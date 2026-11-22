"""Batched GPU cell population engine for Plan 44 M4 (lineages) and M6 (evolution), and Plan 45.

ZH: 與 cell/driver.py 相同的方程式（L2 轉錄轉譯、L3 ATP 帳、L4/L5 脂質生長、L7 複製），
在 torch 上對 B 個細胞做批次向量化；每個細胞有自己的速率倍率（由 genome theta 表現而來）、
自己的轉錄倍率（用來削弱必需基因）、自己的葡萄糖可得度（用於斷養分 / 死亡實驗）。
內層 K 步以 CUDA graph 擷取後重播，避免 M0–M3 的 kernel-launch 瓶頸。

EN: Same equations as cell/driver.py, vectorised over B cells with per-cell rate multipliers
(expressed from genome theta), per-gene transcription multipliers (to weaken an essential gene
class) and per-cell glucose availability (starvation / death). K inner steps are captured in a
CUDA graph and replayed to remove the kernel-launch bottleneck seen in M0-M3.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .dna_link import SCALE_RANGE, THETA_KEYS
from .genome_map import GeneTable
from .params import Params
from .state import balanced_protein, founder_lipids, sphere_area_nm2


def theta_to_scale(theta):
    lo, hi = SCALE_RANGE
    return lo + (hi - lo) * theta


class CellBatch:
    def __init__(self, params: Params, genes: GeneTable, batch: int, dt: float = 1.0,
                 device: str = "cuda", seed: int = 0, steps_per_graph: int = 60,
                 theta: Optional[np.ndarray] = None, tx_mult: Optional[np.ndarray] = None,
                 use_graph: bool = True):
        import torch
        self.torch = torch
        t = torch
        self.p, self.genes, self.B, self.dt = params, genes, batch, float(dt)
        self.dev = t.device(device if (device == "cpu" or t.cuda.is_available()) else "cpu")
        self.use_graph = use_graph and self.dev.type == "cuda"
        self.K = steps_per_graph
        dev, f32 = self.dev, t.float32
        G = len(genes)
        self.G = G
        L = t.tensor(genes.length_bp, device=dev, dtype=f32)
        self.L, self.aa = L, L / 3.0
        cls = t.tensor(genes.gene_class, device=dev)
        self.mask_metabolic = (cls == 1).to(f32)
        self.mask_membrane = (cls == 3).to(f32)
        k_tx0 = params["tx_elongation_nt_per_s"] / L * t.tensor(genes.copies, device=dev, dtype=f32)
        txm = t.ones(batch, G, device=dev) if tx_mult is None else t.as_tensor(tx_mult, device=dev, dtype=f32)
        self.k_tx = (k_tx0.unsqueeze(0) * txm).contiguous()
        self.k_deg = math.log(2.0) / params["mrna_half_life_s"]
        self.per_rib = (params["tl_elongation_aa_per_s"] / self.aa).unsqueeze(0)
        self.ribosomes = params["n_ribosomes"] * params["ribosome_active_fraction"]
        th = np.full((batch, len(THETA_KEYS)), (1.0 - SCALE_RANGE[0]) / (SCALE_RANGE[1] - SCALE_RANGE[0])) \
            if theta is None else np.asarray(theta, dtype=np.float64)
        self.theta = t.as_tensor(th, device=dev, dtype=f32).contiguous()
        self.scale = theta_to_scale(self.theta).contiguous()
        self.gen = t.Generator(device=dev); self.gen.manual_seed(seed)
        t.manual_seed(seed)
        # state
        self.mrna = t.poisson((self.k_tx / self.k_deg).contiguous(), generator=self.gen)
        self.protein = t.tensor(balanced_protein(params, genes), device=dev, dtype=f32).unsqueeze(0).repeat(batch, 1)
        self.pool_per_lipid = params["adenylate_pool"] / founder_lipids(params)   # salvage target
        pool = params["adenylate_pool"]
        self.atp = t.full((batch,), pool * params["atp_initial_fraction"], device=dev, dtype=f32)
        self.adp = pool - self.atp
        self.atp0 = float(pool * params["atp_initial_fraction"])
        area0 = sphere_area_nm2(params["cell_radius_nm"])
        self.area0 = area0
        self.lipids = t.full((batch,), 2.0 * area0 / params["area_per_lipid_nm2"], device=dev, dtype=f32)
        self.lipids_founder = self.lipids.clone()      # exogenous seed lipids (for P2 self-made fraction)
        self.birth_area = self.lipids * params["area_per_lipid_nm2"] / 2.0
        self.repl = t.zeros(batch, device=dev)
        self.t = t.zeros((), device=dev)
        self.birth_t = t.zeros(batch, device=dev)
        self.ready_t = t.full((batch,), float("nan"), device=dev)
        self.frozen = t.zeros(batch, device=dev)       # 1 = ready and waiting for the division handler
        self.glucose = t.ones(batch, device=dev)
        self.low_atp_steps = t.zeros(batch, device=dev)
        self.dead = t.zeros(batch, device=dev)
        self.max_closure = t.zeros((), device=dev)
        self.shortfall = t.zeros(batch, device=dev)
        self.salvaged = t.zeros(batch, device=dev)
        self.generation = t.zeros(batch, device=dev)
        # death criterion (Plan 45 E2): ATP < frac * initial for n consecutive steps
        self.death_frac = t.full((batch,), params["death_atp_fraction"], device=dev)
        self.death_steps = t.full((batch,), params["death_consecutive_steps"], device=dev)
        self.post_death_max_atp = t.zeros(batch, device=dev)
        self.graph = None

    # ------------------------------------------------------------------ physics
    def _step(self):
        t, p, dt = self.torch, self.p, self.dt
        live = (1.0 - self.frozen) * (1.0 - self.dead)
        lv = live.unsqueeze(1)
        atp = self.atp
        sat = atp / (p["atp_km"] + atp)
        prime = atp / (p["glycolysis_atp_priming_km"] + atp)
        n_tx = t.poisson(self.k_tx * dt * lv)
        n_deg = t.minimum(t.poisson(self.mrna * (self.k_deg * dt) * lv), self.mrna + n_tx)
        share = self.mrna / self.mrna.sum(1, keepdim=True).clamp(min=1.0)
        n_tl = t.poisson(self.per_rib * (self.ribosomes * dt) * share * lv)
        self.mrna.add_(n_tx - n_deg)
        self.protein.add_(n_tl)
        atp_tx = (n_tx * self.L).sum(1) * p["atp_per_nt_transcription"]
        atp_tl = (n_tl * self.aa).sum(1) * p["atp_per_aa_translation"]
        memb = (self.protein * self.mask_membrane).sum(1)
        n_lip = p["k_lipid_per_enzyme_per_s"] * self.scale[:, 0] * memb * sat * dt * live
        self.lipids.add_(n_lip)
        remaining = (p["genome_bp"] - self.repl).clamp(min=0.0)
        bp = t.minimum(p["k_replication_bp_per_s"] * self.scale[:, 2] * sat * dt * live, remaining)
        self.repl.add_(bp)
        metab = (self.protein * self.mask_metabolic).sum(1)
        produced = (metab * p["k_glucose_per_transporter_per_s"] * self.scale[:, 1]
                    * p["atp_per_glucose_glycolysis"] * prime * self.glucose * dt * (1.0 - self.frozen))
        consumed = (atp_tx + atp_tl + n_lip * p["atp_per_lipid"] + bp * p["atp_per_bp_replication"]
                    + p["maintenance_atp_per_s"] * self.scale[:, 3] * dt * (1.0 - self.frozen))
        pool_before = self.atp + self.adp
        new_atp = self.atp + produced - consumed
        self.shortfall.add_(t.clamp(-new_atp, min=0.0))
        new_atp = t.minimum(t.clamp(new_atp, min=0.0), pool_before)
        self.atp.copy_(new_atp)
        self.adp.copy_(pool_before - new_atp)
        closure = ((self.atp + self.adp) - pool_before).abs() / pool_before
        self.max_closure.copy_(t.maximum(self.max_closure, closure.max()))
        # nucleotide salvage (cell/metabolism.py salvage_adenylate): booked influx, outside the closure
        deficit = t.clamp(self.pool_per_lipid * self.lipids - (self.atp + self.adp), min=0.0)
        self.adp.add_(deficit)
        self.salvaged.add_(deficit)
        self.t.add_(dt)
        # death bookkeeping (absorbing state)
        low = (self.atp < self.death_frac * self.atp0).float()
        self.low_atp_steps.copy_((self.low_atp_steps + 1.0) * low)
        self.post_death_max_atp.copy_(t.maximum(self.post_death_max_atp, self.atp * self.dead))
        self.dead.copy_(t.maximum(self.dead, (self.low_atp_steps >= self.death_steps).float()))
        # readiness for division
        area = self.lipids * p["area_per_lipid_nm2"] / 2.0
        ready = (area >= 2.0 * self.birth_area) & (self.repl >= p["genome_bp"]) & (self.frozen < 0.5) \
            & (self.dead < 0.5)
        self.ready_t.copy_(t.where(ready, self.t.expand_as(self.ready_t), self.ready_t))
        self.frozen.copy_(t.maximum(self.frozen, ready.float()))

    def _block(self):
        for _ in range(self.K):
            self._step()

    def advance(self):
        """Advance K steps (one CUDA-graph replay when available)."""
        t = self.torch
        if self.use_graph and self.graph is None:
            s = t.cuda.Stream(); s.wait_stream(t.cuda.current_stream())
            snap = self._snapshot()
            with t.cuda.stream(s):
                for _ in range(2):
                    self._step()
            t.cuda.current_stream().wait_stream(s)
            self._restore(snap)
            self.graph = t.cuda.CUDAGraph()
            with t.cuda.graph(self.graph):
                self._block()
            self._restore(snap)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._block()

    _STATE = ("mrna", "protein", "atp", "adp", "lipids", "repl", "t", "ready_t", "frozen",
              "low_atp_steps", "dead", "max_closure", "shortfall", "post_death_max_atp", "salvaged")

    def _snapshot(self):
        return {k: getattr(self, k).clone() for k in self._STATE}

    def _restore(self, snap):
        for k, v in snap.items():
            getattr(self, k).copy_(v)

    # ---------------------------------------------------------------- division
    def divide(self, idx, frac, keep_left, mutate_theta=None):
        """Divide cells `idx` in place: the slot keeps one daughter.

        frac: fraction of the mother's volume/area going to the left daughter (per cell).
        keep_left: bool tensor, which daughter the slot keeps. Molecules are binomially split
        with p = frac (left); lipids/ATP/ADP split by frac. Returns the partition record.
        """
        t, p = self.torch, self.p
        f = frac.unsqueeze(1)
        m_l = t.binomial(self.mrna[idx], f.expand(-1, self.G).contiguous())
        pr_l = t.binomial(self.protein[idx], f.expand(-1, self.G).contiguous())
        kl = keep_left.float()
        kf = t.where(keep_left, frac, 1.0 - frac)
        mrna_parent = self.mrna[idx].sum(1); prot_parent = self.protein[idx].sum(1)
        self.mrna[idx] = m_l * kl.unsqueeze(1) + (self.mrna[idx] - m_l) * (1 - kl).unsqueeze(1)
        self.protein[idx] = pr_l * kl.unsqueeze(1) + (self.protein[idx] - pr_l) * (1 - kl).unsqueeze(1)
        lip_parent = self.lipids[idx].clone()
        self.lipids[idx] = lip_parent * kf
        self.lipids_founder[idx] = self.lipids_founder[idx] * kf
        self.atp[idx] = self.atp[idx] * kf
        self.adp[idx] = self.adp[idx] * kf
        self.repl[idx] = 0.0
        self.birth_area[idx] = self.lipids[idx] * p["area_per_lipid_nm2"] / 2.0
        doubling = self.ready_t[idx] - self.birth_t[idx]
        self.birth_t[idx] = self.t
        self.ready_t[idx] = float("nan")
        self.frozen[idx] = 0.0
        self.generation[idx] += 1
        return {"doubling_s": doubling, "lipids_parent": lip_parent, "kept_fraction": kf,
                "mrna_left": m_l.sum(1), "mrna_parent": mrna_parent,
                "protein_left": pr_l.sum(1), "protein_parent": prot_parent}

    def copy_slot(self, src, dst):
        """Copy whole cell state from slots src to slots dst (Moran replacement)."""
        for name in ("mrna", "protein", "atp", "adp", "lipids", "lipids_founder", "repl", "birth_area",
                     "birth_t", "ready_t", "frozen", "theta", "scale", "k_tx", "generation", "glucose",
                     "low_atp_steps", "dead", "shortfall"):
            ten = getattr(self, name)
            ten[dst] = ten[src]

    def divide_into(self, idx, dst, frac=None):
        """Moran division: mother slot idx keeps the left daughter, slot dst receives the right one."""
        t, p = self.torch, self.p
        n = idx.numel()
        if frac is None:
            frac = t.full((n,), 0.5, device=self.dev)
        doubling = self.ready_t[idx] - self.birth_t[idx]
        self.copy_slot(idx, dst)
        f = frac.unsqueeze(1).expand(-1, self.G).contiguous()
        m_parent, pr_parent = self.mrna[idx].clone(), self.protein[idx].clone()
        m_l = t.binomial(m_parent, f); pr_l = t.binomial(pr_parent, f)
        self.mrna[idx] = m_l; self.mrna[dst] = m_parent - m_l
        self.protein[idx] = pr_l; self.protein[dst] = pr_parent - pr_l
        for name in ("lipids", "lipids_founder", "atp", "adp"):
            ten = getattr(self, name)
            v = ten[idx].clone()
            ten[idx] = v * frac
            ten[dst] = v * (1.0 - frac)
        for slots in (idx, dst):
            self.repl[slots] = 0.0
            self.birth_area[slots] = self.lipids[slots] * p["area_per_lipid_nm2"] / 2.0
            self.birth_t[slots] = self.t
            self.ready_t[slots] = float("nan")
            self.frozen[slots] = 0.0
            self.generation[slots] = self.generation[slots] + 1
        return doubling

    def mutate(self, slots, mu: float, sigma: float):
        """Per-locus mutation of theta for the given slots; returns the number of mutation events."""
        t = self.torch
        th = self.theta[slots]
        hit = (t.rand(th.shape, device=self.dev) < mu).float()
        th = (th + hit * sigma * t.randn(th.shape, device=self.dev)).clamp(0.0, 1.0)
        self.theta[slots] = th
        self.scale[slots] = theta_to_scale(th)
        return hit.sum()
