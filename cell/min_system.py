"""L6: Min-protein reaction-diffusion that positions the division site (Plan 44 M4).

ZH: 以 Huang, Meir & Wingreen 2003（PNAS, DOI 10.1073/pnas.2135445100）的 MinD/MinE 反應循環
為準，化簡成沿細胞長軸的一維模型（假設截面內均勻）。原文用 3D 圓柱並指出 1D 近似可能失準，
本模組以「振盪週期是否接近原文約 40 秒、MinD 時間平均最低點是否在中央」做驗證並據實回報。
syn3A 本身沒有 Min 系統；這是 C6322（Godino 2019）那種「在人工細胞中重建 Min」的設定。

EN: 1D axial reduction of the Huang-Meir-Wingreen 2003 MinD/MinE cycle (cross-section assumed
well mixed). The paper itself warns that 1D can be inaccurate; we validate the reduction against
its ~40 s period and a central MinD minimum and report the measured numbers. syn3A has no Min
system; this models a Min-reconstituted artificial cell as in C6322.

Species per voxel (molecule counts): D_ADP, D_ATP, E (cytosolic, diffuse); d (membrane MinD:ATP),
de (membrane MinD:ATP:MinE complex). Membrane species do not diffuse (as in the paper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .params import Params


@dataclass
class MinParams:
    D_um2_s: float
    s_exch: float     # MinD:ADP -> MinD:ATP, 1/s
    s_D: float        # um/s
    s_dD: float       # um^3/s
    s_de: float       # 1/s
    s_E: float        # um^3/s
    radius_um: float
    mind_per_um: float
    mine_per_um: float

    @classmethod
    def from_params(cls, p: Params) -> "MinParams":
        return cls(D_um2_s=p["min_D_um2_s"], s_exch=p["min_sigma_exch_s"], s_D=p["min_sigma_D_um_s"],
                   s_dD=p["min_sigma_dD_um3_s"], s_de=p["min_sigma_de_s"], s_E=p["min_sigma_E_um3_s"],
                   radius_um=p["rod_radius_um"], mind_per_um=p["min_mind_per_um"],
                   mine_per_um=p["min_mine_per_um"])


def _rates(mp: MinParams, dx: float, nATP, nE, Md, Mde, xp=np):
    """Reaction propensities per voxel (molecules/s) for the 1D reduction.

    Volume concentration = n / (pi R^2 dx); membrane density = M / (2 pi R dx).
    Fluxes per membrane area are multiplied by the voxel membrane area 2 pi R dx.
    """
    area = math.pi * mp.radius_um ** 2
    per = 2.0 * math.pi * mp.radius_um
    m_tot = (Md + Mde) / (per * dx)
    attach = per * (mp.s_D + mp.s_dD * m_tot) * nATP / area
    recruit = mp.s_E * Md * nE / (dx * area)
    hydro = mp.s_de * Mde
    return attach, recruit, hydro


def deterministic_profile(mp: MinParams, length_um: float, dx: float = 0.1, t_end_s: float = 300.0,
                          avg_from_s: float = 100.0, seed: int = 0) -> Dict[str, object]:
    """Explicit-Euler mean-field solution (NumPy). Used for validation and unit tests."""
    rng = np.random.default_rng(seed)
    n = max(int(round(length_um / dx)), 3)
    dt = 0.4 * dx * dx / mp.D_um2_s
    D0 = mp.mind_per_um * length_um / n
    E0 = mp.mine_per_um * length_um / n
    nADP = D0 * (1.0 + 0.01 * rng.standard_normal(n)); nADP *= D0 * n / nADP.sum()
    nATP = np.zeros(n); nE = np.full(n, E0); Md = np.zeros(n); Mde = np.zeros(n)

    def lap(u):
        p = np.concatenate(([u[0]], u, [u[-1]]))
        return (p[2:] - 2 * u + p[:-2]) * (mp.D_um2_s / dx / dx)

    steps = int(t_end_s / dt)
    acc = np.zeros(n); nacc = 0
    asym = []
    every = max(int(1.0 / dt), 1)
    for s in range(steps):
        attach, recruit, hydro = _rates(mp, dx, nATP, nE, Md, Mde)
        exch = mp.s_exch * nADP
        nADP = nADP + dt * (lap(nADP) - exch + hydro)
        nATP = nATP + dt * (lap(nATP) + exch - attach)
        nE = nE + dt * (lap(nE) + hydro - recruit)
        Md = Md + dt * (attach - recruit)
        Mde = Mde + dt * (recruit - hydro)
        if s * dt >= avg_from_s:
            acc += Md + Mde; nacc += 1
        if s % every == 0:
            m = Md + Mde
            asym.append(float(m[: n // 2].sum() - m[n - n // 2:].sum()))
    prof = acc / max(nacc, 1)
    a = np.asarray(asym[int(avg_from_s):])
    z = np.where(np.diff(np.sign(a)) != 0)[0]
    period = float(2.0 * np.mean(np.diff(z))) if z.size > 2 else None
    total = float((nADP + nATP + Md + Mde).sum())
    return {"profile": prof, "site_fraction": site_from_profile(prof), "period_s": period,
            "contrast": float(prof.min() / prof.max()) if prof.max() > 0 else None,
            "mind_conservation": total / (D0 * n), "dt_s": dt, "dx_um": dx}


def site_quadfit(prof: np.ndarray):
    """FtsZ-ring readout (ASSUMED model): vertex of a least-squares quadratic fitted to the whole
    time-averaged membrane-MinD profile, i.e. the ring follows the cell-scale MinD/MinC gradient
    rather than single-voxel noise. Returns (site_fraction, fallback_used). Falls back to the
    parabola-refined argmin when the fit has no trough inside the cell (reported as a fallback).
    """
    n = prof.size
    x = (np.arange(n) + 0.5) / n
    c = np.polyfit(x, prof, 2)
    if c[0] > 0:
        v = -c[1] / (2.0 * c[0])
        if 0.0 < v < 1.0:
            return float(v), False
    return site_from_profile(prof), True


def site_from_profile(prof: np.ndarray) -> float:
    """Division site (fraction of length) = minimum of time-averaged membrane MinD, parabola-refined."""
    n = prof.size
    i = int(np.argmin(prof))
    off = 0.0
    if 0 < i < n - 1:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        den = a - 2 * b + c
        if den > 0:
            off = 0.5 * (a - c) / den
    return float((i + 0.5 + off) / n)


class MinBatchGPU:
    """Stochastic (tau-leaping + binomial hopping) 1D Min model for a batch of cells on CUDA.

    Each row is one cell with its own length (voxels beyond its length are inactive).
    The inner loop is captured once in a CUDA graph and replayed, so the GPU is not
    starved by per-kernel launch overhead (the M0-M3 bottleneck, 32.6% mean GR3D).
    """

    def __init__(self, mp: MinParams, lengths_um, dx: float = 0.1, dt: float = 1e-3,
                 device: str = "cuda", seed: int = 0, steps_per_graph: int = 200):
        import torch
        self.torch = torch
        self.mp, self.dx, self.dt = mp, dx, dt
        self.dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        lengths = torch.as_tensor(np.asarray(lengths_um, dtype=np.float32), device=self.dev)
        self.nvox = torch.clamp(torch.round(lengths / dx), min=3).to(torch.int64)
        B = lengths.numel(); N = int(self.nvox.max().item())
        self.B, self.N = B, N
        idx = torch.arange(N, device=self.dev).unsqueeze(0)
        self.active = (idx < self.nvox.unsqueeze(1)).float()
        self.can_right = (idx < (self.nvox.unsqueeze(1) - 1)).float()
        self.can_left = ((idx > 0) & (idx < self.nvox.unsqueeze(1))).float()
        g = torch.Generator(device=self.dev); g.manual_seed(seed)
        Lc = (self.nvox.float() * dx).unsqueeze(1)
        D_tot = torch.poisson((mp.mind_per_um * Lc).expand(B, 1).contiguous(), generator=g)
        E_tot = torch.poisson((mp.mine_per_um * Lc).expand(B, 1).contiguous(), generator=g)
        # all MinD starts as cytosolic ADP form, spread uniformly (Poisson per voxel); the
        # conserved totals are the realised sums, not the Poisson means
        w = self.active / self.nvox.unsqueeze(1).float()
        self.nADP = torch.poisson(D_tot * w, generator=g)
        self.nATP = torch.zeros(B, N, device=self.dev)
        self.nE = torch.poisson(E_tot * w, generator=g)
        self.D_total = self.nADP.sum(1).clone(); self.E_total = self.nE.sum(1).clone()
        self.Md = torch.zeros(B, N, device=self.dev)
        self.Mde = torch.zeros(B, N, device=self.dev)
        self.acc = torch.zeros(B, N, device=self.dev)
        self.accumulate = torch.zeros((), device=self.dev)
        self.p_hop = mp.D_um2_s * dt / dx / dx
        if self.p_hop > 0.5:
            raise ValueError(f"hop probability {self.p_hop} > 0.5; reduce dt")
        # conditional left-hop probability given no right hop: p / (1 - p_right) per voxel
        # (p_right = 0 at the last active voxel, so there it is exactly p)
        pr = self.p_hop * self.can_right
        self.p_right = pr.unsqueeze(0).expand(3, B, N).contiguous()
        self.p_left_cond = (self.p_hop * self.can_left / (1.0 - pr)).unsqueeze(0).expand(3, B, N).contiguous()
        self.steps_per_graph = steps_per_graph
        self.graph = None
        self.asym = []
        self._seed = seed
        half = (self.nvox // 2).unsqueeze(1)
        self.left_mask = (idx < half).float()
        self.right_mask = ((idx >= self.nvox.unsqueeze(1) - half) & (idx < self.nvox.unsqueeze(1))).float()

    # one stochastic step, all in-place on static tensors (graph-capturable)
    def _step(self):
        t = self.torch
        mp, dt = self.mp, self.dt
        attach, recruit, hydro = _rates(mp, self.dx, self.nATP, self.nE, self.Md, self.Mde, xp=t)
        n_exch = t.minimum(t.poisson(mp.s_exch * self.nADP * dt), self.nADP)
        n_att = t.minimum(t.poisson(attach * dt), self.nATP)
        n_rec = t.minimum(t.minimum(t.poisson(recruit * dt), self.Md), self.nE)
        n_hyd = t.minimum(t.poisson(hydro * dt), self.Mde)
        self.nADP.add_(n_hyd - n_exch)
        self.nATP.add_(n_exch - n_att)
        self.nE.add_(n_hyd - n_rec)
        self.Md.add_(n_att - n_rec)
        self.Mde.add_(n_rec - n_hyd)
        # binomial hopping for the three cytosolic species, stacked into one tensor
        cyt = t.stack((self.nADP, self.nATP, self.nE))
        p = self.p_hop
        right = t.binomial(cyt, self.p_right)
        left = t.binomial(cyt - right, self.p_left_cond)
        moved = cyt - right - left
        moved[..., 1:] += right[..., :-1]
        moved[..., :-1] += left[..., 1:]
        self.nADP.copy_(moved[0]); self.nATP.copy_(moved[1]); self.nE.copy_(moved[2])
        self.acc.add_(self.accumulate * (self.Md + self.Mde))

    def _block(self):
        for _ in range(self.steps_per_graph):
            self._step()

    def run(self, t_end_s: float, avg_from_s: float, use_graph: bool = True, asym_every_s: float = 1.0):
        t = self.torch
        n_blocks = int(round(t_end_s / (self.dt * self.steps_per_graph)))
        avg_block = int(round(avg_from_s / (self.dt * self.steps_per_graph)))
        if use_graph and self.dev.type == "cuda" and self.graph is None:
            t.manual_seed(self._seed)
            s = t.cuda.Stream(); s.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(s):
                for _ in range(2):   # warm-up outside capture (required by CUDA graphs)
                    self._step()
            t.cuda.current_stream().wait_stream(s)
            self.graph = t.cuda.CUDAGraph()
            with t.cuda.graph(self.graph):
                self._block()
        for b in range(n_blocks):
            if b == avg_block:
                self.accumulate.fill_(1.0)
            if self.graph is not None:
                self.graph.replay()
            else:
                self._block()
            m = self.Md + self.Mde
            self.asym.append((m * self.left_mask).sum(1) - (m * self.right_mask).sum(1))
        prof = self.acc * self.active
        return prof

    def results(self, prof) -> Dict[str, np.ndarray]:
        t = self.torch
        prof_np = prof.detach().cpu().numpy()
        nvox = self.nvox.cpu().numpy()
        sites_argmin = np.array([site_from_profile(prof_np[i, : nvox[i]]) for i in range(self.B)])
        q = [site_quadfit(prof_np[i, : nvox[i]]) for i in range(self.B)]
        sites = np.array([a for a, _ in q]); fallback = np.array([b for _, b in q])
        asym = t.stack(self.asym, 1).cpu().numpy() if self.asym else np.zeros((self.B, 0))
        block_s = self.dt * self.steps_per_graph
        periods = np.full(self.B, np.nan)
        for i in range(self.B):
            a = asym[i, asym.shape[1] // 3:]
            z = np.where(np.diff(np.sign(a)) != 0)[0]
            if z.size > 2:
                periods[i] = 2.0 * np.mean(np.diff(z)) * block_s
        conserv = ((self.nADP + self.nATP + self.Md + self.Mde).sum(1) / self.D_total).cpu().numpy()
        conserv_e = ((self.nE + self.Mde).sum(1) / self.E_total).cpu().numpy()
        return {"site_fraction": sites, "site_argmin": sites_argmin, "readout_fallback": fallback,
                "period_s": periods, "mind_conservation": conserv,
                "mine_conservation": conserv_e, "length_um": nvox * self.dx}
