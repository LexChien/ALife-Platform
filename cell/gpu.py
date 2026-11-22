"""GPU-batched ensemble of the same model (Plan 44, D5 amended by Lex 2026-11-22: use CPU+GPU).

ZH: 與 cell/driver.py 相同方程式，改成在 torch 上對 B 個細胞批次向量化；在 Jetson 上跑 CUDA。
EN: Same equations as the NumPy driver, vectorised over a batch of B cells with torch (CUDA on the Jetson).
Falls back to CPU tensors when CUDA is unavailable, and says so in the result.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np

from .genome_map import GeneTable, build_gene_table
from .params import Params, load_params
from .driver import RunConfig, apply_overrides
from .state import balanced_protein, founder_lipids, sphere_area_nm2


def cuda_report() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch optional
        return {"torch": None, "cuda_available": False, "error": repr(exc)[:200]}
    info: Dict[str, Any] = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["cuda_version"] = torch.version.cuda
    return info


def run_batch(batch: int = 64, config: Optional[RunConfig] = None, params: Optional[Params] = None,
              genes: Optional[GeneTable] = None, device: str = "cuda") -> Dict[str, Any]:
    """Run `batch` independent cells in lock-step on the GPU. Returns measured results."""
    import torch

    config = config or RunConfig()
    params = apply_overrides(params or load_params(), config.overrides)
    genes = genes or build_gene_table(params)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    g = torch.Generator(device=dev); g.manual_seed(config.seed)

    L = torch.tensor(genes.length_bp, device=dev, dtype=dtype)
    aa = L / 3.0
    copies = torch.tensor(genes.copies, device=dev, dtype=dtype)
    cls = torch.tensor(genes.gene_class, device=dev)
    mask_metabolic = (cls == 1).to(dtype)
    mask_membrane = (cls == 3).to(dtype)

    k_tx = (params["tx_elongation_nt_per_s"] / L * copies).unsqueeze(0).expand(batch, -1).contiguous()
    k_deg = math.log(2.0) / params["mrna_half_life_s"]
    per_ribosome = params["tl_elongation_aa_per_s"] / aa
    ribosomes = params["n_ribosomes"] * params["ribosome_active_fraction"]

    mrna = torch.poisson(k_tx / k_deg, generator=g)
    protein = torch.tensor(balanced_protein(params, genes), device=dev, dtype=dtype).unsqueeze(0).repeat(batch, 1)
    pool_per_lipid = params["adenylate_pool"] / founder_lipids(params)   # salvage target (cell/metabolism.py)
    pool = params["adenylate_pool"]
    atp = torch.full((batch,), pool * params["atp_initial_fraction"], device=dev, dtype=dtype)
    adp = pool - atp
    area0 = sphere_area_nm2(params["cell_radius_nm"])
    lipids = torch.full((batch,), 2.0 * area0 / params["area_per_lipid_nm2"], device=dev, dtype=dtype)
    birth_area = lipids * params["area_per_lipid_nm2"] / 2.0
    replicated = torch.zeros((batch,), device=dev, dtype=dtype)
    t = 0.0
    divided_at = torch.full((batch,), float("nan"), device=dev, dtype=dtype)
    dt = config.dt_s
    km = params["atp_km"]
    max_closure = 0.0
    t0 = time.perf_counter()
    steps = int(config.max_time_s / dt)
    for _ in range(steps):
        sat = atp / (km + atp)
        n_tx = torch.poisson(k_tx * dt, generator=g)
        n_deg = torch.poisson(mrna * k_deg * dt, generator=g)
        n_deg = torch.minimum(n_deg, mrna + n_tx)
        total_mrna = mrna.sum(dim=1, keepdim=True).clamp(min=1.0)
        k_tl = per_ribosome.unsqueeze(0) * ribosomes * (mrna / total_mrna)
        n_tl = torch.poisson(k_tl * dt, generator=g)
        mrna = mrna + n_tx - n_deg
        protein = protein + n_tl
        atp_tx = (n_tx * L).sum(dim=1) * params["atp_per_nt_transcription"]
        atp_tl = (n_tl * aa).sum(dim=1) * params["atp_per_aa_translation"]

        membrane_enz = (protein * mask_membrane).sum(dim=1)
        n_lipid = params["k_lipid_per_enzyme_per_s"] * membrane_enz * sat * dt
        lipids = lipids + n_lipid
        atp_lip = n_lipid * params["atp_per_lipid"]

        remaining = (params["genome_bp"] - replicated).clamp(min=0.0)
        bp = torch.minimum(params["k_replication_bp_per_s"] * sat * dt, remaining)
        replicated = replicated + bp
        atp_rep = bp * params["atp_per_bp_replication"]

        metabolic = (protein * mask_metabolic).sum(dim=1)
        prime = atp / (params["glycolysis_atp_priming_km"] + atp)
        produced = metabolic * params["k_glucose_per_transporter_per_s"] * params["atp_per_glucose_glycolysis"] * prime * dt
        consumed = atp_tx + atp_tl + atp_lip + atp_rep + params["maintenance_atp_per_s"] * dt
        pool_before = atp + adp
        atp = torch.clamp(atp + produced - consumed, min=0.0)
        atp = torch.minimum(atp, pool_before)
        adp = pool_before - atp
        closure = ((atp + adp) - pool_before).abs() / pool_before
        max_closure = max(max_closure, float(closure.max().item()))
        adp = adp + torch.clamp(pool_per_lipid * lipids - (atp + adp), min=0.0)   # nucleotide salvage

        t += dt
        area = lipids * params["area_per_lipid_nm2"] / 2.0
        done = torch.isnan(divided_at) & (area >= 2.0 * birth_area) & (replicated >= params["genome_bp"])
        if bool(done.any()):
            divided_at = torch.where(done, torch.full_like(divided_at, t), divided_at)
        if not bool(torch.isnan(divided_at).any()):
            break
    wall = time.perf_counter() - t0
    dts = divided_at.detach().cpu().numpy()
    finite = dts[np.isfinite(dts)]
    return {
        "device": str(dev),
        "cuda_used": use_cuda,
        "batch": batch,
        "steps_simulated": int(t / dt),
        "sim_time_s": t,
        "wall_time_s": wall,
        "cells_divided": int(finite.size),
        "doubling_time_s": finite.tolist(),
        "doubling_time_median_s": float(np.median(finite)) if finite.size else None,
        "doubling_time_min_s": float(finite.min()) if finite.size else None,
        "doubling_time_max_s": float(finite.max()) if finite.size else None,
        "max_closure_error": max_closure,
    }
