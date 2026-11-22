"""GPU experiments for Plan 44 M4/M6 and Plan 45 P7/P8. Every function streams raw JSONL rows.

ZH: 所有門檻都只由這裡寫出的原始 JSONL 重算（M7）；本模組不做任何「好看」的後處理。
EN: All gates are recomputed from the raw JSONL written here (M7).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .genome_map import build_gene_table
from .min_system import MinBatchGPU, MinParams
from .params import load_params
from .population import CellBatch

Progress = Callable[[str, float, str], None]


def _noop(stage: str, frac: float, note: str) -> None:
    pass


def _jl(fh, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------- M4 + P2/P3
def m4_lineages(out: Path, batch: int = 1024, generations: int = 12, seed: int = 4,
                min_dt: float = 1e-3, min_dx: float = 0.1, progress: Progress = _noop) -> Dict[str, Any]:
    import torch
    p = load_params(); genes = build_gene_table(p); mp = MinParams.from_params(p)
    cb = CellBatch(p, genes, batch, dt=1.0, seed=seed, steps_per_graph=60)
    rng = torch.Generator(device=cb.dev); rng.manual_seed(seed + 1)
    L_birth = p["rod_birth_length_um"]; area_ref = cb.area0
    window, burn = p["min_window_s"], p["min_burnin_s"]
    max_cycle = 4.0 * p["doubling_time_experimental_s"]
    t0 = time.perf_counter(); min_wall = grow_wall = 0.0
    alive_slots = torch.ones(batch, dtype=torch.bool, device=cb.dev)
    with (out / "m4_divisions.jsonl").open("w") as fh, (out / "m4_generations.jsonl").open("w") as fg:
        for gen in range(generations):
            tg = time.perf_counter(); t_start = float(cb.t)
            while True:
                cb.advance()
                if bool((cb.frozen > 0.5).all()) or float(cb.t) - t_start > max_cycle:
                    break
            grow_wall += time.perf_counter() - tg
            not_ready = int((cb.frozen < 0.5).sum())
            area = cb.lipids * p["area_per_lipid_nm2"] / 2.0
            sav = (area / 1e6) / (4.0 / 3.0 * math.pi * (torch.sqrt(area / (4 * math.pi)) / 1000.0) ** 3)
            lengths = (L_birth * area / area_ref).cpu().numpy()
            tm = time.perf_counter()
            mb = MinBatchGPU(mp, lengths, dx=min_dx, dt=min_dt, seed=seed * 1000 + gen)
            prof = mb.run(window, burn)
            res = mb.results(prof)
            # Root-cause fix (2026-11-23, run attempt 1): in a 220 s window a few cells (~0.5%) had not
            # established the pole-to-pole oscillation (no trough, abnormal period); the argmin fallback
            # then put the site at a pole and the kept 3% minicell died. Real Min oscillates through the
            # whole cycle, so the short window is our artefact: re-simulate only those cells with a
            # longer window (new seed), up to 3 times, and record how many retries each cell needed.
            retries = np.zeros(batch, dtype=int)
            for attempt in range(3):
                bad = np.where(res["readout_fallback"] | ~np.isfinite(res["period_s"]))[0]
                if bad.size == 0:
                    break
                mb2 = MinBatchGPU(mp, lengths[bad], dx=min_dx, dt=min_dt, seed=seed * 1000 + gen + 7919 * (attempt + 1))
                prof2 = mb2.run(2.0 * window, burn)
                r2 = mb2.results(prof2)
                for key in ("site_fraction", "site_argmin", "readout_fallback", "period_s",
                            "mind_conservation", "mine_conservation"):
                    res[key][bad] = r2[key]
                retries[bad] += 1
            torch.cuda.synchronize() if cb.dev.type == "cuda" else None
            min_wall += time.perf_counter() - tm
            frac = torch.as_tensor(res["site_fraction"], device=cb.dev, dtype=torch.float32).clamp(0.02, 0.98)
            keep_left = torch.rand(batch, generator=rng, device=cb.dev) < 0.5
            idx = torch.arange(batch, device=cb.dev)
            was_ready = (cb.frozen > 0.5).cpu().numpy()
            rec = cb.divide(idx, frac, keep_left)
            d = {k: v.detach().cpu().numpy() for k, v in rec.items()}
            fr = frac.cpu().numpy(); kl = keep_left.cpu().numpy()
            sav_np = sav.cpu().numpy(); shortfall = cb.shortfall.cpu().numpy(); dead = cb.dead.cpu().numpy()
            for i in range(batch):
                _jl(fh, {"gen": gen, "slot": i, "was_ready": bool(was_ready[i]), "length_um": float(lengths[i]), "site_fraction": float(fr[i]),
                         "volume_left_fraction": float(fr[i]), "kept_left": bool(kl[i]),
                         "site_argmin": float(res["site_argmin"][i]), "readout_fallback": bool(res["readout_fallback"][i]),
                         "min_retries": int(retries[i]),
                         "doubling_s": float(d["doubling_s"][i]), "min_period_s": float(res["period_s"][i]),
                         "mind_conservation": float(res["mind_conservation"][i]),
                         "mine_conservation": float(res["mine_conservation"][i]),
                         "sa_v_at_division_um_inv": float(sav_np[i]),
                         "mrna_parent": float(d["mrna_parent"][i]), "mrna_left": float(d["mrna_left"][i]),
                         "protein_parent": float(d["protein_parent"][i]), "protein_left": float(d["protein_left"][i]),
                         "atp_shortfall_cum": float(shortfall[i]), "dead": bool(dead[i] > 0.5)})
            self_made = 1.0 - float((cb.lipids_founder / cb.lipids).mean())
            _jl(fg, {"gen": gen, "not_ready": not_ready, "sim_time_s": float(cb.t),
                     "self_made_lipid_fraction_mean": self_made, "max_closure": float(cb.max_closure),
                     "nan_state": bool(torch.isnan(cb.atp).any() or torch.isnan(cb.lipids).any()),
                     "dead": int((cb.dead > 0.5).sum()), "grow_wall_s": grow_wall, "min_wall_s": min_wall})
            fh.flush(); fg.flush()
            progress("M4", (gen + 1) / generations, f"generation {gen + 1}/{generations}, not_ready={not_ready}")
    return {"batch": batch, "generations": generations, "wall_s": time.perf_counter() - t0,
            "grow_wall_s": grow_wall, "min_wall_s": min_wall}


def min_validation(out: Path, lengths=(2.0, 3.0, 4.0, 5.0, 6.0), seed: int = 7) -> Dict[str, Any]:
    """Deterministic 1D reduction vs Huang 2003 (period ~40 s at L = 4 um, centre minimum)."""
    from .min_system import deterministic_profile
    p = load_params(); mp = MinParams.from_params(p)
    rows = []
    with (out / "m4_min_validation.jsonl").open("w") as fh:
        for L in lengths:
            r = deterministic_profile(mp, L, dx=0.1, t_end_s=300.0, avg_from_s=100.0, seed=seed)
            row = {"length_um": L, "period_s": r["period_s"], "site_fraction": r["site_fraction"],
                   "contrast_min_over_max": r["contrast"], "mind_conservation": r["mind_conservation"]}
            _jl(fh, row); rows.append(row)
    return {"rows": rows}


# --------------------------------------------------------------------------- dt check
def dt_validation(out: Path, batch: int = 4096, seed: int = 11) -> Dict[str, Any]:
    import torch
    p = load_params(); genes = build_gene_table(p)
    res = {}
    with (out / "m6_dt_validation.jsonl").open("w") as fh:
        for dt, K in ((1.0, 60), (10.0, 6)):
            cb = CellBatch(p, genes, batch, dt=dt, seed=seed, steps_per_graph=K)
            t0 = time.perf_counter()
            while not bool((cb.frozen > 0.5).all()) and float(cb.t) < 30000:
                cb.advance()
            d = (cb.ready_t - cb.birth_t).cpu().numpy()
            d = d[np.isfinite(d)]
            row = {"dt_s": dt, "n": int(d.size), "median_s": float(np.median(d)), "mean_s": float(d.mean()),
                   "min_s": float(d.min()), "max_s": float(d.max()), "wall_s": time.perf_counter() - t0}
            _jl(fh, row); res[str(dt)] = row
    return res


# --------------------------------------------------------------------------- M6 + P6
def m6_evolution(out: Path, n_pop: int = 1024, replicates: int = 5, generations: int = 2000,
                 mu: float = 0.005, sigma: float = 0.05, dt: float = 10.0, seed: int = 6,
                 window_gen: int = 10, ancestral_gens: int = 30, garden_gens: int = 20,
                 progress: Progress = _noop) -> Dict[str, Any]:
    import torch
    p = load_params(); genes = build_gene_table(p)
    P = 2 * replicates; B = P * n_pop
    weak_mult = p["weakened_class_tx_multiplier"]
    tx = np.ones((B, len(genes)))
    memb = genes.gene_class == 3
    cond = np.repeat(np.array(["control"] * replicates + ["weakened"] * replicates), n_pop)
    tx[np.ix_(cond == "weakened", memb)] = weak_mult
    t0 = time.perf_counter()

    def run_phase(cb, phase, target_gens, mu_phase, fh):
        dev = cb.dev
        pop_of = torch.arange(B, device=dev) // n_pop
        div_count = torch.zeros(P, device=dev); dbl_sum = torch.zeros(P, device=dev)
        dbl_sq = torch.zeros(P, device=dev)
        win_count = torch.zeros(P, device=dev); win_sum = torch.zeros(P, device=dev)
        mut_events = torch.zeros((), device=dev); mut_trials = 0
        next_report = window_gen
        last_print = time.perf_counter()
        while True:
            cb.advance()
            ready = torch.nonzero(cb.frozen > 0.5).squeeze(1)
            if ready.numel():
                pops = pop_of[ready]
                victims = pops * n_pop + torch.randint(0, n_pop, (ready.numel(),), device=dev)
                # victims are drawn from the whole population (any cell can be washed out, as in a
                # turbidostat); a ready cell that is itself someone's victim loses its division, and a
                # victim slot is used by one mother only. (A first version only allowed non-ready
                # victims and deadlocked when a whole population became ready; fixed 2026-11-23.)
                ar = torch.arange(ready.numel(), device=dev)
                owner = torch.full((B,), -1, dtype=torch.long, device=dev)
                owner[victims] = ar
                ok = (owner[victims] == ar) & (owner[ready] == -1) & (victims != ready)
                src, dst = ready[ok], victims[ok]
                if src.numel():
                    dbl = cb.divide_into(src, dst)
                    sp = pops[ok]
                    div_count.index_add_(0, sp, torch.ones_like(dbl)); win_count.index_add_(0, sp, torch.ones_like(dbl))
                    dbl_sum.index_add_(0, sp, dbl); win_sum.index_add_(0, sp, dbl)
                    dbl_sq.index_add_(0, sp, dbl * dbl)
                    if mu_phase > 0:
                        both = torch.cat((src, dst))
                        mut_events += cb.mutate(both, mu_phase, sigma)
                        mut_trials += int(both.numel()) * cb.theta.shape[1]
            gens = div_count / n_pop
            gmin = float(gens.min())
            if gmin >= next_report or gmin >= target_gens:
                th = cb.theta.view(P, n_pop, -1)
                row = {"phase": phase, "sim_time_s": float(cb.t), "gens": gens.tolist(),
                       "window_mean_doubling_s": (win_sum / win_count.clamp(min=1)).tolist(),
                       "window_divisions": win_count.tolist(),
                       "theta_mean": th.mean(1).tolist(), "theta_var": th.var(1).tolist(),
                       "mutation_events": float(mut_events), "mutation_trials": mut_trials,
                       "dead": int((cb.dead > 0.5).sum()), "max_closure": float(cb.max_closure),
                       "wall_s": time.perf_counter() - t0}
                _jl(fh, row); fh.flush()
                win_count.zero_(); win_sum.zero_()
                next_report += window_gen
                if time.perf_counter() - last_print > 20:
                    progress("M6", min(gmin / max(target_gens, 1), 1.0), f"{phase}: min generations {gmin:.1f}/{target_gens}")
                    last_print = time.perf_counter()
            if gmin >= target_gens:
                return {"divisions": div_count.tolist(), "mean_doubling_s": (dbl_sum / div_count.clamp(min=1)).tolist(),
                        "mutation_events": float(mut_events), "mutation_trials": mut_trials}

    with (out / "m6_evolution.jsonl").open("w") as fh:
        # ancestral fitness (mu = 0), fresh founders
        cb = CellBatch(p, genes, B, dt=dt, seed=seed, steps_per_graph=6, tx_mult=tx)
        anc = run_phase(cb, "ancestral_mu0", ancestral_gens, 0.0, fh)
        # evolution from fresh founders
        cb = CellBatch(p, genes, B, dt=dt, seed=seed + 1, steps_per_graph=6, tx_mult=tx)
        evo = run_phase(cb, "evolution", generations, mu, fh)
        theta_final = cb.theta.view(P, n_pop, -1).mean(1).cpu().numpy().tolist()
        garden = run_phase(cb, "evolved_mu0", garden_gens, 0.0, fh)
    return {"populations": P, "n_pop": n_pop, "conditions": ["control"] * replicates + ["weakened"] * replicates,
            "mu": mu, "sigma": sigma, "dt_s": dt, "ancestral": anc, "evolution": evo, "garden": garden,
            "theta_final_mean": theta_final, "wall_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- P8 + P7
def p8_death(out: Path, per_cell: int = 64, seed: int = 8, progress: Progress = _noop) -> Dict[str, Any]:
    import torch
    p = load_params(); genes = build_gene_table(p)
    durations = [0, 2, 5, 10, 20, 30, 60, 120, 600]
    fracs = [0.01, 0.05, 0.10]
    nsteps = [30, 60, 120]
    combos = [(d, f, n) for d in durations for f in fracs for n in nsteps]
    B = len(combos) * per_cell
    cb = CellBatch(p, genes, B, dt=1.0, seed=seed, steps_per_graph=1)   # 1 s control resolution
    dur = torch.tensor(np.repeat([c[0] for c in combos], per_cell), device=cb.dev, dtype=torch.float32)
    cb.death_frac.copy_(torch.tensor(np.repeat([c[1] for c in combos], per_cell), device=cb.dev, dtype=torch.float32))
    cb.death_steps.copy_(torch.tensor(np.repeat([c[2] for c in combos], per_cell), device=cb.dev, dtype=torch.float32))
    t_starve, t_total = 1800.0, 1800.0 + 600.0 + 7200.0
    divided_after = torch.zeros(B, device=cb.dev)
    while float(cb.t) < t_total:
        g = torch.where((dur > 0) & (cb.t >= t_starve) & (cb.t < t_starve + dur), 0.0, 1.0)
        cb.glucose.copy_(g)
        cb.advance()
        rd = torch.nonzero(cb.frozen > 0.5).squeeze(1)
        if rd.numel():
            if float(cb.t) > t_starve:
                divided_after[rd] += 1
            cb.divide(rd, torch.full((rd.numel(),), 0.5, device=cb.dev), torch.ones(rd.numel(), dtype=torch.bool, device=cb.dev))
        if int(float(cb.t)) % 300 == 0:
            progress("P8", min(float(cb.t) / t_total, 1.0), f"t={float(cb.t):.0f}/{t_total:.0f}s")
    dead = cb.dead.cpu().numpy(); post = cb.post_death_max_atp.cpu().numpy()
    thr = cb.death_frac.cpu().numpy() * cb.atp0
    dv = divided_after.cpu().numpy(); atp = cb.atp.cpu().numpy()
    with (out / "p8_death.jsonl").open("w") as fh:
        for i in range(B):
            d, f, n = combos[i // per_cell]
            _jl(fh, {"starve_s": d, "death_frac": f, "death_steps": n, "dead": bool(dead[i] > 0.5),
                     "revived": bool(dead[i] > 0.5 and post[i] > thr[i]), "post_death_max_atp": float(post[i]),
                     "final_atp": float(atp[i]), "divisions_after_starvation": int(dv[i])})
    return {"cells": B, "combos": len(combos)}


def p7_response(out: Path, batch: int = 2048, seed: int = 9, progress: Progress = _noop) -> Dict[str, Any]:
    import torch
    p = load_params(); genes = build_gene_table(p)
    cb = CellBatch(p, genes, batch, dt=1.0, seed=seed, steps_per_graph=10)
    t_step, t_total, level = 13000.0, 13000.0 + 4 * 6300.0, 0.5
    with (out / "p7_response.jsonl").open("w") as fh:
        while float(cb.t) < t_total:
            if float(cb.t) >= t_step:
                cb.glucose.fill_(level)
            cb.advance()
            rd = torch.nonzero(cb.frozen > 0.5).squeeze(1)
            n_div = int(rd.numel())
            dbl = []
            if n_div:
                dbl = (cb.ready_t[rd] - cb.birth_t[rd]).cpu().numpy().tolist()
                cb.divide(rd, torch.full((n_div,), 0.5, device=cb.dev), torch.ones(n_div, dtype=torch.bool, device=cb.dev))
            sat = cb.atp / (p["atp_km"] + cb.atp)
            _jl(fh, {"t_s": float(cb.t), "glucose": float(cb.glucose.mean()), "atp_mean": float(cb.atp.mean()),
                     "sat_mean": float(sat.mean()), "divisions": n_div, "doubling_s": dbl})
            progress("P7", min(float(cb.t) / t_total, 1.0), f"t={float(cb.t):.0f}/{t_total:.0f}s")
    return {"batch": batch, "glucose_step_to": level, "t_step_s": t_step}


# --------------------------------------------------------------------------- full-load v2 benchmark
def fullload_v2(out: Path, batch: int = 32768, seed: int = 12, progress: Progress = _noop) -> Dict[str, Any]:
    """Same single-cycle benchmark as M0-M3 full load, with the CUDA-graph engine (dt = 1 s)."""
    import torch
    p = load_params(); genes = build_gene_table(p)
    t0 = time.perf_counter()
    cb = CellBatch(p, genes, batch, dt=1.0, seed=seed, steps_per_graph=60)
    while not bool((cb.frozen > 0.5).all()) and float(cb.t) < 40000:
        cb.advance()
        progress("FULLLOAD_V2", min(float(cb.t) / 6500.0, 1.0), f"t={float(cb.t):.0f}s")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    d = (cb.ready_t - cb.birth_t).cpu().numpy(); d = d[np.isfinite(d)]
    row = {"batch": batch, "wall_s": wall, "cells_divided": int(d.size), "median_s": float(np.median(d)),
           "cells_per_s": d.size / wall, "max_closure": float(cb.max_closure),
           "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
    with (out / "fullload_v2.jsonl").open("w") as fh:
        _jl(fh, row)
    return row
