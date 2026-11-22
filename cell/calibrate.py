"""Calibrate the one free growth rate to the cited syn3A doubling time.

ZH: k_lipid_per_enzyme_per_s 在文獻中查不到，因此以「中位倍增時間 = 105 分鐘（C0715/C0862 實測）」
    反解它。這是校準，不是獨立預測；M3 的倍增時間門檻因此只是離散度檢查。
EN: k_lipid_per_enzyme_per_s has no literature value, so it is solved for by matching the
    cited 105 min median doubling time. This is calibration, not an independent prediction;
    the M3 doubling-time gate is therefore a dispersion check.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .driver import RunConfig, run_single
from .genome_map import build_gene_table
from .params import load_params


def median_doubling(k_lipid: float, seeds=(0, 1, 2), max_time_s: float = 40000.0) -> float:
    params = load_params()
    genes = build_gene_table(params)
    vals = []
    for s in seeds:
        res = run_single(params, genes, RunConfig(max_time_s=max_time_s, seed=s, max_generations=1,
                                                  overrides={"k_lipid_per_enzyme_per_s": k_lipid}))
        if res.doubling_times_s:
            vals.append(res.doubling_times_s[0])
        else:
            vals.append(float("inf"))
    return float(np.median(vals))


def solve(target_s: float = 6300.0, lo: float = 1e-4, hi: float = 10.0,
          tol_s: float = 60.0, max_iter: int = 30) -> Tuple[float, Dict[str, float]]:
    """Bisection on k_lipid (monotone: more lipid synthesis -> faster doubling)."""
    history = {}
    f_lo = median_doubling(lo); f_hi = median_doubling(hi)
    history[lo] = f_lo; history[hi] = f_hi
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5  # geometric bisection, rates span orders of magnitude
        f_mid = median_doubling(mid)
        history[mid] = f_mid
        if abs(f_mid - target_s) <= tol_s:
            return mid, history
        if f_mid > target_s:      # too slow -> need more lipid synthesis
            lo = mid
        else:
            hi = mid
    best = min(history.items(), key=lambda kv: abs(kv[1] - target_s))
    return best[0], history
