"""CPU multiprocessing ensemble (all cores) + optional GPU batch, per Lex's D5 change."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional

from .driver import RunConfig, run_single
from .genome_map import build_gene_table
from .params import load_params


def _worker(args):
    seed, config_kwargs = args
    params = load_params()
    genes = build_gene_table(params)
    cfg = RunConfig(seed=seed, **config_kwargs)
    res = run_single(params, genes, cfg)
    return {
        "seed": seed,
        "doubling_times_s": res.doubling_times_s,
        "wall_time_s": res.wall_time_s,
        "max_closure_error": res.max_closure_error,
        "atp_shortfall_total": res.atp_shortfall_total,
        "finished_reason": res.finished_reason,
    }


def run_cpu_ensemble(n_runs: int, processes: Optional[int] = None, **config_kwargs) -> Dict[str, Any]:
    procs = processes or os.cpu_count() or 1
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=procs) as pool:
        results: List[Dict[str, Any]] = pool.map(_worker, [(s, config_kwargs) for s in range(n_runs)])
    wall = time.perf_counter() - t0
    return {"processes": procs, "n_runs": n_runs, "wall_time_s": wall, "runs": results}
