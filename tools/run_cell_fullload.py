#!/usr/bin/env python3
"""Saturate CPU (all cores) and the Jetson GPU at the same time with the cell model.

ZH: Lex 2026-11-22 要求 CPU 與 GPU 全頻跑；CPU 多行程 ensemble 與 GPU 批次同時執行。
EN: Runs the multiprocessing CPU ensemble and the CUDA batch concurrently.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cell import driver, ensemble, gpu  # noqa: E402


def _gpu_proc(batch: int, max_time_s: float, seed: int, q):
    try:
        res = gpu.run_batch(batch=batch, config=driver.RunConfig(max_time_s=max_time_s, seed=seed))
        res.pop("doubling_time_s", None)
        q.put(res)
    except Exception as exc:  # pragma: no cover
        q.put({"error": repr(exc)[:300]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-runs", type=int, default=600)
    ap.add_argument("--gpu-batch", type=int, default=16384)
    ap.add_argument("--gpu-repeats", type=int, default=3)
    ap.add_argument("--max-time-s", type=float, default=40000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.perf_counter()
    gpu_jobs = [ctx.Process(target=_gpu_proc, args=(args.gpu_batch, args.max_time_s, 100 + i, q))
                for i in range(args.gpu_repeats)]
    for p in gpu_jobs:
        p.start()
    cpu = ensemble.run_cpu_ensemble(args.cpu_runs, max_time_s=args.max_time_s, max_generations=1)
    gpu_results = []
    for _ in gpu_jobs:
        gpu_results.append(q.get())
    for p in gpu_jobs:
        p.join()
    wall = time.perf_counter() - t0
    dts = [d for r in cpu["runs"] for d in r["doubling_times_s"]]
    report = {
        "wall_time_s": wall,
        "cpu": {"processes": cpu["processes"], "n_runs": cpu["n_runs"], "wall_time_s": cpu["wall_time_s"],
                "cells_divided": len(dts),
                "doubling_time_median_s": (sorted(dts)[len(dts) // 2] if dts else None),
                "doubling_time_min_s": (min(dts) if dts else None),
                "doubling_time_max_s": (max(dts) if dts else None)},
        "gpu": gpu_results,
        "total_cells_simulated": cpu["n_runs"] + args.gpu_batch * args.gpu_repeats,
    }
    (out / "fullload.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
