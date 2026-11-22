#!/usr/bin/env python3
"""Plan 44 M0-M3 gate runner (CPU multiprocessing + Jetson GPU batch).

Usage:
  python tools/run_cell_sim.py gates --out runs/cell_sim/<stamp> [--cpu-runs 24] [--gpu-batch 256]
  python tools/run_cell_sim.py calibrate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cell import load_params, build_gene_table  # noqa: E402
from cell import driver, ensemble, expression_tx_tl, gpu  # noqa: E402


def env_facts() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:
            return f"ERROR {exc!r}"
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "l4t": sh("cat /etc/nv_tegra_release 2>/dev/null | head -1"),
        "nvcc": sh("/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2 | head -1"),
        "nvpmodel": sh("nvpmodel -q 2>/dev/null | head -2 | tr '\\n' ' '"),
        "mem_total_kb": sh("awk '/MemTotal/{print $2}' /proc/meminfo"),
        "git_head": sh(f"git -C {ROOT} rev-parse --short HEAD"),
        "torch": gpu.cuda_report(),
    }


def gate_m1(params, genes, n_rep=8, sim_s=3000.0) -> dict:
    """mRNA steady state vs the analytic birth-death solution, plus the Poisson CV check."""
    analytic = expression_tx_tl.analytic_mrna_steady_state(params, genes)
    tail = []
    for seed in range(n_rep):
        res = driver.run_single(params, genes, driver.RunConfig(max_time_s=sim_s, seed=1000 + seed,
                                                               max_generations=99, telemetry_every_s=10.0))
        rows = [r for r in res.telemetry if r["t_s"] >= sim_s * 0.5]
        tail.extend(r["mrna_total"] for r in rows)
    measured_mean = float(np.mean(tail))
    analytic_total = float(analytic.sum())
    rel_err = abs(measured_mean - analytic_total) / analytic_total
    cv_measured = float(np.std(tail) / measured_mean)
    cv_poisson = 1.0 / math.sqrt(analytic_total)
    cv_ratio = cv_measured / cv_poisson
    return {
        "analytic_total_mrna": analytic_total,
        "measured_mean_total_mrna": measured_mean,
        "relative_error": rel_err,
        "gate_mean_within_5pct": bool(rel_err <= 0.05),
        "cv_measured": cv_measured,
        "cv_poisson_expected": cv_poisson,
        "cv_ratio": cv_ratio,
        "gate_cv_within_20pct": bool(abs(cv_ratio - 1.0) <= 0.20),
        "n_samples": len(tail),
    }


def gate_m2(params, genes, sim_s=6000.0) -> dict:
    res = driver.run_single(params, genes, driver.RunConfig(max_time_s=sim_s, seed=7, max_generations=99,
                                                            telemetry_every_s=30.0))
    bp_total = params["genome_bp"]
    atp_per_bp = params["atp_per_bp_replication"]
    return {
        "max_closure_error": res.max_closure_error,
        "gate_closure_le_1pct": bool(res.max_closure_error <= 0.01),
        "atp_shortfall_total": res.atp_shortfall_total,
        "atp_per_bp_replication": atp_per_bp,
        "gate_atp_per_bp_is_1": bool(abs(atp_per_bp - 1.0) < 1e-12),
        "replication_atp_for_full_genome": bp_total * atp_per_bp,
    }


def gate_m3(cpu_runs: int, gpu_batch: int, max_time_s: float) -> dict:
    cpu = ensemble.run_cpu_ensemble(cpu_runs, max_time_s=max_time_s, max_generations=1)
    cpu_dts = [d for r in cpu["runs"] for d in r["doubling_times_s"]]
    out = {"cpu": {"processes": cpu["processes"], "n_runs": cpu["n_runs"],
                   "wall_time_s": cpu["wall_time_s"],
                   "wall_time_per_run_s": cpu["wall_time_s"] / max(1, cpu["n_runs"]),
                   "n_divided": len(cpu_dts),
                   "doubling_time_median_s": float(np.median(cpu_dts)) if cpu_dts else None,
                   "doubling_time_min_s": float(np.min(cpu_dts)) if cpu_dts else None,
                   "doubling_time_max_s": float(np.max(cpu_dts)) if cpu_dts else None,
                   "max_closure_error": max(r["max_closure_error"] for r in cpu["runs"])}}
    try:
        out["gpu"] = gpu.run_batch(batch=gpu_batch, config=driver.RunConfig(max_time_s=max_time_s, seed=11))
    except Exception as exc:
        out["gpu"] = {"error": repr(exc)[:300]}
    dts = list(cpu_dts)
    g = out.get("gpu", {})
    if isinstance(g, dict):
        dts += list(g.get("doubling_time_s") or [])
    arr = np.asarray(dts, dtype=float)
    lo, hi = 5280.0, 6720.0   # 88-112 min, C0862
    out["combined"] = {
        "n": int(arr.size),
        "median_s": float(np.median(arr)) if arr.size else None,
        "min_s": float(arr.min()) if arr.size else None,
        "max_s": float(arr.max()) if arr.size else None,
        "fraction_in_88_112_min": float(np.mean((arr >= lo) & (arr <= hi))) if arr.size else 0.0,
        "gate_all_in_88_112_min": bool(arr.size and np.all((arr >= lo) & (arr <= hi))),
        "gate_wall_time_per_cycle_le_30min": bool(out["cpu"]["wall_time_per_run_s"] <= 1800.0),
        "note": "k_lipid_per_enzyme_per_s was CALIBRATED to the cited 105 min median; this gate is a dispersion check, not an independent prediction.",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["gates", "calibrate", "env"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu-runs", type=int, default=24)
    ap.add_argument("--gpu-batch", type=int, default=256)
    ap.add_argument("--max-time-s", type=float, default=40000.0)
    args = ap.parse_args()

    params = load_params()
    genes = build_gene_table(params)

    if args.command == "env":
        print(json.dumps(env_facts(), indent=2))
        return 0
    if args.command == "calibrate":
        from cell import calibrate
        k, history = calibrate.solve()
        print(json.dumps({"k_lipid_per_enzyme_per_s": k,
                          "history": {str(kk): vv for kk, vv in history.items()}}, indent=2))
        return 0

    out_dir = Path(args.out or (ROOT / "runs/cell_sim" / time.strftime("%Y%m%d_%H%M")))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "env": env_facts(),
              "provenance": params.provenance_report(),
              "params_source": {k: params.source(k) for k in params.constants}}
    t0 = time.perf_counter()
    report["M1_expression"] = gate_m1(params, genes)
    report["M2_atp_ledger"] = gate_m2(params, genes)
    report["M3_growth"] = gate_m3(args.cpu_runs, args.gpu_batch, args.max_time_s)
    report["total_wall_time_s"] = time.perf_counter() - t0
    (out_dir / "gates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("M")}, indent=2)[:4000])
    print("wrote", out_dir / "gates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
