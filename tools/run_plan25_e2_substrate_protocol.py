#!/usr/bin/env python3
"""Plan 25 E2: frozen multi-seed NCA/Lenia visual protocol under runs/asal/."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from core.artifacts import save_gif, save_image
from core.config import load_config
from core.logger import iso_now, make_run_dir, save_json
from core.runtime import RuntimeManager
from research.asal_engine.engine import ASALEngine


def run_one(config_path: str, profile: str | None, seed: int) -> dict:
    cfg = load_config(config_path, profile=profile)
    cfg.setdefault("search", {})["seed"] = int(seed)
    # Keep tiny for protocol speed
    search = cfg.setdefault("search", {})
    search["iters"] = min(int(search.get("iters", 2)), 2)
    search["pop"] = min(int(search.get("pop", 4)), 4)
    search["keep"] = min(int(search.get("keep", 4)), 4)
    runtime = cfg.setdefault("runtime", {})
    runtime["steps"] = min(int(runtime.get("steps", 64)), 64)
    run_dir = make_run_dir("runs/asal")
    t0 = time.time()
    result = ASALEngine(cfg, run_dir).run()
    elapsed = time.time() - t0
    return {
        "config": config_path,
        "profile": profile,
        "seed": seed,
        "run_dir": str(run_dir),
        "elapsed_sec": elapsed,
        "best_score": result.get("best_score"),
        "substrate": result.get("substrate"),
        "gif": str(run_dir / "best.gif"),
        "png": str(run_dir / "best.png"),
        "num_frames": result.get("num_frames"),
        "narrative_accepted": result.get("narrative_accepted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,21,31")
    ap.add_argument("--outdir", default="runs/asal/plan25_e2_substrate_protocol_20260920")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    matrix = [
        ("configs/asal/nca_baseline.yaml", None, "nca"),
        ("configs/asal/lenia_baseline.yaml", None, "lenia"),
    ]
    for config, profile, label in matrix:
        for seed in seeds:
            print(f"E2_RUN {label} seed={seed}", flush=True)
            try:
                row = run_one(config, profile, seed)
                row["label"] = label
                row["ok"] = True
                row["error"] = None
            except Exception as exc:
                row = {"label": label, "config": config, "seed": seed, "ok": False, "error": str(exc)}
                print(f"E2_FAIL {label} seed={seed}: {exc}", flush=True)
            rows.append(row)
            print(f"E2_DONE {label} seed={seed} ok={row.get('ok')} score={row.get('best_score')}", flush=True)

    # Honest notes: these substrates already implement real update rules;
    # this protocol measures runnable multi-seed artifact production, not curated species quality.
    report = {
        "schema_version": "plan25-e2-1",
        "created_at": iso_now(),
        "purpose": "Frozen multi-seed NCA/Lenia protocol for Plan 25 E2",
        "limits": [
            "NCA uses Distill-style local perception with untrained random weights unless weights_path set",
            "Lenia uses FFT ring kernel + Gaussian growth; default patch is not a curated persistent species",
            "Matched tiny search budget; not a claim of biological cell quality",
        ],
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_total": len(rows),
        "rows": rows,
    }
    save_json(outdir / "protocol_results.json", report)
    md = ["# Plan 25 E2 substrate protocol", "", f"ok {report['n_ok']}/{report['n_total']}", ""]
    for r in rows:
        md.append(f"- {r.get('label')} seed {r.get('seed')}: ok={r.get('ok')} score={r.get('best_score')} run=`{r.get('run_dir')}` err={r.get('error')}")
    (outdir / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("E2_RESULTS", outdir / "protocol_results.json")


if __name__ == "__main__":
    main()
