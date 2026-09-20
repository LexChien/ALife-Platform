#!/usr/bin/env python3
"""Q1 multi-seed ASAL narrative quality runner. Writes JSON results only; does not modify old runs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import build_run_summary, iso_now, make_run_dir, save_json
from core.runtime import RuntimeManager
from research.asal_engine.engine import ASALEngine


def run_one(config_path: str, profile: str, seed: int) -> dict:
    cfg = load_config(config_path, profile=profile)
    cfg.setdefault("search", {})["seed"] = int(seed)
    runtime = RuntimeManager(**cfg.get("runtime_profile", {}))
    run_dir = make_run_dir("runs/asal")
    started = iso_now()
    t0 = time.time()
    result = ASALEngine(cfg, run_dir).run()
    elapsed = time.time() - t0
    completed = iso_now()
    summary = build_run_summary(
        system="asal",
        run_dir=run_dir,
        config_path=config_path,
        mode=result["mode"],
        started_at=started,
        completed_at=completed,
        metrics={
            "best_score": result["best_score"],
            "num_frames": result["num_frames"],
            "search_iters": result["search_iters"],
            "search_pop": result["search_pop"],
            "search_keep": result["search_keep"],
            "replay_matches": result["replay_matches"],
            "narrative_accepted": result.get("narrative_accepted"),
            "q1_elapsed_sec": elapsed,
            "q1_requested_seed": seed,
            "q1_profile": profile,
        },
        artifacts={
            "image": "best.png",
            "gif": result["gif"],
            "mp4": result["mp4"],
            "narrative_summary": result.get("narrative_summary"),
            "trajectory_stats": result.get("trajectory_stats"),
            "raw_trajectory": result["raw_trajectory"],
            "resolved_config": result["resolved_config"],
            "scores": result["scores"],
            "replay_manifest": result["replay_manifest"],
        },
        details={
            "prompt": result["prompt"],
            "best_theta": result["best_theta"],
            "foundation_model": result["foundation_model"],
            "substrate": result["substrate"],
            "runtime": runtime.to_dict(),
            "active_profile": cfg.get("_active_profile"),
            "narrative_enabled": result.get("narrative_enabled"),
            "narrative_keyframes": result.get("narrative_keyframes"),
            "narrative_score": result.get("narrative_score"),
            "narrative_phase_order_valid": result.get("narrative_phase_order_valid"),
            "narrative_failure_reasons": result.get("narrative_failure_reasons"),
            "seed": result["seed"],
            "score_components": result["score_components"],
            "mp4_error": result.get("mp4_error"),
        },
    )
    save_json(run_dir / "summary.json", summary)

    narr_name = result.get("narrative_summary") or "narrative_summary.json"
    narr_path = run_dir / narr_name
    narr = {}
    if narr_path.exists():
        narr = json.loads(narr_path.read_text(encoding="utf-8"))

    return {
        "profile": profile,
        "requested_seed": seed,
        "result_seed": result.get("seed"),
        "run_dir": str(run_dir),
        "elapsed_sec": elapsed,
        "best_score": result.get("best_score"),
        "narrative_score": result.get("narrative_score"),
        "phase_order_valid": result.get("narrative_phase_order_valid"),
        "narrative_accepted": result.get("narrative_accepted"),
        "failure_reasons": result.get("narrative_failure_reasons"),
        "actual_component_sequence": narr.get("actual_component_sequence") or narr.get("dominant_component_sequence"),
        "narrative_summary": narr,
        "gif": result.get("gif"),
        "image": "best.png",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asal/target_cell_fusion_narrative.yaml")
    ap.add_argument("--profile", default="cpu_tiny")
    ap.add_argument("--seeds", default="11,21,31,41,51")
    ap.add_argument("--outdir", default="runs/asal/multiseed_results")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        print(f"Q1_RUN profile={args.profile} seed={seed}", flush=True)
        row = run_one(args.config, args.profile, seed)
        rows.append(row)
        print(
            f"Q1_DONE seed={seed} valid={row.get('phase_order_valid')} "
            f"accepted={row.get('narrative_accepted')} score={row.get('narrative_score')} "
            f"seq={row.get('actual_component_sequence')} dir={row.get('run_dir')}",
            flush=True,
        )
    payload = {
        "config": args.config,
        "profile": args.profile,
        "seeds": seeds,
        "rows": rows,
        "n_valid": sum(1 for r in rows if r.get("phase_order_valid") is True),
        "n_accepted": sum(1 for r in rows if r.get("narrative_accepted") is True),
        "n_total": len(rows),
    }
    out = outdir / f"q1_{args.profile}_results.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Q1_RESULTS", out)


if __name__ == "__main__":
    main()
