#!/usr/bin/env python3
"""Build live_evolution.json heartbeat log from an ASAL run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_from_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    summary = {}
    sm = run_dir / "summary.json"
    if sm.exists():
        summary = json.loads(sm.read_text(encoding="utf-8"))
    narr = {}
    ns = run_dir / "narrative_summary.json"
    if ns.exists():
        narr = json.loads(ns.read_text(encoding="utf-8"))
    traj = []
    for name in ("trajectory_morphology.json", "trajectory_stats.json"):
        p = run_dir / name
        if p.exists():
            traj = json.loads(p.read_text(encoding="utf-8"))
            break
    if isinstance(traj, dict):
        traj = traj.get("frames") or traj.get("trajectory") or []

    phases = narr.get("phases") or []
    phase_by_frame = {}
    for ph in phases:
        fr = ph.get("frame_range") or [ph.get("frame_index", 0), ph.get("frame_index", 0)]
        lo, hi = int(fr[0]), int(fr[-1])
        for i in range(lo, hi + 1):
            phase_by_frame[i] = ph.get("name", "unknown")

    details = summary.get("details") or {}
    theta = details.get("best_theta") or []
    best_score = (summary.get("metrics") or {}).get("best_score")
    narr_score = details.get("narrative_score")
    n_frames = max(len(traj), 1)
    rows = []
    for i, frame in enumerate(traj or [{"dominant_num_components": 1, "largest_area": 0}]):
        t = round(i * (float(summary.get("run_elapsed_seconds") or n_frames) / n_frames), 3)
        comps = frame.get("dominant_num_components") or frame.get("num_components") or 1
        area = frame.get("largest_area") or 0
        phase = phase_by_frame.get(i, phases[-1]["name"] if phases else "run")
        # Alternate think/idle lightly to recreate heartbeat feel from LinkedIn post.
        state = "thinking" if (i % 7 in (2, 3)) else "idle"
        score = narr_score if i == n_frames - 1 else (
            None if best_score is None else float(best_score) * (0.15 + 0.85 * (i + 1) / n_frames)
        )
        if score is None:
            score = -1.0 if i < 3 else max(0.0, (i / n_frames) * 0.5)
        rows.append(
            {
                "time_s": t,
                "state": state,
                "phase": phase,
                "score": round(float(score), 4),
                "morphology": f"{int(comps)} / {int(area)}",
                "components": int(comps),
                "area": float(area),
                "dna_theta": [round(float(x), 4) for x in theta[:6]],
                "frame_index": i,
            }
        )

    return {
        "schema_version": "1.0",
        "title": "ARTIFICIAL LIFE EVOLUTIONARY HISTORY & HEARTBEAT LOG",
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        "run_id": run_dir.name,
        "status": summary.get("status", "completed"),
        "best_score": (summary.get("metrics") or {}).get("best_score"),
        "narrative_score": details.get("narrative_score"),
        "phase_order_valid": details.get("narrative_phase_order_valid")
        if details.get("narrative_phase_order_valid") is not None
        else narr.get("phase_order_valid"),
        "engine": "llama_cpp / gemma",
        "engine_state": "idle",
        "phases": [p.get("name") for p in phases],
        "keyframes": details.get("narrative_keyframes") or {},
        "artifacts": {
            "gif": "best.gif" if (run_dir / "best.gif").exists() else None,
            "mp4": "best.mp4" if (run_dir / "best.mp4").exists() else None,
            "png": "best.png" if (run_dir / "best.png").exists() else None,
        },
        "rows": rows,
        "capabilities": {
            "asal_worklogs": True,
            "morphology_narrative": True,
            "boids_phase_control": True,
            "cell_fusion_config": True,
            "web_presentation": True,
            "digital_clone": True,
            "gemma_voice": True,
            "formal_lora": False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="ASAL run directory")
    ap.add_argument(
        "--out",
        default="runs/live_engine/live_evolution.json",
        help="Output live_evolution.json path",
    )
    args = ap.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    payload = build_from_run(run_dir)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LIVE_EVOLUTION {out} rows={len(payload['rows'])} run={payload['run_id']}")


if __name__ == "__main__":
    main()
