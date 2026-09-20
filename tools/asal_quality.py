#!/usr/bin/env python3
"""Frozen-budget, paired-seed ASAL quality comparison and complete evidence export.

All labels describe measured geometry. The improved simulator uses explicit
phase forces and the default judge is a heuristic, not an OpenCLIP semantic test.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from core.config import load_config
from core.logger import build_run_summary, iso_now, save_json
from research.asal_engine.engine import ASALEngine
from research.asal_engine.provenance import provenance_snapshot, sha256_file, source_hashes
from research.asal_engine.replay import verify_run

STRICT_CLEAN_POLICY = {"min_phase_coverage": 0.65, "min_consecutive_ratio": 0.5,
                       "max_fragment_fraction": 0.1, "min_mass_retention": 0.8}


def strict_clean_gate(narrative, morphology):
    """Additional declared release gate; never alter the existing narrative score."""
    if not narrative:
        return None
    phases = []
    for phase in narrative["phases"]:
        start, end = phase["frame_range"]
        target = phase["target_components"]
        qualified = []
        for stats in morphology[start:end + 1]:
            total = stats["raw_foreground_area"]
            selected = sum(component["area"] for component in stats["components"][:target])
            fragment = 1 - selected / total if total else 1.0
            qualified.append(stats["dominant_num_components"] == target
                             and fragment <= STRICT_CLEAN_POLICY["max_fragment_fraction"])
        longest = current = 0
        for value in qualified:
            current = current + 1 if value else 0
            longest = max(longest, current)
        coverage = sum(qualified) / len(qualified)
        accepted = (coverage >= STRICT_CLEAN_POLICY["min_phase_coverage"]
                    and longest >= np.ceil(len(qualified) * STRICT_CLEAN_POLICY["min_consecutive_ratio"]))
        phases.append({"name": phase["name"], "accepted": bool(accepted),
                       "coverage": coverage, "longest_consecutive": longest})
    retained = narrative["continuity"]["minimum_mass_retention"]
    return {"accepted": bool(narrative["accepted"] and all(p["accepted"] for p in phases)
                             and retained >= STRICT_CLEAN_POLICY["min_mass_retention"]),
            "phases": phases, "minimum_mass_retention": retained, "policy": STRICT_CLEAN_POLICY}


def wilson_interval(successes, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total ** 2)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def contact_sheet(frames, target, labels=None, columns=10):
    width, height = 128, 144
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), (22, 25, 31))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        x, y = (index % columns) * width, (index // columns) * height
        canvas.paste(Image.fromarray(np.asarray(frame)).resize((128, 128)), (x, y))
        draw.text((x + 3, y + 129), labels[index] if labels else f"frame {index}", fill="white")
    canvas.save(target)


def video_audit(directory):
    path = directory / "best.mp4"
    if not path.is_file():
        return {"decoded": False, "error": "MP4 missing"}
    frames = []
    with imageio.get_reader(path) as reader:
        meta = reader.get_meta_data()
        for frame in reader:
            frames.append(frame)
    contact_sheet(frames, directory / "decoded_video_contact_sheet.png")
    return {"decoded": True, "frame_count": len(frames), "fps": float(meta.get("fps", 0)),
            "duration_seconds": float(meta.get("duration", 0)),
            "contact_sheet": "decoded_video_contact_sheet.png"}


def run_case(config, directory, config_path):
    started = iso_now()
    start = time.monotonic()
    result = ASALEngine(config, directory).run()
    elapsed = time.monotonic() - start
    summary = build_run_summary(
        system="asal", run_dir=directory, config_path=config_path, mode=result["mode"],
        started_at=started, completed_at=iso_now(),
        metrics={key: result.get(key) for key in ("best_score", "num_frames", "search_iters",
                 "search_pop", "search_keep", "replay_matches", "narrative_accepted", "narrative_score")},
        artifacts={key: result.get(key) for key in ("gif", "mp4", "raw_trajectory", "simulation_states",
                   "substrate_stats", "resolved_config", "scores", "replay_manifest",
                   "narrative_summary", "trajectory_stats")},
        details={**result, "elapsed_seconds": elapsed},
    )
    save_json(directory / "summary.json", summary)
    replay = verify_run(directory, resimulate=True)
    save_json(directory / "replay_verification.json", replay)
    audit = video_audit(directory)
    save_json(directory / "video_audit.json", audit)
    narrative = {}
    if result.get("narrative_summary"):
        narrative = json.loads((directory / result["narrative_summary"]).read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        frames = data["frames"]
    morphology = []
    if result.get("trajectory_stats"):
        morphology = json.loads((directory / result["trajectory_stats"]).read_text())
    labels = [f"{index:02d} bodies={stats['dominant_num_components']}"
              for index, stats in enumerate(morphology)] or None
    contact_sheet(frames, directory / "all_frames_contact_sheet.png", labels)
    phase_rows = [{key: phase[key] for key in ("name", "accepted", "qualified_sustain_ratio",
                  "longest_qualified_run", "fragment_fraction", "failure_reasons")}
                  for phase in narrative.get("phases", [])]
    return {
        "run_dir": str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
        "seed": config["search"]["seed"], "elapsed_seconds": elapsed,
        "accepted": result.get("narrative_accepted"), "narrative_score": result.get("narrative_score"),
        "failure_reasons": result.get("narrative_failure_reasons"),
        "sequence": narrative.get("actual_component_sequence"), "phases": phase_rows,
        "strict_clean_gate": strict_clean_gate(narrative, morphology),
        "continuity": narrative.get("continuity"), "replay": replay,
        "video": audit, "final_stats": result.get("substrate_final_stats"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seeds", default="11,21,31,41,51")
    parser.add_argument("--include-automata", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be a nonempty list of distinct integers")
    directory = args.outdir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    improved_path = "configs/asal/controlled_cell_fusion.yaml"
    baseline_path = "configs/asal/target_cell_fusion_narrative.yaml"
    improved = load_config(ROOT / improved_path)
    baseline = load_config(ROOT / baseline_path, profile="cpu_smoke")
    baseline["runtime"] = deepcopy(improved["runtime"])
    baseline["narrative"] = deepcopy(improved["narrative"])
    for key in ("iters", "pop", "keep"):
        baseline["search"][key] = improved["search"][key]
    snapshots = provenance_snapshot()
    candidates = improved["search"]["keep"] + improved["search"]["iters"] * improved["search"]["pop"]
    protocol = {
        "schema_version": "1.0", "created_at": iso_now(), "seeds": seeds,
        "search_candidates_per_seed": candidates,
        "frames_per_candidate": improved["runtime"]["steps"],
        "substeps_per_frame": improved["runtime"]["substeps"],
        "additional_winner_reruns": 2,
        "budget_definition": "Equal candidate counts, frames, seeds, scoring, and acceptance; wall time is measured, not fixed.",
        "judge": "morphology_judge_stub (hand-built geometry features; no OpenCLIP semantic claim)",
        "control": "Explicit phase-conditioned external forces on conserved particles; no autonomous reproduction claim.",
        "acceptance": improved["narrative"]["acceptance"],
        "additional_strict_clean_policy": STRICT_CLEAN_POLICY,
        "provenance": snapshots,
        "baseline_config": baseline, "improved_config": improved,
        "scope": "Paired local regression benchmark, not population-wide proof of reliability.",
    }
    save_json(directory / "protocol.json", protocol)
    paths = sorted(set(snapshots["source_sha256"]) | {improved_path, baseline_path,
                   "configs/asal/nca_baseline.yaml", "configs/asal/lenia_baseline.yaml", "tools/asal_quality.py"})
    with tarfile.open(directory / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(ROOT / path, arcname=path, recursive=False)
    rows = []
    for label, config, path in (("boids_baseline", baseline, baseline_path),
                                ("controlled_cells", improved, improved_path)):
        for seed in seeds:
            case = deepcopy(config)
            case["search"]["seed"] = seed
            print(f"RUN {label} seed={seed}", flush=True)
            row = run_case(case, directory / f"{label}_seed{seed}", path)
            row["variant"] = label
            rows.append(row)
            save_json(directory / "progress.json", {"rows": rows})
            print(f"DONE {label} seed={seed} accepted={row['accepted']} score={row['narrative_score']} replay={row['replay']['verified']}", flush=True)
    ablated = deepcopy(improved)
    ablated["search"]["seed"] = seeds[0]
    ablated["search"]["theta_low"][3] = 0.0
    ablated["search"]["theta_high"][3] = 0.0
    print("RUN negative control: split force ablated", flush=True)
    negative_control = run_case(ablated, directory / "split_force_ablated", improved_path)
    negative_control["intervention"] = "Set split-distance bounds to zero; preserve every other budget and rule."
    automata = []
    if args.include_automata:
        for name in ("nca", "lenia"):
            path = f"configs/asal/{name}_baseline.yaml"
            print(f"RUN {name} baseline", flush=True)
            row = run_case(load_config(ROOT / path), directory / f"{name}_baseline", path)
            row["substrate"] = name
            automata.append(row)
    aggregate = {}
    for label in ("boids_baseline", "controlled_cells"):
        values = [row for row in rows if row["variant"] == label]
        success = sum(row["accepted"] is True for row in values)
        clean_success = sum(row["strict_clean_gate"]["accepted"] for row in values)
        aggregate[label] = {"accepted": success, "total": len(values),
                            "success_rate": success / len(values),
                            "wilson_95_interval": wilson_interval(success, len(values)),
                            "strict_clean_accepted": clean_success,
                            "strict_clean_success_rate": clean_success / len(values),
                            "strict_clean_wilson_95_interval": wilson_interval(clean_success, len(values)),
                            "mean_narrative_score": float(np.mean([row["narrative_score"] for row in values])),
                            "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in values)}
    report = {"protocol": "protocol.json", "source_snapshot": "source_snapshot.tar.gz",
              "source_snapshot_sha256": sha256_file(directory / "source_snapshot.tar.gz"),
              "source_stable_during_run": snapshots["source_sha256"] == source_hashes(),
              "aggregate": aggregate, "rows": rows, "automata": automata,
              "negative_control": negative_control,
              "all_replays_verified": all(row["replay"]["verified"] for row in rows + automata + [negative_control]),
              "all_videos_decoded": all(row["video"]["decoded"] for row in rows + automata + [negative_control]),
              "manual_visual_review": "pending; contact sheets must be inspected and recorded separately"}
    save_json(directory / "report.json", report)
    print(json.dumps({"outdir": str(directory), **aggregate}, indent=2), flush=True)
    return 0 if report["all_replays_verified"] and report["all_videos_decoded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
