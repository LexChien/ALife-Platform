"""Verify saved lossless frames with the same scoring formula as search.

Usage: python -m research.asal_engine.replay RUN_DIRECTORY
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import SCORE_ATOL, sha256_file, source_hashes
from .scoring import TrajectoryScorer, resolve_config
from .substrates import substrates


def verify_run(run_dir, resimulate=False):
    directory = Path(run_dir)
    manifest = json.loads((directory / "replay_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported replay manifest version")
    expected_files = {"trajectory.npz", "resolved_config.json", "scores.json"}
    supplied = manifest.get("files_sha256", {})
    file_errors = [name for name in expected_files if name not in supplied]
    for name, expected in supplied.items():
        if Path(name).name != name:
            raise ValueError("Replay artifacts must be files inside the run directory")
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            file_errors.append(name)
    current = source_hashes()
    saved = manifest["provenance"]["source_sha256"]
    changed_sources = sorted(key for key in set(current) | set(saved) if current.get(key) != saved.get(key))
    result = {"verified": False, "file_hashes_match": not file_errors,
              "file_errors": sorted(set(file_errors)), "source_matches": not changed_sources,
              "changed_sources": changed_sources, "score_matches": False}
    if file_errors:
        return result
    cfg = resolve_config(json.loads((directory / "resolved_config.json").read_text(encoding="utf-8")))
    expected = json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        frames = data["frames"]
    if list(frames.shape) != manifest["trajectory"]["shape"] or str(frames.dtype) != manifest["trajectory"]["dtype"]:
        raise ValueError("Trajectory shape or dtype differs from the saved manifest")
    actual, narrative = TrajectoryScorer(cfg).score(frames)

    def same(left, right):
        if left is None or right is None:
            return left is right
        return bool(np.isclose(left, right, atol=SCORE_ATOL, rtol=0))

    component_matches = {key: same(value, expected["components"].get(key))
                         for key, value in actual["components"].items()}
    weights_match = all(same(value, expected["weights"].get(key)) for key, value in actual["weights"].items())
    narrative_matches = True
    if narrative is not None:
        summary_file = directory / "narrative_summary.json"
        if summary_file.is_file():
            saved_narrative = json.loads(summary_file.read_text(encoding="utf-8"))
            narrative_matches = (
                saved_narrative["accepted"] == narrative["accepted"]
                and saved_narrative["failure_reasons"] == narrative["failure_reasons"]
                and same(saved_narrative["total_score"], narrative["total_score"])
            )
        else:
            narrative_matches = False
    score_matches = (all(component_matches.values()) and weights_match and narrative_matches
                     and same(actual["combined"], expected["combined"])
                     and same(actual["combined"], expected["search_best_score"])
                     and expected["search_replay_matches"])
    result.update(
        score_matches=bool(score_matches), component_matches=component_matches,
        weights_match=weights_match, narrative_matches=narrative_matches,
        recomputed_score=actual["combined"], recorded_score=expected["combined"],
        absolute_difference=abs(actual["combined"] - expected["combined"]), absolute_tolerance=SCORE_ATOL,
        narrative_accepted=narrative["accepted"] if narrative else None,
        verified=bool(score_matches and not changed_sources),
    )
    if resimulate:
        substrate = substrates.create(cfg["substrate"]["name"], **cfg["substrate"].get("params", {}))
        narrative_cfg = cfg.get("narrative", {})
        if hasattr(substrate, "configure_narrative") and narrative_cfg.get("enabled", False):
            substrate.configure_narrative(cfg["runtime"]["steps"], narrative_cfg["phases"])
        substrate.reset(manifest["best_theta"], seed=manifest["seed"])
        mismatched_frames = []
        mismatched_states = []
        state_file = directory / "simulation_states.npz"
        states = {}
        if state_file.is_file():
            with np.load(state_file, allow_pickle=False) as saved_states:
                states = {key: saved_states[key] for key in saved_states.files}
        for index, expected_frame in enumerate(frames):
            substrate.step(cfg["runtime"]["substeps"])
            if not np.array_equal(np.asarray(substrate.render()), expected_frame):
                mismatched_frames.append(index)
            for name, values in states.items():
                if not np.array_equal(getattr(substrate, name, None), values[index]):
                    mismatched_states.append({"frame": index, "field": name})
        result.update(resimulated=True, mismatched_frames=mismatched_frames,
                      mismatched_states=mismatched_states,
                      resimulation_matches=not mismatched_frames and not mismatched_states)
        result["verified"] = result["verified"] and result["resimulation_matches"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--resimulate", action="store_true", help="Also reconstruct frames and state from seed/theta")
    args = parser.parse_args()
    result = verify_run(args.run_dir, resimulate=args.resimulate)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
