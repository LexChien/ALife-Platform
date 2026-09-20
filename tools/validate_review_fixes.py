"""Offline review-fix experiments / 離線修復驗證實驗。"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from core.config import load_config
from core.logger import save_json
from foundation_models import foundation_models
from foundation_models.openclip_adapter import OpenCLIPAdapter
from research.asal_engine.engine import ASALEngine
from research.asal_engine.replay import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    output = args.outdir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.monotonic()
    adapter = OpenCLIPAdapter(device="cpu")
    pixels = np.random.RandomState(9).randint(0, 256, (64, 64, 3)).astype(np.uint8)
    inputs = {"pil": Image.fromarray(pixels), "uint8": pixels}
    for dtype in (np.float16, np.float32, np.float64):
        inputs[f"{dtype.__name__}_unit"] = pixels.astype(dtype) / 255
        inputs[f"{dtype.__name__}_byte"] = pixels.astype(dtype)
    embeddings = {name: adapter.img_embed(value) for name, value in inputs.items()}
    report = {"real_openclip_weights": True, "device": adapter.device, "inputs": {}}
    for name, vector in embeddings.items():
        difference = float(np.max(np.abs(vector - embeddings["pil"])))
        report["inputs"][name] = {"shape": list(vector.shape), "finite": bool(np.isfinite(vector).all()),
                                  "max_abs_diff_from_pil": difference}
        if difference != 0:
            raise AssertionError(f"{name}: embeddings differ from PIL by {difference}")
    save_json(output / "image_inputs.json", report)

    cfg = load_config(ROOT / "configs/asal/target_cell_eval.yaml", profile="cpu_smoke")
    cfg["foundation_model"]["params"]["device"] = "cpu"
    create = foundation_models.create

    def cached_model(name, **params):
        return adapter if name == "openclip" else create(name, **params)

    runs = []
    with patch("research.asal_engine.scoring.foundation_models.create", side_effect=cached_model):
        for index in range(2):
            np.random.seed(700 + index)
            run_dir = output / f"rd_smoke_{index}"
            run = ASALEngine(cfg, run_dir).run()
            replay = verify_run(run_dir)
            save_json(run_dir / "replay_verification.json", replay)
            if not replay["verified"]:
                raise AssertionError(replay)
            runs.append({"best_score": run["best_score"], "best_theta": run["best_theta"],
                         "replay": replay, "mp4": run["mp4"]})
    with np.load(output / "rd_smoke_0/trajectory.npz") as first, np.load(output / "rd_smoke_1/trajectory.npz") as second:
        identical = bool(np.array_equal(first["frames"], second["frames"]))
    report["reaction_diffusion"] = {
        "runs": runs, "identical_frames": identical,
        "identical_theta": runs[0]["best_theta"] == runs[1]["best_theta"],
        "identical_score": runs[0]["best_score"] == runs[1]["best_score"],
    }
    if not all(report["reaction_diffusion"][key] for key in ("identical_frames", "identical_theta", "identical_score")):
        raise AssertionError("Fixed-seed repeat mismatch")
    save_json(output / "experiments.json", report)

    cfg = load_config(ROOT / "configs/asal/target_cell_fusion_narrative.yaml", profile="cpu_tiny")
    run_dir = output / "boids_cpu_tiny"
    run = ASALEngine(cfg, run_dir).run()
    replay = verify_run(run_dir)
    save_json(run_dir / "replay_verification.json", replay)
    if not replay["verified"]:
        raise AssertionError(replay)
    summary = json.loads((run_dir / "narrative_summary.json").read_text(encoding="utf-8"))
    report["boids"] = {
        "seed": run["seed"], "steps": run["num_frames"], "best_score": run["best_score"],
        "narrative_accepted": run["narrative_accepted"],
        "failure_reasons": run["narrative_failure_reasons"], "phase_order_valid": run["narrative_phase_order_valid"],
        "component_sequence": summary["actual_component_sequence"],
        "phase_sustain": [p["sustain_ratio"] for p in summary["phases"]],
        "phase_consecutive": [p["longest_qualified_run"] for p in summary["phases"]],
        "continuity_score": summary["continuity_score"], "replay": replay,
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save_json(output / "experiments.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
