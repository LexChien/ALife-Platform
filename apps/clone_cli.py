from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import build_run_summary, iso_now, make_run_dir, save_json
from core.runtime import RuntimeManager
from core.tracking import init_tracker
from digital_clone.engine import DigitalCloneEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile")
    args = ap.parse_args()
    cfg = load_config(args.config, profile=args.profile)
    runtime = RuntimeManager(**cfg.get("runtime_profile", {}))
    tracker = init_tracker(cfg.get("tracking"))
    run_dir = make_run_dir("runs/digital_clone")
    started_at = iso_now()
    result = DigitalCloneEngine(cfg, run_dir).run()
    completed_at = iso_now()
    save_json(run_dir / "digital_clone_outputs.json", result["outputs"])
    summary = build_run_summary(
        system="digital_clone",
        run_dir=run_dir,
        config_path=args.config,
        mode=result["summary"]["mode"],
        started_at=started_at,
        completed_at=completed_at,
        metrics={
            "num_inputs": result["summary"]["num_inputs"],
            "mean_consistency": result["summary"]["mean_consistency"],
            "min_consistency": result["summary"]["min_consistency"],
            "max_consistency": result["summary"]["max_consistency"],
            "retrieval_hit_rate": result["summary"]["retrieval_hit_rate"],
        },
        artifacts={
            "outputs": "digital_clone_outputs.json",
        },
        details={
            "persona_name": result["summary"]["persona_name"],
            "llm_backend": result["summary"].get("llm_backend"),
            "llm_model_family": result["summary"].get("llm_model_family"),
            "llm_runtime": result["summary"].get("llm_runtime"),
            "runtime": runtime.to_dict(),
            "tracking_backend": cfg.get("tracking", {}).get("backend") if tracker else None,
            "active_profile": cfg.get("_active_profile"),
        },
    )
    save_json(run_dir / "summary.json", summary)
    if tracker:
        tracker.log_summary(summary)
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
