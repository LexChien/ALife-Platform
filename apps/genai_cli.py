from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import build_run_summary, iso_now, make_run_dir, save_json
from core.runtime import RuntimeManager
from core.tracking import init_tracker
from genai.multimodal.engine import GenAIEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile")
    args = ap.parse_args()
    cfg = load_config(args.config, profile=args.profile)
    runtime = RuntimeManager(**cfg.get("runtime_profile", {}))
    tracker = init_tracker(cfg.get("tracking"))
    run_dir = make_run_dir("runs/genai")
    started_at = iso_now()
    result = GenAIEngine(cfg, run_dir).run()
    completed_at = iso_now()
    summary = build_run_summary(
        system="genai",
        run_dir=run_dir,
        config_path=args.config,
        mode=result["summary"]["mode"],
        started_at=started_at,
        completed_at=completed_at,
        metrics={
            "text_length": result["summary"]["text_length"],
            "has_context": result["summary"]["has_context"],
            "has_audio": result["summary"]["has_audio"],
            "has_image": result["summary"]["has_image"],
        },
        artifacts={
            "image": "generated.png",
            "output": "genai_output.json",
        },
        details={
            "prompt": result["summary"]["prompt"],
            "llm_backend": result["summary"]["llm_backend"],
            "llm_model_family": result["summary"]["llm_model_family"],
            "llm_runtime": result["summary"]["llm_runtime"],
            "llm_healthcheck": result["summary"]["llm_healthcheck"],
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
