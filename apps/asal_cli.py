import os
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import build_run_summary, iso_now, make_run_dir, save_json
from core.runtime import RuntimeManager
from core.tracking import init_tracker


def _candidate_asal_python():
    override = os.environ.get("ALIFE_ASAL_PYTHON")
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path

    local_env = ROOT / ".venv" / "bin" / "python"
    if local_env.exists():
        return local_env

    legacy_env = Path("/home/lexchien/Documents/ASAL/.venv/bin/python")
    if legacy_env.exists():
        return legacy_env

    return None


def _current_cuda_ok():
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _should_bootstrap(candidate: Path | None):
    if candidate is None:
        return False
    if os.environ.get("ALIFE_ASAL_BOOTSTRAPPED") == "1":
        return False
    try:
        if candidate.samefile(sys.executable):
            return False
    except Exception:
        if str(candidate) == sys.executable:
            return False
    if _current_cuda_ok():
        return False
    return platform.machine() == "aarch64"


def _bootstrap_asal_python():
    candidate = _candidate_asal_python()
    if not _should_bootstrap(candidate):
        return

    env = os.environ.copy()
    env["ALIFE_ASAL_BOOTSTRAPPED"] = "1"

    # Preserve the known-good Jetson cuSPARSELt path used by the historical ASAL env.
    candidate_root = candidate.parent.parent
    cusparselt = candidate_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusparselt" / "lib"
    ld_parts = []
    if cusparselt.exists():
        ld_parts.append(str(cusparselt))
    existing_ld = env.get("LD_LIBRARY_PATH")
    if existing_ld:
        ld_parts.append(existing_ld)
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    os.execve(str(candidate), [str(candidate), *sys.argv], env)


_bootstrap_asal_python()

from research.asal_engine.engine import ASALEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile")
    args = ap.parse_args()
    cfg = load_config(args.config, profile=args.profile)
    runtime = RuntimeManager(**cfg.get("runtime_profile", {}))
    tracker = init_tracker(cfg.get("tracking"))
    if cfg.get("foundation_model", {}).get("name") == "openclip":
        fm_params = cfg.setdefault("foundation_model", {}).setdefault("params", {})
        fm_params.setdefault("device", runtime.device)
    run_dir = make_run_dir("runs/asal")
    started_at = iso_now()
    result = ASALEngine(cfg, run_dir).run()
    completed_at = iso_now()
    summary = build_run_summary(
        system="asal",
        run_dir=run_dir,
        config_path=args.config,
        mode=result["mode"],
        started_at=started_at,
        completed_at=completed_at,
        metrics={
            "best_score": result["best_score"],
            "num_frames": result["num_frames"],
            "search_iters": result["search_iters"],
            "search_pop": result["search_pop"],
            "search_keep": result["search_keep"],
        },
        artifacts={
            "image": "best.png",
            "gif": result["gif"],
            "mp4": result["mp4"],
            "narrative_summary": result.get("narrative_summary"),
            "trajectory_stats": result.get("trajectory_stats"),
        },
        details={
            "prompt": result["prompt"],
            "best_theta": result["best_theta"],
            "foundation_model": result["foundation_model"],
            "substrate": result["substrate"],
            "runtime": runtime.to_dict(),
            "tracking_backend": cfg.get("tracking", {}).get("backend") if tracker else None,
            "active_profile": cfg.get("_active_profile"),
            "narrative_enabled": result.get("narrative_enabled"),
            "narrative_keyframes": result.get("narrative_keyframes"),
            "narrative_score": result.get("narrative_score"),
            "narrative_phase_order_valid": result.get("narrative_phase_order_valid"),
        },
    )
    save_json(run_dir / "summary.json", summary)
    if tracker:
        tracker.log_summary(summary)
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
