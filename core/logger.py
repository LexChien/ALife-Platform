from pathlib import Path
from datetime import datetime
import json

def make_run_dir(base="runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = Path(base) / ts
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def iso_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_summary(
    *,
    system,
    run_dir,
    config_path,
    mode,
    started_at,
    completed_at,
    metrics=None,
    artifacts=None,
    details=None,
    status="completed",
):
    started_dt = datetime.fromisoformat(started_at)
    completed_dt = datetime.fromisoformat(completed_at)
    elapsed = max((completed_dt - started_dt).total_seconds(), 0.0)
    return {
        "schema_version": "1.0",
        "status": status,
        "system": system,
        "mode": mode,
        "config_path": str(config_path),
        "run_dir": str(Path(run_dir)),
        "started_at": started_at,
        "completed_at": completed_at,
        "run_elapsed_seconds": elapsed,
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "details": details or {},
    }
