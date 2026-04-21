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
