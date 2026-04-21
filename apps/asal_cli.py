from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import make_run_dir
from research.asal_engine.engine import ASALEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir("runs/asal")
    ASALEngine(cfg, run_dir).run()
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
