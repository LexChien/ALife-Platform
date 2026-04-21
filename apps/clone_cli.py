from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import make_run_dir, save_json
from digital_clone.engine import DigitalCloneEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir("runs/digital_clone")
    outputs = DigitalCloneEngine(cfg, run_dir).run()
    save_json(run_dir / "digital_clone_outputs.json", outputs)
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
