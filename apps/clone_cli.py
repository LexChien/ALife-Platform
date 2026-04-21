from core.config import load_config
from core.logger import make_run_dir, save_json
from digital_clone.engine import DigitalCloneEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir("runs")
    outputs = DigitalCloneEngine(cfg, run_dir).run()
    save_json(run_dir / "digital_clone_outputs.json", outputs)
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
