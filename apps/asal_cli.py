from core.config import load_config
from core.logger import make_run_dir
from research.asal_engine.engine import ASALEngine

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir("runs")
    ASALEngine(cfg, run_dir).run()
    print("Done. Artifacts:", run_dir)

if __name__ == "__main__":
    main()
