from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="target", required=True)

    clone = sub.add_parser("clone")
    clone.add_argument("--config", default="configs/clone/clone_eval_dataset.yaml")
    clone.add_argument("--outdir", default="runs/clone_validation")

    asal = sub.add_parser("asal")
    asal.add_argument("--config", default="configs/asal/target_cell_eval.yaml")
    asal.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    asal.add_argument("--outdir", default="runs/asal_target_validation_eval")

    args = ap.parse_args()

    if args.target == "clone":
        from digital_clone.eval import run_clone_eval

        summary = run_clone_eval(args.config, args.outdir)
        print(summary)
    elif args.target == "asal":
        from research.asal_engine.eval_target_search import main as asal_main

        sys.argv = [
            "eval_target_search.py",
            "--config",
            args.config,
            "--seeds",
            args.seeds,
            "--outdir",
            args.outdir,
        ]
        asal_main()


if __name__ == "__main__":
    main()
