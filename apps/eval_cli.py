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
    clone.add_argument("--outdir", default="runs/digital_clone/clone_validation")
    clone.add_argument("--profile")

    asal = sub.add_parser("asal")
    asal.add_argument("--config", default="configs/asal/target_cell_eval.yaml")
    asal.add_argument("--profile")
    asal.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    asal.add_argument("--outdir", default="runs/asal/asal_target_validation_eval")

    genai = sub.add_parser("genai")
    genai.add_argument("--config", default="configs/genai/genai_eval_dataset.yaml")
    genai.add_argument("--outdir", default="runs/genai/genai_validation")
    genai.add_argument("--profile")

    args = ap.parse_args()

    if args.target == "clone":
        from digital_clone.eval import run_clone_eval

        summary = run_clone_eval(args.config, args.outdir, profile=args.profile)
        print(summary)
    elif args.target == "asal":
        from research.asal_engine.eval_target_search import main as asal_main

        sys.argv = [
            "eval_target_search.py",
            "--config",
            args.config,
            "--profile",
            args.profile or "",
            "--seeds",
            args.seeds,
            "--outdir",
            args.outdir,
        ]
        asal_main()
    elif args.target == "genai":
        from genai.eval import run_genai_eval

        summary = run_genai_eval(args.config, args.outdir, profile=args.profile)
        print(summary)


if __name__ == "__main__":
    main()
