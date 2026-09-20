"""Validate contracts without training / 僅驗證資料契約，不執行訓練。"""
import argparse
import json
from pathlib import Path

from training.datasets.contracts import PROFILES, load_rows, render_example, validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if bool(args.profile) != bool(args.dataset):
        parser.error("--profile and --dataset must be supplied together")
    results = []
    for profile in ([args.profile] if args.profile else PROFILES):
        validator(profile)
        path = args.dataset or Path(__file__).parents[1] / "datasets" / "examples" / f"{profile}.jsonl"
        rows = load_rows(path, profile)
        for row in rows:
            render_example(profile, row)
        results.append({"profile": profile, "schema_valid": True, "dataset": str(path),
                        "valid_rows": len(rows), "serialization_valid": True})
    print(json.dumps({"status": "valid", "datasets": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
