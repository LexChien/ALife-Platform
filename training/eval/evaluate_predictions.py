"""Format-only evaluation scaffold; does not assess truth or model quality."""
import argparse
import json
import math
from pathlib import Path
import re

from training.datasets.contracts import PROFILES, load_rows


def evaluate(profile, rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("One prediction is required for each dataset row")
    valid = []
    squared_errors = []
    for row, prediction in zip(rows, predictions):
        text = prediction.get("response", "") if isinstance(prediction, dict) else ""
        passed = False
        if isinstance(text, str):
            if profile == "clone":
                passed = bool(re.fullmatch(r"\[[^\]\r\n]+\] tone=.+ principles=.+ response=.+", text))
            elif profile == "judge":
                match = re.fullmatch(r"score=([0-9.eE+-]+) feedback=(.+)", text, flags=re.DOTALL)
                if match:
                    try:
                        score = float(match.group(1))
                        passed = math.isfinite(score) and 0 <= score <= 1
                        if passed:
                            squared_errors.append((score - row["score"]) ** 2)
                    except ValueError:
                        pass
            elif text.startswith("plan=") and " next_step=" in text:
                plan, next_step = text[5:].rsplit(" next_step=", 1)
                try:
                    from training.datasets.contracts import validate_row
                    value = json.loads(plan)
                    validate_row("planner", {"instruction": row["instruction"], **value, "next_step": next_step})
                    passed = True
                except (ValueError, TypeError, KeyError):
                    pass
                except Exception:
                    passed = False
        valid.append(passed)
    return {"scope": "format_only", "rows": len(rows), "format_valid": sum(valid),
            "format_pass_rate": sum(valid) / len(rows) if rows else 0.0,
            "judge_rmse_on_valid_rows": math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None,
            "per_row_format_valid": valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help='JSONL with a "response" string per row')
    args = parser.parse_args()
    rows = load_rows(args.dataset, args.profile)
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(evaluate(args.profile, rows, predictions), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
