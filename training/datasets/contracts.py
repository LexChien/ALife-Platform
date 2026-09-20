"""Validated JSONL rows and explicit runtime-compatible SFT serialization."""
from functools import lru_cache
import json
import math
from pathlib import Path

from jsonschema import Draft7Validator

PROFILES = ("clone", "planner", "judge")


@lru_cache(maxsize=3)
def validator(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unsupported dataset profile: {profile}")
    schema = json.loads(Path(__file__).with_name(f"{profile}.schema.json").read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def _check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Dataset numbers must be finite")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def validate_row(profile, row):
    _check_finite(row)
    validator(profile).validate(row)
    if profile == "planner":
        bounds = row["hyperparameters"]
        if any(low > high for low, high in zip(bounds["theta_low"], bounds["theta_high"])):
            raise ValueError("Every theta_low value must be <= theta_high")
        phases = [item["phase"] for item in row["expected_phases"]]
        ordering = {"birth": 0, "split": 1, "fusion": 2}
        if len(set(phases)) != len(phases) or phases != sorted(phases, key=ordering.get):
            raise ValueError("Phases must be unique and ordered birth, split, fusion")
    return row


def load_rows(path, profile):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(validate_row(profile, json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        except Exception as exc:
            # jsonschema ValidationError carries a long schema; keep the row location.
            raise ValueError(f"{path}:{line_number}: {exc.message if hasattr(exc, 'message') else exc}") from exc
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def render_example(profile, row):
    validate_row(profile, row)
    if profile == "clone":
        return {"prompt_profile": profile, "system": row["system"], "context": row.get("context", ""),
                "prompt": row["prompt"], "response": row["response"]}
    if profile == "judge":
        return {"prompt_profile": profile, "system": "Evaluate the supplied measured morphology features.",
                "context": json.dumps(row["image_metadata"], sort_keys=True, ensure_ascii=False),
                "prompt": row["rubric"], "response": f"score={row['score']:g} feedback={row['feedback']}"}
    plan = {"expected_phases": row["expected_phases"], "hyperparameters": row["hyperparameters"]}
    return {"prompt_profile": profile, "system": "Produce a structured, testable simulation plan.",
            "context": "", "prompt": row["instruction"],
            "response": "plan=" + json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + " next_step=" + row["next_step"]}
