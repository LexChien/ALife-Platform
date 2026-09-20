from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "requires jsonschema; run with system python3")
class TestDatasetContracts(unittest.TestCase):
    def setUp(self):
        from training.datasets.contracts import load_rows, validate_row, render_example
        self.load = load_rows
        self.validate = validate_row
        self.render = render_example
        self.root = Path(__file__).resolve().parents[1] / "training" / "datasets"

    def sample(self, profile):
        return self.load(self.root / "examples" / f"{profile}.jsonl", profile)[0]

    def test_schemas_examples_and_documented_schemas_agree(self):
        from jsonschema import Draft7Validator
        for profile in ("clone", "planner", "judge"):
            schema = json.loads((self.root / f"{profile}.schema.json").read_text())
            Draft7Validator.check_schema(schema)
            documented = re.search(r"```json\n(.*?)\n```", (self.root / f"{profile}_schema.md").read_text(), re.S)
            self.assertEqual(json.loads(documented.group(1)), schema)
            self.validate(profile, self.sample(profile))

    def test_invalid_fields_types_and_nonfinite_numbers_fail(self):
        from jsonschema import ValidationError
        changes = [("judge", "score", 1.1), ("judge", "score", "0.5"), ("judge", "score", float("nan")),
                   ("judge", "feedback", "score=0.5 feedback=duplicate"),
                   ("clone", "response", "missing format"), ("planner", "next_step", "")]
        for profile, key, value in changes:
            row = self.sample(profile)
            row[key] = value
            with self.subTest(profile=profile, key=key), self.assertRaises((ValueError, ValidationError)):
                self.validate(profile, row)
        row = self.sample("clone")
        row["unknown"] = "not in contract"
        with self.assertRaises(ValidationError):
            self.validate("clone", row)

    def test_planner_cross_field_bounds_and_phase_targets(self):
        from jsonschema import ValidationError
        row = self.sample("planner")
        variants = [deepcopy(row) for _ in range(4)]
        variants[0]["hyperparameters"]["theta_low"][0] = 99
        variants[1]["expected_phases"][1]["target_components"] = 1
        variants[2]["expected_phases"][0]["steps"] = 1
        variants[3]["expected_phases"].reverse()
        for variant in variants:
            with self.assertRaises((ValueError, ValidationError)):
                self.validate("planner", variant)

    def test_serialization_and_format_eval_match_runtime_profiles(self):
        from training.eval.evaluate_predictions import evaluate
        for profile in ("clone", "planner", "judge"):
            row = self.sample(profile)
            rendered = self.render(profile, row)
            self.assertEqual(rendered["prompt_profile"], profile)
            self.assertEqual(evaluate(profile, [row], [rendered])["format_pass_rate"], 1.0)
            self.assertEqual(evaluate(profile, [row], [{"response": "invalid"}])["format_pass_rate"], 0.0)
        self.assertEqual(self.render("judge", self.sample("judge"))["response"].count("score="), 1)
        planner = self.render("planner", self.sample("planner"))["response"]
        plan = json.loads(planner[5:].rsplit(" next_step=", 1)[0])
        self.assertEqual(plan["expected_phases"], self.sample("planner")["expected_phases"])

    def test_preflight_records_data_without_claiming_trained_weights(self):
        from training.lora.train_placeholder import prepare
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare("clone", self.root / "examples" / "clone.jsonl", tmp, "declared-test-base")
            self.assertEqual(manifest["status"], "prepared_not_trained")
            self.assertIsNone(manifest["trained_weights"])
            self.assertFalse(manifest["base_model_reference_verified"])
            self.assertEqual(manifest["rows"], 1)
            rows = [json.loads(line) for line in (Path(tmp) / "prepared.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["response"], self.sample("clone")["response"])


if __name__ == "__main__":
    unittest.main()
