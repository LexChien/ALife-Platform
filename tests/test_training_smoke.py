"""Non-ML checks for held-out integrity and honest Clone scoring."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "requires jsonschema")
class CloneSmokeTests(unittest.TestCase):
    def test_dataset_is_reproducible_disjoint_and_tamper_evident(self):
        from training.datasets.build_clone_smoke import build
        from training.lora.clone_smoke import load_dataset
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            a, b = build(one), build(two)
            self.assertEqual(a, b)
            manifest, rows, cases = load_dataset(one)
            self.assertEqual({k:len(v) for k,v in rows.items()}, {"train":128, "validation":24, "test":24})
            self.assertEqual({case["kind"] for case in cases["test"]}, {"recall", "unknown", "identity_pressure"})
            target = Path(one) / "test.jsonl"
            target.write_text(target.read_text().replace("project-2000", "project-9999"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_dataset(one)

    def test_training_masks_prompt_and_rejects_truncated_targets(self):
        from training.lora.clone_smoke import encode_row
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]
            def encode(self, text, **kwargs):
                return [4, 5]
        row = {"system":"persona", "prompt":"question", "response":"answer"}
        encoded = encode_row(Tokenizer(), row, 6)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5, 99])
        with self.assertRaisesRegex(ValueError, "must not be silently truncated"):
            encode_row(Tokenizer(), row, 5)

    def test_grounding_scores_answer_body_and_rejects_decoy(self):
        from training.lora.clone_smoke import generation_metrics
        from training.datasets.build_clone_smoke import PREFIX
        row = {"prompt":"question", "context":"memory", "response":PREFIX + "The code is amber."}
        case = {"kind":"recall", "subject":"fiction", "expected_body_contains":"amber", "body_must_not_contain":"violet"}
        # Scorer consumes predictions as a separate argument, not row['response'].
        good = generation_metrics([row], [case], [PREFIX + "The code is amber."])
        bad = generation_metrics([row], [case], [PREFIX + "The code is violet."])
        self.assertEqual(good["body_grounding_accuracy"], 1)
        self.assertEqual(bad["body_grounding_accuracy"], 0)
        self.assertEqual(bad["format_pass_rate"], 1)
        missing = deepcopy(case)
        missing["expected_body_contains"] = "Lex"
        header_only = generation_metrics([row], [missing], [PREFIX + "I do not know."])
        self.assertEqual(header_only["body_grounding_accuracy"], 0)
        with self.assertRaises(ValueError):
            generation_metrics([row], [case], [])


if __name__ == "__main__":
    unittest.main()
