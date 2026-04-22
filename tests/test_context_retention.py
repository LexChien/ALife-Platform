import json
import tempfile
import unittest
from pathlib import Path

from core.config import load_config
from genai.multimodal.engine import GenAIEngine


class GenAIContextRetentionTests(unittest.TestCase):
    def test_dummy_path_preserves_context_in_request_and_output(self):
        cases = json.loads(Path("tests/fixtures/genai_cases.json").read_text(encoding="utf-8"))
        case = cases[0]
        cfg = load_config("configs/genai/genai_baseline.yaml")
        cfg["prompt"] = case["prompt"]
        cfg["context"] = case["context"]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = GenAIEngine(cfg, tmpdir).run()
        self.assertTrue(result["summary"]["has_context"])
        self.assertIn(case["context"], result["output"]["text"])
        self.assertEqual(result["output"]["context"], case["context"])


if __name__ == "__main__":
    unittest.main()
