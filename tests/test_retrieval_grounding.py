import json
import tempfile
import unittest
from pathlib import Path

from digital_clone.engine import DigitalCloneEngine


class CloneRetrievalGroundingTests(unittest.TestCase):
    def test_retrieved_memory_enters_prompt_and_output(self):
        cases = json.loads(Path("tests/fixtures/clone_cases.json").read_text(encoding="utf-8"))
        case = cases[0]
        cfg = {
            "persona": case["persona"],
            "llm": {"backend": "dummy", "model_family": "dummy"},
            "memory": {"retrieval": {"limit": 5}},
            "inputs": [case["input"]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = DigitalCloneEngine(cfg, tmpdir).run()
        row = result["outputs"][0]
        self.assertIn(case["expected_keyword"], row["context"])
        self.assertIn(case["expected_keyword"], row["output"])
        self.assertTrue(row["retrieved_memories"])
        self.assertIn("retrieval_grounding", row["consistency"]["checks"])
        self.assertTrue(row["consistency"]["checks"]["retrieval_grounding"])


if __name__ == "__main__":
    unittest.main()
