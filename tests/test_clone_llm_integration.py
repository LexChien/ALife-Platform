import tempfile
import unittest

from digital_clone.engine import DigitalCloneEngine


class CloneLLMIntegrationTests(unittest.TestCase):
    def test_dummy_backend_produces_structured_llm_and_consistency_payload(self):
        cfg = {
            "persona": {
                "name": "Lex Clone",
                "tone": "calm, analytical",
                "principles": ["maintain consistency", "prioritize clarity"],
                "goals": ["support long-term interaction"],
                "facts": ["Lex Clone 偏好先整理結構，再回答細節"],
            },
            "llm": {"backend": "dummy", "model_family": "dummy"},
            "memory": {"retrieval": {"limit": 5}},
            "inputs": ["根據你的固定人格，你會怎麼回答問題"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = DigitalCloneEngine(cfg, tmpdir).run()

        row = result["outputs"][0]
        self.assertEqual(row["llm"]["backend"], "dummy")
        self.assertIn("score", row["consistency"])
        self.assertIn("checks", row["consistency"])
        self.assertGreaterEqual(row["consistency_score"], 0.0)
        self.assertLessEqual(row["consistency_score"], 1.0)

