import unittest

from core.config import load_config
from genai.llm.factory import create_llm_adapter


class DummyVsLlamaCppSmokeTests(unittest.TestCase):
    def test_backends_are_observably_different(self):
        dummy_cfg = load_config("configs/genai/genai_baseline.yaml")
        llama_cfg = load_config("configs/genai/gemma_llama_cpp.yaml")
        dummy = create_llm_adapter(dummy_cfg)
        llama = create_llm_adapter(llama_cfg)
        dummy_health = dummy.healthcheck()
        llama_health = llama.healthcheck()
        self.assertEqual(dummy.backend_name, "dummy")
        self.assertEqual(llama.backend_name, "llama_cpp")
        self.assertNotEqual(dummy.backend_name, llama.backend_name)
        self.assertNotEqual(dummy_health["backend"], llama_health["backend"])
        self.assertNotEqual(dummy_health["model_family"], llama_health["model_family"])
        self.assertEqual(dummy_health["ok"], True)
        self.assertIn("driver", llama_health)


if __name__ == "__main__":
    unittest.main()
