import unittest

from core.config import load_config
from genai.llm.factory import create_llm_adapter
from genai.multimodal.engine import GenAIEngine


class GenAIRealConfigTests(unittest.TestCase):
    def test_gemma_llama_cpp_config_is_loadable(self):
        cfg = load_config("configs/genai/gemma_llama_cpp.yaml")
        self.assertEqual(cfg["llm"]["model_id"], "gemma_bootstrap_v1")
        adapter = create_llm_adapter(cfg)
        self.assertEqual(adapter.backend_name, "llama_cpp")
        self.assertEqual(adapter.model_family, "gemma")

    def test_engine_preserves_llm_healthcheck_in_dummy_mode(self):
        cfg = load_config("configs/genai/genai_baseline.yaml")
        result = GenAIEngine(cfg, "runs/test-genai-engine").run()
        self.assertEqual(result["summary"]["llm_backend"], "dummy")
        self.assertIn("ok", result["summary"]["llm_healthcheck"])


if __name__ == "__main__":
    unittest.main()
