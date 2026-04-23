import unittest

from genai.llm.backends.dummy import DummyLLMAdapter
from genai.llm.backends.llama_cpp import LlamaCppAdapter
from genai.llm.factory import create_llm_adapter


class LLMFactoryModelIdTests(unittest.TestCase):
    def test_factory_supports_model_id_for_bootstrap_spec(self):
        adapter = create_llm_adapter(
            {
                "llm": {
                    "model_id": "gemma_bootstrap_v1",
                    "cli_path": "third_party/llama.cpp/build/bin/llama-completion",
                }
            }
        )
        self.assertIsInstance(adapter, LlamaCppAdapter)
        self.assertEqual(adapter.model_family, "gemma")
        self.assertEqual(adapter.model_path, "models/gemma/gemma.gguf")

    def test_factory_keeps_backward_compatibility_for_dummy(self):
        adapter = create_llm_adapter({"llm": {"backend": "dummy", "model_family": "dummy"}})
        self.assertIsInstance(adapter, DummyLLMAdapter)
        self.assertEqual(adapter.model_family, "dummy")

    def test_factory_prefers_model_id_over_inline_backend_identity(self):
        adapter = create_llm_adapter(
            {
                "llm": {
                    "model_id": "gemma_bootstrap_v1",
                    "backend": "dummy",
                    "model_family": "wrong-family",
                    "model_path": "wrong/path.gguf",
                    "cli_path": "third_party/llama.cpp/build/bin/llama-completion",
                }
            }
        )
        self.assertIsInstance(adapter, LlamaCppAdapter)
        self.assertEqual(adapter.model_family, "gemma")
        self.assertEqual(adapter.model_path, "models/gemma/gemma.gguf")


if __name__ == "__main__":
    unittest.main()
