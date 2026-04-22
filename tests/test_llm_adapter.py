import unittest

from genai.llm.adapter import LLMRequest
from genai.llm.backends.dummy import DummyLLMAdapter
from genai.llm.backends.llama_cpp import LlamaCppAdapter
from genai.llm.factory import create_llm_adapter


class LLMAdapterTests(unittest.TestCase):
    def test_dummy_backend_returns_structured_response(self):
        adapter = DummyLLMAdapter()
        response = adapter.generate(
            LLMRequest(prompt="hello", context="ctx", system="sys")
        )
        self.assertEqual(response.backend, "dummy")
        self.assertEqual(response.model_family, "dummy")
        self.assertIn("hello", response.text)
        self.assertIn("ctx", response.text)
        self.assertIn("sys", response.text)
        self.assertEqual(response.runtime["mode"], "dummy")

    def test_factory_builds_dummy_backend(self):
        adapter = create_llm_adapter({"llm": {"backend": "dummy"}})
        self.assertEqual(adapter.backend_name, "dummy")

    def test_llama_cpp_healthcheck_reports_missing_runtime(self):
        adapter = LlamaCppAdapter(
            model_family="gemma",
            model_path="models/gemma/missing.gguf",
        )
        health = adapter.healthcheck()
        self.assertEqual(health["backend"], "llama_cpp")
        self.assertFalse(health["exists"])
        self.assertIn("binding_available", health)


if __name__ == "__main__":
    unittest.main()
