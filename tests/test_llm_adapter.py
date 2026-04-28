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

    def test_llama_cpp_clean_subprocess_output_removes_cli_banner_and_prompt(self):
        adapter = LlamaCppAdapter(
            model_family="gemma",
            model_path="models/gemma/missing.gguf",
        )
        prompt = "上下文:\nprototype baseline\n\n使用者:\n請用一句繁體中文自我介紹。"
        raw = (
            "Loading model...\n\n"
            "available commands:\n"
            "  /exit or Ctrl+C     stop or exit\n\n"
            f"> {prompt}\n\n"
            "我是 Gemma 4，一個由 Google DeepMind 開發的開放權重大型語言模型！\n\n"
            "[ Prompt: 152.6 t/s | Generation: 30.2 t/s ]\n\n"
            "Exiting..."
        )
        cleaned = adapter._clean_subprocess_output(raw, prompt=prompt)
        self.assertEqual(
            cleaned,
            "我是 Gemma 4，一個由 Google DeepMind 開發的開放權重大型語言模型！",
        )

    def test_llama_cpp_decodes_invalid_subprocess_utf8_lossily(self):
        raw = b"\xe7invalid\nfinal answer"
        decoded = LlamaCppAdapter._decode_subprocess_bytes(raw)
        self.assertIn("\ufffd", decoded)
        self.assertIn("final answer", decoded)


if __name__ == "__main__":
    unittest.main()
