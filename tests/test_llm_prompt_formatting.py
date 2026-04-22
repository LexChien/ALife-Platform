import unittest

from genai.llm.adapter import LLMRequest
from genai.llm.backends.dummy import DummyLLMAdapter
from genai.llm.backends.llama_cpp import LlamaCppAdapter


class LLMPromptFormattingTests(unittest.TestCase):
    def test_dummy_adapter_inherits_base_prompt_contract(self):
        adapter = DummyLLMAdapter()
        request = LLMRequest(
            prompt="user prompt",
            context="context block",
            system="system block",
            stop=["END"],
            temperature=0.5,
            json_mode=True,
        )
        prompt = adapter.build_prompt(request)
        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[USER]", prompt)
        self.assertIn("user prompt", prompt)
        self.assertEqual(request.stop, ["END"])
        self.assertTrue(request.json_mode)

    def test_llama_cpp_adapter_can_override_prompt_style(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        request = LLMRequest(prompt="hello", context="ctx", system="sys")
        prompt = adapter.build_prompt(request)
        self.assertIn("<system>", prompt)
        self.assertIn("<context>", prompt)
        self.assertIn("<user>", prompt)


if __name__ == "__main__":
    unittest.main()
