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

    def test_llama_cpp_cleans_thought_channel_output(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        raw = "<|channel>thought\nHere's a thinking process that leads to the suggested concept:\n\n1. idea"
        cleaned = adapter._clean_subprocess_output(raw)
        self.assertNotIn("<|channel>thought", cleaned)
        self.assertNotIn("Here's a thinking process", cleaned)
        self.assertEqual(cleaned, "1. idea")

    def test_llama_cpp_prefers_final_channel_when_present(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        raw = "<|channel>thought\ninternal\n<|channel>final\nFinal answer"
        cleaned = adapter._clean_subprocess_output(raw)
        self.assertEqual(cleaned, "Final answer")

    def test_llama_cpp_detects_reasoning_leak(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        self.assertTrue(adapter._has_reasoning_leak("Thinking Process:\n1. Analyze"))
        self.assertFalse(adapter._has_reasoning_leak("Final concept: luminous bio-cell"))

    def test_llama_cpp_builds_extractor_request(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        request = LLMRequest(prompt="Generate concept", context="prototype", system="You are helpful")
        followup = adapter._extractor_request(request, "Thinking Process:\n1. Analyze")
        self.assertIn("Rewrite the following draft as the final answer only", followup.prompt)
        self.assertEqual(followup.context, "prototype")
        self.assertTrue(followup.metadata["disable_reasoning_extractor"])

    def test_llama_cpp_heuristic_extracts_final_answer_shape(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        raw = (
            "Thinking Process:\n\n"
            "1. **Analyze the Request:** ...\n"
            "2. **Deconstruct Key Terms:** ...\n\n"
            "A futuristic AI cell organism could be a luminous bio-digital colony."
        )
        cleaned = adapter._heuristic_extract_final_answer(raw)
        self.assertIn("A futuristic AI cell organism", cleaned)
        self.assertNotIn("Thinking Process:", cleaned)

    def test_llama_cpp_skips_overcompressed_fallback_candidate(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        original = (
            "Thinking Process:\n"
            "1. Analyze the request.\n"
            "2. Draft a user-facing answer.\n\n"
            "Gemma is a family of open models developed by Google DeepMind. "
            "It can answer questions and generate text in a conversational form."
        )
        candidate = "* Name: Gemma\n* Developer: Google DeepMind"
        should_apply, reason = adapter._should_apply_fallback(original, candidate)
        self.assertFalse(should_apply)
        self.assertEqual(reason, "candidate_fragmented")

    def test_llama_cpp_accepts_non_fragmented_fallback_candidate(self):
        adapter = LlamaCppAdapter(model_family="gemma", model_path="missing.gguf")
        original = (
            "Thinking Process:\n"
            "1. Analyze the request.\n"
            "2. Draft a user-facing answer.\n\n"
            "Gemma is a family of open models developed by Google DeepMind. "
            "It can answer questions and generate text in a conversational form."
        )
        candidate = (
            "Gemma 是 Google DeepMind 開發的開放權重模型家族。"
            "我可以協助回答問題並生成文字內容，並以對話方式提供簡潔的說明與整理。"
            "在一般問答場景中，我會直接給出使用者可讀的答案，而不是列出內部分析步驟。"
        )
        should_apply, reason = adapter._should_apply_fallback(original, candidate)
        self.assertTrue(should_apply)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
