import unittest

from genai.llm.adapter import LLMRequest
from tools.chat_gemma import (
    _build_direct_answer_retry_request,
    _build_turn_context,
    _contains_reasoning,
    _strict_local_cleanup,
    _strict_cleanup_with_retry,
)


class ChatGemmaTests(unittest.TestCase):
    def test_contains_reasoning_detects_thinking_process(self):
        self.assertTrue(_contains_reasoning("Thinking Process:\n1. Analyze the Request"))
        self.assertFalse(_contains_reasoning("相對論指出時間與空間會隨速度改變。"))

    def test_strict_local_cleanup_strips_reasoning_outline(self):
        raw = (
            "Thinking Process:\n\n"
            "1.  **Analyze the Request:** ...\n"
            "2.  **Determine the Goal:** ...\n\n"
            "相對論指出時間與空間不是固定不變，速度與重力都會影響它們的表現。"
        )
        cleaned = _strict_local_cleanup(raw)
        self.assertIn("相對論指出時間與空間", cleaned)
        self.assertNotIn("Thinking Process:", cleaned)
        self.assertNotIn("Analyze the Request", cleaned)

    def test_build_turn_context_includes_recent_history(self):
        context = _build_turn_context(
            "Base context",
            [
                ("user", "你好"),
                ("assistant", "你好，我是 Gemma。"),
                ("user", "請介紹相對論"),
            ],
            history_turns=2,
        )
        self.assertIn("Base context", context)
        self.assertNotIn("User: 你好", context)
        self.assertIn("Assistant: 你好，我是 Gemma。", context)
        self.assertIn("User: 請介紹相對論", context)

    def test_build_direct_answer_retry_request_preserves_user_intent(self):
        request = LLMRequest(
            prompt="請用 100 字介紹相對論。",
            context="用繁體中文直接回答",
            system="ignored",
            max_tokens=120,
            temperature=0.2,
        )
        retry = _build_direct_answer_retry_request(request)
        self.assertEqual(retry.prompt, request.prompt)
        self.assertEqual(retry.context, request.context)
        self.assertEqual(retry.max_tokens, 96)
        self.assertEqual(retry.temperature, 0.0)
        self.assertTrue(retry.metadata["disable_reasoning_extractor"])

    def test_strict_cleanup_with_retry_uses_retry_result(self):
        class StubAdapter:
            def generate(self, request):
                class Response:
                    text = "相對論指出時間與空間會隨速度與重力改變。"
                return Response()

        request = LLMRequest(prompt="請介紹相對論。", context="用繁體中文")
        cleaned, meta = _strict_cleanup_with_retry(
            StubAdapter(),
            request,
            "Thinking Process:\n1. Analyze the Request",
        )
        self.assertEqual(cleaned, "相對論指出時間與空間會隨速度與重力改變。")
        self.assertTrue(meta["strict_cleanup_retry"])
        self.assertEqual(meta["strict_cleanup_retry_strategy"], "direct_answer_retry")


if __name__ == "__main__":
    unittest.main()
