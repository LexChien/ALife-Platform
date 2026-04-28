import unittest

from genai.llm.adapter import LLMRequest
from tools.chat_gemma import (
    _build_direct_answer_rescue_request,
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

    def test_strict_local_cleanup_strips_meta_prompt_echo(self):
        raw = (
            "我收到的請求是要根據提供的上下文和原始用戶請求來重寫草稿。\n\n"
            "原始用戶請求是：請用繁體中文一句話自我介紹。\n\n"
            "我需要提供一個繁體中文的單句自我介紹。\n\n"
            "最終答案應該是：我是 Gemma 4，很高興和你對話。"
        )
        cleaned = _strict_local_cleanup(raw)
        self.assertEqual(cleaned, "我是 Gemma 4，很高興和你對話。")

    def test_contains_reasoning_detects_meta_prompt_echo(self):
        self.assertTrue(_contains_reasoning("我收到的請求是要根據提供的上下文來重寫草稿。"))

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

    def test_build_direct_answer_rescue_request_is_more_forceful(self):
        request = LLMRequest(
            prompt="請用一句話自我介紹。",
            context="用繁體中文",
            max_tokens=120,
        )
        rescue = _build_direct_answer_rescue_request(request)
        self.assertIn("User request", rescue.prompt)
        self.assertEqual(rescue.context, request.context)
        self.assertEqual(rescue.max_tokens, 80)
        self.assertEqual(rescue.temperature, 0.0)
        self.assertIn("Thinking Process:", rescue.stop)
        self.assertTrue(rescue.metadata["disable_reasoning_extractor"])

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

    def test_strict_cleanup_with_retry_uses_rescue_result_after_second_failure(self):
        class StubAdapter:
            def __init__(self):
                self.calls = 0

            def generate(self, request):
                self.calls += 1

                class Response:
                    pass

                response = Response()
                if self.calls == 1:
                    response.text = "Thinking Process:\n1. Analyze the Request"
                else:
                    response.text = "我是 Gemma 4，很高興透過這個網頁和你對話。"
                return response

        request = LLMRequest(prompt="請一句話自我介紹。", context="用繁體中文")
        cleaned, meta = _strict_cleanup_with_retry(
            StubAdapter(),
            request,
            "Thinking Process:\n1. Analyze the Request",
        )
        self.assertEqual(cleaned, "我是 Gemma 4，很高興透過這個網頁和你對話。")
        self.assertTrue(meta["strict_cleanup_rescue"])
        self.assertEqual(meta["strict_cleanup_retry_strategy"], "direct_answer_rescue")

    def test_strict_cleanup_with_retry_falls_back_to_safe_message(self):
        class StubAdapter:
            def generate(self, request):
                class Response:
                    text = "Thinking Process:\n1. Analyze the Request"
                return Response()

        request = LLMRequest(prompt="請一句話自我介紹。", context="用繁體中文")
        cleaned, meta = _strict_cleanup_with_retry(
            StubAdapter(),
            request,
            "Thinking Process:\n1. Analyze the Request",
        )
        self.assertIn("抱歉", cleaned)
        self.assertTrue(meta["strict_cleanup_final_fallback"])

    def test_strict_cleanup_with_retry_strips_chinese_meta_prompt_echo(self):
        class StubAdapter:
            def generate(self, request):
                class Response:
                    text = (
                        "我收到的上一個請求是要求我用繁體中文一句話自我介紹。\n"
                        "我必須只輸出最終答案。\n"
                        "根據上下文和指令，我將生成一個簡潔的自我介紹。<channel|>"
                        "我是 Gemma 4，很高興和你對話。 [end of text]"
                    )
                return Response()

        request = LLMRequest(prompt="請一句話自我介紹。", context="用繁體中文")
        cleaned, meta = _strict_cleanup_with_retry(
            StubAdapter(),
            request,
            "我收到的上一個請求是要求我用繁體中文一句話自我介紹。",
        )
        self.assertEqual(cleaned, "我是 Gemma 4，很高興和你對話。")
        self.assertFalse(meta["strict_cleanup_final_fallback"])

    def test_strict_cleanup_with_retry_retries_chinese_context_echo(self):
        class StubAdapter:
            def generate(self, request):
                class Response:
                    text = "目前 ASAL 最新 run 已完成，best_score 為 0.832，phase order 為 1→2→1。"
                return Response()

        request = LLMRequest(prompt="目前人工生命進度？", context="ASAL progress facts")
        cleaned, meta = _strict_cleanup_with_retry(
            StubAdapter(),
            request,
            (
                "上下文: Local ALife prototype baseline. Digital Clone / ASAL memory: "
                "人工生命原型狀態上下文。 使用者: 目前人工生命進度？ "
                "根據目前的 ASAL 進度摘要，最新 run 已完成。"
            ),
        )
        self.assertEqual(cleaned, "目前 ASAL 最新 run 已完成，best_score 為 0.832，phase order 為 1→2→1。")
        self.assertTrue(meta["strict_cleanup_retry"])

    def test_strict_local_cleanup_strips_life_context_bullets(self):
        raw = (
            "- 最新 ASAL run：20260424-195441，status=completed，config=co ... (truncated)\n\n"
            "人工生命雛形的最新進度是：最新的 ASAL 運行結果是 20260424-195441。"
        )
        cleaned = _strict_local_cleanup(raw)
        self.assertEqual(cleaned, "人工生命雛形的最新進度是：最新的 ASAL 運行結果是 20260424-195441。")


if __name__ == "__main__":
    unittest.main()
