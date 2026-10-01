import unittest

from genai.llm.reasoning import has_reasoning_leak, sanitize_reply, strip_cli_banner, strip_reasoning

# Captured verbatim (structure) from Mac llama-cli b8881 on 2026-10-02.
REAL_CLI_STDOUT = (
    "Loading model... \n\n\n▄▄ ▄▄\n██ ██\n\nbuild      : b8881-0dedb9ef7\nmodel      : gemma.gguf\n"
    "modalities : text\nusing custom system prompt\n\navailable commands:\n"
    "  /exit or Ctrl+C     stop or exit\n  /regen              regenerate the last response\n\n\n"
    "> Context:\nline one about ASAL\nline two\n\nUser request:\n用一句話說明你是誰。\n\n"
    "我是一個大型語言模型，由 Google DeepMind 開發。\n\n"
    "[ Prompt: 209.8 t/s | Generation: 91.3 t/s ]\n\nExiting...\n"
)


class ReasoningHygieneTests(unittest.TestCase):
    def test_strip_real_cli_banner_with_prompt(self):
        prompt = "Context:\nline one about ASAL\nline two\n\nUser request:\n用一句話說明你是誰。"
        self.assertEqual(strip_cli_banner(REAL_CLI_STDOUT, prompt=prompt), "我是一個大型語言模型，由 Google DeepMind 開發。")

    def test_strip_cli_banner_without_prompt_falls_back(self):
        out = strip_cli_banner("Loading model...\navailable commands:\n  /exit\n\n> hi\n\nhello there\n\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]\n\nExiting...")
        self.assertEqual(out, "hello there")

    def test_plain_text_untouched(self):
        self.assertEqual(strip_cli_banner("  普通回答。 "), "普通回答。")

    def test_closed_gemma_thought_block_removed(self):
        raw = "<|channel>thought\nThinking Process:\n1. **Analyze**...<channel|>你好，我在這裡陪你。"
        self.assertEqual(strip_reasoning(raw), "你好，我在這裡陪你。")
        self.assertFalse(has_reasoning_leak(strip_reasoning(raw)))

    def test_think_tags_and_control_tokens_removed(self):
        raw = "<think>plan the answer</think><|channel>final\n答案是 42。<turn|>"
        self.assertEqual(strip_reasoning(raw), "答案是 42。")

    def test_sanitize_reply_combines(self):
        raw = REAL_CLI_STDOUT.replace("我是一個大型", "<think>x</think>我是一個大型")
        prompt = "Context:\nline one about ASAL\nline two\n\nUser request:\n用一句話說明你是誰。"
        self.assertEqual(sanitize_reply(raw, prompt=prompt), "我是一個大型語言模型，由 Google DeepMind 開發。")

    def test_leak_detection(self):
        self.assertTrue(has_reasoning_leak("Thinking Process:\n1. foo"))
        self.assertTrue(has_reasoning_leak("<|channel>thought\nxx"))
        self.assertFalse(has_reasoning_leak("聽到你很難過，我很抱歉。"))


if __name__ == "__main__":
    unittest.main()
