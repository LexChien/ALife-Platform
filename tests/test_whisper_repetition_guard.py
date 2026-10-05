"""Unit tests for Whisper repetition post-filter (no MLX/MPS required)."""
from __future__ import annotations

import time
import unittest

from genai.web.voice import collapse_repeated_ngrams


class CollapseRepeatedNgramsTests(unittest.TestCase):
    def test_collapses_looped_phrase(self):
        unit = "我聽不懂你在說什麼。"
        text = unit * 20
        out = collapse_repeated_ngrams(text, ngram=len(unit), min_repeats=3)
        self.assertEqual(out, unit)
        self.assertLess(len(out), len(text) // 5)

    def test_preserves_normal_text(self):
        text = "今天天氣不錯，我們去河濱公園走走吧。"
        self.assertEqual(collapse_repeated_ngrams(text), text)

    def test_caps_absurd_length(self):
        text = "a" * 5000
        out = collapse_repeated_ngrams(text, max_chars=100, ngram=8, min_repeats=3)
        self.assertLessEqual(len(out), 101)  # 100 + ellipsis
        self.assertTrue(out.endswith("…"))

    def test_empty(self):
        self.assertEqual(collapse_repeated_ngrams(""), "")
        self.assertEqual(collapse_repeated_ngrams(None or ""), "")

    def test_period_sweep_zh_x3_and_x40(self):
        base = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黃宇宙洪荒日月盈昃春秋冬夏"
        for period in range(4, 65):
            unit = base[:period]
            for reps in (3, 40):
                with self.subTest(period=period, reps=reps, lang="zh"):
                    text = unit * reps
                    out = collapse_repeated_ngrams(text)
                    self.assertEqual(out, unit, f"p={period} x{reps}: {out!r}")

    def test_period_sweep_en_x3_and_x40(self):
        base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
        for period in range(4, 65):
            unit = base[:period]
            for reps in (3, 40):
                with self.subTest(period=period, reps=reps, lang="en"):
                    text = unit * reps
                    out = collapse_repeated_ngrams(text)
                    self.assertEqual(out, unit, f"p={period} x{reps}: {out!r}")

    def test_embedded_loop_preserves_prefix_suffix(self):
        head = "好的，我們今天來討論一下這個計畫的進度。"
        loop = "謝謝大家收看，請記得訂閱我的頻道。"  # 17 chars
        tail = "下次見。"
        text = head + loop * 40 + tail
        out = collapse_repeated_ngrams(text)
        self.assertEqual(out, head + loop + tail)
        self.assertTrue(out.startswith(head))
        self.assertTrue(out.endswith(tail))
        self.assertEqual(out.count(loop), 1)

    def test_en_embedded_loop(self):
        head = "Okay so the plan is ready. "
        loop = "Thank you for watching. "  # 24 chars
        tail = "Bye."
        text = head + loop * 30 + tail
        out = collapse_repeated_ngrams(text)
        self.assertEqual(out, head + loop + tail)
        self.assertEqual(out.count("Thank you for watching."), 1)

    def test_separator_variant_loops(self):
        # Same core phrase, trailing punctuation/space differs between copies.
        text = "請訂閱我的頻道。請訂閱我的頻道、請訂閱我的頻道。"
        out = collapse_repeated_ngrams(text)
        self.assertEqual(out.count("請訂閱我的頻道"), 1)
        self.assertIn("請訂閱我的頻道", out)
        # Whitespace variant
        text2 = "Thank you. Thank you. Thank you."
        out2 = collapse_repeated_ngrams(text2)
        self.assertEqual(out2.count("Thank you"), 1)

    def test_normal_sentences_unchanged(self):
        samples = [
            "今天天氣不錯，我們去河濱公園走走吧。",
            "早上七點，城市慢慢醒來，街角的早餐店已經排起長長的隊伍。",
            "在辦公室裡，工程師們一邊喝咖啡，一邊討論新版本的測試計畫。",
            "下午的會議討論了預算、時程與人力安排，大家同意下週再檢討一次。",
            "The quick brown fox jumps over the lazy dog near the river bank.",
            "Please review the pull request before merging into main tomorrow.",
            "We deployed the service at noon and monitored latency for an hour.",
            "哈哈哈",
            "對對對",
            "very very",
            "是的是的，我知道了。",  # only 2 copies of 是的 — below min_repeats
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(collapse_repeated_ngrams(text), text)

    def test_cap_then_collapse(self):
        text = "字" * 50000
        out = collapse_repeated_ngrams(text, max_chars=4000)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 4001)
        # After cap, period-4 run of 字 collapses to one short unit + ellipsis.
        self.assertLess(len(out), 100)

    def test_timing_4000_chars(self):
        # Worst-ish: no long collapse early; varied content padded to 4000.
        chunk = "今天天氣不錯我們去河濱公園走走順便買杯咖啡聊聊專案進度然後回家休息一下。"
        text = (chunk * ((4000 // len(chunk)) + 1))[:4000]
        self.assertEqual(len(text), 4000)
        t0 = time.perf_counter()
        out = collapse_repeated_ngrams(text)
        elapsed = time.perf_counter() - t0
        # O(n * max_period) on 4000 chars should be well under 250 ms on Mac.
        self.assertLess(elapsed, 0.25, f"took {elapsed*1000:.1f} ms")
        self.assertIsInstance(out, str)

    def test_ngram_kwarg_backward_compatible(self):
        unit = "abcdefghij"  # 10
        text = unit * 5
        # Old callers pass ngram=len(unit); must still collapse via period scan.
        out = collapse_repeated_ngrams(text, ngram=len(unit), min_repeats=3)
        self.assertEqual(out, unit)


if __name__ == "__main__":
    unittest.main()
