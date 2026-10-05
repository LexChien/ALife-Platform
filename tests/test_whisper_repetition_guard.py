"""Unit tests for Whisper repetition post-filter (no MLX/MPS required)."""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
