import unittest
from types import SimpleNamespace

from genai.web.emotion_llm import classify_emotion_llm, judge_empathy_llm, parse_label


class _Fake:
    """Parser-only stub (unit test of parsing; NOT evidence of model quality)."""
    def __init__(self, text):
        self.text = text
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        return SimpleNamespace(text=self.text)


class EmotionLLMParsingTests(unittest.TestCase):
    def test_parse_label_first_match(self):
        self.assertEqual(parse_label("anger"), "anger")
        self.assertEqual(parse_label(" Sadness."), "sadness")
        self.assertEqual(parse_label("我覺得是擔心"), "fear")
        self.assertIsNone(parse_label("???"))

    def test_classify_uses_zero_temperature_and_label_set(self):
        fake = _Fake("joy")
        out = classify_emotion_llm(fake, "太棒了")
        self.assertEqual(out["label"], "joy")
        self.assertEqual(fake.requests[0].temperature, 0.0)
        self.assertIn("neutral", fake.requests[0].prompt)

    def test_empathy_judge_parses_three_scores(self):
        self.assertEqual(judge_empathy_llm(_Fake("2 1 2"), "u", "r")["total"], 5)
        self.assertIsNone(judge_empathy_llm(_Fake("good"), "u", "r")["total"])


if __name__ == "__main__":
    unittest.main()


class HybridDetectorTests(unittest.TestCase):
    """Routing logic only (stub adapter for parsing); model quality is measured by tools/run_plan37_emotion_eval.py."""

    def test_confident_lexicon_skips_llm(self):
        from genai.web.emotion_llm import detect_emotion
        fake = _Fake("joy")
        out = detect_emotion(fake, "氣死我了！超級生氣！真的很生氣！", mode="hybrid", threshold=0.6)
        self.assertEqual(out["label"], "anger")
        self.assertEqual(fake.requests, [])

    def test_neutral_lexicon_defers_to_llm(self):
        from genai.web.emotion_llm import detect_emotion
        fake = _Fake("sadness")
        out = detect_emotion(fake, "我養了十二年的狗昨天走了。", mode="hybrid")
        self.assertEqual(out["label"], "sadness")
        self.assertEqual(out["lexicon_label"], "neutral")
        self.assertTrue(out["source"].startswith("llm_gemma"))
        self.assertLess(out["valence"], 0)

    def test_unparsed_llm_falls_back_and_says_so(self):
        from genai.web.emotion_llm import detect_emotion
        out = detect_emotion(_Fake("???"), "今天天氣如何", mode="llm")
        self.assertEqual(out["label"], "neutral")
        self.assertIn("unparsed", out["source"])

    def test_lexicon_mode_never_calls_model(self):
        from genai.web.emotion_llm import detect_emotion
        fake = _Fake("joy")
        detect_emotion(fake, "我養了十二年的狗昨天走了。", mode="lexicon")
        self.assertEqual(fake.requests, [])


class RubricPackedParseTests(unittest.TestCase):
    def test_packed_digits(self):
        from genai.web.emotion_llm import judge_empathy_llm
        r = judge_empathy_llm(_Fake("222"), "我好怕", "別怕，我在。")
        self.assertEqual(r["total"], 6)

    def test_spaced_digits(self):
        from genai.web.emotion_llm import judge_empathy_llm
        r = judge_empathy_llm(_Fake("2 1 0"), "我好怕", "嗯。")
        self.assertEqual(r["scores"], {"A": 2, "B": 1, "C": 0})
