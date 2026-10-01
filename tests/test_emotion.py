import math
import unittest

import numpy as np

from genai.web.emotion import EmotionState, arousal_from_prosody, detect_text_emotion, modulation, prosody_features

# Small self-authored labelled set (Plan 37 E1). Not an external benchmark.
LABELLED = [
    ("我今天很難過，工作被罵了。", "sadness"), ("真的好想哭，覺得好孤單", "sadness"), ("I feel so lonely and sad today", "sadness"),
    ("氣死我了！他們又改需求！", "anger"), ("我受夠了，這太不爽了", "anger"), ("I'm so angry, this is ridiculous", "anger"),
    ("我好擔心明天的面試，睡不著", "fear"), ("壓力好大，很焦慮", "fear"), ("I'm really anxious about the exam", "fear"),
    ("太好了！實驗成功了！", "joy"), ("謝謝你，我好開心", "joy"), ("This is awesome, I'm so happy", "joy"),
    ("天啊，竟然是這樣，沒想到", "surprise"), ("Wow, no way!", "surprise"),
    ("今天天氣如何？", "neutral"), ("請幫我列出三個步驟", "neutral"), ("What time is it?", "neutral"),
]


class TextEmotionTests(unittest.TestCase):
    def test_labelled_accuracy(self):
        hits = sum(detect_text_emotion(t)["label"] == y for t, y in LABELLED)
        self.assertGreaterEqual(hits / len(LABELLED), 0.9, hits)

    def test_negation(self):
        self.assertNotEqual(detect_text_emotion("我不難過")["label"], "sadness")

    def test_valence_sign(self):
        self.assertLess(detect_text_emotion("我很難過")["valence"], 0)
        self.assertGreater(detect_text_emotion("我很開心")["valence"], 0)


class ProsodyTests(unittest.TestCase):
    def _tone(self, hz, amp, vibrato=0.0, sr=16000, secs=1.0):
        t = np.arange(int(sr * secs)) / sr
        freq = hz + vibrato * np.sin(2 * math.pi * 3 * t)
        phase = 2 * math.pi * np.cumsum(freq) / sr
        return (amp * np.sin(phase)).astype(np.float32)

    def test_pitch_estimate(self):
        f = prosody_features(self._tone(200, 0.3))
        self.assertAlmostEqual(f["pitch_hz_mean"], 200, delta=12)
        self.assertGreater(f["voiced_ratio"], 0.8)

    def test_loud_lively_has_higher_arousal(self):
        calm = arousal_from_prosody(prosody_features(self._tone(150, 0.03)))
        lively = arousal_from_prosody(prosody_features(self._tone(220, 0.5, vibrato=80)))
        self.assertGreater(lively, calm)

    def test_silence(self):
        f = prosody_features(np.zeros(16000, dtype=np.float32))
        self.assertEqual(f["voiced_ratio"], 0.0)
        self.assertEqual(arousal_from_prosody(f), 0.0)


class StateTests(unittest.TestCase):
    def test_state_tracks_and_decays(self):
        s = EmotionState()
        s.update(detect_text_emotion("我今天很難過，工作被罵了。"))
        self.assertEqual(s.label, "sadness")
        peak = s.intensity
        for _ in range(4):
            s.update(detect_text_emotion("今天天氣如何？"))
        self.assertLess(s.intensity, peak)
        self.assertEqual(s.label, "neutral")

    def test_anger_not_confused_with_fear(self):
        s = EmotionState().update(detect_text_emotion("氣死我了！客戶又臨時改需求，我整個週末都白做了！"))
        self.assertEqual(s.label, "anger")
        f = EmotionState().update(detect_text_emotion("我好擔心明天的面試，緊張到睡不著。"))
        self.assertEqual(f.label, "fear")

    def test_residual_mood_guidance_answers_directly(self):
        s = EmotionState().update(detect_text_emotion("我今天真的好難過，好想哭。"))
        s.update(detect_text_emotion("今天天氣如何？"))
        g = modulation(s)["system_guidance"]
        self.assertIn("直接", g)
        self.assertNotIn("先用一兩句真誠地接住感受", g)

    def test_modulation_changes_voice_and_guidance(self):
        sad = EmotionState().update(detect_text_emotion("我好難過好想哭"))
        happy = EmotionState().update(detect_text_emotion("太好了我好開心"))
        ms, mh = modulation(sad), modulation(happy)
        self.assertLess(ms["tts"]["rate"], mh["tts"]["rate"])
        self.assertIn("感受", ms["system_guidance"])
        self.assertEqual(ms["avatar"]["expression"], "concerned")
        self.assertEqual(modulation(EmotionState())["system_guidance"], "")


if __name__ == "__main__":
    unittest.main()
