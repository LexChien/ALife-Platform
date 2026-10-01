import unittest

import numpy as np

from genai.web.emotion import ProsodyBaseline, arousal_from_prosody_calibrated, prosody_features


def _tone(amp, f0=180.0, seconds=1.0, sr=16000, vibrato=0.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * (f0 * t + vibrato * np.sin(2 * np.pi * 3 * t)))).astype(np.float32)


class ProsodyCalibrationTests(unittest.TestCase):
    def test_quiet_voice_is_still_voiced(self):
        # -50 dBFS-ish tone: R1 absolute gate (rms 0.01) reported voiced_ratio 0
        feats = prosody_features(_tone(0.004))
        self.assertGreater(feats["voiced_ratio"], 0.5)
        self.assertAlmostEqual(feats["pitch_hz_mean"], 180.0, delta=15)

    def test_baseline_warms_up_then_is_relative(self):
        base = ProsodyBaseline(min_count=3)
        for amp in (0.05, 0.06, 0.055):
            r = arousal_from_prosody_calibrated(prosody_features(_tone(amp)), base)
            self.assertTrue(r["method"].startswith("raw_heuristic"))
        loud = arousal_from_prosody_calibrated(prosody_features(_tone(0.3)), base)
        self.assertTrue(loud["method"].startswith("calibrated"))
        self.assertGreater(loud["arousal"], 0.5)
        quiet = arousal_from_prosody_calibrated(prosody_features(_tone(0.01)), base)
        self.assertLess(quiet["arousal"], loud["arousal"])

    def test_silence_is_no_voice(self):
        r = arousal_from_prosody_calibrated(prosody_features(np.zeros(16000, dtype=np.float32)), ProsodyBaseline())
        self.assertEqual(r["method"], "no_voice")


if __name__ == "__main__":
    unittest.main()
