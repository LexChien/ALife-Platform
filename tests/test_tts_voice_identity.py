"""Plan 38 J2.7: resident Meijia synthesizer must be sample-identical to `say -v Meijia` (+ afconvert) in this context."""
import shutil
import sys
import unittest

from voice.tts_selfcheck import selfcheck


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("say") and shutil.which("swiftc"), "macOS say + swiftc required")
class TTSVoiceIdentityTest(unittest.TestCase):
    def test_resident_matches_say(self):
        res = selfcheck("Meijia", context="unittest", use_resemblyzer=False)
        self.assertEqual(res["resident_vs_say"], "identical", res)
        self.assertEqual(res["provider"], "macos_resident")
        self.assertTrue(res["ok"], res)


if __name__ == "__main__":
    unittest.main()
