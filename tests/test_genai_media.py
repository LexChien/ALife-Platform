import ctypes.util
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
from genai.voice.adapter import EspeakVoiceAdapter
from genai.multimodal.engine import GenAIEngine


class MediaTests(unittest.TestCase):
    def test_dummy_audio_is_not_counted_as_real_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            result = GenAIEngine({"prompt": "A blue cell", "llm": {"backend": "dummy"}}, directory).run()
            self.assertFalse(result["summary"]["has_audio"])
            self.assertFalse(result["summary"]["real_image"])

    @unittest.skipUnless(ctypes.util.find_library("espeak-ng"), "requires eSpeak NG")
    def test_real_english_and_chinese_speech_has_nonzero_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(("The cell divides into two.", "細胞分裂成兩個。")):
                path = Path(directory) / f"voice-{index}.wav"
                result = EspeakVoiceAdapter().synthesize(text, path)
                self.assertEqual(result["status"], "generated")
                with wave.open(str(path)) as wav:
                    self.assertGreater(wav.getnframes() / wav.getframerate(), 0.5)
                    signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                self.assertGreater(float(np.std(signal)), 20)
                self.assertGreater(float(np.count_nonzero(signal)) / len(signal), 0.2)


if __name__ == "__main__":
    unittest.main()
