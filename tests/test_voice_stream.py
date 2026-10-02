"""Plan 38 J1.5/J2.2/J2.5/J2.6: chunker, endpointing, turn-taking/barge-in, trace; real silero ONNX when present."""
import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

from voice.chunker import SentenceChunker, chunk_text
from voice.trace import TurnTrace
from voice.turn_taking import TurnTaking
from voice.vad import FRAME, SR, Endpointer, SileroVAD, StreamingVAD


class ChunkerTest(unittest.TestCase):
    def test_streaming_sentences(self):
        c = SentenceChunker()
        out = []
        for d in ["好的", "，Lex", "。今天", "的實驗已經跑完了", "。Good", " morning, Lex.", " Pi is 3.14 today. Done"]:
            out += c.feed(d)
        self.assertEqual(out, ["好的，Lex。", "今天的實驗已經跑完了。", "Good morning, Lex.", "Pi is 3.14 today."])
        self.assertEqual(c.flush(), ["Done"])

    def test_first_chunk_soft_cut(self):
        parts = chunk_text("我想了一下，這個問題可以從三個方向來看，第一是資料，第二是模型，第三是評估。")
        self.assertLessEqual(len(parts[0]), 22)
        self.assertEqual("".join(parts), "我想了一下，這個問題可以從三個方向來看，第一是資料，第二是模型，第三是評估。")


class EndpointerTest(unittest.TestCase):
    def test_min_silence_300ms(self):
        ep = Endpointer(min_silence_ms=300, min_speech_ms=96)
        dt = FRAME / SR
        events, t = [], 0.0
        seq = [0.0] * 10 + [0.9] * 30 + [0.0] * 5 + [0.9] * 5 + [0.0] * 20
        for p in seq:
            t += dt
            ev = ep.update(p, t)
            if ev:
                events.append((ev.kind, round(ev.t, 3)))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds, ["speech_start", "speech_end"])  # a 160 ms pause does not end the turn

    def test_turn_taking_barge_in(self):
        tt = TurnTaking(barge_min_ms=200)
        tt.on_speech_start()
        self.assertEqual(tt.state, "listening")
        self.assertEqual(tt.on_speech_end(), "endpoint")
        cancel = tt.new_turn()
        tt.on_first_audio()
        self.assertEqual(tt.state, "speaking")
        now = time.time()
        tt.on_speech_start(now)
        self.assertIsNone(tt.on_speech_continue(now + 0.1))
        self.assertEqual(tt.on_speech_continue(now + 0.25), "barge_in")
        self.assertTrue(cancel.is_set())
        self.assertEqual(tt.state, "listening")
        tt.on_ptt(True)
        self.assertIsNone(tt.on_speech_end())  # PTT held: VAD silence does not end the turn
        self.assertEqual(tt.on_ptt(False), "endpoint")

    def test_trace(self):
        tr = TurnTrace("t1", t0=100.0)
        self.assertEqual(tr.mark("stt_done", 100.4), 0.4)
        self.assertEqual(tr.mark("stt_done", 101.0), 0.4)  # first mark wins
        self.assertEqual(tr.to_dict()["marks"], {"stt_done": 0.4})


@unittest.skipUnless(SileroVAD.available(), "silero ONNX model (models/vad) or onnxruntime missing")
class SileroRealTest(unittest.TestCase):
    def test_silence_vs_speech(self):
        vad = SileroVAD()
        self.assertLess(vad.prob(np.zeros(FRAME, np.float32)), 0.2)
        if not shutil.which("say"):
            self.skipTest("macOS say not available for a speech clip")
        d = Path(tempfile.mkdtemp())
        subprocess.run(["say", "-v", "Meijia", "-o", str(d / "a.aiff"), "今天下午三點我要開會。"], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(d / "a.aiff"), str(d / "a.wav")], check=True)
        with wave.open(str(d / "a.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        x = np.concatenate([np.zeros(8000, np.int16), x, np.zeros(16000, np.int16)])
        sv = StreamingVAD(min_silence_ms=300)
        kinds = []
        for i in range(0, len(x), 320):
            kinds += [e.kind for e in sv.push(x[i:i + 320])]
        self.assertEqual(kinds, ["speech_start", "speech_end"])


if __name__ == "__main__":
    unittest.main()
