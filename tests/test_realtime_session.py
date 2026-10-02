"""Plan 38 J1.3/J2.5: realtime VoiceSession state machine (fakes for VAD/STT/LLM; MOCK by design)."""
import time
import unittest
from types import SimpleNamespace

import numpy as np

from genai.web.realtime import VoiceSession
from voice.vad import VadEvent


class FakeVAD:
    def __init__(self):
        self.n = 0
        self.last_prob = 0.0
        self.ep = SimpleNamespace(in_speech=False)
        self.script = []

    def push(self, pcm):
        self.n += len(pcm)
        evs, self.script = self.script, []
        for e in evs:
            self.ep.in_speech = e.kind == "speech_start"
        self.last_prob = 0.95 if self.ep.in_speech else 0.02
        return evs

    def segment(self, a, b):
        return np.zeros(int(max(b - a, 0) * 16000), dtype=np.float32)

    def trim(self, keep_s=30.0):
        pass


class FakeService:
    def __init__(self, delay=0.0):
        self.voice_cfg = {"ack": True}
        self.cog_cfg = {"thought_stream_default": False}
        self.calls = []
        self.delay = delay

    def chat_stream(self, sid, text, *, emit, cancel, **kw):
        self.calls.append(text)
        emit({"type": "thought", "thought": {"private_note": "secret-ish"}})
        emit({"type": "sentence", "index": 0, "text": "好的。"})
        emit({"type": "audio", "index": 0, "url": "/api/tts/x.wav"})
        t0 = time.time()
        while self.delay and time.time() - t0 < self.delay and not cancel.is_set():
            time.sleep(0.01)
        return {"session_id": "s1", "reply": "好的。", "timings": {}, "cancelled": cancel.is_set()}


class FakeSTT:
    def transcribe(self, audio):
        return {"text": "現在幾點", "language": "zh", "elapsed_s": 0.1, "audio_s": len(audio) / 16000}


class FakeAck:
    files = {"zh": ["ack_zh_0.wav"]}

    def pick(self, lang):
        return "ack_zh_0.wav"


FRAME = np.zeros(320, dtype=np.int16).tobytes()


def make(mode="open", delay=0.0, wake=None):
    out = []
    svc = FakeService(delay)
    s = VoiceSession(svc, out.append, vad=FakeVAD(), wake=wake, stt=FakeSTT(), ack=FakeAck(), barge_min_ms=100)
    s.on_control({"type": "hello", "mode": mode})
    return s, svc, out


def wait_turn(s, timeout=3.0):
    t0 = time.time()
    while s.turn_thread is not None and time.time() - t0 < timeout:
        time.sleep(0.01)


def types(out):
    return [e["type"] for e in out]


class TestVoiceSession(unittest.TestCase):
    def test_open_mic_turn_and_thoughts_hidden_by_default(self):
        s, svc, out = make("open")
        s.vad.script = [VadEvent("speech_start", 0.3, 0.1)]
        s.on_audio(FRAME)
        s.vad.script = [VadEvent("speech_end", 1.5, 0.1)]
        s.on_audio(FRAME)
        wait_turn(s)
        self.assertEqual(svc.calls, ["現在幾點"])
        t = types(out)
        for k in ("ack", "transcript", "sentence", "audio", "trace"):
            self.assertIn(k, t)
        self.assertNotIn("thought", t)  # thought stream OFF by default
        self.assertLess(t.index("ack"), t.index("transcript"))

    def test_thoughts_only_when_enabled(self):
        s, svc, out = make("open")
        s.on_control({"type": "set", "thoughts": True})
        s.on_control({"type": "text", "text": "hi"})
        wait_turn(s)
        self.assertIn("thought", types(out))

    def test_ptt_mode_ignores_vad_endpoint(self):
        s, svc, out = make("ptt")
        s.vad.script = [VadEvent("speech_start", 0.3, 0.1)]
        s.on_audio(FRAME)
        s.vad.script = [VadEvent("speech_end", 1.5, 0.1)]
        s.on_audio(FRAME)
        wait_turn(s)
        self.assertEqual(svc.calls, [])
        s.on_control({"type": "ptt", "down": True})
        for _ in range(60):  # 1.2 s of audio
            s.on_audio(FRAME)
        s.on_control({"type": "ptt", "down": False})
        wait_turn(s)
        self.assertEqual(svc.calls, ["現在幾點"])

    def test_wake_mode_requires_wake_word(self):
        class Wake:
            names = ["hey_jarvis"]
            hit = False

            def push(self, pcm):
                return "hey_jarvis" if self.hit else None
        w = Wake()
        s, svc, out = make("wake", wake=w)
        s.vad.script = [VadEvent("speech_start", 0.3, 0.1)]
        s.on_audio(FRAME)
        s.vad.script = [VadEvent("speech_end", 1.5, 0.1)]
        s.on_audio(FRAME)
        wait_turn(s)
        self.assertEqual(svc.calls, [])
        w.hit = True
        s.on_audio(FRAME)
        w.hit = False
        self.assertIn("wake", types(out))
        s.vad.script = [VadEvent("speech_start", 2.0, 1.8)]
        s.on_audio(FRAME)
        s.vad.script = [VadEvent("speech_end", 3.2, 1.8)]
        s.on_audio(FRAME)
        wait_turn(s)
        self.assertEqual(svc.calls, ["現在幾點"])

    def test_barge_in_stops_and_cancels(self):
        s, svc, out = make("open", delay=2.0)
        s.on_control({"type": "text", "text": "講個長故事"})
        t0 = time.time()
        while s.tt.state != "speaking" and time.time() - t0 < 2:
            time.sleep(0.01)
        self.assertEqual(s.tt.state, "speaking")
        s.vad.script = [VadEvent("speech_start", 0.5, 0.3)]
        s.on_audio(FRAME)
        time.sleep(0.12)
        s.on_audio(FRAME)
        stops = [e for e in out if e["type"] == "stop"]
        self.assertEqual(len(stops), 1)
        self.assertTrue(s.tt.cancel.is_set())
        wait_turn(s)
        self.assertLess(time.time() - t0, 1.5)  # generation was cancelled, not run to the 2 s end

    def test_too_short_segment_is_ignored(self):
        s, svc, out = make("open")
        s.vad.script = [VadEvent("speech_start", 0.3, 0.25)]
        s.on_audio(FRAME)
        s.vad.script = [VadEvent("speech_end", 0.4, 0.25)]
        s.on_audio(FRAME)
        wait_turn(s)
        self.assertEqual(svc.calls, [])


class TestWakeModels(unittest.TestCase):
    def test_wake_detector_loads_or_reports(self):
        from voice.wake import WakeDetector, OWW_DIR
        w = WakeDetector()
        if not (OWW_DIR / "hey_jarvis_v0.1.onnx").exists():
            self.skipTest("wake models not present")
        self.assertTrue(w.names or w.errors)
        self.assertIsNone(w.push(np.zeros(16000, dtype=np.int16)))  # silence never wakes


if __name__ == "__main__":
    unittest.main()
