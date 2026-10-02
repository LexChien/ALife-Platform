"""Plan 38 J2.2: Silero VAD (ONNX v5, onnxruntime only - no torch) + endpointing.

``SileroVAD.prob(frame512)`` -> speech probability for 32 ms @16 kHz. ``Endpointer`` turns probabilities into
speech_start / speech_end events with min_silence (default 300 ms, spec J2.2) and min_speech guards.
Model file: models/vad/silero_vad.onnx (private models/ dir; copied from the silero-vad package)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "vad" / "silero_vad.onnx"
SR = 16000
FRAME = 512  # samples per VAD step at 16 kHz (32 ms)
CONTEXT = 64


class SileroVAD:
    def __init__(self, model_path: str | Path = MODEL):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.reset()

    @staticmethod
    def available(model_path: str | Path = MODEL) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return Path(model_path).exists()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FRAME:
            raise ValueError(f"silero expects {FRAME} samples at 16 kHz")
        inp = np.concatenate([self.context, x], axis=1)
        out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": np.array(SR, dtype=np.int64)})
        self.context = inp[:, -CONTEXT:]
        return float(out[0][0])


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end"
    t: float   # stream time in seconds
    start_t: float | None = None


class Endpointer:
    def __init__(self, threshold: float = 0.5, neg_threshold: float | None = None, min_silence_ms: int = 300,
                 min_speech_ms: int = 160, pre_roll_ms: int = 200):
        self.thr = threshold
        self.neg = neg_threshold if neg_threshold is not None else max(0.01, threshold - 0.15)
        self.min_silence = min_silence_ms / 1000
        self.min_speech = min_speech_ms / 1000
        self.pre_roll = pre_roll_ms / 1000
        self.reset()

    def reset(self):
        self.in_speech = False
        self.cand_start = None
        self.silence_start = None
        self.speech_start = None

    def update(self, p: float, t: float) -> VadEvent | None:
        """p = speech prob of the frame ending at time t (seconds)."""
        frame_s = FRAME / SR
        if not self.in_speech:
            if p >= self.thr:
                self.cand_start = self.cand_start if self.cand_start is not None else t - frame_s
                if t - self.cand_start >= self.min_speech:
                    self.in_speech = True
                    self.speech_start = max(0.0, self.cand_start - self.pre_roll)
                    self.silence_start = None
                    return VadEvent("speech_start", t, self.speech_start)
            else:
                self.cand_start = None
            return None
        if p < self.neg:
            self.silence_start = self.silence_start if self.silence_start is not None else t - frame_s
            if t - self.silence_start >= self.min_silence:
                ev = VadEvent("speech_end", self.silence_start, self.speech_start)
                self.reset()
                return ev
        else:
            self.silence_start = None
        return None


class StreamingVAD:
    """Accepts arbitrary-size int16/float PCM chunks at 16 kHz; returns endpoint events and keeps the utterance audio."""

    def __init__(self, vad: SileroVAD | None = None, **endpoint_kw):
        self.vad = vad or SileroVAD()
        self.ep = Endpointer(**endpoint_kw)
        self.buf = np.zeros(0, dtype=np.float32)
        self.audio = []  # rolling recent audio for pre-roll + utterance capture
        self.n = 0
        self.base = 0  # absolute sample index of the first kept sample
        self.last_prob = 0.0

    def reset(self):
        self.vad.reset()
        self.ep.reset()
        self.buf = np.zeros(0, dtype=np.float32)
        self.audio = []
        self.n = 0
        self.base = 0

    def push(self, pcm) -> list[VadEvent]:
        x = np.asarray(pcm)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        self.audio.append(x.astype(np.float32))
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        events = []
        while len(self.buf) >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self.n += FRAME
            self.last_prob = self.vad.prob(frame)
            ev = self.ep.update(self.last_prob, self.n / SR)
            if ev:
                events.append(ev)
        return events

    def segment(self, start_t: float, end_t: float) -> np.ndarray:
        allx = np.concatenate(self.audio) if self.audio else np.zeros(0, dtype=np.float32)
        a = max(0, int(start_t * SR) - self.base)
        b = max(0, int(end_t * SR) - self.base)
        return allx[a:b]

    def trim(self, keep_s: float = 30.0):
        """Bound memory: keep only the last keep_s seconds; stream times stay absolute."""
        allx = np.concatenate(self.audio) if self.audio else np.zeros(0, dtype=np.float32)
        if len(allx) > keep_s * SR:
            drop = len(allx) - int(keep_s * SR)
            self.audio = [allx[drop:]]
            self.base += drop
