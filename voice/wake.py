"""Plan 38 J2.5: wake-word detection (optional; push-to-talk is always available).

* English "Hey Jarvis": openWakeWord ONNX (models/kws/oww), 80 ms frames.
* Chinese "你好管家": sherpa-onnx open-vocabulary KWS (zipformer wenetspeech 3.3M), 100 ms chunks.
Both are lazy and optional: a missing package/model leaves that detector disabled and reports why."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OWW_DIR = ROOT / "models" / "kws" / "oww"
KWS_DIR = ROOT / "models" / "kws" / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
SR = 16000
# ppinyin tokens precomputed with sherpa_onnx.text2token (needs sentencepiece/pypinyin, absent on py3.14):
# runs/plan38/wake/kws_zh_bench.json
KNOWN_TOKENS = {"你好管家": "n ǐ h ǎo g uǎn j iā"}


class OWWJarvis:
    name = "hey_jarvis"

    def __init__(self, threshold: float = 0.5, model_dir: Path = OWW_DIR):
        from openwakeword.model import Model
        self.threshold = threshold
        self.model = Model(wakeword_models=[str(model_dir / "hey_jarvis_v0.1.onnx")],
                           melspec_model_path=str(model_dir / "melspectrogram.onnx"),
                           embedding_model_path=str(model_dir / "embedding_model.onnx"), inference_framework="onnx")
        self.buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def push(self, pcm16: np.ndarray) -> bool:
        self.buf = np.concatenate([self.buf, pcm16.astype(np.int16)])
        hit = False
        while len(self.buf) >= 1280:
            frame, self.buf = self.buf[:1280], self.buf[1280:]
            self.last_score = float(max(self.model.predict(frame).values()))
            if self.last_score >= self.threshold:
                hit = True
        if hit:
            self.model.reset()
        return hit


class SherpaZhKWS:
    name = "你好管家"

    def __init__(self, keyword: str = "你好管家", threshold: float = 0.15, model_dir: Path = KWS_DIR):
        import sherpa_onnx
        if keyword in KNOWN_TOKENS:
            toks = KNOWN_TOKENS[keyword]
        else:
            from sherpa_onnx import text2token
            toks = " ".join(text2token([keyword], tokens=str(model_dir / "tokens.txt"), tokens_type="ppinyin")[0])
        kwf = Path(tempfile.mkdtemp(prefix="kws_")) / "kw.txt"
        kwf.write_text(toks + f" @{keyword}\n", encoding="utf-8")
        self.name = keyword
        self.ks = sherpa_onnx.KeywordSpotter(
            tokens=str(model_dir / "tokens.txt"), encoder=str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            decoder=str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"), num_threads=1,
            keywords_file=str(kwf), keywords_threshold=threshold, keywords_score=1.0, provider="cpu")
        self.stream = self.ks.create_stream()

    def push(self, pcm16: np.ndarray) -> bool:
        self.stream.accept_waveform(SR, pcm16.astype(np.float32) / 32768.0)
        while self.ks.is_ready(self.stream):
            self.ks.decode_stream(self.stream)
        if self.ks.get_result(self.stream):
            self.ks.reset_stream(self.stream)
            return True
        return False


class WakeDetector:
    """Runs every available detector on the same 16 kHz int16 stream; returns the detector name on a hit."""

    def __init__(self, enable=("hey_jarvis", "zh")):
        self.detectors, self.errors = [], {}
        if "hey_jarvis" in enable:
            try:
                self.detectors.append(OWWJarvis())
            except Exception as exc:
                self.errors["hey_jarvis"] = f"{type(exc).__name__}: {exc}"
        if "zh" in enable:
            try:
                self.detectors.append(SherpaZhKWS())
            except Exception as exc:
                self.errors["zh"] = f"{type(exc).__name__}: {exc}"

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.detectors]

    def push(self, pcm16: np.ndarray) -> str | None:
        hit = None
        for d in self.detectors:
            try:
                if d.push(pcm16) and hit is None:
                    hit = d.name
            except Exception as exc:
                self.errors[d.name] = f"{type(exc).__name__}: {exc}"
        return hit
