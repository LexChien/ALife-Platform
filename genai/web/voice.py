"""Plan 37 G3: offline voice I/O for gemma_web.

- WhisperTranscriber: faster-whisper (CTranslate2, CPU int8), fully offline once
  the model is cached under models/whisper/.
- MacSayTTS: macOS `say` (on-device voices, e.g. Meijia zh_TW) -> 16-bit WAV.
- decode_to_pcm: ffmpeg -> 16 kHz mono float32 (also feeds prosody features).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]


def decode_to_pcm(audio_path: Path, sample_rate: int = 16000):
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass
class WhisperResult:
    transcript: str
    input_path: Path
    normalized_path: Path
    language: Optional[str] = None
    elapsed_s: float = 0.0
    pcm: object = None


class WhisperTranscriber:
    provider = "faster_whisper"

    def __init__(self, model_size: str = "small", language: str = "zh", initial_prompt: str = "以下是繁體中文的句子。",
                 download_root: str | Path = ROOT / "models" / "whisper", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.language = language
        self.initial_prompt = initial_prompt
        self.download_root = Path(download_root)
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type,
                                       download_root=str(self.download_root))
        return self._model

    def preload(self) -> None:
        """Load the model in a background thread so the first voice turn is not slowed by model load."""
        def _run():
            try:
                with self._lock:
                    self._load()
            except Exception as exc:  # recorded in healthcheck
                self._error = f"{type(exc).__name__}: {exc}"
        threading.Thread(target=_run, daemon=True).start()

    def healthcheck(self) -> dict:
        cached = any(self.download_root.glob(f"*whisper-{self.model_size}*")) if self.download_root.exists() else False
        return {"provider": self.provider, "model_size": self.model_size, "installed": self.available(),
                "model_cached": cached, "loaded": self._model is not None, "ffmpeg": shutil.which("ffmpeg"),
                "offline": True, "error": self._error, "ok": self.available() and bool(shutil.which("ffmpeg"))}

    def transcribe_bytes(self, *, audio_bytes: bytes, content_type: str, outdir: Path) -> WhisperResult:
        if not audio_bytes:
            raise ValueError("audio payload is empty")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        suffix = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/mp4": ".m4a",
                  "audio/ogg": ".ogg", "audio/aiff": ".aiff", "audio/mpeg": ".mp3"}.get(content_type.split(";")[0].strip(), ".bin")
        raw_path = outdir / f"{stem}{suffix}"
        raw_path.write_bytes(audio_bytes)
        pcm = decode_to_pcm(raw_path)
        t0 = time.time()
        with self._lock:
            model = self._load()
            segments, info = model.transcribe(pcm, language=self.language, initial_prompt=self.initial_prompt, beam_size=5)
            text = to_traditional("".join(seg.text for seg in segments).strip())
        (outdir / f"{stem}.txt").write_text(text, encoding="utf-8")
        return WhisperResult(transcript=text, input_path=raw_path, normalized_path=raw_path,
                             language=getattr(info, "language", None), elapsed_s=round(time.time() - t0, 3), pcm=pcm)


def to_traditional(text: str) -> str:
    """Whisper often emits Simplified for zh; convert to Taiwan Traditional when OpenCC is available."""
    try:
        from opencc import OpenCC
    except Exception:
        return text
    global _OPENCC
    try:
        _OPENCC
    except NameError:
        _OPENCC = OpenCC("s2twp")
    return _OPENCC.convert(text)


class MLXWhisperTranscriber(WhisperTranscriber):
    """Whisper on the Apple GPU via mlx-whisper (Plan 37 R2). Same interface as WhisperTranscriber.

    Synthetic-say sweep (runs/plan37/stt_bench_r2): large-v3 mean CER 0.047, Eddy 0.036 (fixes the
    Eddy 擔心明天 clip, CER 0.333 -> 0.0); large-v3-turbo mean 0.066, 1.24 s/clip under CPU load.
    """

    provider = "mlx_whisper"

    def __init__(self, model_repo: str = "mlx-community/whisper-large-v3-mlx", language: str = "zh",
                 initial_prompt: str = "以下是繁體中文的句子。") -> None:
        super().__init__(model_size=model_repo, language=language, initial_prompt=initial_prompt)
        self.model_repo = model_repo
        self._loaded = False

    @staticmethod
    def available() -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _run(self, pcm):
        import mlx_whisper

        out = mlx_whisper.transcribe(pcm, path_or_hf_repo=self.model_repo, language=self.language,
                                     initial_prompt=self.initial_prompt)
        self._loaded = True
        return out

    def preload(self) -> None:
        def _go():
            try:
                import numpy as np
                with self._lock:
                    self._run(np.zeros(16000, dtype=np.float32))
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
        threading.Thread(target=_go, daemon=True).start()

    def healthcheck(self) -> dict:
        return {"provider": self.provider, "model_size": self.model_repo, "installed": self.available(),
                "loaded": self._loaded, "ffmpeg": shutil.which("ffmpeg"), "offline": True, "error": self._error,
                "ok": self.available() and bool(shutil.which("ffmpeg"))}

    def transcribe_bytes(self, *, audio_bytes: bytes, content_type: str, outdir: Path) -> WhisperResult:
        if not audio_bytes:
            raise ValueError("audio payload is empty")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        suffix = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/mp4": ".m4a",
                  "audio/ogg": ".ogg", "audio/aiff": ".aiff", "audio/mpeg": ".mp3"}.get(content_type.split(";")[0].strip(), ".bin")
        raw_path = outdir / f"{stem}{suffix}"
        raw_path.write_bytes(audio_bytes)
        pcm = decode_to_pcm(raw_path)
        t0 = time.time()
        with self._lock:
            out = self._run(pcm)
        text = to_traditional((out.get("text") or "").strip())
        (outdir / f"{stem}.txt").write_text(text, encoding="utf-8")
        return WhisperResult(transcript=text, input_path=raw_path, normalized_path=raw_path,
                             language=out.get("language"), elapsed_s=round(time.time() - t0, 3), pcm=pcm)


_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


class MacSayTTS:
    provider = "macos_say"

    def __init__(self, voice: str = "Meijia", base_wpm: int = 190) -> None:
        self.voice = voice
        self.base_wpm = base_wpm
        self.say = shutil.which("say")
        self.afconvert = shutil.which("afconvert")

    def healthcheck(self) -> dict:
        return {"provider": self.provider, "voice": self.voice, "say": self.say, "afconvert": self.afconvert,
                "offline": True, "ok": bool(self.say and self.afconvert)}

    @staticmethod
    def speakable(text: str) -> str:
        t = re.sub(r"[*_`#>|]", " ", text or "")
        t = _EMOJI.sub("", t)
        return re.sub(r"\s+", " ", t).strip()[:1200]

    def synthesize(self, text: str, outdir: Path, rate: float = 1.0, pitch: float = 1.0, stem: str | None = None) -> dict:
        if not (self.say and self.afconvert):
            raise RuntimeError("macOS say/afconvert not available")
        spoken = self.speakable(text)
        if not spoken:
            raise ValueError("nothing to speak")
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or uuid.uuid4().hex
        aiff = outdir / f"{stem}.aiff"
        wav = outdir / f"{stem}.wav"
        tmp_wav = outdir / f"{stem}.partial.wav"  # atomic publish: <stem>.wav exists only when complete
        wpm = int(max(90, min(320, self.base_wpm * rate)))
        # [[pbas]] embedded pitch command; honoured by some voices only (recorded as best-effort).
        pbas = int(max(20, min(80, 50 * pitch)))
        t0 = time.time()
        subprocess.run([self.say, "-v", self.voice, "-r", str(wpm), "-o", str(aiff), f"[[pbas {pbas}]] {spoken}"],
                       check=True, capture_output=True, timeout=120)
        subprocess.run([self.afconvert, "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", str(aiff), str(tmp_wav)],
                       check=True, capture_output=True, timeout=60)
        aiff.unlink(missing_ok=True)
        tmp_wav.replace(wav)
        import wave

        with wave.open(str(wav)) as w:
            duration = w.getnframes() / float(w.getframerate())
        return {"file": wav.name, "path": str(wav), "duration_s": round(duration, 3), "wpm": wpm, "pbas": pbas,
                "voice": self.voice, "elapsed_s": round(time.time() - t0, 3), "pitch_best_effort": True}


_SENT_END = re.compile(r"(?<=[。！？!?；;\n])")


def split_for_tts(text: str, min_chars: int = 8, first_max: int = 40) -> list:
    """Split a reply into speakable chunks (sentence ends). The first chunk is kept short so
    time-to-first-audio is small; tiny fragments are merged into their neighbour."""
    parts = [p.strip() for p in _SENT_END.split(text or "") if p and p.strip()]
    chunks: list = []
    for p in parts:
        if chunks and (len(chunks[-1]) < min_chars or len(p) < min_chars // 2):
            chunks[-1] += p
        else:
            chunks.append(p)
    if chunks and len(chunks[0]) > first_max:
        head = chunks[0]
        cut = max((head.rfind(c, 0, first_max) for c in "，,、 "), default=-1)
        if cut >= min_chars:
            chunks[0:1] = [head[:cut + 1].strip(), head[cut + 1:].strip()]
    return [c for c in chunks if c]
