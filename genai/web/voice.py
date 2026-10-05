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

# 2026-10-03: ONE process-wide lock for every MLX (Apple GPU) call. Two MLX Whisper models warming up concurrently in
# different threads (upload large-v3 preload thread + streaming turbo preload on the main thread) aborted gemma_web
# with libc++abi 'There is no Stream(cpu, 0) in current thread' (start 07:41 log). Serialising MLX use fixes the race.
# 2026-10-05: preferred path is genai.web.mlx_worker (single owner thread). MLX_LOCK remains for any leftover direct
# callers and for nested re-entry; mlx_worker.call is what voice/stt_stream use now.
MLX_LOCK = threading.RLock()

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


_SEP_CHARS = frozenset(" \t\n\r\u3000，。、！？,.!?;；：")


def collapse_repeated_ngrams(
    text: str,
    max_chars: int = 4000,
    ngram: int = 12,
    min_repeats: int = 3,
    *,
    min_period: int = 4,
    max_period: int = 64,
) -> str:
    """Post-filter Whisper repetition hallucinations (same phrase looped dozens of times on long audio).

    Detects contiguous character-level repetition of ANY period ``p`` in
    ``[min_period, max_period]`` (default 4..64; works for zh and en). At each
    position picks the period with the longest repeated run; among equal
    coverage prefers the smallest period (collapses nested loops). Runs with
    ``repeats >= min_repeats`` collapse to one copy. Trailing whitespace /
    punctuation variations between repeats are tolerated via core comparison
    (rstrip of separators). Hard-caps output to ``max_chars``.

    ``ngram`` is kept for backward-compatible call sites and is ignored; period
    detection uses ``min_period``/``max_period``. Pure string; no MLX/torch.
    """
    del ngram  # backward-compat alias; period range replaces fixed n-gram
    if not text:
        return text
    s = text.strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    if len(s) < min_period * min_repeats:
        return s

    def _core(chunk: str) -> str:
        return chunk.rstrip("".join(_SEP_CHARS))

    def _count_run(start: int, period: int) -> tuple[int, int, str]:
        """Return (repeats, end_index, unit_to_emit) for an exact/sep-tolerant run."""
        unit = s[start:start + period]
        if len(unit) < period:
            return 1, start + 1, unit
        unit_core = _core(unit)
        if not unit_core:
            return 1, start + 1, unit
        repeats = 1
        j = start + period
        # Exact fixed-width repeats first (common Whisper case).
        while j + period <= len(s) and s[j:j + period] == unit:
            repeats += 1
            j += period
        # text.strip() may remove the final unit's trailing space/punct; absorb a
        # remainder that is exactly unit_core + optional trailing separators.
        if j < len(s) and unit_core:
            rest = s[j:]
            if rest == unit_core or (rest.startswith(unit_core) and _core(rest) == unit_core):
                repeats += 1
                j = len(s)
        if repeats >= min_repeats:
            return repeats, j, unit
        # Separator-tolerant only when the first unit had trailing seps
        # (otherwise equal-width exact matching already covered copies).
        if len(unit) == len(unit_core):
            return repeats, start + period, unit
        repeats = 1
        j = start + period
        while j < len(s):
            if not s.startswith(unit_core, j):
                break
            k = j + len(unit_core)
            # Require at least one trailing sep when the first unit had one,
            # otherwise allow bare core abutting the next content only at EOS.
            had_sep = len(unit) > len(unit_core)
            if had_sep:
                if k >= len(s):
                    repeats += 1
                    j = k
                    break
                if s[k] not in _SEP_CHARS:
                    break
                while k < len(s) and s[k] in _SEP_CHARS:
                    k += 1
            elif k < len(s) and s[k] not in _SEP_CHARS and _core(s[k:k + period]) != unit_core:
                # No separator and not another bare copy — stop.
                # (Bare exact copies already handled above when widths match.)
                break
            repeats += 1
            j = k
        return repeats, j, unit

    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        best_coverage = 0
        best_period = 0
        best_end = i
        best_unit = ""
        # Allow last copy to be shorter than `period` (sep-tolerant / missing
        # trailing whitespace). Bound only by remaining length.
        max_p = min(max_period, n - i)
        for p in range(min_period, max_p + 1):
            # Quick reject: without trailing seps, first two windows must match
            # for an exact run to ever reach min_repeats.
            unit_peek = s[i:i + p]
            if len(unit_peek) < p:
                break
            if i + 2 * p <= n and s[i + p:i + 2 * p] != unit_peek:
                if len(unit_peek) == len(_core(unit_peek)):
                    continue
            repeats, end, unit = _count_run(i, p)
            if repeats < min_repeats:
                continue
            coverage = end - i  # actual span covered (handles sep-variant lengths)
            # Longest coverage wins; tie-break to smallest period (nested loops).
            if coverage > best_coverage or (coverage == best_coverage and (best_period == 0 or p < best_period)):
                best_coverage = coverage
                best_period = p
                best_end = end
                best_unit = unit
        if best_period > 0:
            out.append(best_unit)
            i = best_end
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


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
        from genai.web.mlx_worker import mlx_call, warm_mel_filters

        # Ensure mel_filters is materialised on the worker before any transcribe.
        warm_mel_filters(128)
        return mlx_call(self._run_locked, mlx_whisper, pcm)

    def _run_locked(self, mlx_whisper, pcm):
        # Anti-repetition for long uploads (2026-10-05: 138 s hallucination). kwargs verified against
        # mlx-whisper 0.4.3 / mlx 0.32.3 transcribe() signature.
        out = mlx_whisper.transcribe(
            pcm,
            path_or_hf_repo=self.model_repo,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )
        self._loaded = True
        return out

    def preload(self) -> None:
        def _go():
            try:
                import numpy as np
                from genai.web.mlx_worker import warm_mel_filters
                warm_mel_filters(128)
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
        text = collapse_repeated_ngrams(to_traditional((out.get("text") or "").strip()))
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
