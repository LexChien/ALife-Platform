"""Plan 40 T4.3: resident voice-clone TTS adapter for gemma_web (selectable voice '雅英').

``CloneTTS`` keeps one ``voice/clone_tts_worker.py`` process alive inside the engine's own venv (F5-TTS or
CosyVoice2 need a different Python/torch than gemma_web) and feeds it one sentence at a time, exactly like
``voice.tts_stream.ResidentTTS`` does for Meijia, so the J2 sentence-chunked streaming, ack and barge-in paths are
unchanged. Same ``synthesize()`` return shape as ResidentTTS. Falls back to ``fallback`` (Meijia) on any failure.
The reference clip / weights are private machine-local files (runs/yaying_clone/, ~/yaying_cache), never committed."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "voice" / "clone_tts_worker.py"


class CloneTTS:
    provider = "clone_tts"

    def __init__(self, voice: str, engine: str, python: str, ref_wav: str, ref_text: str, device: str = "mps",
                 nfe: int = 16, fallback=None, ready_timeout: float = 240.0, extra_args: list | None = None):
        self.voice = voice
        self.engine = engine
        self.python = str(Path(python).expanduser())
        self.ref_wav = str((ROOT / ref_wav) if not str(ref_wav).startswith(("/", "~")) else Path(ref_wav).expanduser())
        self.ref_text = ref_text
        self.device = device
        self.nfe = int(nfe)
        self.fallback = fallback
        self.ready_timeout = ready_timeout
        self.extra_args = list(extra_args or [])
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self.ready_info: dict | None = None
        self.start_error: str | None = None

    def available(self) -> bool:
        return Path(self.python).exists() and Path(self.ref_wav).exists() and WORKER.exists()

    def start(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self.ready_info:
                return True
            if not self.available():
                self.start_error = f"clone TTS unavailable (python={Path(self.python).exists()} ref={Path(self.ref_wav).exists()})"
                return False
            cmd = [self.python, str(WORKER), "--engine", self.engine, "--ref", self.ref_wav, "--ref-text", self.ref_text,
                   "--device", self.device, "--nfe", str(self.nfe), *self.extra_args]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          text=True, bufsize=1)
            box: dict = {}

            def _first():
                box["line"] = self._proc.stdout.readline()
            th = threading.Thread(target=_first, daemon=True)
            th.start()
            th.join(self.ready_timeout)
            try:
                self.ready_info = json.loads(box.get("line") or "")
            except Exception:
                self.start_error = f"clone worker not ready: {box.get('line')!r}"
                self._kill()
                return False
            threading.Thread(target=self._reader, daemon=True).start()
            self.start_error = None
            return True

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        self.ready_info = None

    def _reader(self):
        proc = self._proc
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except Exception:
                continue
            jid = msg.get("id")
            if jid in self._events:
                self._pending[jid] = msg
                self._events[jid].set()

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def healthcheck(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        return {"provider": self.provider, "voice": self.voice, "engine": self.engine, "resident": alive,
                "ready": self.ready_info, "error": self.start_error,
                "fallback": getattr(self.fallback, "voice", None), "ok": alive or bool(self.fallback)}

    @staticmethod
    def speakable(text: str) -> str:
        from genai.web.voice import MacSayTTS
        return MacSayTTS.speakable(text)

    def synthesize(self, text: str, outdir: Path, rate: float = 1.0, pitch: float = 1.0, stem: str | None = None,
                   timeout: float = 90.0) -> dict:
        spoken = self.speakable(text)
        if not spoken:
            raise ValueError("nothing to speak")
        if not self.start():
            if self.fallback is None:
                raise RuntimeError(self.start_error or "clone TTS unavailable")
            out = self.fallback.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem)
            out["clone_error"] = self.start_error
            return out
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or uuid.uuid4().hex
        wav = outdir / f"{stem}.wav"
        speed = max(0.7, min(1.4, float(rate)))
        jid = uuid.uuid4().hex
        ev = threading.Event()
        self._events[jid] = ev
        t0 = time.time()
        try:
            with self._lock:
                self._proc.stdin.write(json.dumps({"id": jid, "text": spoken, "out": str(wav), "speed": speed},
                                                  ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            if not ev.wait(timeout):
                raise TimeoutError("clone TTS timed out")
            msg = self._pending.pop(jid)
            if not msg.get("ok"):
                raise RuntimeError(msg.get("error"))
        except (BrokenPipeError, OSError, TimeoutError, RuntimeError) as exc:
            if not isinstance(exc, RuntimeError):
                self._kill()
            if self.fallback is None:
                raise
            out = self.fallback.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem)
            out["clone_error"] = f"{type(exc).__name__}: {exc}"
            return out
        finally:
            self._events.pop(jid, None)
        with wave.open(str(wav)) as w:
            duration = w.getnframes() / float(w.getframerate())
        return {"file": wav.name, "path": str(wav), "duration_s": round(duration, 3), "wpm": None, "pbas": None,
                "speed": speed, "voice": self.voice, "engine": self.engine, "elapsed_s": round(time.time() - t0, 3),
                "worker_synth_s": msg.get("synth_s"), "pitch_best_effort": True, "provider": self.provider}


class HybridTTS:
    """Plan 40 phase 2 fallback route: a fast resident voice speaks chunk 0 (first clause) so first audio stays within the
    J2 budget, the clone voice speaks every later chunk. Services pass ``index`` (chunk number in the turn) when
    ``indexed`` is True; calls without an index (acks, /api/tts) use the clone."""
    provider = "hybrid_clone"
    indexed = True

    def __init__(self, fast, clone, voice: str = "", fast_max_index: int = 0):
        self.fast, self.clone, self.voice = fast, clone, voice or getattr(clone, "voice", "clone")
        self.fast_max_index = fast_max_index
        self.engine = f"hybrid:{getattr(fast, 'voice', 'fast')}+{getattr(clone, 'engine', 'clone')}"
        self.fallback = fast

    def start(self) -> bool:
        return self.clone.start()

    @property
    def start_error(self):
        return getattr(self.clone, "start_error", None)

    def close(self):
        self.clone.close()

    def healthcheck(self) -> dict:
        c = self.clone.healthcheck()
        return {**c, "provider": self.provider, "engine": self.engine, "fast_voice": getattr(self.fast, "voice", None),
                "fast_max_index": self.fast_max_index, "ok": bool(c.get("ok"))}

    speakable = staticmethod(CloneTTS.speakable)

    def synthesize(self, text: str, outdir: Path, rate: float = 1.0, pitch: float = 1.0, stem: str | None = None,
                   index: int | None = None, **kw) -> dict:
        if index is not None and index <= self.fast_max_index:
            out = self.fast.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem)
            return {**out, "hybrid_part": "fast", "provider": self.provider, "engine": self.engine}
        out = self.clone.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem, **kw)
        return {**out, "hybrid_part": "clone", "engine": self.engine, "clone_provider": out.get("provider"),
                "provider": self.provider}
