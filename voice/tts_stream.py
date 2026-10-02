"""Plan 38 J2.4: resident Meijia synthesizer.

``ResidentTTS`` keeps one ``tts_daemon`` process (voice/native/tts_daemon.swift, NSSpeechSynthesizer = the engine
behind ``say``) alive and feeds it one sentence at a time, removing the per-sentence ``say`` + ``afconvert``
process spawns. Output is sample-identical to ``say -v Meijia -r <wpm> "[[pbas N]] text"`` + ``afconvert
LEI16@22050`` (verified in J2.7, voice/tts_selfcheck.py). Falls back to ``MacSayTTS`` when the daemon is unavailable.
``AckCache`` pre-renders short acknowledgements in the same voice so something is audible right after the endpoint."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "voice" / "native" / "tts_daemon.swift"
BINARY = ROOT / "runs" / "bin" / "tts_daemon"


def build_daemon(force: bool = False) -> Path | None:
    if BINARY.exists() and not force and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return BINARY
    swiftc = shutil.which("swiftc")
    if not swiftc:
        return None
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([swiftc, "-O", str(SOURCE), "-o", str(BINARY)], capture_output=True, text=True, timeout=300)
    return BINARY if proc.returncode == 0 and BINARY.exists() else None


class ResidentTTS:
    provider = "macos_resident"

    def __init__(self, voice: str = "Meijia", base_wpm: int = 190, fallback=None):
        self.voice = voice
        self.base_wpm = base_wpm
        self.fallback = fallback
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self.ready_info: dict | None = None
        self.start_error: str | None = None

    # -------------------------------------------------------------- process
    def start(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            binary = build_daemon()
            if binary is None:
                self.start_error = "tts_daemon unavailable (needs macOS + swiftc)"
                return False
            self._proc = subprocess.Popen([str(binary), self.voice], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
            line = self._proc.stdout.readline()
            try:
                self.ready_info = json.loads(line)
            except Exception:
                self.start_error = f"bad ready line: {line!r}"
                return False
            if not self.ready_info.get("voice_found"):
                self.start_error = f"voice {self.voice} not installed"
            threading.Thread(target=self._reader, daemon=True).start()
            return True

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
                self._proc.wait(3)
            except Exception:
                self._proc.kill()
            self._proc = None

    def healthcheck(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        return {"provider": self.provider, "voice": self.voice, "resident": alive, "ready": self.ready_info,
                "error": self.start_error, "fallback": getattr(self.fallback, "provider", None),
                "ok": alive or bool(self.fallback)}

    # -------------------------------------------------------------- synthesis
    @staticmethod
    def speakable(text: str) -> str:
        from genai.web.voice import MacSayTTS
        return MacSayTTS.speakable(text)

    def synthesize(self, text: str, outdir: Path, rate: float = 1.0, pitch: float = 1.0, stem: str | None = None,
                   timeout: float = 60.0) -> dict:
        spoken = self.speakable(text)
        if not spoken:
            raise ValueError("nothing to speak")
        if not self.start():
            if self.fallback is None:
                raise RuntimeError(self.start_error or "resident TTS unavailable")
            return self.fallback.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or uuid.uuid4().hex
        wav = outdir / f"{stem}.wav"
        wpm = int(max(90, min(320, self.base_wpm * rate)))
        pbas = int(max(20, min(80, 50 * pitch)))
        jid = uuid.uuid4().hex
        ev = threading.Event()
        self._events[jid] = ev
        t0 = time.time()
        try:
            with self._lock:
                self._proc.stdin.write(json.dumps({"id": jid, "text": f"[[pbas {pbas}]] {spoken}", "voice": self.voice,
                                                   "wpm": wpm, "out": str(wav)}, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            if not ev.wait(timeout):
                raise TimeoutError("resident TTS timed out")
            msg = self._pending.pop(jid)
        except (BrokenPipeError, OSError, TimeoutError) as exc:
            self._proc = None
            if self.fallback is None:
                raise
            out = self.fallback.synthesize(text, outdir, rate=rate, pitch=pitch, stem=stem)
            out["resident_error"] = f"{type(exc).__name__}: {exc}"
            return out
        finally:
            self._events.pop(jid, None)
        if not msg.get("ok"):
            raise RuntimeError(f"resident TTS failed: {msg.get('error')}")
        with wave.open(str(wav)) as w:
            duration = w.getnframes() / float(w.getframerate())
        return {"file": wav.name, "path": str(wav), "duration_s": round(duration, 3), "wpm": wpm, "pbas": pbas,
                "voice": self.voice, "elapsed_s": round(time.time() - t0, 3), "pitch_best_effort": True,
                "provider": self.provider}


ACKS = {"zh": ["嗯。", "好的。", "我在聽。"], "en": ["Mm-hmm.", "Right.", "On it."]}


class AckCache:
    """Pre-rendered acknowledgements in the SAME voice, played immediately after the user's endpoint."""

    def __init__(self, tts, outdir: Path):
        self.tts = tts
        self.outdir = Path(outdir)
        self.files: dict[str, list[str]] = {}
        self._i = 0

    def warm(self) -> dict:
        # Plan 40: a non-default voice gets its own file names (no browser-cache mix-up after a voice switch)
        tag = "" if getattr(self.tts, "provider", "") in ("macos_resident", "macos_say", "") else \
            "_" + __import__("hashlib").sha1(str(getattr(self.tts, "voice", "")).encode()).hexdigest()[:8]
        files: dict[str, list[str]] = {}
        for lang, phrases in ACKS.items():
            files[lang] = []
            for k, p in enumerate(phrases):
                out = self.tts.synthesize(p, self.outdir, stem=f"ack{tag}_{lang}_{k}")
                files[lang].append(out["file"])
        self.files = files
        return self.files

    def retarget(self, tts) -> dict:
        """Plan 40: re-render the acknowledgements in the newly selected voice (same object, sessions keep it)."""
        self.tts = tts
        return self.warm()

    def pick(self, lang: str) -> str | None:
        files = self.files.get(lang) or self.files.get("zh")
        if not files:
            return None
        self._i += 1
        return files[self._i % len(files)]
