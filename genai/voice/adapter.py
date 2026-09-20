import json
from pathlib import Path
import subprocess
import sys


class DummyVoiceAdapter:
    def synthesize(self, text, output=None):
        return {"status": "stub", "text": text}


class EspeakVoiceAdapter:
    def __init__(self, voice="auto", rate=165):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text, output):
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        voice = self.voice
        if voice == "auto":
            voice = "cmn" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
        request = {"text": text, "output": str(Path(output).resolve()), "voice": voice, "rate": self.rate}
        proc = subprocess.run([sys.executable, "-m", "genai.voice.espeak_worker"],
                              input=json.dumps(request), text=True, capture_output=True, timeout=120)
        if proc.returncode:
            raise RuntimeError(f"Speech synthesis failed: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout)


def create_voice_adapter(config):
    cfg = dict(config or {})
    backend = cfg.pop("backend", "dummy")
    if backend == "dummy":
        return DummyVoiceAdapter()
    if backend == "espeak_ng":
        return EspeakVoiceAdapter(**cfg)
    raise ValueError(f"Unknown voice backend: {backend}")
