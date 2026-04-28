from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TranscriptionResult:
    transcript: str
    input_path: Path
    normalized_path: Path


class MacOSSpeechTranscriber:
    def __init__(
        self,
        *,
        locale: str = "zh-TW",
        ffmpeg_path: str | None = None,
        xcrun_path: str | None = None,
        swift_script: str | None = None,
    ) -> None:
        self.locale = locale
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self.xcrun_path = xcrun_path or shutil.which("xcrun")
        self.swift_script = Path(swift_script) if swift_script else ROOT / "tools" / "transcribe_audio.swift"

    def healthcheck(self) -> dict:
        return {
            "provider": "macos_speech",
            "locale": self.locale,
            "ffmpeg": self.ffmpeg_path,
            "xcrun": self.xcrun_path,
            "swift_script": str(self.swift_script),
            "ok": bool(self.ffmpeg_path and self.xcrun_path and self.swift_script.exists()),
        }

    def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        content_type: str,
        outdir: Path,
    ) -> TranscriptionResult:
        if not audio_bytes:
            raise ValueError("audio payload is empty")
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg not found")
        if not self.xcrun_path:
            raise RuntimeError("xcrun not found")
        if not self.swift_script.exists():
            raise RuntimeError(f"swift script not found: {self.swift_script}")

        outdir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        suffix = self._suffix_for_content_type(content_type)
        raw_path = outdir / f"{stem}{suffix}"
        normalized_path = outdir / f"{stem}.aiff"
        raw_path.write_bytes(audio_bytes)

        ffmpeg_cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(raw_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(normalized_path),
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        swift_cmd = [
            self.xcrun_path,
            "swift",
            str(self.swift_script),
            str(normalized_path),
            self.locale,
        ]
        proc = subprocess.run(swift_cmd, capture_output=True, text=True, check=True)
        transcript = proc.stdout.strip()
        if not transcript:
            raise RuntimeError("empty transcript")
        return TranscriptionResult(
            transcript=transcript,
            input_path=raw_path,
            normalized_path=normalized_path,
        )

    @staticmethod
    def _suffix_for_content_type(content_type: str) -> str:
        normalized = (content_type or "").lower()
        if "webm" in normalized:
            return ".webm"
        if "ogg" in normalized:
            return ".ogg"
        if "wav" in normalized:
            return ".wav"
        if "mp4" in normalized or "m4a" in normalized:
            return ".m4a"
        return ".bin"
