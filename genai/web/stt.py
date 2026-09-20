from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
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
        self.swift_app_script = Path(__file__).with_name("transcribe_audio_fileout.swift")
        self.open_path = shutil.which("open")
        self.codesign_path = shutil.which("codesign")
        self.helper_root = Path(os.environ.get("ALIFE_SPEECH_HELPER_DIR", "/private/tmp/alife_speech_helper"))

    def healthcheck(self) -> dict:
        return {
            "provider": "macos_speech",
            "locale": self.locale,
            "ffmpeg": self.ffmpeg_path,
            "xcrun": self.xcrun_path,
            "swift_script": str(self.swift_script),
            "swift_app_script": str(self.swift_app_script),
            "open": self.open_path,
            "codesign": self.codesign_path,
            "helper_root": str(self.helper_root),
            "ok": bool(
                self.ffmpeg_path
                and self.xcrun_path
                and self.open_path
                and self.swift_app_script.exists()
            ),
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
        if not self.open_path:
            raise RuntimeError("open command not found")
        if not self.swift_app_script.exists():
            raise RuntimeError(f"swift app script not found: {self.swift_app_script}")

        outdir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        suffix = self._suffix_for_content_type(content_type)
        raw_path = outdir / f"{stem}{suffix}"
        normalized_path = outdir / f"{stem}.normalized.aiff"
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
        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"ffmpeg failed: {detail}") from exc

        helper_app = self._ensure_helper_app()
        transcript_path = outdir / f"{stem}.txt"
        app_cmd = [
            self.open_path,
            "-W",
            str(helper_app),
            "--args",
            str(normalized_path.resolve()),
            str(transcript_path.resolve()),
            self.locale,
        ]
        try:
            subprocess.run(app_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"speech transcription app failed: {detail}") from exc
        if not transcript_path.exists():
            raise RuntimeError("speech transcription failed: no transcript output")
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if transcript.startswith("ERROR: "):
            detail = transcript.removeprefix("ERROR: ").strip()
            if "speech authorization denied" in detail:
                detail = (
                    "macOS Speech Recognition permission denied for ALifeTranscribeAudio. "
                    "Enable ALifeTranscribeAudio in System Settings > Privacy & Security > Speech Recognition."
                )
            raise RuntimeError(f"speech transcription failed: {detail}")
        if not transcript:
            raise RuntimeError("empty transcript")
        return TranscriptionResult(
            transcript=transcript,
            input_path=raw_path,
            normalized_path=normalized_path,
        )

    def _ensure_helper_app(self) -> Path:
        app_root = self.helper_root / "ALifeTranscribeAudio.app"
        contents_dir = app_root / "Contents"
        macos_dir = contents_dir / "MacOS"
        exe_path = macos_dir / "ALifeTranscribeAudio"
        info_path = contents_dir / "Info.plist"
        module_cache = self.helper_root / "module_cache"
        source_mtime = self.swift_app_script.stat().st_mtime
        needs_build = not exe_path.exists() or exe_path.stat().st_mtime < source_mtime
        if not needs_build:
            return app_root

        macos_dir.mkdir(parents=True, exist_ok=True)
        module_cache.mkdir(parents=True, exist_ok=True)
        info_path.write_text(
            """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\"
  \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key>
  <string>ALifeTranscribeAudio</string>
  <key>CFBundleIdentifier</key>
  <string>com.lexchien.alife.transcribe-audio</string>
  <key>CFBundleName</key>
  <string>ALifeTranscribeAudio</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>NSSpeechRecognitionUsageDescription</key>
  <string>ALife uses local macOS Speech Recognition to transcribe uploaded voice recordings.</string>
</dict>
</plist>
""",
            encoding="utf-8",
        )
        build_cmd = [
            self.xcrun_path,
            "swiftc",
            "-module-cache-path",
            str(module_cache),
            str(self.swift_app_script),
            "-o",
            str(exe_path),
        ]
        try:
            subprocess.run(build_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"speech helper build failed: {detail}") from exc
        if self.codesign_path:
            sign_cmd = [self.codesign_path, "--force", "--deep", "--sign", "-", str(app_root)]
            try:
                subprocess.run(sign_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or str(exc)).strip()
                raise RuntimeError(f"speech helper codesign failed: {detail}") from exc
        return app_root

    @staticmethod
    def _suffix_for_content_type(content_type: str) -> str:
        normalized = (content_type or "").lower()
        if "webm" in normalized:
            return ".webm"
        if "ogg" in normalized:
            return ".ogg"
        if "wav" in normalized:
            return ".wav"
        if "aiff" in normalized or "aif" in normalized:
            return ".aiff"
        if "mp4" in normalized or "m4a" in normalized:
            return ".m4a"
        return ".bin"
