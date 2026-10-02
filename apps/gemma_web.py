from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genai.web.service import GemmaWebService


STATIC_DIR = ROOT / "web" / "gemma_chat"
# Plan 38: extra static scripts + private mouth keyframes rendered from the fixed avatar (guard-checked, never in git)
STATIC_SCRIPTS = {"/hud.js", "/avatar.js", "/realtime.js", "/pcm-worklet.js"}
MOUTH_DIR = ROOT / "runs" / "plan38" / "avatar" / "mouth"


def _read_static_asset(name: str) -> tuple[bytes, str]:
    path = STATIC_DIR / name
    if name.endswith(".html"):
        content_type = "text/html; charset=utf-8"
    elif name.endswith(".js"):
        content_type = "application/javascript; charset=utf-8"
    elif name.endswith(".css"):
        content_type = "text/css; charset=utf-8"
    elif name.endswith(".mp4"):
        content_type = "video/mp4"
    elif name.endswith(".jpg"):
        content_type = "image/jpeg"
    else:
        content_type = "application/octet-stream"
    return path.read_bytes(), content_type


class GemmaWebHandler(BaseHTTPRequestHandler):
    server_version = "GemmaWeb/0.1"

    @property
    def app(self) -> GemmaWebService:
        return self.server.app

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._serve_static("index.html")
            return
        if path == "/app.js":
            self._serve_static("app.js")
            return
        if path == "/styles.css":
            self._serve_static("styles.css")
            return
        if path == "/generated_video-3.mp4":
            self._serve_static("generated_video-3.mp4")
            return
        if path == "/avatar.jpg":
            self._serve_static("avatar.jpg")
            return
        if path in STATIC_SCRIPTS:
            self._serve_static(path.lstrip("/"))
            return
        if path.startswith("/avatar/clips/"):
            self._serve_clip(path.removeprefix("/avatar/clips/"))
            return
        if path.startswith("/avatar/mouth/"):
            self._serve_mouth(path.removeprefix("/avatar/mouth/"))
            return
        if path == "/api/thoughts":
            n = int((parse_qs(parsed.query).get("n") or ["20"])[0])
            self._send_json(self.app.thoughts_payload(n))
            return
        if path in {"/mic_check", "/mic_check.html"}:
            self._serve_static("mic_check.html")
            return
        if path == "/api/health":
            self._send_json(self.app.health_payload())
            return
        if path == "/api/life":
            self._send_json(self.app.life_payload())
            return
        if path == "/api/dna":
            self._send_json({"ok": True, **self.app.dna_payload()})
            return
        if path == "/api/profile":  # Plan 40: active persona/voice profile + available ones
            self._send_json(self.app.profile_payload())
            return
        if path == "/api/emotion":
            sid = (parse_qs(parsed.query).get("session_id") or [None])[0]
            self._send_json(self.app.emotion_payload(sid))
            return
        if path.startswith("/api/tts/"):
            try:
                body = self.app.read_tts(path.removeprefix("/api/tts/"))
            except FileNotFoundError:
                self._send_json({"ok": False, "error": "tts_missing"}, status=HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/artifacts/asal/"):
            self._serve_life_artifact(path)
            return
        self._send_json(
            {"ok": False, "error": "not_found", "path": path},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:
        if self.path == "/api/transcribe":
            self._handle_transcribe()
            return
        if self.path == "/api/voice_chat":
            self._handle_transcribe(voice_chat=True)
            return
        try:
            payload = self._read_json_body()
            if self.path == "/api/chat":
                response = self.app.chat(
                    payload.get("session_id"),
                    payload.get("message", ""),
                    want_tts=bool(payload.get("tts", False)),
                )
                self._send_json(response)
                return
            if self.path == "/api/mic_check_log":
                self._send_json(self.app.save_mic_check(payload))
                return
            if self.path == "/api/tts":
                self._send_json(self.app.synthesize(payload.get("session_id"), payload.get("text", "")))
                return
            if self.path == "/api/profile":  # Plan 40: {"profile": "yaying"|"digiclone", "voice": optional}
                try:
                    if payload.get("profile"):
                        self._send_json(self.app.apply_profile(str(payload["profile"]), voice=payload.get("voice")))
                    elif payload.get("voice"):
                        self._send_json({"ok": True, "tts": self.app.set_voice(str(payload["voice"])), **self.app.profile_payload()})
                    else:
                        raise ValueError("need 'profile' or 'voice'")
                except (KeyError, ValueError) as exc:
                    raise ValueError(str(exc)) from exc
                return
            if self.path == "/api/reset":
                response = self.app.reset(payload.get("session_id"))
                self._send_json(response)
                return
            self._send_json(
                {"ok": False, "error": "not_found", "path": self.path},
                status=HTTPStatus.NOT_FOUND,
            )
        except ValueError as exc:
            self._send_json(
                {"ok": False, "error": "bad_request", "detail": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": "internal_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_transcribe(self, voice_chat: bool = False) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("audio body is required")
            audio_bytes = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "application/octet-stream")
            session_id = self.headers.get("X-Session-ID")
            if voice_chat:
                response = self.app.voice_chat(session_id, audio_bytes, content_type)
            else:
                response = self.app.transcribe(session_id, audio_bytes, content_type)
            self._send_json(response)
        except ValueError as exc:
            self._send_json(
                {"ok": False, "error": "bad_request", "detail": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": "internal_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args) -> None:
        return

    def _serve_static(self, asset_name: str) -> None:
        try:
            body, content_type = _read_static_asset(asset_name)
        except FileNotFoundError:
            self._send_json(
                {"ok": False, "error": "asset_missing", "asset": asset_name},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_clip(self, name: str) -> None:
        """Plan 38 J3.1: guard-passed LivePortrait clips of the fixed avatar (web/gemma_chat/clips, private)."""
        import re as _re
        clip_dir = STATIC_DIR / "clips"
        if not _re.fullmatch(r"(?:idle|smile|concerned|talking)\.mp4|manifest\.json", name) or not (clip_dir / name).exists():
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = (clip_dir / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _serve_mouth(self, name: str) -> None:
        import re as _re
        if not _re.fullmatch(r"k[0-7]\.jpg|manifest\.json", name) or not (MOUTH_DIR / name).exists():
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = (MOUTH_DIR / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg" if name.endswith(".jpg") else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _serve_life_artifact(self, path: str) -> None:
        rel = unquote(path.removeprefix("/artifacts/asal/"))
        run_id, sep, asset = rel.partition("/")
        if not sep:
            self._send_json(
                {"ok": False, "error": "artifact_missing", "path": path},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        try:
            body, content_type = self.app.read_life_artifact(run_id, asset)
        except FileNotFoundError:
            self._send_json(
                {"ok": False, "error": "artifact_missing", "path": path},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local Gemma web chat UI with text input and browser microphone support."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--ws-port", type=int, default=None, help="realtime WebSocket port (default: port+1; 0 disables)")
    args = parser.parse_args()

    app = GemmaWebService(
        config_path=args.config,
        profile=args.profile,
        host=args.host,
        port=args.port,
        history_turns=args.history_turns,
    )
    server = ThreadingHTTPServer((args.host, args.port), GemmaWebHandler)
    server.app = app
    ws_port = args.port + 1 if args.ws_port is None else args.ws_port
    if ws_port and app.loop is not None:
        from genai.web.realtime import RealtimeServer
        rt = RealtimeServer(app, host=args.host, port=ws_port)
        rt.warm()
        ok = rt.start()
        app.realtime = rt
        app.ws_port = ws_port if ok else None
        print(f"Realtime WebSocket {'listening on ws://%s:%d/ws/session' % (args.host, ws_port) if ok else 'failed: ' + str(rt.error)}")
    print(f"Gemma web chat listening on http://{args.host}:{args.port}")
    print(f"Run artifacts: {app.run_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Gemma web chat.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
