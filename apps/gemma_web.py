from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genai.web.service import GemmaWebService


STATIC_DIR = ROOT / "web" / "gemma_chat"


def _read_static_asset(name: str) -> tuple[bytes, str]:
    path = STATIC_DIR / name
    if name.endswith(".html"):
        content_type = "text/html; charset=utf-8"
    elif name.endswith(".js"):
        content_type = "application/javascript; charset=utf-8"
    elif name.endswith(".css"):
        content_type = "text/css; charset=utf-8"
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
        if path == "/api/health":
            self._send_json(self.app.health_payload())
            return
        if path == "/api/life":
            self._send_json(self.app.life_payload())
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
        try:
            payload = self._read_json_body()
            if self.path == "/api/chat":
                response = self.app.chat(
                    payload.get("session_id"),
                    payload.get("message", ""),
                )
                self._send_json(response)
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

    def _handle_transcribe(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("audio body is required")
            audio_bytes = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "application/octet-stream")
            session_id = self.headers.get("X-Session-ID")
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
