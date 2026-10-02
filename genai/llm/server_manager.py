"""Plan 38 J1.1: resident llama-server lifecycle (start / health / restart / watchdog).

The server runs in its own tmux session (default ``gemma_llm``) so it survives gemma_web restarts and agent shell
cleanup; gemma_web adopts a healthy server on the configured port instead of reloading the model. A watchdog thread
restarts it after a crash (kill -9 target: healthy again <= 10 s). Reasoning is OFF via ``-rea off`` only: Plan 38
measured 0/6 leaks for ``-rea off`` and 5/6 meta-narration leaks for ``--reasoning-budget 0`` alone, so the budget
flag is not passed (runs/plan38/llm/reasoning_modes.json, re-verified in J1).

CLI: python -m genai.llm.server_manager {ensure,status,stop,restart} [--port 8091] [--model ...]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "third_party" / "llama.cpp" / "build-plan38" / "bin" / "llama-server"
DEFAULT_MODEL = ROOT / "models" / "gemma" / "gemma.gguf"


@dataclass
class ServerConfig:
    binary: str = str(DEFAULT_BINARY)
    model: str = str(DEFAULT_MODEL)
    host: str = "127.0.0.1"
    port: int = 8091
    ctx: int = 16384
    n_parallel: int = 2
    n_gpu_layers: int = 99
    cache_reuse: int = 256
    reasoning_budget_zero: bool = False  # kept only for A/B tests; -rea off is what disables thinking
    extra_args: list[str] = field(default_factory=list)
    tmux_session: str = "gemma_llm"
    log_path: str = str(ROOT / "runs" / "live_engine" / "llama_server.log")
    launcher: str = "auto"  # auto -> tmux when available, else subprocess

    @classmethod
    def from_dict(cls, d: dict | None) -> "ServerConfig":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("binary", "model", "log_path"):
            if k in known and not os.path.isabs(str(known[k])):
                known[k] = str(ROOT / str(known[k]))
        return cls(**known)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        cmd = [self.binary, "-m", self.model, "--host", self.host, "--port", str(self.port), "-ngl", str(self.n_gpu_layers),
               "-c", str(self.ctx), "--jinja", "-rea", "off", "-np", str(self.n_parallel), "--cache-reuse",
               str(self.cache_reuse), "--metrics"]
        if self.reasoning_budget_zero:
            cmd += ["--reasoning-budget", "0"]
        return cmd + list(self.extra_args)


def http_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except Exception:
        return None


class LlamaServerManager:
    def __init__(self, config: ServerConfig | dict | None = None):
        self.cfg = config if isinstance(config, ServerConfig) else ServerConfig.from_dict(config)
        self._proc: subprocess.Popen | None = None
        self._watchdog: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.events: list[dict] = []
        self.adopted = False

    # ------------------------------------------------------------------ status
    def health(self) -> dict:
        h = http_json(self.cfg.url + "/health")
        ok = bool(h) and h.get("status") == "ok"
        return {"ok": ok, "url": self.cfg.url, "status": (h or {}).get("status", "down"), "pid": self.pid(),
                "launcher": self._launcher(), "adopted": self.adopted, "restarts": sum(e["event"] == "restart" for e in self.events),
                "last_event": self.events[-1] if self.events else None}

    def is_healthy(self) -> bool:
        return self.health()["ok"]

    def pid(self) -> int | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        try:
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{self.cfg.port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                                 text=True, timeout=3).stdout.split()
            return int(out[0]) if out else None
        except Exception:
            return None

    def _launcher(self) -> str:
        if self.cfg.launcher == "auto":
            return "tmux" if shutil.which("tmux") else "subprocess"
        return self.cfg.launcher

    # ------------------------------------------------------------------ lifecycle
    def start(self, wait_s: float = 90.0) -> dict:
        with self._lock:
            if self.is_healthy():
                self.adopted = self._proc is None
                self._event("adopt" if self.adopted else "already_up")
                return self.health()
            if not Path(self.cfg.binary).exists():
                raise FileNotFoundError(f"llama-server binary not found: {self.cfg.binary}")
            if not Path(self.cfg.model).exists():
                raise FileNotFoundError(f"GGUF model not found: {self.cfg.model}")
            Path(self.cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
            cmd = self.cfg.command()
            t0 = time.time()
            if self._launcher() == "tmux":
                subprocess.run(["tmux", "kill-session", "-t", self.cfg.tmux_session], capture_output=True)
                quoted = " ".join(_sh_quote(c) for c in cmd)
                subprocess.run(["tmux", "new-session", "-d", "-s", self.cfg.tmux_session,
                                f"exec {quoted} >> {_sh_quote(self.cfg.log_path)} 2>&1"], check=True)
            else:
                log = open(self.cfg.log_path, "ab")
                self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            ok = self._wait_healthy(wait_s)
            self.adopted = False
            self._event("start", ok=ok, load_s=round(time.time() - t0, 2))
            if not ok:
                raise RuntimeError(f"llama-server did not become healthy in {wait_s}s (log: {self.cfg.log_path})")
            return self.health()

    def _wait_healthy(self, wait_s: float) -> bool:
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.is_healthy():
                return True
            if self._proc is not None and self._proc.poll() is not None:
                return False
            time.sleep(0.2)
        return False

    def stop(self) -> None:
        with self._lock:
            if self._launcher() == "tmux":
                subprocess.run(["tmux", "kill-session", "-t", self.cfg.tmux_session], capture_output=True)
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            pid = self.pid()
            if pid:
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
            self._proc = None
            self._event("stop")

    def restart(self, reason: str = "manual") -> dict:
        t0 = time.time()
        self.stop()
        time.sleep(0.3)
        h = self.start()
        self._event("restart", reason=reason, recovery_s=round(time.time() - t0, 2))
        return h

    def ensure(self) -> dict:
        return self.health() if self.is_healthy() else self.start()

    # ------------------------------------------------------------------ watchdog
    def start_watchdog(self, interval_s: float = 1.0, failures_before_restart: int = 2) -> None:
        if self._watchdog and self._watchdog.is_alive():
            return

        def loop():
            fails, down_since = 0, None
            while not self._stop.wait(interval_s):
                if self.is_healthy():
                    fails, down_since = 0, None
                    continue
                fails += 1
                down_since = down_since or time.time()
                if fails >= failures_before_restart:
                    try:
                        self.restart(reason="watchdog")
                        self._event("recovered", downtime_s=round(time.time() - down_since, 2))
                    except Exception as exc:  # keep watching; record the failure
                        self._event("restart_failed", error=f"{type(exc).__name__}: {exc}")
                    fails, down_since = 0, None
        self._watchdog = threading.Thread(target=loop, name="llama-server-watchdog", daemon=True)
        self._watchdog.start()

    def stop_watchdog(self) -> None:
        self._stop.set()

    def _event(self, event: str, **kw) -> None:
        self.events.append({"event": event, "t": round(time.time(), 3), **kw})
        self.events = self.events[-50:]


def _sh_quote(s: str) -> str:
    import shlex
    return shlex.quote(str(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["ensure", "status", "stop", "restart"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("GEMMA_LLM_PORT", "8091")))
    ap.add_argument("--model", default=os.environ.get("GEMMA_LLM_MODEL", str(DEFAULT_MODEL)))
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()
    m = LlamaServerManager(ServerConfig(port=a.port, model=a.model, ctx=a.ctx))
    out = {"ensure": m.ensure, "status": m.health, "restart": m.restart}.get(a.action, None)
    if a.action == "stop":
        m.stop()
        print(json.dumps({"stopped": True}))
        return 0
    res = out()
    print(json.dumps({**res, "events": m.events}))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
