"""Loopback-first, persistent single-worker API for platform engines."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


class JobRequest(BaseModel):
    target: Literal["asal", "clone", "genai"]
    config: str = Field(min_length=1, max_length=300)
    profile: str | None = Field(default=None, max_length=80)
    prompt: str | None = Field(default=None, max_length=4000)


class JobManager:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT / "runs/service").resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=16)
        self.stopping = threading.Event()
        self.process = None
        self.lock = threading.Lock()
        for path in self.directory.glob("*/job.json"):
            job = json.loads(path.read_text())
            if job["status"] in ("running", "queued"):
                job.update(status="interrupted", error="Service restarted before job completion", completed_at=now())
                self.save(job)
        self.thread = threading.Thread(target=self.work, name="alife-jobs", daemon=True)

    def save(self, job):
        path = self.directory / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2, ensure_ascii=False))
        temporary.replace(path)

    def get(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise HTTPException(404, "Unknown job")
        path = self.directory / identifier / "job.json"
        if not path.is_file():
            raise HTTPException(404, "Unknown job")
        return json.loads(path.read_text())

    def submit(self, request):
        config = (ROOT / request.config).resolve()
        expected = (ROOT / "configs" / request.target).resolve()
        if not config.is_relative_to(expected) or config.suffix not in (".yaml", ".yml") or not config.is_file():
            raise HTTPException(400, "Config must be an existing YAML under configs/<target>")
        try:
            resolved = load_config(str(config), profile=request.profile)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        if request.prompt is not None:
            if request.target == "clone":
                resolved["inputs"] = [request.prompt]
            else:
                resolved["prompt"] = request.prompt
        with self.lock:
            if self.pending.full() or self.stopping.is_set():
                raise HTTPException(503, "Job queue unavailable")
            identifier = uuid.uuid4().hex
            directory = self.directory / identifier
            directory.mkdir()
            job = {"id": identifier, "target": request.target, "config": str(config.relative_to(ROOT)),
                   "profile": request.profile, "resolved_config": resolved,
                   "run_dir": str(directory / "artifacts"), "status": "queued", "created_at": now()}
            self.save(job)
            self.pending.put_nowait(identifier)
        return job

    def work(self):
        while not self.stopping.is_set():
            try:
                identifier = self.pending.get(timeout=0.25)
            except queue.Empty:
                continue
            job = self.get(identifier)
            job.update(status="running", started_at=now())
            self.save(job)
            path = self.directory / identifier
            try:
                with (path / "worker.log").open("w") as log:
                    self.process = subprocess.Popen(
                        [sys.executable, str(ROOT / "tools/run_python.py"), "-m", "apps.job_worker", str(path / "job.json")],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    code = self.process.wait(timeout=3600)
                job.update(status="completed" if code == 0 else "failed", exit_code=code)
                if code:
                    job["error"] = (path / "worker.log").read_text()[-3000:]
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
                job.update(status="failed", error="Job exceeded 3600-second limit")
            except Exception as exc:
                job.update(status="failed", error=str(exc))
            finally:
                self.process = None
                job["completed_at"] = now()
                self.save(job)
                self.pending.task_done()

    def close(self):
        self.stopping.set()
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        self.thread.join(timeout=5)


def create_app(directory=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = JobManager(directory)
        app.state.jobs.thread.start()
        yield
        app.state.jobs.close()

    app = FastAPI(title="ALife Platform", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def authorize(request: Request, call_next):
        token = os.environ.get("ALIFE_API_TOKEN")
        if token:
            import hmac
            from fastapi.responses import JSONResponse
            if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        packages = {}
        for name in ("numpy", "torch", "chromadb", "mlflow", "transformers", "diffusers"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {"status": "ready", "version": app.version, "packages": packages,
                "execution": "serialized subprocesses", "targets": ["asal", "clone", "genai"]}

    @app.get("/configs")
    def configs():
        return sorted(str(path.relative_to(ROOT)) for path in (ROOT / "configs").glob("*/*.yaml"))

    @app.post("/jobs", status_code=202)
    def submit(request: JobRequest):
        return app.state.jobs.submit(request)

    @app.get("/jobs/{identifier}")
    def get_job(identifier: str):
        return app.state.jobs.get(identifier)

    @app.get("/jobs/{identifier}/artifacts/{filename:path}")
    def artifact(identifier: str, filename: str):
        job = app.state.jobs.get(identifier)
        root = Path(job["run_dir"]).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path)

    return app


app = create_app()
