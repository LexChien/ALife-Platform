import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from apps.service import create_app


class ServiceTests(unittest.TestCase):
    def test_http_job_executes_and_serves_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                result = client.post("/jobs", json={"target": "genai", "config": "configs/genai/genai_baseline.yaml"})
                self.assertEqual(result.status_code, 202, result.text)
                job = result.json()
                deadline = time.monotonic() + 45
                while job["status"] in ("queued", "running") and time.monotonic() < deadline:
                    time.sleep(0.1)
                    job = client.get(f"/jobs/{job['id']}").json()
                self.assertEqual(job["status"], "completed", job)
                payload = client.get(f"/jobs/{job['id']}/artifacts/result.json")
                self.assertEqual(payload.status_code, 200)
                self.assertEqual(payload.json()["output"]["llm"]["backend"], "dummy")
                forbidden = client.get(f"/jobs/{job['id']}/artifacts/%2E%2E%2Fjob.json")
                self.assertEqual(forbidden.status_code, 404)
            with TestClient(create_app(directory)) as restarted:
                self.assertEqual(restarted.get(f"/jobs/{job['id']}").json()["status"], "completed")

    def test_rejects_config_escape_and_requires_configured_token(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(directory)) as client:
                for config in ("README.md", "configs/clone/clone_baseline.yaml", "configs/genai/../../README.md"):
                    response = client.post("/jobs", json={"target": "genai", "config": config})
                    self.assertEqual(response.status_code, 400)
                with patch.dict("os.environ", {"ALIFE_API_TOKEN": "test-token"}):
                    self.assertEqual(client.get("/health").status_code, 401)
                    self.assertEqual(client.get("/health", headers={"Authorization": "Bearer test-token"}).status_code, 200)

    def test_marks_unfinished_job_interrupted_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            identifier = "a" * 32
            path = Path(directory) / identifier
            path.mkdir()
            (path / "job.json").write_text(json.dumps({"id": identifier, "status": "running"}))
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get(f"/jobs/{identifier}").json()["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
