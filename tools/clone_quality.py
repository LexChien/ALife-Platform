#!/usr/bin/env python3
"""Run real Clone memory/behavior verification in independent OS processes.

The writer only stores supplied user memories. Readers receive prompts without
expected answers; the parent scores returned generations against labeled cases.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def worker(payload_path):
    from digital_clone.engine import DigitalCloneEngine
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    engine = DigitalCloneEngine(payload["config"], payload["run_dir"])
    initial_user_items = [item for item in engine.memory.items if item["role"] == "user"]
    for memory in payload.get("seed_memories", []):
        engine.memory.add("user", memory)
    result = engine.run()
    result.update({"pid": os.getpid(), "initial_current_session_user_memories": len(initial_user_items),
                   "action": payload["action"]})
    write_json(payload["result_path"], result)


def run(config_path, outdir, backends):
    from core.config import load_config
    from core.model_registry import get_model_spec
    from digital_clone.eval import score_case, summarize, render_markdown
    cfg = load_config(str(config_path))
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Require a fresh output/database to prevent accidental contamination.
    if (out / "lineage.json").exists():
        raise ValueError(f"Output already used: {out}; use a new directory")
    spec = get_model_spec(cfg["llm"]["model_id"])
    model_path = Path(spec.model_path)
    source_paths = sorted(list((ROOT / "digital_clone").rglob("*.py")) + [Path(__file__), Path(config_path)])
    lineage = {"version": cfg["version"], "created_at": datetime.now(timezone.utc).isoformat(),
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_sha256": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): digest(path) for path in source_paths},
               "model_id": cfg["llm"]["model_id"], "model_path": str(model_path),
               "model_sha256": digest(model_path), "python": sys.executable,
               "evaluator": "labeled_substring_behavior_v2", "backends": backends,
               "note": "Model prompts exclude evaluation labels. No generated answer is replaced by a target."}
    write_json(out / "lineage.json", lineage)
    write_json(out / "resolved_config.json", cfg)
    reports = {}
    for backend in backends:
        backend_out = out / backend
        backend_out.mkdir()
        engine_cfg = {key: deepcopy(cfg[key]) for key in ("persona", "llm", "memory")}
        engine_cfg["memory"]["persist_directory"] = str(backend_out / "memory")
        if backend == "dummy":
            engine_cfg["llm"] = {"backend": "dummy", "model_family": "dummy"}
        jobs = [{"id": "writer", "action": "write", "seed_memories": cfg["seed_memories"], "cases": []}]
        jobs += [{**session, "action": "read"} for session in cfg["sessions"]]
        all_results, processes = [], []
        for job in jobs:
            session_out = backend_out / job["id"]
            session_out.mkdir()
            session_cfg = deepcopy(engine_cfg)
            session_cfg["persona"]["id"] = job.get("persona_id", engine_cfg["persona"]["id"])
            session_cfg["persona"]["name"] = job.get("persona_name", engine_cfg["persona"]["name"])
            # Only input text is sent to the worker, never expected answer labels.
            session_cfg["inputs"] = [case["input"] for case in job["cases"]]
            payload = {"action": job["action"], "config": session_cfg, "run_dir": str(session_out),
                       "result_path": str(session_out / "outputs.json"), "seed_memories": job.get("seed_memories", [])}
            write_json(session_out / "worker_input.json", payload)
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(session_out / "worker_input.json")],
                                     cwd=ROOT, capture_output=True, text=True, timeout=900,
                                     env={**os.environ, "ANONYMIZED_TELEMETRY": "False", "PYTHONDONTWRITEBYTECODE": "1"})
            (session_out / "stdout.txt").write_text(process.stdout, encoding="utf-8")
            (session_out / "stderr.txt").write_text(process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"Session {backend}/{job['id']} failed; inspect {session_out / 'stderr.txt'}")
            result = json.loads((session_out / "outputs.json").read_text(encoding="utf-8"))
            processes.append({"session": job["id"], "pid": result["pid"], "action": result["action"],
                              "initial_current_session_user_memories": result["initial_current_session_user_memories"],
                              "memory_collection": result["summary"]["memory_collection"],
                              "memory_backend": result["summary"]["memory_backend"]})
            if len(result["outputs"]) != len(job["cases"]):
                raise RuntimeError("Mismatch between expected cases and generated outputs")
            scored = [score_case(session_cfg["persona"], case, output, {})
                      for case, output in zip(job["cases"], result["outputs"])]
            for row in scored:
                row["session_id"] = job["id"]
            all_results.extend(scored)
            write_json(session_out / "scores.json", scored)
            print(f"{backend}/{job['id']}: {sum(row['pass'] for row in scored)}/{len(scored)}", flush=True)
        summary = summarize(all_results)
        summary["process_isolation"] = {
            "fresh_user_state": all(item["initial_current_session_user_memories"] == 0 for item in processes),
            "real_persistence": all(item["memory_backend"] == "chromadb" for item in processes),
            "num_sessions": len(processes), "processes": processes,
        }
        summary["all_passed"] = summary["all_passed"] and all(summary["process_isolation"][key] for key in ("fresh_user_state", "real_persistence"))
        write_json(backend_out / "results.json", all_results)
        write_json(backend_out / "summary.json", summary)
        (backend_out / "report.md").write_text(render_markdown(str(config_path), summary, all_results, []), encoding="utf-8")
        reports[backend] = summary
    reports["acceptance"] = {"real_model_passed": reports.get("llama_cpp", {}).get("all_passed", False),
                              "dummy_negative_control_rejected": not reports.get("dummy", {}).get("all_passed", True),
                              "scope": "7 fixed cases, 5 reader sessions plus writer per backend; not unrestricted persona quality."}
    write_json(out / "summary.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clone/clone_quality.yaml")
    parser.add_argument("--outdir", default="runs/digital_clone/quality_v2")
    parser.add_argument("--backends", nargs="+", choices=["dummy", "llama_cpp"], default=["dummy", "llama_cpp"])
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    result = run(Path(args.config).resolve(), args.outdir, args.backends)
    print(json.dumps(result["acceptance"], ensure_ascii=False))
    if "llama_cpp" in args.backends:
        return 0 if result["acceptance"]["real_model_passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
