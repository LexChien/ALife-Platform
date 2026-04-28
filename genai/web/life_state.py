from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import json
import re


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class ASALProgressIndex:
    def __init__(self, *, root: Path | None = None, config: dict | None = None) -> None:
        self.root = Path(root or ROOT)
        self.config = config or {}
        run_roots = self.config.get("run_roots") or [
            "runs/asal",
            "research/asal_engine/runs",
        ]
        history_roots = self.config.get("history_roots") or [
            "log",
            "research/asal_engine/history/log",
        ]
        self.run_roots = [self.root / root for root in run_roots]
        self.history_roots = [self.root / root for root in history_roots]

    def snapshot(self) -> dict:
        runs = self._collect_runs()
        latest = runs[0] if runs else None
        capabilities = self._capabilities(latest)
        tasks = self._tasks()
        worklogs = self._latest_worklogs()
        summary = self._summary(latest, capabilities, tasks)
        return {
            "enabled": True,
            "schema_version": "1.0",
            "summary": summary,
            "latest_run": latest,
            "recent_runs": runs[:5],
            "capabilities": capabilities,
            "tasks": tasks,
            "worklogs": worklogs,
        }

    def context_text(self, snapshot: dict | None = None) -> str:
        snap = snapshot or self.snapshot()
        latest = snap.get("latest_run") or {}
        metrics = latest.get("metrics") or {}
        details = latest.get("details") or {}
        capabilities = snap.get("capabilities") or []
        tasks = snap.get("tasks") or {}

        lines = [
            "人工生命原型狀態上下文。這些是本機專案事實，回答時不得誇大成產品完成：",
            f"- ASAL 進度摘要：{snap.get('summary')}",
        ]
        if latest:
            lines.extend(
                [
                    f"- 最新 ASAL run：{latest.get('run_id')}，status={latest.get('status')}，config={latest.get('config_path')}",
                    f"- substrate={details.get('substrate')}，foundation_model={details.get('foundation_model')}",
                    f"- best_score={metrics.get('best_score')}，narrative_score={details.get('narrative_score')}",
                    f"- narrative phase order valid={details.get('narrative_phase_order_valid')}，component_sequence={details.get('actual_component_sequence')}",
                ]
            )
        if tasks:
            lines.append(
                f"- TASKS.md：完成 {tasks.get('done_count', 0)} 項，未完成 {tasks.get('pending_count', 0)} 項。"
            )
        done = [item["label"] for item in capabilities if item.get("state") == "done"]
        pending = [item["label"] for item in capabilities if item.get("state") != "done"]
        if done:
            lines.append("- 已接上的能力：" + "；".join(done[:6]))
        if pending:
            lines.append("- 明確限制/待辦：" + "；".join(pending[:6]))
        lines.append(
            "- 對話策略：你是 ASAL 生命體進度觀測 + Digital Clone 記憶 + Gemma 推理/語音介面的雛形。"
        )
        return "\n".join(lines)

    def resolve_artifact(self, run_id: str, asset: str) -> Path:
        if not re.match(r"^[A-Za-z0-9_.-]+$", run_id or ""):
            raise FileNotFoundError("invalid run id")
        if not asset or asset.startswith("/") or ".." in Path(asset).parts:
            raise FileNotFoundError("invalid artifact path")
        run_dir = self._find_run_dir(run_id)
        target = (run_dir / asset).resolve()
        run_dir_resolved = run_dir.resolve()
        if not _path_is_inside(target, run_dir_resolved) or not target.is_file():
            raise FileNotFoundError(asset)
        return target

    def _collect_runs(self) -> list[dict]:
        runs = []
        for root in self.run_roots:
            if not root.exists():
                continue
            for summary_path in root.glob("*/summary.json"):
                run = self._summarize_run(summary_path, root)
                if run:
                    runs.append(run)
        sorted_runs = sorted(
            runs,
            key=lambda item: (
                item.get("_sort_ts") or 0.0,
                item.get("run_id") or "",
            ),
            reverse=True,
        )
        for run in sorted_runs:
            run.pop("_sort_ts", None)
        return sorted_runs

    def _summarize_run(self, summary_path: Path, root: Path) -> dict | None:
        data = _read_json(summary_path)
        if not isinstance(data, dict):
            return None
        run_dir = summary_path.parent
        run_id = run_dir.name
        artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        if not metrics:
            metrics = {
                "best_score": data.get("best_score"),
                "num_frames": data.get("num_frames"),
                "search_iters": data.get("search_iters"),
                "search_pop": data.get("search_pop"),
                "search_keep": data.get("search_keep"),
            }
        if not details:
            details = {
                "prompt": data.get("prompt"),
                "best_theta": data.get("best_theta"),
                "foundation_model": data.get("foundation_model"),
                "substrate": data.get("substrate"),
                "narrative_enabled": data.get("narrative_enabled"),
                "narrative_score": data.get("narrative_score"),
                "narrative_phase_order_valid": data.get("narrative_phase_order_valid"),
            }
        self._merge_narrative_summary(run_dir, artifacts, details)
        asset_urls = {}
        for key in ("gif", "image", "mp4"):
            rel = artifacts.get(key)
            if rel:
                asset_urls[key] = self._artifact_url(run_id, rel)
        for key, rel in (details.get("narrative_keyframes") or {}).items():
            asset_urls[f"keyframe_{key}"] = self._artifact_url(run_id, rel)
        for phase in details.get("phase_scores") or []:
            name = phase.get("name")
            if name and asset_urls.get(f"keyframe_{name}"):
                phase["keyframe_url"] = asset_urls[f"keyframe_{name}"]

        sort_dt = _parse_dt(data.get("completed_at")) or _parse_dt(data.get("started_at"))
        if sort_dt is None:
            sort_dt = self._dt_from_run_id(run_id)

        return {
            "run_id": run_id,
            "source": str(root.relative_to(self.root)) if _path_is_inside(root, self.root) else str(root),
            "status": data.get("status", "completed" if artifacts or data.get("best_score") is not None else "unknown"),
            "mode": data.get("mode"),
            "config_path": data.get("config_path"),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
            "run_elapsed_seconds": data.get("run_elapsed_seconds"),
            "metrics": {
                "best_score": _float_or_none(metrics.get("best_score")),
                "num_frames": metrics.get("num_frames"),
                "search_iters": metrics.get("search_iters"),
                "search_pop": metrics.get("search_pop"),
                "search_keep": metrics.get("search_keep"),
            },
            "details": details,
            "artifacts": artifacts,
            "asset_urls": asset_urls,
            "timeline_label": self._timeline_label(data, run_id),
            "_sort_ts": sort_dt.timestamp() if sort_dt else 0.0,
        }

    def _merge_narrative_summary(self, run_dir: Path, artifacts: dict, details: dict) -> None:
        rel = artifacts.get("narrative_summary") or details.get("narrative_summary")
        if not rel:
            return
        payload = _read_json(run_dir / rel)
        if not isinstance(payload, dict):
            return
        details["narrative_score"] = payload.get("total_score", details.get("narrative_score"))
        details["narrative_phase_order_valid"] = payload.get(
            "phase_order_valid",
            details.get("narrative_phase_order_valid"),
        )
        details["actual_component_sequence"] = payload.get("actual_component_sequence")
        details["expected_component_sequence"] = payload.get("expected_component_sequence")
        phase_scores = []
        for phase in payload.get("phases") or []:
            phase_scores.append(
                {
                    "name": phase.get("name"),
                    "score": phase.get("score"),
                    "target_components": phase.get("target_components"),
                }
            )
        details["phase_scores"] = phase_scores

    def _capabilities(self, latest: dict | None) -> list[dict]:
        boids_text = _read_text(self.root / "research/asal_engine/substrates/boids.py")
        config_text = _read_text(self.root / "configs/asal/target_cell_fusion_narrative.yaml")
        latest_details = (latest or {}).get("details") or {}
        avatar_asset = self.root / "web/gemma_chat/avatar.jpg"
        return [
            {
                "key": "asal_logs",
                "label": "ASAL 舊工作日誌已匯入並可搜尋",
                "state": "done" if (self.root / "research/asal_engine/history/log").exists() else "pending",
            },
            {
                "key": "morphology",
                "label": "morphology analyzer / narrative scoring 已存在",
                "state": "done"
                if (self.root / "research/asal_engine/morphology.py").exists()
                and (self.root / "research/asal_engine/narrative_scores.py").exists()
                else "pending",
            },
            {
                "key": "phase_controller",
                "label": "boids phase-controlled dynamics prototype 已接入",
                "state": "done" if "narrative_controller" in boids_text and "split_push" in boids_text else "pending",
            },
            {
                "key": "fusion_config",
                "label": "cell fusion narrative config 已存在",
                "state": "done" if "birth" in config_text and "split" in config_text and "fusion" in config_text else "pending",
            },
            {
                "key": "latest_artifact",
                "label": "最新 ASAL 生命體 artifact 可供網頁呈現",
                "state": "done" if latest and (latest.get("asset_urls") or {}).get("gif") else "pending",
            },
            {
                "key": "phase_order",
                "label": "最新 run 有 1->2->1 phase order 檢查",
                "state": "done" if latest_details.get("narrative_phase_order_valid") is True else "pending",
            },
            {
                "key": "voice",
                "label": "Gemma web 語音輸入/播報路徑存在",
                "state": "done" if (self.root / "genai/web/stt.py").exists() else "pending",
            },
            {
                "key": "human_avatar",
                "label": "真人/人形 avatar 尚未修好，不納入本雛形成功條件",
                "state": "pending" if not avatar_asset.exists() else "done",
            },
        ]

    def _tasks(self) -> dict:
        path = self.root / "context/TASKS.md"
        lines = _read_text(path).splitlines()
        done = []
        pending = []
        for line in lines:
            match = re.match(r"- \[(x| )\]\s+(.+)", line)
            if not match:
                continue
            target = done if match.group(1) == "x" else pending
            target.append(match.group(2).strip())
        return {
            "done_count": len(done),
            "pending_count": len(pending),
            "done": done,
            "pending": pending,
        }

    def _latest_worklogs(self) -> list[dict]:
        entries = []
        for root in self.history_roots:
            if not root.exists():
                continue
            for path in root.glob("**/*.md"):
                if path.name == "READ_WORK_LOG.md":
                    continue
                title = ""
                for line in _read_text(path).splitlines():
                    if line.startswith("# "):
                        title = line.lstrip("#").strip()
                        break
                try:
                    rel = str(path.relative_to(self.root))
                except ValueError:
                    rel = str(path)
                entries.append({"path": rel, "title": title or path.stem})
        return sorted(entries, key=lambda item: item["path"], reverse=True)[:6]

    def _summary(self, latest: dict | None, capabilities: list[dict], tasks: dict) -> str:
        done_count = sum(1 for item in capabilities if item.get("state") == "done")
        total_count = len(capabilities)
        if not latest:
            return f"ASAL records exist, but no local run summary was found. Capabilities {done_count}/{total_count}."
        metrics = latest.get("metrics") or {}
        details = latest.get("details") or {}
        score = metrics.get("best_score")
        narrative_score = details.get("narrative_score")
        phase_valid = details.get("narrative_phase_order_valid")
        task_text = f"TASKS done={tasks.get('done_count', 0)}, pending={tasks.get('pending_count', 0)}"
        return (
            f"latest run {latest.get('run_id')} completed with best_score={score}, "
            f"narrative_score={narrative_score}, phase_order_valid={phase_valid}. "
            f"Capabilities {done_count}/{total_count}; {task_text}."
        )

    def _find_run_dir(self, run_id: str) -> Path:
        candidates = []
        for root in self.run_roots:
            candidate = root / run_id
            if candidate.exists():
                candidates.append(candidate)
        if not candidates:
            raise FileNotFoundError(run_id)
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    def _artifact_url(self, run_id: str, asset: str) -> str:
        return f"/artifacts/asal/{quote(run_id)}/{quote(str(asset))}"

    def _dt_from_run_id(self, run_id: str) -> datetime | None:
        match = re.match(r"^(\d{8})-(\d{6})$", run_id)
        if not match:
            return None
        try:
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def _timeline_label(self, data: dict, run_id: str) -> str:
        dt = _parse_dt(data.get("completed_at")) or _parse_dt(data.get("started_at")) or self._dt_from_run_id(run_id)
        if dt:
            return dt.strftime("%m/%d %H:%M")
        return run_id
