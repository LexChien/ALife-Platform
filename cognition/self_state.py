"""Plan 38 J4: persisted self-state (runs/digiclone/self_state.json): mood distribution, last appraisal, recent
thought ids (for HUD traceability, C5), turn counter. Written atomically after every turn."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cognition.appraisal import appraise, dominant, update_mood

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "runs" / "digiclone" / "self_state.json"


class SelfState:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.data = {"schema": "digiclone.self_state.v1", "turns": 0, "mood": None, "feeling": "calm",
                     "last_user_emotion": "neutral", "last_intent": "answer", "last_brief": "", "last_summary": "",
                     "recent_thought_ids": [], "updated_at": None}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:
                pass
        if not self.data.get("mood"):
            self.data["mood"] = update_mood(None, "calm")

    @property
    def feeling(self) -> str:
        return self.data.get("feeling", "calm")

    def apply_thought(self, th: dict) -> dict:
        feeling = appraise(th.get("user_emotion"))
        self.data["mood"] = update_mood(self.data.get("mood"), feeling)
        self.data.update({"turns": int(self.data.get("turns", 0)) + 1, "feeling": feeling,
                          "dominant_mood": dominant(self.data["mood"]), "last_user_emotion": th.get("user_emotion"),
                          "last_intent": th.get("intent"), "last_brief": (th.get("speech_brief") or "")[:200],
                          "last_summary": th.get("summary") or "", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        ids = (self.data.get("recent_thought_ids") or []) + [th.get("id")]
        self.data["recent_thought_ids"] = ids[-50:]
        self.save()
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def public(self) -> dict:
        """What the HUD may show (no briefs, no notes)."""
        return {k: self.data.get(k) for k in ("turns", "feeling", "dominant_mood", "mood", "last_user_emotion", "updated_at")}
