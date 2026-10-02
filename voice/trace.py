"""Plan 38 J2.6: per-turn latency trace (all times relative to the user's speech end / text submit)."""
from __future__ import annotations

import json
import time
from pathlib import Path


class TurnTrace:
    MARKS = ("speech_end", "stt_done", "ack_audio", "llm_first_token", "first_sentence", "first_audio_ready",
             "first_audio_played", "llm_done", "turn_done", "barge_in", "audio_stopped")

    def __init__(self, turn_id: str, t0: float | None = None, source: str = "voice"):
        self.turn_id = turn_id
        self.t0 = t0 if t0 is not None else time.time()
        self.source = source
        self.marks: dict[str, float] = {}
        self.extra: dict = {}

    def mark(self, name: str, t: float | None = None) -> float:
        dt = round((t if t is not None else time.time()) - self.t0, 4)
        return self.marks.setdefault(name, dt)

    def to_dict(self) -> dict:
        return {"turn_id": self.turn_id, "source": self.source, "t0": round(self.t0, 4), "marks": dict(self.marks),
                **self.extra}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
