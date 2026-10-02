"""Plan 38 (J6 preview): deterministic proactive-speech salience gate (promoted from tools/plan38_cognitive_loop_v2.py;
16 events with labels fixed BEFORE running: gate 15/16 vs LLM few-shot 11/16)."""
from __future__ import annotations

import math

THRESH = 0.2  # a priori, not tuned on the labels


def salience(e: dict) -> float:
    urg = 0.0 if e.get("deadline_min") is None else math.exp(-e["deadline_min"] / 30.0)
    s = (0.45 * urg + 0.35 * e.get("goal", 0) * e.get("novelty", 0) + 0.9 * e.get("risk", 0) - 0.25 * e.get("focus", 0)
         - (0.3 if e.get("dnd") else 0) - (0.2 if e.get("quiet") else 0))
    if e.get("risk", 0) >= 0.9:
        s = max(s, 1.0)  # safety override
    return round(s, 3)


def should_speak(e: dict, thresh: float = THRESH) -> bool:
    return salience(e) >= thresh


def eval_events() -> dict:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from plan38_cognitive_loop_v2 import EVENTS  # labelled a priori in the Plan 38 experiment
    rows = [{"event": e["text"], "label": lab, "salience": salience(e), "speak": should_speak(e)} for e, lab in EVENTS]
    acc = sum(r["speak"] == r["label"] for r in rows) / len(rows)
    return {"kind": "DETERMINISTIC_GATE_ON_LABELLED_EVENTS", "n": len(rows), "accuracy": round(acc, 3), "rows": rows}
