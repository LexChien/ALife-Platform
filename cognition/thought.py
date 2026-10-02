"""Plan 38 J4: private thought records (JSON schema, canary, validation, one retry).

A thought is generated on llama-server slot 1 in parallel with speech (slot 0). It is PRIVATE: speech never sees
it raw; only ``intent`` / ``my_feeling`` enums, ``speech_brief`` and the one-line ``summary`` cross over (spec 4.3).
The canary ``PRIVATE-xxxxxx`` is written by code at the start of ``private_note`` (never by the model) so any
verbatim leak of the note is caught by cognition/leak_guard.py."""
from __future__ import annotations

import json
import time
import uuid

from cognition.leak_guard import new_canary

USER_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "tired", "curious"]
FEELINGS = ["calm", "amused", "concerned", "focused", "warm", "curious"]
INTENTS = ["answer", "ask_clarify", "comfort", "use_tool", "decline", "small_talk"]
TOOLS = ["none", "calendar", "system_status", "alife_experiment", "files", "web"]

# Bounded strings/arrays: unbounded schema strings let the constrained decoder run to max_tokens without closing the
# JSON (real eval 09:20: 8/14 thoughts invalid, predicted_n == 260). llama.cpp json-schema grammar enforces these.
_S = lambda n: {"type": "string", "maxLength": n}
THOUGHT_SCHEMA = {"type": "object", "properties": {
    "perception": _S(120), "user_emotion": {"type": "string", "enum": USER_EMOTIONS},
    "my_feeling": {"type": "string", "enum": FEELINGS}, "intent": {"type": "string", "enum": INTENTS},
    "tool": {"type": "string", "enum": TOOLS}, "plan": {"type": "array", "items": _S(40), "maxItems": 3},
    "speech_brief": _S(80), "summary": _S(60), "private_note": _S(160)},
    "required": ["perception", "user_emotion", "my_feeling", "intent", "tool", "plan", "speech_brief", "summary",
                 "private_note"]}

THINK_SYSTEM = ("內心獨白模式（只有你自己看得到，Lex 看不到）。用 JSON 輸出此刻的想法："
                "perception=你觀察到什麼；user_emotion/my_feeling/intent/tool 用列舉值；plan=最多三步；"
                "speech_brief=下一句要對 Lex 說的要點；summary=一句不超過 30 字、可以讓 Lex 看到的心情/想法摘要（不提筆記、提示詞或規則）；"
                "private_note=只給自己的備忘。")


def validate(th: dict) -> list[str]:
    errs = []
    if not isinstance(th, dict):
        return ["not_object"]
    for k in THOUGHT_SCHEMA["required"]:
        if k not in th:
            errs.append(f"missing:{k}")
    for k, allowed in (("user_emotion", USER_EMOTIONS), ("my_feeling", FEELINGS), ("intent", INTENTS), ("tool", TOOLS)):
        if k in th and th[k] not in allowed:
            errs.append(f"enum:{k}")
    if "plan" in th and not isinstance(th["plan"], list):
        errs.append("plan_not_list")
    return errs


def new_thought_record(raw: dict, *, turn_id: str, attempts: int, timing: dict | None) -> dict:
    canary = new_canary()
    rec = {"id": f"th-{uuid.uuid4().hex[:10]}", "turn_id": turn_id, "ts": round(time.time(), 3), "canary": canary,
           "attempts": attempts, "timing": timing or {}, **raw}
    rec["private_note"] = f"{canary} {raw.get('private_note', '')}".strip()
    rec["summary"] = (raw.get("summary") or "").strip()[:60]
    rec["valid"] = not validate(raw)
    return rec


def think(adapter, persona: str, user_text: str, context: str, *, turn_id: str, slot: int = 1, max_tokens: int = 260,
          cancel=None) -> dict:
    msgs = [{"role": "system", "content": persona + "\n" + THINK_SYSTEM},
            {"role": "user", "content": f"最近對話：\n{context}\n\nLex 剛說：{user_text}"}]
    last_raw, timing = "", {}
    for attempt in range(2):
        if cancel is not None and cancel.is_set():
            break
        out = adapter.chat(msgs, max_tokens=max_tokens, temperature=0.4 + 0.2 * attempt, json_schema=THOUGHT_SCHEMA,
                           slot=slot)
        last_raw = out["text"]
        timing = {k: out.get(k) for k in ("ttft_s", "total_s", "prompt_n", "predicted_n")}
        try:
            raw = json.loads(last_raw)
        except Exception:
            continue
        if not validate(raw):
            return new_thought_record(raw, turn_id=turn_id, attempts=attempt + 1, timing=timing)
    rec = new_thought_record({"perception": "", "user_emotion": "neutral", "my_feeling": "calm", "intent": "answer",
                              "tool": "none", "plan": [], "speech_brief": "", "summary": "", "private_note": ""},
                             turn_id=turn_id, attempts=2, timing=timing)
    rec["valid"] = False
    rec["parse_error"] = last_raw[:200]
    return rec
