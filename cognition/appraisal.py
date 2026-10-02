"""Plan 38 J4: deterministic appraisal (her feeling from the user's state; not LLM-chosen, which collapsed to
'amused' in E-COG v1) + a mood distribution with decay where no single mood may exceed 60 % (spec J4)."""
from __future__ import annotations

from cognition.thought import FEELINGS

APPRAISAL = {"sadness": "concerned", "tired": "warm", "fear": "concerned", "anger": "calm", "joy": "warm",
             "curious": "focused", "neutral": "calm"}
ZH_TONE = {"calm": "平靜", "warm": "溫暖", "concerned": "關切", "focused": "專注", "amused": "輕鬆幽默", "curious": "好奇"}
EN_TONE = {"calm": "calm", "warm": "warm", "concerned": "gently concerned", "focused": "focused", "amused": "lightly amused",
           "curious": "curious"}
MAX_SHARE = 0.60
DECAY = 0.7


def appraise(user_emotion: str | None) -> str:
    return APPRAISAL.get(user_emotion or "neutral", "calm")


def cap_distribution(mood: dict[str, float], max_share: float = MAX_SHARE) -> dict[str, float]:
    total = sum(mood.values()) or 1.0
    m = {k: v / total for k, v in mood.items()}
    for _ in range(10):
        over = {k: v for k, v in m.items() if v > max_share + 1e-9}
        if not over:
            break
        excess = sum(v - max_share for v in over.values())
        rest = [k for k in m if k not in over]
        for k in over:
            m[k] = max_share
        rest_total = sum(m[k] for k in rest)
        for k in rest:
            m[k] += excess * (m[k] / rest_total if rest_total > 0 else 1 / len(rest))
    return {k: round(v, 4) for k, v in m.items()}


def update_mood(mood: dict[str, float] | None, feeling: str) -> dict[str, float]:
    base = {f: 1.0 / len(FEELINGS) for f in FEELINGS}
    m = {f: (mood or base).get(f, 0.0) * DECAY for f in FEELINGS}
    m[feeling if feeling in m else "calm"] += 1 - DECAY
    return cap_distribution(m)


def dominant(mood: dict[str, float]) -> str:
    return max(mood.items(), key=lambda kv: kv[1])[0] if mood else "calm"


def tone_line(feeling: str, lang: str) -> str:
    return f"語氣：{ZH_TONE.get(feeling, '平靜')}。" if lang == "zh" else f"Tone: {EN_TONE.get(feeling, 'calm')}."
