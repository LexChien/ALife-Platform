"""Plan 38 per-turn language lock (E-LANG variant D: lock line at the END of the system prompt + verify once)."""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LAT = re.compile(r"[A-Za-z]")

LOCK = {"zh": "【本回合語言】Lex 這句用的是中文。你必須只用繁體中文回答（專有名詞可保留英文）。",
        "en": "[Language for this turn] Lex wrote in English. Reply in English only."}
RETRY_SUFFIX = {"zh": "（上一個草稿語言錯誤，請只用繁體中文重寫。）", "en": "(The previous draft used the wrong language; rewrite in English.)"}


def detect_lang(text: str) -> str:
    s = text or ""
    cjk = len(_CJK.findall(s))
    lat = len(_LAT.findall(s))
    return "zh" if cjk >= max(2, 0.3 * (cjk + lat / 4)) else ("en" if lat else "zh")


def lock_line(lang: str) -> str:
    return LOCK.get(lang, LOCK["zh"])


def matches(reply: str, lang: str) -> bool:
    return detect_lang(reply) == lang
