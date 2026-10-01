"""LLM-assisted emotion classification and empathy judging with the real local model (Plan 37 R2 E-track).

classify_emotion_llm: zero-shot label in LABELS, parsed strictly; None if unparseable.
judge_empathy_llm: rubric judge (0-2 per criterion) for a reply to an emotional user message.
Both use the same llama.cpp Gemma adapter as the product; no mock path.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from genai.llm.adapter import LLMRequest
from genai.web.emotion import LABELS

_ZH = {"joy": "開心", "sadness": "悲傷", "anger": "憤怒", "fear": "恐懼/擔心", "surprise": "驚訝", "neutral": "平淡"}
_ALIASES = {
    "joy": ["joy", "happy", "開心", "快樂", "高興"], "sadness": ["sadness", "sad", "悲傷", "難過", "傷心"],
    "anger": ["anger", "angry", "憤怒", "生氣"], "fear": ["fear", "afraid", "恐懼", "擔心", "害怕", "焦慮"],
    "surprise": ["surprise", "surprised", "驚訝", "驚奇"], "neutral": ["neutral", "平淡", "中性", "無情緒"],
}

CLASSIFY_SYSTEM = ("You are an emotion classifier for Traditional Chinese and English chat messages. "
                   "Read the user's message, including sarcasm and implied feelings, and output exactly one "
                   "English label from: joy, sadness, anger, fear, surprise, neutral. Output only the label.")


def parse_label(text: str) -> Optional[str]:
    low = (text or "").strip().lower()
    hits = []
    for label, words in _ALIASES.items():
        for w in words:
            i = low.find(w)
            if i >= 0:
                hits.append((i, label))
    return min(hits)[1] if hits else None


def classify_emotion_llm(adapter, text: str, max_tokens: int = 8) -> Dict[str, object]:
    labels = ", ".join(f"{l} ({_ZH[l]})" for l in LABELS)
    prompt = f"Labels: {labels}\nMessage: 「{text}」\nLabel:"
    out = adapter.generate(LLMRequest(prompt=prompt, system=CLASSIFY_SYSTEM, max_tokens=max_tokens, temperature=0.0))
    return {"label": parse_label(out.text), "raw": (out.text or "")[:80]}


EMPATHY_SYSTEM = ("你是嚴格的對話品質評審。根據評分準則給分，只輸出三個以空白分隔的整數（每項 0、1 或 2），不要解釋。")
EMPATHY_RUBRIC = (
    "評分準則：\n"
    "A 情緒辨識：回覆是否正確辨識並說出使用者的情緒（0=沒有或錯誤，1=模糊，2=明確且正確）\n"
    "B 情緒回應：回覆是否先回應感受、語氣恰當（0=冷漠或說教，1=部分，2=溫和且恰當）\n"
    "C 有幫助：回覆是否提供具體支持或下一步，且不輕視問題（0=沒有，1=泛泛，2=具體）\n")


def judge_empathy_llm(adapter, user_text: str, reply: str) -> Dict[str, object]:
    prompt = f"{EMPATHY_RUBRIC}\n使用者：「{user_text}」\n回覆：「{reply}」\n請輸出 A B C 三個分數："
    out = adapter.generate(LLMRequest(prompt=prompt, system=EMPATHY_SYSTEM, max_tokens=12, temperature=0.0))
    nums = [int(x) for x in re.findall(r"(?<!\d)[012](?!\d)", out.text or "")][:3]
    if len(nums) < 3:
        return {"scores": None, "total": None, "raw": (out.text or "")[:80]}
    return {"scores": dict(zip("ABC", nums)), "total": sum(nums), "raw": (out.text or "")[:80]}


def detect_emotion(adapter, text: str, mode: str = "hybrid", threshold: float = 0.7) -> Dict[str, object]:
    """Product-path detector. mode: lexicon | llm | hybrid.

    hybrid: keep the lexicon result when it is non-neutral and confident (>= threshold), otherwise ask the
    real model. Independent zh-TW public set (runs/plan37/emotion_r2): lexicon acc 0.45, llm 0.83, hybrid 0.825.
    Returns a dict compatible with detect_text_emotion (label/probs/valence/arousal/confidence/source).
    """
    from genai.web.emotion import ANCHORS, detect_text_emotion

    lex = detect_text_emotion(text)
    if mode == "lexicon" or adapter is None:
        return lex
    if mode == "hybrid" and lex["label"] != "neutral" and float(lex["confidence"]) >= threshold:
        return {**lex, "source": "text_lexicon_v1 (hybrid: confident)"}
    try:
        r = classify_emotion_llm(adapter, text)
    except Exception as exc:  # model failure: fall back, but say so
        return {**lex, "source": f"text_lexicon_v1 (llm failed: {type(exc).__name__})"}
    label = r["label"]
    if not label:
        return {**lex, "source": "text_lexicon_v1 (llm unparsed)", "llm_raw": r["raw"]}
    probs = {l: (0.85 if l == label else 0.03) for l in LABELS}
    v, a = ANCHORS[label]
    return {**lex, "label": label, "probs": probs, "valence": round(v, 3), "arousal": round(a, 3),
            "confidence": 0.8, "source": f"llm_gemma ({mode})", "lexicon_label": lex["label"], "llm_raw": r["raw"]}
