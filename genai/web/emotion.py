"""Plan 37 G4: emotion detection (text + voice prosody), emotion state, reply modulation.

Text detection is a deterministic zh-TW/EN lexicon model (no network, no model
download). Prosody features are computed from raw PCM with numpy. This is a
companion-style affect signal, not a clinical assessment.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LABELS = ("joy", "sadness", "anger", "fear", "surprise", "neutral")
# Russell circumplex anchors (valence, arousal) in [-1, 1].
ANCHORS: Dict[str, tuple] = {
    "joy": (0.8, 0.5), "sadness": (-0.7, -0.4), "anger": (-0.7, 0.7),
    "fear": (-0.6, 0.6), "surprise": (0.2, 0.8), "neutral": (0.0, 0.0),
}
LEXICON: Dict[str, List[str]] = {
    "joy": ["開心", "高興", "快樂", "太好了", "好棒", "興奮", "期待", "感謝", "謝謝", "喜歡", "幸福", "滿意", "順利", "成功", "哈哈",
            "happy", "glad", "great", "awesome", "excited", "love", "thanks", "thank you", "wonderful", "yay"],
    "sadness": ["難過", "傷心", "沮喪", "失落", "哭", "心痛", "孤單", "寂寞", "失望", "低落", "憂鬱", "想哭", "被罵", "遺憾", "累了", "好累", "絕望",
                "sad", "depressed", "lonely", "upset", "cry", "heartbroken", "disappointed", "down", "miserable", "tired"],
    "anger": ["生氣", "氣死", "憤怒", "火大", "討厭", "煩死", "不爽", "受夠", "可惡", "爛透", "氣炸",
              "angry", "furious", "mad", "hate", "annoyed", "pissed", "fed up", "ridiculous"],
    "fear": ["害怕", "擔心", "焦慮", "緊張", "恐懼", "不安", "壓力", "怕", "慌", "睡不著",
             "afraid", "scared", "worried", "anxious", "nervous", "panic", "stress", "stressed", "fear"],
    "surprise": ["驚訝", "沒想到", "竟然", "居然", "嚇一跳", "真的假的", "天啊", "哇",
                 "surprised", "wow", "unexpected", "no way", "omg", "shocked"],
}
INTENSIFIERS = ["很", "非常", "超", "好", "太", "真的", "極", "特別", "so ", "very ", "really ", "extremely "]
NEGATIONS = ["不", "沒", "沒有", "別", "不會", "not ", "no ", "never ", "don't ", "isn't "]


def _contains_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 4):start]
    return any(window.endswith(n) or n in window[-3:] for n in NEGATIONS)


def detect_text_emotion(text: str) -> Dict[str, object]:
    raw = text or ""
    low = raw.lower()
    scores = {label: 0.0 for label in LABELS}
    cues: List[str] = []
    for label, words in LEXICON.items():
        for word in words:
            for match in re.finditer(re.escape(word), low):
                if _contains_negated(low, match.start()):
                    cues.append(f"negated:{word}")
                    continue
                weight = 1.0
                pre = low[max(0, match.start() - 3):match.start()]
                if any(i.strip() and i.strip() in pre for i in INTENSIFIERS):
                    weight = 1.5
                scores[label] += weight
                cues.append(word)
    exclaim = raw.count("!") + raw.count("！")
    question = raw.count("?") + raw.count("？")
    total = sum(scores.values())
    if total == 0:
        scores["neutral"] = 1.0
        label, confidence = "neutral", 0.5
    else:
        label = max((l for l in LABELS if l != "neutral"), key=lambda l: scores[l])
        confidence = round(min(0.95, 0.5 + 0.5 * scores[label] / (total + 1.0)), 3)
    norm = sum(scores.values()) or 1.0
    probs = {l: round(scores[l] / norm, 3) for l in LABELS}
    valence = sum(probs[l] * ANCHORS[l][0] for l in LABELS)
    arousal = sum(probs[l] * ANCHORS[l][1] for l in LABELS) + min(0.3, 0.1 * exclaim)
    return {
        "label": label, "confidence": confidence, "probs": probs,
        "valence": round(max(-1.0, min(1.0, valence)), 3), "arousal": round(max(-1.0, min(1.0, arousal)), 3),
        "cues": cues[:12], "exclamations": exclaim, "questions": question, "source": "text_lexicon_v1",
    }


def prosody_features(samples, sample_rate: int = 16000) -> Dict[str, float]:
    """RMS energy, autocorrelation pitch (70-400 Hz) and voiced ratio from mono float PCM in [-1, 1]."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).flatten()
    duration = len(x) / float(sample_rate) if sample_rate else 0.0
    if len(x) < sample_rate // 10:
        return {"duration_s": round(duration, 3), "rms_db": -120.0, "pitch_hz_mean": 0.0, "pitch_hz_std": 0.0, "voiced_ratio": 0.0}
    frame = int(0.04 * sample_rate)
    hop = frame // 2
    rms_list, pitches = [], []
    lo, hi = int(sample_rate / 400), int(sample_rate / 70)
    for start in range(0, len(x) - frame, hop):
        f = x[start:start + frame]
        rms = float(np.sqrt(np.mean(f * f)) + 1e-12)
        rms_list.append(rms)
        if rms < 0.01:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, mode="full")[frame - 1:]
        if ac[0] <= 0 or hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.3:
            pitches.append(sample_rate / lag)
    rms_arr = np.array(rms_list)
    voiced = len(pitches) / max(1, len(rms_list))
    return {
        "duration_s": round(duration, 3),
        "rms_db": round(float(20 * np.log10(np.mean(rms_arr) + 1e-12)), 2),
        "pitch_hz_mean": round(float(np.mean(pitches)) if pitches else 0.0, 1),
        "pitch_hz_std": round(float(np.std(pitches)) if pitches else 0.0, 1),
        "voiced_ratio": round(float(voiced), 3),
    }


def arousal_from_prosody(features: Dict[str, float]) -> float:
    """Map loudness and pitch variability to arousal in [-1, 1] (heuristic, uncalibrated)."""
    if not features or features.get("voiced_ratio", 0) <= 0:
        return 0.0
    loud = (features["rms_db"] + 30.0) / 15.0          # ~-45 dB -> -1, ~-15 dB -> +1
    var = (features["pitch_hz_std"] - 25.0) / 35.0       # flat -> negative, lively -> positive
    return round(max(-1.0, min(1.0, 0.6 * loud + 0.4 * var)), 3)


@dataclass
class EmotionState:
    """Per-session affect state with exponential decay toward neutral."""

    valence: float = 0.0
    arousal: float = 0.0
    label: str = "neutral"
    intensity: float = 0.0
    turns: int = 0
    history: List[Dict[str, object]] = field(default_factory=list)

    def update(self, observation: Dict[str, object], voice_arousal: Optional[float] = None, decay: float = 0.5) -> "EmotionState":
        obs_v = float(observation.get("valence", 0.0))
        obs_a = float(observation.get("arousal", 0.0))
        if voice_arousal is not None:
            obs_a = 0.6 * obs_a + 0.4 * float(voice_arousal)
        conf = float(observation.get("confidence", 0.5))
        if observation.get("label") == "neutral" and voice_arousal is None:
            # no affect evidence this turn: decay toward neutral
            keep = 0.5 + 0.5 * decay
            self.valence = round(self.valence * keep, 3)
            self.arousal = round(self.arousal * keep, 3)
        else:
            w = max(0.0, min(1.0, (1.0 - decay) * (0.5 + conf)))
            self.valence = round((1 - w) * self.valence + w * obs_v, 3)
            self.arousal = round((1 - w) * self.arousal + w * obs_a, 3)
        self.intensity = round(min(1.0, math.hypot(self.valence, self.arousal)), 3)
        self.label = self._nearest_label() if self.intensity >= 0.2 else "neutral"
        self.turns += 1
        self.history.append({"turn": self.turns, "observed": observation.get("label"), "valence": self.valence,
                             "arousal": self.arousal, "label": self.label, "voice_arousal": voice_arousal})
        self.history = self.history[-20:]
        return self

    def _nearest_label(self) -> str:
        return min((l for l in LABELS if l != "neutral"),
                   key=lambda l: (ANCHORS[l][0] - self.valence) ** 2 + (ANCHORS[l][1] - self.arousal) ** 2)

    def to_dict(self) -> Dict[str, object]:
        return {"label": self.label, "valence": self.valence, "arousal": self.arousal,
                "intensity": self.intensity, "turns": self.turns, "history": list(self.history[-5:])}


GUIDANCE = {
    "sadness": "使用者目前情緒低落。先用一兩句真誠地接住感受（不說教、不急著給建議），再溫和地詢問是否想多聊。The user seems sad: validate feelings first, no lecturing.",
    "anger": "使用者目前很生氣。先承認其不滿合理之處，語氣冷靜不防衛，不爭辯，再一起看可以怎麼處理。The user is angry: acknowledge, stay calm, don't argue.",
    "fear": "使用者目前焦慮或擔心。用穩定、放慢的語氣回應，提供安全感與一個具體的小步驟。The user is anxious: be steady, offer one small concrete step.",
    "joy": "使用者心情很好。自然地分享喜悅、回應其好消息。The user is happy: share the joy naturally.",
    "surprise": "使用者感到驚訝。先回應這份意外感，再釐清狀況。The user is surprised: acknowledge, then clarify.",
}
AVATAR_EXPRESSION = {"sadness": "concerned", "anger": "calm", "fear": "reassuring", "joy": "smile", "surprise": "attentive", "neutral": "neutral"}


def modulation(state: EmotionState, base_voice: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    base = base_voice or {"rate": 1.0, "pitch": 1.0}
    label = state.label
    rate, pitch = float(base.get("rate", 1.0)), float(base.get("pitch", 1.0))
    if label in ("sadness", "fear"):
        rate *= 0.88
        pitch *= 0.96
    elif label == "anger":
        rate *= 0.92
    elif label in ("joy", "surprise"):
        rate *= 1.06
        pitch *= 1.05
    guidance = GUIDANCE.get(label, "")
    if guidance and state.intensity >= 0.5:
        guidance += " 情緒強度高，回覆要更短、更溫和。"
    return {
        "label": label,
        "system_guidance": guidance,
        "tts": {"rate": round(rate, 3), "pitch": round(pitch, 3)},
        "avatar": {"expression": AVATAR_EXPRESSION.get(label, "neutral"), "intensity": state.intensity},
    }
