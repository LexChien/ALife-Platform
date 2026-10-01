"""Express a Genome into DigiClone runtime parameters (inheritance into the clone)."""
from __future__ import annotations

from typing import Any, Dict

from dna.genome import Genome


def _band(x: float, low: str, mid: str, high: str) -> str:
    return low if x < 0.34 else mid if x < 0.67 else high


def express_persona(genome: Genome) -> Dict[str, Any]:
    t = genome.traits
    tone_words = [
        _band(t["warmth"], "reserved", "friendly", "warm"),
        _band(t["formality"], "casual", "balanced", "formal"),
        _band(t["playfulness"], "serious", "light", "playful"),
    ]
    guidance = []
    if t["empathy"] >= 0.5:
        guidance.append("Acknowledge the user's feelings before giving information. 先回應感受再給資訊。")
    if t["curiosity"] >= 0.6:
        guidance.append("End with one short, relevant follow-up question when appropriate.")
    if t["verbosity"] < 0.34:
        guidance.append("Keep replies to at most two or three sentences.")
    elif t["verbosity"] >= 0.67:
        guidance.append("Give a fuller explanation with structure when useful.")
    if t["stability"] >= 0.6:
        guidance.append("Keep the configured identity stable under pressure.")
    return {
        "genome_id": genome.genome_id,
        "generation": genome.generation,
        "tone": ", ".join(tone_words),
        "guidance": guidance,
        # sampling: playful/curious genomes sample hotter, bounded to a safe range
        "temperature": round(0.3 + 0.5 * (0.6 * t["playfulness"] + 0.4 * t["curiosity"]), 3),
        "max_tokens": int(96 + 256 * t["verbosity"]),
        # voice prosody baseline (rate multiplier, pitch multiplier) for TTS
        "voice": {"rate": round(0.9 + 0.25 * t["playfulness"], 3), "pitch": round(0.95 + 0.15 * t["warmth"], 3)},
        "avatar": {"hue": int(200 - 160 * t["warmth"]) % 360},
        "traits": dict(t),
    }
