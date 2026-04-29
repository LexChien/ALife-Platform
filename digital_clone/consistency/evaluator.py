from typing import Any, Dict
from evaluation.heuristics import contains_all, contains_any, make_result


class ConsistencyEvaluator:
    def score(self, persona, text, user_text="", retrieved_memories=None, prompt_components=None):
        retrieved_memories = retrieved_memories or []
        prompt_components = prompt_components or {}
        context = prompt_components.get("context")
        memory_lines = context.splitlines() if context else []
        ask_memory_sensitive = any(key in user_text for key in ["記憶", "memory", "風格", "特徵"])

        checks = {
            "persona_tag": text.startswith(f"[{persona.name}]"),
            "tone_match": f"tone={persona.tone}" in text,
            "principle_presence": contains_all(text, persona.principles[:2]),
            "user_echo": user_text in text if user_text else True,
            "prompt_context_hit": contains_any(text, memory_lines) if memory_lines else True,
            "retrieval_grounding": contains_any(text, retrieved_memories) if retrieved_memories else True,
            "memory_sensitive_answer": (
                contains_any(text, persona.facts[:2])
                or persona.tone in text
                or contains_any(text, retrieved_memories)
            ) if persona.facts and ask_memory_sensitive else True,
        }
        return make_result(checks)

    def evaluate_life_drift(
        self,
        persona: Dict[str, Any],
        current_life_phase: str,
        life_likeness: float,
    ) -> Dict[str, Any]:
        """
        新增：計算 life_drift
        若 persona 與當前 ASAL 生命階段不符 → drift 上升
        """
        expected_tone = {
            "birth": "calm, nurturing",
            "split": "analytical, exploratory",
            "fusion": "integrated, reflective",
        }.get(current_life_phase, "neutral")

        actual_tone = persona.get("tone", "")
        tone_match = 1.0 if expected_tone in actual_tone else 0.6

        drift = (1.0 - tone_match) * 0.5 + (abs(life_likeness - 0.5) * 0.5)
        return {
            "life_drift": round(float(drift), 4),
            "tone_match": float(tone_match),
            "expected_tone": expected_tone,
        }
