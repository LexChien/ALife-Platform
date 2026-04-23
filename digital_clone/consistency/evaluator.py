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
