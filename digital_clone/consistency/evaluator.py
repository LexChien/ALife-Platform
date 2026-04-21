class ConsistencyEvaluator:
    def score(self, persona, text, user_text="", retrieved_memories=None):
        retrieved_memories = retrieved_memories or []
        checks = [
            text.startswith(f"[{persona.name}]"),
            f"tone={persona.tone}" in text,
            all(p in text for p in persona.principles[:2]),
            user_text in text if user_text else True,
        ]
        if retrieved_memories:
            checks.append(any(memory in text for memory in retrieved_memories))
        if persona.facts and any(key in user_text for key in ["記憶", "memory", "風格", "特徵"]):
            checks.append(
                any(fact in text for fact in persona.facts[:2])
                or persona.tone in text
                or any(memory in text for memory in retrieved_memories)
            )
        return sum(1.0 for ok in checks if ok) / max(len(checks), 1)
