class DecisionPolicy:
    def select(self, persona, memory, user_text):
        retrieved = memory.retrieve(user_text, n=2, kinds=["dialogue"])
        profile_notes = memory.retrieve(user_text, n=2, kinds=["profile_fact"])
        memory_notes = []
        for item in profile_notes + retrieved:
            if item["content"] == user_text:
                continue
            if item["content"] not in memory_notes:
                memory_notes.append(item["content"])

        reply_parts = [f"我已收到你的問題：{user_text}"]
        lowered = user_text.lower()

        if any(key in user_text for key in ["風格", "特徵", "identity", "persona"]):
            reply_parts.append(f"我會維持 {persona.tone} 的風格，並遵守 {', '.join(persona.principles[:2])}")

        if any(key in user_text for key in ["記憶", "memory", "記住", "之前"]):
            if memory_notes:
                reply_parts.append(f"我目前可用的相關記憶是：{' | '.join(memory_notes)}")
            elif persona.facts:
                reply_parts.append(f"我目前固定的人格記憶是：{' | '.join(persona.facts[:2])}")

        if "一句話" in user_text:
            reply = reply_parts[0]
        else:
            reply = "；".join(reply_parts)

        return {
            "tone": persona.tone,
            "principles": persona.principles[:2],
            "reply": reply,
            "retrieved_memories": memory_notes,
        }
