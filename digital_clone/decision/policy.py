class DecisionPolicy:
    def select(self, persona, memory, user_text):
        return {
            "tone": persona.tone,
            "principles": persona.principles[:2],
            "reply": f"我已收到你的問題：{user_text}"
        }
