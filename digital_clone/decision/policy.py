from __future__ import annotations


class ClonePromptBuilder:
    def build(self, persona, memories, user_text, extra_context=None):
        system = (
            f"You are {persona.name}. "
            f"Tone: {persona.tone}. "
            f"Principles: {'; '.join(persona.principles)}."
        )
        if getattr(persona, "goals", None):
            system += f" Goals: {'; '.join(persona.goals)}."

        memory_lines = [m["content"] for m in memories if m.get("content")]
        if extra_context:
            memory_lines.append(extra_context)
        context = "\n".join(memory_lines).strip() if memory_lines else None
        return {
            "system": system,
            "context": context,
            "prompt": user_text,
        }
