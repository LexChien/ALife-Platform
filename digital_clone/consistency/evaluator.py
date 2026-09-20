"""Observable response checks; no claim of general persona or tone understanding."""
from evaluation.heuristics import make_result


class ConsistencyEvaluator:
    def score(self, persona, text, user_text="", retrieved_memories=None, prompt_components=None):
        memories = retrieved_memories or []
        checks = {
            "nonempty_answer": bool(text.strip()),
            "not_dummy_echo": "[DummyLLM]" not in text,
            "no_reasoning_channel": not any(marker in text for marker in
                ("<|channel>thought", "Thinking Process:", "<think>")),
        }
        result = make_result(checks)
        result["method"] = "observable_response_checks_v2"
        result["scope"] = "Sanity checks only; behavioral persona fidelity requires labeled evaluation cases."
        # Correct paraphrases need not repeat an entire archived message verbatim.
        def _grounded(answer: str, memory_list) -> bool:
            answer_l = (answer or "").lower()
            for memory in memory_list:
                content = str(memory or "")
                if not content:
                    continue
                if content in answer:
                    return True
                # Token overlap for distinctive codes (e.g. BLUE-ORBIT-7741).
                tokens = [tok for tok in content.replace("，", " ").replace(",", " ").split() if len(tok) >= 6]
                if tokens and any(tok.lower() in answer_l for tok in tokens):
                    return True
            return False

        result["evidence"] = {
            "retrieval_grounding": _grounded(text, memories),
            "retrieved_count": len(memories),
        }
        return result
