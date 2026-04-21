from digital_clone.persona.model import PersonaModel
from digital_clone.memory.store import MemoryStore
from digital_clone.consistency.evaluator import ConsistencyEvaluator
from digital_clone.decision.policy import DecisionPolicy

class DigitalCloneEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        p = config["persona"]
        self.persona = PersonaModel(
            name=p["name"],
            tone=p["tone"],
            principles=p["principles"],
            goals=p["goals"],
            facts=p.get("facts", []),
        )
        self.memory = MemoryStore()
        self.memory.add_profile_facts(self.persona.facts)
        self.consistency = ConsistencyEvaluator()
        self.policy = DecisionPolicy()

    def run(self):
        outputs = []
        for text in self.config["inputs"]:
            self.memory.add("user", text)
            plan = self.policy.select(self.persona, self.memory, text)
            reply = f"[{self.persona.name}] tone={plan['tone']} principles={', '.join(plan['principles'])} response={plan['reply']}"
            score = self.consistency.score(
                self.persona,
                reply,
                user_text=text,
                retrieved_memories=plan.get("retrieved_memories", []),
            )
            self.memory.add("assistant", reply)
            outputs.append(
                {
                    "input": text,
                    "output": reply,
                    "consistency": score,
                    "retrieved_memories": plan.get("retrieved_memories", []),
                }
            )
        return outputs
