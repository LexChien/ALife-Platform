from pathlib import Path

from digital_clone.persona.model import PersonaModel
from digital_clone.memory.store import MemoryStore
from digital_clone.consistency.evaluator import ConsistencyEvaluator
from digital_clone.decision.policy import ClonePromptBuilder
from genai.llm.adapter import LLMRequest
from genai.llm.factory import create_llm_adapter

class DigitalCloneEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        self.prompt_builder = ClonePromptBuilder()
        self.llm = create_llm_adapter(config)

    def run(self):
        outputs = []
        for text in self.config["inputs"]:
            self.memory.add("user", text)
            memories = self.memory.retrieve_for_prompt(
                text,
                limit=self.config.get("memory", {}).get("retrieval", {}).get("limit", 5),
            )
            built = self.prompt_builder.build(self.persona, memories, text)
            request = LLMRequest(
                prompt=built["prompt"],
                context=built["context"],
                system=built["system"],
                max_tokens=self.config.get("llm", {}).get("max_tokens"),
                temperature=self.config.get("llm", {}).get("temperature"),
                stop=self.config.get("llm", {}).get("stop"),
                json_mode=self.config.get("llm", {}).get("json_mode", False),
            )
            health = self.llm.healthcheck()
            response = self.llm.generate(request)
            response_text = f"我已收到你的問題：{text}"
            if response.text:
                response_text = f"{response_text}；{response.text}"
            reply = (
                f"[{self.persona.name}] "
                f"tone={self.persona.tone} "
                f"principles={', '.join(self.persona.principles[:2])} "
                f"response={response_text}"
            )
            score = self.consistency.score(
                self.persona,
                reply,
                user_text=text,
                retrieved_memories=[m["content"] for m in memories],
                prompt_components=built,
            )
            self.memory.add("assistant", reply)
            outputs.append(
                {
                    "input": text,
                    "output": reply,
                    "consistency": score,
                    "retrieved_memories": [m["content"] for m in memories],
                    "system": built["system"],
                    "context": built["context"],
                    "llm": {
                        "backend": response.backend,
                        "model_family": response.model_family,
                        "runtime": response.runtime,
                        "healthcheck": health,
                    },
                }
            )
        consistencies = [float(row["consistency"]) for row in outputs]
        retrieval_hits = sum(1 for row in outputs if row["retrieved_memories"])
        return {
            "outputs": outputs,
            "summary": {
                "mode": "digital_clone_session",
                "num_inputs": len(outputs),
                "mean_consistency": sum(consistencies) / max(len(consistencies), 1),
                "min_consistency": min(consistencies) if consistencies else 0.0,
                "max_consistency": max(consistencies) if consistencies else 0.0,
                "retrieval_hit_rate": retrieval_hits / max(len(outputs), 1),
                "persona_name": self.persona.name,
                "llm_backend": outputs[0]["llm"]["backend"] if outputs else None,
                "llm_model_family": outputs[0]["llm"]["model_family"] if outputs else None,
                "llm_runtime": outputs[0]["llm"]["runtime"] if outputs else None,
            },
        }
