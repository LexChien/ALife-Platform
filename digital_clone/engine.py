from pathlib import Path
from dataclasses import asdict

from digital_clone.persona.model import PersonaModel
from digital_clone.persona.identity import persona_collection_name, resolve_persona_id
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
        self.persona_id = resolve_persona_id(p)
        collection_name = persona_collection_name(self.persona_id)
        memory_cfg = config.get("memory", {})
        self.memory = MemoryStore(
            collection_name=collection_name,
            persist_directory=memory_cfg.get("persist_directory"),
            require_persistence=memory_cfg.get("require_persistence", False),
        )
        self.memory.add_profile_facts(self.persona.facts)
        self.consistency = ConsistencyEvaluator()
        self.prompt_builder = ClonePromptBuilder()
        self.llm = create_llm_adapter(config)

    def run(self):
        outputs = []
        for text in self.config["inputs"]:
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
                metadata={"disable_prompt_profile": True, "disable_reasoning_extractor": True},
            )
            health = self.llm.healthcheck()
            response = self.llm.generate(request)
            # Never inject expected persona keywords or the question into answers.
            # Preserve provider payloads separately to audit post-processing.
            reply = response.text
            if not (reply or "").strip():
                # One short retry when post-process emptied a reasoning-only dump.
                retry = LLMRequest(
                    prompt=built["prompt"] + "\n\nAnswer in one or two short sentences only.",
                    context=built["context"],
                    system=(
                        built["system"]
                        + " Reply in at most two short sentences. "
                        + "Never output Thinking Process, analysis steps, or channel tags."
                    ),
                    max_tokens=min(int(self.config.get("llm", {}).get("max_tokens") or 160), 96),
                    temperature=0.1,
                    stop=self.config.get("llm", {}).get("stop"),
                    json_mode=self.config.get("llm", {}).get("json_mode", False),
                    metadata={
                        "disable_prompt_profile": True,
                        "disable_reasoning_extractor": True,
                        "empty_reply_retry": True,
                    },
                )
                response = self.llm.generate(retry)
                reply = response.text
            consistency = self.consistency.score(
                self.persona,
                reply,
                user_text=text,
                retrieved_memories=[m["content"] for m in memories],
                prompt_components=built,
            )
            self.memory.add("user", text)
            self.memory.add("assistant", reply)
            outputs.append(
                {
                    "input": text,
                    "output": reply,
                    "generated_response": response.text,
                    "raw_generation": response.raw,
                    "consistency": consistency,
                    "consistency_score": float(consistency["score"]),
                    "retrieved_memories": [m["content"] for m in memories],
                    "retrieved_records": memories,
                    "request": asdict(request),
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
        consistencies = [float(row["consistency_score"]) for row in outputs]
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
                "persona_id": self.persona_id,
                "memory_backend": "chromadb" if self.memory.use_vector_db else "in_memory",
                "memory_persistent": bool(self.memory.use_vector_db),
                "memory_collection": self.memory.collection_name,
                "llm_backend": outputs[0]["llm"]["backend"] if outputs else None,
                "llm_model_family": outputs[0]["llm"]["model_family"] if outputs else None,
                "llm_runtime": outputs[0]["llm"]["runtime"] if outputs else None,
            },
        }
