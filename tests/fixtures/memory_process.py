"""Real ChromaDB persistence worker with deterministic, offline test embeddings."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chromadb.api.types import EmbeddingFunction
from digital_clone.memory.store import MemoryStore
from digital_clone.persona.identity import persona_collection_name, resolve_persona_id


class TestEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return [[(byte + 1) / 256.0 for byte in hashlib.sha256(text.encode()).digest()]
                for text in input]

    @staticmethod
    def name():
        return "alife-offline-test-embedding"

    @staticmethod
    def build_from_config(config):
        return TestEmbedding()

    def get_config(self):
        return {}


def main():
    action, database = sys.argv[1:]
    personas = [{"name": "A B"}, {"name": "A_B"},
                {"name": "長" * 80 + "甲"}, {"name": "長" * 80 + "乙"},
                {"id": "stable-persona", "name": "Before" if action == "write" else "After"}]
    results = []
    for index, persona in enumerate(personas):
        key = persona_collection_name(resolve_persona_id(persona))
        store = MemoryStore(key, database, require_persistence=True,
                            embedding_function=TestEmbedding())
        if action == "write":
            store.add_profile_facts([f"profile-{index}", f"profile-{index}"])
            store.add("user", f"private-memory-{index}")
        results.append({"collection": key, "current_items": len(store.items),
                        "memories": store.retrieve("private memory", n=100),
                        "count": store.vector_store.collection.count()})
    # An ambiguous legacy collection must not be silently assigned to either persona.
    if action == "write":
        legacy = MemoryStore("clone_a_b", database, require_persistence=True,
                             embedding_function=TestEmbedding())
        legacy.add("user", "legacy-owner-unknown")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
