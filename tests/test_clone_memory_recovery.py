import tempfile
import unittest
from unittest.mock import MagicMock, patch

from digital_clone.engine import DigitalCloneEngine
from digital_clone.memory.store import MemoryStore
from digital_clone.persona.identity import persona_collection_name, resolve_persona_id


class TestCloneMemoryRecoveryAndIsolation(unittest.TestCase):
    def make_store(self, documents):
        vector = MagicMock()
        vector.query.return_value = [documents]
        with patch("digital_clone.memory.store.storage_registry") as registry:
            registry.create.return_value = vector
            store = MemoryStore(collection_name="test_collection")
        return store, vector

    def test_reconstructs_archived_metadata_despite_identical_current_text(self):
        store, vector = self.make_store([
            {"id": "old-profile", "document": "same text",
             "metadata": {"role": "system", "kind": "profile_fact"}},
            {"id": "old-dialogue", "document": "same text",
             "metadata": {"role": "user", "kind": "dialogue"}},
        ])
        store.add("assistant", "same text")
        result = store.retrieve("same", n=2, kinds=["profile_fact", "dialogue"])
        self.assertEqual([(r["id"], r["role"], r["kind"]) for r in result],
                         [("old-profile", "system", "profile_fact"),
                          ("old-dialogue", "user", "dialogue")])
        self.assertEqual(vector.query.call_args.kwargs["where"],
                         {"kind": {"$in": ["profile_fact", "dialogue"]}})

    def test_missing_metadata_and_empty_documents(self):
        store, _ = self.make_store([
            {"id": "legacy", "document": "old message", "metadata": None},
            {"id": "empty", "document": None, "metadata": {}},
        ])
        self.assertEqual(store.retrieve("message")[0]["kind"], "dialogue")
        self.assertEqual(store.retrieve("message", kinds=["profile_fact"]), [])

    def test_profile_facts_are_idempotent(self):
        store, vector = self.make_store([])
        store.add_profile_facts(["fact", "fact"])
        self.assertEqual(len(store.items), 1)
        self.assertEqual(vector.add_documents.call_args_list[0].kwargs["ids"],
                         vector.add_documents.call_args_list[1].kwargs["ids"])

    def test_zero_limits_do_not_return_memory(self):
        store, vector = self.make_store([])
        store.add("user", "private")
        self.assertEqual(store.recent(0), [])
        self.assertEqual(store.retrieve("private", n=0), [])
        self.assertEqual(store.retrieve("private", kinds=[]), [])
        self.assertEqual(store.retrieve_for_prompt("query", limit=0), [])
        vector.query.assert_not_called()

    def test_required_persistence_fails_explicitly(self):
        with patch("digital_clone.memory.store.storage_registry") as registry:
            registry.create.side_effect = ImportError("missing")
            with self.assertRaisesRegex(RuntimeError, "Persistent memory is required"):
                MemoryStore(require_persistence=True)
        store, vector = self.make_store([])
        store.require_persistence = True
        vector.add_documents.side_effect = RuntimeError("write failure")
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            store.add("user", "must be saved")
        self.assertEqual(store.items, [])

    def test_lossy_name_collisions_are_isolated(self):
        names = ["A B", "A_B", "a b", "長" * 80 + "甲", "長" * 80 + "乙"]
        keys = [persona_collection_name(resolve_persona_id({"name": n})) for n in names]
        self.assertEqual(len(set(keys)), len(names))
        self.assertTrue(all(key.isascii() and len(key) <= 63 for key in keys))

    def test_explicit_identity_survives_rename(self):
        before = resolve_persona_id({"id": "persona-1", "name": "Old"})
        after = resolve_persona_id({"id": "persona-1", "name": "New"})
        self.assertEqual(persona_collection_name(before), persona_collection_name(after))
        self.assertNotEqual(persona_collection_name(before), persona_collection_name("persona-2"))
        for invalid in ("", " ", None, 42):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_persona_id({"id": invalid, "name": "valid name"})

    @patch("digital_clone.engine.MemoryStore")
    @patch("digital_clone.engine.create_llm_adapter")
    def test_engine_uses_stable_id_and_persistence_settings(self, create_llm, memory_class):
        cfg = {
            "persona": {"id": "stable-id", "name": "A B", "tone": "calm",
                        "principles": [], "goals": [], "facts": []},
            "memory": {"persist_directory": "configured-db", "require_persistence": True},
            "llm": {"backend": "dummy", "model_family": "dummy"}, "inputs": [],
        }
        with tempfile.TemporaryDirectory() as run_dir:
            engine = DigitalCloneEngine(cfg, run_dir)
        self.assertEqual(engine.persona_id, "stable-id")
        memory_class.assert_called_once_with(
            collection_name=persona_collection_name("stable-id"),
            persist_directory="configured-db", require_persistence=True,
        )


if __name__ == "__main__":
    unittest.main()
