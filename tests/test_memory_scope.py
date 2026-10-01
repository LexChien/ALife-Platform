import unittest

from digital_clone.memory.store import MemoryStore


class _NoChroma(MemoryStore):
    def __init__(self):
        self.items = []
        self.collection_name = "t"
        self.require_persistence = False
        self.vector_store = None
        self.use_vector_db = False


class MemoryScopeTests(unittest.TestCase):
    def _store(self, persistent):
        if persistent:
            import tempfile
            self._tmp = tempfile.TemporaryDirectory()
            return MemoryStore(collection_name="scope_test", persist_directory=self._tmp.name)
        return _NoChroma()

    def _check(self, store):
        store.add("user", "我今天很難過，工作被罵了。", scope="A")
        store.add("user", "請記住：我的暗語是 BLUE-ORBIT-7741。", kind="user_fact", scope="A")
        store.add("user", "Lenia 是什麼？", scope="B")
        got_b = [m["content"] for m in store.retrieve_for_prompt("暗語是什麼", limit=5, scope="B")]
        self.assertTrue(any("BLUE-ORBIT-7741" in c for c in got_b), got_b)   # global user fact
        self.assertFalse(any("難過" in c for c in got_b), got_b)              # A's dialogue isolated
        got_a = [m["content"] for m in store.retrieve_for_prompt("心情", limit=5, scope="A")]
        self.assertTrue(any("難過" in c for c in got_a), got_a)
        unscoped = [m["content"] for m in store.retrieve_for_prompt("心情", limit=5)]
        self.assertTrue(any("難過" in c for c in unscoped))                   # legacy/global behaviour unchanged

    def test_in_memory_scope(self):
        self._check(self._store(False))

    def test_chroma_scope(self):
        store = self._store(True)
        if not store.use_vector_db:
            self.skipTest("chromadb unavailable")
        self._check(store)


if __name__ == "__main__":
    unittest.main()
