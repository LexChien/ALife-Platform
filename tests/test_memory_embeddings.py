import tempfile
import unittest

try:
    import sentence_transformers  # noqa: F401
    import chromadb  # noqa: F401
    HAVE = True
except Exception:
    HAVE = False

from digital_clone.memory.embeddings import collection_suffix


class CollectionSuffixTests(unittest.TestCase):
    def test_suffix_is_stable_and_safe(self):
        self.assertEqual(collection_suffix("intfloat/multilingual-e5-small"), "__multilingual_e5_small")
        self.assertEqual(collection_suffix("BAAI/bge-small-zh-v1.5"), "__bge_small_zh_v1_5")


@unittest.skipUnless(HAVE, "requires sentence-transformers + chromadb (installed on Mac)")
class MultilingualMemoryTests(unittest.TestCase):
    """Real model (intfloat/multilingual-e5-small from local HF cache); no mock."""

    @classmethod
    def setUpClass(cls):
        from digital_clone.memory.embeddings import PrefixedSentenceTransformerEF
        try:
            cls.ef = PrefixedSentenceTransformerEF()
        except Exception as exc:  # model not cached / offline
            raise unittest.SkipTest(f"e5-small unavailable: {exc}")

    def test_query_and_doc_prefixes_differ(self):
        d = self.ef(["我的貓叫麻糬"])[0]
        q = self.ef.embed_query(["我的貓叫麻糬"])[0]
        self.assertGreater(float((d * q).sum()), 0.8)
        self.assertLess(float((d * q).sum()), 0.9999)

    def test_chinese_paraphrase_retrieval_and_migration(self):
        from digital_clone.memory.embeddings import migrate_collection
        from digital_clone.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            old = MemoryStore(collection_name="mem", persist_directory=td, require_persistence=True)
            facts = ["我對花生過敏，吃到會起疹子。", "我每天早上六點去河濱公園慢跑。", "我最喜歡的電影是《神隱少女》。"]
            for i, f in enumerate(facts):
                old.add("user", f, kind="user_fact", memory_id=f"f{i}")
            new = MemoryStore(collection_name="mem" + collection_suffix(self.ef.model_name), persist_directory=td,
                              require_persistence=True, embedding_function=self.ef)
            self.assertEqual(migrate_collection(td, "mem", new.vector_store), 3)
            self.assertEqual(migrate_collection(td, "mem", new.vector_store), 0)  # idempotent
            got = new.retrieve("有什麼食物我不能碰？", n=1, kinds=["user_fact"])
            self.assertEqual(got[0]["id"], "f0")
            self.assertEqual(got[0]["content"], facts[0])  # stored text has no "passage: " prefix


    def test_second_switch_keeps_memories_from_intermediate_collection(self):
        from digital_clone.memory.embeddings import migrate_collection, migration_sources
        from digital_clone.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            base = MemoryStore(collection_name="mem", persist_directory=td, require_persistence=True)
            base.add("user", "我的貓叫麻糬。", kind="user_fact", memory_id="a")
            mid = MemoryStore(collection_name="mem__first", persist_directory=td, require_persistence=True,
                              embedding_function=self.ef)
            self.assertEqual(migrate_collection(td, migration_sources(td, "mem", "mem__first"), mid.vector_store), 1)
            mid.add("user", "My sister lives in Osaka.", kind="user_fact", memory_id="b")  # written after 1st switch
            target = MemoryStore(collection_name="mem__second", persist_directory=td, require_persistence=True,
                                 embedding_function=self.ef)
            sources = migration_sources(td, "mem", "mem__second")
            self.assertEqual(sources, ["mem__first", "mem"])
            self.assertEqual(migrate_collection(td, sources, target.vector_store), 2)  # union, de-duplicated
            self.assertEqual(target.vector_store.collection.count(), 2)
            self.assertEqual(base.vector_store.collection.count(), 1)  # sources untouched



class ConcurrentEncodeLockTests(unittest.TestCase):
    """Fake model: concurrent _encode must never overlap (MPS race guard). No real model/MPS."""

    def test_encode_is_serialised(self):
        import threading
        import time
        import numpy as np
        from digital_clone.memory import embeddings as emb

        class FakeModel:
            def __init__(self):
                self.concurrent = 0
                self.max_concurrent = 0
                self._lock = threading.Lock()

            def encode(self, texts, **kwargs):
                with self._lock:
                    self.concurrent += 1
                    self.max_concurrent = max(self.max_concurrent, self.concurrent)
                time.sleep(0.04)
                with self._lock:
                    self.concurrent -= 1
                return np.zeros((len(texts), 4), dtype=np.float32)

        fake = FakeModel()
        key = ("__test_fake_model__", "cpu")
        with emb._CACHE_LOCK:
            emb._MODEL_CACHE[key] = fake
        try:
            ef = emb.PrefixedSentenceTransformerEF.__new__(emb.PrefixedSentenceTransformerEF)
            ef.model_name = key[0]
            ef.device = key[1]
            ef.doc_prefix = ""
            ef.query_prefix = ""
            ef._model = fake
            errors = []

            def worker():
                try:
                    ef._encode(["hello world"])
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertEqual(fake.max_concurrent, 1)
        finally:
            with emb._CACHE_LOCK:
                emb._MODEL_CACHE.pop(key, None)

if __name__ == "__main__":
    unittest.main()
