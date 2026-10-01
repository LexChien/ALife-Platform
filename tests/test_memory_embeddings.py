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


if __name__ == "__main__":
    unittest.main()
