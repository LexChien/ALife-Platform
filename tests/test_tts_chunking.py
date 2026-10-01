import shutil
import tempfile
import time
import unittest
from pathlib import Path

from genai.web.voice import MacSayTTS, split_for_tts


class SplitForTTSTests(unittest.TestCase):
    def test_sentences_split_and_short_merged(self):
        chunks = split_for_tts("聽到你這樣說，我很難過。你願意多說一點嗎？好。我會陪你。")
        self.assertEqual("".join(chunks), "聽到你這樣說，我很難過。你願意多說一點嗎？好。我會陪你。")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) >= 2 for c in chunks))

    def test_long_first_sentence_cut_at_comma(self):
        text = "這是一個非常非常長的第一句，裡面有很多很多很多的內容需要被唸出來，所以應該在逗號處切開。第二句。"
        chunks = split_for_tts(text, first_max=30)
        self.assertLessEqual(len(chunks[0]), 31)
        self.assertEqual("".join(chunks), text)

    def test_single_and_empty(self):
        self.assertEqual(split_for_tts("好的"), ["好的"])
        self.assertEqual(split_for_tts(""), [])


@unittest.skipUnless(shutil.which("say") and shutil.which("afconvert"), "macOS say/afconvert required")
class ChunkedServiceTTSTests(unittest.TestCase):
    """Real macOS say synthesis (no mock)."""

    def test_first_chunk_ready_rest_arrive(self):
        from tests.test_gemma_web_plan37 import _service
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp)
            svc.tts = MacSayTTS()
            text = "聽到你這樣說，我很難過。你願意多說一點嗎？我會一直在這裡陪你。"
            t0 = time.time()
            out = svc._synthesize_chunked(text, {"rate": 1.0, "pitch": 1.0})
            first_s = time.time() - t0
            self.assertTrue(out["ok"] and out["chunked"])
            urls = [c["url"] for c in out["chunks"]]
            self.assertEqual(out["url"], urls[0])
            for u in urls:  # read_tts waits for pending chunks
                data = svc.read_tts(Path(u).name)
                self.assertGreater(len(data), 1000)
                self.assertEqual(data[:4], b"RIFF")
            self.assertFalse(svc._tts_pending)
            self.assertLess(first_s, 5.0)


if __name__ == "__main__":
    unittest.main()
