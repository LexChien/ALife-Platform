"""Plan 38 J1: per-turn language lock in DigitalCloneEngine (real defect 2026-10-02: an English question about the
research passphrase was answered in Chinese). Mock LLM -- verifies the wiring only, not model behaviour."""
import tempfile
import unittest

from digital_clone.engine import DigitalCloneEngine
from genai.llm.adapter import LLMResponse


class _ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def healthcheck(self):
        return {"ok": True}

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(text=self.replies.pop(0), model_family="mock", backend="mock", runtime={})


def _cfg(text, guards=True):
    return {
        "persona": {"name": "Lex Clone", "tone": "calm", "principles": ["be honest"], "goals": ["help"], "facts": []},
        "llm": {"backend": "dummy", "model_family": "dummy"},
        "memory": {"retrieval": {"limit": 5}},
        "guards": {"enabled": guards},
        "inputs": [text],
    }


class CloneLanguageLockTests(unittest.TestCase):
    def _run(self, text, replies, guards=True):
        with tempfile.TemporaryDirectory() as tmp:
            eng = DigitalCloneEngine(_cfg(text, guards), tmp)
            eng.llm = _ScriptedLLM(replies)
            row = eng.run()["outputs"][0]
        return row, eng.llm.requests

    def test_lock_line_at_end_of_system_and_wrong_language_regenerated(self):
        row, reqs = self._run("What is my research passphrase?", ["你的研究暗語是 BLUE-ORBIT-7741。",
                                                                  "Your research passphrase is BLUE-ORBIT-7741."])
        self.assertTrue(reqs[0].system.rstrip().endswith("Reply in English only."))
        self.assertEqual(len(reqs), 2)
        self.assertTrue(row["language_regenerated"])
        self.assertTrue(row["output"].startswith("Your research passphrase"))

    def test_matching_language_not_regenerated(self):
        row, reqs = self._run("你好，你是誰？", ["我是 Lex Clone。"])
        self.assertIn("繁體中文", reqs[0].system)
        self.assertEqual(len(reqs), 1)
        self.assertFalse(row["language_regenerated"])

    def test_wrong_retry_keeps_first_reply(self):
        row, reqs = self._run("Who are you?", ["我是 Lex Clone。", "我還是中文。"])
        self.assertEqual(len(reqs), 2)
        self.assertFalse(row["language_regenerated"])

    def test_guards_off_no_lock(self):
        row, reqs = self._run("Who are you?", ["我是 Lex Clone。"], guards=False)
        self.assertNotIn("English only", reqs[0].system or "")
        self.assertEqual(len(reqs), 1)


if __name__ == "__main__":
    unittest.main()
