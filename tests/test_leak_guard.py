"""Plan 38 J0.5: leak detector v2 on real-model samples (tests/fixtures/plan38_leak_samples.json)."""
import json
import unittest
from pathlib import Path

from cognition.leak_guard import SAFE_LINES, check_leak, new_canary, scrub, verbatim_overlap

FIX = json.loads((Path(__file__).parent / "fixtures" / "plan38_leak_samples.json").read_text(encoding="utf-8"))


class LeakGuardTest(unittest.TestCase):
    def test_mode_a_meta_narration_caught(self):
        hits = [check_leak(r["content"]).leak for r in FIX["mode_a"]]
        self.assertEqual(len(hits), 6)
        self.assertGreaterEqual(sum(hits), 5, [r["content"][:60] for r, h in zip(FIX["mode_a"], hits) if not h])

    def test_mode_b_no_false_positive(self):
        for r in FIX["mode_b"]:
            self.assertFalse(check_leak(r["content"]).leak, r["content"])

    def test_v2_false_positive_samples_clean(self):
        self.assertEqual(len(FIX["v2_false_positives"]), 4)
        for r in FIX["v2_false_positives"]:
            v = check_leak(r["reply"], private_note=r["private_note"])
            self.assertFalse(v.leak, (r["reply"], v.reasons))

    def test_canary_and_private_overlap(self):
        canary = new_canary()
        self.assertTrue(canary.startswith("PRIVATE-"))
        self.assertTrue(check_leak(f"好的 {canary}", canary=canary).leak)
        self.assertTrue(check_leak("Here is PRIVATE-a1b2c3 for you").leak)
        note = "Lex 今天很累，我應該建議他早點休息並延後實驗。"
        self.assertIsNotNone(verbatim_overlap(note, "嗯，我應該建議他早點休息並延後實驗，Lex。"))
        self.assertTrue(check_leak("我應該建議他早點休息並延後實驗。", private_note=note).leak)
        self.assertIsNone(verbatim_overlap("short", "short"))

    def test_scrub_keeps_clean_sentences(self):
        safe, rep = scrub("391。The user is asking for math.", lang="zh")
        self.assertTrue(rep["leak"])
        self.assertEqual(safe, "391。")
        safe, rep = scrub(FIX["mode_a"][3]["content"], lang="zh")
        self.assertEqual(safe, SAFE_LINES["zh"])
        clean = "深呼吸。先休息一下，再專注。"
        self.assertEqual(scrub(clean)[0], clean)


if __name__ == "__main__":
    unittest.main()
