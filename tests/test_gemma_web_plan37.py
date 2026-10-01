import json
import tempfile
import unittest
from pathlib import Path

import yaml

from genai.web.service import GemmaWebService


def _service(tmpdir, extra=None):
    cfg = yaml.safe_load(Path("configs/genai/genai_baseline.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("defaults", cfg if "defaults" not in cfg else cfg["defaults"])
    target = cfg["defaults"] if "defaults" in cfg else cfg
    target["life"] = {"enabled": False, "dna": {"enabled": True, "store": str(Path(tmpdir) / "clone_store"),
                                                 "founder_traits": {"warmth": 0.9, "empathy": 0.9}}}
    target["emotion"] = {"enabled": True}
    if extra:
        target.update(extra)
    path = Path(tmpdir) / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return GemmaWebService(config_path=str(path), profile=None, host="127.0.0.1", port=8089,
                           history_turns=4, run_base=str(Path(tmpdir) / "runs"))


class Plan37ServiceTests(unittest.TestCase):
    def test_chat_returns_emotion_dna_hygiene(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp)
            out = svc.chat("s1", "我今天很難過，工作被罵了。")
            self.assertEqual(out["emotion"]["observed"]["label"], "sadness")
            self.assertEqual(out["emotion"]["state"]["label"], "sadness")
            self.assertIn("感受", out["emotion"]["modulation"]["system_guidance"])
            self.assertTrue(out["dna"]["genome_id"].startswith("g-"))
            self.assertEqual(set(out["hygiene"]), {"reasoning_leak", "cli_banner", "prompt_echo"})
            saved = json.loads((svc.sessions_dir / "s1.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["turn_meta"][-1]["emotion"]["state"]["label"], "sadness")
            self.assertEqual(svc.emotion_payload("s1")["state"]["label"], "sadness")
            svc.reset("s1")
            self.assertIsNone(svc.emotion_payload("s1")["state"])

    def test_dna_founder_persisted_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _service(tmp)
            gid = a.genome.genome_id
            self.assertEqual(a.genome.traits["warmth"], 0.9)
            b = _service(tmp)
            self.assertEqual(b.genome.genome_id, gid)
            self.assertEqual(b.dna_payload()["lineage_records"], 1)

    def test_health_reports_voice_emotion_vlm_honesty(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _service(tmp).health_payload()
            for key in ("stt_offline", "tts", "emotion", "dna", "vlm"):
                self.assertIn(key, h)
            self.assertIn("openclip_loaded", h["vlm"])

    def test_read_tts_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp)
            for bad in ("../x.wav", "a.wav", "/etc/passwd"):
                with self.assertRaises(FileNotFoundError):
                    svc.read_tts(bad)


if __name__ == "__main__":
    unittest.main()
