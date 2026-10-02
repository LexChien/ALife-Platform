"""Plan 40: switchable persona/voice profiles (digiclone <-> yaying) without breaking the original persona."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from genai.web.clone_profiles import BASE, CloneProfiles, build_voice
from voice.clone_tts import CloneTTS
from voice.tts_stream import AckCache

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = {
    "system": "你是 ALife Prototype。",
    "voice": {"tts_voice": "Meijia"},
    "cognition": {"speech_rules": "Speak like a calm assistant."},
    "life": {"persona": {"name": "ALife Prototype", "tone": "calm, factual, precise", "principles": ["p1"], "goals": ["g1"]}},
}


class FakeTTS:
    def __init__(self, voice, provider):
        self.voice, self.provider, self.calls = voice, provider, []

    def synthesize(self, text, outdir, stem=None, **kw):
        self.calls.append(stem)
        return {"file": f"{stem}.wav"}


class CloneProfilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.persona = self.tmp / "persona_yaying.yaml"
        self.persona.write_text(
            "persona:\n  name: 雅英\n  tone: warm, playful\nsystem: 你是雅英。\nspeech_style: 句尾常用「喔」。\n", encoding="utf-8")
        os.environ.pop("GEMMA_WEB_CLONE_PROFILE", None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("GEMMA_WEB_CLONE_PROFILE", None)

    def cfg(self, default="yaying", persona_file=None):
        c = json.loads(json.dumps(BASE_CFG))
        c["clone_profiles"] = {"default": default, "profiles": {"yaying": {
            "label": "雅英", "persona_file": str(persona_file or self.persona), "tts_voice": "雅英"}}}
        c["voices"] = {"雅英": {"provider": "clone_tts", "engine": "f5", "python": str(self.tmp / "nope/python"),
                               "ref_wav": str(self.tmp / "ref.wav"), "ref_text_file": str(self.tmp / "ref.txt")}}
        return c

    def test_base_profile_is_original_config(self):
        p = CloneProfiles(BASE_CFG, root=self.tmp)
        self.assertEqual(p.active, BASE)
        b = p.get()
        self.assertEqual(b["persona"]["name"], "ALife Prototype")
        self.assertEqual(b["system"], BASE_CFG["system"])
        self.assertEqual(b["speech_rules"], BASE_CFG["cognition"]["speech_rules"])
        self.assertEqual(b["tts_voice"], "Meijia")

    def test_yaying_profile_overlays_without_mutating_base(self):
        p = CloneProfiles(self.cfg(), root=self.tmp)
        self.assertEqual(p.active, "yaying")
        y = p.get()
        self.assertEqual(y["persona"]["name"], "雅英")
        self.assertEqual(y["persona"]["principles"], ["p1"])  # inherited from base
        self.assertEqual(y["system"], "你是雅英。")
        self.assertIn("Speak like a calm assistant.", y["speech_rules"])
        self.assertIn("喔", y["speech_rules"])
        self.assertEqual(y["tts_voice"], "雅英")
        self.assertEqual(p.get(BASE)["persona"]["name"], "ALife Prototype")
        p.switch(BASE)
        self.assertEqual(p.active, BASE)
        p.switch("yaying")
        self.assertEqual(p.active, "yaying")

    def test_missing_private_persona_falls_back_to_base(self):
        p = CloneProfiles(self.cfg(persona_file=self.tmp / "missing.yaml"), root=self.tmp)
        self.assertEqual(p.active, BASE)
        self.assertFalse(p.public()["profiles"]["yaying"]["available"])
        with self.assertRaises(ValueError):
            p.switch("yaying")
        with self.assertRaises(KeyError):
            p.switch("nobody")

    def test_env_override(self):
        os.environ["GEMMA_WEB_CLONE_PROFILE"] = BASE
        self.assertEqual(CloneProfiles(self.cfg(), root=self.tmp).active, BASE)

    def test_public_lists_voices(self):
        pub = CloneProfiles(self.cfg(), root=self.tmp).public()
        self.assertEqual(set(pub["profiles"]), {BASE, "yaying"})
        self.assertIn("Meijia", pub["voices"])
        self.assertIn("雅英", pub["voices"])

    def test_build_voice_reads_ref_text_and_falls_back(self):
        (self.tmp / "ref.txt").write_text("大家好。", encoding="utf-8")
        fb = FakeTTS("Meijia", "macos_resident")
        v = build_voice(self.cfg()["voices"]["雅英"], fallback=fb, root=self.tmp, name="雅英")
        self.assertIsInstance(v, CloneTTS)
        self.assertEqual(v.ref_text, "大家好。")
        self.assertFalse(v.available())  # python missing -> unavailable
        out = v.synthesize("你好", self.tmp)  # falls back to Meijia instead of failing the turn
        self.assertEqual(out["file"], "None.wav")
        self.assertIn("clone_error", out)
        with self.assertRaises(ValueError):
            build_voice({"provider": "other"})

    def test_ack_retarget_renames_for_clone_voice(self):
        meijia = FakeTTS("Meijia", "macos_resident")
        ack = AckCache(meijia, self.tmp)
        ack.warm()
        self.assertEqual(ack.files["zh"][0], "ack_zh_0.wav")  # J2 names unchanged for the default voice
        clone = FakeTTS("雅英", "clone_tts")
        ack.retarget(clone)
        self.assertTrue(ack.files["zh"][0].startswith("ack_") and ack.files["zh"][0] != "ack_zh_0.wav")
        self.assertTrue(clone.calls)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_avatar_clip_mapping(self):
        js = (ROOT / "web/gemma_chat/avatar.js").read_text(encoding="utf-8")
        start = js.index("  function pickClip(")
        end = js.index("  function applyClip()")
        code = js[start:end] + """
const all = {idle:1, smile:1, concerned:1, yaying_smile:1, yaying_glance:1, yaying_head_tilt:1, yaying_idle_sway:1, yaying_speaking:1};
const j3 = {idle:1, smile:1, concerned:1};
console.log(JSON.stringify([
  pickClip("idle","neutral","digiclone",all), pickClip("idle","smile","digiclone",all), pickClip("speaking","neutral","digiclone",all),
  pickClip("idle","concerned","digiclone",all), pickClip("idle","neutral","yaying",all), pickClip("idle","smile","yaying",all),
  pickClip("thinking","neutral","yaying",all), pickClip("listening","neutral","yaying",all), pickClip("speaking","smile","yaying",all),
  pickClip("idle","neutral","yaying",j3), pickClip("speaking","neutral","yaying",j3)]));"""
        out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), ["idle", "smile", None, "concerned", "yaying_idle_sway", "yaying_smile",
                                                  "yaying_glance", "yaying_head_tilt", "yaying_speaking", "idle", None])


class ClipWhitelistTest(unittest.TestCase):
    def test_plan40_clip_names_whitelisted_and_traversal_blocked(self):
        import re
        src = (ROOT / "apps/gemma_web.py").read_text(encoding="utf-8")
        pat = re.search(r're\.fullmatch\(r"(\(\?:idle[^"]+)"', src).group(1)
        for ok in ("idle.mp4", "yaying_smile.mp4", "yaying_speaking.mp4", "manifest.json"):
            self.assertTrue(re.fullmatch(pat, ok), ok)
        for bad in ("../avatar.jpg", "yaying_x.mp4", "yaying_smile.mp4/../../a", "raw.mp4"):
            self.assertFalse(re.fullmatch(pat, bad), bad)


if __name__ == "__main__":
    unittest.main()
