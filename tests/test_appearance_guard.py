"""Plan 38 J0.2: appearance guard thresholds + a real 3-image smoke (needs the insightface venv; skipped otherwise)."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from avatar.appearance_guard import T_CLIP, T_ID, T_NEG, T_WARN, negative_ok, summarize, verdict
from avatar.identity import AVATAR_PATH, ROOT, VIDEO_PATH

GUARD_PY = Path(os.environ.get("APPEARANCE_GUARD_PYTHON", str(Path.home() / "plan38_cache/venv/bin/python")))


class GuardVerdictTest(unittest.TestCase):
    def test_threshold_order(self):
        # spec 2.4: hardest negative 0.316 < T_NEG < T_ID < T_WARN < lowest positive 0.736
        self.assertLess(0.316, T_NEG)
        self.assertLess(T_NEG, T_ID)
        self.assertLess(T_ID, T_WARN)
        self.assertLess(T_WARN, 0.736)
        self.assertAlmostEqual(T_CLIP, 0.78)

    def test_verdicts(self):
        self.assertEqual(verdict(None), "FAIL")
        self.assertEqual(verdict(0.54, 0.9), "FAIL")
        self.assertEqual(verdict(0.60, 0.9), "WARN")
        self.assertEqual(verdict(0.90, 0.70), "WARN")
        self.assertEqual(verdict(0.90, 0.90), "PASS")

    def test_summary_worst_case(self):
        s = summarize([{"arcface": 0.9, "clip_face": 0.9, "face_found": True},
                       {"arcface": None, "clip_face": None, "face_found": False}])
        self.assertEqual(s["verdict"], "FAIL")
        self.assertEqual(s["faces_found"], 1)
        self.assertTrue(negative_ok({"arcface": {"max": 0.2}}))
        self.assertFalse(negative_ok({"arcface": {"max": 0.45}}))


@unittest.skipUnless(GUARD_PY.exists() and AVATAR_PATH.exists() and shutil.which("ffmpeg"),
                     "insightface guard venv / private avatar / ffmpeg not available")
class GuardSmokeTest(unittest.TestCase):
    def test_three_image_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="guard_smoke_"))
        pos = tmp / "pos"
        pos.mkdir()
        shutil.copy(AVATAR_PATH, pos / "a_ref.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(VIDEO_PATH), "-frames:v", "1", str(pos / "b_video.png")],
                       check=True)
        neg = tmp / "neg_astronaut.png"
        subprocess.run([str(GUARD_PY), "-c", f"from skimage import data; from PIL import Image; "
                        f"Image.fromarray(data.astronaut()).resize((256,256)).save('{neg}')"], check=True)
        report = tmp / "report.json"
        proc = subprocess.run([str(GUARD_PY), str(ROOT / "tools/appearance_guard.py"), "check", f"pos={pos}",
                               "--every", "1", "--negative", f"neg={neg}", "--report", str(report)],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        data = json.loads(report.read_text())
        self.assertEqual(data["positive"]["pos"]["faces_found"], 2)
        self.assertGreaterEqual(data["positive"]["pos"]["arcface"]["min"], T_ID)
        self.assertLess(data["negative"]["neg"]["arcface"]["max"], T_NEG)


if __name__ == "__main__":
    unittest.main()
