"""Plan 38 J0.1: avatar identity lock (SHA-256) and 'avatar.jpg never in the public repo'."""
import subprocess
import unittest

from avatar.identity import (AVATAR_PATH, AVATAR_SHA256, AVATAR_SIZE, ROOT, VIDEO_PATH, VIDEO_SHA256, sha256_file,
                             verify_assets)


class AvatarIdentityLockTest(unittest.TestCase):
    def test_reference_video_sha_locked(self):
        self.assertTrue(VIDEO_PATH.exists(), "generated_video-3.mp4 is tracked and must exist")
        self.assertEqual(sha256_file(VIDEO_PATH), VIDEO_SHA256)

    def test_avatar_sha_locked(self):
        if not AVATAR_PATH.exists():
            self.skipTest("avatar.jpg is private (private sync only); not present in this checkout")
        self.assertEqual(sha256_file(AVATAR_PATH), AVATAR_SHA256)
        from PIL import Image
        with Image.open(AVATAR_PATH) as im:
            self.assertEqual(im.size, AVATAR_SIZE)

    def test_avatar_not_tracked_by_git(self):
        try:
            proc = subprocess.run(["git", "ls-files", "--error-unmatch", "web/gemma_chat/avatar.jpg"], cwd=ROOT,
                                  capture_output=True, text=True)
        except FileNotFoundError:
            self.skipTest("git not available")
        if "not a git repository" in (proc.stderr or ""):
            self.skipTest("not a git checkout")
        self.assertNotEqual(proc.returncode, 0, "avatar.jpg must NOT be committed to the public repo")

    def test_verify_assets_payload(self):
        status = verify_assets()
        self.assertIn("video", status)
        self.assertTrue(status["video"]["ok"])
        if AVATAR_PATH.exists():
            self.assertTrue(status["ok"])


if __name__ == "__main__":
    unittest.main()
