"""Plan 38 J0.1: the female avatar is FIXED. These hashes are the identity lock; any animation or UI change
must keep these files byte-identical and pass the appearance guard (avatar/appearance_guard.py).
avatar.jpg is private (never committed to the public repo); it is distributed by the private sync only."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AVATAR_PATH = ROOT / "web" / "gemma_chat" / "avatar.jpg"
VIDEO_PATH = ROOT / "web" / "gemma_chat" / "generated_video-3.mp4"
AVATAR_SHA256 = "afa001b33a616e1bd64e5e9606d77849cf67700295bfdd47ada62cc49b25176d"
VIDEO_SHA256 = "4635265b7d8dcae52e1892b4903465ee9ea45b1fdc06d77cc44aa3fc71e81daf"
AVATAR_SIZE = (464, 688)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_assets() -> dict:
    """Return the lock status of both reference assets (missing avatar.jpg is reported, not raised)."""
    out = {}
    for name, path, expected in (("avatar", AVATAR_PATH, AVATAR_SHA256), ("video", VIDEO_PATH, VIDEO_SHA256)):
        if not path.exists():
            out[name] = {"present": False, "ok": False}
            continue
        got = sha256_file(path)
        out[name] = {"present": True, "ok": got == expected, "sha256": got}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out
