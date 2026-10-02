"""Plan 38 J0.2: appearance-invariance guard for the FIXED avatar (promoted from tools/plan38_appearance_guard.py).

Metrics vs the reference avatar.jpg: ArcFace identity cosine (insightface buffalo_l, with rotation search because
the reference face needs a -30 deg rotation), OpenCLIP ViT-B-32 whole-image and face-crop cosine.
Thresholds (spec section 2.4): T_ID=0.55 (below -> FAIL, fall back to the untouched avatar.jpg), T_WARN=0.70
(0.55-0.70 -> human review), T_CLIP=0.78 (CLIP-face), T_NEG=0.40 (negative controls must stay below, otherwise the
guard model itself is broken); face detection must be 100 %.
Heavy deps (insightface, open_clip, torch) are imported lazily; the pure verdict logic is dependency-free."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

T_ID = 0.55
T_WARN = 0.70
T_CLIP = 0.78
T_NEG = 0.40
ROTATIONS = (0, -30, 30, -60, 60, 90, -90)


def verdict(arcface: float | None, clip_face: float | None = None) -> str:
    """PASS / WARN / FAIL for one animated frame compared with the reference avatar."""
    if arcface is None:
        return "FAIL"  # no face found = identity cannot be verified
    if arcface < T_ID:
        return "FAIL"
    if arcface < T_WARN or (clip_face is not None and clip_face < T_CLIP):
        return "WARN"
    return "PASS"


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-frame rows ({'arcface', 'clip_face', 'face_found'}) into a set-level verdict."""
    def stat(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        vals = sorted(vals)
        return {"min": round(vals[0], 4), "median": round(vals[len(vals) // 2], 4), "max": round(vals[-1], 4)}
    verdicts = [verdict(r.get("arcface"), r.get("clip_face")) for r in rows]
    worst = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
    return {"n": len(rows), "faces_found": sum(bool(r.get("face_found")) for r in rows),
            "arcface": stat("arcface"), "clip_full": stat("clip_full"), "clip_face": stat("clip_face"),
            "verdicts": {v: verdicts.count(v) for v in ("PASS", "WARN", "FAIL")}, "verdict": worst if rows else "FAIL"}


def negative_ok(set_summary: dict) -> bool:
    arc = set_summary.get("arcface")
    return arc is None or arc["max"] < T_NEG


def frames_from_video(path: str | Path, every: int = 3) -> list:
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="guard_frames_"))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=not(mod(n\\,{every}))", "-vsync", "vfr",
                    str(tmp / "f_%05d.png")], check=True)
    return [Image.open(f).convert("RGB") for f in sorted(tmp.glob("*.png"))]


def load_images(spec: str | Path, every: int = 3) -> list:
    from PIL import Image
    p = Path(spec)
    if p.is_dir():
        files = sorted([*p.glob("*.png"), *p.glob("*.jpg")])
        return [Image.open(f).convert("RGB") for f in files][::max(1, every)]
    if p.suffix.lower() in {".mp4", ".mov", ".webm"}:
        return frames_from_video(p, every=every)
    return [Image.open(p).convert("RGB")]


class AppearanceGuard:
    def __init__(self, reference: str | Path):
        import numpy as np
        import open_clip
        import torch
        from insightface.app import FaceAnalysis
        from PIL import Image
        self.np, self.torch = np, torch
        self.clip, _, self.prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip.eval()
        self.fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.fa.prepare(ctx_id=-1, det_size=(640, 640))
        self.reference_path = Path(reference)
        self.ref = self.measure(Image.open(reference).convert("RGB"))
        if "arcface" not in self.ref:
            raise RuntimeError("no face detected in the reference avatar")

    def _clip(self, img):
        with self.torch.no_grad():
            e = self.clip.encode_image(self.prep(img).unsqueeze(0))[0].numpy()
        return e / self.np.linalg.norm(e)

    def face(self, img):
        best = None
        for ang in ROTATIONS:
            im = img.rotate(ang, expand=True) if ang else img
            for f in self.fa.get(self.np.array(im)[:, :, ::-1]):
                if best is None or f.det_score > best[0].det_score:
                    best = (f, ang, im)
            if best and best[0].det_score > 0.7:
                break
        return best

    def measure(self, img) -> dict:
        r = {"clip_full": self._clip(img)}
        b = self.face(img)
        if b:
            f, ang, im = b
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            pad = int(0.25 * (x1 - x0))
            crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
            r.update({"arcface": f.normed_embedding, "clip_face": self._clip(crop), "det_score": float(f.det_score),
                      "rot": ang})
        return r

    def compare(self, img) -> dict:
        m = self.measure(img)
        dot = lambda a, b: float(self.np.dot(a, b)) if a is not None and b is not None else None  # noqa: E731
        row = {"arcface": dot(self.ref["arcface"], m.get("arcface")), "clip_full": dot(self.ref["clip_full"], m["clip_full"]),
               "clip_face": dot(self.ref.get("clip_face"), m.get("clip_face")), "face_found": "arcface" in m,
               "rot": m.get("rot")}
        row["verdict"] = verdict(row["arcface"], row["clip_face"])
        return row

    def check_set(self, images: list) -> dict:
        rows = [self.compare(im) for im in images]
        return {**summarize(rows), "rows": rows}
