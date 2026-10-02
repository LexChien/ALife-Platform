#!/usr/bin/env python3
"""Plan 38 J0.2 appearance guard CLI (needs insightface + open_clip; on the Mac use ~/plan38_cache/venv/bin/python).

  appearance_guard.py check NAME=PATH [NAME=PATH ...] [--negative NAME=PATH ...] [--report out.json] [--every 3]
      PATH = image, directory of png/jpg, or mp4. Exit 1 if any positive set FAILs (min ArcFace < T_ID or a frame
      without a face) or any negative set reaches T_NEG.
  appearance_guard.py calibrate [--report out.json]
      reference video frames (positive) + built-in negatives (another woman's photo, cat, blurred-face avatar)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avatar.appearance_guard import T_CLIP, T_ID, T_NEG, T_WARN, AppearanceGuard, load_images, negative_ok  # noqa: E402
from avatar.identity import AVATAR_PATH, VIDEO_PATH, sha256_file  # noqa: E402


def builtin_negatives(guard):
    from PIL import Image, ImageFilter
    from skimage import data
    neg = {"neg_other_woman_astronaut": [Image.fromarray(data.astronaut())], "neg_cat": [Image.fromarray(data.chelsea())]}
    ref_img = Image.open(AVATAR_PATH).convert("RGB")
    f, ang, im = guard.face(ref_img)
    x0, y0, x1, y1 = [int(v) for v in f.bbox]
    blurred = im.copy()
    blurred.paste(im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(12)), (x0, y0))
    neg["neg_blurred_face_avatar"] = [blurred.rotate(-ang, expand=False) if ang else blurred]
    return neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "calibrate"])
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--negative", nargs="*", default=[])
    ap.add_argument("--report")
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    t0 = time.perf_counter()
    guard = AppearanceGuard(AVATAR_PATH)
    pos, neg = {}, {}
    if a.mode == "calibrate":
        pos["ref_video_frames"] = load_images(VIDEO_PATH, every=6)
        neg.update(builtin_negatives(guard))
    for spec in a.sets:
        name, path = spec.split("=", 1)
        pos[name] = load_images(path, every=a.every)
    for spec in a.negative:
        name, path = spec.split("=", 1)
        neg[name] = load_images(path, every=a.every)
    report = {"tool": "tools/appearance_guard.py", "mode": a.mode,
              "thresholds": {"T_ID": T_ID, "T_WARN": T_WARN, "T_CLIP": T_CLIP, "T_NEG": T_NEG},
              "reference": {"path": str(AVATAR_PATH.relative_to(ROOT)), "sha256": sha256_file(AVATAR_PATH),
                            "det_rotation": guard.ref.get("rot"), "det_score": guard.ref.get("det_score")},
              "video": {"path": str(VIDEO_PATH.relative_to(ROOT)), "sha256": sha256_file(VIDEO_PATH)},
              "positive": {}, "negative": {}, "argv": sys.argv[1:]}
    ok = True
    for name, imgs in pos.items():
        s = guard.check_set(imgs)
        s_ok = s["verdict"] != "FAIL" and s["faces_found"] == s["n"]
        ok &= s_ok
        report["positive"][name] = {**{k: v for k, v in s.items() if k != "rows"}, "ok": s_ok,
                                    "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in s["rows"]]}
        print(name, json.dumps({k: v for k, v in report["positive"][name].items() if k != "rows"}))
    for name, imgs in neg.items():
        s = guard.check_set(imgs)
        n_ok = negative_ok(s)
        ok &= n_ok
        report["negative"][name] = {**{k: v for k, v in s.items() if k not in ("rows", "verdict", "verdicts")}, "ok": n_ok}
        print(name, json.dumps(report["negative"][name]))
    report["ok"] = bool(ok)
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1))
    print("GUARD", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
