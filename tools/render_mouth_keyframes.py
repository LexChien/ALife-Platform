#!/usr/bin/env python3
"""Plan 38 J3.2: render K mouth-openness keyframes of the FIXED avatar (2D ROI warp, promoted from
tools/plan38_lipsync_warp.py) and guard every frame. Needs insightface + skimage (~/plan38_cache/venv).
Output (PRIVATE, never committed): runs/plan38/avatar/mouth/k{0..K-1}.jpg + manifest.json; k0 = untouched avatar.jpg."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from avatar.identity import AVATAR_PATH, AVATAR_SHA256, sha256_file
from avatar.lipsync import K

OUT = ROOT / "runs/plan38/avatar/mouth"


def main():
    from insightface.app import FaceAnalysis
    from skimage.transform import PiecewiseAffineTransform, warp
    assert sha256_file(AVATAR_PATH) == AVATAR_SHA256, "avatar.jpg changed - identity lock violated"
    OUT.mkdir(parents=True, exist_ok=True)
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    img = np.array(Image.open(AVATAR_PATH).convert("RGB")); H, W = img.shape[:2]
    f = max(fa.get(img[:, :, ::-1]), key=lambda z: z.det_score); lm = f.landmark_2d_106
    mouth = lm[52:72]; c = mouth.mean(0); mw = np.ptp(mouth[:, 0])
    eyes = (lm[35:43].mean(0) + lm[89:97].mean(0)) / 2; down = (c - eyes) / np.linalg.norm(c - eyes)
    lower = [p for p in mouth if np.dot(p - c, down) > 0]; upper = [p for p in mouth if np.dot(p - c, down) <= 0]
    x0, y0 = np.maximum(c - 1.6 * mw, 0).astype(int); x1, y1 = np.minimum(c + 1.6 * mw, [W - 1, H - 1]).astype(int)
    border = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]])
    chin_near = [p for p in lm[0:33] if np.linalg.norm(p - c) < 1.5 * mw]
    src = np.vstack([border, upper, lower] + ([np.array(chin_near)] if chin_near else []))
    frames = []
    for k in range(K):
        a = k / (K - 1); path = OUT / f"k{k}.jpg"
        if k == 0:
            shutil.copy(AVATAR_PATH, path)
        else:
            disp = a * 0.22 * mw * down
            dst = np.vstack([border, upper, np.array(lower) + disp] + ([np.array(chin_near) + 0.6 * disp] if chin_near else []))
            tf = PiecewiseAffineTransform(); tf.estimate(dst - [x0, y0], src - [x0, y0])
            out = img.copy(); roi = img[y0:y1, x0:x1].astype(np.float32) / 255
            out[y0:y1, x0:x1] = (warp(roi, tf, mode="edge") * 255).astype(np.uint8)
            Image.fromarray(out).save(path, quality=95)
        changed = float((np.abs(np.array(Image.open(path).convert("RGB")).astype(int) - img.astype(int)).sum(2) > 6).mean())
        frames.append({"k": k, "openness": round(a, 3), "file": path.name, "changed_pixel_fraction": round(changed, 4)})
    man = {"avatar_sha256": AVATAR_SHA256, "K": K, "roi": [int(x0), int(y0), int(x1), int(y1)], "frames": frames,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": "2D piecewise-affine ROI warp (mouth/jaw only)"}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man))


if __name__ == "__main__":
    main()
