#!/usr/bin/env python3
"""Plan 40 demo v4: keyframe side-by-sides (base | v4) and a consecutive-frame mouth strip for visual jitter review.
  ~/yaying_cache/venv_mt/bin/python tools/yaying_demo_v4_keyframes.py
"""
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs/yaying_clone/demo_v2"
K = D / "keyframes"


def main():
    base = np.load(D / "work/v4_base.npy", mmap_mode="r"); fin = np.load(D / "work/v4_final.npy", mmap_mode="r")
    rows = json.load(open(D / "yaying_v4_gate.json"))["rows"]
    ls = [r["i"] for r in rows if r.get("lipsync") and r["alpha"] >= 0.5]
    K.mkdir(exist_ok=True)
    picks = [ls[int(round(q * (len(ls) - 1)))] for q in np.linspace(0.05, 0.95, 6)]
    out = []
    for n, i in enumerate(picks):
        img = np.concatenate([base[i], np.full((base.shape[1], 8, 3), 255, np.uint8), fin[i]], 1)
        p = K / f"v4_sbs_{n}_f{i:03d}.png"; cv2.imwrite(str(p), img[:, :, ::-1]); out.append(str(p.relative_to(ROOT)))
    # mouth strip: 12 consecutive frames from the middle of the utterance, base (top) vs v4 (bottom), 2x zoom
    mid = ls[len(ls) // 2 - 6: len(ls) // 2 + 6]
    h, w = base.shape[1:3]
    y0, y1, x0, x1 = int(h * 0.20), int(h * 0.36), int(w * 0.30), int(w * 0.70)
    rb = np.concatenate([cv2.resize(np.ascontiguousarray(base[i][y0:y1, x0:x1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for i in mid], 1)
    rf = np.concatenate([cv2.resize(np.ascontiguousarray(fin[i][y0:y1, x0:x1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for i in mid], 1)
    strip = np.concatenate([rb, np.full((6, rb.shape[1], 3), 255, np.uint8), rf], 0)
    p = K / f"v4_mouth_strip_f{mid[0]:03d}-{mid[-1]:03d}.png"; cv2.imwrite(str(p), strip[:, :, ::-1]); out.append(str(p.relative_to(ROOT)))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
