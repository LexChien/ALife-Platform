#!/usr/bin/env python3
"""Plan 40 demo v2: identity check of Option A frames vs the nearest ORIGINAL frame + 6 keyframe side-by-side PNGs
(original | output) for Option A and Option B. Reads the decoded output MP4s (what the viewer sees).

  ~/plan38_cache/venv/bin/python tools/yaying_demo_v2_check.py [--which A|B|AB]
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs/yaying_clone/demo_v2"


def frames(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, p.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sbs(a, b, la, lb, out, zoom_box=None):
    row = np.concatenate([label(a, la), np.full((a.shape[0], 6, 3), 255, np.uint8), label(b, lb)], 1)
    if zoom_box is not None:
        x1, y1, x2, y2 = zoom_box
        za = cv2.resize(a[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        zb = cv2.resize(b[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        z = np.concatenate([za, np.full((za.shape[0], 6, 3), 255, np.uint8), zb], 1)
        pad = np.zeros((z.shape[0], row.shape[1], 3), np.uint8); pad[:, :z.shape[1]] = z[:, :row.shape[1]]
        row = np.concatenate([row, np.full((6, row.shape[1], 3), 255, np.uint8), pad], 0)
    cv2.imwrite(str(out), row[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="AB")
    a = ap.parse_args()
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    face = lambda im: (lambda fs: max(fs, key=lambda f: f.det_score) if fs else None)(fa.get(np.ascontiguousarray(im[:, :, ::-1])))  # noqa
    src = frames(D / "work/src.mp4")
    tl = json.load(open(D / "yaying_v2_A2_original_timeline.json"))
    pos = np.array(tl["timeline"])
    kd = D / "keyframes"; kd.mkdir(exist_ok=True)
    src_emb = {}
    res = {}
    if "A" in a.which:
        A = frames(D / "yaying_v2_A2_original.mp4")
        arcs, ss, nof = [], [], 0
        for i in range(0, len(A), 1):
            j = int(round(pos[i]))
            if j not in src_emb:
                f = face(src[j]); src_emb[j] = f.normed_embedding if f is not None else None
            f = face(A[i])
            if f is None or src_emb[j] is None:
                nof += 1; continue
            arcs.append(float(np.dot(f.normed_embedding, src_emb[j])))
        res["A"] = {"frames": len(A), "no_face": nof, "arcface_vs_nearest_original_min": round(min(arcs), 4),
                    "arcface_vs_nearest_original_median": round(float(np.median(arcs)), 4),
                    "arcface_p01": round(float(np.percentile(arcs, 1)), 4)}
        print(res["A"], flush=True)
        for k, i in enumerate(np.linspace(30, len(A) - 8, 6).astype(int)):
            j = int(round(pos[i]))
            sbs(src[j], A[i], f"original frame {j}", f"A2 t={i / 24:.2f}s pos={pos[i]:.2f}", kd / f"A2_key{k + 1}_t{i:04d}.png")
    if "B" in a.which:
        B = frames(D / "yaying_v2_B_lipsync.mp4")
        g = json.load(open(D / "yaying_v2_B_lipsync_gate.json"))
        base = np.load(D / "work/A2_frames.npy", mmap_mode="r")
        rows = g["rows"]
        cand = [r for r in rows if r.get("gate") == "PASS" and r.get("blend", 0) >= 0.5]
        # pick 6 spread over time with the largest mouth-region change vs base
        picks = []
        if cand:
            groups = np.array_split(np.array([r["i"] for r in cand]), 6)
            for grp in groups:
                if len(grp) == 0:
                    continue
                best = max(grp, key=lambda i: float(np.abs(B[i].astype(np.int16) - np.array(base[i]).astype(np.int16)).mean()))
                picks.append(int(best))
        for k, i in enumerate(picks):
            f = face(np.array(base[i]))
            x1, y1, x2, y2 = f.bbox.astype(int) if f is not None else (0, 0, 80, 80)
            cy = int(y1 + 0.62 * (y2 - y1)); cx = int((x1 + x2) / 2); r = int(0.32 * (x2 - x1))
            box = (max(0, cx - r), max(0, cy - r), cx + r, cy + r)
            r_ = rows[i]
            sbs(np.array(base[i]), B[i], f"original (A2 base) t={i / 24:.2f}s",
                f"B lip-sync blend {r_.get('blend')} ArcFace {r_['arcface']} SSIMout {r_['ssim_outside']}", kd / f"B_key{k + 1}_t{i:04d}.png", box)
        res["B_keyframes"] = picks
    json.dump(res, open(D / "check.json", "w"), indent=1)
    print("CHECK_DONE", res)


if __name__ == "__main__":
    main()
