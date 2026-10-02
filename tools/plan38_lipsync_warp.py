#!/usr/bin/env python3
"""Plan 38 E-LIP-A: identity-preserving 2D mouth warp on the FIXED avatar, driven by a TTS WAV RMS envelope.
Only the mouth/jaw region (insightface 2d106 landmarks) is displaced via piecewise-affine warp; every other
pixel is the original avatar. Output runs/plan38/lipsync/warp_<name>.mp4 (+audio) and timing json.
This is a feasibility baseline, not production lip-sync (no visemes, no teeth/tongue synthesis)."""
import json, sys, time, subprocess, tempfile
from pathlib import Path
import numpy as np, soundfile as sf
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "web/gemma_chat/avatar.jpg"; OUT = ROOT / "runs/plan38/lipsync"; OUT.mkdir(parents=True, exist_ok=True)
FPS = 25

def main(wav, name):
    from insightface.app import FaceAnalysis
    from skimage.transform import PiecewiseAffineTransform, warp
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
    img = np.array(Image.open(REF).convert("RGB")); H, W = img.shape[:2]
    faces = fa.get(img[:, :, ::-1]); assert faces, "no face"
    f = max(faces, key=lambda z: z.det_score); lm = f.landmark_2d_106  # (106,2) x,y
    mouth = lm[52:72]; c = mouth.mean(0); mw = np.ptp(mouth[:, 0])
    # face 'down' direction (eyes->mouth) since the avatar is tilted
    eyes = (lm[35:43].mean(0) + lm[89:97].mean(0)) / 2; down = (c - eyes) / np.linalg.norm(c - eyes)
    lower = [p for p in mouth if np.dot(p - c, down) > 0]
    x0, y0 = np.maximum(c - 1.6 * mw, 0).astype(int); x1, y1 = np.minimum(c + 1.6 * mw, [W - 1, H - 1]).astype(int)
    border = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]])
    chin = lm[0:33]; chin_near = [p for p in chin if np.linalg.norm(p - c) < 1.5 * mw]
    x, sr = sf.read(wav, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    hop = sr // FPS; env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1); env = np.convolve(env, [0.25, 0.5, 0.25], "same")
    d = Path(tempfile.mkdtemp()); times = []
    upper = [p for p in mouth if np.dot(p - c, down) <= 0]
    src = np.vstack([border, upper, lower, chin_near]) if chin_near else np.vstack([border, upper, lower])
    for k, a in enumerate(env):
        t0 = time.perf_counter(); disp = a * 0.22 * mw * down
        dst = np.vstack([border, upper, np.array(lower) + disp] + ([np.array(chin_near) + 0.6 * disp] if chin_near else []))
        roi = img[y0:y1, x0:x1].astype(np.float32) / 255
        tf = PiecewiseAffineTransform(); tf.estimate(dst - [x0, y0], src - [x0, y0])  # inverse map
        out = img.copy(); out[y0:y1, x0:x1] = (warp(roi, tf, mode="edge") * 255).astype(np.uint8)
        times.append(time.perf_counter() - t0); Image.fromarray(out).save(d / f"f_{k:05d}.png")
    mp4 = OUT / f"warp_{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "f_%05d.png"), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(mp4)], check=True)
    rep = {"wav": str(wav), "frames": len(env), "fps": FPS, "ms_per_frame_median": round(float(np.median(times)) * 1000, 1),
           "realtime_factor": round(float(np.sum(times)) / (len(env) / FPS), 3), "roi": [int(x0), int(y0), int(x1), int(y1)],
           "roi_fraction_of_image": round((x1 - x0) * (y1 - y0) / (W * H), 4), "mouth_width_px": round(float(mw), 1), "mp4": str(mp4.relative_to(ROOT))}
    (OUT / f"warp_{name}.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))

if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "sample")
