#!/usr/bin/env python3
"""Plan 40 phase 2: lip-synced 雅英 demo. Base frames = an installed, identity-guarded clip (default yaying_idle_sway,
LivePortrait re-enactment of the fixed avatar), looped to the clone-audio length; per frame, only the mouth/jaw region
(insightface 2d106) is displaced by the clone WAV RMS envelope (same method as tools/plan38_lipsync_warp.py E-LIP-A).
Every other pixel is the clip frame. Output mp4 (+audio) is then checked with tools/appearance_guard.py.

  ~/plan38_cache/venv/bin/python tools/yaying_lipsync_demo.py WAV OUT.mp4 [--clip web/gemma_chat/clips/yaying_idle_sway.mp4]"""
import argparse, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, soundfile as sf
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
FPS = 25


def frames_of(clip, n, d):
    src = d / "src"; src.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", str(clip), "-vf", f"fps={FPS}", "-frames:v", str(n),
                    str(src / "s_%05d.png")], check=True)
    return sorted(src.glob("s_*.png"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("wav"); ap.add_argument("out")
    ap.add_argument("--clip", default=str(ROOT / "web/gemma_chat/clips/yaying_idle_sway.mp4")); ap.add_argument("--gain", type=float, default=0.22)
    a = ap.parse_args()
    from insightface.app import FaceAnalysis
    from skimage.transform import PiecewiseAffineTransform, warp
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
    x, sr = sf.read(a.wav, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    hop = sr // FPS; env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1); env = np.convolve(env, [0.25, 0.5, 0.25], "same")
    d = Path(tempfile.mkdtemp(prefix="yy_lip_")); srcs = frames_of(a.clip, len(env), d); times = []; nofaces = 0; last = None
    for k, amp in enumerate(env):
        t0 = time.perf_counter(); img = np.array(Image.open(srcs[k % len(srcs)]).convert("RGB")); H, W = img.shape[:2]
        faces = fa.get(img[:, :, ::-1])
        if faces:
            last = max(faces, key=lambda z: z.det_score).landmark_2d_106
        elif last is None:
            nofaces += 1; Image.fromarray(img).save(d / f"f_{k:05d}.png"); continue
        else:
            nofaces += 1
        lm = last; mouth = lm[52:72]; c = mouth.mean(0); mw = np.ptp(mouth[:, 0])
        eyes = (lm[35:43].mean(0) + lm[89:97].mean(0)) / 2; down = (c - eyes) / np.linalg.norm(c - eyes)
        upper = [p for p in mouth if np.dot(p - c, down) <= 0]; lower = [p for p in mouth if np.dot(p - c, down) > 0]
        x0, y0 = np.maximum(c - 1.6 * mw, 0).astype(int); x1, y1 = np.minimum(c + 1.6 * mw, [W - 1, H - 1]).astype(int)
        border = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1], [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]])
        chin = [p for p in lm[0:33] if np.linalg.norm(p - c) < 1.5 * mw]
        disp = amp * a.gain * mw * down
        src = np.vstack([border, upper, lower] + ([np.array(chin)] if chin else []))
        dst = np.vstack([border, upper, np.array(lower) + disp] + ([np.array(chin) + 0.6 * disp] if chin else []))
        roi = img[y0:y1, x0:x1].astype(np.float32) / 255
        tf = PiecewiseAffineTransform(); tf.estimate(dst - [x0, y0], src - [x0, y0])
        out = img.copy(); out[y0:y1, x0:x1] = (warp(roi, tf, mode="edge") * 255).astype(np.uint8)
        times.append(time.perf_counter() - t0); Image.fromarray(out).save(d / f"f_{k:05d}.png")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(d / "f_%05d.png"), "-i", a.wav, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)
    rep = {"wav": a.wav, "clip": a.clip, "frames": int(len(env)), "fps": FPS, "frames_without_face_detect": nofaces,
           "ms_per_frame_median": round(float(np.median(times)) * 1000, 1) if times else None, "mp4": str(out), "method": "RMS-envelope mouth warp (E-LIP-A) on guarded clip frames; no visemes"}
    out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
