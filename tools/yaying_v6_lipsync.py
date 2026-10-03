#!/usr/bin/env python3
"""yaying v6 high-fidelity lip pass: LatentSync 1.6 (stage2_512, 512 px mouth crop) on Apple MPS (patched, fp16).

Anticipatory coarticulation: the *driving* audio is advanced by --lead frames (default 2 @25 fps = 80 ms) so visemes
(b/p/m closures, u/o rounding) form slightly before the sound, as in natural Mandarin speech; the final mux uses the
ORIGINAL untouched audio. Writes <out>_meta.json with timings. Run with ~/gen_cache/venv_gen python.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LS = Path(os.path.expanduser("~/gen_cache/LatentSync"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--lead", type=int, default=2); ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--deepcache", type=int, default=1); ap.add_argument("--seed", type=int, default=1247)
    a = ap.parse_args()
    out = Path(a.out).resolve(); wd = out.parent / (out.stem + "_work"); wd.mkdir(parents=True, exist_ok=True)
    vid = Path(a.video).resolve(); aud = Path(a.audio).resolve()
    # LatentSync expects 25 fps video
    v25 = wd / "in25.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-an", "-vf", "fps=25", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(v25)], check=True)
    drv = wd / "drive.wav"; lead_s = a.lead / a.fps
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aud), "-af", f"atrim=start={lead_s:.4f},asetpts=PTS-STARTPTS,apad=pad_dur={lead_s:.4f}",
                    "-ar", "16000", "-ac", "1", str(drv)], check=True)
    raw = wd / "ls_raw.mp4"; t0 = time.time()
    cmd = [sys.executable, "-m", "scripts.inference", "--unet_config_path", "configs/unet/stage2_512.yaml",
           "--inference_ckpt_path", "checkpoints/latentsync_unet.pt", "--inference_steps", str(a.steps), "--guidance_scale", str(a.guidance),
           "--video_path", str(v25), "--audio_path", str(drv), "--video_out_path", str(raw), "--temp_dir", str(wd / "tmp"), "--seed", str(a.seed)]
    if a.deepcache: cmd.append("--enable_deepcache")
    env = dict(os.environ, LS_DEVICE=os.environ.get("LS_DEVICE", "mps"), PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run(cmd, cwd=str(LS), env=env, check=True)
    gen = time.time() - t0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(aud), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    meta = {"model": "LatentSync 1.6 (stage2_512, MPS fp16 port)", "args": vars(a), "lead_s": lead_s, "gen_s": round(gen, 1), "video_in": str(vid), "audio": str(aud)}
    json.dump(meta, open(str(out.with_suffix("")) + "_meta.json", "w"), indent=1)
    print("LIPSYNC_DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
