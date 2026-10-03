#!/usr/bin/env python3
"""yaying v6 candidate: EchoMimic v3 Flash-Pro, multi-window long-audio generation on Apple MPS (models loaded once).

The whole voice track (lines 0-2, 17.4 s) is embedded once with chinese-wav2vec2 (so audio context is continuous across
window seams). Video is generated in windows of --win frames; window k+1 starts from window k's LAST frame (overlap of one
frame, dropped on assembly), so pose/light are continuous. Per-window text prompts follow the Muse-style motion plan
(glance away -> locked eye contact; slow blink + upward gaze; slow lean + head tilt); hands are NOT prompted (finger gate).
NOTE: the flash pipeline's CFG is audio-only (cond vs zero-audio); the negative prompt is not used in denoising.
MPS fixes: chunked SDPA (yaying_mps_sdpa), Wan-VAE in an fp32 copy on --vae-dev (bf16 conv3d OOM'd), input->weight dtype
cast for Conv3d/Linear (cuda.amp.autocast is a no-op on MPS). Writes <out>, <out>_silent.mp4 and <out>_meta.json.
  ~/gen_cache/venv_gen/bin/python tools/yaying_v6_emv3w.py --image ref.png --audio lines012.wav --out x.mp4
"""
import argparse, json, os, runpy, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EM = Path.home() / "gen_cache/echomimic_v3"
BASE = ("A young East Asian woman with long black hair lies on white bedsheets in warm golden morning sunlight streaming through "
        "window blinds, speaking softly and intimately to the camera in a breathy near-whisper, natural breathing, slow deliberate "
        "subtle movements, warm directional sunlight with soft shadows, photorealistic, same face and hairstyle. ")
CUES = {
    "open": "At first she glances slightly away, then slowly turns her eyes and face to lock gentle eye contact with the viewer; lips slightly parted, a soft smile.",
    "l0": "She holds calm, gentle eye contact with the viewer; between phrases her lips rest slightly parted with a soft smile.",
    "l1": "She blinks slowly, then looks up at the viewer through relaxed, lowered eyelids, tilting her head gently toward her shoulder, soft smile between phrases.",
    "l2": "She leans slowly closer toward the camera with a gentle head tilt, eyes locked on the viewer, lips slightly parted, a soft inviting smile.",
}
NEG = ("deformed fingers, extra fingers, fused fingers, distorted face, asymmetric eyes, blurry, flicker, jelly hair, nudity, explicit, "
       "cartoon, text, watermark, fast motion, camera shake")


def patches(vae_dev):
    import copy, torch
    sys.path.insert(0, str(ROOT / "tools")); import yaying_mps_sdpa; yaying_mps_sdpa.patch(1.5)
    for cls in (torch.nn.Conv3d, torch.nn.Linear, torch.nn.Conv2d):
        f0 = cls.forward
        def fwd(self, x, _f0=f0):
            if x.is_floating_point() and x.dtype != self.weight.dtype:
                x = x.to(self.weight.dtype)
            return _f0(self, x)
        cls.forward = fwd
    import src.wan_vae as wv
    def model_on(self, dev):
        c = getattr(self, "_y_models", {})
        if dev not in c:
            c[dev] = copy.deepcopy(self.model).float().to(dev).eval(); self._y_models = c
        return c[dev]
    def run(self, fn, xs, dev):
        m = model_on(self, dev); sc = [t.float().to(dev) if hasattr(t, "to") else t for t in self.scale]
        with torch.no_grad():
            out = [getattr(m, fn)(u.unsqueeze(0).float().to(dev), sc).squeeze(0) for u in xs]
        if dev == "mps": torch.mps.empty_cache()
        return torch.stack(out)
    def _encode(self, x):
        try:
            h = run(self, "encode", x, vae_dev)
        except RuntimeError as e:
            print(f"[emv3w] VAE encode on {vae_dev} failed ({str(e)[:120]}) -> CPU", flush=True); h = run(self, "encode", x, "cpu")
        return h.to(x.device, x.dtype)
    def _decode(self, zs):
        try:
            d = run(self, "decode", zs, vae_dev)
        except RuntimeError as e:
            print(f"[emv3w] VAE decode on {vae_dev} failed ({str(e)[:120]}) -> CPU", flush=True); d = run(self, "decode", zs, "cpu")
        return wv.DecoderOutput(sample=d.clamp_(-1, 1).to(zs.device, zs.dtype))
    wv.AutoencoderKLWan._encode = _encode; wv.AutoencoderKLWan._decode = _decode


def plan_windows(n_total, win, line_ends_s, fps=25):
    """Equal windows (4n+1 frames, <= win) chained with a 1-frame overlap, covering >= n_total frames."""
    import math
    k = math.ceil((n_total - 1) / (win - 1)); step = math.ceil((n_total - 1) / k / 4) * 4
    wins = []
    for i in range(k):
        s = i * step; t = (s + step / 2) / fps
        cue = "open" if i == 0 else ("l0" if t < line_ends_s[0] else "l1" if t < line_ends_s[1] else "l2")
        wins.append({"start": s, "len": step + 1, "cue": cue})
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=49); ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=6.0); ap.add_argument("--aguidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=43); ap.add_argument("--sample", default="768,768")
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--vae-dev", default="mps")
    ap.add_argument("--line-ends", default="5.584,11.39"); ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--anchors", default="", help="comma list of n_windows+1 ORIGINAL frames: window k = anchor k -> anchor k+1 (no chaining drift)")
    a = ap.parse_args()
    out = Path(a.out).resolve(); wd = out.parent / (out.stem + "_win"); wd.mkdir(parents=True, exist_ok=True)
    img = str((ROOT / a.image).resolve()) if not os.path.isabs(a.image) else a.image
    aud = str((ROOT / a.audio).resolve()) if not os.path.isabs(a.audio) else a.audio
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", aud]))
    n_total = int(np.ceil(dur * 25))
    wins = plan_windows(n_total, a.win, [float(x) for x in a.line_ends.split(",")])
    if a.max_windows: wins = wins[:a.max_windows]
    if a.anchors:
        an = [str((ROOT / x).resolve()) if not os.path.isabs(x) else x for x in a.anchors.split(",")]
        assert len(an) >= len(wins) + 1, f"need {len(wins) + 1} anchors, got {len(an)}"
        for k, w in enumerate(wins):
            w["image"], w["end"] = an[k], an[k + 1]
    for k, w in enumerate(wins):
        w["prompt"] = BASE + CUES[w["cue"]]; w["seed"] = a.seed + k
    spec = {"audio": aud, "image": img, "windows": wins}
    sp = wd / "spec.json"; json.dump(spec, open(sp, "w"), indent=1, ensure_ascii=False)
    print(f"[emv3w] {dur:.3f}s -> {n_total} frames, {len(wins)} windows of <= {a.win}", flush=True)
    os.chdir(EM); sys.path.insert(0, str(EM)); patches(a.vae_dev)
    os.environ["EMV3_WINDOWS"] = str(sp)
    t0 = time.time()
    sys.argv = ["infer_flash_multi.py", "--image_path", img, "--audio_path", aud, "--prompt", wins[0]["prompt"], "--negative_prompt", NEG,
                "--num_inference_steps", str(a.steps), "--config_path", "config/config.yaml",
                "--model_name", str(EM / "models/Wan2.1-Fun-V1.1-1.3B-InP"),
                "--transformer_path", str(EM / "models/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"),
                "--save_path", str(wd), "--wav2vec_model_dir", str(EM / "models/wav2vec2-base-chinese"), "--sampler_name", "Flow_Unipc",
                "--video_length", str(a.win), "--guidance_scale", str(a.guidance), "--audio_guidance_scale", str(a.aguidance),
                "--audio_scale", "1.0", "--neg_scale", "1.0", "--neg_steps", "0", "--seed", str(a.seed), "--enable_teacache",
                "--teacache_threshold", "0.1", "--num_skip_start_steps", str(min(5, a.steps)), "--riflex_k", "6", "--weight_dtype", a.dtype,
                "--sample_size", *a.sample.split(","), "--fps", "25", "--shift", "5.0"]
    runpy.run_path(str(EM / "infer_flash_multi.py"), run_name="__main__")
    gen_s = time.time() - t0
    frames = []
    for k in range(len(wins)):
        f = np.load(wd / f"win{k:02d}.npy"); frames.extend(f if k == 0 else f[1:])
    frames = np.stack(frames); h, w = frames.shape[1:3]
    silent = Path(str(out.with_suffix("")) + "_silent.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes()); p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", aud, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", str(out)], check=True)
    wm = json.load(open(wd / "windows_meta.json"))
    meta = {"model": "EchoMimicV3-flash-pro (Wan2.1-Fun-V1.1-1.3B-InP + audio, chinese-wav2vec2), MPS", "args": vars(a), "frames": int(len(frames)),
            "seconds": round(len(frames) / 25, 3), "gen_total_s": round(gen_s, 1), "windows": wm, "negative_note": "flash CFG is audio-only; NEG unused", "anchored": bool(a.anchors)}
    json.dump(meta, open(str(out.with_suffix("")) + "_meta.json", "w"), indent=1, ensure_ascii=False)
    print("EMV3W_DONE", out, len(frames), round(gen_s, 1), flush=True)


if __name__ == "__main__":
    main()
