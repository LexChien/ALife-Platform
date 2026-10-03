#!/usr/bin/env python3
"""yaying v6: Wan2.2-TI2V-5B image-to-video candidate (Apple MPS, diffusers 0.35, venv_wan).

Muse-style motion plan, one prompt per voice line; segments chain on the previous segment's last frame
(or restart from an original frame with --chain 0). Output is silent 24 fps; the lip pass + mux happen later.
Real runs only; writes <out>_meta.json with load / per-segment timings.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ("A young East Asian woman with long black hair lies on white bedsheets in warm golden morning sunlight streaming "
        "through window blinds, close-up portrait, she speaks softly and intimately to the camera in a breathy near-whisper, "
        "natural skin texture, realistic, gentle slow motion, lingering pauses with soft visible breathing. ")
SEG = [
    "At first she glances slightly away from the camera, then slowly turns her head and locks calm, gentle eye contact with the viewer; "
    "her lips rest slightly parted and a soft smile appears.",
    "She blinks slowly, then looks up at the viewer through relaxed, lowered eyelids; she tilts her head gently, cheek toward her shoulder, "
    "with a soft smile between phrases; her hand stays still and relaxed.",
    "She leans slowly and deliberately toward the camera with a gentle head tilt, the camera pushes in slowly to a close-up of her face, "
    "eyes locked on the viewer, lips slightly parted, warm sunlight on her cheek.",
]
NEG = ("deformed fingers, extra fingers, fused fingers, merged fingers, twisted hands, distorted face, asymmetric eyes, blurry, "
       "flicker, jelly hair, morphing hair, nudity, explicit, cartoon, painting, text, subtitles, watermark, fast motion, camera shake, "
       "overexposed, low quality, static image")


def decode(pipe, lat):
    import torch
    pipe.transformer.to("cpu"); pipe.text_encoder.to("cpu"); torch.mps.empty_cache()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    v = pipe.vae
    lat = lat.to("mps", v.dtype)
    mean = torch.tensor(v.config.latents_mean).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    std = 1.0 / torch.tensor(v.config.latents_std).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    with torch.no_grad():
        video = v.decode(lat / std + mean, return_dict=False)[0]
    out = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    del video; torch.mps.empty_cache()
    pipe.transformer.to("mps"); pipe.text_encoder.to("mps")
    return out


def frames_for(sec, fps):
    n = int(np.ceil(sec * fps)); return ((n - 1 + 3) // 4) * 4 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--durations", default="5.584,5.806,6.01"); ap.add_argument("--segments", default="0,1,2")
    ap.add_argument("--height", type=int, default=704); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chain", type=int, default=1); ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--latents_only", type=int, default=1)
    ap.add_argument("--model", default=os.path.expanduser("~/gen_cache/wan22_ti2v5b"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "tools")); import yaying_mps_sdpa; yaying_mps_sdpa.patch(1.5)
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from diffusers.utils import export_to_video
    out = Path(a.out).resolve(); wd = out.parent / (out.stem + "_parts"); wd.mkdir(parents=True, exist_ok=True)
    dev = "mps"; t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    tdt = getattr(torch, a.dtype)
    from diffusers import WanTransformer3DModel
    tr = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=tdt)
    pipe = WanImageToVideoPipeline.from_pretrained(a.model, vae=vae, transformer=tr, torch_dtype=torch.bfloat16).to(dev)
    t_load = time.time() - t0; print(f"[wan22] loaded in {t_load:.1f}s", flush=True)
    meta = {"model": f"Wan2.2-TI2V-5B (diffusers WanImageToVideoPipeline, MPS, transformer {a.dtype}, T5 bf16, VAE fp32)", "args": vars(a), "load_s": t_load,
            "negative": NEG, "segments": []}
    durs = [float(x) for x in a.durations.split(",")]; segs = [int(x) for x in a.segments.split(",")]
    img = Image.open(a.image).convert("RGB").resize((a.width, a.height), Image.LANCZOS)
    allf = []
    for k in segs:
        nf = frames_for(durs[k], a.fps)
        if a.max_frames: nf = min(nf, a.max_frames)
        prompt = BASE + SEG[k]; ts = time.time()
        g = torch.Generator("cpu").manual_seed(a.seed + k)
        lat = pipe(image=img, prompt=prompt, negative_prompt=NEG, height=a.height, width=a.width, num_frames=nf,
                   num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g, output_type="latent").frames
        torch.save(lat.cpu(), wd / f"seg{k}_latents.pt")
        t_den = time.time() - ts
        if a.latents_only:
            meta["segments"].append({"seg": k, "prompt": prompt, "frames": nf, "denoise_s": t_den, "s_per_step": t_den / a.steps,
                                     "latents": str(wd / f"seg{k}_latents.pt")})
            json.dump(meta, open(str(out.with_suffix("")) + "_meta.json", "w"), indent=2, ensure_ascii=False)
            print(f"[wan22] seg{k}: latents saved, denoise {t_den:.1f}s", flush=True)
            continue
        res = decode(pipe, lat)            # VAE decode with the DiT/T5 parked on CPU (decode OOM'd at 40.8 GB cap otherwise)
        el = time.time() - ts
        fr = (np.clip(res, 0, 1) * 255).round().astype(np.uint8)
        export_to_video([f.astype(np.float32) / 255.0 for f in fr], str(wd / f"seg{k}.mp4"), fps=a.fps)
        print(f"[wan22] seg{k}: {nf} frames in {el:.1f}s ({el / a.steps:.1f}s/step)", flush=True)
        meta["segments"].append({"seg": k, "prompt": prompt, "frames": nf, "gen_s": el, "denoise_s": t_den, "s_per_step": t_den / a.steps})
        allf.extend(fr if not allf else fr[1:] if a.chain else fr)
        if a.chain: img = Image.fromarray(fr[-1])
        json.dump(meta, open(str(out.with_suffix("")) + "_meta.json", "w"), indent=2, ensure_ascii=False)
    export_to_video([f.astype(np.float32) / 255.0 for f in allf], str(out), fps=a.fps)
    meta["total_s"] = time.time() - t0; meta["frames_total"] = len(allf)
    json.dump(meta, open(str(out.with_suffix("")) + "_meta.json", "w"), indent=2, ensure_ascii=False)
    print("[wan22] done", out, len(allf), flush=True)


if __name__ == "__main__":
    main()
