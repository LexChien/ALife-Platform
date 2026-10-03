#!/usr/bin/env python3
"""yaying v6: decode saved Wan2.2-TI2V-5B latents (seg*_latents.pt) in a separate process with ONLY the VAE loaded.
In-process decode OOM'd twice on MPS (MPSGraph 'other allocations' grow to ~26 GB during the Wan VAE decode).
Default CPU fp32, untiled (MPS decode OOMs; tiled decode gave inverted colours). Writes seg<k>.mp4 (24 fps) and prints timings.
  ~/gen_cache/venv_wan/bin/python tools/yaying_v6_wan22_decode.py --latents dir/seg0_latents.pt --out dir/seg0.mp4 [--dev mps|cpu]"""
import argparse, os, time
import numpy as np, torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--latents", required=True); ap.add_argument("--out", required=True); ap.add_argument("--dev", default="cpu")
ap.add_argument("--fps", type=int, default=24); ap.add_argument("--model", default=os.path.expanduser("~/gen_cache/wan22_ti2v5b"))
a = ap.parse_args()
vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).eval()
vp = VideoProcessor(vae_scale_factor=16)
lat = torch.load(a.latents)


def run(dev):
    v = vae.to(dev); l = lat.to(dev, torch.float32)
    # NOTE: no enable_tiling() - diffusers 0.35.1 tiled decode of the Wan2.2 (z48) VAE produced inverted/posterised colours
    mean = torch.tensor(v.config.latents_mean).view(1, v.config.z_dim, 1, 1, 1).to(dev)
    std = 1.0 / torch.tensor(v.config.latents_std).view(1, v.config.z_dim, 1, 1, 1).to(dev)
    with torch.no_grad():
        return v.decode(l / std + mean, return_dict=False)[0].cpu()


t = time.time()
try:
    video = run(a.dev); used = a.dev
except RuntimeError as e:
    print("decode on", a.dev, "failed:", str(e)[:160], "-> cpu", flush=True)
    torch.mps.empty_cache(); video = run("cpu"); used = "cpu"
fr = ((video[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()   # C,T,H,W -> T,H,W,C
export_to_video([f.astype(np.float32) / 255.0 for f in fr], a.out, fps=a.fps)  # export_to_video multiplies np frames by 255
np.save(a.out.rsplit(".", 1)[0] + ".npy", fr)
print(f"DECODED {a.out} frames={len(fr)} dev={used} {time.time() - t:.1f}s", flush=True)
