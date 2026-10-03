#!/usr/bin/env python3
"""Plan 40 v6 candidate B: EchoMimic v3 Flash (Ant Group; Wan2.1-Fun-1.3B-InP + audio cross-attention, chinese-wav2vec2)
= audio-driven + text-prompted generation of NEW whole-frame motion (head, eyes, hands, breathing) from ONE original
frame of her footage. Runs on Apple MPS: third-party clone ~/gen_cache/echomimic_v3 patched (device -> MPS, float64
-> float32 sinusoid, complex64 RoPE); SDPA chunked by tools/yaying_mps_sdpa.py (flash-attn unavailable on Mac).
Long audio is generated line by line; each next line starts from the previous segment's last frame (--chain) or
again from the original reference (--no-chain); segments are concatenated without re-timing.
  ~/gen_cache/venv_gen/bin/python tools/yaying_v6_emv3.py --image <png> --audios a.wav,b.wav --out <mp4>
"""
import argparse, json, os, runpy, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM = Path.home() / "gen_cache/echomimic_v3"
PROMPT = ("A young East Asian woman with long black hair lies on white bedsheets in warm golden morning sunlight streaming "
          "through window blinds, speaking softly and intimately to the camera in a breathy near-whisper. She first glances "
          "slightly away, then turns her eyes to lock gentle eye contact with the viewer, blinks slowly and looks up through "
          "relaxed lowered eyelids; between phrases her lips rest slightly parted with a soft smile. She tilts her head gently "
          "toward her shoulder, leans a little closer, and slowly brings her fingertips near her lips, pausing. Natural "
          "breathing, slow deliberate subtle movements, warm directional sunlight with soft shadows, photorealistic, same "
          "face and hairstyle, cinematic close shot.")
NEG = ("deformed fingers, extra fingers, fused fingers, merged fingers, twisted hands, distorted face, asymmetric eyes, blurry, "
       "flicker, jelly hair, morphing, nudity, explicit, cartoon, painting, text, watermark, fast motion, camera shake")


def _cpu_vae_patch():
    """MPS conv3d in the Wan VAE OOMs (81.6 GB cap hit on the masked-video encode), so run VAE encode/decode on CPU in fp32."""
    import copy, torch
    import src.wan_vae as wv
    def _cpu(self):
        if getattr(self, "_cpu_model", None) is None:
            self._cpu_model = copy.deepcopy(self.model).float().cpu().eval()
        return self._cpu_model
    def _encode(self, x):
        m = _cpu(self); sc = [t.float().cpu() if hasattr(t, "cpu") else t for t in self.scale]
        with torch.no_grad():
            h = torch.stack([m.encode(u.unsqueeze(0).float().cpu(), sc).squeeze(0) for u in x])
        if x.device.type == "mps": torch.mps.empty_cache()
        return h.to(x.device, x.dtype)
    def _decode(self, zs):
        m = _cpu(self); sc = [t.float().cpu() if hasattr(t, "cpu") else t for t in self.scale]
        with torch.no_grad():
            d = torch.stack([m.decode(u.unsqueeze(0).float().cpu(), sc).clamp_(-1, 1).squeeze(0) for u in zs])
        return wv.DecoderOutput(sample=d.to(zs.device, zs.dtype))
    wv.AutoencoderKLWan._encode = _encode; wv.AutoencoderKLWan._decode = _decode
    print("[emv3] VAE encode/decode patched to CPU fp32", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--audios", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8); ap.add_argument("--sample", default="768,768")
    ap.add_argument("--guidance", type=float, default=6.0); ap.add_argument("--aguidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=43); ap.add_argument("--max-frames", type=int, default=161)
    ap.add_argument("--chain", type=int, default=1); ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    out = Path(a.out).resolve(); wd = out.parent / (out.stem + "_parts"); wd.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import yaying_mps_sdpa; yaying_mps_sdpa.patch(1.5)
    os.chdir(EM); sys.path.insert(0, str(EM))
    if os.environ.get("EMV3_CPU_VAE", "1") == "1":
        _cpu_vae_patch()
    ref = str(Path(ROOT / a.image).resolve()) if not os.path.isabs(a.image) else a.image
    parts, meta = [], {"model": "EchoMimicV3-flash-pro (Wan2.1-Fun-V1.1-1.3B-InP base)", "segments": [], "prompt": a.prompt, "negative": NEG, "args": vars(a)}
    t_all = time.time()
    for k, aud in enumerate(a.audios.split(",")):
        aud = str((ROOT / aud).resolve()) if not os.path.isabs(aud) else aud
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", aud]))
        vl = min(a.max_frames, int(dur * 25) // 4 * 4 + 1 + 4)
        sd = wd / f"seg{k}"; sd.mkdir(exist_ok=True)
        t0 = time.time()
        sys.argv = ["infer_flash.py", "--image_path", ref, "--audio_path", aud, "--prompt", a.prompt, "--negative_prompt", NEG,
                    "--num_inference_steps", str(a.steps), "--config_path", "config/config.yaml",
                    "--model_name", str(EM / "models/Wan2.1-Fun-V1.1-1.3B-InP"),
                    "--transformer_path", str(EM / "models/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"),
                    "--save_path", str(sd), "--wav2vec_model_dir", str(EM / "models/wav2vec2-base-chinese"), "--sampler_name", "Flow_Unipc",
                    "--video_length", str(vl), "--guidance_scale", str(a.guidance), "--audio_guidance_scale", str(a.aguidance),
                    "--audio_scale", "1.0", "--neg_scale", "1.0", "--neg_steps", "0", "--seed", str(a.seed + k), "--enable_teacache",
                    "--teacache_threshold", "0.1", "--num_skip_start_steps", str(min(5, a.steps)), "--riflex_k", "6", "--weight_dtype", os.environ.get("EMV3_DTYPE", "float32"),
                    "--sample_size", *a.sample.split(","), "--fps", "25", "--shift", "5.0"]
        sys.modules.pop("infer_flash", None)
        runpy.run_path(str(EM / "infer_flash.py"), run_name="__main__")
        vid = next(sd.glob("*_output.mp4"))
        parts.append(vid)
        meta["segments"].append({"audio": aud, "audio_s": round(dur, 3), "video_length": vl, "ref": ref, "gen_s": round(time.time() - t0, 1), "video": str(vid)})
        print("SEGMENT_DONE", meta["segments"][-1], flush=True)
        if a.chain:
            last = sd / "last.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.05", "-i", str(vid), "-frames:v", "1", str(last)], check=True)
            ref = str(last)
    lst = wd / "list.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    meta["total_gen_s"] = round(time.time() - t_all, 1)
    json.dump(meta, open(str(out).replace(".mp4", "_meta.json"), "w"), indent=1, ensure_ascii=False)
    print("EMV3_DONE", json.dumps(meta, ensure_ascii=False)[:600], flush=True)


if __name__ == "__main__":
    main()
