"""Run real diffusion in a short-lived process so weights are released."""
import hashlib
import json
from pathlib import Path
import sys
import time


def generate(request):
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    start = time.monotonic()
    cfg = request["config"]
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA image generation but CUDA is unavailable")
    torch.set_num_threads(int(cfg.get("threads", 2)))
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    path = cfg.get("model_path", "models/segmind_tiny_sd")
    pipe = StableDiffusionPipeline.from_pretrained(
        path, torch_dtype=dtype, local_files_only=True,
        use_safetensors=False, low_cpu_mem_usage=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.to(device)
    seed = int(cfg.get("seed", 42))
    size = int(request.get("size", 256))
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(request["prompt"], height=size, width=size,
                      num_inference_steps=int(cfg.get("steps", 20)),
                      guidance_scale=float(cfg.get("guidance_scale", 7.5)),
                      generator=generator)
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output)
    manifest = Path(path) / "provenance.json"
    return {"status": "generated", "backend": "diffusers", "model_path": path,
            "model": json.loads(manifest.read_text()) if manifest.exists() else {"verified": False},
            "device": device, "seed": seed, "steps": int(cfg.get("steps", 20)),
            "width": size, "height": size, "path": str(output),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "elapsed_seconds": time.monotonic() - start,
            "nsfw_content_detected": result.nsfw_content_detected}


if __name__ == "__main__":
    print(json.dumps(generate(json.load(sys.stdin))))
