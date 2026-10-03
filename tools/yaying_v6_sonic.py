#!/usr/bin/env python3
"""Plan 40 v6 candidate A: Sonic (Tencent, SVD-xt based, audio-driven global head/expression motion) on Apple MPS.
Reference = an ORIGINAL frame of her footage (full frame, no crop), audio = the approved F5 lines (full, untrimmed).
Third-party clone ~/gen_cache/Sonic (device line patched to MPS via SONIC_DEVICE). Weights: LeonJoe13/Sonic,
stabilityai/stable-video-diffusion-img2vid-xt (fp16 subset), openai/whisper-tiny.
  ~/gen_cache/venv_gen/bin/python tools/yaying_v6_sonic.py --image <png> --audio <wav> --out <mp4>
"""
import argparse, json, os, sys, time
from pathlib import Path

SONIC = Path.home() / "gen_cache/Sonic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=25); ap.add_argument("--dynamic-scale", type=float, default=1.0)
    ap.add_argument("--min-res", type=int, default=512); ap.add_argument("--seed", type=int, default=72589)
    a = ap.parse_args()
    img, aud, out = (str(Path(p).resolve()) for p in (a.image, a.audio, a.out))
    here = Path(__file__).resolve().parent
    os.chdir(SONIC); sys.path.insert(0, str(SONIC))
    import torch
    sys.path.insert(0, str(here)); import yaying_mps_sdpa; yaying_mps_sdpa.patch(2.0)
    from sonic import Sonic
    t0 = time.time()
    pipe = Sonic(0); t_load = time.time() - t0
    info = pipe.preprocess(img, expand_ratio=0.5); print(info, flush=True)
    t1 = time.time()
    rc = pipe.process(img, aud, out, min_resolution=a.min_res, inference_steps=a.steps, dynamic_scale=a.dynamic_scale, seed=a.seed)
    meta = {"model": "Sonic (SVD-xt + Sonic unet/audio2token/audio2bucket, RIFE x2)", "device": str(pipe.device), "rc": rc,
            "load_s": round(t_load, 1), "gen_s": round(time.time() - t1, 1), "args": vars(a), "torch": torch.__version__,
            "mps_peak_alloc_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if torch.backends.mps.is_available() else None}
    json.dump(meta, open(out.replace(".mp4", "_meta.json"), "w"), indent=1); print("SONIC_DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
