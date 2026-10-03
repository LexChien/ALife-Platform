#!/usr/bin/env python3
"""Plan 40 phase 2: post-enhance clone TTS output with DeepFilterNet3 (CPU) and register it as a new bench candidate
<cand>_df in tts_bench/bench.json. Latency fields include the measured enhancement time (head wav for first_audio_s).

  ~/yaying_cache/venv/bin/python tools/yaying_tts_enhance.py --cands f5q_nfe16,f5mlxnc_nfe4_s2"""
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, librosa, torch
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--cands", required=True); ap.add_argument("--bench", default="runs/yaying_clone/tts_bench")
A = ap.parse_args(); B = ROOT / A.bench
from df.enhance import enhance, init_df
model, st, _ = init_df(); torch.set_num_threads(4)


def enh(src, dst):
    y, sr = sf.read(src, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
    t = time.time(); z = librosa.resample(y, orig_sr=sr, target_sr=48000) if sr != 48000 else y
    e = enhance(model, st, torch.from_numpy(z[None].astype(np.float32)))[0].numpy()
    el = time.time() - t; sf.write(dst, e, 48000, subtype="PCM_16"); return el


bench = json.loads((B / "bench.json").read_text())
for cand in A.cands.split(","):
    src = bench[cand]; out = B / f"{cand}_df"; out.mkdir(exist_ok=True); rows = []
    for r in src["rows"]:
        if "error" in r: rows.append(r); continue
        p = ROOT / r["wav"]; q = out / p.name; el = enh(p, q); row = {**r, "wav": str(q.relative_to(ROOT)), "enhance_s": round(el, 3)}
        row["latency_s"] = round(r["latency_s"] + el, 3); row["rtf"] = round(row["latency_s"] / r["dur_s"], 3)
        if "first_audio_s" in r:
            h = p.with_name(p.stem + "_head.wav")
            row["first_audio_s"] = round(r["first_audio_s"] + (enh(h, out / h.name) if h.exists() else el), 3)
        rows.append(row); print(cand + "_df", {k: row.get(k) for k in ("lang", "i", "latency_s", "first_audio_s", "enhance_s")}, flush=True)
    bench = json.loads((B / "bench.json").read_text())  # re-read: other benches may have written meanwhile
    bench[cand + "_df"] = {"info": {**src["info"], "post": "DeepFilterNet3 48k CPU"}, "rows": rows}
    (B / "bench.json").write_text(json.dumps(bench, ensure_ascii=False, indent=1))
print("ENH_DONE")
