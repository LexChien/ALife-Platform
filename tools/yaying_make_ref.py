#!/usr/bin/env python3
"""Plan 40 T4.0: pick the zero-shot TTS reference clip from the clean 雅英 dataset (rest = held-out for scoring).

Score per target segment = z(snr_wada) + z(avg_word_prob) + z(cos_to_target); greedily take the best segments of ONE
video (contiguous order kept) until 5-11 s, joined with 0.25 s silence. Writes tts_ref/ref.wav (24 kHz), ref.txt, ref.json."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "runs/yaying_clone"
ds = json.loads((W / "dataset.json").read_text()); T = [s for s in ds["segments"] if s.get("target")]
z = lambda k: (lambda v: (v - v.mean()) / (v.std() + 1e-9))(np.array([s[k] for s in T], float))
# WADA-SNR saturates (~97 dB) on DeepFilterNet output -> not informative; use separation SNR + duration instead
sc = z("snr_sep_db") + z("avg_word_prob") + 2 * z("cos_to_target") + z("dur")
for s, v in zip(T, sc): s["ref_score"] = float(v)
best = None
for vid in sorted({s["video"] for s in T}):
    cand = sorted([s for s in T if s["video"] == vid and 1.0 <= s["dur"] <= 11.5], key=lambda s: -s["ref_score"])
    pick, tot = [], 0.0
    for s in cand:
        if tot + s["dur"] > 11.5: continue
        pick.append(s); tot += s["dur"]
        if tot >= 7.0: break
    if tot >= 5.0:
        q = float(np.mean([s["ref_score"] for s in pick]))
        if best is None or q > best[0]: best = (q, vid, sorted(pick, key=lambda s: s["start"]), tot)
if best is None:
    print("no video has >=5 s of good target speech", file=sys.stderr); sys.exit(1)
q, vid, pick, tot = best
sil = np.zeros(int(0.25 * 24000), np.float32); parts = []
for s in pick:
    y, sr = sf.read(W / s["wav"]); assert sr == 24000; parts += [y.astype(np.float32), sil]
out = W / "tts_ref"; out.mkdir(exist_ok=True)
y = np.concatenate(parts[:-1]); sf.write(out / "ref.wav", y, 24000, subtype="PCM_16")
text = "，".join(s["text"].strip("，。 ") for s in pick) + "。"
# documented manual fix: Whisper writes her name 李雅英 as the homophone-ish 李雅音 (proper-name error only)
whisper_text = text; text = text.replace("李雅音", "李雅英")
(out / "ref.txt").write_text(text, encoding="utf-8")
info = {"video": vid, "used_segments": [s["wav"] for s in pick], "dur_s": round(len(y) / 24000, 2), "text": text, "mean_ref_score": round(q, 3), "whisper_text": whisper_text}
(out / "ref.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print(json.dumps(info, ensure_ascii=False))
