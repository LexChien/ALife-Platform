#!/usr/bin/env python3
"""yaying v6: refine the Mandarin character timing with CTC forced alignment (torchaudio MMS_FA, wav2vec2 trained on
1100+ languages, romanised input). Each character's tone-less pinyin is one 'word'; start/end come from its token spans.
Keeps Whisper times as start_whisper/end_whisper; recomputes onset_frame and the anticipatory shape_from_frame.
  ~/gen_cache/venv_gen/bin/python tools/yaying_v6_align_mms.py [--align in.json] [--audio 16k.wav] [--out out.json]"""
import argparse, json, re
import torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--align", default="runs/yaying_clone/demo_v6/audio/lines012_align.json")
ap.add_argument("--audio", default="runs/yaying_clone/demo_v6/audio/lines012_16k.wav")
ap.add_argument("--out", default="runs/yaying_clone/demo_v6/audio/lines012_align_mms.json")
a = ap.parse_args()
al = json.load(open(a.align)); fps = al.get("fps", 25); ant = al.get("anticip_frames", 2)
chars = [c for c in al["chars"] if c.get("pinyin")]
words = [re.sub(r"[^a-z]", "", c["pinyin"].lower().replace("ü", "u").replace("v", "u")) for c in chars]
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval(); tok = bundle.get_tokenizer(); aligner = bundle.get_aligner()
wav, sr = torchaudio.load(a.audio); wav = wav.mean(0, keepdim=True)
if sr != bundle.sample_rate: wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
with torch.inference_mode():
    em, _ = model(wav)
    spans = aligner(em[0], tok(words))
ratio = wav.shape[1] / em.shape[1] / bundle.sample_rate
for c, sp in zip(chars, spans):
    c["start_whisper"], c["end_whisper"] = c.get("start"), c.get("end")
    c["start"] = round(sp[0].start * ratio, 3); c["end"] = round(sp[-1].end * ratio, 3)
    c["score"] = round(float(sum(s.score * len(s) for s in sp) / sum(len(s) for s in sp)), 3)
    c["onset_frame"] = int(round(c["start"] * fps)); c["shape_from_frame"] = c["onset_frame"] - ant
al["aligner"] = "torchaudio MMS_FA (CTC forced alignment, tone-less pinyin)"
dev = [abs(c["start"] - c["start_whisper"]) for c in chars if c.get("start_whisper") is not None]
al["mms_vs_whisper_start_abs_dev_s"] = {"median": round(sorted(dev)[len(dev) // 2], 3), "max": round(max(dev), 3)}
al["mms_score_mean"] = round(sum(c["score"] for c in chars) / len(chars), 3)
json.dump(al, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({"chars": len(chars), "dev": al["mms_vs_whisper_start_abs_dev_s"], "score_mean": al["mms_score_mean"],
                  "first": [(c["c"], c["start"], c["end"]) for c in chars[:6]]}, ensure_ascii=False))
