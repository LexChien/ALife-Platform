#!/usr/bin/env python3
"""Plan 40 T4.2: score TTS candidates (real models only).

  ~/yaying_cache/venv/bin/python tools/yaying_tts_eval.py [--bench runs/yaying_clone/tts_bench]

Speaker similarity: cosine to the centroid of HELD-OUT real 雅英 clips (dataset segments not used as the TTS
reference), with (a) speechbrain ECAPA-TDNN voxceleb and (b) microsoft/wavlm-base-plus-sv x-vectors.
Intelligibility: Whisper large-v3 (mlx) -> OpenCC s2twp -> CER (zh) / WER (en) vs the input text.
Quality: UTMOS22 strong (SpeechMOS) and DNSMOS P.835 (sig/bak/ovr, Microsoft onnx).
Also scores the held-out real clips themselves (ceiling) with the same metrics."""
import argparse, json, re, string, sys
from pathlib import Path
import numpy as np, librosa, torch

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--bench", default="runs/yaying_clone/tts_bench")
ap.add_argument("--dataset", default="runs/yaying_clone/dataset.json"); ap.add_argument("--ref-info", default="runs/yaying_clone/tts_ref/ref.json")
ap.add_argument("--out", default="eval.json"); ap.add_argument("--cands", default="")
A = ap.parse_args(); B = ROOT / A.bench
bench = json.loads((B / "bench.json").read_text()); ds = json.loads((ROOT / A.dataset).read_text())
ref_info = json.loads((ROOT / A.ref_info).read_text())
held = [s for s in ds["segments"] if s.get("target") and s["wav"] not in ref_info["used_segments"] and s["dur"] >= 1.0]
print("held-out real clips", len(held), round(sum(s["dur"] for s in held), 1), "s")

from speechbrain.inference.speaker import EncoderClassifier
ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(Path.home() / "yaying_cache/ecapa"))
from transformers import AutoFeatureExtractor, WavLMForXVector
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
import onnxruntime as ort
dns = ort.InferenceSession(str(Path.home() / "yaying_cache/dnsmos/sig_bak_ovr.onnx"))
import mlx_whisper, opencc, jiwer
cc = opencc.OpenCC("s2twp")


def load16(p):
    y, _ = librosa.load(ROOT / p if not str(p).startswith("/") else p, sr=16000); return y.astype(np.float32)


def emb(y):
    with torch.no_grad():
        e1 = ecapa.encode_batch(torch.from_numpy(y[None])).squeeze().numpy()
        inp = fe(y, sampling_rate=16000, return_tensors="pt"); e2 = wavlm(**inp).embeddings.squeeze().numpy()
    return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)


def dnsmos(y):
    L = int(9.01 * 16000)
    if len(y) < L: y = np.tile(y, int(np.ceil(L / len(y))))[:L]
    outs = []
    for s in range(0, max(1, len(y) - L + 1), 16000):
        seg = y[s:s + L]
        if len(seg) < L: break
        sig, bak, ovr = dns.run(None, {"input_1": seg[None].astype(np.float32)})[0][0]
        # P.835 polynomial calibration (DNS-Challenge dnsmos_local.py, personalized=False)
        sig = -0.08397278 * sig ** 2 + 1.22083953 * sig + 0.0052439; bak = -0.13166888 * bak ** 2 + 1.60915514 * bak - 0.39604546
        ovr = -0.06766283 * ovr ** 2 + 1.11546468 * ovr + 0.04602535; outs.append((sig, bak, ovr))
    return [float(np.mean([o[i] for o in outs])) for i in range(3)]


def norm_zh(t):
    t = cc.convert(t); return re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE).lower()


def norm_en(t):
    return " ".join(t.lower().translate(str.maketrans("", "", string.punctuation)).split())


def score(path, text=None, lang="zh"):
    y = load16(path); e1, e2 = emb(y)
    with torch.no_grad(): u = float(utmos(torch.from_numpy(y[None]), 16000))
    sig, bak, ovr = dnsmos(y)
    r = {"ecapa": e1, "wavlm": e2, "utmos": u, "dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovr": ovr}
    if text is not None:
        for attempt in range(4):  # Metal "Command buffer execution failed" under heavy GPU load -> retry
            try:
                hyp = mlx_whisper.transcribe(str(ROOT / path), path_or_hf_repo="mlx-community/whisper-large-v3-mlx", language=lang,
                                             condition_on_previous_text=False)["text"]
                break
            except RuntimeError as exc:
                if attempt == 3: raise
                print("whisper retry", attempt, exc, flush=True); __import__("time").sleep(10)
        r["hyp"] = cc.convert(hyp).strip()
        r["cer" if lang == "zh" else "wer"] = (jiwer.cer(norm_zh(text), norm_zh(hyp)) if lang == "zh" else jiwer.wer(norm_en(text), norm_en(hyp)))
    return r


H = [score(str(ROOT / "runs/yaying_clone" / s["wav"])) for s in held]  # dataset wav paths are relative to runs/yaying_clone
c1 = np.mean([h["ecapa"] for h in H], 0); c1 /= np.linalg.norm(c1); c2 = np.mean([h["wavlm"] for h in H], 0); c2 /= np.linalg.norm(c2)
# ceiling: leave-one-out similarity of each held-out real clip to the centroid of the others
loo1 = [float(H[i]["ecapa"] @ (lambda c: c / np.linalg.norm(c))(np.sum([h["ecapa"] for j, h in enumerate(H) if j != i], 0))) for i in range(len(H))]
loo2 = [float(H[i]["wavlm"] @ (lambda c: c / np.linalg.norm(c))(np.sum([h["wavlm"] for j, h in enumerate(H) if j != i], 0))) for i in range(len(H))]
report = {"held_out": {"n": len(H), "dur_s": round(sum(s["dur"] for s in held), 2), "ecapa_loo_mean": round(float(np.mean(loo1)), 4),
                       "wavlm_loo_mean": round(float(np.mean(loo2)), 4), "utmos_mean": round(float(np.mean([h["utmos"] for h in H])), 3),
                       "dnsmos_ovr_mean": round(float(np.mean([h["dnsmos_ovr"] for h in H])), 3)}, "candidates": {}}
print("REAL", report["held_out"], flush=True)
for cand, res in bench.items():
    if A.cands and cand not in A.cands.split(","): continue
    rows = []
    for r in res["rows"]:
        if "error" in r: rows.append(r); continue
        s = score(r["wav"], r["text"], r["lang"])
        row = {**r, "sim_ecapa": round(float(s["ecapa"] @ c1), 4), "sim_wavlm": round(float(s["wavlm"] @ c2), 4),
               "utmos": round(s["utmos"], 3), "dnsmos_ovr": round(s["dnsmos_ovr"], 3), "dnsmos_sig": round(s["dnsmos_sig"], 3),
               "hyp": s["hyp"], **({"cer": round(s["cer"], 4)} if "cer" in s else {"wer": round(s["wer"], 4)})}
        rows.append(row); print(cand, {k: row[k] for k in ("lang", "i", "sim_ecapa", "sim_wavlm", "utmos", "dnsmos_ovr")}, row.get("cer", row.get("wer")), flush=True)
    ok = [r for r in rows if "error" not in r]; zh = [r for r in ok if r["lang"] == "zh"]; en = [r for r in ok if r["lang"] == "en"]
    m = lambda k, rr: round(float(np.mean([r[k] for r in rr])), 4) if rr else None
    p = lambda k, rr, q: round(float(np.percentile([r[k] for r in rr], q)), 3) if rr else None
    report["candidates"][cand] = {"info": res["info"], "n_ok": len(ok), "n_err": len(rows) - len(ok),
        "sim_ecapa_zh": m("sim_ecapa", zh), "sim_wavlm_zh": m("sim_wavlm", zh), "sim_ecapa_en": m("sim_ecapa", en),
        "cer_zh": m("cer", zh), "wer_en": m("wer", en), "utmos": m("utmos", ok), "dnsmos_ovr": m("dnsmos_ovr", ok),
        "latency_p50_s": p("latency_s", zh, 50), "latency_p90_s": p("latency_s", zh, 90), "rtf_mean": m("rtf", ok), "rows": rows}
(B / A.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
print("\n| cand | sim ECAPA zh | sim WavLM zh | CER zh | WER en | UTMOS | DNSMOS ovr | latency p50 | RTF |")
for c, v in report["candidates"].items():
    print(f"| {c} | {v['sim_ecapa_zh']} | {v['sim_wavlm_zh']} | {v['cer_zh']} | {v['wer_en']} | {v['utmos']} | {v['dnsmos_ovr']} | {v['latency_p50_s']} | {v['rtf_mean']} |")
print("WROTE", B / A.out)
