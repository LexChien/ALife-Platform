#!/usr/bin/env python3
"""Plan 38 E-VAD / E-WAKE (SYNTHETIC audio).
VAD: silero-vad vs webrtcvad on constructed streams (noise gap + say clip + gap ...), clean-ish / white 15 dB / white 5 dB.
WAKE: openWakeWord pretrained 'hey_jarvis' on say-generated positives vs all corpus clips + confusers.
Writes runs/plan38/vad/vad_bench.json and runs/plan38/wake/wake_bench.json"""
import json, time, subprocess, tempfile, os
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
SR = 16000
man = json.loads((ROOT / "runs/plan38/audio/manifest.json").read_text())

def clip(path):
    x, sr = sf.read(ROOT / path, dtype="float32"); return x

def true_region(x):
    e = np.abs(x); thr = 0.02 * e.max(); idx = np.where(e > thr)[0]
    return idx[0] / SR, idx[-1] / SR

def build_stream(clips, snr, seed):
    rng = np.random.default_rng(seed); parts = []; segs = []; t = 0.0
    speech_pow = np.mean(np.concatenate(clips) ** 2)
    noise_sd = np.sqrt(speech_pow / 10 ** (snr / 10)) if snr is not None else 1e-4
    for c in clips:
        gap = np.zeros(int(SR * rng.uniform(0.8, 1.6)), dtype="float32"); parts.append(gap); t += len(gap) / SR
        s, e = true_region(c); segs.append((t + s, t + e)); parts.append(c); t += len(c) / SR
    parts.append(np.zeros(int(SR * 1.5), dtype="float32"))
    x = np.concatenate(parts)
    x = x + rng.normal(0, noise_sd, x.shape).astype("float32")
    return np.clip(x, -1, 1), segs

def frame_labels(segs, n_frames, hop):
    y = np.zeros(n_frames, bool)
    for s, e in segs: y[int(s / hop): int(e / hop) + 1] = True
    return y

def score(pred_segs, segs, total, hop=0.03):
    n = int(total / hop) + 1
    y = frame_labels(segs, n, hop); p = frame_labels(pred_segs, n, hop)
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    onsets = []; offsets = []
    for s, e in segs:
        ov = [ps for ps in pred_segs if ps[1] > s and ps[0] < e]
        if ov: onsets.append(ov[0][0] - s); offsets.append(ov[-1][1] - e)
    return {"frame_precision": round(float(prec), 3), "frame_recall": round(float(rec), 3),
            "f1": round(float(2 * prec * rec / max(1e-9, prec + rec)), 3),
            "utterances_detected": f"{len(onsets)}/{len(segs)}", "n_pred_segments": len(pred_segs),
            "onset_err_ms_median": round(float(np.median(onsets) * 1000), 1) if onsets else None,
            "offset_err_ms_median": round(float(np.median(offsets) * 1000), 1) if offsets else None}

def silero_segments(x, model, get_ts):
    import torch
    t0 = time.perf_counter()
    ts = get_ts(torch.from_numpy(x), model, sampling_rate=SR, min_silence_duration_ms=300, speech_pad_ms=30, threshold=0.5)
    dt = time.perf_counter() - t0
    return [(d["start"] / SR, d["end"] / SR) for d in ts], dt

def webrtc_segments(x, aggr, hang_ms=300):
    import webrtcvad
    v = webrtcvad.Vad(aggr); fl = int(0.03 * SR); pcm = (x * 32767).astype("<i2")
    t0 = time.perf_counter(); flags = []
    for i in range(0, len(pcm) - fl, fl): flags.append(v.is_speech(pcm[i:i + fl].tobytes(), SR))
    dt = time.perf_counter() - t0
    segs = []; start = None; sil = 0; hang = hang_ms // 30
    for k, f in enumerate(flags):
        if f:
            if start is None: start = k
            sil = 0
        elif start is not None:
            sil += 1
            if sil > hang: segs.append((start * 0.03, (k - sil + 1) * 0.03)); start = None; sil = 0
    if start is not None: segs.append((start * 0.03, len(flags) * 0.03))
    segs = [s for s in segs if s[1] - s[0] >= 0.12]
    return segs, dt, len(flags)

def vad_bench():
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    clips = [clip(c["path"]) for c in man[::4]][:20]
    out = {"n_utterances": len(clips), "source": "SYNTHETIC say clips + synthetic white noise", "conditions": {}}
    for name, snr in (("quiet", None), ("white15db", 15), ("white5db", 5)):
        x, segs = build_stream(clips, snr, seed=7); total = len(x) / SR
        ps, dt = silero_segments(x, model, get_speech_timestamps)
        row = {"stream_s": round(total, 1), "silero": {**score(ps, segs, total), "cpu_s": round(dt, 3), "rtf": round(dt / total, 4)}}
        for ag in (1, 3):
            ws, dt2, nf = webrtc_segments(x, ag)
            row[f"webrtc_aggr{ag}"] = {**score(ws, segs, total), "cpu_s": round(dt2, 3), "us_per_30ms_frame": round(dt2 / nf * 1e6, 1)}
        # pure noise false-positive test (30 s)
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.003 if snr is None else 0.02 * (2 if snr == 5 else 1), SR * 30).astype("float32")
        ps_n, _ = silero_segments(noise, model, get_speech_timestamps); ws_n, _, _ = webrtc_segments(noise, 3)
        row["noise_only_30s_false_segments"] = {"silero": len(ps_n), "webrtc_aggr3": len(ws_n)}
        out["conditions"][name] = row; print(name, json.dumps(row)[:600])
    p = ROOT / "runs/plan38/vad/vad_bench.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))

def say_wav(voice, text, rate=None):
    d = tempfile.mkdtemp(); a = Path(d) / "x.aiff"; w = Path(d) / "x.wav"
    cmd = ["say", "-v", voice, "-o", str(a)] + (["-r", str(rate)] if rate else []) + [text]
    subprocess.run(cmd, check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(w)], check=True)
    x, _ = sf.read(w, dtype="float32"); return x

def oww_scores(model, x):
    model.reset(); pcm = (np.concatenate([np.zeros(SR // 2, "float32"), x, np.zeros(SR, "float32")]) * 32767).astype(np.int16)
    sc = []; t0 = time.perf_counter()
    for i in range(0, len(pcm) - 1280, 1280):
        p = model.predict(pcm[i:i + 1280]); sc.append(max(p.values()))
    return np.array(sc), time.perf_counter() - t0, len(sc)

def wake_bench():
    import openwakeword
    from openwakeword.model import Model
    openwakeword.utils.download_models(model_names=["hey_jarvis"])
    m = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    voices = ["Samantha", "Daniel", "Fred", "Kathy", "Albert", "Ralph", "Flo (英文（美國）)", "Shelley (英文（美國）)",
              "Sandy (英文（美國）)", "Eddy (英文（美國）)", "Reed (英文（美國）)", "Rocko (英文（美國）)", "Grandma (英文（美國）)", "Meijia", "Flo (中文（台灣）)"]
    pos = []; rng = np.random.default_rng(3)
    for v in voices:
        for rate in (None, 220):
            for text in ("Hey Jarvis", "Hey Jarvis, what's the status?"):
                x = say_wav(v, text, rate)
                for cond in ("clean", "white15db"):
                    xx = x if cond == "clean" else x + rng.normal(0, np.sqrt(np.mean(x ** 2) / 10 ** 1.5), x.shape).astype("float32")
                    sc, dt, n = oww_scores(m, xx)
                    pos.append({"voice": v, "rate": rate, "text": text, "cond": cond, "max": round(float(sc.max()), 3),
                                "first_frame_over_0.5": int(np.argmax(sc > 0.5)) if (sc > 0.5).any() else None})
    neg = []; neg_dur = 0.0
    confusers = ["Hey Travis", "Hey Jarvis is a movie character", "Hey Charles", "Harvest", "Hey, are you there?", "嘿 賈維斯", "Hey service"]
    for c in man:
        x = clip(c["path"]); sc, dt, n = oww_scores(m, x); neg_dur += len(x) / SR + 1.5
        neg.append({"src": c["path"], "max": round(float(sc.max()), 3)})
    conf = []
    for t in confusers:
        for v in ("Samantha", "Daniel", "Meijia"):
            x = say_wav(v, t); sc, _, _ = oww_scores(m, x); conf.append({"text": t, "voice": v, "max": round(float(sc.max()), 3)})
    # per-frame cost
    sc, dt, n = oww_scores(m, np.zeros(SR * 10, "float32"))
    out = {"model": "openWakeWord hey_jarvis (pretrained, onnx)", "source": "SYNTHETIC say", "per_80ms_frame_ms": round(dt / n * 1000, 3),
           "positives": pos, "negatives_corpus": neg, "confusers": conf}
    for thr in (0.3, 0.5, 0.7):
        out[f"thr_{thr}"] = {"tpr": round(float(np.mean([p["max"] > thr for p in pos])), 3),
                             "tpr_clean": round(float(np.mean([p["max"] > thr for p in pos if p["cond"] == "clean"])), 3),
                             "false_accepts_corpus": int(sum(n_["max"] > thr for n_ in neg)), "corpus_minutes": round(neg_dur / 60, 2),
                             "false_accepts_confusers": [c["text"] + "/" + c["voice"] for c in conf if c["max"] > thr]}
        print(thr, out[f"thr_{thr}"])
    p = ROOT / "runs/plan38/wake/wake_bench.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    import sys
    if "vad" in sys.argv[1:] or len(sys.argv) == 1: vad_bench()
    if "wake" in sys.argv[1:] or len(sys.argv) == 1: wake_bench()
