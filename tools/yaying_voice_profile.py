#!/usr/bin/env python3
"""Plan 40 T3: '本質語氣' voice-style profile of 雅英 from the clean dataset (measured, no guesses).

  ~/yaying_cache/venv/bin/python tools/yaying_voice_profile.py  -> runs/yaying_clone/voice_profile.json + voice_profile.md

F0: librosa pYIN (65-700 Hz) per target segment -> Hz + semitone stats, range, per-segment slope (declination),
final-200 ms movement. Rate: syllables/s (Han char = 1 syllable, Latin words via vowel groups) over speech time and
over phonation time (articulation rate from Whisper word timestamps). Pauses: inter-segment gaps (VAD) + intra-
segment word gaps >= 150 ms. Energy: frame RMS dB stats / dynamic range / 4 Hz-band modulation. Spectral tilt: LTAS
regression slope dB/octave 100-5000 Hz; breathiness: H1-H2 (dB, F0-normalised harmonics) + CPP (dB). SER:
emotion2vec_plus_large (FunASR) per segment if installed, else superb wav2vec2 ER. Phrases: char n-grams, particles, fillers."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, librosa

ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "runs/yaying_clone"
ds = json.loads((W / "dataset.json").read_text()); T = [s for s in ds["segments"] if s.get("target")]
st = lambda x: {k: round(float(v), 3) for k, v in dict(mean=np.mean(x), std=np.std(x), p05=np.percentile(x, 5), p50=np.median(x),
                                                        p95=np.percentile(x, 95), min=np.min(x), max=np.max(x)).items()} if len(x) else None
f0_all, slopes, finals, rms_all, mods, tilts, h1h2, cpps = [], [], [], [], [], [], [], []
for s in T:
    y, sr = librosa.load(W / s["wav"], sr=16000)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr, frame_length=1024, hop_length=160)
    v = f0[~np.isnan(f0)]
    if len(v) > 10:
        f0_all.append(v); semi = 12 * np.log2(v / 100.0); tt = np.arange(len(semi)) * 0.01
        slopes.append(np.polyfit(tt, semi, 1)[0]); finals.append(semi[-20:].mean() - semi[-40:-20].mean() if len(semi) >= 40 else 0)
        # H1-H2 on voiced frames
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=160)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        for k in np.where(~np.isnan(f0))[0][::3]:
            if k >= S.shape[1]: break
            g = f0[k]; b1 = np.argmin(np.abs(fr - g)); b2 = np.argmin(np.abs(fr - 2 * g))
            a1 = S[max(0, b1 - 2):b1 + 3, k].max(); a2 = S[max(0, b2 - 2):b2 + 3, k].max()
            h1h2.append(20 * np.log10((a1 + 1e-9) / (a2 + 1e-9)))
        # CPP (Hillenbrand-style, per voiced frame, 60-700 Hz quefrency window)
        frames = librosa.util.frame(np.pad(y, 512), frame_length=1024, hop_length=160)
        for k in np.where(~np.isnan(f0))[0][::3]:
            if k >= frames.shape[1]: break
            x = frames[:, k] * np.hanning(1024); spec = 20 * np.log10(np.abs(np.fft.rfft(x, 2048)) + 1e-9)
            cep = 20 * np.log10(np.abs(np.fft.irfft(spec)) + 1e-9)[:1024]; q = np.arange(1024) / sr
            m = (q >= 1 / 700) & (q <= 1 / 60); qi = np.where(m)[0]; pk = qi[np.argmax(cep[m])]
            line = np.polyval(np.polyfit(q[qi], cep[qi], 1), q[pk]); cpps.append(cep[pk] - line)
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]; db = 20 * np.log10(r + 1e-6); act = db[db > db.max() - 40]
    rms_all.append(act)
    env = r - r.mean(); spec = np.abs(np.fft.rfft(env)); fq = np.fft.rfftfreq(len(env), 0.01); band = (fq > 1) & (fq < 12)
    if band.any(): mods.append(float(fq[band][np.argmax(spec[band])]))
    L = np.mean(np.abs(librosa.stft(y, n_fft=1024)) ** 2, 1); f = librosa.fft_frequencies(sr=sr, n_fft=1024); m = (f >= 100) & (f <= 5000)
    tilts.append(np.polyfit(np.log2(f[m]), 10 * np.log10(L[m] + 1e-12), 1)[0])
F0 = np.concatenate(f0_all); semi = 12 * np.log2(F0 / 100)
# rate + pauses from whisper words
def syl(text):
    han = len(re.findall(r"[\u4e00-\u9fff]", text)); lat = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in re.findall(r"[A-Za-z]+", text))
    hang = len(re.findall(r"[\uac00-\ud7a3]", text)); return han + lat + hang
speech_t = sum(s["dur"] for s in T); n_syl = sum(syl(s["text"]) for s in T)
word_gaps, phon_t = [], 0.0
for vid in sorted({s["video"] for s in T}):
    wr = json.loads((W / "voice" / f"whisper_{vid}.json").read_text()); words = [w for sg in wr["segments"] for w in sg.get("words", [])]
    segs = [s for s in T if s["video"] == vid]
    for s in segs:
        ws = [w for w in words if w["start"] < s["end"] and w["end"] > s["start"]]
        phon_t += sum(min(w["end"], s["end"]) - max(w["start"], s["start"]) for w in ws)
        word_gaps += [b["start"] - a["end"] for a, b in zip(ws, ws[1:]) if b["start"] - a["end"] >= 0.15]
seg_gaps = []
for vid in sorted({s["video"] for s in ds["segments"]}):
    ss = sorted([s for s in ds["segments"] if s["video"] == vid], key=lambda s: s["start"])
    seg_gaps += [b["start"] - a["end"] for a, b in zip(ss, ss[1:]) if a.get("target") and b.get("target")]
pauses = np.array(word_gaps + seg_gaps)
# SER
ser = {"model": None}
try:
    from funasr import AutoModel
    em = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)
    labs = Counter(); probs = []
    for s in T:
        r = em.generate(str(W / s["wav"]), granularity="utterance", extract_embedding=False)[0]
        names = [l.split("/")[-1] for l in r["labels"]]; sc = np.array(r["scores"]); labs[names[int(np.argmax(sc))]] += s["dur"]; probs.append(dict(zip(names, sc)))
    tot = sum(labs.values()); ser = {"model": "emotion2vec_plus_large (FunASR, hf)", "duration_weighted_top_label": {k: round(v / tot, 3) for k, v in labs.most_common()},
                                    "mean_probs": {k: round(float(np.mean([p.get(k, 0) for p in probs])), 3) for k in probs[0]}}
except Exception as exc:
    ser["error_emotion2vec"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from transformers import pipeline
        clf = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er"); labs = Counter()
        for s in T:
            y, _ = librosa.load(W / s["wav"], sr=16000); top = clf(y, top_k=1)[0]; labs[top["label"]] += s["dur"]
        tot = sum(labs.values()); ser.update({"model": "superb/wav2vec2-base-superb-er (IEMOCAP 4-class, English-trained)",
                                              "duration_weighted_top_label": {k: round(v / tot, 3) for k, v in labs.most_common()}})
    except Exception as exc2:
        ser["error_fallback"] = f"{type(exc2).__name__}: {exc2}"[:300]
# phrases
texts = [s["text"] for s in T]; joined = "".join(texts)
PART = ["呢", "喔", "哦", "啦", "耶", "嘛", "吧", "欸", "啊", "呀", "哇", "嗯", "齁", "捏", "唷", "啊"]
FILL = ["嗯", "那個", "就是", "然後", "對", "真的", "好", "我們", "大家", "謝謝", "非常", "很", "可以"]
ngr = Counter(); [ngr.update(t[i:i + n] for i in range(len(t) - n + 1)) for t in texts for n in (2, 3, 4)]
ngr = Counter({k: v for k, v in ngr.items() if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]+", k) and v >= 2})
final_particles = Counter(re.sub(r"[，。！？!?、 ~～]+$", "", t)[-1:] for t in re.split(r"[，。！？!?]", joined) if t.strip())
prof = {
    "speaker": "雅英", "consent": ds.get("consent"), "n_segments": len(T), "speech_s": round(speech_t, 2),
    "f0_hz": st(F0), "f0_semitones_re_100hz": st(semi), "f0_range_semitones_p05_p95": round(float(np.percentile(semi, 95) - np.percentile(semi, 5)), 2),
    "f0_contour": {"segment_slope_semitones_per_s": st(np.array(slopes)), "final_200ms_move_semitones": st(np.array(finals)),
                   "rising_final_fraction": round(float(np.mean(np.array(finals) > 0.5)), 3), "falling_final_fraction": round(float(np.mean(np.array(finals) < -0.5)), 3)},
    "speaking_rate": {"syllables": n_syl, "syl_per_s_speech_time": round(n_syl / speech_t, 3), "articulation_syl_per_s_phonation": round(n_syl / max(phon_t, 1e-6), 3)},
    "pauses_s": {"n": int(len(pauses)), **(st(pauses) or {}), "n_word_gaps": len(word_gaps), "n_segment_gaps": len(seg_gaps),
                 "hist_bins_s": [0.15, 0.3, 0.5, 1.0, 2.0, 99], "hist": np.histogram(pauses, [0.15, 0.3, 0.5, 1.0, 2.0, 99])[0].tolist() if len(pauses) else []},
    "energy_db": {**st(np.concatenate(rms_all)), "dynamic_range_p95_p05": round(float(np.percentile(np.concatenate(rms_all), 95) - np.percentile(np.concatenate(rms_all), 5)), 2),
                  "envelope_modulation_peak_hz": st(np.array(mods))},
    "spectral_tilt_db_per_octave_100_5000": st(np.array(tilts)), "breathiness": {"h1_h2_db": st(np.array(h1h2)), "cpp_db": st(np.array(cpps))},
    "ser": ser,
    "phrases": {"top_ngrams": ngr.most_common(25), "sentence_final_chars": final_particles.most_common(12),
                "particle_counts": {p: joined.count(p) for p in dict.fromkeys(PART) if joined.count(p)},
                "filler_counts": {f: joined.count(f) for f in FILL if joined.count(f)},
                "latin_tokens": Counter(re.findall(r"[A-Za-z]+", joined)).most_common(10), "hangul_chars": len(re.findall(r"[\uac00-\ud7a3]", joined))},
    "transcripts": [{"video": s["video"], "start": s["start"], "text": s["text"]} for s in T],
}
(W / "voice_profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in prof.items() if k != "transcripts"}, ensure_ascii=False, indent=1))
