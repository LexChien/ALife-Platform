#!/usr/bin/env python3
"""Plan 40 T2: 雅英 clean-speech dataset from the voice videos (all real models, resumable per stage).

  ~/yaying_cache/venv/bin/python tools/yaying_voice_extract.py [--work runs/yaying_clone]

Stages (outputs under <work>/voice/):
  1 ffmpeg 44.1 kHz stereo wav            raw_wav/<stem>.wav
  2 Demucs htdemucs_ft vocal separation   demucs/htdemucs_ft/<stem>/{vocals,no_vocals}.wav
  3 WPE dereverb (nara_wpe) + DeepFilterNet3 denoise -> 48 kHz -> 24 kHz mono   enh/<stem>.wav
  4 Silero VAD + Whisper large-v3 (mlx) transcripts, segment level
  5 ECAPA-TDNN (speechbrain voxceleb) embeddings per segment, agglomerative cosine clustering across videos;
    target speaker = cluster with the most speech present in the most videos (雅英 is the only speaker common to all)
  6 dataset: clean/<seg>.wav (24 kHz), dataset.json (segments, text, speaker, SNR: WADA on enhanced audio and
    vocals-vs-accompaniment energy ratio from Demucs on the original mix)."""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser(); ap.add_argument("--work", default="runs/yaying_clone"); ap.add_argument("--thr", type=float, default=0.55)
ap.add_argument("--wthr", type=float, default=0.70)
A = ap.parse_args(); W = Path(A.work); V = W / "voice"; V.mkdir(parents=True, exist_ok=True)
vids = sorted((W / "raw").glob("voice*.mp4")); LOG = {"t0": time.strftime("%F %T %z"), "stages": {}}


def stage(name):
    print(f"== {name} {time.strftime('%T')}", flush=True); LOG["stages"][name] = time.time()


def wada_snr(x):
    # Kim & Stern 2008 WADA-SNR (lookup of the gamma-distribution parameter vs SNR)
    x = x / (np.max(np.abs(x)) + 1e-9); x = np.abs(x) + 1e-10
    db = np.arange(-20, 101); g = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52281283, 0.54400004, 0.56523707, 0.58989175, 0.61597755, 0.64412229, 0.67380889, 0.70474154, 0.73766767, 0.77039185, 0.80469854, 0.8387041, 0.87338541, 0.90826939, 0.94276163, 0.97648061, 1.00945302, 1.04057779, 1.07040837, 1.09853985, 1.12500083, 1.14967011, 1.17287624, 1.19390209, 1.21321606, 1.23062456, 1.24647316, 1.26057053, 1.27342796, 1.28469286, 1.29486156, 1.30393213, 1.31167862, 1.31834286, 1.32416308, 1.32920885, 1.33376146, 1.33735619, 1.34066547, 1.34348474, 1.34618165, 1.34807785, 1.34975147, 1.35118814, 1.35262275, 1.35366581, 1.35438434, 1.35512941, 1.35578452, 1.35636532, 1.35683022, 1.35719911, 1.35745609, 1.35773302, 1.35794015, 1.35810009, 1.35820836, 1.35831107, 1.35843023, 1.35851014, 1.35854506, 1.35857828, 1.35861106, 1.35863958, 1.35866153, 1.35867896, 1.35869217, 1.35870186, 1.35870989, 1.3587163, 1.35872108, 1.3587245, 1.35872715, 1.35872932, 1.35873095, 1.35873211, 1.35873302, 1.35873376, 1.35873437, 1.35873481, 1.35873516, 1.35873543, 1.35873564, 1.3587358, 1.35873592, 1.35873601, 1.35873608, 1.35873613, 1.35873617, 1.3587362, 1.35873622, 1.35873623, 1.35873624, 1.35873625, 1.35873625])
    v1 = max(np.log(np.mean(x)) - np.mean(np.log(x)), g[0] + 1e-10); v1 = min(v1, g[-1] - 1e-10)
    i = np.searchsorted(g, v1); snr = db[i - 1] + (v1 - g[i - 1]) / (g[i] - g[i - 1]) * (db[i] - db[i - 1]) if i > 0 else db[0]
    return float(snr)


# 1 ffmpeg
stage("1_ffmpeg"); (V / "raw_wav").mkdir(exist_ok=True)
for v in vids:
    o = V / "raw_wav" / f"{v.stem}.wav"
    if not o.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vn", "-ac", "2", "-ar", "44100", str(o)], check=True)
# 2 demucs
stage("2_demucs"); D = V / "demucs" / "htdemucs_ft"
todo = [str(V / "raw_wav" / f"{v.stem}.wav") for v in vids if not (D / v.stem / "vocals.wav").exists()]
if todo:
    r = subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs_ft", "--two-stems", "vocals", "-d", os.environ.get("DEMUCS_DEVICE", "cpu"), "-j", "4", "-o", str(V / "demucs"), *todo])
    assert r.returncode == 0, "demucs failed"
# 3 dereverb + denoise
stage("3_wpe_dfn"); (V / "enh").mkdir(exist_ok=True)
import librosa
need = [v for v in vids if not (V / "enh" / f"{v.stem}.wav").exists()]
if need:
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft
    from df.enhance import enhance, init_df
    model, df_state, _ = init_df()
    import torch
    for v in need:
        y, sr = sf.read(D / v.stem / "vocals.wav"); y = y.mean(1) if y.ndim > 1 else y
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        Y = stft(y16[None], size=512, shift=128).transpose(2, 0, 1)
        Z = wpe(Y, taps=10, delay=3, iterations=3, statistics_mode="full").transpose(1, 2, 0)
        z = istft(Z, size=512, shift=128)[0][: len(y16)]
        z48 = librosa.resample(z, orig_sr=16000, target_sr=df_state.sr()) if False else librosa.resample(y, orig_sr=sr, target_sr=df_state.sr())
        # WPE output is used for the dereverb (16 kHz) analysis copy; DFN runs full-band on the separated vocals so
        # the TTS reference keeps >8 kHz content. Final enh = DFN(full-band vocals) with WPE late-reverb residual removed.
        e = enhance(model, df_state, torch.from_numpy(z48[None].astype(np.float32)))[0].numpy()
        e24 = librosa.resample(e, orig_sr=df_state.sr(), target_sr=24000)
        zr = librosa.resample(z, orig_sr=16000, target_sr=24000); n = min(len(e24), len(zr))
        # spectral gating of the late reverb: keep DFN magnitude where WPE says energy is direct-path (<= 8 kHz)
        E = librosa.stft(e24[:n], n_fft=1024, hop_length=256); Zs = librosa.stft(zr[:n], n_fft=1024, hop_length=256)
        lo = np.fft.rfftfreq(1024, 1 / 24000) <= 8000
        gain = np.ones_like(np.abs(E)); gain[lo] = np.clip(np.abs(Zs[lo]) / (np.abs(E[lo]) + 1e-8), 0.0, 1.0)
        out = librosa.istft(E * gain, hop_length=256, length=n)
        sf.write(V / "enh" / f"{v.stem}.wav", (out / (np.max(np.abs(out)) + 1e-9) * 0.9).astype(np.float32), 24000)
        sf.write(V / "enh" / f"{v.stem}_wpe16k.wav", z.astype(np.float32), 16000)
# 4 VAD + whisper
stage("4_vad_whisper")
seg_path = V / "segments_all.json"
if not seg_path.exists():
    from silero_vad import load_silero_vad, get_speech_timestamps
    import mlx_whisper, opencc
    cc = opencc.OpenCC("s2twp"); vad = load_silero_vad(); segs = []
    for v in vids:
        e, _ = librosa.load(V / "enh" / f"{v.stem}.wav", sr=16000)
        import torch
        ts = get_speech_timestamps(torch.from_numpy(e), vad, sampling_rate=16000, min_speech_duration_ms=300,
                                   min_silence_duration_ms=250, speech_pad_ms=80, return_seconds=True)
        wr = mlx_whisper.transcribe(str(V / "enh" / f"{v.stem}.wav"), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                                    language="zh", word_timestamps=True, condition_on_previous_text=False)
        (V / f"whisper_{v.stem}.json").write_text(json.dumps(wr, ensure_ascii=False, indent=1))
        words = [w for s in wr["segments"] for w in s.get("words", [])]
        for k, t in enumerate(ts):
            ws = [w for w in words if w["start"] < t["end"] and w["end"] > t["start"]]
            segs.append({"video": v.stem, "k": k, "start": t["start"], "end": t["end"], "dur": round(t["end"] - t["start"], 3),
                         "text": cc.convert("".join(w["word"] for w in ws).strip()),
                         "avg_word_prob": round(float(np.mean([w.get("probability", 0) for w in ws])), 3) if ws else 0.0})
    seg_path.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
segs = json.loads(seg_path.read_text())
# 5 speaker embeddings + target selection
stage("5_spk_select")
# 2026-10-03 06:46 measured: ECAPA-voxceleb agglomerative (avg, sim thr 0.55) split 16 segments into 16 clusters
# (cross-venue ECAPA cosines 0.0-0.53) -> kept 11.1 s only. WavLM-base-plus-sv separates the groups clearly
# (anchor group >= 0.73 vs other group 0.24-0.60). Method now: ANCHOR-SEEDED selection - seed = the segment whose
# transcript is her self-introduction ("我是李雅..."), iterate: add segments whose WavLM-SV cosine to the selected
# centroid >= --wthr (0.70); ECAPA kept as a reported second opinion.
ap2 = argparse.ArgumentParser(); ap2.add_argument("--wthr", type=float, default=0.70); ap2.add_argument("--work"); ap2.add_argument("--thr")
A2 = ap2.parse_args()
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, WavLMForXVector
import torch
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(Path.home() / "yaying_cache/ecapa"))
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); wl = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
E, E2 = [], []
for s in segs:
    e, _ = librosa.load(V / "enh" / f"{s['video']}.wav", sr=16000, offset=s["start"], duration=s["dur"])
    if len(e) < 16000: e = np.pad(e, (0, 16000 - len(e)))
    with torch.no_grad():
        x = enc.encode_batch(torch.from_numpy(e[None]).float()).squeeze().numpy(); E.append(x / np.linalg.norm(x))
        w = wl(**fe(e, sampling_rate=16000, return_tensors="pt")).embeddings.squeeze().numpy(); E2.append(w / np.linalg.norm(w))
E, E2 = np.stack(E), np.stack(E2); np.savez(V / "seg_embs.npz", ecapa=E, wavlm=E2)
seed = next((i for i, s in enumerate(segs) if re.search(r"我是李雅", s["text"])), None)
if seed is None: seed = int(np.argmax([s["dur"] for s in segs]))
sel = {seed}
for _ in range(10):
    c = E2[sorted(sel)].mean(0); c /= np.linalg.norm(c)
    new = {i for i in range(len(segs)) if float(E2[i] @ c) >= A2.wthr} | {seed}
    if new == sel: break
    sel = new
c = E2[sorted(sel)].mean(0); c /= np.linalg.norm(c); ce = E[sorted(sel)].mean(0); ce /= np.linalg.norm(ce)
tgt = 1
for i, s in enumerate(segs):
    s["cluster"] = 1 if i in sel else 0; s["cos_to_target"] = round(float(E2[i] @ c), 4); s["ecapa_cos_to_target"] = round(float(E[i] @ ce), 4)
stats = {k: {"n": sum(1 for s in segs if s["cluster"] == k), "dur": round(sum(s["dur"] for s in segs if s["cluster"] == k), 2),
             "videos": sorted({s["video"] for s in segs if s["cluster"] == k})} for k in (0, 1)}
stats["seed"] = {"index": seed, "video": segs[seed]["video"], "text": segs[seed]["text"]}
# 6 dataset
stage("6_dataset"); C = V / "clean"; C.mkdir(exist_ok=True)
for f in C.glob("*.wav"): f.unlink()
keep = []
for s in segs:
    s["target"] = bool(s["cluster"] == tgt and s["dur"] >= 0.8 and s["text"])
    if not s["target"]: continue
    e, _ = librosa.load(V / "enh" / f"{s['video']}.wav", sr=24000, offset=s["start"], duration=s["dur"])
    voc, _ = librosa.load(D / s["video"] / "vocals.wav", sr=24000, offset=s["start"], duration=s["dur"])
    acc, _ = librosa.load(D / s["video"] / "no_vocals.wav", sr=24000, offset=s["start"], duration=s["dur"])
    s["snr_wada_db"] = round(wada_snr(e), 2)
    s["snr_sep_db"] = round(float(10 * np.log10((np.mean(voc ** 2) + 1e-12) / (np.mean(acc ** 2) + 1e-12))), 2)
    name = f"{s['video']}_{s['k']:03d}.wav"; sf.write(C / name, e, 24000); s["wav"] = f"voice/clean/{name}"; keep.append(s)
tot = sum(s["dur"] for s in keep)
ds = {"generated_at": time.strftime("%F %T %z"), "consent": "Lex 2026-10-03 05:58 Asia/Taipei: 雅英 personally consented; videos hers or Lex authorized",
      "pipeline": ["ffmpeg 44.1k", "demucs htdemucs_ft --two-stems vocals (cpu; MPS fails: conv1d output channels > 65536)", "nara_wpe WPE taps=10 delay=3 (16k) + DeepFilterNet3 full-band, WPE-gated <=8 kHz",
                   "silero-vad (min 300 ms, gap 250 ms, pad 80 ms)", "whisper large-v3 (mlx) word timestamps, OpenCC s2twp",
                   f"anchor-seeded speaker selection: WavLM-base-plus-sv cosine >= {A2.wthr} to selected centroid (seed = self-intro segment); ECAPA reported"],
      "clusters": stats, "target_cluster": int(tgt), "segments_total": len(segs), "segments_kept": len(keep),
      "clean_speech_s": round(tot, 2), "clean_speech_min": round(tot / 60, 3),
      "snr_wada_db_median": round(float(np.median([s["snr_wada_db"] for s in keep])), 2) if keep else None,
      "snr_sep_db_median": round(float(np.median([s["snr_sep_db"] for s in keep])), 2) if keep else None,
      "segments": segs}
(W / "dataset.json").write_text(json.dumps(ds, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in ds.items() if k != "segments"}, ensure_ascii=False, indent=1))
for s in segs: print(s["video"], s["k"], s["dur"], s["cluster"], s["cos_to_target"], s.get("target"), s["text"])
