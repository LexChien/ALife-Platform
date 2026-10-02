#!/usr/bin/env python3
"""Plan 38 J2 voice bench (REAL models, SYNTHETIC `say` audio from runs/plan38/audio/manifest.json).
  vad   : voice/vad.py StreamingVAD (silero ONNX) on 20-utterance streams (quiet, white 15 dB): frame F1, endpoint lag
  stt   : mlx-whisper large-v3-turbo vs large-v3 on all 98 clips (2.1-4.3 s): latency p50/p95 + CER/WER
  e2e   : 30 clips through endpoint -> STT -> llama-server stream -> first sentence -> resident Meijia TTS
          (first audio READY on the server; browser playback not included) + pre-rendered ack timing
Writes runs/plan38/j2/<ts>/report.json"""
import json, re, statistics, sys, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from voice.vad import SileroVAD, StreamingVAD, SR
from voice.stt_stream import UtteranceSTT, TURBO, LARGE_V3
from voice.chunker import SentenceChunker
from voice.tts_stream import ResidentTTS
from cognition.language import detect_lang, lock_line
from cognition.leak_guard import check_leak
from genai.llm.backends.llama_server import LlamaServerAdapter

MAN = json.loads((ROOT / "runs/plan38/audio/manifest.json").read_text())
PERSONA = ("你是 Lex 的 DigiClone：一位以《鋼鐵人》J.A.R.V.I.S. 為範本的女性 AI 管家。冷靜、忠誠、精準、帶一點英式冷幽默；稱呼使用者為 Lex。"
           "回答簡短（口語 1–3 句），使用者用什麼語言就用什麼語言回答（中文用繁體）。沒有工具結果時不要說已經做了。")

def load(p):
    with wave.open(str(ROOT / p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; lo = int(k); hi = min(lo + 1, len(v) - 1); return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 4)

def true_region(x):
    e = np.abs(x); idx = np.where(e > 0.02 * e.max())[0]; return idx[0] / SR, idx[-1] / SR

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]

def err_rate(ref, hyp, lang):
    if lang == "en":
        r = re.findall(r"[a-z0-9']+", ref.lower()); h = re.findall(r"[a-z0-9']+", hyp.lower())
    else:
        norm = lambda s: re.sub(r"[\W_]+", "", s.lower())  # noqa: E731
        r, h = list(norm(ref)), list(norm(hyp))
    return lev(r, h) / max(1, len(r))

def bench_vad(rep):
    out = {}
    for cond, snr in (("quiet", None), ("white15db", 15)):
        rng = np.random.default_rng(7); clips = [load(m["path"]) for m in MAN[:20]]
        sp = np.mean(np.concatenate(clips) ** 2); sd = np.sqrt(sp / 10 ** (snr / 10)) if snr else 1e-4
        parts, segs, t = [], [], 0.0
        for c in clips:
            g = np.zeros(int(SR * rng.uniform(0.8, 1.6)), np.float32); parts.append(g); t += len(g) / SR
            s, e = true_region(c); segs.append((t + s, t + e)); parts.append(c); t += len(c) / SR
        parts.append(np.zeros(int(SR * 1.5), np.float32)); x = np.concatenate(parts); x = np.clip(x + rng.normal(0, sd, x.shape).astype(np.float32), -1, 1)
        sv = StreamingVAD(SileroVAD(), min_silence_ms=300); pred = []; start = None; lags = []; t0 = time.perf_counter()
        for i in range(0, len(x), 320):  # 20 ms frames like the browser
            for ev in sv.push(x[i:i + 320]):
                if ev.kind == "speech_start": start = ev.start_t
                else:
                    pred.append((start, ev.t)); emitted = (i + 320) / SR
                    tru = min(segs, key=lambda s: abs(s[1] - ev.t)); lags.append(emitted - tru[1])
        cpu = time.perf_counter() - t0
        hop = 0.03; n = int(len(x) / SR / hop) + 1
        def lab(ss):
            y = np.zeros(n, bool)
            for s, e in ss: y[int(s / hop):int(e / hop) + 1] = True
            return y
        y, p = lab(segs), lab(pred); tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out[cond] = {"f1": round(float(2 * pr * rc / max(1e-9, pr + rc)), 3), "precision": round(float(pr), 3), "recall": round(float(rc), 3),
                     "segments": f"{len(pred)}/{len(segs)}", "endpoint_lag_s_p50": pct(lags, .5), "endpoint_lag_s_p95": pct(lags, .95),
                     "rtf": round(cpu / (len(x) / SR), 4)}
        print("VAD", cond, out[cond])
    rep["vad"] = out

def bench_stt(rep):
    out = {}
    for repo in (TURBO, LARGE_V3):
        stt = UtteranceSTT(repo); load_s = stt.preload(); rows = []
        for m in MAN:
            x = load(m["path"]); r = stt.transcribe(x)
            rows.append({"path": m["path"], "lang": m["lang"], "s": r["stt_s"], "dur": m["duration_s"], "err": round(err_rate(m["text"], r["text"], "en" if m["lang"] == "en" else "zh"), 3), "hyp": r["text"]})
        lat = [r["s"] for r in rows]; short = [r["s"] for r in rows if 2.0 <= r["dur"] <= 3.0]
        out[repo] = {"load_s": load_s, "n": len(rows), "p50_s": pct(lat, .5), "p95_s": pct(lat, .95), "p50_2to3s_clips": pct(short, .5) if short else None,
                     "n_2to3s": len(short), "zh_cer_mean": round(statistics.mean(r["err"] for r in rows if r["lang"] == "zh"), 4),
                     "en_wer_mean": round(statistics.mean(r["err"] for r in rows if r["lang"] == "en"), 4),
                     "mixed_err_mean": round(statistics.mean(r["err"] for r in rows if r["lang"] == "mixed"), 4), "rows": rows}
        print("STT", repo, {k: v for k, v in out[repo].items() if k != "rows"})
    rep["stt"] = out

def bench_e2e(rep, stt_repo):
    stt = UtteranceSTT(stt_repo); stt.preload(); tts = ResidentTTS(); tts.start()
    tts.synthesize("嗯。", ROOT / "runs/plan38/j2/tmp", stem="warm")
    llm = LlamaServerAdapter(); rows = []
    picks = [m for m in MAN if m["lang"] in ("zh", "en")][::3][:30]
    for m in picks:
        clip = load(m["path"]); x = np.concatenate([np.zeros(SR // 2, np.float32), clip, np.zeros(SR, np.float32)])
        s, e = true_region(clip); true_end = 0.5 + e
        sv = StreamingVAD(SileroVAD(), min_silence_ms=300); ep_t = None; start = None
        for i in range(0, len(x), 320):
            for ev in sv.push(x[i:i + 320]):
                if ev.kind == "speech_start": start = ev.start_t
                elif ep_t is None: ep_t = (i + 320) / SR; seg = sv.segment(start, ev.t)
            if ep_t: break
        if ep_t is None: rows.append({"path": m["path"], "error": "no endpoint"}); continue
        endpoint_lag = ep_t - true_end
        t0 = time.perf_counter(); r = stt.transcribe(seg); t_stt = time.perf_counter() - t0
        lang = detect_lang(r["text"]); msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": f"{lock_line(lang)}\n---\n{r['text']}"}]
        ch = SentenceChunker(); t1 = time.perf_counter(); first = None; first_tok = None; reply = ""
        for d in llm.stream(msgs, max_tokens=120, temperature=0.6, slot=0):
            first_tok = first_tok or time.perf_counter() - t1; reply += d
            sents = ch.feed(d)
            if sents and first is None:
                first = (sents[0], time.perf_counter() - t1)
        if first is None:
            rest = ch.flush(); first = (rest[0] if rest else reply, time.perf_counter() - t1)
        t2 = time.perf_counter(); tts.synthesize(first[0], ROOT / "runs/plan38/j2/tmp"); t_tts = time.perf_counter() - t2
        total = endpoint_lag + t_stt + first[1] + t_tts
        rows.append({"path": m["path"], "lang": m["lang"], "transcript": r["text"], "reply": reply.strip(), "first_sentence": first[0],
                     "endpoint_lag_s": round(endpoint_lag, 3), "stt_s": round(t_stt, 3), "llm_ttft_s": round(first_tok or -1, 3),
                     "llm_first_sentence_s": round(first[1], 3), "tts_first_s": round(t_tts, 3), "first_audio_ready_s": round(total, 3),
                     "ack_ready_s": round(endpoint_lag, 3), "leak": check_leak(reply).leak})
        print("E2E", rows[-1]["first_audio_ready_s"], rows[-1]["endpoint_lag_s"], rows[-1]["stt_s"], rows[-1]["llm_first_sentence_s"], rows[-1]["tts_first_s"], "|", first[0][:40])
    ok = [r for r in rows if "error" not in r]; fa = [r["first_audio_ready_s"] for r in ok]
    rep["e2e"] = {"stt_model": stt_repo, "n": len(ok), "first_audio_ready_p50_s": pct(fa, .5), "first_audio_ready_p95_s": pct(fa, .95),
                  "stage_p50": {k: pct([r[k] for r in ok], .5) for k in ("endpoint_lag_s", "stt_s", "llm_ttft_s", "llm_first_sentence_s", "tts_first_s")},
                  "ack_ready_p50_s": pct([r["ack_ready_s"] for r in ok], .5), "leaks": sum(r["leak"] for r in ok), "rows": rows,
                  "note": "server-side first audio READY (wav written); browser fetch+decode+play not included; SYNTHETIC say audio"}
    print("E2E", {k: v for k, v in rep["e2e"].items() if k != "rows"})

if __name__ == "__main__":
    what = sys.argv[1:] or ["vad", "stt", "e2e"]
    ts = time.strftime("%Y%m%d-%H%M%S"); out = ROOT / f"runs/plan38/j2/{ts}"; out.mkdir(parents=True, exist_ok=True)
    rep = {"kind": "REAL_MODELS_SYNTHETIC_AUDIO", "ts": ts}
    import os; rep["loadavg_start"] = os.getloadavg()
    if "vad" in what: bench_vad(rep)
    if "stt" in what: bench_stt(rep)
    e2e_model = next((w.split("=", 1)[1] for w in what if w.startswith("e2e_model=")), TURBO)
    if "e2e" in what: bench_e2e(rep, e2e_model)
    rep["loadavg_end"] = os.getloadavg()
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print("wrote", out)
