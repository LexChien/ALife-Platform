#!/usr/bin/env python3
"""Plan 38 E-TTS: local female-voice TTS candidates on Mac.
Measures per engine/voice: synth latency (first sentence = first-audio proxy for sentence-chunked streaming),
RTF, intelligibility via faster-whisper medium round-trip (CER/WER), and speaker-embedding consistency
(resemblyzer) between zh and en output of the same voice identity. Writes runs/plan38/tts/tts_bench.json
and sample WAVs under runs/plan38/tts/samples/."""
import json, time, subprocess, tempfile, re, sys
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from plan38_stt_bench import err
CACHE = Path.home() / "plan38_cache/tts"
OUT = ROOT / "runs/plan38/tts"; (OUT / "samples").mkdir(parents=True, exist_ok=True)
ZH = ["好的，Lex。", "我已經把今天的實驗結果整理好了，存活率比昨天高了百分之十二。", "提醒你，三點的會議還有十分鐘開始。"]
EN = ["Of course, Lex.", "I have summarized today's experiment; survival is twelve percent higher than yesterday.", "Your three o'clock meeting starts in ten minutes."]

def to16k(x, sr):
    import soxr
    if x.ndim > 1: x = x.mean(1)
    return soxr.resample(x.astype("float32"), sr, 16000) if sr != 16000 else x.astype("float32")

def say_engine(voice):
    def f(text):
        d = tempfile.mkdtemp(); a = Path(d) / "o.aiff"
        t0 = time.perf_counter(); subprocess.run(["say", "-v", voice, "-o", str(a), text], check=True); dt = time.perf_counter() - t0
        x, sr = sf.read(a, dtype="float32"); return x, sr, dt
    return f

def kokoro_engine(voice, lang):
    from kokoro_onnx import Kokoro
    k = kokoro_engine.k = getattr(kokoro_engine, "k", None) or Kokoro(str(CACHE / "kokoro-v1.0.onnx"), str(CACHE / "voices-v1.0.bin"))
    def f(text):
        t0 = time.perf_counter(); x, sr = k.create(text, voice=voice, speed=1.0, lang=lang); return x, sr, time.perf_counter() - t0
    return f

def piper_engine(model):
    from piper import PiperVoice
    v = PiperVoice.load(str(CACHE / model))
    def f(text):
        t0 = time.perf_counter(); chunks = list(v.synthesize(text))
        x = np.concatenate([c.audio_float_array for c in chunks]); return x, chunks[0].sample_rate, time.perf_counter() - t0
    return f

CANDIDATES = [
    ("say", "Meijia", "zh", lambda: say_engine("Meijia")),
    ("say", "Flo-zhTW", "zh", lambda: say_engine("Flo (中文（台灣）)")),
    ("say", "Flo-enUS", "en", lambda: say_engine("Flo (英文（美國）)")),
    ("say", "Sandy-zhTW", "zh", lambda: say_engine("Sandy (中文（台灣）)")),
    ("say", "Sandy-enUS", "en", lambda: say_engine("Sandy (英文（美國）)")),
    ("say", "Shelley-zhTW", "zh", lambda: say_engine("Shelley (中文（台灣）)")),
    ("say", "Shelley-enUS", "en", lambda: say_engine("Shelley (英文（美國）)")),
    ("say", "Samantha", "en", lambda: say_engine("Samantha")),
    # cross-language identity probe: the zh-TW Meijia voice reading English (run with filter "say_x")
    ("say_x", "Meijia-zh", "zh", lambda: say_engine("Meijia")),
    ("say_x", "Meijia-en", "en", lambda: say_engine("Meijia")),
    ("kokoro", "zf_xiaobei", "zh", lambda: kokoro_engine("zf_xiaobei", "cmn")),
    ("kokoro", "zf_xiaoxiao", "zh", lambda: kokoro_engine("zf_xiaoxiao", "cmn")),
    ("kokoro", "af_heart", "en", lambda: kokoro_engine("af_heart", "en-us")),
    ("kokoro", "af_bella", "en", lambda: kokoro_engine("af_bella", "en-us")),
    ("piper", "zh_CN-huayan-medium", "zh", lambda: piper_engine("zh_CN-huayan-medium.onnx")),
    ("piper", "en_US-amy-medium", "en", lambda: piper_engine("en_US-amy-medium.onnx")),
]

def main():
    # Usage: plan38_tts_bench.py [ENGINE_FILTER] [OUT_NAME]
    # NOTE (measured 07:04): macOS `say` silently falls back to the default voice (Meijia/Samantha) for
    # downloaded voices like "Flo (中文（台灣）)" when spawned inside tmux; run the say subset from a login shell.
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_name = sys.argv[2] if len(sys.argv) > 2 else "tts_bench.json"
    from faster_whisper import WhisperModel
    from resemblyzer import VoiceEncoder, preprocess_wav
    asr = WhisperModel("medium", device="cpu", compute_type="int8", download_root=str(ROOT / "models/whisper"))
    enc = VoiceEncoder("cpu")
    res = {"started": time.strftime("%F %T"), "voices": {}}
    embs = {}
    for eng, name, lang, mk in CANDIDATES:
        if only and eng != only: continue
        try:
            f = mk(); f(ZH[0] if lang == "zh" else EN[0])  # warm-up
        except Exception as e:
            res["voices"][f"{eng}:{name}"] = {"error": repr(e)[:300]}; print(eng, name, "ERROR", e); continue
        rows = []; wavs = []
        for i, text in enumerate(ZH if lang == "zh" else EN):
            x, sr, dt = f(text); d = len(x) / sr
            x16 = to16k(x, sr); wavs.append(x16)
            sf.write(OUT / "samples" / f"{eng}_{name}_{i}.wav", x16, 16000)
            segs, _ = asr.transcribe(x16, language=lang, beam_size=1)
            hyp = "".join(s.text for s in segs).strip()
            rows.append({"text": text, "synth_s": round(dt, 3), "audio_s": round(d, 3), "rtf": round(dt / d, 3), "asr": hyp, "err": round(err(text, hyp, lang), 3)})
        e = enc.embed_utterance(preprocess_wav(np.concatenate(wavs), source_sr=16000)); embs[f"{eng}:{name}"] = e
        res["voices"][f"{eng}:{name}"] = {"engine": eng, "lang": lang, "rows": rows,
            "first_sentence_synth_s": rows[0]["synth_s"], "median_rtf": float(np.median([r["rtf"] for r in rows])),
            "mean_roundtrip_err": round(float(np.mean([r["err"] for r in rows])), 3)}
        print(eng, name, json.dumps({k: v for k, v in res["voices"][f"{eng}:{name}"].items() if k != "rows"}, ensure_ascii=False))
    keys = list(embs); sim = {a: {b: round(float(np.dot(embs[a], embs[b])), 3) for b in keys} for a in keys}
    res["speaker_similarity_resemblyzer"] = sim
    (OUT / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=1)); print("wrote")

if __name__ == "__main__": main()
