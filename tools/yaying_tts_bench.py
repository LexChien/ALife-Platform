#!/usr/bin/env python3
"""Plan 40 T4.1: synthesize the fixed bench sentences with every TTS candidate through the PRODUCTION adapters.

  .venv/bin/python tools/yaying_tts_bench.py --cands meijia,sandy,f5,cosyvoice2 [--out runs/yaying_clone/tts_bench]

Candidates: meijia (ResidentTTS, current default), sandy (macOS say 'Sandy (中文（台灣）)'), f5 (F5-TTS v1 Base
zero-shot, MPS), cosyvoice2 (CosyVoice2-0.5B zero-shot, CPU) - clone candidates use voice/clone_tts.CloneTTS with the
reference clip runs/yaying_clone/tts_ref/ref.wav. Records per sentence: request->wav latency (= first audio of a
sentence in the J2 chunked pipeline), audio duration, RTF; plus worker cold-start. Scoring: tools/yaying_tts_eval.py."""
import argparse, json, subprocess, sys, time, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ZH = ["你好，我在這裡，今天想聊什麼呢？", "我剛剛幫你看了一下，系統一切正常。", "這個問題有點難，讓我想一想再回答你。",
      "謝謝你一直陪著我，我真的很開心。", "外面在下雨，出門記得帶傘喔。", "我們明天早上九點開會，別忘了。",
      "哇，這個好可愛，我好喜歡！", "如果你累了，就先休息一下吧。"]
EN = ["Hello, I am here. What would you like to talk about?", "Thank you so much, that makes me really happy."]


# clone candidate variants: name -> engine, venv python, nfe, device  (Plan 40 phase 2: live-latency routes)
CLONES = {"f5": ("f5", "~/yaying_cache/venv_f5/bin/python", 16, "mps"),
          "cosyvoice2": ("cosyvoice2", "~/yaying_cache/venv_cv/bin/python", 16, "cpu"),
          "f5q_nfe16": ("f5", "~/yaying_cache/venv_f5/bin/python", 16, "mps"),
          "f5q_nfe8": ("f5", "~/yaying_cache/venv_f5/bin/python", 8, "mps"),
          "f5mlx_nfe16": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 16, "mlx"),
          "f5mlx_nfe8": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 8, "mlx"),
          "f5mlx_nfe8_s5": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 8, "mlx", "runs/yaying_clone/tts_ref/ref_s5.wav"),
          "f5mlx_nfe8_s2": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 8, "mlx", "runs/yaying_clone/tts_ref/ref_s2.wav"),
          "f5mlx_nfe16_s5": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 16, "mlx", "runs/yaying_clone/tts_ref/ref_s5.wav"),
          "f5mlx_nfe4_s2": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 4, "mlx", "runs/yaying_clone/tts_ref/ref_s2.wav"),
          "f5mlx_nfe6_s2": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 6, "mlx", "runs/yaying_clone/tts_ref/ref_s2.wav"),
          "f5mlx_nfe6_s5": ("f5mlx", "~/yaying_cache/venv_f5mlx/bin/python", 6, "mlx", "runs/yaying_clone/tts_ref/ref_s5.wav"),
          "f5q_nfe8_s5": ("f5", "~/yaying_cache/venv_f5/bin/python", 8, "mps", "runs/yaying_clone/tts_ref/ref_s5.wav"),
          "gsv": ("gsv", "~/yaying_cache/venv_gsv/bin/python", 0, "mps")}


def clone_spec(name, default_ref):
    c = CLONES[name]; ref = ROOT / (c[4] if len(c) > 4 else default_ref)
    return c[0], c[1], c[2], c[3], ref, ref.with_suffix(".txt").read_text(encoding="utf-8").strip()


def first_chunk(text):
    """Production-like first chunk: the clause up to the first comma (the cognitive loop emits clauses), >=2 chars."""
    for i, ch in enumerate(text):
        if ch in "，,、" and i >= 1:
            return text[:i + 1]
    return text


def concat_xfade(a_path, b_path, out, xfade_s=0.03):
    import numpy as np, soundfile as sf, librosa
    a, sa = sf.read(a_path, dtype="float32"); b, sb = sf.read(b_path, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if b.ndim > 1: b = b.mean(1)
    if sa != sb: a = librosa.resample(a, orig_sr=sa, target_sr=sb)
    n = int(xfade_s * sb); n = min(n, len(a), len(b))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32); mid = a[-n:] * (1 - r) + b[:n] * r
        y = np.concatenate([a[:-n], mid, b[n:]])
    else:
        y = np.concatenate([a, b])
    sf.write(out, y, sb, subtype="PCM_16")


def wav_dur(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / float(w.getframerate())


def say_tts(voice, text, out):
    aiff = out.with_suffix(".aiff")
    t = time.time()
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff), str(out)], check=True)
    aiff.unlink()
    return time.time() - t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cands", default="meijia,sandy,f5,cosyvoice2")
    ap.add_argument("--out", default="runs/yaying_clone/tts_bench"); ap.add_argument("--ref", default="runs/yaying_clone/tts_ref/ref.wav")
    ap.add_argument("--limit-zh", type=int, default=0); ap.add_argument("--no-en", action="store_true"); ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--first-chunk", action="store_true", help="also time the first clause alone (J2 first-audio)")
    a = ap.parse_args(); out = ROOT / a.out
    ref_text = (ROOT / a.ref).with_suffix(".txt").read_text(encoding="utf-8").strip()
    items = [("zh", i, t) for i, t in enumerate(ZH[: a.limit_zh or None])] + ([] if a.no_en else [("en", i, t) for i, t in enumerate(EN)])
    allres = json.loads((out / "bench.json").read_text()) if (out / "bench.json").exists() else {}
    for cand in a.cands.split(","):
        d = out / cand; d.mkdir(parents=True, exist_ok=True); rows = []; info = {}
        if cand == "meijia":
            from voice.tts_stream import ResidentTTS
            from genai.web.voice import MacSayTTS
            t = time.time(); eng = ResidentTTS(voice="Meijia", fallback=MacSayTTS(voice="Meijia")); eng.start(); info["cold_start_s"] = round(time.time() - t, 2)
            fn = lambda text, p: eng.synthesize(text, d, stem=p.stem)["elapsed_s"]
        elif cand == "sandy":
            fn = lambda text, p: say_tts("Sandy (中文（台灣）)", text, p); info["cold_start_s"] = 0.0
        elif cand.startswith("hybrid_"):  # hybrid_<fast>_<clone>: fast voice speaks the first clause, clone the rest
            _, fast_name, clone_name = cand.split("_", 2)
            from voice.tts_stream import ResidentTTS
            from genai.web.voice import MacSayTTS
            from voice.clone_tts import CloneTTS
            fast = ResidentTTS(voice=fast_name.capitalize(), fallback=MacSayTTS(voice=fast_name.capitalize())); fast.start()
            eng_c, py, nfe, dev, rw, rt = clone_spec(clone_name, a.ref)
            clone = CloneTTS(voice="雅英", engine=eng_c, python=py, ref_wav=str(rw), ref_text=rt, device=dev, nfe=nfe, ready_timeout=900)
            t = time.time(); ok = clone.start(); info["cold_start_s"] = round(time.time() - t, 2); info["ready"] = clone.ready_info
            if not ok:
                info["error"] = clone.start_error; allres[cand] = {"info": info, "rows": []}; print(cand, info); continue
            import concurrent.futures as cf
            pool = cf.ThreadPoolExecutor(2)
            def fn(text, p, fast=fast, clone=clone):
                head = first_chunk(text); rest = text[len(head):].strip()
                t0 = time.time()
                fr = pool.submit(lambda: fast.synthesize(head, d, stem=p.stem + "_head"))
                cr = pool.submit(lambda: clone.synthesize(rest, d, stem=p.stem + "_rest", timeout=a.timeout)) if rest else None
                h = fr.result(); first = time.time() - t0
                if cr is not None:
                    c = cr.result(); rest_s = time.time() - t0
                    if c.get("provider") != "clone_tts": raise RuntimeError(c.get("clone_error"))
                    concat_xfade(str(d / (p.stem + "_head.wav")), str(d / (p.stem + "_rest.wav")), str(p))
                    hd = wav_dur(d / (p.stem + "_head.wav"))
                    fn.extra = {"first_audio_s": round(first, 3), "rest_ready_s": round(rest_s, 3), "head_text": head,
                                "gap_s": round(max(0.0, rest_s - (first + hd)), 3)}
                else:
                    __import__("shutil").copy(d / (p.stem + "_head.wav"), p); fn.extra = {"first_audio_s": round(first, 3), "gap_s": 0.0}
                return time.time() - t0
        else:
            from voice.clone_tts import CloneTTS
            eng_c, py, nfe, dev, rw, rt = clone_spec(cand, a.ref)
            eng = CloneTTS(voice="雅英", engine=eng_c, python=py, ref_wav=str(rw), ref_text=rt,
                           device=dev, nfe=nfe, ready_timeout=900)
            t = time.time(); ok = eng.start(); info["cold_start_s"] = round(time.time() - t, 2); info["ready"] = eng.ready_info
            info.update({"engine": eng_c, "nfe": nfe, "device": dev, "ref": str(rw.relative_to(ROOT))})
            if not ok:
                info["error"] = eng.start_error; allres[cand] = {"info": info, "rows": []}; print(cand, info); continue
            def fn(text, p, eng=eng):
                r = eng.synthesize(text, d, stem=p.stem, timeout=a.timeout)
                if r.get("provider") != "clone_tts": raise RuntimeError(r.get("clone_error"))
                return r["elapsed_s"]
        for lang, i, text in items:
            p = d / f"{lang}_{i:02d}.wav"
            try:
                fn.extra = {}
                lat = fn(text, p); dur = wav_dur(p); extra = dict(getattr(fn, "extra", {}) or {})
                if "first_audio_s" not in extra and a.first_chunk:  # J2 first audio = synth time of the first clause
                    head = first_chunk(text)
                    extra["first_audio_s"] = round(fn(head, d / f"{lang}_{i:02d}_head.wav"), 3) if head != text else round(lat, 3)
                    extra["head_text"] = head
                rows.append({"lang": lang, "i": i, "text": text, "wav": str(p.relative_to(ROOT)), "latency_s": round(lat, 3),
                             "dur_s": round(dur, 3), "rtf": round(lat / dur, 3), **extra})
            except Exception as exc:
                rows.append({"lang": lang, "i": i, "text": text, "error": f"{type(exc).__name__}: {exc}"})
            print(cand, rows[-1], flush=True)
        if cand in CLONES: eng.close()
        if cand.startswith("hybrid_"): clone.close()
        allres[cand] = {"info": info, "rows": rows}
        (out / "bench.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print("WROTE", out / "bench.json")


if __name__ == "__main__":
    main()
