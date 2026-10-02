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
        else:
            from voice.clone_tts import CloneTTS
            py = {"f5": "~/yaying_cache/venv_f5/bin/python", "cosyvoice2": "~/yaying_cache/venv_cv/bin/python"}[cand]
            eng = CloneTTS(voice="雅英", engine=cand, python=py, ref_wav=str(ROOT / a.ref), ref_text=ref_text,
                           device="mps" if cand == "f5" else "cpu", nfe=16, ready_timeout=900)
            t = time.time(); ok = eng.start(); info["cold_start_s"] = round(time.time() - t, 2); info["ready"] = eng.ready_info
            if not ok:
                info["error"] = eng.start_error; allres[cand] = {"info": info, "rows": []}; print(cand, info); continue
            def fn(text, p, eng=eng):
                r = eng.synthesize(text, d, stem=p.stem, timeout=a.timeout)
                if r.get("provider") != "clone_tts": raise RuntimeError(r.get("clone_error"))
                return r["elapsed_s"]
        for lang, i, text in items:
            p = d / f"{lang}_{i:02d}.wav"
            try:
                lat = fn(text, p); dur = wav_dur(p)
                rows.append({"lang": lang, "i": i, "text": text, "wav": str(p.relative_to(ROOT)), "latency_s": round(lat, 3),
                             "dur_s": round(dur, 3), "rtf": round(lat / dur, 3)})
            except Exception as exc:
                rows.append({"lang": lang, "i": i, "text": text, "error": f"{type(exc).__name__}: {exc}"})
            print(cand, rows[-1], flush=True)
        if cand in ("f5", "cosyvoice2"): eng.close()
        allres[cand] = {"info": info, "rows": rows}
        (out / "bench.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print("WROTE", out / "bench.json")


if __name__ == "__main__":
    main()
