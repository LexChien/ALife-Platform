#!/usr/bin/env python3
"""Plan 40 demo v2 step 1: synthesize persona lines with the approved 雅英 F5 clone voice (PyTorch F5, long ref).

Writes runs/yaying_clone/demo_v2/voice/line_XX.wav, voice.wav (48 kHz mono, lines joined by pauses) and
voice.json (phrase start/end times used by the audio-reactive edit). Private output (runs/ is never committed).
"""
import argparse, json, sys, time, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice.clone_tts import CloneTTS  # noqa: E402

LINES = [
    "嗨，你終於來了。我等你好久了喔。",
    "陽光好暖，可是你在我旁邊，我更喜歡。",
    "靠近一點嘛，我想聽你說說今天的事。",
    "你在想我嗎？說真的，不可以騙我喔。",
]


def read_wav(p):
    with wave.open(str(p)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(n), dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / (32768.0 if sw == 2 else 2147483648.0)
    return x, sr


def resample(x, sr, to):
    if sr == to:
        return x
    t = np.arange(int(len(x) * to / sr)) * sr / to
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/voice")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--lead", type=float, default=0.6)
    ap.add_argument("--gap", type=float, default=0.55)
    ap.add_argument("--tail", type=float, default=0.8)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ref = "runs/yaying_clone/tts_ref/ref.wav"
    tts = CloneTTS("雅英", "f5", "~/yaying_cache/venv_f5/bin/python", ref,
                   (ROOT / "runs/yaying_clone/tts_ref/ref.txt").read_text().strip(), device="mps", nfe=a.nfe,
                   ready_timeout=900)
    t0 = time.time()
    assert tts.start(), tts.start_error
    print("ready", round(time.time() - t0, 1), flush=True)
    sr = 48000
    parts, meta, cur = [np.zeros(int(a.lead * sr), np.float32)], [], a.lead
    for i, text in enumerate(LINES):
        r = tts.synthesize(text, out, stem=f"line_{i:02d}", timeout=600)
        x, xsr = read_wav(r["path"])
        x = resample(x, xsr, sr)
        # trim leading/trailing near-silence so phrase timing is accurate
        env = np.convolve(np.abs(x), np.ones(480) / 480, "same")
        idx = np.where(env > 0.01)[0]
        if len(idx):
            x = x[max(0, idx[0] - 480): idx[-1] + 2400]
        meta.append({"i": i, "text": text, "start": round(cur, 3), "end": round(cur + len(x) / sr, 3),
                     "synth_s": r.get("elapsed_s"), "engine": r.get("engine"), "src_sr": xsr})
        print(meta[-1], flush=True)
        parts += [x, np.zeros(int(a.gap * sr), np.float32)]
        cur += len(x) / sr + a.gap
    parts[-1] = np.zeros(int(a.tail * sr), np.float32)
    y = np.concatenate(parts)
    y = y / max(1e-6, np.abs(y).max()) * 0.89
    with wave.open(str(out / "voice.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    json.dump({"engine": "f5 (PyTorch, mps)", "nfe": a.nfe, "ref": ref, "sr": sr, "duration_s": round(len(y) / sr, 3),
               "lines": meta}, open(out / "voice.json", "w"), ensure_ascii=False, indent=1)
    tts.close()
    print("VOICE_DONE", round(len(y) / sr, 2), "s", flush=True)


if __name__ == "__main__":
    main()
