#!/usr/bin/env python3
"""Plan 38 E-AUD: synthesize SYNTHETIC test clips with macOS `say` (not human speech).
Writes runs/plan38/audio/<voice>_<lang>_<i>.wav (16 kHz mono) + manifest.json, and measures
say->file synthesis latency (TTS candidate data) per voice."""
import json, os, subprocess, sys, time, wave, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan38_corpus import ZH, EN, MIXED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/plan38/audio"
OUT.mkdir(parents=True, exist_ok=True)
STT_VOICES = {
    "zh": ["Meijia", "Flo (中文（台灣）)", "Sandy (中文（台灣）)", "Eddy (中文（台灣）)", "Reed (中文（台灣）)"],
    "en": ["Samantha", "Daniel", "Fred", "Flo (英文（美國）)", "Eddy (英文（美國）)"],
    "mixed": ["Meijia", "Flo (中文（台灣）)"],
}

def slug(v):
    base = v.split(" ")[0]
    tag = "tw" if "台灣" in v else ("us" if "美國" in v else "")
    return (base + ("_" + tag if tag else "")).lower()

def dur(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()

def synth(voice, text, path):
    aiff = path.with_suffix(".aiff")
    t0 = time.perf_counter()
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    t_say = time.perf_counter() - t0
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True)
    aiff.unlink()
    return t_say

manifest = []
sets = {"zh": ZH, "en": EN, "mixed": MIXED}
for lang, voices in STT_VOICES.items():
    for v in voices:
        for i, text in enumerate(sets[lang]):
            p = OUT / f"{slug(v)}_{lang}_{i:02d}.wav"
            t = synth(v, text, p)
            manifest.append({"path": str(p.relative_to(ROOT)), "voice": v, "lang": lang, "idx": i,
                             "text": text, "duration_s": round(dur(p), 3), "say_synth_s": round(t, 3),
                             "source": "SYNTHETIC macOS say"})
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
print(len(manifest), "clips")
