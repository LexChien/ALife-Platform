"""Plan 38 J2.7: voice identity self-check (the spec found `say` can silently swap voices in some contexts).

Renders a fixed sentence through the resident synthesizer AND a fresh ``say`` process in the CURRENT context
(foreground / tmux / launchd) and compares 16-bit PCM. Identity levels:
  identical  - sample-identical PCM (same engine, same voice)  -> PASS
  similar    - resemblyzer speaker cosine >= 0.85 (plan38 venv) -> PASS (reported)
  different  - otherwise                                        -> FAIL (caller should fall back / alert)
A reference PCM hash recorded in a foreground login shell is stored in runs/plan38/voice/selfcheck_reference.json;
later checks also compare against it so a context that silently swaps the voice in BOTH paths is still caught.
CLI: python -m voice.tts_selfcheck [--record-reference] [--context NAME]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "runs" / "plan38" / "voice" / "selfcheck_reference.json"
SENTENCE = "好的，Lex。今天的實驗已經準備好了。Good morning."
THRESHOLD = 0.85


def pcm_hash(path: str | Path) -> tuple[str, int]:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    return hashlib.sha256(frames).hexdigest(), len(frames) // 2


def render_say(text: str, voice: str, out: Path, wpm: int = 190, pbas: int = 50) -> Path:
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(wpm), "-o", str(aiff), f"[[pbas {pbas}]] {text}"], check=True, timeout=60)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", str(aiff), str(out)], check=True, timeout=60)
    aiff.unlink(missing_ok=True)
    return out


def resemblyzer_cosine(a: Path, b: Path) -> float | None:
    py = Path(os.environ.get("RESEMBLYZER_PYTHON", str(Path.home() / "plan38_cache/venv/bin/python")))
    if not py.exists():
        return None
    code = ("import sys,numpy as np;from resemblyzer import VoiceEncoder,preprocess_wav;e=VoiceEncoder('cpu');"
            "x=e.embed_utterance(preprocess_wav(sys.argv[1]));y=e.embed_utterance(preprocess_wav(sys.argv[2]));"
            "print(float(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)))")
    proc = subprocess.run([str(py), "-c", code, str(a), str(b)], capture_output=True, text=True, timeout=300)
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def selfcheck(voice: str = "Meijia", context: str | None = None, tts=None, use_resemblyzer: bool = True) -> dict:
    from voice.tts_stream import ResidentTTS
    tmp = Path(tempfile.mkdtemp(prefix="tts_selfcheck_"))
    own = tts is None
    tts = tts or ResidentTTS(voice=voice)
    t0 = time.time()
    res = tts.synthesize(SENTENCE, tmp, stem="resident")
    say_wav = render_say(SENTENCE, voice, tmp / "say.wav")
    h_res, n_res = pcm_hash(res["path"])
    h_say, n_say = pcm_hash(say_wav)
    out = {"context": context or os.environ.get("TTS_SELFCHECK_CONTEXT") or ("tmux" if os.environ.get("TMUX") else "foreground"),
           "voice": voice, "provider": res.get("provider"), "resident_vs_say": "identical" if h_res == h_say else "different",
           "pcm_sha256": h_res, "samples": n_res, "say_samples": n_say, "elapsed_s": round(time.time() - t0, 3)}
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        out["vs_reference"] = "identical" if ref.get("pcm_sha256") == h_res else "different"
        out["reference_context"] = ref.get("context")
    cos = None
    if (out["resident_vs_say"] != "identical" or out.get("vs_reference") == "different") and use_resemblyzer:
        cos = resemblyzer_cosine(Path(res["path"]), say_wav)
        out["resemblyzer_cosine"] = cos
    identical = out["resident_vs_say"] == "identical" and out.get("vs_reference", "identical") == "identical"
    out["ok"] = bool(identical or (cos is not None and cos >= THRESHOLD))
    out["level"] = "identical" if identical else ("similar" if out["ok"] else "different")
    if own:
        tts.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Meijia")
    ap.add_argument("--context")
    ap.add_argument("--record-reference", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = selfcheck(a.voice, a.context)
    if a.record_reference:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps({**res, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sentence": SENTENCE}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
