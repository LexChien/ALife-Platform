#!/usr/bin/env python3
"""Plan 38 E-STT: faster-whisper (CTranslate2 CPU int8) sizes and mlx-whisper (Metal) on SYNTHETIC
zh / en / code-switch clips (runs/plan38/audio), clean + white noise @10 dB SNR; also language auto-detect.
Writes runs/plan38/stt/stt_bench.json. Self-made corpus; not an external benchmark."""
import json, re, sys, time, argparse
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
try:
    import opencc; CC = opencc.OpenCC("s2twp")
except Exception:
    CC = None

def norm(s, lang):
    s = s.lower()
    if CC: s = CC.convert(s)
    s = re.sub(r"[\s，。！？、,.!?;:；：「」\"'()（）\-]", "", s)
    return s

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def err(ref, hyp, lang):
    if lang == "en":
        r = re.sub(r"[^a-z0-9 ]", " ", ref.lower()).split(); h = re.sub(r"[^a-z0-9 ]", " ", hyp.lower()).split()
        return lev(r, h) / max(1, len(r))
    r, h = norm(ref, lang), norm(hyp, lang)
    return lev(r, h) / max(1, len(r))

def load(path, snr=None, seed=0):
    x, sr = sf.read(path, dtype="float32"); assert sr == 16000
    if snr is not None:
        rng = np.random.default_rng(seed); p = np.mean(x ** 2) + 1e-12
        x = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), x.shape).astype("float32")
    return x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engines", default="fw:small,fw:medium,fw:large-v3-turbo,mlx:large-v3-turbo")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stride", type=int, default=2); a = ap.parse_args()
    man = json.loads((ROOT / "runs/plan38/audio/manifest.json").read_text())
    man = man[:: a.stride]
    if a.limit: man = man[: a.limit]
    results = {"started": time.strftime("%F %T"), "n_clips": len(man), "source": "SYNTHETIC macOS say", "engines": {}}
    for eng in a.engines.split(","):
        kind, size = eng.split(":")
        t_load = time.perf_counter()
        if kind == "fw":
            from faster_whisper import WhisperModel
            name = {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}.get(size, size)
            m = WhisperModel(name, device="cpu", compute_type="int8", download_root=str(ROOT / "models/whisper"))
            def tr(x, lang):
                segs, info = m.transcribe(x, language=lang, beam_size=1, vad_filter=False, condition_on_previous_text=False)
                return "".join(s.text for s in segs), info.language
        else:
            import mlx_whisper
            repo = "mlx-community/whisper-large-v3-turbo"
            def tr(x, lang):
                r = mlx_whisper.transcribe(x, path_or_hf_repo=repo, language=lang, condition_on_previous_text=False)
                return r["text"], r.get("language")
            tr(np.zeros(16000, dtype="float32"), "zh")
        load_s = time.perf_counter() - t_load
        rows = []
        for cond, snr in (("clean", None), ("noise10db", 10)):
            for c in man:
                x = load(ROOT / c["path"], snr, seed=c["idx"])
                lang = "en" if c["lang"] == "en" else "zh"
                t0 = time.perf_counter(); hyp, _ = tr(x, lang); dt = time.perf_counter() - t0
                rows.append({"cond": cond, "lang": c["lang"], "voice": c["voice"], "ref": c["text"], "hyp": hyp.strip(),
                             "err": round(err(c["text"], hyp, c["lang"]), 4), "latency_s": round(dt, 3), "dur_s": c["duration_s"]})
        # language auto-detect on clean set
        lid = []
        for c in man:
            x = load(ROOT / c["path"])
            t0 = time.perf_counter(); hyp, det = tr(x, None); dt = time.perf_counter() - t0
            exp = "en" if c["lang"] == "en" else "zh"
            lid.append({"lang": c["lang"], "detected": det, "ok": det == exp, "err_autolang": round(err(c["text"], hyp, c["lang"]), 4), "latency_s": round(dt, 3)})
        summ = {}
        for cond in ("clean", "noise10db"):
            for lg in ("zh", "en", "mixed"):
                rs = [r for r in rows if r["cond"] == cond and r["lang"] == lg]
                if rs:
                    summ[f"{cond}/{lg}"] = {"n": len(rs), "mean_err": round(float(np.mean([r["err"] for r in rs])), 4),
                                            "median_latency_s": round(float(np.median([r["latency_s"] for r in rs])), 3),
                                            "rtf": round(float(np.sum([r["latency_s"] for r in rs]) / np.sum([r["dur_s"] for r in rs])), 3)}
        summ["autolang"] = {"accuracy": round(float(np.mean([l["ok"] for l in lid])), 4),
                            "mean_err": round(float(np.mean([l["err_autolang"] for l in lid])), 4),
                            "by_lang": {lg: round(float(np.mean([l["ok"] for l in lid if l["lang"] == lg])), 3) for lg in ("zh", "en", "mixed")}}
        results["engines"][eng] = {"load_s": round(load_s, 2), "summary": summ, "rows": rows, "lid": lid}
        print(eng, json.dumps(summ, ensure_ascii=False))
        out = ROOT / "runs/plan38/stt/stt_bench.json"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))

if __name__ == "__main__": main()
