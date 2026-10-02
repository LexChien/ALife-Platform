#!/usr/bin/env python3
"""Plan 38 E-WAKE-ZH: open-vocabulary Chinese wake phrase with sherpa-onnx KWS
(zipformer wenetspeech 3.3M). Keyword chosen for the test: 「你好管家」 (placeholder until Lex names her, D1).
SYNTHETIC say audio. Writes runs/plan38/wake/kws_zh_bench.json"""
import json, subprocess, tempfile, time, sys
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
M = Path.home() / "plan38_cache/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
man = json.loads((ROOT / "runs/plan38/audio/manifest.json").read_text())

def say(v, t, rate=None):
    d = Path(tempfile.mkdtemp()); a = d / "a.aiff"; w = d / "a.wav"
    subprocess.run(["say", "-v", v, "-o", str(a)] + (["-r", str(rate)] if rate else []) + [t], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(w)], check=True)
    return sf.read(w, dtype="float32")[0]

def main():
    import sherpa_onnx
    from sherpa_onnx import text2token
    kw_raw = ["你好管家"]
    toks = text2token(kw_raw, tokens=str(M / "tokens.txt"), tokens_type="ppinyin")
    kwf = Path(tempfile.mkdtemp()) / "kw.txt"; kwf.write_text("\n".join(" ".join(t) + f" @{k}" for t, k in zip(toks, kw_raw)) + "\n")
    def spotter(thr):
        return sherpa_onnx.KeywordSpotter(tokens=str(M / "tokens.txt"), encoder=str(M / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            decoder=str(M / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"), joiner=str(M / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
            num_threads=1, keywords_file=str(kwf), keywords_threshold=thr, keywords_score=1.0, provider="cpu")
    def detect(ks, x):
        s = ks.create_stream(); x = np.concatenate([np.zeros(8000, "float32"), x, np.zeros(16000, "float32")])
        hits = []; t0 = time.perf_counter()
        for i in range(0, len(x), 1600):
            s.accept_waveform(16000, x[i:i + 1600])
            while ks.is_ready(s): ks.decode_stream(s)
            r = ks.get_result(s)
            if r: hits.append((i / 16000, r)); ks.reset_stream(s)
        return hits, time.perf_counter() - t0
    voices = ["Meijia", "Flo (中文（台灣）)", "Sandy (中文（台灣）)", "Shelley (中文（台灣）)", "Eddy (中文（台灣）)", "Reed (中文（台灣）)",
              "Grandma (中文（台灣）)", "Grandpa (中文（台灣）)", "Tingting", "Sinji"]
    pos_text = ["你好管家", "你好管家，現在幾點？"]; conf_text = ["你好嗎", "你好", "管家婆", "回家", "你好棒", "我好想家"]
    res = {"keyword": kw_raw, "tokens": toks, "model": M.name, "source": "SYNTHETIC say"}
    for thr in (0.15, 0.25, 0.35):
        ks = spotter(thr); pos = []; neg = 0; negdur = 0; conf = []; cost = 0; dur = 0
        rng = np.random.default_rng(5)
        for v in voices:
            for rate in (None, 230):
                for t in pos_text:
                    x = say(v, t, rate)
                    for cond in ("clean", "white15db"):
                        xx = x if cond == "clean" else x + rng.normal(0, np.sqrt(np.mean(x ** 2) / 10 ** 1.5), x.shape).astype("float32")
                        h, c = detect(ks, xx); cost += c; dur += len(xx) / 16000
                        pos.append({"voice": v, "rate": rate, "text": t, "cond": cond, "hit": bool(h)})
        for c_ in man:
            x = sf.read(ROOT / c_["path"], dtype="float32")[0]; h, _ = detect(ks, x); neg += len(h); negdur += len(x) / 16000 + 1.5
        for t in conf_text:
            for v in ("Meijia", "Flo (中文（台灣）)"):
                h, _ = detect(ks, say(v, t)); conf.append({"text": t, "voice": v, "hit": bool(h)})
        res[f"thr_{thr}"] = {"tpr": round(float(np.mean([p["hit"] for p in pos])), 3), "tpr_clean": round(float(np.mean([p["hit"] for p in pos if p["cond"] == "clean"])), 3),
                             "n_pos": len(pos), "false_accepts_corpus": neg, "corpus_minutes": round(negdur / 60, 2),
                             "confuser_hits": [f"{c['text']}/{c['voice']}" for c in conf if c["hit"]], "rtf": round(cost / dur, 4),
                             "missed": [f"{p['voice']}|{p['rate']}|{p['cond']}|{p['text']}" for p in pos if not p["hit"]][:12]}
        print(thr, json.dumps(res[f"thr_{thr}"], ensure_ascii=False))
    p = ROOT / "runs/plan38/wake/kws_zh_bench.json"; p.write_text(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__": main()
