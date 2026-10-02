#!/usr/bin/env python3
"""Plan 38 E-LLM: streaming first-token latency + decode speed against a llama-server
(OpenAI-compatible /v1/chat/completions). Real model only. Writes runs/plan38/llm/ttft_<tag>.json"""
import argparse, json, time, urllib.request, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def stream_chat(url, messages, max_tokens=160, temperature=0.6, extra=None):
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "cache_prompt": True}
    if extra: body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = ""; n = 0; timings = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            obj = json.loads(data)
            if "timings" in obj: timings = obj["timings"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    if first is None: first = time.perf_counter() - t0
                    text += d; n += 1
    total = time.perf_counter() - t0
    return {"ttft_s": first, "total_s": total, "chunks": n, "text": text, "server_timings": timings}

FILLER = ("Lex 的研究筆記：ALife-Platform 結合 ASAL 人工生命搜尋、NCA、Lenia、Boids、DigiClone 記憶、"
          "DNA 基因組演化、Gemma 本機推理與語音互動。")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="default"); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = {"tag": a.tag, "url": a.url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}
    for sys_chars in (400, 2400, 6000):
        system = ("你是一位冷靜、機智、忠誠的 AI 管家，以繁體中文回答。" + FILLER * (sys_chars // len(FILLER)))[:sys_chars]
        for q in ("早安，用一句話告訴我今天要注意什麼。", "Give me a two-sentence status of the lab, in English."):
            runs = []
            for i in range(a.reps):
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": q + ("" if i == 0 else f"（第{i+1}次）")}]
                r = stream_chat(a.url, msgs); runs.append(r)
            st = [r["server_timings"] or {} for r in runs]
            out["cases"].append({
                "system_chars": sys_chars, "question": q,
                "ttft_cold_s": runs[0]["ttft_s"], "ttft_warm_median_s": statistics.median([r["ttft_s"] for r in runs[1:]]) if len(runs) > 1 else None,
                "prompt_n": [s.get("prompt_n") for s in st], "prompt_ms": [s.get("prompt_ms") for s in st],
                "predicted_per_second": [s.get("predicted_per_second") for s in st],
                "predicted_n": [s.get("predicted_n") for s in st],
                "total_s": [r["total_s"] for r in runs], "sample_reply": runs[0]["text"][:300]})
            print(json.dumps(out["cases"][-1], ensure_ascii=False)[:400])
    p = ROOT / f"runs/plan38/llm/ttft_{a.tag}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("wrote", p)

if __name__ == "__main__": main()
