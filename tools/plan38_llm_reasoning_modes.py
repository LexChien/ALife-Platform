#!/usr/bin/env python3
"""Plan 38 E-LLM-R: does llama-server leak Gemma 4 'thinking' into message.content under different
reasoning flags? Restarts a SEPARATE server on :8091 per mode (gemma_web :8080 untouched). Real model.
Writes runs/plan38/llm/reasoning_modes.json"""
import json, subprocess, time, urllib.request, sys, os, signal, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from genai.llm.reasoning import has_reasoning_leak
BIN = ROOT / "third_party/llama.cpp/build-plan38/bin/llama-server"
MODEL = ROOT / "models/gemma/gemma.gguf"
EXTRA_MARKERS = ["Thinking Process", "Analyze the Request", "**分析", "用戶要求", "用户要求", "我需要根據", "<|channel>", "<think>"]
PROMPTS = [("你是冷靜、機智、忠誠的 AI 管家，以繁體中文簡短回答。", "早安，用一句話告訴我今天要注意什麼。"),
           ("You are a calm, witty, loyal AI butler. Answer briefly in English.", "Give me a two-sentence status of the lab."),
           ("你是冷靜、機智、忠誠的 AI 管家，以繁體中文簡短回答。", "17 乘以 23 等於多少？"),
           ("你是冷靜、機智、忠誠的 AI 管家，以繁體中文簡短回答。", "我今天好累，可是報告還沒寫完。"),
           ("You are a calm, witty, loyal AI butler. Answer briefly in English.", "Should I run the Lenia experiment tonight or tomorrow morning?"),
           ("你是冷靜、機智、忠誠的 AI 管家，以繁體中文簡短回答。", "幫我想三個實驗名稱。")]
MODES = {
    "A_budget0": ["--reasoning-budget", "0"],
    "B_rea_off": ["-rea", "off"],
    "C_rea_on_deepseek": ["-rea", "on", "--reasoning-format", "deepseek"],
    "D_rea_off_budget0_kwargs": ["-rea", "off", "--reasoning-budget", "0", "--chat-template-kwargs", '{"enable_thinking": false}'],
}

def leak(text):
    return has_reasoning_leak(text) or any(m in text for m in EXTRA_MARKERS)

def chat(system, user):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 320,
            "temperature": 0.6, "stream": True}
    req = urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first_c = first_r = None; content = reasoning = ""; timings = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            o = json.loads(line[5:]); timings = o.get("timings", timings)
            for ch in o.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"):
                    first_c = first_c or time.perf_counter() - t0; content += d["content"]
                if d.get("reasoning_content"):
                    first_r = first_r or time.perf_counter() - t0; reasoning += d["reasoning_content"]
    return {"ttft_content_s": first_c, "ttft_reasoning_s": first_r, "total_s": time.perf_counter() - t0, "content": content,
            "reasoning_content": reasoning, "leak": leak(content), "predicted_n": (timings or {}).get("predicted_n")}

def main():
    modes = sys.argv[1:] or list(MODES)
    outp = ROOT / "runs/plan38/llm/reasoning_modes.json"
    res = json.loads(outp.read_text()) if outp.exists() else {}
    for m in modes:
        subprocess.run(["tmux", "kill-session", "-t", "p38llm"], capture_output=True); time.sleep(2)
        cmd = [str(BIN), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8091", "-ngl", "99", "-c", "8192", "--jinja", "-np", "1"] + MODES[m]
        log = ROOT / f"runs/plan38/llm/server_{m}.log"
        subprocess.run(["tmux", "new-session", "-d", "-s", "p38llm", " ".join(f"'{c}'" for c in cmd) + f" > {log} 2>&1"])
        for _ in range(60):
            try:
                if b"ok" in urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2).read(): break
            except Exception: pass
            time.sleep(1)
        rows = [{"system": s, "user": u, **chat(s, u)} for s, u in PROMPTS]
        res[m] = {"flags": MODES[m], "leaks": sum(r["leak"] for r in rows), "n": len(rows),
                  "reasoning_nonempty": sum(bool(r["reasoning_content"]) for r in rows), "rows": rows,
                  "median_ttft_content_s": sorted([r["ttft_content_s"] or 99 for r in rows])[len(rows) // 2]}
        print(m, "leaks", res[m]["leaks"], "/", len(rows), "reasoning_nonempty", res[m]["reasoning_nonempty"], "ttft", res[m]["median_ttft_content_s"])
        for r in rows: print("   ", repr(r["content"][:90]))
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__": main()
