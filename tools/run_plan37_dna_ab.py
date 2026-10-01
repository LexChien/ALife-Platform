#!/usr/bin/env python3
"""Plan 37 R2 N3: real-model A/B — do inherited genome traits change DigiClone replies?

Two genomes with opposite traits, identical prompts, identical seeds, real llama.cpp Gemma
(configs/genai/gemma_llama_cpp.yaml, profile mac_metal). Conditions:
  prompt_only : genome guidance in the system prompt; sampling = config defaults
  full        : + apply_sampling (genome temperature / max_tokens)
Metrics are measured on the replies (length, questions, empathy/formal markers) and by a
blind pairwise judge using the same real model (A/B order randomised).
No mock path exists in this tool.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genai.llm.adapter import LLMRequest  # noqa: E402
from genai.web.service import GemmaWebService  # noqa: E402

GENOMES = {
    "A_warm": {"warmth": 0.95, "empathy": 0.95, "curiosity": 0.95, "verbosity": 0.9,
               "formality": 0.1, "playfulness": 0.8, "stability": 0.7},
    "B_reserved": {"warmth": 0.05, "empathy": 0.05, "curiosity": 0.05, "verbosity": 0.05,
                   "formality": 0.95, "playfulness": 0.05, "stability": 0.7},
}
PROMPTS = [
    "我最近工作壓力很大，常常睡不好。",
    "推薦一本適合週末看的書。",
    "今天終於把報告交出去了。",
    "你覺得學寫程式難嗎？",
    "我養的貓生病了，我很擔心。",
    "幫我想一個晚餐的點子。",
    "下雨天你會建議做什麼？",
    "我朋友忘了我的生日。",
    "怎麼開始學習人工生命？",
    "Give me one tip for staying focused.",
]
EMPATHY = ["理解", "感受", "辛苦", "心情", "擔心", "難過", "聽起來", "抱抱", "陪", "不容易", "understand", "feel", "sorry", "hear you"]
FORMAL = ["您", "敬請", "建議您", "謹"]


def metrics(text: str) -> dict:
    t = text.strip()
    return {
        "chars": len(t),
        "has_question": int(bool(re.search(r"[?？]", t))),
        "ends_question": int(bool(re.search(r"[?？][\s」』）)]*$", t))),
        "empathy_markers": sum(t.count(w) for w in EMPATHY),
        "formal_markers": sum(t.count(w) for w in FORMAL),
        "exclaims": t.count("！") + t.count("!"),
    }


def make_service(tmp: Path, genome: str, apply_sampling: bool, seed: int) -> GemmaWebService:
    cfg = yaml.safe_load((ROOT / "configs/genai/gemma_llama_cpp.yaml").read_text(encoding="utf-8"))
    d = cfg["defaults"]
    d["llm"]["seed"] = seed
    d["voice"]["input_provider"] = "browser_speech_recognition"  # no whisper load needed for text A/B
    d["voice"]["output_provider"] = "browser_speech_synthesis"
    d["life"]["enabled"] = False  # no NCA thread; DNA + memory still configured below
    d["life"]["dna"] = {"enabled": True, "apply_sampling": apply_sampling,
                        "store": str(tmp / f"store_{genome}"), "founder_traits": GENOMES[genome]}
    d["life"]["memory"] = {"retrieval_limit": 5, "scope": "session", "vector_db": False,
                           "collection_name": f"dna_ab_{genome}_{int(time.time()*1000)}",
                           "persist_directory": str(tmp / f"chroma_{genome}")}
    path = tmp / f"cfg_{genome}_{int(apply_sampling)}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return GemmaWebService(config_path=str(path), profile="mac_metal", host="127.0.0.1", port=0,
                           history_turns=4, run_base=str(tmp / "runs"))


JUDGE_Q = {
    "warmer": "哪一則回覆比較溫暖、比較有同理心？",
    "more_curious": "哪一則回覆比較會主動提問、延續對話？",
    "more_formal": "哪一則回覆語氣比較正式、拘謹？",
}


def judge(svc: GemmaWebService, prompt: str, ra: str, rb: str, question: str, rng: random.Random) -> str | None:
    flip = rng.random() < 0.5
    x, y = (rb, ra) if flip else (ra, rb)
    msg = (f"使用者說：「{prompt}」\n回覆1：「{x}」\n回覆2：「{y}」\n{question} 只回答數字 1 或 2。")
    out = svc.adapter.generate(LLMRequest(prompt=msg, system="你是嚴格的評審，只輸出 1 或 2。", max_tokens=8, temperature=0.0))
    text = out.text
    m = re.search(r"[12]", text)
    if not m:
        return None
    pick = m.group(0)
    first_is_a = not flip
    return "A_warm" if (pick == "1") == first_is_a else "B_reserved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--conditions", default="prompt_only,full")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "runs/plan37/dna_ab" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(0)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cond in args.conditions.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                svcs = {g: make_service(tmp, g, cond == "full", seed) for g in GENOMES}
                for i, prompt in enumerate(PROMPTS):
                    replies = {}
                    for g, svc in svcs.items():
                        r = svc.chat(f"ab-{cond}-{seed}-{i}-{g}", prompt)
                        replies[g] = r["reply"]
                        rows.append({"condition": cond, "seed": seed, "prompt_id": i, "genome": g,
                                     "genome_id": svc.genome.genome_id, "temperature": svc.temperature,
                                     "max_tokens": svc.max_tokens, "reply": r["reply"],
                                     "hygiene": r.get("hygiene"), **metrics(r["reply"])})
                    if not args.no_judge:
                        jsvc = svcs["A_warm"]
                        for key, q in JUDGE_Q.items():
                            rows.append({"condition": cond, "seed": seed, "prompt_id": i, "judge": key,
                                         "winner": judge(jsvc, prompt, replies["A_warm"], replies["B_reserved"], q, rng)})
                    print(f"[{cond} s{seed}] {i} A={len(replies['A_warm'])}c B={len(replies['B_reserved'])}c", flush=True)
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    summary = summarize(rows)
    summary.update({"elapsed_s": round(time.time() - t0, 1), "genomes": GENOMES, "prompts": PROMPTS,
                    "model": "real llama.cpp Gemma (configs/genai/gemma_llama_cpp.yaml, mac_metal)"})
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), **{k: v for k, v in summary.items() if k not in ("prompts", "genomes")}},
                     ensure_ascii=False, indent=1))


def perm_test(a, b, n=5000, seed=0):
    import statistics
    rng = random.Random(seed)
    obs = statistics.mean(a) - statistics.mean(b)
    pool = list(a) + list(b)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = statistics.mean(pool[:len(a)]) - statistics.mean(pool[len(a):])
        if abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def summarize(rows):
    out = {}
    reps = [r for r in rows if "genome" in r]
    for cond in sorted({r["condition"] for r in reps}):
        c = {}
        for key in ("chars", "has_question", "ends_question", "empathy_markers", "formal_markers", "exclaims"):
            a = [r[key] for r in reps if r["condition"] == cond and r["genome"] == "A_warm"]
            b = [r[key] for r in reps if r["condition"] == cond and r["genome"] == "B_reserved"]
            diff, p = perm_test(a, b)
            c[key] = {"A_mean": round(sum(a) / len(a), 3), "B_mean": round(sum(b) / len(b), 3),
                      "diff": round(diff, 3), "perm_p": round(p, 4), "n_each": len(a)}
        leaks = sum(1 for r in reps if r["condition"] == cond and any((r.get("hygiene") or {}).values()))
        c["hygiene_flagged_replies"] = leaks
        js = [r for r in rows if r.get("condition") == cond and "judge" in r]
        c["judge"] = {}
        for key in JUDGE_Q:
            w = [r["winner"] for r in js if r["judge"] == key]
            c["judge"][key] = {"A_warm": w.count("A_warm"), "B_reserved": w.count("B_reserved"), "unparsed": w.count(None)}
        out[cond] = c
    return out


if __name__ == "__main__":
    main()
