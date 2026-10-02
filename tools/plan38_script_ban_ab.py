#!/usr/bin/env python3
"""Plan 38 hygiene A/B (REAL llama-server): stray non-zh/en script rate and TTFT with/without logit_bias ban."""
import json, re, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cognition.loop import frame_user_message
from cognition.language import lock_line
voc = json.loads((ROOT / "runs/plan38/llm/vocab_scripts.json").read_text())
KEEP = {"LATIN", "GREEK", "MATHEMATICAL", "COMBINING", "MODIFIER", "?", "VARIATION", "SUBSCRIPT", "EXTENDED", "SUPERSCRIPT"}
ban_all = sorted(i for sc, ids in voc["ban"].items() if sc not in KEEP for i in ids)
ban_bn = sorted(voc["ban"]["BENGALI"])
FOREIGN = re.compile(r"[\u0900-\u0DFF\u0E00-\u0EFF\u0600-\u06FF\u0400-\u04FF\uAC00-\uD7AF\u1100-\u11FF\u0590-\u05FF]")
sysm = open(sys.argv[1]).read() if len(sys.argv) > 1 else "你是 ALife Prototype，一個數位分身。"
prompts = ["你好，你是誰？", "記住：我的暗語是藍色海豚。", "你今天過得怎麼樣？", "介紹一下你自己", "我有點累", "幫我想一個晚餐的點子",
           "你喜歡什麼音樂？", "現在的專案進度如何？", "你是 Gemma 嗎？", "跟我說個笑話"]
def call(user, bias, seed):
    body = {"messages": [{"role": "system", "content": sysm}, {"role": "user", "content": frame_user_message([lock_line("zh")], user, "zh")}],
            "max_tokens": 80, "temperature": 0.6, "seed": seed, "cache_prompt": True, "id_slot": 0}
    if bias is not None:
        body["logit_bias"] = [[i, False] for i in bias]
    data = json.dumps(body).encode()
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", data,
                   {"Content-Type": "application/json"}), timeout=120).read())
    return r["choices"][0]["message"]["content"], time.perf_counter() - t, len(data)
res = {}
for name, bias in (("none", None), ("bengali", ban_bn), ("all_scripts", ban_all)):
    rows = []
    for seed in (1, 2, 3):
        for p in prompts:
            out, dt, n = call(p, bias, seed)
            rows.append({"p": p, "seed": seed, "out": out, "s": round(dt, 3), "foreign": bool(FOREIGN.search(out))})
    lat = sorted(r["s"] for r in rows)
    res[name] = {"n": len(rows), "foreign": sum(r["foreign"] for r in rows), "p50_s": lat[len(lat)//2], "body_bytes": n,
                 "n_banned": len(bias or []), "examples": [r for r in rows if r["foreign"]][:5]}
    print(name, {k: v for k, v in res[name].items() if k != "examples"}, flush=True)
    for e in res[name]["examples"]: print("  ", e["p"], "->", e["out"][:80])
out = ROOT / "runs/plan38/llm" / f"script_ban_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps({"source": "REAL gemma-4-E2B Q4_K_M llama-server", **res}, ensure_ascii=False, indent=1))
print("saved", out)
