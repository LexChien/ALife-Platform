#!/usr/bin/env python3
"""Plan 38 E-LANG (REAL model on :8091, -rea off): why did cognitive-loop v2 answer 10/10 Chinese turns in English,
and which language-lock fixes it? Same 12 TURNS as v2, run sequentially with each variant's own history.
 A  v2 baseline: PERSONA + memory + '語氣：<english style label>'
 B  style label in Chinese
 C  B + per-turn lock line at the END of the system prompt (detected input language)
 D  C + verify-and-regenerate once if the reply language mismatches
Writes runs/plan38/cognitive_lang/<ts>/report.json"""
import json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from plan38_cognitive_loop import call, PERSONA
from plan38_cognitive_loop_v2 import TURNS
from genai.llm.reasoning import sanitize_reply

def lang(s):
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s)); lat = len(re.findall(r"[A-Za-z]", s))
    return "zh" if cjk >= max(2, 0.3 * (cjk + lat / 4)) else "en"
ZH_STYLE = {"calm": "平靜", "warm": "溫暖", "concerned": "關切", "focused": "專注"}
LOCK = {"zh": "【本回合語言】Lex 這句用的是中文。你必須只用繁體中文回答（專有名詞可保留英文）。",
        "en": "[Language for this turn] Lex wrote in English. Reply in English only."}

def run(variant):
    hist, rows = [], []
    for u, _ in TURNS:
        ul = lang(u); style = "calm"
        sysmsg = PERSONA + (f"\n語氣：{style}。" if variant == "A" else f"\n語氣：{ZH_STYLE[style]}。")
        if variant in ("C", "D"): sysmsg += "\n" + LOCK[ul]
        msgs = [{"role": "system", "content": sysmsg}, *[{"role": "user" if r == "Lex" else "assistant", "content": t} for r, t in hist[-6:]], {"role": "user", "content": u}]
        reply, t = call(msgs, max_tokens=120); reply = sanitize_reply(reply); regen = False; t2 = None
        if variant == "D" and lang(reply) != ul:
            regen = True; msgs[0]["content"] += "\n" + LOCK[ul] + "（上一個草稿語言錯誤，請重寫。）"
            reply, t2 = call(msgs, max_tokens=120); reply = sanitize_reply(reply)
        rows.append({"user": u, "user_lang": ul, "reply": reply, "reply_lang": lang(reply), "match": lang(reply) == ul, "regen": regen,
                     "ttft_s": t["ttft_s"], "ttft_regen_s": (t2 or {}).get("ttft_s")})
        hist += [("Lex", u), ("DigiClone", reply)]
    n_zh = sum(r["user_lang"] == "zh" for r in rows)
    return {"lang_match": f"{sum(r['match'] for r in rows)}/{len(rows)}", "zh_match": f"{sum(r['match'] for r in rows if r['user_lang']=='zh')}/{n_zh}",
            "en_match": f"{sum(r['match'] for r in rows if r['user_lang']=='en')}/{len(rows)-n_zh}", "regens": sum(r["regen"] for r in rows),
            "ttft_median_s": sorted(r["ttft_s"] for r in rows)[len(rows) // 2], "rows": rows}

def main():
    ts = time.strftime("%Y%m%d-%H%M%S"); out = ROOT / f"runs/plan38/cognitive_lang/{ts}"; out.mkdir(parents=True, exist_ok=True)
    rep = {"started": ts, "server": "llama-server :8091 -rea off -np 2", "variants": {}}
    for v in sys.argv[1:] or ["A", "B", "C", "D"]:
        rep["variants"][v] = run(v); r = rep["variants"][v]
        print(v, r["lang_match"], "zh", r["zh_match"], "en", r["en_match"], "regens", r["regens"], "ttft", r["ttft_median_s"])
        for x in r["rows"]: print("   ", x["user_lang"], "->", x["reply_lang"], "|", x["reply"][:70].replace("\n", " "))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print("wrote", out)

if __name__ == "__main__": main()
