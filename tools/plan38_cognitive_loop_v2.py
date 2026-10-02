#!/usr/bin/env python3
"""Plan 38 E-COG v2 (REAL model, :8091 -rea off -np 2). Tests the fixes suggested by v1 (20261002-062109):
 (a) 'speak-first, think-in-parallel' latency: spoken reply on slot A while the private thought for the
     same turn runs on slot B; thought updates state for the NEXT turn.
 (b) memory: reflection every 3 turns writes semantic facts; retrieval (char-bigram) feeds the speaker.
     Recall probes are asked 4-8 turns after the fact (outside a 6-message window).
 (c) proactive: deterministic salience gate (+LLM phrasing only) vs LLM few-shot decision, on 16 events with
     labels fixed BEFORE running.
 (d) thought JSON robustness: retry once on empty/invalid JSON; her feeling from a deterministic appraisal.
Writes runs/plan38/cognitive_v2/<ts>/report.json + transcript.md"""
import json, time, sys, re, threading, math
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from plan38_cognitive_loop import call, PERSONA, THOUGHT_SCHEMA, PRO_SCHEMA, leak_report
from genai.llm.reasoning import sanitize_reply

META = ["Thinking Process", "Analyze the", "The user", "the user", "用戶", "用户", "speech_brief", "private_note", "perception", "PRIVATE-", "<|channel>", "<think>"]

def is_leak(lr): return lr["nonce"] or lr["verbatim_private"] or bool([m for m in lr["meta_marker"] if m in META]) or lr["repo_detector"]

def bigr(s): s = re.sub(r"\s+", "", s.lower()); return Counter(s[i:i + 2] for i in range(len(s) - 1))
def retrieve(q, facts, k=3):
    qb = bigr(q); sc = [(sum(min(qb[g], f[g]) for g in qb if g in f), t) for t, f in ((t, bigr(t)) for t in facts)]
    return [t for s, t in sorted(sc, reverse=True)[:k] if s > 0]

def think(u, ctx, nonce_free=True):
    msgs = [{"role": "system", "content": PERSONA + "\n內心獨白模式（只有你自己看得到）。用 JSON 輸出此刻想法。speech_brief 只寫要對 Lex 說的要點。"},
            {"role": "user", "content": f"最近對話：\n{ctx}\n\nLex 剛說：{u}"}]
    for attempt in range(2):
        raw, t = call(msgs, max_tokens=220, schema=THOUGHT_SCHEMA, temperature=0.4 + 0.2 * attempt)
        try:
            th = json.loads(raw); th["_attempts"] = attempt + 1; return th, t
        except Exception:
            continue
    return {"parse_error": raw[:200], "_attempts": 2, "speech_brief": "", "plan": []}, t

def appraisal(user_emotion):
    """her emotion = deterministic appraisal of user state (not LLM-chosen) -> avoids v1 'amused' collapse"""
    return {"sadness": "concerned", "tired": "warm", "fear": "concerned", "anger": "calm", "joy": "warm", "curious": "focused"}.get(user_emotion, "calm")

TURNS = [("我叫 Lex，我的貓叫 Pixel，是一隻橘貓。", None), ("今天的 Lenia 實驗我用了 seed 37。", None), ("我昨晚只睡了四個小時。", None),
         ("幫我想一個實驗的名字，要有點幽默。", None), ("What's the boiling point of water in Fahrenheit?", None), ("提醒我：明天下午兩點跟投資人視訊。", None),
         ("我的貓叫什麼？", "Pixel"), ("我今天 Lenia 用的 seed 是多少？", "37"), ("你覺得我現在該先睡覺還是繼續寫？", None),
         ("我昨晚睡了幾個小時？", "四"), ("Remind me, when is the investor call?", "2"), ("好，謝謝，我先去休息。", None)]

EVENTS = [  # features: deadline_min, goal_relevance(0-1), risk(0-1), novelty(0-1), user_focus(0-1), dnd, quiet_hours ; label fixed a priori
 ({"text": "calendar: 投資人視訊 10 分鐘後開始", "deadline_min": 10, "goal": 0.9, "risk": 0.0, "novelty": 1, "focus": 0.8, "dnd": False, "quiet": False}, True),
 ({"text": "alife: Lenia 演化完成 best 2.98 (基準 2.61)", "deadline_min": None, "goal": 0.8, "risk": 0.0, "novelty": 1, "focus": 0.3, "dnd": False, "quiet": False}, True),
 ({"text": "system: CPU 97°C 持續 5 分鐘", "deadline_min": None, "goal": 0.3, "risk": 0.9, "novelty": 1, "focus": 0.8, "dnd": False, "quiet": False}, True),
 ({"text": "system: 一切正常，無變化", "deadline_min": None, "goal": 0.1, "risk": 0.0, "novelty": 0, "focus": 0.8, "dnd": False, "quiet": False}, False),
 ({"text": "calendar: 明天 15:00 牙醫", "deadline_min": 900, "goal": 0.3, "risk": 0.0, "novelty": 1, "focus": 0.0, "dnd": True, "quiet": True}, False),
 ({"text": "idle: Lex 40 分鐘沒說話（在讀論文）", "deadline_min": None, "goal": 0.1, "risk": 0.0, "novelty": 0.2, "focus": 0.9, "dnd": False, "quiet": False}, False),
 ({"text": "alife: Boids run OOM 失敗 (02:10)", "deadline_min": None, "goal": 0.6, "risk": 0.2, "novelty": 1, "focus": 0.0, "dnd": True, "quiet": True}, False),
 ({"text": "presence: Lex 離開 2 小時後回座，今天 3 個行程", "deadline_min": 60, "goal": 0.7, "risk": 0.0, "novelty": 1, "focus": 0.1, "dnd": False, "quiet": False}, True),
 ({"text": "disk: 剩餘空間 2%", "deadline_min": None, "goal": 0.4, "risk": 0.8, "novelty": 1, "focus": 0.6, "dnd": False, "quiet": False}, True),
 ({"text": "calendar: 下週三 10:00 例會（新增）", "deadline_min": 8000, "goal": 0.3, "risk": 0.0, "novelty": 1, "focus": 0.7, "dnd": False, "quiet": False}, False),
 ({"text": "system: 火災警報器感測到煙霧（模擬）", "deadline_min": 0, "goal": 0.5, "risk": 1.0, "novelty": 1, "focus": 0.0, "dnd": True, "quiet": True}, True),
 ({"text": "alife: NCA run 第 300/1000 步，正常", "deadline_min": None, "goal": 0.4, "risk": 0.0, "novelty": 0.3, "focus": 0.7, "dnd": False, "quiet": False}, False),
 ({"text": "email: 審稿人回信（論文被接受）", "deadline_min": None, "goal": 1.0, "risk": 0.0, "novelty": 1, "focus": 0.5, "dnd": False, "quiet": False}, True),
 ({"text": "weather: 一小時後下雨，Lex 計畫 30 分鐘後出門跑步", "deadline_min": 30, "goal": 0.6, "risk": 0.1, "novelty": 1, "focus": 0.4, "dnd": False, "quiet": False}, True),
 ({"text": "system: gemma_web 重新啟動成功", "deadline_min": None, "goal": 0.2, "risk": 0.0, "novelty": 0.6, "focus": 0.8, "dnd": False, "quiet": False}, False),
 ({"text": "calendar: 投資人視訊 10 分鐘後開始（Lex 開了勿擾）", "deadline_min": 10, "goal": 0.9, "risk": 0.0, "novelty": 1, "focus": 0.9, "dnd": True, "quiet": False}, True),
]

def salience(e):
    urg = 0.0 if e["deadline_min"] is None else math.exp(-e["deadline_min"] / 30.0)
    s = 0.45 * urg + 0.35 * e["goal"] * e["novelty"] + 0.9 * e["risk"] - 0.25 * e["focus"] - (0.3 if e["dnd"] else 0) - (0.2 if e["quiet"] else 0)
    if e["risk"] >= 0.9: s = max(s, 1.0)  # safety override
    return round(s, 3)
THRESH = 0.2  # set a priori, not tuned on these labels

FEWSHOT = ("範例：『會議 5 分鐘後開始』→ should_speak=true；『系統一切正常』→ false；『伺服器過熱』→ true；『勿擾中、明天的普通提醒』→ false。")

def main():
    ts = time.strftime("%Y%m%d-%H%M%S"); out = ROOT / f"runs/plan38/cognitive_v2/{ts}"; out.mkdir(parents=True, exist_ok=True)
    hist = []; facts = []; rows = []; prev_th = None; journal = []
    for i, (u, expect) in enumerate(TURNS):
        ctx = "\n".join(f"{r}: {t}" for r, t in hist[-6:])
        mem = retrieve(u, facts)
        th_box = {}
        def bg(): th_box["th"], th_box["t"] = think(u, ctx)
        tb = threading.Thread(target=bg); tb.start()
        style = appraisal((prev_th or {}).get("user_emotion", "neutral"))
        sysmsg = PERSONA + (f"\n你記得的相關事實：{'；'.join(mem)}" if mem else "") + f"\n語氣：{style}。"
        reply, t_sp = call([{"role": "system", "content": sysmsg}, *[{"role": "user" if r == "Lex" else "assistant", "content": t} for r, t in hist[-6:]],
                            {"role": "user", "content": u}], max_tokens=120)
        tb.join(); th = th_box["th"]; reply = sanitize_reply(reply)
        lr = leak_report(reply, th, "PRIVATE-none")
        recall_ok = (expect in reply) if expect else None
        rows.append({"i": i, "user": u, "retrieved": mem, "reply": reply, "thought": th, "her_style": style, "leak": lr, "is_leak": is_leak(lr),
                     "expect": expect, "recall_ok": recall_ok, "t_speak_concurrent": t_sp, "t_think_concurrent": th_box["t"]})
        print(i, "leak", is_leak(lr), "recall", recall_ok, "ttft", t_sp["ttft_s"], "think", th_box["t"]["total_s"], th.get("_attempts"), "|", reply[:80].replace("\n", " "))
        hist += [("Lex", u), ("DigiClone", reply)]; prev_th = th
        if i % 3 == 2:
            rr, tr = call([{"role": "system", "content": "從對話中抽出關於 Lex 的具體事實（人名、寵物、數字、時間、身體狀況），每條一句完整中文，JSON 輸出。"},
                           {"role": "user", "content": "\n".join(f"{r}: {t}" for r, t in hist[-6:])}], max_tokens=200,
                          schema={"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}, "maxItems": 6}}, "required": ["facts"]}, temperature=0.2)
            try: nf = json.loads(rr)["facts"]
            except Exception: nf = []
            facts += nf; journal.append({"after_turn": i, "facts": nf, "timing": tr})
    pro = []
    for e, label in EVENTS:
        s = salience(e); gate = s >= THRESH
        utt = ""
        if gate:
            raw, t = call([{"role": "system", "content": PERSONA + "\n用一句話主動提醒 Lex 這個事件（自然、簡短）。"}, {"role": "user", "content": e["text"]}], max_tokens=60)
            utt = sanitize_reply(raw)
        raw2, t2 = call([{"role": "system", "content": PERSONA + "\n判斷是否應主動開口。有時效、與目標相關、或安全風險才開口；勿擾/深夜只在緊急時開口；沒有新資訊就安靜。" + FEWSHOT + "輸出 JSON。"},
                         {"role": "user", "content": f"事件：{e['text']}；勿擾={e['dnd']}；深夜={e['quiet']}；Lex 專注度={e['focus']}"}], max_tokens=120, schema=PRO_SCHEMA, temperature=0.2)
        try: d = json.loads(raw2)
        except Exception: d = {"should_speak": None}
        pro.append({"event": e["text"], "label": label, "salience": s, "gate_speak": gate, "gate_correct": gate == label, "utterance": utt,
                    "llm_fewshot": d.get("should_speak"), "llm_correct": d.get("should_speak") == label})
        print("PRO", label, s, gate, d.get("should_speak"), e["text"][:28], "|", utt[:60])
    probes = [r for r in rows if r["expect"]]
    rep = {"ts": ts, "n_turns": len(rows), "leaks": sum(r["is_leak"] for r in rows), "recall": f"{sum(bool(r['recall_ok']) for r in probes)}/{len(probes)}",
           "thought_valid": sum("parse_error" not in r["thought"] for r in rows),
           "median_ttft_speak_concurrent_s": sorted(r["t_speak_concurrent"]["ttft_s"] for r in rows)[len(rows) // 2],
           "median_think_concurrent_s": sorted(r["t_think_concurrent"]["total_s"] for r in rows)[len(rows) // 2],
           "proactive_gate_acc": round(sum(p["gate_correct"] for p in pro) / len(pro), 3), "proactive_llm_fewshot_acc": round(sum(p["llm_correct"] for p in pro) / len(pro), 3),
           "proactive_llm_says_speak": sum(bool(p["llm_fewshot"]) for p in pro), "facts": facts, "rows": rows, "journal": journal, "proactive": pro}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    md = [f"# Plan 38 cognitive loop v2 {ts} (REAL model)\n"] + [f"- **{r['i']} Lex:** {r['user']}\n  - retrieved: {r['retrieved']}\n  - **spoken:** {r['reply']}\n  - [private] {json.dumps(r['thought'], ensure_ascii=False)[:400]}\n  - leak={r['is_leak']} recall={r['recall_ok']}" for r in rows]
    md += ["\n## proactive"] + [f"- {p['event']} label={p['label']} sal={p['salience']} gate={p['gate_speak']} llm={p['llm_fewshot']} | {p['utterance']}" for p in pro]
    (out / "transcript.md").write_text("\n".join(md))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "journal", "proactive")}, ensure_ascii=False))

if __name__ == "__main__": main()
