#!/usr/bin/env python3
"""Plan 38 E-MEM: zh/en/cross-lingual memory retrieval with candidate embedding models vs the current
Chroma default (all-MiniLM-L6-v2) and a char-bigram lexical baseline. Self-written set (NOT external).
Writes runs/plan38/memory/embed_bench.json"""
import json, time, math, re
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MEM = [  # (id, text)
 ("m01", "Lex 的研究暗語是 RED-COMET-2208。"), ("m02", "Lex 最喜歡的人工生命基質是 Lenia。"),
 ("m03", "Lex 每天早上九點喝黑咖啡，不加糖。"), ("m04", "Lex's sister lives in Kaohsiung and visits in December."),
 ("m05", "The Mac is an M1 Max with 64 GB of memory."), ("m06", "Ubuntu 機器是 Jetson Orin，只有 7 GB 記憶體。"),
 ("m07", "Lex 不喜歡在會議中被打斷。"), ("m08", "Lex prefers short answers in the morning and detailed ones at night."),
 ("m09", "上週 Lenia 演化實驗的 fitness 被標準差項利用，產生滿版波紋。"), ("m10", "The DNA genome has seven traits including warmth and empathy."),
 ("m11", "Lex 對花生過敏。"), ("m12", "Lex's favourite film is Iron Man (2008)."),
 ("m13", "每週五下午四點是實驗室例會。"), ("m14", "The whisper medium model had CER 0.036 on synthetic Chinese speech."),
 ("m15", "Lex 的母親住在台中，生日是三月十四日。"), ("m16", "Lex jogs along the riverside on Sunday mornings."),
 ("m17", "gemma_web 跑在 Mac 的 8080 埠。"), ("m18", "Lex wants the avatar's original appearance to never change."),
 ("m19", "Lex 正在學習大提琴，每週二上課。"), ("m20", "The ASAL experiments use NCA, Lenia and Boids substrates."),
 ("m21", "Lex 覺得壓力大時喜歡聽爵士樂。"), ("m22", "Lex's laptop password hint is stored offline only."),
 ("m23", "明天下午兩點要跟投資人視訊。"), ("m24", "Lex dislikes overly cheerful small talk."),
 ("m25", "研究團隊有三個人：Lex、Amy 和 Kevin。"), ("m26", "The thought stream must never be spoken aloud."),
 ("m27", "Lex 的貓叫做 Pixel，是一隻橘貓。"), ("m28", "Lex usually works until midnight on Thursdays."),
 ("m29", "上一次語音往返延遲大約是七到八秒。"), ("m30", "Lex's dentist appointment is on October 15."),
]
Q = [  # (query, gold id, type)
 ("我的研究暗語是什麼？", "m01", "zh-zh"), ("我喜歡哪一種人工生命？", "m02", "zh-zh"), ("我早上習慣喝什麼？", "m03", "zh-zh"),
 ("我有什麼食物過敏？", "m11", "zh-zh"), ("實驗室例會是什麼時候？", "m13", "zh-zh"), ("我媽媽住哪裡？", "m15", "zh-zh"),
 ("我的貓叫什麼名字？", "m27", "zh-zh"), ("我心情不好時喜歡聽什麼？", "m21", "zh-zh"),
 ("What's my research codeword?", "m01", "en-zh"), ("Am I allergic to anything?", "m11", "en-zh"), ("What is my cat called?", "m27", "en-zh"),
 ("When is the weekly lab meeting?", "m13", "en-zh"), ("Which instrument am I learning?", "m19", "en-zh"),
 ("我姊姊住在哪裡？", "m04", "zh-en"), ("我最喜歡的電影是什麼？", "m12", "zh-en"), ("我星期天早上會做什麼運動？", "m16", "zh-en"),
 ("我什麼時候要看牙醫？", "m30", "zh-en"), ("我早上喜歡長的還是短的回答？", "m08", "zh-en"),
 ("How much RAM does the Mac have?", "m05", "en-en"), ("What did the whisper model score?", "m14", "en-en"),
 ("Do I like cheerful chit-chat?", "m24", "en-en"), ("How many traits does the genome have?", "m10", "en-en"),
]
MODELS = [
 ("sentence-transformers/all-MiniLM-L6-v2", "", ""),  # = Chroma default embedding
 ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
 ("intfloat/multilingual-e5-small", "query: ", "passage: "),
 ("BAAI/bge-m3", "", ""),
]

def bigrams(s):
    s = re.sub(r"\s+", "", s.lower()); return Counter(s[i:i + 2] for i in range(len(s) - 1))

def lexical(qs, docs):
    D = [bigrams(d) for d in docs]; N = len(D); df = Counter(g for d in D for g in d)
    out = []
    for q in qs:
        qb = bigrams(q); out.append([sum(min(qb[g], d[g]) * math.log(1 + N / df[g]) for g in qb if g in d) for d in D])
    return np.array(out)

def evaluate(S, tag, extra):
    ids = [m[0] for m in MEM]; rows = []
    for (q, gold, typ), s in zip(Q, S):
        order = [ids[i] for i in np.argsort(-s)]; rank = order.index(gold) + 1
        rows.append({"q": q, "type": typ, "rank": rank, "top3": order[:3]})
    summ = {"recall@1": np.mean([r["rank"] == 1 for r in rows]), "recall@3": np.mean([r["rank"] <= 3 for r in rows]),
            "mrr": np.mean([1 / r["rank"] for r in rows])}
    by = {t: round(float(np.mean([r["rank"] == 1 for r in rows if r["type"] == t])), 3) for t in ("zh-zh", "en-zh", "zh-en", "en-en")}
    return {"model": tag, **{k: round(float(v), 3) for k, v in summ.items()}, "recall@1_by_type": by, **extra, "rows": rows}

def main():
    from sentence_transformers import SentenceTransformer
    res = {"started": time.strftime("%F %T"), "n_mem": len(MEM), "n_q": len(Q), "note": "self-written set", "results": []}
    S = lexical([q for q, _, _ in Q], [m[1] for m in MEM]); r = evaluate(S, "char-bigram-idf (lexical)", {}); res["results"].append(r)
    print(r["model"], r["recall@1"], r["recall@3"], r["recall@1_by_type"])
    for name, qp, dp in MODELS:
        t0 = time.perf_counter(); m = SentenceTransformer(name, device="cpu"); load = time.perf_counter() - t0
        D = m.encode([dp + t for _, t in MEM], normalize_embeddings=True)
        t0 = time.perf_counter(); Qe = m.encode([qp + q for q, _, _ in Q], normalize_embeddings=True, batch_size=1); qt = (time.perf_counter() - t0) / len(Q)
        ms = []
        for q, _, _ in Q[:8]:
            t1 = time.perf_counter(); m.encode([qp + q], normalize_embeddings=True); ms.append(time.perf_counter() - t1)
        params = sum(p.numel() for p in m.parameters())
        r = evaluate(Qe @ D.T, name, {"load_s": round(load, 2), "ms_per_query": round(float(np.median(ms)) * 1000, 1), "params_M": round(params / 1e6, 1), "dim": int(D.shape[1])})
        # hybrid: dense + 0.15 * normalised lexical
        L = S / (S.max(1, keepdims=True) + 1e-9); rh = evaluate(Qe @ D.T + 0.15 * L, name + " + lexical(0.15)", {})
        res["results"] += [r, rh]
        for x in (r, rh): print(x["model"], x["recall@1"], x["recall@3"], x["recall@1_by_type"], x.get("ms_per_query"))
    p = ROOT / "runs/plan38/memory/embed_bench.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__": main()
