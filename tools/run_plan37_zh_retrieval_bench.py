#!/usr/bin/env python3
"""Plan 37 R2 C-track: Chinese retrieval benchmark for DigiClone memory embeddings.

30 memory facts (zh-TW, some mixed zh/en) + 1 paraphrased query each (queries avoid copying
the fact wording where possible). Each embedding is evaluated through a real Chroma collection
(same path as MemoryStore). Metrics: recall@1, recall@3, MRR. Dataset is self-authored
(stated in the report); real embedding models only, no mock.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = [
    ("我的貓叫做麻糬，是一隻橘色的公貓。", "我家那隻寵物叫什麼名字？"),
    ("我對花生過敏，吃到會起疹子。", "有什麼食物我不能碰？"),
    ("我每天早上六點去河濱公園慢跑。", "我通常幾點運動？"),
    ("我在新竹的一家半導體公司當工程師。", "我的職業是什麼？"),
    ("我最喜歡的電影是《神隱少女》。", "我偏好哪部動畫片？"),
    ("我姊姊下個月要在台中結婚。", "家裡最近有什麼喜事？"),
    ("我正在學習日文，目標是明年考 N2。", "我在準備什麼語言檢定？"),
    ("我開的是一台白色的 Toyota Corolla。", "我的車子是什麼款式？"),
    ("我小時候住在花蓮的海邊。", "我童年在哪裡長大？"),
    ("我不喝咖啡，只喝烏龍茶。", "我平常喝什麼飲料？"),
    ("我的生日是十一月三號。", "我哪天出生？"),
    ("我最怕的是蛇。", "什麼動物會讓我害怕？"),
    ("我週末喜歡去爬山，特別是合歡山。", "假日我常去哪裡戶外活動？"),
    ("我的專案代號是 RED-COMET-2208。", "那個專案的代碼是多少？"),
    ("我左手手腕去年骨折過。", "我以前受過什麼傷？"),
    ("我女兒今年七歲，讀小學一年級。", "我的小孩多大了？"),
    ("我睡前習慣聽 podcast 放鬆。", "我晚上怎麼讓自己入睡？"),
    ("我大學念的是台大資工系。", "我是哪個科系畢業的？"),
    ("我正在存錢，打算明年去冰島看極光。", "我的旅行計畫是什麼？"),
    ("我是素食者，已經吃素五年了。", "我的飲食有什麼限制？"),
    ("我最近在練習彈吉他，每天半小時。", "我在學什麼樂器？"),
    ("我的血型是 O 型。", "我是什麼血型？"),
    ("我很討厭下雨天，因為通勤很麻煩。", "哪種天氣讓我心情不好？"),
    ("我奶奶住在台南，我每個月回去看她。", "我多久探望一次長輩？"),
    ("我的筆電是 MacBook Pro M1 Max。", "我用什麼電腦工作？"),
    ("Lenia 是連續型細胞自動機，是 ALife 的模擬基質。", "那個人工生命的模擬底層是什麼？"),
    ("我對貓毛不過敏，但對塵蟎過敏。", "什麼東西會讓我過敏打噴嚏？"),
    ("我最喜歡的顏色是深藍色。", "我偏愛什麼色調？"),
    ("我下週二要去看牙醫。", "我最近有什麼醫療預約？"),
    ("我在學校擔任羽球社的社長。", "我在社團裡是什麼角色？"),
]


# R2 (after Plan 38 benchmark note: e5-small 0/5 on zh->en): cross-language set, self-authored.
# (fact, query, direction) — direction = memory language -> query language. The 30 zh PAIRS facts stay in
# the index as distractors.
XLANG = [
    ("Please remember: my sister lives in Osaka.", "我姊姊住在哪個城市？", "en->zh"),
    ("My favourite movie is Spirited Away.", "我最喜歡的電影是哪一部？", "en->zh"),
    ("I go swimming every Sunday morning.", "我星期天早上都做什麼運動？", "en->zh"),
    ("My dentist appointment is next Thursday at 3 pm.", "我什麼時候要去看牙醫？", "en->zh"),
    ("I prefer short answers in the morning.", "早上我喜歡長一點還是短一點的回答？", "en->zh"),
    ("My laptop is a MacBook Pro with 64 GB of RAM.", "我的筆電記憶體有多大？", "en->zh"),
    ("I am learning to play the cello.", "我正在學哪一種樂器？", "en->zh"),
    ("My best friend is called Kenji.", "我最好的朋友叫什麼名字？", "en->zh"),
    ("I am allergic to shellfish.", "我對什麼食物過敏？", "en->zh"),
    ("My flight to Berlin leaves on the 14th.", "我飛柏林的班機是幾號？", "en->zh"),
    ("The project deadline is the end of November.", "專案的截止日期是什麼時候？", "en->zh"),
    ("My daughter's name is Mia.", "我女兒叫什麼名字？", "en->zh"),
    ("我的貓叫做麻糬。", "What is my cat's name?", "zh->en"),
    ("我每天早上喝一杯黑咖啡。", "What do I drink every morning?", "zh->en"),
    ("我媽媽住在台南。", "Which city does my mother live in?", "zh->en"),
    ("實驗室每週三下午兩點開例會。", "When is the weekly lab meeting?", "zh->en"),
    ("我心情不好時喜歡聽爵士樂。", "What music do I listen to when I feel down?", "zh->en"),
    ("我的車是白色的 Toyota。", "What colour is my car?", "zh->en"),
    ("我下個月要搬到新竹。", "Where am I moving next month?", "zh->en"),
    ("我不吃牛肉。", "Is there any meat I avoid?", "zh->en"),
    ("我的生日是七月九日。", "When is my birthday?", "zh->en"),
    ("我在學日文，目前 N3 程度。", "Which language am I studying?", "zh->en"),
    ("我哥哥是消防員。", "What does my brother do for a living?", "zh->en"),
    ("我最怕的動物是蛇。", "Which animal am I most afraid of?", "zh->en"),
]


def evaluate_xlang(name: str) -> dict:
    from digital_clone.memory.store import MemoryStore
    ef, desc = load_embedding(name)
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(collection_name=f"xl_bench_{int(time.time()*1000)}", persist_directory=td,
                            require_persistence=True, embedding_function=ef)
        for i, (fact, _) in enumerate(PAIRS):
            store.add("user", fact, kind="user_fact", memory_id=f"d{i}")
        for i, (fact, _, _) in enumerate(XLANG):
            store.add("user", fact, kind="user_fact", memory_id=f"x{i}")
        t1 = time.time()
        rows = []
        for i, (_, query, direction) in enumerate(XLANG):
            ids = [g["id"] for g in store.retrieve(query, n=10, kinds=["user_fact"])]
            rows.append({"direction": direction, "rank": ids.index(f"x{i}") + 1 if f"x{i}" in ids else None})
        t_query = time.time() - t1
    out = {"embedding": desc, "n": len(rows), "index_size": len(PAIRS) + len(XLANG),
           "ms_per_query": round(1000 * t_query / len(rows), 1)}
    for d in ("en->zh", "zh->en", "all"):
        rs = [r["rank"] for r in rows if d == "all" or r["direction"] == d]
        out[d] = {"recall@1": round(sum(1 for r in rs if r == 1) / len(rs), 3),
                  "recall@3": round(sum(1 for r in rs if r and r <= 3) / len(rs), 3),
                  "mrr": round(sum(1 / r for r in rs if r) / len(rs), 3), "ranks": rs}
    return out


def load_embedding(name: str):
    if name == "chroma_default":
        return None, "chromadb default (ONNX all-MiniLM-L6-v2, English-trained)"
    if name.startswith("alife:"):
        from digital_clone.memory.embeddings import PrefixedSentenceTransformerEF
        model = name.split(":", 1)[1]
        return PrefixedSentenceTransformerEF(model), f"{model} via PrefixedSentenceTransformerEF (product path)"
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=name, device="cpu"), name


def evaluate(name: str) -> dict:
    from digital_clone.memory.store import MemoryStore
    ef, desc = load_embedding(name)
    manual = "e5" in name and not name.startswith("alife:")
    prefix_doc = "passage: " if manual else ""
    prefix_q = "query: " if manual else ""
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        store = MemoryStore(collection_name=f"zh_bench_{int(time.time()*1000)}", persist_directory=td,
                            require_persistence=True, embedding_function=ef)
        for i, (fact, _) in enumerate(PAIRS):
            store.add("user", prefix_doc + fact, kind="user_fact", memory_id=f"f{i}")
        t_index = time.time() - t0
        ranks = []
        t1 = time.time()
        for i, (_, query) in enumerate(PAIRS):
            got = store.retrieve(prefix_q + query, n=10, kinds=["user_fact"])
            ids = [g["id"] for g in got]
            ranks.append(ids.index(f"f{i}") + 1 if f"f{i}" in ids else None)
        t_query = time.time() - t1
    n = len(PAIRS)
    return {"embedding": desc, "n": n,
            "recall@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            "recall@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "ranks": ranks, "index_s": round(t_index, 2), "query_s": round(t_query, 2)}


def main():
    args = sys.argv[1:]
    xlang = "--xlang" in args
    args = [a for a in args if a != "--xlang"]
    names = args or ["chroma_default", "BAAI/bge-small-zh-v1.5", "intfloat/multilingual-e5-small",
                             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    out = ROOT / "runs/plan37/zh_retrieval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        try:
            r = evaluate(name)
            if xlang:
                r["xlang"] = evaluate_xlang(name)
        except Exception as exc:  # report, do not hide
            r = {"embedding": name, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "ranks"}, ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps({"pairs": PAIRS, "xlang_pairs": XLANG if xlang else None, "results": results}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("out", out)


if __name__ == "__main__":
    main()
