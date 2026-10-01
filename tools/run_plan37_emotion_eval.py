#!/usr/bin/env python3
"""Plan 37 R2 E-track: independent emotion evaluation (real model; no mock).

Sets:
  public : Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset (zh-TW, human-annotated, HF).
           Mapped 平淡→neutral 開心→joy 悲傷→sadness 憤怒→anger 驚奇→surprise.
           關切/厭惡/疑問 have no 1:1 label in our 6-class space → excluded from the main score
           (厭惡→anger reported separately). The dataset has NO fear class.
  hard   : 42 hand-written hard cases (sarcasm, implicit, negation, colloquial, mixed, English).
           Written by the agent AFTER having seen the lexicon → disclosed; the public set is the
           primary independent measure.
Classifiers: lexicon (genai.web.emotion.detect_text_emotion), llm (real Gemma via
genai.web.emotion_llm), hybrid (lexicon if confident & non-neutral, else llm).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from genai.llm.factory import create_llm_adapter  # noqa: E402
from genai.web.emotion import LABELS, detect_text_emotion  # noqa: E402
from genai.web.emotion_llm import classify_emotion_llm  # noqa: E402

MAP = {"平淡語氣": "neutral", "開心語調": "joy", "悲傷語調": "sadness", "憤怒語調": "anger", "驚奇語調": "surprise"}
HARD = [
    # sarcasm (surface positive, actually negative)
    ("太好了，又要加班到半夜，真是開心死了。", "anger", "sarcasm"),
    ("哇，你真準時，才遲到一個小時而已。", "anger", "sarcasm"),
    ("好棒喔，雨下這麼大我的傘剛好壞掉。", "sadness", "sarcasm"),
    ("恭喜我自己，第三次被同一間公司拒絕。", "sadness", "sarcasm"),
    ("真是謝謝你喔，把我的秘密告訴全班。", "anger", "sarcasm"),
    ("Great, my laptop died right before the deadline. Just perfect.", "anger", "sarcasm"),
    # implicit (no emotion word)
    ("我養了十二年的狗昨天走了。", "sadness", "implicit"),
    ("明天要開刀，醫生說成功率只有一半。", "fear", "implicit"),
    ("我剛收到錄取通知，下個月就去東京上班！", "joy", "implicit"),
    ("他當著所有人的面把我的報告撕了。", "anger", "implicit"),
    ("半夜三點一直有人在敲我家的門。", "fear", "implicit"),
    ("原來那個匿名捐款的人是我爸。", "surprise", "implicit"),
    ("房東突然說下週就要我們搬走。", "fear", "implicit"),
    ("我們結婚二十年了，今天他還記得送我花。", "joy", "implicit"),
    # negation
    ("我一點都不開心。", "sadness", "negation"),
    ("其實我沒有生氣，只是有點累。", "sadness", "negation"),
    ("考試結果出來了，我不再擔心了，過了！", "joy", "negation"),
    ("別擔心，我沒事，只是想安靜一下。", "neutral", "negation"),
    # colloquial Taiwanese Mandarin
    ("靠北喔，又被插隊。", "anger", "colloquial"),
    ("嚇死寶寶了，剛剛差點被車撞。", "fear", "colloquial"),
    ("心好累，什麼都不想做。", "sadness", "colloquial"),
    ("爽啦！終於抽到演唱會門票！", "joy", "colloquial"),
    ("蛤？你說他們分手了？真的假的？", "surprise", "colloquial"),
    ("超傻眼，老闆居然自己承認錯誤。", "surprise", "colloquial"),
    ("氣到不行，客服一直踢皮球。", "anger", "colloquial"),
    ("好怕怕，明天要上台報告。", "fear", "colloquial"),
    ("今天就普普通通，吃飯上班睡覺。", "neutral", "colloquial"),
    # mixed (label = dominant feeling)
    ("升職了是很開心，但想到要搬去外縣市又有點捨不得家人。", "joy", "mixed"),
    ("雖然比賽輸了，但我真的很氣裁判的判決。", "anger", "mixed"),
    ("我很高興他回來了，可是我好怕他又會離開。", "fear", "mixed"),
    ("畢業了，終於自由，可是一想到要跟大家分開就好想哭。", "sadness", "mixed"),
    ("中樂透了？！我現在手都在抖，不知道該笑還是該哭。", "surprise", "mixed"),
    # neutral distractors containing emotion words
    ("《悲慘世界》是一本很長的小說。", "neutral", "distractor"),
    ("請問開心果一斤多少錢？", "neutral", "distractor"),
    ("這部電影的分類是恐怖片嗎？", "neutral", "distractor"),
    ("明天的會議改到下午三點。", "neutral", "distractor"),
    # English
    ("I can't stop crying since she left.", "sadness", "english"),
    ("What?! You sold the house without telling me?", "surprise", "english"),
    ("I'm terrified the biopsy results will be bad.", "fear", "english"),
    ("This is the best day of my life!", "joy", "english"),
    ("Stop lying to me. I'm done.", "anger", "english"),
    ("The meeting notes are attached.", "neutral", "english"),
]


def load_public(n_per_class: int, seed: int):
    rows = []
    with open(ROOT / "runs/plan37/emotion_r2/dataset/data.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"].strip(), r["emotion"].strip()))
    by = defaultdict(list)
    for t, e in rows:
        by[e].append(t)
    rng = random.Random(seed)
    main, extra = [], []
    for e, texts in sorted(by.items()):
        rng.shuffle(texts)
        if e in MAP:
            main += [(t, MAP[e], e) for t in texts[:n_per_class]]
        elif e == "厭惡語調":
            extra += [(t, "anger", e) for t in texts[:n_per_class // 2]]
    return main, extra, Counter(e for _, e in rows)


def scores(items, preds):
    labels = sorted({g for _, g, _ in items} | {p for p in preds if p})
    acc = sum(p == g for (_, g, _), p in zip(items, preds)) / len(items)
    f1s = {}
    for l in sorted({g for _, g, _ in items}):
        tp = sum(p == l and g == l for (_, g, _), p in zip(items, preds))
        fp = sum(p == l and g != l for (_, g, _), p in zip(items, preds))
        fn = sum(p != l and g == l for (_, g, _), p in zip(items, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[l] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    conf = defaultdict(Counter)
    for (_, g, _), p in zip(items, preds):
        conf[g][p or "unparsed"] += 1
    return {"n": len(items), "accuracy": round(acc, 3), "macro_f1": round(sum(f1s.values()) / len(f1s), 3),
            "f1": f1s, "confusion": {g: dict(c) for g, c in conf.items()}, "labels_seen": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--hybrid-threshold", type=float, default=0.7)
    args = ap.parse_args()
    out = ROOT / "runs/plan37/emotion_r2" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    public, disgust, dist = load_public(args.n_per_class, args.seed)
    hard = [(t, g, c) for t, g, c in HARD]
    adapter = None
    if not args.no_llm:
        cfg = load_config(str(ROOT / "configs/genai/gemma_llama_cpp.yaml"), profile="mac_metal")
        adapter = create_llm_adapter(cfg)
    report = {"label": "REAL lexicon + REAL Gemma (llama.cpp, mac_metal)" if adapter else "lexicon only",
              "dataset_distribution": dict(dist), "args": vars(args)}
    records = []
    t0 = time.time()
    for set_name, items in (("public", public), ("public_disgust_as_anger", disgust), ("hard", hard)):
        lex, llm, hyb = [], [], []
        for text, gold, cat in items:
            d = detect_text_emotion(text)
            lp = d["label"]
            lex.append(lp)
            if adapter:
                r = classify_emotion_llm(adapter, text)
                llm.append(r["label"])
                hp = lp if (lp != "neutral" and float(d["confidence"]) >= args.hybrid_threshold) else (r["label"] or lp)
                hyb.append(hp)
            records.append({"set": set_name, "text": text, "gold": gold, "category": cat, "lexicon": lp,
                            "lexicon_conf": d["confidence"], "llm": llm[-1] if adapter else None,
                            "llm_raw": r["raw"] if adapter else None, "hybrid": hyb[-1] if adapter else None})
        report[set_name] = {"lexicon": scores(items, lex)}
        if adapter:
            report[set_name]["llm"] = scores(items, llm)
            report[set_name]["hybrid"] = scores(items, hyb)
        if set_name == "hard":
            per_cat = {}
            for cat in sorted({c for _, _, c in items}):
                idx = [i for i, (_, _, c) in enumerate(items) if c == cat]
                per_cat[cat] = {k: round(sum(v[i] == items[i][1] for i in idx) / len(idx), 3)
                                for k, v in (("lexicon", lex), ("llm", llm), ("hybrid", hyb)) if v}
            report["hard"]["per_category_accuracy"] = per_cat
        print(set_name, {k: (v["accuracy"], v["macro_f1"]) for k, v in report[set_name].items() if isinstance(v, dict) and "accuracy" in v}, flush=True)
    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    print("out", out)


if __name__ == "__main__":
    main()
