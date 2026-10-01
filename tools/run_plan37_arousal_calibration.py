#!/usr/bin/env python3
"""Plan 37 R2 — calibrate voice-arousal on RAVDESS (public, acted English speech, 24 actors).

Binary arousal labels: high = happy/angry/fearful/surprised, low = neutral/calm/sad (disgust excluded).
Reports ROC-AUC of (a) the round-1 heuristic arousal_from_prosody on raw features, (b) the same heuristic on
per-speaker-normalized features, (c) a logistic model on per-speaker z-scored features with leave-actors-out CV.
Honest scope: acted English speech, studio level; not Lex's voice, not Mandarin.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genai.web.emotion import arousal_from_prosody, prosody_features  # noqa: E402

HIGH = {"03", "05", "06", "08"}
LOW = {"01", "02", "04"}
FEATS = ["rms_db", "pitch_hz_mean", "pitch_hz_std", "voiced_ratio"]


def load16k(path: Path) -> np.ndarray:
    from scipy.signal import resample_poly
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
    x /= 32768.0 if sw == 2 else 2147483648.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="models/datasets/ravdess")
    ap.add_argument("--out", default="runs/plan37/arousal_calib")
    args = ap.parse_args()
    files = sorted((ROOT / args.data).rglob("03-01-*.wav"))
    rows = []
    t0 = time.time()
    for f in files:
        p = f.stem.split("-")
        emo, actor = p[2], int(p[6])
        if emo not in HIGH | LOW:
            continue
        feats = prosody_features(load16k(f), 16000)
        rows.append({"file": f.name, "actor": actor, "emotion": emo, "intensity": p[3], "y": int(emo in HIGH), **feats})
    print(f"{len(rows)} clips in {time.time()-t0:.0f}s", flush=True)
    y = np.array([r["y"] for r in rows])
    actors = np.array([r["actor"] for r in rows])
    X = np.array([[r[k] for k in FEATS] for r in rows], dtype=float)
    heur_raw = np.array([arousal_from_prosody(r) for r in rows])
    # per-speaker z-score (simulates a per-session baseline)
    Z = X.copy()
    for a in np.unique(actors):
        m = actors == a
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
        Z[m] = (X[m] - mu) / sd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    cv_scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=6).split(Z, y, actors):
        clf = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        cv_scores[te] = clf.decision_function(Z[te])
    full = LogisticRegression(max_iter=1000).fit(Z, y)
    single = {k: auc(y, Z[:, i]) for i, k in enumerate(FEATS)}
    heur_z = 0.6 * Z[:, 0] + 0.4 * Z[:, 2]
    from sklearn.metrics import accuracy_score
    report = {
        "dataset": "RAVDESS speech (Livingstone & Russo 2018, Zenodo 1188976), 24 actors, English, acted",
        "n_clips": len(rows), "n_high": int(y.sum()), "n_low": int(len(y) - y.sum()),
        "labels": {"high": "happy/angry/fearful/surprised", "low": "neutral/calm/sad", "excluded": "disgust"},
        "auc_heuristic_raw_r1": auc(y, heur_raw),
        "auc_heuristic_same_weights_per_speaker_z": auc(y, heur_z),
        "auc_logistic_per_speaker_z_leave_actors_out": auc(y, cv_scores),
        "acc_logistic_cv_at_0": float(accuracy_score(y, (cv_scores > 0).astype(int))),
        "auc_single_feature_z": single,
        "logistic_full_fit": {"features": FEATS, "coef": [round(float(c), 4) for c in full.coef_[0]],
                              "intercept": round(float(full.intercept_[0]), 4)},
        "scope_caveat": "Acted English speech at studio level; per-speaker z needs a per-session baseline in product.",
    }
    out = ROOT / args.out / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (out / "features.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OUT", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
