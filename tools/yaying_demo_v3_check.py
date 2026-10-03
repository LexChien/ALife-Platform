#!/usr/bin/env python3
"""Plan 40 demo v3 verification on the decoded yaying_v3.mp4: (1) frame-diff no-reverse check (each output frame is
matched to its nearest ORIGINAL source frame by mean-abs-diff; the matched index sequence must never decrease),
(2) ArcFace of every output frame vs its nearest original frame, (3) 6 side-by-side keyframes (original | v3, with a
3x mouth zoom). Gate / sharpness / flow numbers are copied from yaying_v3_gate.json and work/v3_plan.json.
  ~/plan38_cache/venv/bin/python tools/yaying_demo_v3_check.py
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yaying_demo_v2_check import frames, sbs  # noqa: E402
D = ROOT / "runs/yaying_clone/demo_v2"


def main():
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    face = lambda im: (lambda fs: max(fs, key=lambda f: f.det_score) if fs else None)(fa.get(np.ascontiguousarray(im[:, :, ::-1])))  # noqa
    src = frames(D / "work/src.mp4"); out = frames(D / "yaying_v3.mp4")
    plan = json.load(open(D / "work/v3_plan.json")); pos = np.array(plan["timeline"])
    g = json.load(open(D / "yaying_v3_gate.json"))
    sm = lambda f: cv2.resize(f, (116, 172), interpolation=cv2.INTER_AREA).astype(np.float32)  # noqa
    S = np.stack([sm(f) for f in src])
    match = []
    for f in out:
        d = np.abs(S - sm(f)[None]).mean((1, 2, 3)); match.append(int(np.argmin(d)))
    match = np.array(match)
    back = np.where(np.diff(match) < 0)[0]
    # tolerate +-1 ambiguity in held / interpolated stretches: count backward jumps > 1 frame as reverse playback
    rev = [int(k) for k in back if match[k] - match[k + 1] > 1]
    emb = {}
    arcs = []
    for k, f in enumerate(out):
        j = int(round(pos[k]))
        if j not in emb:
            fs = face(src[j]); emb[j] = fs.normed_embedding if fs is not None else None
        fo = face(f)
        if fo is not None and emb[j] is not None:
            arcs.append(float(np.dot(fo.normed_embedding, emb[j])))
    rows = g["rows"]
    ls = [r for r in rows if r.get("lipsync") and r["gate"] == "PASS"]
    picks = [r["i"] for r in sorted(ls, key=lambda r: -r["blend"])][:0]
    sp = [r["i"] for r in ls]
    if sp:
        picks = [int(v) for v in np.array(sp)[np.linspace(0, len(sp) - 1, 4).astype(int)]]
    picks += [int(np.searchsorted(pos, 40)), int(np.searchsorted(pos, 100))]
    kd = D / "keyframes_v3"; kd.mkdir(exist_ok=True)
    for n_, i in enumerate(picks):
        j = int(round(pos[i])); fs = face(src[j])
        x1, y1, x2, y2 = fs.bbox.astype(int)
        cy = int(y1 + 0.62 * (y2 - y1)); cx = int((x1 + x2) / 2); r = int(0.32 * (x2 - x1))
        rr = rows[i]
        lab = (f"v3 t={i / 24:.2f}s pos={pos[i]:.2f} " + (f"blend {rr['blend']} ArcFace {rr['arcface']} LVdev {rr['lv_ratio_dev']}" if rr.get("lipsync") else "no lip-sync (silent)"))
        sbs(src[j], out[i], f"original frame {j}", lab, kd / f"v3_key{n_ + 1}_t{i:04d}.png",
            (max(0, cx - r), max(0, cy - r), cx + r, cy + r))
    res = {"frames": len(out), "matched_source_index_monotonic": len(rev) == 0, "reverse_jumps_gt1": rev,
           "backward_1frame_ambiguities": int(len(back) - len(rev)),
           "arcface_vs_nearest_original_min": round(min(arcs), 4), "arcface_vs_nearest_original_median": round(float(np.median(arcs)), 4),
           "faces": len(arcs), "keyframes": picks, "gate": g["summary"],
           "plan": {k: plan[k] for k in ("duration_s", "slowdown_max_x", "reverse_steps", "interpolated_frames", "original_frames",
                                         "max_flow_p99_on_interpolated_pairs", "max_flow_p99_native_pairs", "flow_thr_px", "spans", "speech", "voice_tempo")}}
    json.dump(res, open(D / "check_v3.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "gate"}, ensure_ascii=False)); print("V3_CHECK_DONE")


if __name__ == "__main__":
    main()
