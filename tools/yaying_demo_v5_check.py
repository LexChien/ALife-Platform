#!/usr/bin/env python3
"""Plan 40 demo v5 checks: eye-edit gate (ArcFace >= 0.90 vs the original timeline frame, SSIM outside the eye/brow
ROI >= 0.995, changed pixels outside the mask), whole-clip ArcFace for v5 and v5_noeye, and 6 side-by-side keyframes
(original source frame | v5) + an eye-zoom strip.
  ~/yaying_cache/venv_mt/bin/python tools/yaying_demo_v5_check.py
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yaying_demo_v2_lipsync import ssim_map  # noqa: E402
from yaying_demo_v2_edit import read_frames  # noqa: E402
D = ROOT / "runs/yaying_clone/demo_v2"; K = D / "keyframes"


def main():
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    def emb(img):
        fs = fa.get(np.ascontiguousarray(img[:, :, ::-1]))
        return max(fs, key=lambda f: f.det_score).normed_embedding if fs else None
    tl = json.load(open(D / "work/v5_timeline.json")); pos = np.array(tl["pos"])
    base = np.load(D / "work/v5_base.npy", mmap_mode="r"); eyes = np.load(D / "work/v5_eyes.npy", mmap_mode="r")
    v5 = np.load(D / "work/yaying_v5_final.npy", mmap_mode="r"); ne = np.load(D / "work/yaying_v5_noeye_final.npy", mmap_mode="r")
    erows = {r["i"]: r for r in json.load(open(D / "work/v5_eyes_rows.json"))["rows"]}
    n, h, w = base.shape[:3]
    EB = [emb(base[i]) for i in range(n)]
    res = {"eye_edit": [], "v5": [], "v5_noeye": []}
    for i in range(n):
        r = erows.get(i, {})
        if r.get("edited"):
            e = emb(eyes[i]); arc = float(np.dot(e, EB[i])) if (e is not None and EB[i] is not None) else None
            x0, y0, x1, y1 = r["roi"]; out = np.ones((h, w), bool); out[y0:y1, x0:x1] = False
            ss = float(ssim_map(cv2.cvtColor(np.ascontiguousarray(eyes[i]), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.ascontiguousarray(base[i]), cv2.COLOR_RGB2GRAY))[out].mean())
            res["eye_edit"].append({"i": i, "arcface": arc, "ssim_outside_roi": ss, "changed_px_outside_mask": r["changed_px_outside_mask"],
                                    "eye_factor": r["eye_factor"], "gaze_x": r["gaze_x"], "brow": r["brow"],
                                    "pass": arc is not None and arc >= 0.90 and ss >= 0.995})
        for key, arr in (("v5", v5), ("v5_noeye", ne)):
            e = emb(arr[i]); res[key].append(float(np.dot(e, EB[i])) if (e is not None and EB[i] is not None) else None)
    ee = res["eye_edit"]
    summ = {"eye_edit_frames": len(ee), "eye_gate_pass": sum(x["pass"] for x in ee),
            "eye_arcface_min": min(x["arcface"] for x in ee if x["arcface"]), "eye_arcface_median": float(np.median([x["arcface"] for x in ee if x["arcface"]])),
            "eye_ssim_outside_roi_min": min(x["ssim_outside_roi"] for x in ee), "eye_changed_px_outside_mask_total": sum(x["changed_px_outside_mask"] for x in ee)}
    for key in ("v5", "v5_noeye"):
        v = [x for x in res[key] if x is not None]
        summ[f"{key}_arcface_vs_original_min"] = min(v); summ[f"{key}_arcface_vs_original_median"] = float(np.median(v))
        summ[f"{key}_frames_no_face"] = sum(x is None for x in res[key])
    # keyframes: original source frame (nearest real frame at that source position) | v5
    src, _, _ = read_frames(D / "work/src.mp4")
    K.mkdir(exist_ok=True)
    picks = {"open_glance": 0.55, "open_halflid": 2.3, "open_slowblink": 2.85, "speech_hai": 4.4, "speech_handrise": 8.2, "end_liptouch_lids": 12.7}
    kf = []
    for name, t in picks.items():
        i = min(n - 1, int(round(t * 24))); so = int(round(pos[i]))
        img = np.concatenate([src[so], np.full((h, 8, 3), 255, np.uint8), v5[i]], 1)
        p = K / f"v5_sbs_{len(kf)}_{name}_f{i:03d}_src{so:03d}.png"; cv2.imwrite(str(p), img[:, :, ::-1]); kf.append(str(p.relative_to(ROOT)))
    zs = []
    for name, t in picks.items():
        i = min(n - 1, int(round(t * 24))); r = erows.get(i, {}); x0, y0, x1, y1 = r.get("roi", [100, 170, 340, 320])
        a_ = cv2.resize(np.ascontiguousarray(base[i][y0:y1, x0:x1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        b_ = cv2.resize(np.ascontiguousarray(v5[i][y0:y1, x0:x1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        z = np.concatenate([a_, np.full((a_.shape[0], 6, 3), 255, np.uint8), b_], 1)
        zs.append(cv2.resize(z, (480, int(480 * z.shape[0] / z.shape[1]))))
    p = K / "v5_eye_zoom_strip.png"; cv2.imwrite(str(p), np.concatenate(zs, 0)[:, :, ::-1]); kf.append(str(p.relative_to(ROOT)))
    json.dump({"summary": summ, "keyframes": kf, "rows": res}, open(D / "yaying_v5_check.json", "w"), indent=1)
    print(json.dumps({"summary": summ, "keyframes": kf}, indent=1))


if __name__ == "__main__":
    main()
