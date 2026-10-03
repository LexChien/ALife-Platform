#!/usr/bin/env python3
"""Plan 40 v6: measured gates/metrics for a generated candidate video (all real measurements; nothing estimated).

  identity   : insightface buffalo_l ArcFace per frame vs her ORIGINAL footage (max cosine over all 145 original
               frames, so pose changes are compared to the closest real pose) -> min / median / frames >= 0.88
  look       : mean Lab of the frame and of the face box vs the original (ΔE76) = morning-sunlight look kept?
  sharpness  : Laplacian variance of the face box, ratio to the original median
  lips       : SyncNet (syncnet_v2.model, LatentSync eval code) LSE-D (min dist) / LSE-C (confidence) / AV offset,
               on a 25 fps re-encode, S3FD face tracks (min_track 50)
  visemes    : MediaPipe FaceMesh inner-lip gap / mouth width at the forced-aligned events (closure b/p/m within
               +-2 frames: gap <= 25 % of the speech p90 gap; round: width < speech median; spread: width > median;
               open: gap > median)
  jitter     : 2nd temporal difference of the 40 MediaPipe lip landmarks normalised by mouth width (mean, p95),
               mouth-crop consecutive MAD and 2nd-order flicker
  hands      : MediaPipe Hands per frame: hands found, per-hand fingertip collapse ("merged fingers" = two
               adjacent fingertips < 0.12 x palm size), wrist-normalised landmark jerk (frame-to-frame 2nd diff)
  ~/gen_cache/venv_gen/bin/python tools/yaying_v6_eval.py --video <mp4> --align <json> [--no-sync]
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/yaying_clone/demo_v2/work/src.mp4"
LS = Path.home() / "gen_cache/LatentSync"
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def frames_of(path, fps=None):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[:, :, ::-1].copy())
    return out, cap.get(cv2.CAP_PROP_FPS) or fps


def lab_mean(img, box=None):
    x = img if box is None else img[box[1]:box[3], box[0]:box[2]]
    return cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32).mean(0) * np.array([100 / 255, 1, 1]) - np.array([0, 128, 128])


def sync_metrics(video, tmp):
    sys.path.insert(0, str(LS)); cwd = os.getcwd(); os.chdir(LS)
    try:
        import torch
        from eval.syncnet import SyncNetEval
        from eval.syncnet_detect import SyncNetDetector
        v25 = os.path.join(tmp, "v25.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-r", "25", "-c:v", "libx264", "-crf", "16", "-c:a", "aac", v25], check=True)
        dev = "cpu"
        sn = SyncNetEval(device=dev); sn.loadParameters(str(LS / "checkpoints/auxiliary/syncnet_v2.model"))
        det = SyncNetDetector(device=dev, detect_results_dir=os.path.join(tmp, "det"))
        det(video_path=v25, min_track=50)
        crops = sorted(os.listdir(os.path.join(tmp, "det", "crop")))
        res = [sn.evaluate(video_path=os.path.join(tmp, "det", "crop", c), temp_dir=os.path.join(tmp, "sn")) for c in crops]
        if not res:
            return {"lse_error": "no face track"}
        return {"lse_d": round(float(np.mean([r[1] for r in res])), 3), "lse_c": round(float(np.mean([r[2] for r in res])), 3),
                "av_offset": int(round(np.mean([r[0] for r in res]))), "tracks": len(res)}
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--align", default="runs/yaying_clone/demo_v6/audio/lines012_align.json")
    ap.add_argument("--out", default=None); ap.add_argument("--no-sync", action="store_true"); ap.add_argument("--t-id", type=float, default=0.88)
    a = ap.parse_args()
    t0 = time.time()
    import mediapipe as mp
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    def face(img):
        fs = fa.get(np.ascontiguousarray(img[:, :, ::-1]))
        return max(fs, key=lambda f: f.det_score) if fs else None
    src, _ = frames_of(SRC)
    SF = [face(f) for f in src]
    SE = np.stack([f.normed_embedding for f in SF if f is not None])
    s_box = [f.bbox.astype(int) for f in SF if f is not None]
    s_lv = np.median([cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(src[k][max(0, b[1]):b[3], max(0, b[0]):b[2]]), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                      for k, b in zip([k for k, f in enumerate(SF) if f is not None], s_box)])
    s_lab = np.mean([lab_mean(f) for f in src], 0)
    s_lab_face = np.mean([lab_mean(src[k], np.clip(b, 0, None)) for k, b in zip([k for k, f in enumerate(SF) if f is not None], s_box)], 0)
    vid, fps = frames_of(a.video)
    n = len(vid); h, w = vid[0].shape[:2]
    arc, lvr, dE, dEf, boxes = [], [], [], [], []
    for f in vid:
        fc = face(f)
        if fc is None:
            arc.append(None); boxes.append(None); continue
        arc.append(float((SE @ fc.normed_embedding).max()))
        b = np.clip(fc.bbox.astype(int), 0, None); boxes.append(b)
        lvr.append(cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(f[b[1]:b[3], b[0]:b[2]]), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() / s_lv)
        dEf.append(float(np.linalg.norm(lab_mean(f, b) - s_lab_face)))
        dE.append(float(np.linalg.norm(lab_mean(f) - s_lab)))
    av = [x for x in arc if x is not None]
    res = {"video": a.video, "frames": n, "fps": fps, "size": [w, h], "duration_s": round(n / fps, 3),
           "identity": {"arcface_min": round(min(av), 4) if av else None, "arcface_median": round(float(np.median(av)), 4) if av else None,
                        "frames_ge_t": int(sum(x >= a.t_id for x in av)), "frames_no_face": int(sum(x is None for x in arc)),
                        "all_frames_pass": bool(av and len(av) == n and min(av) >= a.t_id)},
           "look": {"deltaE_frame_mean": round(float(np.mean(dE)), 2), "deltaE_face_mean": round(float(np.mean(dEf)), 2)},
           "sharpness": {"face_lv_ratio_median": round(float(np.median(lvr)), 3), "face_lv_ratio_p10": round(float(np.percentile(lvr, 10)), 3)}}
    # lips / visemes via FaceMesh
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    L, gap, width = [], [], []
    for f in vid:
        r = fm.process(np.ascontiguousarray(f))
        if not r.multi_face_landmarks:
            L.append(None); gap.append(np.nan); width.append(np.nan); continue
        p = np.array([[q.x * w, q.y * h] for q in r.multi_face_landmarks[0].landmark], np.float32)
        iod = np.linalg.norm(p[33] - p[263])
        L.append(p[LIPS]); gap.append(float(np.linalg.norm(p[13] - p[14]) / iod)); width.append(float(np.linalg.norm(p[61] - p[291]) / iod))
    gap = np.array(gap); width = np.array(width)
    ok = [k for k in range(1, n - 1) if all(L[j] is not None for j in (k - 1, k, k + 1))]
    jit = [float(np.linalg.norm(L[k + 1] - 2 * L[k] + L[k - 1], axis=1).mean() / max(1e-6, np.ptp(L[k][:, 0]))) for k in ok]
    fl, fl2 = [], []
    for k in range(1, n - 1):
        if L[k] is None:
            continue
        x0, y0 = (L[k].min(0) - 8).astype(int); x1, y1 = (L[k].max(0) + 8).astype(int)
        c = [vid[j][max(0, y0):y1, max(0, x0):x1].astype(np.float32) for j in (k - 1, k, k + 1)]
        fl.append(float(np.abs(c[1] - c[0]).mean())); fl2.append(float(np.abs(c[2] - 2 * c[1] + c[0]).mean()))
    res["lip_motion"] = {"jitter_mean": round(float(np.mean(jit)), 5) if jit else None, "jitter_p95": round(float(np.percentile(jit, 95)), 5) if jit else None,
                         "mouth_flicker_mad": round(float(np.mean(fl)), 3) if fl else None, "mouth_flicker2": round(float(np.mean(fl2)), 3) if fl2 else None,
                         "facemesh_frames": int(sum(x is not None for x in L))}
    if a.align and Path(ROOT / a.align).exists():
        al = json.load(open(ROOT / a.align)); sc = fps / al["fps"]
        sp = [int(round(ch["onset_frame"] * sc)) for ch in al["chars"]]
        lo, hi = max(0, min(sp)), min(n - 1, max(sp) + int(0.3 * fps))
        g_sp, w_sp = gap[lo:hi + 1], width[lo:hi + 1]
        g90, gmed, wmed = np.nanpercentile(g_sp, 90), np.nanmedian(g_sp), np.nanmedian(w_sp)
        hits = {v: [0, 0] for v in ("closure", "round", "spread", "open")}
        for ch, k in zip(al["chars"], sp):
            win = slice(max(0, k - 2), min(n, k + 3))
            for v in ch["visemes"]:
                if v not in hits or k >= n:
                    continue
                hits[v][1] += 1
                if v == "closure":
                    hits[v][0] += int(np.nanmin(gap[win]) <= 0.25 * g90)
                elif v == "round":
                    hits[v][0] += int(np.nanmean(width[win]) < wmed)
                elif v == "spread":
                    hits[v][0] += int(np.nanmean(width[win]) > wmed)
                elif v == "open":
                    hits[v][0] += int(np.nanmax(gap[win]) > gmed)
        res["visemes"] = {v: f"{h_}/{t_}" for v, (h_, t_) in hits.items()}
        tot = sum(t_ for _, t_ in hits.values()); res["visemes"]["score"] = round(sum(h_ for h_, _ in hits.values()) / max(1, tot), 3)
    # hands
    hd = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)
    nh, merged, H = [], 0, []
    for f in vid:
        r = hd.process(np.ascontiguousarray(f))
        hl = r.multi_hand_landmarks or []
        nh.append(len(hl))
        if hl:
            p = np.array([[q.x * w, q.y * h] for q in hl[0].landmark], np.float32)
            palm = np.linalg.norm(p[0] - p[9]) + 1e-6
            tips = p[[4, 8, 12, 16, 20]]
            merged += int(any(np.linalg.norm(tips[j] - tips[j + 1]) < 0.12 * palm for j in range(1, 4)))
            H.append((p - p[0]) / palm)
        else:
            H.append(None)
    hj = [float(np.linalg.norm(H[k + 1] - 2 * H[k] + H[k - 1], axis=1).mean()) for k in range(1, n - 1) if all(H[j] is not None for j in (k - 1, k, k + 1))]
    res["hands"] = {"frames_with_hand": int(sum(x > 0 for x in nh)), "frames_merged_fingertips": merged,
                    "hand_jerk_mean": round(float(np.mean(hj)), 4) if hj else None, "hand_jerk_p95": round(float(np.percentile(hj, 95)), 4) if hj else None}
    if not a.no_sync:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                res["sync"] = sync_metrics(str(Path(a.video).resolve()), tmp)
            except Exception as e:  # report, never hide
                res["sync"] = {"error": repr(e)[:300]}
    res["per_frame_arcface"] = [round(x, 4) if x is not None else None for x in arc]
    res["elapsed_s"] = round(time.time() - t0, 1)
    out = a.out or str(Path(a.video).with_suffix("")) + "_eval.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame_arcface"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
