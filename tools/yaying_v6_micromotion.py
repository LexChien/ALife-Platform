#!/usr/bin/env python3
"""yaying v6: natural micro-motion profile (to catch 'too static / stiff' results).
MediaPipe FaceMesh (refine_landmarks) per frame:
  head : nose-tip and face-centre motion, normalised by inter-ocular distance (IOD): std, mean |velocity| (IOD/s), path length/s
  eyes : eye-aspect-ratio (EAR) per eye -> blinks (EAR < 0.6*median for >=2 frames), blinks/min; iris offset inside the eye
         (gaze proxy, fraction of eye width): std and mean |velocity|
  brow : brow-to-eye distance / IOD std (expression liveliness)
  python tools/yaying_v6_micromotion.py --video v.mp4 [--frames 0:40] --out mm.json"""
import argparse, json, subprocess
import numpy as np
import mediapipe as mp

L_EYE = (33, 133, 159, 145, 158, 153, 160, 144); R_EYE = (263, 362, 386, 374, 385, 380, 387, 373)
L_IRIS, R_IRIS = 468, 473; L_BROW, R_BROW = 105, 334


def frames(path):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path]))["streams"][0]
    w, h = s["width"], s["height"]; n, d = map(int, s["r_frame_rate"].split("/"))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), n / d


def ear(P, e):
    c0, c1, a1, b1, a2, b2, a3, b3 = (P[i] for i in e)
    return (np.linalg.norm(a1 - b1) + np.linalg.norm(a2 - b2) + np.linalg.norm(a3 - b3)) / (3 * np.linalg.norm(c0 - c1) + 1e-6)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--frames", default="")
    a = ap.parse_args(); F, fps = frames(a.video)
    if a.frames: s, e = (int(x) for x in a.frames.split(":")); F = F[s:e]
    h, w = F.shape[1:3]
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    rows = []
    for f in F:
        r = fm.process(f)
        if not r.multi_face_landmarks: rows.append(None); continue
        P = np.array([[l.x * w, l.y * h] for l in r.multi_face_landmarks[0].landmark])
        iod = np.linalg.norm(P[33] - P[263]); ctr = P[[33, 263, 1, 152]].mean(0)
        el, er = ear(P, L_EYE), ear(P, R_EYE)
        gl = np.dot(P[L_IRIS] - (P[33] + P[133]) / 2, (P[133] - P[33])) / (np.linalg.norm(P[133] - P[33]) ** 2)
        gr = np.dot(P[R_IRIS] - (P[263] + P[362]) / 2, (P[362] - P[263])) / (np.linalg.norm(P[362] - P[263]) ** 2)
        brow = (np.linalg.norm(P[L_BROW] - P[159]) + np.linalg.norm(P[R_BROW] - P[386])) / 2 / iod
        rows.append([P[1, 0], P[1, 1], ctr[0], ctr[1], iod, (el + er) / 2, (gl + gr) / 2, brow])
    ok = [r for r in rows if r is not None]; A = np.array(ok); iod = np.median(A[:, 4]); dur = len(F) / fps
    nose = A[:, 0:2] / iod; ctr = A[:, 2:4] / iod
    v = np.linalg.norm(np.diff(nose, axis=0), axis=1) * fps
    E = A[:, 5]; thr = 0.6 * np.median(E); closed = E < thr; blinks = 0; run = 0
    for c in closed:
        run = run + 1 if c else 0
        if run == 2: blinks += 1
    g = A[:, 6]; br = A[:, 7]
    res = {"video": a.video, "frames": len(F), "fps": fps, "face_frames": len(ok), "duration_s": round(dur, 2),
           "head": {"nose_std_iod": [round(float(x), 4) for x in nose.std(0)], "centre_std_iod": [round(float(x), 4) for x in ctr.std(0)],
                    "nose_speed_mean_iod_s": round(float(v.mean()), 4), "nose_speed_p95_iod_s": round(float(np.percentile(v, 95)), 4),
                    "still_frac_speed_lt_0.05": round(float(np.mean(v < 0.05)), 3)},
           "eyes": {"blinks": blinks, "blinks_per_min": round(blinks / dur * 60, 1), "ear_median": round(float(np.median(E)), 4),
                    "ear_cv": round(float(E.std() / E.mean()), 4), "gaze_std": round(float(g.std()), 4),
                    "gaze_speed_mean_s": round(float(np.abs(np.diff(g)).mean() * fps), 4)},
           "brow_std_iod": round(float(br.std()), 4), "iod_px": round(float(iod), 1)}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))


if __name__ == "__main__":
    main()
