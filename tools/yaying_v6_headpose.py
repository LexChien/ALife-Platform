#!/usr/bin/env python3
"""yaying v6: head-pose trajectory (MediaPipe FaceMesh + solvePnP yaw/pitch/roll, nose-tip position) and an oscillation check.
Reports per-axis range/std, direction reversals per second (smoothed, amplitude >= --amp deg), dominant period (FFT), and the
pose values at the ping-pong turnaround frames (anchor f30 / f0) so back-and-forth motion from the anchor order is visible.
  python tools/yaying_v6_headpose.py --video cand.mp4 --out pose.json [--turn 60,120,...] [--plot pose.png]"""
import argparse, json, subprocess
import numpy as np, cv2
import mediapipe as mp

P3 = np.array([[0, 0, 0], [0, -63.6, -12.5], [-43.3, 32.7, -26], [43.3, 32.7, -26], [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]], float)
IDX = [1, 152, 33, 263, 61, 291]


def frames(path):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path]))["streams"][0]
    w, h = s["width"], s["height"]; n, d = map(int, s["r_frame_rate"].split("/"))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), n / d


def reversals(x, amp):
    """count turning points whose swing from the previous turning point is >= amp"""
    ext = [x[0]]; d = 0; c = 0
    for v in x[1:]:
        if d >= 0 and v < ext[-1] - amp: d = -1; ext.append(v); c += 1
        elif d <= 0 and v > ext[-1] + amp: d = 1; ext.append(v); c += 1
        elif (d >= 0 and v > ext[-1]) or (d <= 0 and v < ext[-1]): ext[-1] = v
    return max(c - 1, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--turn", default=""); ap.add_argument("--amp", type=float, default=1.5); ap.add_argument("--plot", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(); F, fps = frames(a.video)
    if a.max_frames: F = F[:a.max_frames]
    h, w = F.shape[1:3]; K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    ypr, nose = [], []
    for f in F:
        r = fm.process(f)
        if not r.multi_face_landmarks: ypr.append([np.nan] * 3); nose.append([np.nan] * 2); continue
        L = r.multi_face_landmarks[0].landmark; p2 = np.array([[L[i].x * w, L[i].y * h] for i in IDX], float)
        ok, rv, tv = cv2.solvePnP(P3, p2, K, None, flags=cv2.SOLVEPNP_ITERATIVE); R, _ = cv2.Rodrigues(rv)
        sy = np.hypot(R[0, 0], R[1, 0]); pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2])); yaw = np.degrees(np.arctan2(-R[2, 0], sy)); roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        ypr.append([yaw, pitch, roll]); nose.append([L[1].x * w, L[1].y * h])
    ypr = np.array(ypr); nose = np.array(nose); ypr[:, 1] = np.unwrap(np.radians(ypr[:, 1])) * 180 / np.pi
    k = np.ones(5) / 5; res = {"video": a.video, "frames": len(F), "fps": fps, "amp_deg": a.amp, "axes": {}}
    for j, name in enumerate(["yaw", "pitch", "roll"]):
        x = ypr[:, j]; x = np.where(np.isnan(x), np.nanmedian(x), x); xs = np.convolve(np.pad(x, 2, mode="edge"), k, "valid")
        spec = np.abs(np.fft.rfft(xs - xs.mean())); fr = np.fft.rfftfreq(len(xs), 1 / fps); i = 1 + int(np.argmax(spec[1:]))
        res["axes"][name] = {"range": round(float(np.ptp(xs)), 2), "std": round(float(xs.std()), 2), "reversals": reversals(xs, a.amp),
                             "reversals_per_s": round(reversals(xs, a.amp) / (len(xs) / fps), 3), "dominant_period_s": round(float(1 / fr[i]), 2),
                             "vel_p95_deg_s": round(float(np.percentile(np.abs(np.diff(xs)) * fps, 95)), 2)}
        if a.turn:
            T = [int(t) for t in a.turn.split(",") if int(t) < len(xs)]; res["axes"][name]["at_turns"] = [round(float(xs[t]), 2) for t in T]
    nz = np.where(np.isnan(nose), np.nanmedian(nose, 0), nose); ns = np.stack([np.convolve(np.pad(nz[:, c], 2, mode="edge"), k, "valid") for c in range(2)], 1)
    res["nose_px"] = {"range_x": round(float(np.ptp(ns[:, 0])), 1), "range_y": round(float(np.ptp(ns[:, 1])), 1),
                      "reversals_x": reversals(ns[:, 0], 2.0), "reversals_y": reversals(ns[:, 1], 2.0), "face_w_px": w}
    res["no_face_frames"] = int(np.isnan(ypr[:, 0]).sum()); res["yaw_pitch_roll"] = np.round(ypr, 2).tolist(); res["nose"] = np.round(nose, 1).tolist()
    json.dump(res, open(a.out, "w"), indent=1)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = np.arange(len(F)) / fps; fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        for j, name in enumerate(["yaw", "pitch", "roll"]): ax[0].plot(t, ypr[:, j], label=name)
        ax[0].legend(); ax[0].set_ylabel("deg"); ax[1].plot(t, nose[:, 0], label="nose x"); ax[1].plot(t, nose[:, 1], label="nose y"); ax[1].legend(); ax[1].set_ylabel("px")
        for tt in ([int(x) for x in a.turn.split(",")] if a.turn else []):
            for axx in ax: axx.axvline(tt / fps, color="k", alpha=0.25, ls="--")
        ax[1].set_xlabel("s"); fig.tight_layout(); fig.savefig(a.plot, dpi=90)
    print(json.dumps({k: v for k, v in res.items() if k not in ("yaw_pitch_roll", "nose")}))


if __name__ == "__main__":
    main()
