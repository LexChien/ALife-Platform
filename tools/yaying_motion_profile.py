#!/usr/bin/env python3
"""Plan 40 T5.1: face blendshapes / head pose / upper-body pose from the 雅英 motion videos (MediaPipe Tasks).

  ~/yaying_cache/venv_mp/bin/python tools/yaying_motion_profile.py  (mediapipe 0.10.21; 1.0.1 Tasks abort: graph_service "Service is unavailable") runs/yaying_clone/raw/motion*.mp4 [--out runs/yaying_clone/motion]

Per video: per-frame series (npz) + stats: blink rate, smile dynamics, gaze, head yaw/pitch/roll, brow/lip motion,
tempo (dominant frequency of head/body motion). Largest detected face per frame is used. Read-only on inputs."""
import argparse, json, math, sys
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MP = Path.home() / "yaying_cache/mp"
KEYS = ["eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight", "jawOpen", "browInnerUp", "browOuterUpLeft",
        "browOuterUpRight", "browDownLeft", "browDownRight", "eyeLookOutLeft", "eyeLookInLeft", "eyeLookOutRight",
        "eyeLookInRight", "eyeLookUpLeft", "eyeLookDownLeft", "mouthPucker", "cheekSquintLeft", "cheekSquintRight",
        "eyeSquintLeft", "eyeSquintRight", "mouthFunnel", "mouthRollLower", "mouthPressLeft"]


def euler(m):
    r = np.asarray(m)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2])); yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def dom_freq(x, fps):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 16: return None
    x = x - x.mean(); f = np.fft.rfftfreq(len(x), 1 / fps); p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (f >= 0.15) & (f <= 4.0)
    return round(float(f[band][np.argmax(p[band])]), 3) if band.any() else None


def st(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x): return None
    return {k: round(float(v), 4) for k, v in dict(mean=x.mean(), std=x.std(), p05=np.percentile(x, 5), p50=np.median(x),
                                                  p95=np.percentile(x, 95), min=x.min(), max=x.max()).items()}


def blinks(b, t, fps, thr=0.55):
    # count closures only inside contiguous face-tracked runs (gaps would fake transitions)
    on = b > thr; cont = np.diff(t) < 1.5 / fps
    n = int(np.sum(on[1:] & ~on[:-1] & cont)); return n, round(n / (len(b) / fps) * 60, 2)


def analyse(path: Path, out: Path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / 30)); efps = fps / step
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MP / "face_landmarker.task"), delegate=mpt.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.IMAGE,
        num_faces=3, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MP / "pose_landmarker_full.task"), delegate=mpt.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.VIDEO))
    rows, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step: i += 1; continue
        H0, W0 = fr.shape[:2]; s = min(1.0, 1280 / max(H0, W0))
        small = cv2.resize(fr, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else fr
        ts = int(i / fps * 1000)
        p = pl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), ts)
        # face: crop the head at FULL resolution around pose landmarks 0-10 (dance shots have small faces)
        crop = fr; cropped = False
        if p.pose_landmarks:
            P = p.pose_landmarks[0][:11]; xs = [q.x * W0 for q in P]; ys = [q.y * H0 for q in P]
            cx, cy = float(np.mean(xs)), float(np.mean(ys)); half = max(80.0, 1.6 * (max(xs) - min(xs)), 1.6 * (max(ys) - min(ys)))
            x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W0, cx + half)), int(min(H0, cy + half))
            if x1 - x0 > 40 and y1 - y0 > 40: crop = fr[y0:y1, x0:x1]; cropped = True
        if max(crop.shape[:2]) > 1280:
            c = 1280 / max(crop.shape[:2]); crop = cv2.resize(crop, (int(crop.shape[1] * c), int(crop.shape[0] * c)), interpolation=cv2.INTER_AREA)
        r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)))
        if not r.face_landmarks and cropped:
            r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        row = {"t": i / fps, "faces": len(r.face_landmarks)}
        if r.face_landmarks:
            areas = [(max(l.x for l in L) - min(l.x for l in L)) * (max(l.y for l in L) - min(l.y for l in L)) for L in r.face_landmarks]
            k = int(np.argmax(areas)); row["face_area"] = areas[k]
            bs = {c.category_name: c.score for c in r.face_blendshapes[k]}
            row.update({kk: bs.get(kk, np.nan) for kk in KEYS})
            row["yaw"], row["pitch"], row["roll"] = euler(r.facial_transformation_matrixes[k])
            L = r.face_landmarks[k]; row["nose_x"], row["nose_y"] = L[1].x, L[1].y
        if p.pose_landmarks:
            P = p.pose_landmarks[0]; ls, rs = P[11], P[12]
            if min(ls.visibility, rs.visibility) > 0.5:
                row["shoulder_tilt"] = math.degrees(math.atan2(ls.y - rs.y, ls.x - rs.x))
                row["sh_mid_x"], row["sh_mid_y"] = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
        rows.append(row); i += 1
    cap.release()
    keys = sorted({k for r in rows for k in r})
    arr = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}
    np.savez_compressed(out / f"{path.stem}.npz", fps=efps, **arr)
    has = ~np.isnan(arr.get("yaw", np.full(len(rows), np.nan)))
    res = {"video": path.name, "fps_analysed": round(efps, 3), "frames": len(rows), "face_frames": int(has.sum()),
           "face_coverage": round(float(has.mean()), 3), "multi_face_frames": int(np.sum(arr["faces"] > 1))}
    if has.sum() > 10:
        b = np.nan_to_num((arr["eyeBlinkLeft"] + arr["eyeBlinkRight"]) / 2)[has]
        smile = ((arr["mouthSmileLeft"] + arr["mouthSmileRight"]) / 2)[has]
        gaze_h = ((arr["eyeLookOutLeft"] - arr["eyeLookInLeft"]) - (arr["eyeLookOutRight"] - arr["eyeLookInRight"]))[has] / 2
        nb, rate = blinks(b, arr['t'][has], efps)
        res.update({
            "blinks": nb, "blink_rate_per_min": rate, "smile": st(smile),
            "smile_fraction_gt_0.5": round(float(np.mean(smile > 0.5)), 3),
            "smile_onset_speed_per_s": round(float(np.percentile(np.diff(smile) * efps, 95)), 3),
            "smile_asymmetry_mean": round(float(np.nanmean((arr["mouthSmileLeft"] - arr["mouthSmileRight"])[has])), 4),
            "gaze_horizontal": st(gaze_h), "gaze_up": st(arr["eyeLookUpLeft"][has]), "gaze_down": st(arr["eyeLookDownLeft"][has]),
            "eye_squint": st(((arr["eyeSquintLeft"] + arr["eyeSquintRight"]) / 2)[has]),
            "head_yaw_deg": st(arr["yaw"][has]), "head_pitch_deg": st(arr["pitch"][has]), "head_roll_deg": st(arr["roll"][has]),
            "head_roll_abs_gt_8deg_fraction": round(float(np.mean(np.abs(arr["roll"][has]) > 8)), 3),
            "brow_inner_up": st(arr["browInnerUp"][has]), "brow_outer_up": st(((arr["browOuterUpLeft"] + arr["browOuterUpRight"]) / 2)[has]),
            "jaw_open": st(arr["jawOpen"][has]), "mouth_pucker": st(arr["mouthPucker"][has]),
            "lip_motion_speed": round(float(np.nanmean(np.abs(np.diff(arr["jawOpen"][has])) * efps)), 4),
            "head_angular_speed_deg_s": round(float(np.nanmean(np.sqrt(np.diff(arr["yaw"][has]) ** 2 + np.diff(arr["pitch"][has]) ** 2
                                                                       + np.diff(arr["roll"][has]) ** 2) * efps)), 2),
            "tempo_hz": {"yaw": dom_freq(arr["yaw"], efps), "roll": dom_freq(arr["roll"], efps), "pitch": dom_freq(arr["pitch"], efps),
                         "smile": dom_freq(arr["mouthSmileLeft"], efps)}})
    if "shoulder_tilt" in arr:
        hs = ~np.isnan(arr["shoulder_tilt"])
        res["body"] = {"pose_frames": int(hs.sum()), "shoulder_tilt_deg": st(arr["shoulder_tilt"]),
                       "sway_x_std": round(float(np.nanstd(arr["sh_mid_x"])), 4), "bob_y_std": round(float(np.nanstd(arr["sh_mid_y"])), 4),
                       "sway_tempo_hz": dom_freq(arr["sh_mid_x"], efps), "bob_tempo_hz": dom_freq(arr["sh_mid_y"], efps)}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--out", default="runs/yaying_clone/motion")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allr = {}
    for v in a.videos:
        r = analyse(Path(v), out); allr[Path(v).stem] = r; print(json.dumps(r, ensure_ascii=False)[:600], flush=True)
    (out / "motion_profile.json").write_text(json.dumps(allr, indent=1, ensure_ascii=False))
    print("WROTE", out / "motion_profile.json")


if __name__ == "__main__":
    sys.exit(main())
