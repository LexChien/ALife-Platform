#!/usr/bin/env python3
"""Plan 40 demo v5: SUBTLE eye/brow-only animation (half-lid softening, sidelong glance to camera, slow blink, slight
brow lift) with LivePortrait used ONLY as a motion-field generator for the eye/brow region.

Per frame of work/v5_base.npy: LivePortrait renders the frame twice from its own crop - (a) neutral reconstruction,
(b) the same with eye-only deltas: eye-open retargeting (stitching_retargeting_module['eye'], relative to the
neutral retarget so the learned bias cancels), eyeball direction and eyebrow keypoint offsets (the official
image-retargeting formulas), lips untouched, stitching on. The DIS optical flow (b -> a) inside a feathered
eye+brow mask is then applied to the ORIGINAL pixels (cubic remap) - so her exact skin, lashes, light and grain are
kept and nothing outside the mask changes. Only for the deepest part of a slow blink (open factor < --blend-below)
is a colour-matched share of render (b) mixed into the eye core. Frames with neutral parameters are copied untouched.
Keyframed curves (cosine-eased) are in output seconds; --strength scales every deviation from neutral.
  cd ~/plan38_cache/LivePortrait && ../venv_lp/bin/python <repo>/tools/yaying_demo_v5_eyes.py
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
LP = Path.home() / "plan38_cache/LivePortrait"
sys.path.insert(0, str(LP))
D = ROOT / "runs/yaying_clone/demo_v2"
FPS = 24

# (t_seconds, value) keyframes; cosine-eased between keys. Sections (v5 timeline): opening 0-3.54 s (finger at lips),
# speech 3.54-8.83 s, ending 8.83-15.7 s (hand to lips, lingering touch).
# EYE = eye-open factor applied to the measured source eye-open ratio (1 = as filmed, 0 = closed). Measured 12:41: the
# retarget module closes the far (viewer-left, foreshortened) eye much less at partial targets, so the blink goes to
# ~0.04 for a symmetric close; half-lid softening is 0.66-0.8.
EYE = [(0, 1.0), (1.3, 1.0), (1.9, 0.7), (2.5, 0.7), (2.78, 0.04), (2.9, 0.04), (3.3, 0.8), (3.54, 0.85),
       (5.68, 0.85), (5.9, 0.04), (6.0, 0.04), (6.4, 0.85), (8.0, 0.88), (8.8, 1.0), (9.3, 1.0), (9.9, 0.7), (10.6, 0.92),
       (12.4, 0.8), (12.65, 0.04), (12.77, 0.04), (13.2, 0.75), (14.2, 0.66), (15.8, 0.72)]
GAZE_X = [(0, 7.0), (0.35, 7.0), (1.05, 0.0), (5.0, 0.0), (5.25, 3.5), (5.6, 3.5), (5.9, 0.0), (9.5, 0.0), (10.0, -6.0),
          (10.7, -6.0), (11.3, 0.0), (15.8, 0.0)]
BROW = [(0, 0.0), (0.6, 0.0), (1.0, 5.0), (1.9, 0.0), (3.6, 0.0), (3.95, 3.0), (4.7, 0.0), (11.1, 0.0), (11.45, 4.0),
        (12.3, 0.0), (15.8, 0.0)]


def curve(keys, t):
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1]
    if t >= ts[-1]:
        return keys[-1][1]
    j = int(np.searchsorted(ts, t, side="right")) - 1
    (t0, v0), (t1, v1) = keys[j], keys[j + 1]
    u = (t - t0) / max(1e-9, t1 - t0)
    return v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * u))


def eye_mask(lmk, h, w, core=False):
    m = np.zeros((h, w), np.float32)
    ew_all = []
    for sl in (slice(0, 24), slice(24, 48)):
        pts = lmk[sl]; c = pts.mean(0); ew = float(np.ptp(pts[:, 0])); ew_all.append(ew)
        if core:
            cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (int(0.62 * ew), int(0.42 * ew)), 0, 0, 360, 1.0, -1)
        else:
            cv2.ellipse(m, (int(round(c[0])), int(round(c[1] - 0.3 * ew))), (int(0.9 * ew), int(0.82 * ew)), 0, 0, 360, 1.0, -1)
    s = 0.15 * float(np.mean(ew_all))
    return cv2.GaussianBlur(m, (0, 0), max(1.0, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-npy", default="runs/yaying_clone/demo_v2/work/v5_base.npy")
    ap.add_argument("--out-npy", default="runs/yaying_clone/demo_v2/work/v5_eyes.npy")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--blend-below", type=float, default=0.3)
    ap.add_argument("--blend-max", type=float, default=0.6)
    ap.add_argument("--frames", default="", help="debug: comma list of frames to render only")
    ap.add_argument("--part", default="", help="k/N: render frames k, k+N, ... (parallel parts); merge with --merge N")
    ap.add_argument("--merge", type=int, default=0)
    a = ap.parse_args()
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.utils.camera import get_rotation_matrix
    from src.utils.retargeting_utils import calc_eye_close_ratio
    t0 = time.time()
    if a.merge:
        base = np.load(ROOT / a.base_npy); out = base.copy(); rows = []
        for k in range(a.merge):
            pk = np.load(ROOT / a.out_npy.replace(".npy", f"_p{k}.npy"), mmap_mode="r"); out[k::a.merge] = pk[k::a.merge]
            rows += json.load(open(ROOT / a.out_npy.replace(".npy", f"_p{k}_rows.json")))["rows"]
        rows.sort(key=lambda r: r["i"]); np.save(ROOT / a.out_npy, out)
        json.dump({"params": vars(a), "keys": {"eye": EYE, "gaze_x": GAZE_X, "brow": BROW}, "rows": rows},
                  open(ROOT / a.out_npy.replace(".npy", "_rows.json"), "w"), indent=1)
        print("MERGED", len(rows), sum(r.get("edited", False) for r in rows)); return
    if a.part:
        kk, NN = (int(v) for v in a.part.split("/")); a.out_npy = a.out_npy.replace(".npy", f"_p{kk}.npy")
    inf_cfg = InferenceConfig(); crop_cfg = CropConfig()
    W = LivePortraitWrapper(inference_cfg=inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg, image_type="human_face", flag_force_cpu=False)
    dev = W.device
    base = np.load(ROOT / a.base_npy); n, h, wd = base.shape[:3]
    out = base.copy()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rows = []
    idx = [int(v) for v in a.frames.split(",")] if a.frames else (range(kk, n, NN) if a.part else range(n))
    for i in idx:
        t = i / FPS
        f = 1.0 + (curve(EYE, t) - 1.0) * a.strength
        gx = curve(GAZE_X, t) * a.strength
        br = float(curve(BROW, t) * a.strength); f = float(f); gx = float(gx)
        row = {"i": i, "t": round(t, 3), "eye_factor": round(f, 3), "gaze_x": round(gx, 2), "brow": round(br, 2)}
        if abs(f - 1) < 0.01 and abs(gx) < 0.1 and abs(br) < 0.1:
            row["edited"] = False; rows.append(row); continue
        img = base[i]
        ci = cropper.crop_source_image(img, crop_cfg)
        if ci is None:
            row["edited"] = False; row["note"] = "no face"; rows.append(row); continue
        with torch.no_grad():
            I_s = W.prepare_source(ci["img_crop_256x256"])
            info = W.get_kp_info(I_s); f_s = W.extract_feature_3d(I_s); x_s = W.transform_keypoint(info)
            R = get_rotation_matrix(info["pitch"], info["yaw"], info["roll"])
            exp0 = info["exp"].clone(); exp1 = info["exp"].clone()
            gxt = float(gx)
            if gx > 0:
                exp1[0, 11, 0] += gxt * 0.0007; exp1[0, 15, 0] += gxt * 0.001
            else:
                exp1[0, 11, 0] += gxt * 0.001; exp1[0, 15, 0] += gxt * 0.0007
            if br > 0:
                exp1[0, 1, 1] += br * 0.001; exp1[0, 2, 1] += br * -0.001
            elif br < 0:
                exp1[0, 1, 0] += br * -0.001; exp1[0, 2, 0] += br * 0.001; exp1[0, 1, 1] += br * 0.0003; exp1[0, 2, 1] += br * -0.0003
            xd0 = info["scale"] * (info["kp"] @ R + exp0) + info["t"]
            xd1 = info["scale"] * (info["kp"] @ R + exp1) + info["t"]
            lmk = ci["lmk_crop"]
            cs = calc_eye_close_ratio(lmk[None]); m_ = float(cs.mean())
            if abs(f - 1) >= 0.01:
                cst = torch.from_numpy(cs).float().to(dev)
                d1 = W.retarget_eye(x_s, torch.cat([cst, torch.tensor([[f * m_]], device=dev)], 1))
                d0 = W.retarget_eye(x_s, torch.cat([cst, torch.tensor([[m_]], device=dev)], 1))
                xd1 = xd1 + (d1 - d0)
            xd0 = W.stitching(x_s, xd0); xd1 = W.stitching(x_s, xd1)
            r0 = W.parse_output(W.warp_decode(f_s, x_s, xd0)["out"])[0]
            r1 = W.parse_output(W.warp_decode(f_s, x_s, xd1)["out"])[0]
        if a.frames:
            cv2.imwrite(str(D / f"work/dbg_eye_r_{i:03d}.png"), np.concatenate([r0, r1], 1)[:, :, ::-1])
        M = ci["M_c2o"][:2].astype(np.float32)
        R0 = cv2.warpAffine(r0, M, (wd, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        R1 = cv2.warpAffine(r1, M, (wd, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mk = eye_mask(lmk, h, wd)
        ys, xs = np.where(mk > 0.01)
        y0, y1 = max(0, ys.min() - 8), min(h, ys.max() + 9); x0, x1 = max(0, xs.min() - 8), min(wd, xs.max() + 9)
        g1 = cv2.cvtColor(R1[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY); g0 = cv2.cvtColor(R0[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        fl = dis.calc(g1, g0, None)
        fl = cv2.GaussianBlur(fl, (0, 0), 1.0)
        mag = np.linalg.norm(fl, axis=2)
        gate = np.clip((mag - 0.08) / 0.25, 0, 1)
        mr = mk[y0:y1, x0:x1] * gate
        gy_, gx_ = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        warped = cv2.remap(img, gx_ + fl[..., 0] * mr, gy_ + fl[..., 1] * mr, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        roi = img[y0:y1, x0:x1].astype(np.float32)
        wsoft = cv2.GaussianBlur(mk[y0:y1, x0:x1], (0, 0), 1.0)[..., None]
        res = roi * (1 - wsoft) + warped.astype(np.float32) * wsoft
        wb = float(np.clip((a.blend_below - f) / 0.25, 0, 1)) * a.blend_max
        if wb > 0:                                        # deepest blink: colour-matched share of the render
            core = eye_mask(lmk, h, wd, core=True)[y0:y1, x0:x1]
            rr = R1[y0:y1, x0:x1].astype(np.float32); sel = core > 0.3
            for c in range(3):
                mu_r, sd_r = rr[..., c][sel].mean(), rr[..., c][sel].std() + 1e-3
                mu_o, sd_o = res[..., c][sel].mean(), res[..., c][sel].std() + 1e-3
                rr[..., c] = (rr[..., c] - mu_r) * (sd_o / sd_r) + mu_o
            res = res * (1 - (core * wb)[..., None]) + rr * (core * wb)[..., None]
        out[i, y0:y1, x0:x1] = np.clip(res.round(), 0, 255).astype(np.uint8)
        full_mask = np.zeros((h, wd), np.float32); full_mask[y0:y1, x0:x1] = wsoft[..., 0]
        diff_out = np.abs(out[i].astype(np.int16) - img.astype(np.int16)).max(2) > 0
        row.update({"edited": True, "flow_max_px": round(float((mag * mk[y0:y1, x0:x1]).max()), 2), "render_blend": round(wb, 3),
                    "roi": [int(x0), int(y0), int(x1), int(y1)], "changed_px_outside_mask": int((diff_out & (full_mask <= 0.01)).sum())})
        rows.append(row)
        if i % 20 == 0:
            print(i, row, round(time.time() - t0, 1), flush=True)
    if not a.frames:
        np.save(ROOT / a.out_npy, out)
    else:
        for i in idx:
            cv2.imwrite(str(D / f"work/dbg_eye_{i:03d}.png"), np.concatenate([base[i], out[i]], 1)[:, :, ::-1])
    json.dump({"params": vars(a), "keys": {"eye": EYE, "gaze_x": GAZE_X, "brow": BROW}, "rows": rows},
              open(ROOT / (a.out_npy.replace(".npy", "_rows.json")), "w"), indent=1)
    print("EYES_DONE", sum(r.get("edited", False) for r in rows), "edited of", len(rows), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
