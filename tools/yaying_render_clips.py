#!/usr/bin/env python3
"""Plan 40 T5.2: alluring/charming (non-explicit) expression + idle-motion clips of the FIXED avatar, driven by
雅英's OWN motion (MediaPipe series from tools/yaying_motion_profile.py), identity-guarded before install.

  ~/plan38_cache/venv/bin/python tools/yaying_render_clips.py [--only yaying_smile,...]

For each clip: pick the best fully face-tracked window from the motion/voice videos by a measured criterion,
trim + downscale it with ffmpeg, re-enact web/gemma_chat/avatar.jpg (read-only /tmp copy) with LivePortrait
(--flag-crop-driving-video, MPS), boomerang-loop at the avatar's exact pixel size, then tools/appearance_guard.py
(ArcFace + CLIP; Plan 38 J3 thresholds T_ID 0.55 / T_WARN 0.70). Only verdict PASS with a face in every checked frame
is installed into web/gemma_chat/clips/ (private, git-excluded). On a non-PASS the driving multiplier is lowered
(ladder) and re-rendered; a clip that never passes is REJECTED. Existing clips are never touched."""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avatar.identity import AVATAR_PATH, AVATAR_SHA256, VIDEO_PATH, VIDEO_SHA256, sha256_file  # noqa: E402

LP = Path.home() / "plan38_cache/LivePortrait"; LP_PY = Path.home() / "plan38_cache/venv_lp/bin/python"
GUARD_PY = Path.home() / "plan38_cache/venv/bin/python"
W = ROOT / "runs/yaying_clone"; CLIP_DIR = ROOT / "web/gemma_chat/clips"; WORK = W / "clips"
EXP = ["--driving-option", "expression-friendly"]
# name -> (sources, window seconds, criterion, LivePortrait region, multiplier ladder)
SPECS = {
    "yaying_smile": (["motion"], 3.0, "smile_rise", "exp", [0.8, 0.6, 0.45]),
    # 07:45 measured: glance with region "all" drifted (ArcFace min 0.43 @0.7, 0.27 @0.5) -> eyes/expression only
    "yaying_glance": (["motion"], 3.0, "gaze_swing", "exp", [0.8, 0.6, 0.45]),
    "yaying_head_tilt": (["motion"], 3.0, "roll_range", "pose", [0.8, 0.6, 0.4]),
    "yaying_idle_sway": (["motion"], 4.0, "calm_sway", "pose", [0.5, 0.35, 0.25]),
    "yaying_speaking": (["voice"], 4.0, "jaw_var", "lip", [1.0, 0.8, 0.6]),
}


def series(kind):
    d = W / ("motion" if kind == "motion" else "motion_voicevids")
    for f in sorted(d.glob(f"{kind}*.npz")):
        z = np.load(f); yield f.stem, float(z["fps"]), {k: z[k] for k in z.files if k != "fps"}


def pick(spec_name):
    kinds, win, crit, _, _ = SPECS[spec_name]; best = None
    for kind in kinds:
        for stem, fps, s in series(kind):
            if "yaw" not in s: continue
            n = int(win * fps); t = s["t"]
            for a in range(0, len(t) - n, max(1, int(fps / 5))):
                sl = slice(a, a + n)
                if np.isnan(s["yaw"][sl]).any() or np.max(s.get("faces", np.ones(len(t)))[sl]) > 1: continue
                if s.get("face_area") is not None and np.nanmin(s["face_area"][sl]) < 0.004: continue
                sm = (s["mouthSmileLeft"][sl] + s["mouthSmileRight"][sl]) / 2
                gz = ((s["eyeLookOutLeft"] - s["eyeLookInLeft"]) - (s["eyeLookOutRight"] - s["eyeLookInRight"]))[sl] / 2
                rot_speed = np.mean(np.abs(np.diff(s["yaw"][sl])) + np.abs(np.diff(s["pitch"][sl])) + np.abs(np.diff(s["roll"][sl]))) * fps
                if crit == "smile_rise": score = (sm[-n // 3:].mean() - sm[: n // 3].mean()) + 0.3 * sm.max() - 0.004 * rot_speed
                elif crit == "gaze_swing": score = np.ptp(gz) + 0.02 * np.ptp(s["yaw"][sl]) - 0.003 * rot_speed
                elif crit == "roll_range": score = np.ptp(s["roll"][sl]) / 20 - 0.002 * rot_speed
                elif crit == "calm_sway": score = -np.std(s["jawOpen"][sl]) * 4 + min(np.ptp(s["roll"][sl]) + np.ptp(s["yaw"][sl]), 25) / 25 - 0.004 * rot_speed
                else: score = np.std(s["jawOpen"][sl]) * 5 - 0.004 * rot_speed
                if best is None or score > best[0]:
                    best = (float(score), stem, float(t[a]), float(t[a + n - 1]),
                            {"smile_mean": round(float(sm.mean()), 3), "roll_ptp": round(float(np.ptp(s["roll"][sl])), 2),
                             "yaw_ptp": round(float(np.ptp(s["yaw"][sl])), 2), "gaze_ptp": round(float(np.ptp(gz)), 3),
                             "jaw_std": round(float(np.std(s["jawOpen"][sl])), 3), "rot_speed_deg_s": round(float(rot_speed), 1)})
    return best


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=",".join(SPECS)); ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    assert sha256_file(AVATAR_PATH) == AVATAR_SHA256 and sha256_file(VIDEO_PATH) == VIDEO_SHA256, "identity lock violated"
    from PIL import Image
    Wd, Hd = Image.open(AVATAR_PATH).size; W2, H2 = Wd - Wd % 2, Hd - Hd % 2
    WORK.mkdir(parents=True, exist_ok=True)
    man_path = CLIP_DIR / "manifest.json"; manifest = json.loads(man_path.read_text())
    res_path = WORK / "results.json"; results = json.loads(res_path.read_text()) if res_path.exists() else {}
    src = Path(tempfile.mkdtemp(prefix="yy_clip_")) / "avatar.jpg"; shutil.copy(AVATAR_PATH, src)
    for name in [n for n in a.only.split(",") if n]:
        if results.get(name, {}).get("installed"): print("skip", name); continue
        kinds, win, crit, region, ladder = SPECS[name]
        sel = pick(name)
        if sel is None: results[name] = {"error": "no fully tracked window"}; continue
        score, stem, t0, t1, feats = sel
        out = WORK / name; shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        drv = out / "driving.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", ROOT / f"runs/yaying_clone/raw/{stem}.mp4",
             "-vf", "scale=-2:1280,fps=25", "-an", "-c:v", "libx264", "-crf", "16", drv])
        entry = {"driving_source": f"{stem}.mp4", "window_s": [round(t0, 2), round(t1, 2)], "criterion": crit, "criterion_score": round(score, 4),
                 "window_features": feats, "region": region, "attempts": [], "installed": False, "guard_pass": False,
                 "source_sha256": sha256_file(src)}
        for mult in ladder:
            r0 = time.time(); lo = out / f"m{mult}"; lo.mkdir()
            args = ["--animation-region", region, "--driving-multiplier", str(mult), "--flag-crop-driving-video", *EXP]
            r = run([LP_PY, "inference.py", "-s", src, "-d", drv, "-o", lo, *args], cwd=LP, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
            (lo / "liveportrait.log").write_text(r.stdout[-3000:] + r.stderr[-3000:])
            raw = [p for p in lo.glob("*.mp4") if "concat" not in p.name]
            att = {"multiplier": mult, "args": args, "lp_exit": r.returncode, "render_s": round(time.time() - r0, 1)}
            if r.returncode != 0 or not raw: att["error"] = "liveportrait failed"; entry["attempts"].append(att); continue
            loop = lo / f"{name}.mp4"
            vf = (f"[0:v]scale={Wd}:{Hd}:flags=lanczos,crop={W2}:{H2}:0:0,setsar=1,split[a][b];[b]reverse[r];"
                  f"[a][r]concat=n=2:v=1:a=0,format=yuv420p[v]")
            f = run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw[0], "-filter_complex", vf, "-map", "[v]", "-an", "-c:v", "libx264",
                     "-crf", "18", "-preset", "slow", "-movflags", "+faststart", loop])
            if f.returncode != 0: att["error"] = "ffmpeg " + f.stderr[-200:]; entry["attempts"].append(att); continue
            rep = lo / "guard.json"
            g = run([GUARD_PY, ROOT / "tools/appearance_guard.py", "check", f"{name}={loop}", "--every", a.every, "--report", rep], cwd=ROOT)
            s = (json.loads(rep.read_text()) if rep.exists() else {}).get("positive", {}).get(name, {})
            att.update({"guard_exit": g.returncode, "frames_checked": s.get("n"), "faces_found": s.get("faces_found"), "arcface": s.get("arcface"),
                        "clip_full": s.get("clip_full"), "verdicts": s.get("verdicts"), "verdict": s.get("verdict"),
                        "guard_report": str(rep.relative_to(ROOT))})
            entry["attempts"].append(att); print(name, json.dumps(att), flush=True)
            if s.get("verdict") == "PASS" and s.get("n") and s.get("faces_found") == s.get("n"):
                shutil.copy(loop, CLIP_DIR / f"{name}.mp4")
                entry.update({"installed": True, "guard_pass": True, "multiplier": mult, "args": args, "arcface": s.get("arcface"),
                              "clip_full": s.get("clip_full"), "verdicts": s.get("verdicts"), "verdict": "PASS", "frames_checked": s.get("n"),
                              "faces_found": s.get("faces_found"), "guard_report": att["guard_report"],
                              "thresholds": {"T_ID": 0.55, "T_WARN": 0.7, "T_CLIP": 0.78, "T_NEG": 0.4},
                              "sha256": sha256_file(CLIP_DIR / f"{name}.mp4"), "tool": "tools/yaying_render_clips.py", "plan": "40"})
                break
        if not entry["installed"]: entry["status"] = "rejected"
        results[name] = entry; res_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        if entry["installed"]:
            manifest = json.loads(man_path.read_text()); manifest["clips"][name] = entry
            manifest["updated_at_plan40"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    assert sha256_file(AVATAR_PATH) == AVATAR_SHA256 and sha256_file(VIDEO_PATH) == VIDEO_SHA256, "identity lock violated during render"
    print("DONE", json.dumps({k: (v.get("installed"), (v.get("arcface") or {}).get("min")) for k, v in results.items()}))


if __name__ == "__main__":
    main()
