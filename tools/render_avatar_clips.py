#!/usr/bin/env python3
"""Plan 38 J3.1: offline LivePortrait expression clips of the FIXED avatar, each identity-guarded before install.

  render_avatar_clips.py [--only idle,smile,concerned] [--every 2]

For every clip: LivePortrait (KwaiVGI; ~/plan38_cache/LivePortrait + venv_lp, MPS) re-enacts web/gemma_chat/avatar.jpg
(read-only copy in /tmp) with a stock driving motion, the result is scaled to the avatar's exact pixel size, made a
seamless boomerang loop (H.264, no audio), then checked with tools/appearance_guard.py (ArcFace + CLIP, T_ID 0.55 /
T_WARN 0.70). Only verdict == PASS with a face in every checked frame is INSTALLED into web/gemma_chat/clips/
(private, git-excluded); WARN = needs Lex review, FAIL = rejected. manifest.json records every result.
The avatar SHA-256 is asserted before and after (identity lock). Nothing here edits avatar.jpg."""
import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avatar.identity import AVATAR_PATH, AVATAR_SHA256, sha256_file  # noqa: E402

LP = Path.home() / "plan38_cache/LivePortrait"
LP_PY = Path.home() / "plan38_cache/venv_lp/bin/python"
GUARD_PY = Path.home() / "plan38_cache/venv/bin/python"
CLIP_DIR = ROOT / "web/gemma_chat/clips"
WORK = ROOT / "runs/plan38/avatar/clips"
# name -> (driving template, LivePortrait extra args). idle = eyes only (blink/gaze, no head/mouth motion).
SPECS = {
    "idle": ("d0.mp4", ["--animation_region", "eyes"]),
    "smile": ("laugh.pkl", ["--animation_region", "exp", "--driving_multiplier", "0.6"]),
    "concerned": ("aggrieved.pkl", ["--animation_region", "exp", "--driving_multiplier", "0.6"]),
}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SPECS))
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    assert sha256_file(AVATAR_PATH) == AVATAR_SHA256, "avatar.jpg changed - identity lock violated"
    from PIL import Image
    W, H = Image.open(AVATAR_PATH).size
    W2, H2 = W - W % 2, H - H % 2  # yuv420p needs even dims; crop 1 px max, no rescale of the face
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    man_path = CLIP_DIR / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"clips": {}}
    src = Path(tempfile.mkdtemp(prefix="p38_clip_")) / "avatar.jpg"
    shutil.copy(AVATAR_PATH, src)
    for name in [n for n in a.only.split(",") if n]:
        drv, extra = SPECS[name]
        out = WORK / name
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        t0 = time.time()
        r = run([LP_PY, "inference.py", "-s", src, "-d", f"assets/examples/driving/{drv}", "-o", out, *extra], cwd=LP,
                env={**__import__("os").environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        (out / "liveportrait.log").write_text(r.stdout[-4000:] + r.stderr[-4000:])
        raw = [p for p in out.glob("*.mp4") if "concat" not in p.name]
        entry = {"driving": drv, "args": extra, "render_s": round(time.time() - t0, 1), "lp_exit": r.returncode,
                 "source_sha256": sha256_file(src), "guard_pass": False, "installed": False}
        if r.returncode != 0 or not raw:
            entry["error"] = "liveportrait failed"; manifest["clips"][name] = entry; print(name, entry); continue
        loop = out / f"{name}.mp4"
        vf = (f"[0:v]scale={W}:{H}:flags=lanczos,crop={W2}:{H2}:0:0,setsar=1,split[a][b];[b]reverse[r];"
              f"[a][r]concat=n=2:v=1:a=0,format=yuv420p[v]")
        f = run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw[0], "-filter_complex", vf, "-map", "[v]", "-an",
                 "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", loop])
        if f.returncode != 0:
            entry["error"] = "ffmpeg: " + f.stderr[-300:]; manifest["clips"][name] = entry; print(name, entry); continue
        rep_path = out / "guard.json"
        g = run([GUARD_PY, ROOT / "tools/appearance_guard.py", "check", f"{name}={loop}", "--every", a.every,
                 "--report", rep_path], cwd=ROOT)
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        s = (rep.get("positive") or {}).get(name, {})
        entry.update({"guard_exit": g.returncode, "frames_checked": s.get("n"), "faces_found": s.get("faces_found"),
                      "arcface": s.get("arcface"), "clip_full": s.get("clip_full"), "verdicts": s.get("verdicts"),
                      "verdict": s.get("verdict"), "guard_report": str(rep_path.relative_to(ROOT)),
                      "thresholds": rep.get("thresholds")})
        entry["guard_pass"] = bool(s.get("verdict") == "PASS" and s.get("n") and s.get("faces_found") == s.get("n"))
        if entry["guard_pass"]:
            shutil.copy(loop, CLIP_DIR / f"{name}.mp4")
            entry["installed"] = True
            entry["sha256"] = sha256_file(CLIP_DIR / f"{name}.mp4")
        else:
            (CLIP_DIR / f"{name}.mp4").unlink(missing_ok=True)
            entry["status"] = "needs_lex_review" if s.get("verdict") == "WARN" else "rejected"
        manifest["clips"][name] = entry
        print(name, json.dumps({k: entry.get(k) for k in ("render_s", "verdict", "arcface", "faces_found",
                                                         "frames_checked", "installed")}), flush=True)
    manifest.update({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "avatar_sha256": sha256_file(AVATAR_PATH),
                     "tool": "tools/render_avatar_clips.py"})
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    assert sha256_file(AVATAR_PATH) == AVATAR_SHA256, "avatar.jpg changed during render"
    print("manifest", man_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
