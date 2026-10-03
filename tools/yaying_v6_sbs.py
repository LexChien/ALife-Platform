#!/usr/bin/env python3
"""yaying v6: side-by-side keyframes (original reference frame vs candidate at N times) + window-seam metric.
Seam metric: mean |frame(t) - frame(t-1)| (luma) at given seam frames divided by the median step elsewhere (1.0 = invisible).
  python tools/yaying_v6_sbs.py --video cand.mp4 --ref ref.png --out sbs.png [--seams 44,88,...] [--times 0.5,3,...]"""
import argparse, json, subprocess
import numpy as np
from PIL import Image, ImageDraw


def read_frames(path):
    info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                                    "-of", "json", path]); s = json.loads(info)["streams"][0]
    w, h = s["width"], s["height"]; n, d = map(int, s["r_frame_rate"].split("/"))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), n / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--times", default=""); ap.add_argument("--seams", default=""); ap.add_argument("--label", default="")
    a = ap.parse_args()
    fr, fps = read_frames(a.video); n = len(fr)
    times = [float(t) for t in a.times.split(",")] if a.times else list(np.linspace(0.2, n / fps - 0.2, 6))
    H = 330; ref = Image.open(a.ref).convert("RGB"); rw = int(ref.width * H / ref.height); ref = ref.resize((rw, H))
    tiles = [ref]
    for t in times:
        i = min(n - 1, int(round(t * fps))); im = Image.fromarray(fr[i]); im = im.resize((int(im.width * H / im.height), H))
        ImageDraw.Draw(im).text((4, 4), f"t={t:.1f}s", fill=(255, 255, 0)); tiles.append(im)
    W = sum(t.width for t in tiles); canvas = Image.new("RGB", (W, H + 16), (0, 0, 0)); x = 0
    for k, t in enumerate(tiles):
        canvas.paste(t, (x, 16)); ImageDraw.Draw(canvas).text((x + 4, 2), "original ref" if k == 0 else a.label, fill=(255, 255, 255)); x += t.width
    canvas.save(a.out)
    res = {"video": a.video, "frames": n, "fps": fps, "times": times, "sbs": a.out}
    if a.seams:
        y = fr[..., 0] * 0.299 + fr[..., 1] * 0.587 + fr[..., 2] * 0.114
        step = np.abs(np.diff(y, axis=0)).mean(axis=(1, 2))           # step[i] = |f[i+1]-f[i]|
        seams = [int(s) for s in a.seams.split(",") if 0 < int(s) < n]
        other = np.delete(step, [s - 1 for s in seams]); med = float(np.median(other))
        res["seam_ratio"] = {str(s): round(float(step[s - 1]) / med, 3) for s in seams}
        res["seam_ratio_max"] = max(res["seam_ratio"].values()); res["step_median"] = round(med, 4)
        res["step_p99_nonseam"] = round(float(np.percentile(other, 99)) / med, 3)
    json.dump(res, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1); print(json.dumps(res))


if __name__ == "__main__":
    main()
