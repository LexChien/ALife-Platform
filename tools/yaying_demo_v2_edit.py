#!/usr/bin/env python3
"""Plan 40 demo v2, Option A: NO face edit. Audio-reactive ping-pong edit of the ORIGINAL generated_video-3.mp4.

The source (a 6 s single take) is never modified; frames are read from a working copy. Timeline (24 fps):
  intro   : silent, plays the finger-at-lips end of the take in reverse (frame 144 -> 62) at 1x
  phrase k: sweeps through the mouth-visible window [0, 62] (alternating direction); every direction change is
            a velocity-0 "hold" placed at the midpoint of the pause between two phrases (cubic Hermite easing,
            velocity-continuous everywhere -> no jump cuts, no seams)
  outro   : after the last phrase the take plays forward 62 -> 144 at 1x (hand rises, finger to lips) + 0.5 s hold
Fractional positions are rendered with DIS optical-flow interpolation between the two neighbouring original
frames (integer positions are the original frames bit-for-bit before encoding). Audio = the line WAVs laid out at
the phrase slots, 48 kHz. Writes timeline.json (per output frame source position) and seam statistics.

  ~/plan38_cache/venv/bin/python tools/yaying_demo_v2_edit.py
"""
import argparse, json, subprocess, wave
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
FPS = 24


def read_frames(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = map(int, p.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), w, h


def read_wav(p):
    with wave.open(str(p)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32).reshape(-1, ch).mean(1) / 32768.0
    return x, sr


def resample(x, sr, to):
    t = np.arange(int(len(x) * to / sr)) * sr / to
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def trim(x, sr):
    env = np.convolve(np.abs(x), np.ones(sr // 100) / (sr // 100), "same")
    idx = np.where(env > 0.01)[0]
    return x[max(0, idx[0] - sr // 100): idx[-1] + sr // 20] if len(idx) else x


def hermite(p0, p1, v0, v1, n):
    """n samples (t in [0,1)) of a cubic Hermite from p0 to p1 with end velocities v0,v1 in frames/output-frame."""
    t = np.arange(n) / n
    m0, m1 = v0 * n, v1 * n
    h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def build_timeline(lines_dur, last, win_hi, lead_gap, gap, hold_end):
    """Return (positions per output frame, phrase slots [(start_s,end_s)], audio length s)."""
    intro_n = int(round((last - win_hi) * 1.0))           # 1x reverse 144 -> 62
    t = intro_n / FPS
    slots = []
    for i, d in enumerate(lines_dur):
        slots.append((t, t + d))
        t += d + gap
    t_end_speech = slots[-1][1]
    # segment boundaries: phrase 0 starts at win_hi moving backward; turns at pause midpoints; last phrase ends at
    # win_hi moving forward at 1x. Number of turns = len(lines)-1; parity requires an odd count of turns when we start
    # backward and end forward -> with 4 lines (3 turns) it works; otherwise the last turn is skipped (slow pass).
    turns = [round((slots[i][1] + slots[i + 1][0]) / 2 * FPS) for i in range(len(slots) - 1)]
    pos = list(np.linspace(last, win_hi, intro_n, endpoint=False))
    knots = [intro_n] + turns + [round(t_end_speech * FPS)]
    targets = []
    cur = win_hi
    for k in range(len(knots) - 1):
        targets.append(0.0 if cur == win_hi else float(win_hi))
        cur = targets[-1]
    if targets[-1] != win_hi:                              # even number of turns: drop the last turn
        knots.pop(-2); targets.pop(-1); targets[-1] = float(win_hi)
    cur = float(win_hi)
    for k in range(len(knots) - 1):
        n = knots[k + 1] - knots[k]
        v0 = -1.0 if k == 0 else 0.0
        v1 = 1.0 if k == len(knots) - 2 else 0.0
        pos += list(hermite(cur, targets[k], v0, v1, n))
        cur = targets[k]
    outro_n = int(round(last - win_hi)) + int(round(hold_end * FPS))   # 1x out of the last phrase, eased to rest
    pos += list(hermite(float(win_hi), float(last), 1.0, 0.0, outro_n))
    pos += [float(last)] * 6
    return np.clip(np.array(pos), 0, last), slots, len(pos) / FPS


class Interp:
    def __init__(self, frames):
        self.f = frames
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in frames]
        self.cache = {}
        h, w = frames.shape[1:3]
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def flows(self, i):
        if i not in self.cache:
            self.cache[i] = (self.dis.calc(self.gray[i], self.gray[i + 1], None),
                             self.dis.calc(self.gray[i + 1], self.gray[i], None))
        return self.cache[i]

    def at(self, p):
        i = int(np.floor(p)); a = p - i
        if a < 1e-3 or i >= len(self.f) - 1:
            return self.f[min(i, len(self.f) - 1)].copy(), True
        if a > 1 - 1e-3:
            return self.f[i + 1].copy(), True
        f01, f10 = self.flows(i)
        w0 = cv2.remap(self.f[i], self.gx - a * f01[..., 0], self.gy - a * f01[..., 1], cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
        w1 = cv2.remap(self.f[i + 1], self.gx - (1 - a) * f10[..., 0], self.gy - (1 - a) * f10[..., 1],
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return np.clip((1 - a) * w0.astype(np.float32) + a * w1.astype(np.float32), 0, 255).astype(np.uint8), False


def encode(frames_iter, w, h, wav, out, n):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
           "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-shortest", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k, fr in enumerate(frames_iter):
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/yaying_clone/demo_v2/work/src.mp4")
    ap.add_argument("--voice", default="runs/yaying_clone/demo_v2/voice")
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v2_A_original.mp4")
    ap.add_argument("--win-hi", type=int, default=62, help="last source frame with the mouth unoccluded")
    ap.add_argument("--gap", type=float, default=0.8)
    ap.add_argument("--hold-end", type=float, default=0.5)
    ap.add_argument("--dump-frames", help="also save the exact pre-encode RGB frames (.npy) as the Option B base")
    a = ap.parse_args()
    src, vdir, out = ROOT / a.src, ROOT / a.voice, ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, w, h = read_frames(src)
    last = len(frames) - 1
    sr = 48000
    lines = sorted(vdir.glob("line_*.wav"))
    clips = []
    for p in lines:
        x, xsr = read_wav(p)
        clips.append(trim(resample(x, xsr, sr), sr))
    pos, slots, dur = build_timeline([len(c) / sr for c in clips], last, a.win_hi, 0, a.gap, a.hold_end)
    audio = np.zeros(int(round(dur * sr)) + sr, np.float32)
    for (s, e), c in zip(slots, clips):
        i0 = int(round(s * sr)); audio[i0:i0 + len(c)] += c
    audio = audio[:int(round(len(pos) / FPS * sr))]
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.89
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr)
        ww.writeframes((audio * 32767).astype(np.int16).tobytes())
    ip = Interp(frames)
    exact = []
    diffs = []
    prev = [None]

    def gen():
        for p in pos:
            fr, ex = ip.at(float(p))
            exact.append(ex)
            if prev[0] is not None:
                diffs.append(float(np.abs(fr.astype(np.int16) - prev[0].astype(np.int16)).mean()))
            prev[0] = fr
            yield fr
    dump = np.lib.format.open_memmap(ROOT / a.dump_frames, "w+", np.uint8, (len(pos), h, w, 3)) if a.dump_frames else None

    def gen2():
        for k, fr in enumerate(gen()):
            if dump is not None:
                dump[k] = fr
            yield fr
    encode(gen2(), w, h, wav, out, len(pos))
    if dump is not None:
        dump.flush()
    src_d = [float(np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)).mean()) for i in range(last)]
    vel = np.diff(pos)
    stats = {"source": str(a.src), "frames_out": len(pos), "duration_s": round(len(pos) / FPS, 3), "fps": FPS,
             "size": [w, h], "win_hi": a.win_hi, "phrase_slots_s": [[round(s, 3), round(e, 3)] for s, e in slots],
             "exact_original_frames": int(sum(exact)), "interpolated_frames": int(len(exact) - sum(exact)),
             "speed_abs_max": round(float(np.abs(vel).max()), 3), "speed_abs_mean": round(float(np.abs(vel).mean()), 3),
             "max_velocity_jump": round(float(np.abs(np.diff(vel)).max()), 4),
             "seam_mad_out_max": round(max(diffs), 3), "seam_mad_out_p99": round(float(np.percentile(diffs, 99)), 3),
             "seam_mad_src_max": round(max(src_d), 3), "seam_mad_src_p50": round(float(np.median(src_d)), 3),
             "timeline": [round(float(p), 3) for p in pos]}
    stats["no_visible_seam"] = stats["seam_mad_out_max"] <= stats["seam_mad_src_max"] * 1.05
    json.dump(stats, open(out.with_name(out.stem + "_timeline.json"), "w"), indent=1)
    print({k: v for k, v in stats.items() if k != "timeline"})
    print("A_DONE", out)


if __name__ == "__main__":
    main()
