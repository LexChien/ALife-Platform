#!/usr/bin/env python3
"""Plan 40 demo v3 step 1: motion-aware FORWARD-ONLY timeline + tightened voice, over the ORIGINAL video.

Rules (review 11:11): never interpolate through high-motion spans; slowdown <= 1.5x and only where the per-pair
DIS optical-flow magnitude (p99 over the frame) is below --flow-thr; high-motion pairs play at native speed using
original frames only; extra duration comes from short holds, not heavy interpolation. Speech must end before the
hand reaches the face (source frame --speech-end-max), so the finger never crosses a speaking mouth.
Voice: re-cut from the approved F5 sample line_00 ("嗨" + "我等你好久了喔。"), WSOLA-tightened (ffmpeg atempo)
only as much as needed to fit. Writes work/v3_plan.json (+ v3.wav 48 kHz) and the RIFE base frames work/v3_base.npy.

  cd ~/yaying_cache/Practical-RIFE && ../venv_mt/bin/python <repo>/tools/yaying_demo_v3_plan.py
"""
import argparse, json, subprocess, sys, wave
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yaying_demo_v2_edit import FPS, read_frames, read_wav, resample  # noqa: E402
D = ROOT / "runs/yaying_clone/demo_v2"


def atempo(x, sr, tempo):
    if abs(tempo - 1) < 1e-3:
        return x
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-af",
                        f"atempo={tempo:.4f}", "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-thr", type=float, default=4.0, help="p99 flow (px) above which a pair is high-motion")
    ap.add_argument("--min-speed", type=float, default=1 / 1.5)
    ap.add_argument("--speech-end-max", type=float, default=60.0)
    ap.add_argument("--lead-hold", type=float, default=0.25)
    ap.add_argument("--end-hold", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.15)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    frames, w, h = read_frames(D / "work/src.mp4")
    last = len(frames) - 1
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    G = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    p99 = np.array([np.percentile(np.linalg.norm(dis.calc(G[i], G[i + 1], None), axis=2), 99) for i in range(last)])
    # a pair may be slowed only if it AND its neighbours are low-motion (the source alternates near-duplicates)
    nb = np.maximum.reduce([p99, np.r_[p99[1:], p99[-1]], np.r_[p99[0], p99[:-1]]])
    low = nb <= a.flow_thr
    # spans of equal class; low spans shorter than 4 pairs are played native (avoid speed judder)
    spans, i = [], 0
    while i < last:
        j = i
        while j < last and low[j] == low[i]:
            j += 1
        spans.append([i, j, bool(low[i])]); i = j
    for s in spans:
        if s[2] and s[1] - s[0] < 4:
            s[2] = False
    merged = []
    for s in spans:
        if merged and merged[-1][2] == s[2]:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    # positions: integer steps in native spans; low spans get n = floor(len*1.5) uniform steps (speed >= 1/1.5)
    pos = [0.0] * int(round(a.lead_hold * FPS))
    for s0, s1, lo in merged:
        L = s1 - s0
        n = int(np.floor(L / a.min_speed + 1e-9)) if lo else L
        pos += list(s0 + np.arange(n) * (L / n))
    pos += [float(last)] * (int(round(a.end_hold * FPS)) + 1)
    pos = np.array(pos)
    assert np.all(np.diff(pos) >= 0)
    t_of = lambda p: (np.searchsorted(pos, p) / FPS)  # noqa: E731
    t_speech_end = float(t_of(a.speech_end_max))
    # voice: "嗨" [0.66,1.42] + "我等你好久了喔。" [2.92,5.40] from approved line_00
    x, xsr = read_wav(D / "voice/line_00.wav")
    sr = 48000
    x = resample(x, xsr, sr)
    hi = x[int(0.66 * sr):int(1.42 * sr)]
    wo = x[int(2.92 * sr):int(5.40 * sr)]
    start = 0.05
    natural = len(hi) / sr + a.gap + len(wo) / sr
    avail = t_speech_end - start - 0.04
    tempo = max(1.0, natural / avail)
    if tempo > 1.15:
        raise SystemExit(f"voice needs tempo {tempo:.3f} > 1.15 to fit; shorten the line instead")
    hi2, wo2 = atempo(hi, sr, tempo).copy(), atempo(wo, sr, tempo).copy()
    fade = int(0.008 * sr)
    for seg in (hi2, wo2):
        seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
    n_out = len(pos)
    audio = np.zeros(int(round(n_out / FPS * sr)), np.float32)
    i0 = int(start * sr); audio[i0:i0 + len(hi2)] += hi2
    i1 = i0 + len(hi2) + int(a.gap / tempo * sr); audio[i1:i1 + len(wo2)] += wo2
    speech = [[round(start, 3), round(i0 / sr + len(hi2) / sr, 3), "嗨"],
              [round(i1 / sr, 3), round((i1 + len(wo2)) / sr, 3), "我等你好久了喔。"]]
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.89
    with wave.open(str(D / "work/v3.wav"), "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr)
        ww.writeframes((audio * 32767).astype(np.int16).tobytes())
    frac = [k for k, p in enumerate(pos) if abs(p - round(p)) > 1e-3]
    interp_pairs = sorted({int(np.floor(pos[k])) for k in frac})
    plan = {"flow_thr_px": a.flow_thr, "flow_p99_per_pair": [round(float(v), 2) for v in p99],
            "spans": [[s0, s1, "low(slowed<=1.5x)" if lo else "native"] for s0, s1, lo in merged],
            "frames_out": n_out, "duration_s": round(n_out / FPS, 3), "lead_hold_s": a.lead_hold, "end_hold_s": a.end_hold,
            "speed_min": round(float(min(np.diff(pos)[np.diff(pos) > 0])), 4), "slowdown_max_x": round(1 / float(min(np.diff(pos)[np.diff(pos) > 0])), 3),
            "reverse_steps": int((np.diff(pos) < 0).sum()), "interpolated_frames": len(frac),
            "original_frames": n_out - len(frac), "interp_pairs": interp_pairs,
            "max_flow_p99_on_interpolated_pairs": round(float(max(p99[interp_pairs])) if interp_pairs else 0.0, 2),
            "max_flow_p99_native_pairs": round(float(p99.max()), 2),
            "speech": speech, "voice_tempo": round(tempo, 4), "speech_end_t": t_speech_end,
            "speech_end_max_pos": a.speech_end_max, "voice_source": "voice/line_00.wav (approved F5, nfe32) re-cut",
            "timeline": [round(float(p), 4) for p in pos]}
    json.dump(plan, open(D / "work/v3_plan.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in plan.items() if k not in ("timeline", "flow_p99_per_pair")}, flush=True)
    if a.no_render:
        return
    from yaying_demo_v2_forward import Rife
    rife = Rife()
    base = np.lib.format.open_memmap(D / "work/v3_base.npy", "w+", np.uint8, (n_out, h, w, 3))
    for k, p in enumerate(pos):
        i = int(np.floor(p)); al = p - i
        base[k] = frames[min(i, last)] if (al < 1e-3 or i >= last) else (frames[i + 1] if al > 1 - 1e-3 else rife(frames[i], frames[i + 1], al))
    base.flush()
    print("V3_BASE_DONE", n_out)


if __name__ == "__main__":
    main()
