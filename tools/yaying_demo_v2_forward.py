#!/usr/bin/env python3
"""Plan 40 demo v2 (revised after review 10:19): FORWARD-ONLY timeline over the ORIGINAL generated_video-3.mp4,
no reverse playback, RIFE v4.26 (Practical-RIFE) frame interpolation for slow motion and for any loop seam.

Loop analysis (--analyze): mean-abs-diff matrix D of the 145 source frames (116x172). If the best loop pair inside
the mouth-visible window has D <= --max-loop-d (default 2x the median adjacent-frame step) a loop is allowed and its
seam is bridged with --bridge RIFE frames; otherwise the take is NOT looped and the speech is carried by one
continuous forward pass, slowed with RIFE in-betweens (documented in timeline.json -> loop_analysis).

Timeline (24 fps): t=0 at source frame 0; constant slow-mo speed s1 while she speaks so the mouth stays visible
until the end of the last line (source frame --speech-end-pos); outro = cubic Hermite (velocity-continuous) up to
the last frame (finger to the lips) and a 0.25 s rest. Output A2 (no face edit) + base frames .npy for Option B.

  cd ~/yaying_cache/Practical-RIFE && ../venv_mt/bin/python <repo>/tools/yaying_demo_v2_forward.py --lines 0,3
"""
import argparse, json, sys, time, wave
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yaying_demo_v2_edit import FPS, encode, hermite, read_frames, read_wav, resample, trim  # noqa: E402

RIFE_DIR = Path.home() / "yaying_cache" / "Practical-RIFE"


class Rife:
    def __init__(self):
        import torch
        sys.path.insert(0, str(RIFE_DIR))
        from train_log.RIFE_HDv3 import Model
        self.torch = torch
        self.dev = torch.device("cpu")  # MPS grid_sample lacks padding_mode=border (torch 2.5)
        self.m = Model()
        sd = torch.load(RIFE_DIR / "train_log/flownet.pkl", map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = self.m.flownet.load_state_dict(sd, strict=False)
        assert not missing, f"RIFE weights missing keys: {missing[:5]}"
        self.m.eval(); self.m.flownet.to(self.dev)
        self.version = getattr(self.m, "version", None)

    def __call__(self, a, b, t):
        torch = self.torch
        h, w = a.shape[:2]
        ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
        def ten(x):
            x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.dev)
            return torch.nn.functional.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
        with torch.no_grad():
            y = self.m.inference(ten(a), ten(b), float(t), 1.0)
        return (y[0, :, :h, :w].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/yaying_clone/demo_v2/work/src.mp4")
    ap.add_argument("--voice", default="runs/yaying_clone/demo_v2/voice")
    ap.add_argument("--lines", default="0,3")
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v2_A2_original.mp4")
    ap.add_argument("--dump-frames", default="runs/yaying_clone/demo_v2/work/A2_frames.npy")
    ap.add_argument("--lead", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--speech-end-pos", type=float, default=66.0)
    ap.add_argument("--mouth-visible-max", type=float, default=64.0)
    ap.add_argument("--outro-speed", type=float, default=0.6, help="mean speed of the eased outro")
    ap.add_argument("--max-loop-d", type=float, default=None)
    ap.add_argument("--bridge", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    frames, w, h = read_frames(ROOT / a.src)
    last = len(frames) - 1
    small = np.stack([cv2.resize(f, (116, 172), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames])
    D = np.abs(small[:, None] - small[None]).mean((2, 3, 4))
    adj = np.array([D[i, i + 1] for i in range(last)])
    thr = a.max_loop_d if a.max_loop_d is not None else 2 * float(np.median(adj))
    mv = int(a.mouth_visible_max)
    pairs = sorted((float(D[i, j]), i, j) for j in range(0, mv) for i in range(j + 16, mv + 1))
    loop = {"adjacent_step_median": round(float(np.median(adj)), 3), "adjacent_step_max": round(float(adj.max()), 3),
            "threshold": round(thr, 3), "best_pairs_in_mouth_window": [[round(d, 2), i, j] for d, i, j in pairs[:5]],
            "best_pair_from_last_frame": [round(float(D[last, j]), 2) for j in [0, 30, 60]]}
    loop["loop_used"] = bool(pairs and pairs[0][0] <= thr)
    if loop["loop_used"]:
        raise SystemExit("a clean loop exists; loop rendering path not needed for this source (re-run with analysis)")
    loop["decision"] = ("no clean loop (best D %.2f vs threshold %.2f: continuous camera push/head turn) -> one forward "
                        "pass, RIFE slow motion, zero seams" % (pairs[0][0], thr))
    print(loop, flush=True)
    sr = 48000
    ids = [int(x) for x in a.lines.split(",")]
    clips = []
    for i in ids:
        x, xsr = read_wav(ROOT / a.voice / f"line_{i:02d}.wav")
        clips.append(trim(resample(x, xsr, sr), sr))
    slots, t = [], a.lead
    for c in clips:
        slots.append((t, t + len(c) / sr)); t += len(c) / sr + a.gap
    t_speech_end = slots[-1][1]
    n1 = int(round(t_speech_end * FPS))
    s1 = a.speech_end_pos / n1
    pos = list(np.arange(n1) * s1)
    dist = last - a.speech_end_pos
    n2 = int(round(dist / a.outro_speed))
    pos += list(hermite(a.speech_end_pos, float(last), s1, 0.0, n2))
    pos += [float(last)] * 6
    pos = np.clip(np.array(pos), 0, last)
    assert np.all(np.diff(pos) >= -1e-9), "timeline must be forward-only"
    n = len(pos)
    audio = np.zeros(int(round(n / FPS * sr)), np.float32)
    for (s, e), c in zip(slots, clips):
        i0 = int(round(s * sr)); audio[i0:i0 + len(c)] += c[:max(0, len(audio) - i0)]
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.89
    out = ROOT / a.out
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr)
        ww.writeframes((audio * 32767).astype(np.int16).tobytes())
    rife = Rife()
    dump = np.lib.format.open_memmap(ROOT / a.dump_frames, "w+", np.uint8, (n, h, w, 3))
    exact = 0
    diffs, prev = [], None
    for k, p in enumerate(pos):
        i = int(np.floor(p)); al = p - i
        if al < 1e-3 or i >= last:
            fr = frames[min(i, last)].copy(); exact += 1
        elif al > 1 - 1e-3:
            fr = frames[i + 1].copy(); exact += 1
        else:
            fr = rife(frames[i], frames[i + 1], al)
        dump[k] = fr
        if prev is not None:
            diffs.append(float(np.abs(fr.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = fr
        if k % 60 == 0:
            print("rife", k, "/", n, round(time.time() - t0, 1), "s", flush=True)
    dump.flush()
    encode((dump[k] for k in range(n)), w, h, wav, out, n)
    vel = np.diff(pos)
    st = {"mode": "forward_only_rife", "rife": "Practical-RIFE v4.26 (anon0077/RIFE-Models mirror), cpu",
          "source": a.src, "lines": ids, "frames_out": n, "duration_s": round(n / FPS, 3), "fps": FPS, "size": [w, h],
          "phrase_slots_s": [[round(s, 3), round(e, 3)] for s, e in slots], "speech_speed_x": round(s1, 4),
          "speech_end_pos": a.speech_end_pos, "mouth_visible_max": a.mouth_visible_max, "win_hi": a.mouth_visible_max,
          "outro_mean_speed_x": a.outro_speed, "speed_max_x": round(float(vel.max()), 3),
          "reverse_steps": int((vel < -1e-9).sum()), "max_velocity_jump": round(float(np.abs(np.diff(vel)).max()), 4),
          "exact_original_frames": exact, "rife_frames": n - exact, "loop_analysis": loop, "seams": [],
          "frame_step_mad_out_max": round(max(diffs), 3), "frame_step_mad_out_p99": round(float(np.percentile(diffs, 99)), 3),
          "elapsed_s": round(time.time() - t0, 1), "timeline": [round(float(p), 4) for p in pos]}
    json.dump(st, open(out.with_name(out.stem + "_timeline.json"), "w"), indent=1)
    print({k: v for k, v in st.items() if k != "timeline"})
    print("A2_DONE", out)


if __name__ == "__main__":
    main()
