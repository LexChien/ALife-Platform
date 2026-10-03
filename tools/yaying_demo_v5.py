#!/usr/bin/env python3
"""Plan 40 demo v5 (Lex feedback 12:23 on v4): 15-18 s, expressive opening, full line, smooth lips.

Timeline (ORIGINAL frames, forward playback inside each section, ONE hard cut):
  1. opening  = the finger-at-lips / glance stretch of the source (--open-start .. last), native speed on moving pairs,
                <= 1.5x on low-motion spans, gentle holds only on near-static pairs (teasing touch) -> expressive start
  2. hard cut -> speech section = source 0 .. --speech-end-max (mouth visible, no hand over the mouth); the FULL,
                untrimmed approved line_00 (whole wav incl. its own leading silence) is placed so its onset is
                --speech-lead s after the cut; holds spread over many near-static pairs (each <= --hold-max s) so the
                start never freezes; the head tilt + hand rise (34-60) play at native speed under "我等你好久了喔"
  3. ending   = source --speech-end-max .. --open-start: hand arrives at the lips at native speed, then lingering
                holds on near-static pairs (lip-touch) + end hold to reach --total-s.
Stages:
  --stage base : timeline + RIFE base frames + wav  -> work/v5_base.npy, work/v5.wav, work/v5_timeline.json
  --stage lips : MuseTalk v1.5 + GFPGAN mouth with the v4 temporal smoothing (SG bbox/landmarks, smoothed reference
                 latents, 3-tap whisper + latent smoothing, light flow-aligned pixel blend, constant alpha with cosine
                 ramps, SG-smoothed per-frame sharpen) on --base-npy (eye-edited or not) -> --out
  cd ~/yaying_cache/MuseTalk && ../venv_mt/bin/python <repo>/tools/yaying_demo_v5.py --stage base|lips ...
"""
import argparse, json, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import yaying_demo_v2_lipsync as v2  # noqa: E402
from yaying_demo_v2_lipsync import MOUTH, ssim_map  # noqa: E402
from yaying_demo_v2_edit import FPS, read_frames, read_wav, resample  # noqa: E402
from yaying_demo_v3_lipsync import Restorer, lapvar, regions, sharpen  # noqa: E402
from yaying_demo_v4 import detect, mouth_metrics  # noqa: E402
MT = Path.home() / "yaying_cache" / "MuseTalk"
sys.path.insert(0, str(MT))
D = ROOT / "runs/yaying_clone/demo_v2"


def section(p99, s0, s1, a, target_frames, hold_flow, hold_max):
    """forward positions for source pairs s0..s1-1 (+ final frame s1 not included). Steps: 1.0 native, 1.5 on low
    spans, plus holds on near-static pairs to reach target_frames (each hold <= hold_max frames)."""
    nb = np.maximum.reduce([p99, np.r_[p99[1:], p99[-1]], np.r_[p99[0], p99[:-1]]])
    low = nb <= a.flow_thr
    steps = {p: (1.5 if low[p] else 1.0) for p in range(s0, s1)}
    base_len = sum(steps.values())
    cands = sorted([p for p in range(s0, s1) if p99[p] <= hold_flow], key=lambda p: p99[p])
    def build(per):
        holds = {p: per for p in cands} if per > 0 else {}
        pos, carry = [], 0.0
        for p in range(s0, s1):
            n = steps[p] + holds.get(p, 0.0) + carry
            if steps[p] == 1.0 and p not in holds:
                k, carry = 1, 0.0
            else:
                k = int(np.floor(n + 1e-9)); carry = n - k
            pos += list(p + np.arange(k) / k) if k > 0 else []
        return pos, holds
    per = min(hold_max, max(0.0, target_frames - base_len) / len(cands)) if cands else 0.0
    pos, holds = build(per)
    while cands and len(pos) < target_frames and per < hold_max:   # floor/carry losses: grow holds until it fits
        per = min(hold_max, per + 0.25); pos, holds = build(per)
    return pos, holds, max(0.0, target_frames - len(pos))


def stage_base(a):
    from yaying_demo_v2_forward import Rife
    t0 = time.time()
    frames, w, h = read_frames(D / "work/src.mp4"); last = len(frames) - 1
    p99 = np.array(json.load(open(D / "work/v3_plan.json"))["flow_p99_per_pair"])
    x, xsr = read_wav(D / a.line); sr = 48000; x = resample(x, xsr, sr)
    env = np.convolve(np.abs(x), np.ones(480) / 480, "same"); vi = np.where(env > 0.01)[0]
    on_w, off_w = vi[0] / sr, vi[-1] / sr                       # voiced span inside the approved wav
    S = int(a.speech_end_max); O = int(a.open_start)
    # 1. opening
    op, op_holds, _ = section(p99, O, last, a, a.open_s * FPS, a.hold_flow, a.open_hold_max * FPS)
    op += [float(last)]
    t_cut = len(op)
    # 2. speech section: onset at t_cut + lead; needs to end (+0.12 s) before the mouth-limit frame S
    need = (a.speech_lead + (off_w - on_w) + 0.12) * FPS
    sp, sp_holds, sp_def = section(p99, 0, S, a, need, a.hold_flow_speech, a.hold_max * FPS)
    assert sp_def <= 0.5, f"speech does not fit: deficit {sp_def:.1f} frames"
    # 3. ending (S .. O), lingering holds, end hold
    end_hold = int(round(a.end_hold * FPS))
    tgt_end = a.total_s * FPS - len(op) - len(sp) - end_hold - 1
    en, en_holds, en_def = section(p99, S, O, a, tgt_end, a.hold_flow_end, a.end_hold_max * FPS)
    end_hold += int(np.ceil(en_def))
    pos = np.array(op + sp + en + [float(O)] * (end_hold + 1))
    n = len(pos)
    cuts = [int(k) for k in np.where(np.diff(pos) < -1e-9)[0] + 1]
    assert cuts == [t_cut], cuts
    on_t = t_cut / FPS + a.speech_lead; off_t = on_t + (off_w - on_w)
    sp_end_k = t_cut + len(sp)                                   # first output frame past the mouth limit
    audio = np.zeros(int(round(n / FPS * sr)), np.float32)
    st = int(round((on_t - on_w) * sr)); audio[st:st + len(x)] = x[:max(0, len(audio) - st)]
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.89
    with wave.open(str(D / "work/v5.wav"), "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr); ww.writeframes((audio * 32767).astype(np.int16).tobytes())
    frac = [k for k, p in enumerate(pos) if abs(p - round(p)) > 1e-3]
    ip = sorted({int(np.floor(pos[k])) for k in frac})
    tl = {"frames_out": n, "duration_s": round(n / FPS, 3), "fps": FPS,
          "sections": {"opening": [0, t_cut, f"src {O}->{last}"], "speech": [t_cut, sp_end_k, f"src 0->{S}"],
                       "ending": [sp_end_k, n, f"src {S}->{O} + end hold"]},
          "hard_cuts_at_frame": cuts, "reverse_playback_steps": 0,
          "holds_frames": {"opening": {int(k): round(v, 2) for k, v in op_holds.items()},
                           "speech": {int(k): round(v, 2) for k, v in sp_holds.items()},
                           "ending": {int(k): round(v, 2) for k, v in en_holds.items()}},
          "end_hold_frames": end_hold, "speech_on_s": round(on_t, 3), "speech_off_s": round(off_t, 3),
          "speech_wav_voiced_s": [round(on_w, 3), round(off_w, 3)], "line": a.line, "line_trimmed": False,
          "mouth_visible_until_s": round(sp_end_k / FPS, 3), "speech_fits": bool(off_t + 0.04 <= sp_end_k / FPS),
          "interpolated_frames": len(frac), "original_frames": n - len(frac),
          "max_flow_p99_on_interpolated_pairs": round(float(p99[ip].max()) if ip else 0.0, 2),
          "max_flow_p99_on_held_pairs": round(float(max([p99[p] for d_ in (op_holds, sp_holds, en_holds) for p in d_] or [0])), 2),
          "slowdown_max_outside_holds_x": 1.5, "pos": [round(float(p), 4) for p in pos]}
    print({k: v for k, v in tl.items() if k != "pos"}, flush=True)
    assert tl["speech_fits"]
    rife = Rife()
    base = np.zeros((n, h, w, 3), np.uint8)
    for k, p in enumerate(pos):
        i = int(np.floor(p)); al = p - i
        base[k] = frames[min(i, last)] if (al < 1e-3 or i >= last) else (frames[i + 1] if al > 1 - 1e-3 else rife(frames[i], frames[i + 1], al))
    np.save(D / "work/v5_base.npy", base)
    json.dump(tl, open(D / "work/v5_timeline.json", "w"), indent=1, ensure_ascii=False)
    print("BASE_DONE", round(time.time() - t0, 1), flush=True)


def stage_lips(a):
    import torch
    import mediapipe as mp
    from insightface.app import FaceAnalysis
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    t0 = time.time()
    v2.MASK_SCALE = a.mask_scale
    tl = json.load(open(D / "work/v5_timeline.json"))
    base = np.load(ROOT / a.base_npy)
    orig = np.load(D / "work/v5_base.npy", mmap_mode="r")
    n, h, w = base.shape[:3]
    wav = D / "work/v5.wav"
    on, off = tl["speech_on_s"], tl["speech_off_s"]
    cut = tl["sections"]["speech"][0]; sp_end = tl["sections"]["speech"][1]
    k_on, k_off = int(np.floor(on * FPS)) - 2, int(np.ceil(off * FPS)) + 2
    env_a = np.zeros(n); env_a[k_on:k_off + 1] = 1.0
    for r in range(1, a.ramp + 1):
        v = 0.5 - 0.5 * np.cos(np.pi * (a.ramp - r + 1) / (a.ramp + 1))
        if k_on - r >= 0: env_a[k_on - r] = v
        if k_off + r < n: env_a[k_off + r] = v
    env_a[:cut] = 0.0; env_a[sp_end:] = 0.0                      # never across the cut / past the mouth limit
    todo = [i for i in range(n) if env_a[i] > 1e-3]
    dev = torch.device(a.device)
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    F = {i: detect(fa, base[i]) for i in todo}
    assert all(F[i] is not None for i in todo)
    FO = {i: detect(fa, np.ascontiguousarray(orig[i])) for i in todo}
    wl = min(9, len(todo) - (1 - len(todo) % 2))
    sg = lambda arr: savgol_filter(arr, wl, 2, axis=0, mode="interp")  # noqa: E731
    BB = sg(np.stack([F[i].bbox for i in todo])); KP = sg(np.stack([F[i].kps for i in todo])); LM = sg(np.stack([F[i].landmark_2d_106 for i in todo]))
    geo = {i: (BB[k], KP[k], LM[k]) for k, i in enumerate(todo)}
    crops = {}
    for i in todo:
        (x1, y1, x2, y2), kp, _ = geo[i]
        half = (kp[0][1] + kp[1][1]) / 2 * 0.45 + kp[2][1] * 0.55
        crops[i] = [int(round(max(0, x1))), int(round(max(0, half - (y2 - half)))), int(round(min(w, x2))), int(round(min(h, y2 + 10)))]
    vae = VAE(model_path=str(MT / "models/sd-vae")); vae.vae = vae.vae.to(dev)
    unet = UNet(unet_config=str(MT / "models/musetalkV15/musetalk.json"), model_path=str(MT / "models/musetalkV15/unet.pth"), device=dev)
    unet.model = unet.model.to(dev); pe = PositionalEncoding(d_model=384).to(dev)
    apx = AudioProcessor(feature_extractor_path=str(MT / "models/whisper"))
    whisper = WhisperModel.from_pretrained(str(MT / "models/whisper")).to(dev).eval()
    feats, L = apx.get_audio_feature(str(wav))
    chunks = apx.get_whisper_chunk(feats, dev, unet.model.dtype, whisper, L, fps=FPS, audio_padding_length_left=2, audio_padding_length_right=2)
    tstep = torch.tensor([0], device=dev)

    def tsmooth(x, wn):
        if wn <= 0 or len(x) < 3:
            return x
        xp = torch.cat([x[:1], x, x[-1:]], 0)
        return wn * xp[:-2] + (1 - 2 * wn) * xp[1:-1] + wn * xp[2:]
    gens = {}
    with torch.no_grad():
        LAT = torch.cat([vae.get_latents_for_unet(cv2.resize(np.ascontiguousarray(base[i][c[1]:c[3], c[0]:c[2], ::-1]), (256, 256), interpolation=cv2.INTER_LANCZOS4))
                         for i, c in ((i, crops[i]) for i in todo)], 0).to(dev)
        if a.ref_sigma > 0:
            rad = int(3 * a.ref_sigma); kk = np.exp(-0.5 * (np.arange(-rad, rad + 1) / a.ref_sigma) ** 2)
            ref = LAT[:, 4:].clone()
            for t in range(len(todo)):
                js = np.clip(np.arange(t - rad, t + rad + 1), 0, len(todo) - 1); wk = torch.tensor(kk / kk.sum(), device=dev, dtype=ref.dtype)
                ref[t] = (LAT[js, 4:] * wk[:, None, None, None]).sum(0)
            LAT = torch.cat([LAT[:, :4], ref], 1)
        AUD = tsmooth(torch.stack([chunks[i] for i in todo]).to(dev), a.audio_smooth)
        PRED = torch.cat([unet.model(LAT[b0:b0 + 8], tstep, encoder_hidden_states=pe(AUD[b0:b0 + 8])).sample for b0 in range(0, len(todo), 8)], 0)
        PRED = tsmooth(PRED, a.latent_smooth)
        for b0 in range(0, len(todo), 8):
            for i, r in zip(todo[b0:b0 + 8], vae.decode_latents(PRED[b0:b0 + 8])):
                gens[i] = r
    rest = Restorer(dev)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    R, M = {}, {}
    for i in todo:
        b = base[i]; c = crops[i]; _, kp, lm = geo[i]
        g = b.copy()
        g[c[1]:c[3], c[0]:c[2]] = cv2.resize(gens[i].astype(np.uint8), (c[2] - c[0], c[3] - c[1]), interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1]
        m, _ = v2.mouth_mask(lm, h, w, feather=max(9, int(0.35 * np.ptp(lm[MOUTH][:, 0]))))
        g = v2.match_color(g, b, m, 0.0); g = rest(g, kp, a.fidelity); g = v2.match_color(g, b, m, 0.0)
        hm = np.zeros((h, w), np.float32)
        for hl in (hands.process(np.ascontiguousarray(b)).multi_hand_landmarks or []):
            pp = np.array([[q.x * w, q.y * h] for q in hl.landmark], np.float32)
            cv2.fillConvexPoly(hm, cv2.convexHull(pp.astype(np.int32)), 1.0)
        if hm.any():
            hm = cv2.GaussianBlur(cv2.dilate(hm, np.ones((31, 31), np.uint8)), (21, 21), 7)
        R[i] = g; M[i] = m * (1 - np.clip(hm, 0, 1))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    gray = {i: cv2.cvtColor(base[i], cv2.COLOR_RGB2GRAY) for i in todo}
    Rs = {}
    for i in todo:
        acc, wsum = R[i].astype(np.float32) * (1 - 2 * a.pix_blend), 1 - 2 * a.pix_blend
        for j in (i - 1, i + 1):
            if j in R:
                fl = dis.calc(gray[i], gray[j], None)
                acc += a.pix_blend * cv2.remap(R[j], gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                wsum += a.pix_blend
        Rs[i] = np.clip(acc / wsum, 0, 255).astype(np.uint8)

    def lv_dev(i, comp):
        b = base[i]; mouth_r, cheek_r = regions(geo[i][2], M[i], h, w, cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
        if cheek_r is None:
            return None
        gb, gc = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
        rb = lapvar(gb, mouth_r) / max(1e-6, lapvar(gb, cheek_r)); ro = lapvar(gc, mouth_r) / max(1e-6, lapvar(gc, cheek_r))
        return ro / rb - 1
    grain = {}
    for i in todo:
        _, cheek_r = regions(geo[i][2], M[i], h, w, gray[i])
        hp = gray[i].astype(np.float32) - cv2.GaussianBlur(gray[i], (0, 0), 1.0).astype(np.float32)
        grain[i] = float(hp[cheek_r].std()) if cheek_r is not None else 0.0
    KF = {}

    def compose(i, alpha, k=None):
        b = base[i]; m = M[i] * alpha
        g = sharpen(Rs[i], KF[i] if k is None else k).astype(np.float32)
        if grain[i] > 0:
            g = g + np.random.default_rng(1000 + i).normal(0, grain[i] * a.grain, g.shape[:2])[..., None]
        comp = (b.astype(np.float32) * (1 - m[..., None]) + np.clip(g, 0, 255) * m[..., None]).round()
        return np.where(m[..., None] > 0, comp, b).astype(np.uint8)
    full_ = [i for i in todo if env_a[i] >= 0.99]
    ks = [min((abs(lv_dev(i, compose(i, 0.5, k)) or 9), k) for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0))[1] for i in full_]
    ksm = np.clip(savgol_filter(np.array(ks, float), min(9, len(ks) - (1 - len(ks) % 2)), 1, mode="interp"), 0, 6)
    for i in todo:
        KF[i] = float(ksm[int(np.argmin([abs(i - j) for j in full_]))])
    print("sharpen k per frame (SG-smoothed)", [round(v, 2) for v in ksm], flush=True)

    def gate(i, comp):
        fc = detect(fa, comp)
        arc = float(np.dot(fc.normed_embedding, F[i].normed_embedding)) if fc is not None else None
        arc_o = float(np.dot(fc.normed_embedding, FO[i].normed_embedding)) if (fc is not None and FO[i] is not None) else None
        out = M[i] <= 0
        ss = float(ssim_map(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), gray[i])[out].mean())
        return arc, ss, arc is not None and arc >= a.t_id and ss >= a.t_ssim, arc_o
    chosen, res = None, None
    for A0 in [float(v) for v in a.alphas.split(",")]:
        r = {i: gate(i, compose(i, A0 * env_a[i])) for i in todo}
        rate = sum(v[2] for v in r.values()) / len(r)
        print("alpha", A0, "pass rate", round(rate, 3), flush=True)
        chosen, res = A0, r
        if rate >= 0.97:
            break
    alpha = np.array([chosen * env_a[i] for i in range(n)])
    for i in todo:
        if not res[i][2]:
            for sc in (0.75, 0.5, 0.3):
                if gate(i, compose(i, alpha[i] * sc))[2]:
                    alpha[i] *= sc; break
            else:
                alpha[i] = 0.0
    for _ in range(2):
        for i in range(1, n):
            alpha[i] = min(alpha[i], alpha[i - 1] + 0.08) if alpha[i] > 0 else 0.0
        for i in range(n - 2, -1, -1):
            alpha[i] = min(alpha[i], alpha[i + 1] + 0.08) if alpha[i] > 0 else 0.0
    rows, final = [], []
    for i in range(n):
        b = base[i]; row = {"i": i, "alpha": round(float(alpha[i]), 3)}
        if i in R and alpha[i] > 1e-3:
            comp = compose(i, alpha[i]); arc, ss, ok, arc_o = gate(i, comp)
            dv = lv_dev(i, comp)
            row.update({"lipsync": True, "arcface": round(arc, 4) if arc else None, "arcface_vs_original": round(arc_o, 4) if arc_o else None,
                        "ssim_outside": round(ss, 5), "lv_ratio_dev": round(dv, 4) if dv is not None else None, "gate": "PASS" if ok else "FAIL"})
            final.append(comp if ok else b)
        else:
            row.update({"lipsync": False, "gate": "BASE"}); final.append(b)
        rows.append(row)
    final = np.stack(final)
    out = ROOT / a.out
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE); p.stdin.write(final.tobytes()); p.stdin.close(); assert p.wait() == 0
    np.save(D / f"work/{out.stem}_final.npy", final)
    ls = [r for r in rows if r.get("lipsync")]
    arcs = [r["arcface"] for r in ls if r["arcface"]]; dvs = [r["lv_ratio_dev"] for r in ls if r["lv_ratio_dev"] is not None]
    seg = list(range(full_[0], full_[-1] + 1))
    met = {"out": mouth_metrics(final, fa, seg), "base": mouth_metrics(base, fa, seg)}
    met["flicker_ratio_vs_base"] = round(met["out"]["flicker_mad_mean"] / met["base"]["flicker_mad_mean"], 3)
    met["flicker2_ratio_vs_base"] = round(met["out"]["flicker2_mean"] / met["base"]["flicker2_mean"], 3)
    met["jitter_ratio_vs_base"] = round(met["out"]["jitter_mean"] / met["base"]["jitter_mean"], 3)
    summ = {"timeline": {k: v for k, v in tl.items() if k != "pos"}, "base_npy": a.base_npy, "lipsync_frames": len(ls),
            "gate_pass": sum(r["gate"] == "PASS" for r in ls), "gate_fail_fallback": sum(r["gate"] == "FAIL" for r in ls),
            "alpha_global": chosen, "alpha_max_step": float(np.abs(np.diff(alpha)).max()), "ramp_frames": a.ramp,
            "sharpen_k_range": [float(ksm.min()), float(ksm.max())], "arcface_min": min(arcs), "arcface_median": float(np.median(arcs)),
            "arcface_vs_original_min": min(r["arcface_vs_original"] for r in ls if r.get("arcface_vs_original")),
            "ssim_outside_min": min(r["ssim_outside"] for r in ls), "lv_within_tol": sum(abs(v) <= a.lv_tol for v in dvs),
            "lv_frames": len(dvs), "lv_dev_median": float(np.median(dvs)), "lv_dev_abs_max": float(max(abs(v) for v in dvs)),
            "jitter_flicker": met, "params": vars(a), "elapsed_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "rows": rows}, open(out.with_name(out.stem + "_gate.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False)); print("V5_DONE", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "lips"], required=True)
    ap.add_argument("--line", default="voice/line_00.wav")
    ap.add_argument("--base-npy", default="runs/yaying_clone/demo_v2/work/v5_base.npy")
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v5.mp4")
    ap.add_argument("--flow-thr", type=float, default=4.0)
    ap.add_argument("--open-start", type=float, default=100)
    ap.add_argument("--open-s", type=float, default=3.4)
    ap.add_argument("--open-hold-max", type=float, default=0.35)
    ap.add_argument("--speech-end-max", type=float, default=60.0)
    ap.add_argument("--speech-lead", type=float, default=0.4)
    ap.add_argument("--hold-flow", type=float, default=1.2)
    ap.add_argument("--hold-flow-speech", type=float, default=1.6)
    ap.add_argument("--hold-max", type=float, default=0.3)
    ap.add_argument("--hold-flow-end", type=float, default=1.6)
    ap.add_argument("--end-hold-max", type=float, default=0.9)
    ap.add_argument("--end-hold", type=float, default=0.6)
    ap.add_argument("--total-s", type=float, default=15.6)
    ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--alphas", default="0.5,0.45,0.4")
    ap.add_argument("--t-id", type=float, default=0.88)
    ap.add_argument("--t-ssim", type=float, default=0.99)
    ap.add_argument("--mask-scale", type=float, default=0.8)
    ap.add_argument("--fidelity", type=float, default=0.7)
    ap.add_argument("--grain", type=float, default=0.5)
    ap.add_argument("--lv-tol", type=float, default=0.15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ref-sigma", type=float, default=4.0)
    ap.add_argument("--audio-smooth", type=float, default=0.25)
    ap.add_argument("--latent-smooth", type=float, default=0.25)
    ap.add_argument("--pix-blend", type=float, default=0.15)
    a = ap.parse_args()
    stage_base(a) if a.stage == "base" else stage_lips(a)


if __name__ == "__main__":
    main()
