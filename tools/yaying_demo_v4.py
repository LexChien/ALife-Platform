#!/usr/bin/env python3
"""Plan 40 demo v4 (Lex feedback 11:38 on v3): FULL approved line, video fitted to the speech, smooth lips.

Timeline (forward-only, ORIGINAL frames): the full, untrimmed approved F5 line_00 ("嗨，你終於來了。我等你好久了喔。")
must finish while the mouth is visible (source frame <= --speech-end-max). Time is found by (1) a lead hold on
frame 0, (2) low-motion holds = stretching near-static source pairs (p99 flow <= --hold-flow px, inside low-motion
spans) with RIFE in-betweens, (3) <= 1.5x slowdown only in low-motion spans; high-motion spans play native speed with
original frames only. Mouth (MuseTalk v1.5 + GFPGAN v1.4 fidelity 0.7) is temporally stabilised:
  * face bbox / 5 kps / 106 landmarks Savitzky-Golay filtered over time (no per-frame recrop jitter)
  * MuseTalk+GFPGAN mouth renders blended over 3 frames (0.25/0.5/0.25) after aligning the neighbours with the base
    frames' DIS optical flow
  * constant blend alpha over the whole utterance (no RMS gating -> no closed-mouth snaps mid-word), cosine ramps of
    --ramp frames in/out; one global sharpen amount (no per-frame sharpness flicker)
Gate per frame: ArcFace(composite, base) >= 0.88, SSIM outside mask >= 0.99 (global alpha lowered until >= 97 % pass;
residual failures get a slope-limited local dip, then fallback). Metrics: mouth-landmark jitter (2nd difference,
normalised by mouth width) and mouth-region flicker (consecutive-frame MAD vs the base), also for v3.
  cd ~/yaying_cache/MuseTalk && ../venv_mt/bin/python <repo>/tools/yaying_demo_v4.py
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
MT = Path.home() / "yaying_cache" / "MuseTalk"
sys.path.insert(0, str(MT))
D = ROOT / "runs/yaying_clone/demo_v2"


def build_timeline(p99, last, a, need_until):
    nb = np.maximum.reduce([p99, np.r_[p99[1:], p99[-1]], np.r_[p99[0], p99[:-1]]])
    low = nb <= a.flow_thr
    spans, i = [], 0
    while i < last:
        j = i
        while j < last and low[j] == low[i]:
            j += 1
        spans.append([i, j, bool(low[i]) and j - i >= 4]); i = j
    steps = {}                                   # pair -> number of output steps
    for s0, s1, lo in spans:
        for p in range(s0, s1):
            steps[p] = 1.5 if lo else 1.0
    # holds on near-static pairs inside low spans before the mouth limit
    cands = [p for s0, s1, lo in spans if lo for p in range(s0, s1) if p < a.speech_end_max - 1 and p99[p] <= a.hold_flow]
    lead = int(round(a.lead_hold * FPS))
    t_speech_avail = lead + sum(steps[p] for p in range(int(a.speech_end_max)))
    deficit = need_until * FPS - t_speech_avail
    holds = {}
    if deficit > 0 and cands:
        per = min(a.hold_max * FPS, deficit / len(cands))
        for p in cands:
            holds[p] = per
        deficit -= per * len(cands)
    if deficit > 0:
        lead += int(np.ceil(deficit))
    pos = [0.0] * lead
    carry = 0.0
    for p in range(last):
        n = steps[p] + holds.get(p, 0.0) + carry
        k = int(np.floor(n + 1e-9)); carry = n - k
        if steps[p] == 1.0 and p not in holds:
            k, carry = 1, 0.0                     # native pairs: exactly the original frame
        pos += list(p + np.arange(k) / k) if k > 0 else []
    pos += [float(last)] * (int(round(a.end_hold * FPS)) + 1)
    return np.array(pos), spans, holds, lead


def detect(fa, img):
    fs = fa.get(np.ascontiguousarray(img[:, :, ::-1]))
    return max(fs, key=lambda f: f.det_score) if fs else None


def mouth_metrics(frames_seq, fa, idx):
    """2nd-difference jitter of the 20 mouth landmarks (normalised by mouth width) and consecutive-frame MAD of the
    mouth box, over consecutive frame indices idx."""
    L, boxes = [], []
    for i in idx:
        f = detect(fa, frames_seq[i])
        L.append(f.landmark_2d_106[MOUTH] if f is not None else None)
    ok = [k for k in range(1, len(idx) - 1) if all(L[j] is not None for j in (k - 1, k, k + 1))]
    jit = [float(np.linalg.norm(L[k + 1] - 2 * L[k] + L[k - 1], axis=1).mean() / max(1e-6, np.ptp(L[k][:, 0]))) for k in ok]
    fl, fl2 = [], []
    for k in range(1, len(idx)):
        if L[k] is None:
            continue
        x0, y0 = L[k].min(0).astype(int) - 6; x1, y1 = L[k].max(0).astype(int) + 6
        a_, b_ = frames_seq[idx[k]][y0:y1, x0:x1].astype(np.float32), frames_seq[idx[k - 1]][y0:y1, x0:x1].astype(np.float32)
        fl.append(float(np.abs(a_ - b_).mean()))
        if k + 1 < len(idx):
            c_ = frames_seq[idx[k + 1]][y0:y1, x0:x1].astype(np.float32)
            fl2.append(float(np.abs(c_ - 2 * a_ + b_).mean()))
    return {"flicker2_mean": round(float(np.mean(fl2)), 3) if fl2 else None, "jitter_mean": round(float(np.mean(jit)), 5) if jit else None, "jitter_p95": round(float(np.percentile(jit, 95)), 5) if jit else None,
            "flicker_mad_mean": round(float(np.mean(fl)), 3) if fl else None, "n": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="voice/line_00.wav")
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v4.mp4")
    ap.add_argument("--flow-thr", type=float, default=4.0)
    ap.add_argument("--hold-flow", type=float, default=1.2)
    ap.add_argument("--hold-max", type=float, default=0.8)
    ap.add_argument("--lead-hold", type=float, default=0.5)
    ap.add_argument("--end-hold", type=float, default=0.8)
    ap.add_argument("--speech-end-max", type=float, default=60.0)
    ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--alphas", default="0.55,0.45")
    ap.add_argument("--t-id", type=float, default=0.88)
    ap.add_argument("--t-ssim", type=float, default=0.99)
    ap.add_argument("--mask-scale", type=float, default=0.8)
    ap.add_argument("--fidelity", type=float, default=0.7)
    ap.add_argument("--grain", type=float, default=0.5)
    ap.add_argument("--lv-tol", type=float, default=0.15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reuse-base", type=int, default=1)
    ap.add_argument("--ref-sigma", type=float, default=4.0, help="temporal Gaussian sigma (frames) on MuseTalk reference latents")
    ap.add_argument("--audio-smooth", type=float, default=0.25, help="neighbour weight of the 3-tap smoothing of whisper chunks")
    ap.add_argument("--latent-smooth", type=float, default=0.25, help="neighbour weight of the 3-tap smoothing of predicted latents")
    ap.add_argument("--pix-blend", type=float, default=0.15, help="neighbour weight of the flow-aligned pixel blend")
    a = ap.parse_args()
    v2.MASK_SCALE = a.mask_scale
    import torch
    import mediapipe as mp
    from insightface.app import FaceAnalysis
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    sys.path.insert(0, str(ROOT / "tools"))
    from yaying_demo_v2_forward import Rife
    t0 = time.time()
    frames, w, h = read_frames(D / "work/src.mp4"); last = len(frames) - 1
    p99 = np.array(json.load(open(D / "work/v3_plan.json"))["flow_p99_per_pair"])
    # ---- audio: FULL line, untrimmed content; leading/trailing silence kept as in the approved file
    x, xsr = read_wav(D / a.line); sr = 48000; x = resample(x, xsr, sr)
    env = np.convolve(np.abs(x), np.ones(480) / 480, "same"); vi = np.where(env > 0.01)[0]
    on, off = vi[0] / sr, vi[-1] / sr
    pos, spans, holds, lead = build_timeline(p99, last, a, need_until=off + 0.12)
    n = len(pos)
    assert np.all(np.diff(pos) >= 0)
    audio = np.zeros(int(round(n / FPS * sr)), np.float32); audio[:len(x)] = x[:len(audio)]
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.89
    wav = D / "work/v4.wav"
    with wave.open(str(wav), "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr); ww.writeframes((audio * 32767).astype(np.int16).tobytes())
    t_mouth_end = np.searchsorted(pos, a.speech_end_max) / FPS
    frac = [k for k, p in enumerate(pos) if abs(p - round(p)) > 1e-3]
    ip = sorted({int(np.floor(pos[k])) for k in frac})
    tl = {"frames_out": n, "duration_s": round(n / FPS, 3), "lead_hold_frames": lead, "holds_pairs_frames": {int(k): round(v, 2) for k, v in holds.items()},
          "spans": [[s0, s1, "low" if lo else "native"] for s0, s1, lo in spans], "speech_on_s": round(on, 3), "speech_off_s": round(off, 3),
          "mouth_visible_until_s": round(float(t_mouth_end), 3), "speech_fits": bool(off <= t_mouth_end),
          "reverse_steps": int((np.diff(pos) < -1e-9).sum()), "interpolated_frames": len(frac), "original_frames": n - len(frac),
          "max_flow_p99_on_interpolated_pairs": round(float(p99[ip].max()) if ip else 0.0, 2),
          "slowdown_max_outside_holds_x": 1.5, "line": a.line, "line_trimmed": False}
    print(tl, flush=True)
    assert tl["speech_fits"], "speech does not fit the mouth-visible window"
    bp = D / "work/v4_base.npy"
    if a.reuse_base and bp.exists() and np.load(bp, mmap_mode="r").shape == (n, h, w, 3):
        base = np.load(bp)
        pos_iter = []
    else:
        rife = Rife(); base = np.zeros((n, h, w, 3), np.uint8); pos_iter = list(enumerate(pos))
    for k, p in pos_iter:
        i = int(np.floor(p)); al = p - i
        base[k] = frames[min(i, last)] if (al < 1e-3 or i >= last) else (frames[i + 1] if al > 1 - 1e-3 else rife(frames[i], frames[i + 1], al))
    np.save(D / "work/v4_base.npy", base)
    print("base", round(time.time() - t0, 1), flush=True)
    # ---- alpha: constant over the utterance, cosine ramps
    k_on, k_off = int(np.floor(on * FPS)) - 2, int(np.ceil(off * FPS)) + 2
    env_a = np.zeros(n)
    env_a[k_on:k_off + 1] = 1.0
    for r in range(1, a.ramp + 1):
        v = 0.5 - 0.5 * np.cos(np.pi * (a.ramp - r + 1) / (a.ramp + 1))
        if k_on - r >= 0: env_a[k_on - r] = v
        if k_off + r < n: env_a[k_off + r] = v
    vis = np.clip((a.speech_end_max + 2 - pos) / 2, 0, 1)
    env_a *= vis
    todo = [i for i in range(n) if env_a[i] > 1e-3]
    dev = torch.device(a.device)
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    F = {i: detect(fa, base[i]) for i in todo}
    assert all(F[i] is not None for i in todo)
    T = np.array(todo)
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
    gens = {}
    tstep = torch.tensor([0], device=dev)
    def tsmooth(x, wn):                            # 3-tap temporal smoothing along dim 0 (edge-replicated)
        if wn <= 0 or len(x) < 3:
            return x
        xp = torch.cat([x[:1], x, x[-1:]], 0)
        return wn * xp[:-2] + (1 - 2 * wn) * xp[1:-1] + wn * xp[2:]
    with torch.no_grad():
        LAT = torch.cat([vae.get_latents_for_unet(cv2.resize(np.ascontiguousarray(base[i][c[1]:c[3], c[0]:c[2], ::-1]), (256, 256), interpolation=cv2.INTER_LANCZOS4))
                         for i, c in ((i, crops[i]) for i in todo)], 0).to(dev)
        if a.ref_sigma > 0:                        # reference half: temporally smoothed -> stable teeth/lip texture
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
    # flow-aligned 3-frame temporal blend of the restored renders (alignment from the BASE frames' motion)
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
    # one global sharpen amount: median of the per-frame best k on every 3rd frame (at alpha 0.65)
    def lv_dev(i, comp):
        b = base[i]; mouth_r, cheek_r = regions(geo[i][2], M[i], h, w, cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
        if cheek_r is None:
            return None
        gb, gc = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
        rb = lapvar(gb, mouth_r) / max(1e-6, lapvar(gb, cheek_r)); ro = lapvar(gc, mouth_r) / max(1e-6, lapvar(gc, cheek_r))
        return ro / rb - 1
    rng = np.random.default_rng(0)
    grain = {}
    for i in todo:
        _, cheek_r = regions(geo[i][2], M[i], h, w, gray[i])
        hp = gray[i].astype(np.float32) - cv2.GaussianBlur(gray[i], (0, 0), 1.0).astype(np.float32)
        grain[i] = float(hp[cheek_r].std()) if cheek_r is not None else 0.0

    def compose(i, alpha, k, seed=None):
        b = base[i]; m = M[i] * alpha
        g = sharpen(Rs[i], KF[i] if k is None else k).astype(np.float32)
        if grain[i] > 0:
            g = g + np.random.default_rng(1000 + i).normal(0, grain[i] * a.grain, g.shape[:2])[..., None]
        comp = (b.astype(np.float32) * (1 - m[..., None]) + np.clip(g, 0, 255) * m[..., None]).round()
        return np.where(m[..., None] > 0, comp, b).astype(np.uint8)
    full_ = [i for i in todo if env_a[i] >= 0.99]
    ks = [min((abs(lv_dev(i, compose(i, 0.55, k)) or 9), k) for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0))[1] for i in full_]
    ksm = np.clip(savgol_filter(np.array(ks, float), min(9, len(ks) - (1 - len(ks) % 2)), 1, mode="interp"), 0, 6)
    KF = {i: float(ksm[int(np.argmin([abs(i - j) for j in full_]))]) for i in todo}
    K = float(np.median(ksm))
    print("sharpen k per frame (SG-smoothed)", [round(v, 2) for v in ksm], flush=True)

    def gate(i, comp):
        fc = detect(fa, comp)
        arc = float(np.dot(fc.normed_embedding, F[i].normed_embedding)) if fc is not None else None
        out = M[i] <= 0
        ss = float(ssim_map(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), gray[i])[out].mean())
        return arc, ss, arc is not None and arc >= a.t_id and ss >= a.t_ssim
    chosen, res = None, None
    for A0 in [float(v) for v in a.alphas.split(",")]:
        r = {i: gate(i, compose(i, A0 * env_a[i], None)) for i in todo}
        rate = sum(v[2] for v in r.values()) / len(r)
        print("alpha", A0, "pass rate", round(rate, 3), flush=True)
        chosen, res = A0, r
        if rate >= 0.97:
            break
    alpha = np.array([chosen * env_a[i] for i in range(n)])
    for i in todo:                                   # local slope-limited dips for residual failures
        if not res[i][2]:
            for sc in (0.75, 0.5, 0.3):
                if gate(i, compose(i, alpha[i] * sc, None))[2]:
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
        b = base[i]; row = {"i": i, "pos": round(float(pos[i]), 3), "alpha": round(float(alpha[i]), 3)}
        if i in R and alpha[i] > 1e-3:
            comp = compose(i, alpha[i], None); arc, ss, ok = gate(i, comp)
            dv = lv_dev(i, comp)
            row.update({"lipsync": True, "arcface": round(arc, 4) if arc else None, "ssim_outside": round(ss, 5),
                        "lv_ratio_dev": round(dv, 4) if dv is not None else None, "gate": "PASS" if ok else "FAIL"})
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
    np.save(D / "work/v4_final.npy", final)
    ls = [r for r in rows if r.get("lipsync")]
    arcs = [r["arcface"] for r in ls if r["arcface"]]; dvs = [r["lv_ratio_dev"] for r in ls if r["lv_ratio_dev"] is not None]
    full = [i for i in todo if env_a[i] >= 0.99]
    seg = list(range(full[0], full[-1] + 1))
    met = {"v4_out": mouth_metrics(final, fa, seg), "v4_base": mouth_metrics(base, fa, seg)}
    v3f, v3b = D / "work/v3_final.npy", D / "work/v3_base.npy"
    if v3f.exists():
        g3 = json.load(open(D / "yaying_v3_gate.json"))["rows"]
        s3 = [r["i"] for r in g3 if r.get("lipsync")]
        seg3 = list(range(s3[0], s3[-1] + 1))
        met["v3_out"] = mouth_metrics(np.load(v3f, mmap_mode="r"), fa, seg3)
        met["v3_base"] = mouth_metrics(np.load(v3b, mmap_mode="r"), fa, seg3)
    for k in ("v3", "v4"):
        if f"{k}_out" in met and met[f"{k}_base"]["flicker_mad_mean"]:
            met[f"{k}_flicker_ratio_vs_base"] = round(met[f"{k}_out"]["flicker_mad_mean"] / met[f"{k}_base"]["flicker_mad_mean"], 3)
            met[f"{k}_jitter_ratio_vs_base"] = round(met[f"{k}_out"]["jitter_mean"] / met[f"{k}_base"]["jitter_mean"], 3)
            met[f"{k}_flicker2_ratio_vs_base"] = round(met[f"{k}_out"]["flicker2_mean"] / met[f"{k}_base"]["flicker2_mean"], 3)
    summ = {"timeline": tl, "lipsync_frames": len(ls), "gate_pass": sum(r["gate"] == "PASS" for r in ls),
            "gate_fail_fallback": sum(r["gate"] == "FAIL" for r in ls), "alpha_global": chosen,
            "alpha_min_in_speech": float(min(alpha[i] for i in full)), "alpha_max_step": float(np.abs(np.diff(alpha)).max()),
            "ramp_frames": a.ramp, "sharpen_k_median": K, "sharpen_k_range": [float(ksm.min()), float(ksm.max())], "arcface_min": min(arcs), "arcface_median": float(np.median(arcs)),
            "ssim_outside_min": min(r["ssim_outside"] for r in ls), "lv_within_tol": sum(abs(v) <= a.lv_tol for v in dvs),
            "lv_frames": len(dvs), "lv_dev_median": float(np.median(dvs)), "lv_dev_abs_max": float(max(abs(v) for v in dvs)),
            "jitter_flicker": met, "params": vars(a), "elapsed_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "rows": rows}, open(out.with_name(out.stem + "_gate.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False)); print("V4_DONE", out)


if __name__ == "__main__":
    main()
