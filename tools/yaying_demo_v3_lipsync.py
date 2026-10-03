#!/usr/bin/env python3
"""Plan 40 demo v3 step 2: mouth-only lip-sync over the v3 motion-aware base (work/v3_base.npy, ORIGINAL frames or
low-motion RIFE in-betweens), fixing the review findings on v2-B:
  * soft/low-res MuseTalk mouth -> restore the composited face with GFPGAN v1.4 (aligned 512 crop via the 5 kps,
    fidelity blend --fidelity 0.7 = 70% restored + 30% input), then match sharpness and grain to the surrounding skin:
    Laplacian-variance ratio LV(mouth)/LV(cheek) of the output must be within +-15% of the same ratio in the base
    frame (unsharp/blur amount searched per frame; grain = Gaussian noise at the cheek high-pass std)
  * finger halo -> MediaPipe Hands mask (dilated hull of 21 landmarks) removed from the mouth composite; speech is
    already placed before the hand reaches the face (see v3_plan.json)
  * per-frame gate: ArcFace(composite, base) >= 0.88 and SSIM outside the mask >= 0.99; blend retried at 0.75x and
    0.5x, then a 3-frame minimum filter on the accepted blend (no opacity flicker) and a final re-gate; failures fall
    back to the base frame. Pixels outside the mask are copied bit-for-bit.
  cd ~/yaying_cache/MuseTalk && ../venv_mt/bin/python <repo>/tools/yaying_demo_v3_lipsync.py
"""
import argparse, json, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import yaying_demo_v2_lipsync as v2  # noqa: E402
from yaying_demo_v2_lipsync import MOUTH, ssim_map  # noqa: E402
MT = Path.home() / "yaying_cache" / "MuseTalk"
sys.path.insert(0, str(MT))
D = ROOT / "runs/yaying_clone/demo_v2"
FPS = 24
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)


def lapvar(gray, mask):
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=3)
    return float(lap[mask].var()) if mask.sum() > 20 else None


class Restorer:
    def __init__(self, dev):
        import torch
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        self.torch, self.dev = torch, dev
        self.net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                                 fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
                                 sft_half=True)
        sd = torch.load(Path.home() / "yaying_cache/gfpgan/GFPGANv1.4.pth", map_location="cpu")
        self.net.load_state_dict(sd["params_ema"], strict=True)
        self.net.eval().to(dev)

    def __call__(self, img, kps, fidelity):
        torch = self.torch
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
        al = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        x = torch.from_numpy(al).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            y = self.net(x, return_rgb=False)[0]
        y = ((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        y = (fidelity * y.astype(np.float32) + (1 - fidelity) * al.astype(np.float32)).round().astype(np.uint8)
        Mi = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        return cv2.warpAffine(y, Mi, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def regions(lm, m_dil, h, w, gray):
    pts = lm[MOUTH].astype(np.float32)
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts.astype(np.int32)), 1)
    face = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(lm[:33].astype(np.int32)), 1)
    c = pts.mean(0); mw = np.ptp(pts[:, 0]); r = int(max(6, 0.17 * mw))
    best = None
    for sx in (-1, 1):
        cx, cy = int(c[0] + sx * 0.95 * mw), int(c[1] - 0.45 * mw)
        box = np.zeros((h, w), bool); box[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = True
        box &= (face > 0) & (m_dil <= 0)
        if box.sum() < 40:
            continue
        score = box.sum() * float(gray[box].mean() > 50)
        if best is None or score > best[0]:
            best = (score, box)
    return hull > 0, (best[1] if best else None)


def sharpen(img, k):
    if abs(k) < 1e-3:
        return img
    bl = cv2.GaussianBlur(img, (0, 0), 1.0)
    if k > 0:
        return cv2.addWeighted(img, 1 + k, bl, -k, 0)
    return cv2.addWeighted(img, 1 + k, bl, -k, 0)          # k<0: blend toward blur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v3.mp4")
    ap.add_argument("--t-id", type=float, default=0.88)
    ap.add_argument("--t-ssim", type=float, default=0.99)
    ap.add_argument("--alpha-max", type=float, default=0.8)
    ap.add_argument("--mask-scale", type=float, default=0.8)
    ap.add_argument("--fidelity", type=float, default=0.7)
    ap.add_argument("--lv-tol", type=float, default=0.15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--grain", type=float, default=0.8, help="grain noise std as a fraction of the cheek high-pass std")
    ap.add_argument("--blend-slope", type=float, default=0.08, help="max blend change per frame (anti-flicker)")
    a = ap.parse_args()
    v2.MASK_SCALE = a.mask_scale
    import torch
    import mediapipe as mp
    from insightface.app import FaceAnalysis
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    t0 = time.time()
    plan = json.load(open(D / "work/v3_plan.json"))
    pos = np.array(plan["timeline"])
    base = np.load(D / "work/v3_base.npy", mmap_mode="r")
    n, h, w, _ = base.shape
    wav = D / "work/v3.wav"
    dev = torch.device(a.device)
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                      allowed_modules=["detection", "landmark_2d_106", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    best_face = lambda im: (lambda fs: max(fs, key=lambda f: f.det_score) if fs else None)(fa.get(np.ascontiguousarray(im[:, :, ::-1])))  # noqa
    with wave.open(str(wav)) as ww:
        sr = ww.getframerate(); x = np.frombuffer(ww.readframes(ww.getnframes()), np.int16).astype(np.float32) / 32768
    hop = sr // FPS
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    act = np.convolve((rms > 0.02).astype(np.float32), np.ones(5) / 5, "same")
    act = np.clip(np.maximum.reduce([act, np.roll(act, 1), np.roll(act, -1)]) * 1.5, 0, 1)
    vis = np.clip((plan["speech_end_max_pos"] + 2 - pos) / 2, 0, 1)
    alpha = act * vis * a.alpha_max
    todo = [i for i in range(n) if alpha[i] > 0.01]
    faces = {i: best_face(np.array(base[i])) for i in todo}
    todo = [i for i in todo if faces[i] is not None]
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    vae = VAE(model_path=str(MT / "models/sd-vae")); vae.vae = vae.vae.to(dev)
    unet = UNet(unet_config=str(MT / "models/musetalkV15/musetalk.json"), model_path=str(MT / "models/musetalkV15/unet.pth"), device=dev)
    unet.model = unet.model.to(dev)
    pe = PositionalEncoding(d_model=384).to(dev)
    apx = AudioProcessor(feature_extractor_path=str(MT / "models/whisper"))
    whisper = WhisperModel.from_pretrained(str(MT / "models/whisper")).to(dev).eval()
    feats, L = apx.get_audio_feature(str(wav))
    chunks = apx.get_whisper_chunk(feats, dev, unet.model.dtype, whisper, L, fps=FPS, audio_padding_length_left=2,
                                   audio_padding_length_right=2)
    rest = Restorer(dev)
    crops, gens = {}, {}
    for i in todo:
        f = faces[i]; x1, y1, x2, y2 = f.bbox; k = f.kps
        half = (k[0][1] + k[1][1]) / 2 * 0.45 + k[2][1] * 0.55
        crops[i] = [int(max(0, x1)), int(max(0, half - (y2 - half))), int(min(w, x2)), int(min(h, y2 + 10))]
    tstep = torch.tensor([0], device=dev)
    with torch.no_grad():
        for b0 in range(0, len(todo), 8):
            ids = todo[b0:b0 + 8]
            lat = torch.cat([vae.get_latents_for_unet(cv2.resize(np.ascontiguousarray(np.array(base[i])[c[1]:c[3], c[0]:c[2], ::-1]), (256, 256), interpolation=cv2.INTER_LANCZOS4))
                             for i, c in ((i, crops[i]) for i in ids)], 0).to(dev)
            pred = unet.model(lat, tstep, encoder_hidden_states=pe(torch.stack([chunks[i] for i in ids]).to(dev))).sample
            for i, r in zip(ids, vae.decode_latents(pred)):
                gens[i] = r
    print("musetalk", len(gens), round(time.time() - t0, 1), "s", flush=True)
    prep = {}
    for i in todo:
        b = np.array(base[i]); c = crops[i]; f = faces[i]
        gen = b.copy()
        gen[c[1]:c[3], c[0]:c[2]] = cv2.resize(gens[i].astype(np.uint8), (c[2] - c[0], c[3] - c[1]), interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1]
        m, _ = v2.mouth_mask(f.landmark_2d_106, h, w, feather=max(9, int(0.35 * np.ptp(f.landmark_2d_106[MOUTH][:, 0]))))
        gen = v2.match_color(gen, b, m, 0.0)
        gen = rest(gen, f.kps, a.fidelity)
        gen = v2.match_color(gen, b, m, 0.0)
        hm = np.zeros((h, w), np.float32)
        res = hands.process(np.ascontiguousarray(b))
        nh = 0
        for hl in (res.multi_hand_landmarks or []):
            p = np.array([[q.x * w, q.y * h] for q in hl.landmark], np.float32)
            cv2.fillConvexPoly(hm, cv2.convexHull(p.astype(np.int32)), 1.0); nh += 1
        if nh:
            hm = cv2.GaussianBlur(cv2.dilate(hm, np.ones((31, 31), np.uint8)), (21, 21), 7)
        m = m * (1 - np.clip(hm, 0, 1))
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        mouth_r, cheek_r = regions(f.landmark_2d_106, m, h, w, gray_b)
        hp = gray_b.astype(np.float32) - cv2.GaussianBlur(gray_b, (0, 0), 1.0).astype(np.float32)
        grain = float(hp[cheek_r].std()) if cheek_r is not None else 0.0
        prep[i] = dict(b=b, gen=gen, m=m, mouth=mouth_r, cheek=cheek_r, grain=grain, hand=nh,
                       hand_overlap_px=int(((hm > 0.05) & (m > 0)).sum()))
    rng = np.random.default_rng(0)

    def compose(i, blend):
        P = prep[i]; b, gen, m = P["b"], P["gen"], P["m"] * blend
        noise = rng.normal(0, P["grain"] * a.grain, gen.shape[:2])[..., None] if P["grain"] > 0 else 0
        best = None
        rb = None
        if P["cheek"] is not None:
            gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            rb = lapvar(gb, P["mouth"]) / max(1e-6, lapvar(gb, P["cheek"]))
        for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, -0.3, -0.6):
            g = np.clip(sharpen(gen, k).astype(np.float32) + noise, 0, 255)
            comp = (b.astype(np.float32) * (1 - m[..., None]) + g * m[..., None]).round()
            comp = np.where(m[..., None] > 0, comp, b).astype(np.uint8)
            if rb is None:
                return comp, None, None, k
            gc = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            ro = lapvar(gc, P["mouth"]) / max(1e-6, lapvar(gc, P["cheek"]))
            dev_ = ro / rb - 1
            if best is None or abs(dev_) < abs(best[2]):
                best = (comp, ro, dev_, k)
            if abs(dev_) <= a.lv_tol:
                break
        return best[0], best[1], best[2], best[3]

    def gate(i, comp):
        fc = best_face(comp)
        arc = float(np.dot(fc.normed_embedding, faces[i].normed_embedding)) if fc is not None else None
        outside = prep[i]["m"] <= 0
        ss = float(ssim_map(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), cv2.cvtColor(prep[i]["b"], cv2.COLOR_RGB2GRAY))[outside].mean())
        return arc, ss, (arc is not None and arc >= a.t_id and ss >= a.t_ssim)

    accepted = np.zeros(n)
    for i in todo:
        for sc in (1.0, 0.75, 0.5):
            comp, ro, dv, k = compose(i, alpha[i] * sc)
            arc, ss, ok = gate(i, comp)
            if ok:
                accepted[i] = alpha[i] * sc; break
    smooth = accepted.copy()
    for i in todo:                                      # 5-frame minimum filter, then a slope limiter
        nbr = [accepted[j] for j in range(i - 2, i + 3) if 0 <= j < n and j in prep and accepted[j] > 0]
        smooth[i] = min(nbr) if accepted[i] > 0 else 0.0
    for i in range(1, n):
        smooth[i] = min(smooth[i], smooth[i - 1] + a.blend_slope) if smooth[i] > 0 else 0.0
    for i in range(n - 2, -1, -1):
        smooth[i] = min(smooth[i], smooth[i + 1] + a.blend_slope) if smooth[i] > 0 else 0.0
    rows, final = [], []
    for i in range(n):
        b = np.array(base[i]); row = {"i": i, "pos": round(float(pos[i]), 3), "alpha": round(float(alpha[i]), 3)}
        if i in prep and smooth[i] > 0.01:
            comp, ro, dv, k = compose(i, smooth[i])
            arc, ss, ok = gate(i, comp)
            outside = prep[i]["m"] * smooth[i] <= 0
            row.update({"lipsync": True, "blend": round(float(smooth[i]), 3), "arcface": round(arc, 4) if arc else None,
                        "ssim_outside": round(ss, 5), "lv_ratio_out": round(ro, 4) if ro else None,
                        "lv_ratio_dev": round(dv, 4) if dv is not None else None, "sharpen_k": k,
                        "hands": prep[i]["hand"], "hand_overlap_px_removed": prep[i]["hand_overlap_px"],
                        "outside_pixel_identical": bool(np.array_equal(comp[outside], b[outside])),
                        "gate": "PASS" if ok else "FAIL"})
            final.append(comp if ok else b)
        else:
            row.update({"lipsync": False, "gate": "BASE" if i not in prep else "FALLBACK_NO_PASSING_BLEND"})
            final.append(b)
        rows.append(row)
    out = ROOT / a.out
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-profile:v",
           "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in final:
        p.stdin.write(fr.tobytes())
    p.stdin.close(); assert p.wait() == 0
    np.save(D / "work/v3_final.npy", np.stack(final))
    ls = [r for r in rows if r.get("lipsync")]
    arcs = [r["arcface"] for r in ls if r["arcface"]]; dvs = [r["lv_ratio_dev"] for r in ls if r["lv_ratio_dev"] is not None]
    summ = {"frames": n, "speech_frames": len([i for i in range(n) if alpha[i] > 0.01]), "lipsync_frames": len(ls),
            "gate_pass": sum(r["gate"] == "PASS" for r in ls), "gate_fail_fallback": sum(r["gate"] == "FAIL" for r in ls),
            "no_passing_blend_fallback": sum(r["gate"] == "FALLBACK_NO_PASSING_BLEND" for r in rows),
            "arcface_min": min(arcs) if arcs else None, "arcface_median": float(np.median(arcs)) if arcs else None,
            "ssim_outside_min": min(r["ssim_outside"] for r in ls) if ls else None,
            "lv_ratio_dev_abs_max": float(max(abs(v) for v in dvs)) if dvs else None,
            "lv_ratio_dev_median": float(np.median(dvs)) if dvs else None,
            "lv_within_tol": sum(abs(v) <= a.lv_tol for v in dvs), "frames_with_hand": sum(r.get("hands", 0) > 0 for r in ls),
            "hand_px_removed_total": int(sum(r.get("hand_overlap_px_removed", 0) for r in ls)),
            "blend_min": float(min(r["blend"] for r in ls)) if ls else None, "blend_median": float(np.median([r["blend"] for r in ls])) if ls else None,
            "blend_max_step": float(max(abs(ls[k]["blend"] - ls[k - 1]["blend"]) for k in range(1, len(ls)) if ls[k]["i"] == ls[k - 1]["i"] + 1)) if len(ls) > 1 else None,
            "outside_pixel_identical_all": all(r["outside_pixel_identical"] for r in ls),
            "params": vars(a), "models": "MuseTalk v1.5 + GFPGAN v1.4 (fidelity blend) + insightface buffalo_l + MediaPipe Hands 0.10.14",
            "elapsed_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "rows": rows}, open(out.with_name(out.stem + "_gate.json"), "w"), indent=1)
    print(summ); print("V3_DONE", out)


if __name__ == "__main__":
    main()
