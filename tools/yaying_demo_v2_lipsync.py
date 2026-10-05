#!/usr/bin/env python3
"""Plan 40 demo v2, Option B: mouth-region-only lip-sync (MuseTalk 1.5) over the Option A base frames of the
ORIGINAL generated_video-3.mp4, with a per-frame identity/pixel gate.

Base frames = Option A2's exact pre-encode frames (work/A2_frames.npy: forward-only timeline, original frames or
RIFE v4.26 in-betweens of neighbouring original frames; no face edit, no reverse playback). For each output frame:
  1. insightface buffalo_l: face bbox, 5 kps, 106 landmarks (mouth = points 52-71)
  2. MuseTalk v1.5 (UNet single step + sd-vae-ft-mse + whisper-tiny audio features at 24 fps) re-renders the
     MuseTalk face crop; ONLY a feathered mouth mask (lip hull dilated toward the jaw) is composited back, scaled by
     a voice-activity alpha (0 in pauses, so silent frames stay the base frame), and 0 when the source position is
     past the mouth-visible window (fingers at the lips).
  3. gate: ArcFace(composite) . ArcFace(base) >= 0.90 AND SSIM(gray) outside the mask >= 0.99, else the base frame
     is used unchanged (fallback). Pixels where mask == 0 are copied from the base frame bit-for-bit.
Run (Mac): cd ~/yaying_cache/MuseTalk && ../venv_mt/bin/python <repo>/tools/yaying_demo_v2_lipsync.py
"""
import argparse, json, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
MT = Path.home() / "yaying_cache" / "MuseTalk"
sys.path.insert(0, str(MT))
import os
FPS = int(os.environ.get("YY_FPS", "24"))  # v6 candidates are 25 fps
MOUTH = list(range(52, 72))
MASK_SCALE = 1.0


def ssim_map(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu1, mu2 = blur(a), blur(b)
    s1, s2, s12 = blur(a * a) - mu1 ** 2, blur(b * b) - mu2 ** 2, blur(a * b) - mu1 * mu2
    return ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))


def mouth_mask(lm, h, w, feather):
    pts = lm[MOUTH].astype(np.float32)
    c = pts.mean(0)
    mw = np.ptp(pts[:, 0])
    # dilate the lip hull: 35% sideways, 45% upward, 110% downward (jaw drop while speaking)
    d = pts - c
    d[:, 0] *= 1.35 * MASK_SCALE
    d[:, 1] = np.where(d[:, 1] < 0, d[:, 1] * 1.45 - 0.06 * mw, d[:, 1] * 2.1 + 0.18 * mw) * MASK_SCALE
    hull = cv2.convexHull((c + d).astype(np.int32))
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, hull, 1.0)
    k = int(feather) | 1
    m = cv2.GaussianBlur(m, (k, k), feather / 3)
    return np.clip(m * 1.15, 0, 1), mw


def match_color(gen, base, m, sharpen):
    """Lab mean/std transfer of the MuseTalk render onto the base frame statistics inside the mouth mask (MuseTalk's
    VAE desaturates the red glossy lips), plus an unsharp mask to recover the VAE-softened detail."""
    sel = m > 0.5
    if sel.sum() < 50:
        return gen
    g = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        gm, gs = g[..., c][sel].mean(), g[..., c][sel].std() + 1e-3
        bm, bs = b[..., c][sel].mean(), b[..., c][sel].std() + 1e-3
        g[..., c] = (g[..., c] - gm) * (bs / gs) + bm
    out = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if sharpen > 0:
        bl = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1 + sharpen, bl, -sharpen, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/yaying_clone/demo_v2/work/A2_frames.npy")
    ap.add_argument("--timeline", default="runs/yaying_clone/demo_v2/yaying_v2_A2_original_timeline.json")
    ap.add_argument("--wav", default="runs/yaying_clone/demo_v2/yaying_v2_A2_original.wav")
    ap.add_argument("--out", default="runs/yaying_clone/demo_v2/yaying_v2_B_lipsync.mp4")
    ap.add_argument("--t-id", type=float, default=0.90)
    ap.add_argument("--t-ssim", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--color-match", type=int, default=1)
    ap.add_argument("--retry", type=int, default=1, help="retry a failing frame at 0.75x and 0.5x blend before fallback")
    ap.add_argument("--alpha-max", type=float, default=1.0, help="cap of the mouth-composite opacity")
    ap.add_argument("--sharpen", type=float, default=0.8)
    ap.add_argument("--mask-scale", type=float, default=1.0, help="<1 shrinks the dilated lip mask")
    ap.add_argument("--only", help="comma list of frame indices (diagnostics; skips encoding)")
    ap.add_argument("--debug-dir", help="write base|musetalk-crop|composite PNGs per processed frame")
    a = ap.parse_args()
    global MASK_SCALE
    MASK_SCALE = a.mask_scale
    import torch
    from insightface.app import FaceAnalysis
    from transformers import WhisperModel
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    t0 = time.time()
    base = np.load(ROOT / a.base, mmap_mode="r")
    tl = json.load(open(ROOT / a.timeline))
    pos = np.array(tl["timeline"])
    win_hi = tl.get("mouth_visible_max", tl["win_hi"])
    n, h, w, _ = base.shape
    dev = torch.device(a.device)
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                      allowed_modules=["detection", "landmark_2d_106", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    # voice activity alpha per output frame
    with wave.open(str(ROOT / a.wav)) as ww:
        sr = ww.getframerate()
        x = np.frombuffer(ww.readframes(ww.getnframes()), np.int16).astype(np.float32) / 32768
    hop = sr // FPS
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    act = (rms > 0.02).astype(np.float32)
    act = np.convolve(act, np.ones(5) / 5, "same")          # ~2 frame attack/release
    act = np.maximum.reduce([act, np.roll(act, 1), np.roll(act, -1)])
    act = np.clip(act * 1.5, 0, 1)
    vis = np.clip((win_hi + 2 - pos) / 2, 0, 1)             # fades to 0 over 2 source frames past the mouth-visible limit
    alpha = act * vis * a.alpha_max
    # faces
    faces = []
    keep_only = {int(v) for v in a.only.split(",")} if a.only else None
    for i in range(n):
        if keep_only is not None and i not in keep_only:
            faces.append(None); continue
        fs = fa.get(np.ascontiguousarray(base[i][:, :, ::-1]))
        faces.append(max(fs, key=lambda f: f.det_score) if fs else None)
    print("faces", sum(f is not None for f in faces), "/", n, round(time.time() - t0, 1), "s", flush=True)
    # MuseTalk
    vae = VAE(model_path=str(MT / "models/sd-vae"))
    vae.vae = vae.vae.to(dev)
    unet = UNet(unet_config=str(MT / "models/musetalkV15/musetalk.json"),
                model_path=str(MT / "models/musetalkV15/unet.pth"), device=dev)
    unet.model = unet.model.to(dev)
    pe = PositionalEncoding(d_model=384).to(dev)
    ap_ = AudioProcessor(feature_extractor_path=str(MT / "models/whisper"))
    whisper = WhisperModel.from_pretrained(str(MT / "models/whisper")).to(dev).eval()
    feats, L = ap_.get_audio_feature(str(ROOT / a.wav))
    chunks = ap_.get_whisper_chunk(feats, dev, unet.model.dtype, whisper, L, fps=FPS,
                                   audio_padding_length_left=2, audio_padding_length_right=2)
    print("whisper chunks", len(chunks), "frames", n, flush=True)
    todo = [i for i in range(n) if alpha[i] > 0.01 and faces[i] is not None and i < len(chunks)]
    if a.only:
        keep = {int(v) for v in a.only.split(",")}
        todo = [i for i in todo if i in keep]
    crops = {}
    for i in todo:
        f = faces[i]
        x1, y1, x2, y2 = f.bbox
        kps = f.kps
        half = (kps[0][1] + kps[1][1]) / 2 * 0.45 + kps[2][1] * 0.55      # ~ nose-bridge midpoint (68-pt #29)
        yb = y2 + 10
        ya = max(0, half - (y2 - half))
        crops[i] = [int(max(0, x1)), int(ya), int(min(w, x2)), int(min(h, yb))]
    out_frames = {}
    tstep = torch.tensor([0], device=dev)
    with torch.no_grad():
        for b0 in range(0, len(todo), a.batch):
            ids = todo[b0:b0 + a.batch]
            lat = []
            for i in ids:
                cx1, cy1, cx2, cy2 = crops[i]
                crop = cv2.resize(np.ascontiguousarray(base[i][cy1:cy2, cx1:cx2, ::-1]), (256, 256),
                                  interpolation=cv2.INTER_LANCZOS4)
                lat.append(vae.get_latents_for_unet(crop))
            lat = torch.cat(lat, 0).to(dev)
            wb = torch.stack([chunks[i] for i in ids]).to(dev)
            pred = unet.model(lat, tstep, encoder_hidden_states=pe(wb)).sample
            for i, r in zip(ids, vae.decode_latents(pred)):
                out_frames[i] = r                                               # BGR 256x256
            print("musetalk", b0 + len(ids), "/", len(todo), round(time.time() - t0, 1), "s", flush=True)
    rows, final = [], []
    for i in range(n):
        b = np.array(base[i])
        row = {"i": i, "pos": round(float(pos[i]), 2), "alpha": round(float(alpha[i]), 3), "face": faces[i] is not None}
        if i in out_frames:
            cx1, cy1, cx2, cy2 = crops[i]
            gen = b.copy()
            gen[cy1:cy2, cx1:cx2] = cv2.resize(out_frames[i].astype(np.uint8), (cx2 - cx1, cy2 - cy1),
                                               interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1]
            m, mw = mouth_mask(faces[i].landmark_2d_106, h, w, feather=max(9, int(0.35 * np.ptp(
                faces[i].landmark_2d_106[MOUTH][:, 0]))))
            if a.color_match:
                gen = match_color(gen, b, m, a.sharpen)
            m0 = m
            tried = []
            for scale in [1.0] + [v for v in (0.75, 0.5) if a.retry]:      # adaptive: weaker blend before fallback
                m = m0 * float(alpha[i]) * scale
                comp = (b.astype(np.float32) * (1 - m[..., None]) + gen.astype(np.float32) * m[..., None]).round()
                comp = np.where(m[..., None] > 0, comp, b).astype(np.uint8)
                fs = fa.get(np.ascontiguousarray(comp[:, :, ::-1]))
                fc = max(fs, key=lambda f: f.det_score) if fs else None
                arc = float(np.dot(fc.normed_embedding, faces[i].normed_embedding)) if fc is not None else None
                outside = m <= 0.0
                sm = ssim_map(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
                ss = float(sm[outside].mean())
                ok = arc is not None and arc >= a.t_id and ss >= a.t_ssim
                tried.append([round(float(alpha[i]) * scale, 3), round(arc, 4) if arc is not None else None])
                if ok:
                    break
            ident = bool(np.array_equal(comp[outside], b[outside]))
            row.update({"lipsync": True, "arcface": round(arc, 4) if arc is not None else None,
                        "ssim_outside": round(ss, 5), "outside_pixel_identical": ident, "tried": tried,
                        "blend": tried[-1][0] if ok else 0.0,
                        "mask_px": int((m > 0).sum()), "gate": "PASS" if ok else "FAIL"})
            final.append(comp if ok else b)
            if a.debug_dir:
                dd = Path(a.debug_dir); dd.mkdir(parents=True, exist_ok=True)
                cx1, cy1, cx2, cy2 = crops[i]
                tile = lambda im: cv2.resize(im[cy1:cy2, cx1:cx2], (256, 256))  # noqa: E731
                mm = (np.repeat(m[..., None], 3, 2) * 255).astype(np.uint8)
                cv2.imwrite(str(dd / f"f{i:04d}_arc{arc}.png"), np.concatenate(
                    [tile(b), tile(gen), tile(comp), tile(mm)], 1)[:, :, ::-1])
        else:
            row.update({"lipsync": False, "gate": "BASE"})
            final.append(b)
        rows.append(row)
    if a.only:
        print([{k: r.get(k) for k in ("i", "alpha", "arcface", "ssim_outside", "gate")} for r in rows if r.get("lipsync")])
        return
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-i", str(ROOT / a.wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow",
           "-crf", "12", "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-shortest", str(ROOT / a.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in final:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    np.save(ROOT / a.base.replace("A2_frames", "B_frames"), np.stack(final))
    ls = [r for r in rows if r.get("lipsync")]
    arcs = [r["arcface"] for r in ls if r["arcface"] is not None]
    sss = [r["ssim_outside"] for r in ls]
    summ = {"frames": n, "speech_frames_alpha_gt0": int((alpha > 0.01).sum()), "lipsync_attempted": len(ls),
            "gate_pass": sum(r["gate"] == "PASS" for r in ls), "gate_fail_fallback": sum(r["gate"] == "FAIL" for r in ls),
            "base_frames": sum(r["gate"] == "BASE" for r in rows),
            "arcface_min": min(arcs) if arcs else None, "arcface_median": float(np.median(arcs)) if arcs else None,
            "ssim_outside_min": min(sss) if sss else None, "ssim_outside_median": float(np.median(sss)) if sss else None,
            "outside_pixel_identical_all": all(r["outside_pixel_identical"] for r in ls),
            "faces_found_base": sum(f is not None for f in faces),
            "blend_median_pass": float(np.median([r["blend"] for r in ls if r["gate"] == "PASS"])) if any(r["gate"] == "PASS" for r in ls) else None,
            "params": {"alpha_max": a.alpha_max, "mask_scale": a.mask_scale, "color_match": a.color_match, "sharpen": a.sharpen, "retry": a.retry}, "thresholds": {"arcface": a.t_id, "ssim_outside": a.t_ssim},
            "model": "MuseTalk v1.5 (TMElyralab/MuseTalk musetalkV15/unet.pth, sd-vae-ft-mse, whisper-tiny)",
            "device": a.device, "elapsed_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "rows": rows}, open((ROOT / a.out).with_name(Path(a.out).stem + "_gate.json"), "w"), indent=1)
    print(summ)
    print("B_DONE")


if __name__ == "__main__":
    main()
