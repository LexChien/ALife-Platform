#!/usr/bin/env python3
"""Plan 38 E-APP: appearance-invariance guard for the FIXED avatar (web/gemma_chat/avatar.jpg).
Metrics: (1) ArcFace identity cosine (insightface buffalo_l), (2) OpenCLIP ViT-B-32 whole-image cosine,
(3) OpenCLIP face-crop cosine. Sets: reference video frames (same character), CSS emotion filters used today,
lip-sync prototype frames, optional extra dirs (e.g. LivePortrait output), and NEGATIVE controls
(another woman's photo = skimage astronaut, blurred-face avatar, non-face images).
Usage: plan38_appearance_guard.py [--extra name=dir_or_mp4 ...]   -> runs/plan38/appearance/guard_report.json"""
import json, sys, time, hashlib, argparse, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "web/gemma_chat/avatar.jpg"; VID = ROOT / "web/gemma_chat/generated_video-3.mp4"
OUT = ROOT / "runs/plan38/appearance"; OUT.mkdir(parents=True, exist_ok=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frames_from_mp4(p, every=6):
    d = Path(tempfile.mkdtemp()); subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"select=not(mod(n\\,{every}))", "-vsync", "vfr", str(d / "f_%04d.png")], check=True)
    return [Image.open(f).convert("RGB") for f in sorted(d.glob("*.png"))]

def css_filter(img, sat=1.0, hue_deg=0.0, bright=1.0, contrast=1.0):
    x = ImageEnhance.Contrast(img).enhance(contrast); x = ImageEnhance.Brightness(x).enhance(bright); x = ImageEnhance.Color(x).enhance(sat)
    if hue_deg:
        h, s, v = x.convert("HSV").split(); h = h.point(lambda p: int((p + hue_deg / 360 * 255) % 255)); x = Image.merge("HSV", (h, s, v)).convert("RGB")
    return x

class Guard:
    def __init__(self):
        import open_clip, torch
        self.torch = torch
        self.clip, _, self.prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip.eval()
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); self.fa.prepare(ctx_id=-1, det_size=(640, 640))
    def clip_emb(self, img):
        with self.torch.no_grad():
            e = self.clip.encode_image(self.prep(img).unsqueeze(0))[0].numpy()
        return e / np.linalg.norm(e)
    def face(self, img):
        """detect with rotation search (avatar is lying down / tilted)"""
        best = None; arr0 = np.array(img)[:, :, ::-1]
        for ang in (0, -30, 30, -60, 60, 90, -90):
            im = img.rotate(ang, expand=True) if ang else img
            fs = self.fa.get(np.array(im)[:, :, ::-1])
            for f in fs:
                if best is None or f.det_score > best[0].det_score: best = (f, ang, im)
            if best and best[0].det_score > 0.7: break
        return best
    def measure(self, img):
        r = {"clip_full": self.clip_emb(img)}
        b = self.face(img)
        if b:
            f, ang, im = b; x0, y0, x1, y1 = [int(v) for v in f.bbox]
            pad = int(0.25 * (x1 - x0)); crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
            r.update({"arcface": f.normed_embedding, "clip_face": self.clip_emb(crop), "det_score": float(f.det_score), "rot": ang})
        return r

def cos(a, b): return float(np.dot(a, b)) if a is not None and b is not None else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", nargs="*", default=[]); a = ap.parse_args()
    t0 = time.perf_counter(); g = Guard(); load_s = time.perf_counter() - t0
    ref_img = Image.open(REF).convert("RGB"); ref = g.measure(ref_img)
    assert "arcface" in ref, "no face detected in reference avatar"
    sets = {}
    sets["ref_self"] = [ref_img]
    sets["ref_video_frames"] = frames_from_mp4(VID, every=6)
    sets["css_filters_current"] = [css_filter(ref_img, contrast=1.1, bright=0.85),  # .noir-image-back base
        css_filter(ref_img, sat=0.75, hue_deg=-12), css_filter(ref_img, sat=0.85, bright=0.97), css_filter(ref_img, sat=0.7),
        css_filter(ref_img, sat=1.2, bright=1.05), css_filter(ref_img, contrast=1.08)]
    sets["mirror"] = [ref_img.transpose(Image.FLIP_LEFT_RIGHT)]
    for spec in a.extra:
        name, path = spec.split("=", 1); p = Path(path)
        sets[name] = frames_from_mp4(p, every=3) if p.suffix == ".mp4" else [Image.open(f).convert("RGB") for f in sorted(p.glob("*.png"))][::3]
    # negatives
    from skimage import data
    neg = {"neg_other_woman_astronaut": [Image.fromarray(data.astronaut())], "neg_cat": [Image.fromarray(data.chelsea())]}
    fb = g.face(ref_img); f, ang, im = fb; x0, y0, x1, y1 = [int(v) for v in f.bbox]
    blurred = im.copy(); blurred.paste(im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(12)), (x0, y0))
    neg["neg_blurred_face_avatar"] = [blurred.rotate(-ang, expand=False) if ang else blurred]
    asal = sorted((ROOT / "runs").glob("asal/*/**/*.png"))[:1]
    if asal: neg["neg_asal_organism"] = [Image.open(asal[0]).convert("RGB")]
    sets.update(neg)
    report = {"reference": {"path": str(REF.relative_to(ROOT)), "sha256": sha(REF), "size": ref_img.size, "det_score": ref["det_score"], "det_rotation": ref["rot"]},
              "video": {"path": str(VID.relative_to(ROOT)), "sha256": sha(VID)}, "model_load_s": round(load_s, 2), "sets": {}}
    for name, imgs in sets.items():
        rows = []; t1 = time.perf_counter()
        for im_ in imgs:
            m = g.measure(im_)
            rows.append({"arcface": cos(ref["arcface"], m.get("arcface")), "clip_full": cos(ref["clip_full"], m["clip_full"]),
                         "clip_face": cos(ref["clip_face"], m.get("clip_face")), "face_found": "arcface" in m})
        dt = (time.perf_counter() - t1) / max(1, len(imgs))
        def stat(k):
            v = [r[k] for r in rows if r[k] is not None]
            return {"min": round(min(v), 4), "median": round(float(np.median(v)), 4), "max": round(max(v), 4)} if v else None
        report["sets"][name] = {"n": len(rows), "faces_found": sum(r["face_found"] for r in rows), "arcface": stat("arcface"),
                                "clip_full": stat("clip_full"), "clip_face": stat("clip_face"), "s_per_image": round(dt, 3)}
        print(name, json.dumps(report["sets"][name]))
    pos = [s for n, s in report["sets"].items() if not n.startswith("neg_") and s["arcface"]]
    negs = [s for n, s in report["sets"].items() if n.startswith("neg_") and s["arcface"]]
    report["separation"] = {"min_positive_arcface": min(s["arcface"]["min"] for s in pos),
                            "max_negative_arcface": max([s["arcface"]["max"] for s in negs] or [None])}
    (OUT / "guard_report.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report["separation"]))

if __name__ == "__main__": main()
