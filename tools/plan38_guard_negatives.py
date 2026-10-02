"""Plan 38 E-GUARD-NEG: harder negative controls for the appearance guard.
Scores every *other* person's portrait in LivePortrait's assets/examples/source/*.jpg (several young women,
incl. East-Asian faces) against the FIXED reference web/gemma_chat/avatar.jpg with the same ArcFace/CLIP guard.
Writes runs/plan38/appearance/guard_negatives_lp.json"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g", ROOT / "tools/plan38_appearance_guard.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
SRC = Path.home() / "plan38_cache/LivePortrait/assets/examples/source"

def main():
    G = g.Guard(); ref = G.measure(Image.open(ROOT / "web/gemma_chat/avatar.jpg").convert("RGB"))
    rows = []
    for p in sorted(SRC.glob("*.jpg")):
        t0 = time.perf_counter(); m = G.measure(Image.open(p).convert("RGB"))
        rows.append({"img": p.name, "face": "arcface" in m, "arcface": g.cos(ref["arcface"], m.get("arcface")),
                     "clip_face": g.cos(ref["clip_face"], m.get("clip_face")), "clip_full": g.cos(ref["clip_full"], m["clip_full"]),
                     "s": round(time.perf_counter() - t0, 2)})
        print(json.dumps(rows[-1]))
    af = [r["arcface"] for r in rows if r["arcface"] is not None]; cf = [r["clip_face"] for r in rows if r["clip_face"] is not None]
    rep = {"source": str(SRC), "n": len(rows), "faces": len(af), "arcface_max": max(af), "arcface_p95": float(np.percentile(af, 95)),
           "clip_face_max": max(cf), "clip_full_max": max(r["clip_full"] for r in rows), "rows": rows}
    out = ROOT / "runs/plan38/appearance/guard_negatives_lp.json"; out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}))

if __name__ == "__main__": main()
