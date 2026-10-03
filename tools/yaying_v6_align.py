#!/usr/bin/env python3
"""Plan 40 v6: Mandarin character-level forced alignment of the approved lines -> pinyin -> viseme timeline.

mlx-whisper large-v3 (word_timestamps, zh) gives per-token times; the KNOWN approved text is aligned to the ASR
characters (difflib on Hanzi), unmatched characters are interpolated inside their phrase. Each character is mapped
with pypinyin to (initial, final) and to viseme events used by the lip pass and the compliance metric:
  closure  : initial b/p/m (lips must close at the onset)        labiodental : f
  round    : finals u/o/ü-type (u, o, uo, ou, ong, iong, ü, üe, un...)
  spread   : finals i-type (i, ie, in, ing, ian, -i after zh/ch/sh/r/z/c/s) and e/ei
  open     : a-type (a, ai, ao, an, ang, ia, iao, ua...)
Anticipatory coarticulation: each event carries a 'shape_from' time = onset - --anticip frames.
  ~/yaying_cache/venv/bin/python tools/yaying_v6_align.py --audio <wav> --out <json>
"""
import argparse, difflib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINES = ["嗨，你終於來了。我等你好久了喔。", "陽光好暖，可是你在我旁邊，我更喜歡。", "靠近一點嘛，我想聽你說說今天的事。", "你在想我嗎？說真的，不可以騙我喔。"]


def viseme(ini, fin):
    ev = []
    if ini in ("b", "p", "m"):
        ev.append("closure")
    if ini == "f":
        ev.append("labiodental")
    f = fin
    if ini in ("j", "q", "x", "y") and f.startswith("u"):
        f = "v" + f[1:]                                            # ü written as u after j/q/x/y
    if f in ("i",) and ini in ("zh", "ch", "sh", "r", "z", "c", "s"):
        ev.append("spread")
    elif f.startswith(("u", "o", "v")) or f in ("ou", "ong", "iong", "iu", "uo"):
        ev.append("round")
    elif f.startswith("i") or f in ("e", "ei", "en", "eng", "er"):
        ev.append("spread")
    elif f.startswith("a") or f in ("ia", "iao", "ian", "iang", "ua", "uai", "uan", "uang"):
        ev.append("open")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lines", default="0,1,2"); ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--anticip", type=int, default=2)
    a = ap.parse_args()
    import mlx_whisper
    from pypinyin import pinyin, Style
    res = mlx_whisper.transcribe(a.audio, path_or_hf_repo="mlx-community/whisper-large-v3-mlx", language="zh", word_timestamps=True,
                                 initial_prompt="".join(LINES[int(i)] for i in a.lines.split(",")))
    asr = []
    for seg in res["segments"]:
        for w in seg.get("words", []):
            chars = [c for c in w["word"] if re.match(r"[\u4e00-\u9fff]", c)]
            if not chars:
                continue
            d = (w["end"] - w["start"]) / len(chars)
            for k, c in enumerate(chars):
                asr.append({"c": c, "start": w["start"] + k * d, "end": w["start"] + (k + 1) * d})
    text = "".join(LINES[int(i)] for i in a.lines.split(","))
    known = [c for c in text if re.match(r"[\u4e00-\u9fff]", c)]
    # whisper may output simplified chars: compare on pinyin (tone-less) instead of glyphs
    pk = [p[0] for p in pinyin(known, style=Style.NORMAL)]
    pa = [p[0] for p in pinyin([x["c"] for x in asr], style=Style.NORMAL)]
    sm = difflib.SequenceMatcher(a=pk, b=pa, autojunk=False)
    t = [None] * len(known)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            t[blk.a + k] = (asr[blk.b + k]["start"], asr[blk.b + k]["end"])
    matched = sum(x is not None for x in t)
    for k in range(len(known)):                                   # interpolate unmatched chars
        if t[k] is None:
            lo = next((t[j][1] for j in range(k - 1, -1, -1) if t[j] is not None), 0.0)
            hi_j = next((j for j in range(k + 1, len(known)) if t[j] is not None), None)
            hi = t[hi_j][0] if hi_j is not None else lo + 0.25
            nmiss = (hi_j if hi_j is not None else len(known)) - k
            span = (hi - lo) / max(1, nmiss + 0)
            t[k] = (lo, lo + span)
    ini = [p[0] for p in pinyin(known, style=Style.INITIALS, strict=False)]
    fin = [p[0] for p in pinyin(known, style=Style.FINALS, strict=False)]
    chars = []
    for k, c in enumerate(known):
        ev = viseme(ini[k], fin[k])
        s, e = t[k]
        chars.append({"c": c, "pinyin": pk[k], "initial": ini[k], "final": fin[k], "start": round(s, 3), "end": round(e, 3),
                      "onset_frame": int(round(s * a.fps)), "shape_from_frame": int(round(s * a.fps)) - a.anticip,
                      "visemes": ev, "matched": k < len(t)})
    out = {"audio": a.audio, "fps": a.fps, "anticip_frames": a.anticip, "asr_text": res["text"], "chars_known": len(known),
           "chars_matched_to_asr": matched, "chars": chars,
           "counts": {v: sum(v in ch["visemes"] for ch in chars) for v in ("closure", "labiodental", "round", "spread", "open")}}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "chars"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
