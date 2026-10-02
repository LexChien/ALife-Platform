#!/usr/bin/env python3
"""Plan 38 hygiene: find GGUF vocab tokens containing scripts other than Latin/CJK/kana/punctuation (for logit_bias ban)."""
import collections, json, sys, time, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
from gguf import GGUFReader

def allowed(o):
    return (o <= 0x2FF or 0x2000 <= o <= 0x206F or 0x2E80 <= o <= 0x9FFF or 0x3000 <= o <= 0x30FF or 0xF900 <= o <= 0xFAFF
            or 0xFE30 <= o <= 0xFE4F or 0xFF00 <= o <= 0xFFEF or 0x20000 <= o <= 0x2FFFF or 0x2100 <= o <= 0x2BFF
            or 0x1F000 <= o <= 0x1FAFF or o in (0xFFFD, 0x2581))

t = time.time()
r = GGUFReader(str(ROOT / "models/gemma/gemma.gguf"))
f = r.fields["tokenizer.ggml.tokens"]
toks = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
cnt, ban = collections.Counter(), {}
for i, s in enumerate(toks):
    for ch in s:
        if not allowed(ord(ch)):
            try:
                sc = unicodedata.name(ch).split()[0]
            except ValueError:
                sc = "?"
            cnt[sc] += 1
            ban.setdefault(sc, []).append(i)
            break
out = {"n_vocab": len(toks), "read_s": round(time.time() - t, 1), "scripts": cnt.most_common(),
       "bengali_examples": [toks[i] for i in ban.get("BENGALI", [])[:20]], "ban": ban}
p = ROOT / "runs/plan38/llm/vocab_scripts.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("n_vocab", "read_s", "bengali_examples")}, ensure_ascii=False))
print(cnt.most_common(40))
