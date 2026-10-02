"""Plan 38 hygiene: ban vocabulary tokens of scripts Lex never uses (logit_bias false) on llama-server.

Measured (REAL, runs/plan38/llm/script_ban_ab_20261002-085715.json, DigiClone system prompt, 30 replies):
no ban 3/30 replies contained Bengali 「হলো」 (e.g. 「我 হলো ALife Prototype」, the round-2 stray-Bengali defect);
Bengali ban 0/30, p50 0.249 s vs 0.269 s. The foreign-script reply guard stays as the second layer."""
from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "runs" / "digiclone"
DEFAULT_SCRIPTS = ("BENGALI", "DEVANAGARI")


def _script_of(token: str) -> str | None:
    for ch in token:
        o = ord(ch)
        if o < 0x0900 or o > 0x0DFF and not (0x0E00 <= o <= 0x0EFF):
            continue
        try:
            return unicodedata.name(ch).split()[0]
        except ValueError:
            return None
    return None


def scan_gguf(model_path: str | Path, scripts=DEFAULT_SCRIPTS) -> list[int]:
    sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(model_path))
    f = r.fields["tokenizer.ggml.tokens"]
    want = set(scripts)
    ids = []
    for i, k in enumerate(f.data):
        tok = bytes(f.parts[k]).decode("utf-8", "replace")
        if _script_of(tok) in want:
            ids.append(i)
    return ids


def load_ban_ids(model_path: str | Path, scripts=DEFAULT_SCRIPTS, cache_dir: Path = CACHE_DIR) -> list[int]:
    p = Path(model_path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{','.join(sorted(scripts))}".encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"script_ban_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())["ids"]
    ids = scan_gguf(p, scripts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": str(p), "scripts": sorted(scripts), "n": len(ids), "ids": ids}))
    return ids
