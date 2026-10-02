#!/usr/bin/env python3
"""Plan 38 J1 (REAL llama-cli, Metal): does the llama-cli fallback still need --reasoning-budget 0 next to -rea off?
Same 6 system/user pairs as runs/plan38/llm/reasoning_modes.json; leak = cognition.leak_guard v2 (+ empty replies).
Writes runs/plan38/llm/cli_budget_ab_<ts>.json"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core.config import load_config
from genai.llm.factory import _resolve_llm_config
from genai.llm.backends.llama_cpp import LlamaCppAdapter
from genai.llm.adapter import LLMRequest
from cognition.leak_guard import check_leak

rows_src = json.loads((ROOT / "runs/plan38/llm/reasoning_modes.json").read_text())["B_rea_off"]["rows"]
cfg = load_config(str(ROOT / "configs/genai/gemma_llama_cpp.yaml"), profile="mac_metal")
llm_cfg = _resolve_llm_config(cfg)
rep = {"kind": "REAL_LLAMA_CLI_AB", "ts": time.strftime("%Y%m%d-%H%M%S"), "variants": {}}
for name, flag in (("rea_off_plus_budget0", True), ("rea_off_only", False)):
    ad = LlamaCppAdapter.from_config({**llm_cfg, "reasoning_budget_flag": flag})
    rows = []
    for r in rows_src:
        t0 = time.time()
        out = ad.generate(LLMRequest(prompt=r["user"], system=r["system"], max_tokens=160, temperature=0.6))
        v = check_leak(out.text)
        rows.append({"user": r["user"], "reply": out.text, "leak": v.leak or not out.text.strip(), "reasons": v.reasons, "s": round(time.time() - t0, 2)})
        print(name, rows[-1]["leak"], rows[-1]["s"], "|", out.text[:70].replace("\n", " "))
    rep["variants"][name] = {"leaks": sum(x["leak"] for x in rows), "n": len(rows),
                             "median_s": sorted(x["s"] for x in rows)[len(rows) // 2], "rows": rows}
out = ROOT / f"runs/plan38/llm/cli_budget_ab_{rep['ts']}.json"
out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
print({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rep["variants"].items()}, out)
