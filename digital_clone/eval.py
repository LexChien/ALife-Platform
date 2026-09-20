"""Labeled behavioral evaluation independent of persona wrappers and retrieval hits."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from core.config import load_config
from digital_clone.engine import DigitalCloneEngine
from evaluation.heuristics import make_result


def load_eval_config(path: str, profile: str | None = None) -> dict:
    return load_config(path, profile=profile)


def score_case(persona: dict, case: dict, output: dict, evaluation: dict) -> dict:
    # If a caller also supplies a display wrapper, it must never supply the answer.
    text = output.get("generated_response", output["output"])
    folded = text.casefold()
    retrieved = output.get("retrieved_memories", [])
    groups = case.get("answer_groups", [[item] for item in case.get("must_include", [])])
    required = case.get("must_recall", [])
    forbidden = case.get("must_not_include", [])
    matches = [{"alternatives": group, "matched": [term for term in group if term.casefold() in folded]}
               for group in groups]
    forbidden_matches = [term for term in forbidden if term.casefold() in folded]
    observed_backend = (output.get("llm") or {}).get("backend")
    checks = {
        "generated_answer_nonempty": bool(text.strip()),
        "real_model": observed_backend not in {None, "dummy"} and "[DummyLLM]" not in text,
        "expected_answer": all(item["matched"] for item in matches),
        "forbidden_absent": not forbidden_matches,
        "no_reasoning_leak": not any(marker in text for marker in ("<|channel>thought", "Thinking Process:", "<think>")),
        "no_runtime_sentinel": "[end of text]" not in text,
    }
    if case.get("max_answer_chars"):
        checks["concise_answer"] = len(text) <= case["max_answer_chars"]
    behavior = make_result(checks)
    retrieval_checks = {
        term: any(term.casefold() in memory.casefold() for memory in retrieved)
        for term in required
    }
    forbidden_retrieval = case.get("must_not_retrieve", [])
    for term in forbidden_retrieval:
        retrieval_checks[f"absent:{term}"] = all(term.casefold() not in memory.casefold() for memory in retrieved)
    retrieval = make_result(retrieval_checks) if retrieval_checks else {"checks": {}, "score": None, "pass": True}
    return {
        "id": case["id"], "input": case["input"], "category": case.get("category", "behavior"),
        "output": text, "generated_response": text,
        "consistency": output.get("consistency", {}),
        "consistency_score": output.get("consistency_score", 0.0),
        "retrieved_memories": retrieved, "llm": output.get("llm"),
        "checks": checks, "score": behavior["score"],
        "behavior": behavior, "retrieval": retrieval,
        "pass": behavior["pass"] and retrieval["pass"],
        "reasons": {"answer_matches": matches, "forbidden_matches": forbidden_matches,
                    "missing_retrieval": [term for term, present in retrieval_checks.items() if not present],
                    "failed_checks": [key for key, passed in checks.items() if not passed]},
    }


def summarize(case_results: list[dict]) -> dict:
    count = len(case_results)
    return {
        "num_cases": count,
        "mean_score": sum(row["score"] for row in case_results) / max(count, 1),
        "pass_rate": sum(bool(row["pass"]) for row in case_results) / max(count, 1),
        "all_passed": bool(count) and all(row["pass"] for row in case_results),
        "behavior_pass_rate": sum(row["behavior"]["pass"] for row in case_results) / max(count, 1),
        "retrieval_pass_rate": sum(row["retrieval"]["pass"] for row in case_results) / max(count, 1),
    }


def summarize_comparison(clone_results: list[dict], baseline_results: list[dict]) -> dict:
    clone, baseline = summarize(clone_results), summarize(baseline_results)
    differences = [a["score"] - b["score"] for a, b in zip(clone_results, baseline_results)]
    return {**clone, "clone_mean_score": clone["mean_score"], "baseline_mean_score": baseline["mean_score"],
            "clone_pass_rate": clone["pass_rate"], "baseline_pass_rate": baseline["pass_rate"],
            "mean_improvement": sum(differences) / max(len(differences), 1),
            "win_rate": sum(value > 0 for value in differences) / max(len(differences), 1),
            "non_negative_rate": sum(value >= 0 for value in differences) / max(len(differences), 1)}


def render_markdown(config_path: str, summary: dict, clone_results: list[dict], baseline_results: list[dict]) -> str:
    lines = ["# Digital Clone behavioral validation / Digital Clone 行為驗收", "",
             f"Config / 設定：`{config_path}`", "",
             "Answers are scored only from model-generated text. Retrieval is scored separately; it cannot substitute for a correct answer.",
             "僅對模型實際生成文字評分。檢索單獨評分；取回資料不代表回答正確。", "",
             "| Case / 案例 | Behavior / 行為 | Retrieval / 檢索 | Pass / 通過 |",
             "|---|---:|---|---|"]
    for row in clone_results:
        lines.append(f"| {row['id']} | {row['score']:.3f} | {row['retrieval']['pass']} | {row['pass']} |")
    lines += ["", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
              "This fixed labeled suite is evidence for the tested tasks only; substring checks do not prove unrestricted persona coherence or long-term drift resistance.",
              "固定標記案例僅驗證受測任務；字串規則無法證明無限制人格一致性或長期抗漂移能力。", ""]
    return "\n".join(lines)


def run_naive_baseline(persona: dict, cases: list[dict]) -> list[dict]:
    return [{"method": "naive_baseline", "input": case["input"], "output": case["input"],
             "consistency": {}, "retrieved_memories": []} for case in cases]


def run_clone_eval(config_path: str, outdir: str, profile: str | None = None) -> dict:
    cfg = load_eval_config(config_path, profile=profile)
    persona, cases, evaluation = cfg["persona"], cfg["cases"], cfg.get("evaluation", {})
    engine_cfg = deepcopy(cfg)
    engine_cfg.setdefault("memory", {}).setdefault("persist_directory", str(Path(outdir).resolve() / "memory"))
    engine_cfg["inputs"] = [case["input"] for case in cases]
    engine_cfg.pop("cases", None)
    engine_cfg.pop("evaluation", None)
    clone_run = DigitalCloneEngine(engine_cfg, Path(outdir)).run()
    clone_outputs = clone_run["outputs"]
    clone_results = [score_case(persona, case, output, evaluation) for case, output in zip(cases, clone_outputs)]
    baseline_results = [score_case(persona, case, output, evaluation)
                        for case, output in zip(cases, run_naive_baseline(persona, cases))]
    summary = summarize_comparison(clone_results, baseline_results)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in {"clone_outputs": clone_outputs, "results": clone_run,
                          "clone_results": clone_results, "baseline_results": baseline_results,
                          "summary": summary}.items():
        (out / f"{name}.json").write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(config_path, summary, clone_results, baseline_results), encoding="utf-8")
    return summary
