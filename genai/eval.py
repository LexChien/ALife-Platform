import json
from pathlib import Path

from core.config import load_config
from genai.multimodal.engine import GenAIEngine


def load_eval_config(path: str, profile: str | None = None) -> dict:
    return load_config(path, profile=profile)


def run_naive_baseline(case: dict) -> dict:
    prompt = case["prompt"]
    context = case.get("context")
    text = prompt if context is None else f"{prompt} | {context}"
    return {
        "output": {"text": text, "audio": None, "prompt": prompt},
        "summary": {
            "mode": "genai_naive_baseline",
            "prompt": prompt,
            "has_context": context is not None,
            "text_length": len(text),
            "has_audio": False,
            "has_image": False,
        },
    }


def score_case(case: dict, result: dict, method: str) -> dict:
    output = result["output"]
    summary = result["summary"]
    text = output["text"]
    prompt = case["prompt"]
    context = case.get("context")
    expected_prefix = case.get("expected_prefix", "[DummyLLM]")
    must_include = case.get("must_include", [])

    checks = {
        "expected_prefix": text.startswith(expected_prefix) if expected_prefix else True,
        "prompt_reflected": prompt in text,
        "context_reflected": context in text if context else not summary["has_context"] or context is None,
        "required_substrings": all(token in text for token in must_include),
        "has_audio": bool(summary["has_audio"]),
        "has_image": bool(summary["has_image"]),
    }
    score = sum(1.0 for ok in checks.values() if ok) / len(checks)
    return {
        "id": case["id"],
        "method": method,
        "prompt": prompt,
        "context": context,
        "text": text,
        "llm": output.get("llm"),
        "checks": checks,
        "score": score,
        "pass": all(checks.values()),
    }


def summarize_comparison(genai_results: list[dict], baseline_results: list[dict]) -> dict:
    genai_scores = [row["score"] for row in genai_results]
    baseline_scores = [row["score"] for row in baseline_results]
    improvements = [a - b for a, b in zip(genai_scores, baseline_scores)]
    return {
        "num_cases": len(genai_results),
        "genai_mean_score": sum(genai_scores) / max(len(genai_scores), 1),
        "baseline_mean_score": sum(baseline_scores) / max(len(baseline_scores), 1),
        "genai_pass_rate": sum(1 for row in genai_results if row["pass"]) / max(len(genai_results), 1),
        "baseline_pass_rate": sum(1 for row in baseline_results if row["pass"]) / max(len(baseline_results), 1),
        "mean_improvement": sum(improvements) / max(len(improvements), 1),
        "win_rate": sum(1 for value in improvements if value > 0.0) / max(len(improvements), 1),
        "non_negative_rate": sum(1 for value in improvements if value >= 0.0) / max(len(improvements), 1),
    }


def render_markdown(config_path: str, summary: dict, genai_results: list[dict], baseline_results: list[dict]) -> str:
    lines = [
        "# GenAI Validation Report",
        "",
        "## Hypothesis",
        "",
        "The current GenAI baseline preserves prompt/context formatting and emits multimodal artifacts more consistently than a naive text-only baseline.",
        "",
        "## Setup",
        "",
        f"- Config: `{config_path}`",
        f"- Cases: `{summary['num_cases']}`",
        "- Metrics: expected prefix, prompt reflection, context reflection, required substrings, audio presence, image presence",
        "- Comparator: naive baseline without multimodal artifact generation",
        "",
        "## Results",
        "",
        "| Case | GenAI Score | Baseline Score | Improvement | GenAI Pass | Baseline Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for genai_row, base_row in zip(genai_results, baseline_results):
        lines.append(
            f"| {genai_row['id']} | {genai_row['score']:.3f} | {base_row['score']:.3f} | {genai_row['score'] - base_row['score']:+.3f} | {'yes' if genai_row['pass'] else 'no'} | {'yes' if base_row['pass'] else 'no'} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        f"- GenAI mean score: `{summary['genai_mean_score']:.3f}`",
        f"- Baseline mean score: `{summary['baseline_mean_score']:.3f}`",
        f"- Mean improvement: `{summary['mean_improvement']:+.3f}`",
        f"- GenAI pass rate: `{summary['genai_pass_rate']:.2%}`",
        f"- Baseline pass rate: `{summary['baseline_pass_rate']:.2%}`",
        f"- Win rate: `{summary['win_rate']:.2%}`",
        f"- Non-negative rate: `{summary['non_negative_rate']:.2%}`",
        "",
        "## Interpretation",
        "",
        "This is still a baseline validation for formatting and artifact existence rather than generation quality. It is sufficient to close the minimal evaluation-layer gap for the GenAI path.",
        "",
    ]
    return "\n".join(lines)


def run_genai_eval(config_path: str, outdir: str, profile: str | None = None) -> dict:
    cfg = load_eval_config(config_path, profile=profile)
    cases = cfg["cases"]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    genai_results = []
    baseline_results = []
    per_case_outputs = []
    for case in cases:
        case_dir = out / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        engine_cfg = {
            "prompt": case["prompt"],
            "context": case.get("context"),
        }
        if "system" in cfg:
            engine_cfg["system"] = cfg["system"]
        if "llm" in cfg:
            engine_cfg["llm"] = cfg["llm"]
        genai_result = GenAIEngine(engine_cfg, case_dir).run()
        baseline_result = run_naive_baseline(case)
        genai_results.append(score_case(case, genai_result, "genai"))
        baseline_results.append(score_case(case, baseline_result, "naive_baseline"))
        per_case_outputs.append(
            {
                "id": case["id"],
                "genai_output": genai_result["output"],
                "genai_summary": genai_result["summary"],
                "baseline_output": baseline_result["output"],
                "baseline_summary": baseline_result["summary"],
            }
        )

    summary = summarize_comparison(genai_results, baseline_results)
    (out / "genai_results.json").write_text(json.dumps(genai_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "baseline_results.json").write_text(json.dumps(baseline_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "outputs.json").write_text(json.dumps(per_case_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(config_path, summary, genai_results, baseline_results), encoding="utf-8")
    return summary
