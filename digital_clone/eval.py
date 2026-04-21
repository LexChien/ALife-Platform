from pathlib import Path
import json

import yaml

from digital_clone.engine import DigitalCloneEngine


def load_eval_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def score_case(persona: dict, case: dict, output: dict, evaluation: dict) -> dict:
    text = output["output"]
    required_principles = case.get("expected_principles", persona["principles"][:2])
    required_substrings = case.get("must_include", [])
    forbidden_substrings = case.get("must_not_include", [])
    response_prefix = case.get(
        "expected_response_prefix",
        evaluation.get("expected_response_prefix", "response=我已收到你的問題："),
    )
    consistency_min = float(case.get("consistency_min", evaluation.get("consistency_min", 1.0)))

    checks = {
        "persona_tag": text.startswith(f"[{persona['name']}]"),
        "tone_match": f"tone={persona['tone']}" in text,
        "response_prefix": response_prefix in text,
        "echo_input": case.get("input", "") in text if case.get("expect_echo", True) else True,
        "consistency_threshold": float(output["consistency"]) >= consistency_min,
        "required_principles": all(p in text for p in required_principles),
        "required_substrings": all(s in text for s in required_substrings),
        "forbidden_absent": all(s not in text for s in forbidden_substrings),
    }
    score = sum(1.0 for ok in checks.values() if ok) / len(checks)
    return {
        "id": case["id"],
        "input": case["input"],
        "method": output.get("method", "clone"),
        "output": output["output"],
        "consistency": float(output["consistency"]),
        "checks": checks,
        "score": score,
        "pass": all(checks.values()),
    }


def summarize(case_results: list[dict]) -> dict:
    scores = [row["score"] for row in case_results]
    passes = [row["pass"] for row in case_results]
    return {
        "num_cases": len(case_results),
        "mean_score": sum(scores) / max(len(scores), 1),
        "pass_rate": sum(1 for p in passes if p) / max(len(passes), 1),
        "all_passed": all(passes),
    }


def summarize_comparison(clone_results: list[dict], baseline_results: list[dict]) -> dict:
    clone_scores = [row["score"] for row in clone_results]
    baseline_scores = [row["score"] for row in baseline_results]
    clone_pass = [row["pass"] for row in clone_results]
    baseline_pass = [row["pass"] for row in baseline_results]
    improvements = [a - b for a, b in zip(clone_scores, baseline_scores)]
    return {
        "num_cases": len(clone_results),
        "clone_mean_score": sum(clone_scores) / max(len(clone_scores), 1),
        "baseline_mean_score": sum(baseline_scores) / max(len(baseline_scores), 1),
        "clone_pass_rate": sum(1 for p in clone_pass if p) / max(len(clone_pass), 1),
        "baseline_pass_rate": sum(1 for p in baseline_pass if p) / max(len(baseline_pass), 1),
        "mean_improvement": sum(improvements) / max(len(improvements), 1),
        "win_rate": sum(1 for x in improvements if x > 0.0) / max(len(improvements), 1),
        "non_negative_rate": sum(1 for x in improvements if x >= 0.0) / max(len(improvements), 1),
    }


def render_markdown(config_path: str, summary: dict, clone_results: list[dict], baseline_results: list[dict]) -> str:
    lines = [
        "# Digital Clone Validation Report",
        "",
        "## Hypothesis",
        "",
        "The current Digital Clone baseline preserves persona tag, tone, required principles, response format, and input echo better than a naive echo baseline across a fixed test set.",
        "",
        "## Setup",
        "",
        f"- Config: `{config_path}`",
        f"- Cases: `{summary['num_cases']}`",
        "- Metrics: persona tag, tone match, principle coverage, response prefix, input echo, forbidden absence, consistency threshold",
        "- Comparator: naive echo baseline without persona/tone/principle formatting",
        "",
        "## Results",
        "",
        "| Case | Clone Score | Baseline Score | Improvement | Clone Pass | Baseline Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for clone_row, base_row in zip(clone_results, baseline_results):
        lines.append(
            f"| {clone_row['id']} | {clone_row['score']:.3f} | {base_row['score']:.3f} | {clone_row['score'] - base_row['score']:+.3f} | {'yes' if clone_row['pass'] else 'no'} | {'yes' if base_row['pass'] else 'no'} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- Clone mean score: `{summary['clone_mean_score']:.3f}`",
        f"- Baseline mean score: `{summary['baseline_mean_score']:.3f}`",
        f"- Mean improvement: `{summary['mean_improvement']:+.3f}`",
        f"- Clone pass rate: `{summary['clone_pass_rate']:.2%}`",
        f"- Baseline pass rate: `{summary['baseline_pass_rate']:.2%}`",
        f"- Win rate: `{summary['win_rate']:.2%}`",
        f"- Non-negative rate: `{summary['non_negative_rate']:.2%}`",
        "",
        "## Interpretation",
        "",
        "This provides stronger evidence than self-checking alone because the current clone output is compared against a weaker baseline. It still does not prove long-term memory quality, drift resistance, or open-ended identity coherence because the memory and consistency modules remain stubs.",
        "",
    ]
    return "\n".join(lines)


def run_naive_baseline(persona: dict, cases: list[dict]) -> list[dict]:
    outputs = []
    for case in cases:
        outputs.append(
            {
                "method": "naive_baseline",
                "input": case["input"],
                "output": case["input"],
                "consistency": 0.0,
            }
        )
    return outputs


def run_clone_eval(config_path: str, outdir: str) -> dict:
    cfg = load_eval_config(config_path)
    persona = cfg["persona"]
    cases = cfg["cases"]
    evaluation = cfg.get("evaluation", {})
    engine_cfg = {
        "persona": persona,
        "inputs": [case["input"] for case in cases],
    }
    clone_outputs = DigitalCloneEngine(engine_cfg, Path(outdir)).run()
    baseline_outputs = run_naive_baseline(persona, cases)
    clone_results = [
        score_case(persona, case, output, evaluation)
        for case, output in zip(cases, clone_outputs)
    ]
    baseline_results = [
        score_case(persona, case, output, evaluation)
        for case, output in zip(cases, baseline_outputs)
    ]
    summary = summarize_comparison(clone_results, baseline_results)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clone_results.json").write_text(json.dumps(clone_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "baseline_results.json").write_text(json.dumps(baseline_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(config_path, summary, clone_results, baseline_results), encoding="utf-8")
    return summary
