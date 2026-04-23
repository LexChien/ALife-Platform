from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from foundation_models import foundation_models
from research.asal_engine.scores import supervised_target_score
from research.asal_engine.search.optim import evo_search
from research.asal_engine.substrates.reaction_diffusion import ReactionDiffusion

def make_evaluator(config: dict):
    fm = foundation_models.create(
        config["foundation_model"]["name"],
        **config["foundation_model"].get("params", {}),
    )
    morphology_cfg = config.get("morphology_judge")
    morphology_fm = None
    morphology_txt_emb = None
    morphology_weight = 0.0
    if morphology_cfg and morphology_cfg.get("enabled", False):
        morphology_fm = foundation_models.create(
            morphology_cfg["name"],
            **morphology_cfg.get("params", {}),
        )
        morphology_txt_emb = morphology_fm.txt_embed(config["prompt"])
        morphology_weight = float(morphology_cfg.get("weight", 0.0))
    substrate = ReactionDiffusion(**config["substrate"].get("params", {}))
    txt_emb = fm.txt_embed(config["prompt"])
    runtime = config["runtime"]

    def evaluate(theta):
        substrate.reset(theta)
        for _ in range(runtime["steps"]):
            substrate.step(runtime["substeps"])
        img = substrate.render()
        img_emb = fm.img_embed(img)
        score = supervised_target_score([img_emb], txt_emb)
        if morphology_fm is not None and morphology_weight > 0.0:
            morph_score = supervised_target_score([morphology_fm.img_embed(img)], morphology_txt_emb)
            score = (1.0 - morphology_weight) * score + morphology_weight * morph_score
        return score

    return evaluate


def run_evolution(config: dict, seed: int) -> dict:
    search = config["search"]
    low = np.array(search["theta_low"], dtype=float)
    high = np.array(search["theta_high"], dtype=float)
    rs = np.random.RandomState(seed)
    np.random.seed(seed)
    init = [rs.uniform(low, high) for _ in range(search["keep"])]
    evaluate = make_evaluator(config)
    pool, scores, best, best_score = evo_search(
        init,
        evaluate,
        iters=search["iters"],
        pop=search["pop"],
        keep=search["keep"],
        sigma=search["sigma"],
        bounds=(low, high),
    )
    return {
        "method": "evolution",
        "seed": seed,
        "best_score": float(best_score),
        "best_theta": np.asarray(best).tolist(),
        "final_pool_best": float(max(scores)),
        "evaluations": int(search["keep"] + search["iters"] * search["pop"]),
    }


def run_random_search(config: dict, seed: int) -> dict:
    search = config["search"]
    low = np.array(search["theta_low"], dtype=float)
    high = np.array(search["theta_high"], dtype=float)
    budget = int(search["keep"] + search["iters"] * search["pop"])
    rs = np.random.RandomState(seed)
    evaluate = make_evaluator(config)
    best_score = None
    best_theta = None
    for _ in range(budget):
        theta = rs.uniform(low, high)
        score = evaluate(theta)
        if best_score is None or score > best_score:
            best_score = score
            best_theta = theta
    return {
        "method": "random",
        "seed": seed,
        "best_score": float(best_score),
        "best_theta": np.asarray(best_theta).tolist(),
        "evaluations": budget,
    }


def summarize(results: list[dict]) -> dict:
    evo_scores = np.array([r["best_score"] for r in results if r["method"] == "evolution"], dtype=float)
    rnd_scores = np.array([r["best_score"] for r in results if r["method"] == "random"], dtype=float)
    diffs = evo_scores - rnd_scores
    rs = np.random.RandomState(0)
    if len(diffs) > 0:
        bootstrap_means = []
        for _ in range(2000):
            sample = rs.choice(diffs, size=len(diffs), replace=True)
            bootstrap_means.append(float(sample.mean()))
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    else:
        ci_low, ci_high = 0.0, 0.0
    return {
        "num_trials": int(len(evo_scores)),
        "evolution_mean_best": float(evo_scores.mean()),
        "random_mean_best": float(rnd_scores.mean()),
        "evolution_median_best": float(np.median(evo_scores)),
        "random_median_best": float(np.median(rnd_scores)),
        "mean_improvement": float(diffs.mean()),
        "median_improvement": float(np.median(diffs)),
        "win_rate": float(np.mean(diffs > 0.0)),
        "non_negative_rate": float(np.mean(diffs >= 0.0)),
        "improvement_std": float(diffs.std(ddof=0)),
        "mean_improvement_ci95": [float(ci_low), float(ci_high)],
    }


def render_markdown(config_path: str, seeds: list[int], results: list[dict], summary: dict) -> str:
    by_seed = {}
    for row in results:
        by_seed.setdefault(row["seed"], {})[row["method"]] = row

    lines = [
        "# ASAL Target Search Validation Report",
        "",
        "## Hypothesis",
        "",
        "Under the same evaluation budget, evolutionary search achieves higher best target similarity than random search on the current reaction-diffusion substrate.",
        "",
        "## Setup",
        "",
        f"- Config: `{config_path}`",
        f"- Seeds: `{', '.join(str(s) for s in seeds)}`",
        f"- Trials: `{summary['num_trials']}`",
        "- Metric: `best_score` from `supervised_target_score`",
        "- Baseline: random search with matched evaluation budget",
        "",
        "## Results",
        "",
        "| Seed | Evolution Best | Random Best | Improvement |",
        "|---|---:|---:|---:|",
    ]

    for seed in seeds:
        evo = by_seed[seed]["evolution"]["best_score"]
        rnd = by_seed[seed]["random"]["best_score"]
        lines.append(f"| {seed} | {evo:.6f} | {rnd:.6f} | {evo - rnd:+.6f} |")

    lines += [
        "",
        "## Aggregate",
        "",
        f"- Evolution mean best: `{summary['evolution_mean_best']:.6f}`",
        f"- Random mean best: `{summary['random_mean_best']:.6f}`",
        f"- Mean improvement: `{summary['mean_improvement']:+.6f}`",
        f"- Improvement std: `{summary['improvement_std']:.6f}`",
        f"- Median improvement: `{summary['median_improvement']:+.6f}`",
        f"- 95% bootstrap CI of mean improvement: `[{summary['mean_improvement_ci95'][0]:+.6f}, {summary['mean_improvement_ci95'][1]:+.6f}]`",
        f"- Win rate: `{summary['win_rate']:.2%}`",
        f"- Non-negative rate: `{summary['non_negative_rate']:.2%}`",
        "",
        "## Interpretation",
        "",
        "This validates only the current minimal pipeline and reward setup. It does not yet validate a stronger biological-life theory, because the reward model is still a stub foundation model and the substrate/search stack is still a reduced baseline.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asal/target_cell.yaml")
    ap.add_argument("--profile")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--outdir", default="runs/asal_target_validation")
    args = ap.parse_args()

    config = load_config(args.config, profile=args.profile)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        results.append(run_evolution(config, seed))
        results.append(run_random_search(config, seed))

    summary = summarize(results)
    report = render_markdown(args.config, seeds, results, summary)

    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved report to {outdir}")


if __name__ == "__main__":
    main()
