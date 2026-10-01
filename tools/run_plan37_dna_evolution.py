#!/usr/bin/env python3
"""Plan 37 N2/N4: evolve Lenia theta + persona traits with a real Lenia simulation.

Fitness is a hand-designed morphology heuristic on the real FFT Lenia substrate
(research/asal_engine/substrates/lenia.py) — NOT a CLIP/VLM score:
  persistence (alive, not saturated over the last third of the run)
  + structure (spatial std of the final grid)
  - instability (mean |d activity| over the last third).
Artifacts: runs/asal/plan37_dna_lenia_<ts>/ (genome store, lineage, history, keyframes, GIF).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dna import Genome, GenomeStore, evolve, express_persona  # noqa: E402
from research.asal_engine.substrates.lenia import Lenia  # noqa: E402

BOUNDS = [(0.01, 0.95), (0.005, 0.3), (0.001, 0.2), (0.03, 0.4), (0.05, 1.0)]


def simulate(theta, size, steps, seed, keep_frames=False):
    sim = Lenia(size=size)
    sim.reset(theta, seed=seed)
    acts, frames = [], []
    for i in range(steps):
        sim.step()
        acts.append(float(sim.grid.mean()))
        if keep_frames and i % max(1, steps // 12) == 0:
            frames.append(sim.render())
    return sim, np.array(acts), frames


def fitness(theta, size, steps, seeds):
    scores = []
    for seed in seeds:
        sim, acts, _ = simulate(theta, size, steps, seed)
        tail = acts[-max(3, steps // 3):]
        alive = float(np.mean((tail > 0.005) & (tail < 0.6)))
        structure = float(sim.grid.std()) * 4.0
        instability = float(np.mean(np.abs(np.diff(tail)))) * 50.0
        scores.append(alive + structure - instability)
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--eval-seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    out = ROOT / "runs/asal" / f"plan37_dna_lenia_{time.strftime('%Y%m%d-%H%M%S')}"
    store = GenomeStore(out / "genome_store")
    founder = Genome(theta=[0.15, 0.035, 0.1, 0.15, 0.5], substrate="lenia", meta={"origin": "lenia_default_theta"})
    t0 = time.time()
    result = evolve([founder], lambda g: fitness(g.theta, args.size, args.steps, seeds),
                    generations=args.generations, population=args.population, seed=args.seed,
                    theta_sigma=0.03, trait_sigma=0.05, theta_bounds=BOUNDS, on_genome=store.put)
    best = result["best"]
    store.put(best)
    store.set_current(best.genome_id)
    # held-out seed check (not used during selection)
    held_out = {"founder": fitness(founder.theta, args.size, args.steps, [99]),
                "best": fitness(best.theta, args.size, args.steps, [99])}
    frames = {}
    for name, g in (("founder", founder), ("best", best)):
        _, acts, fr = simulate(g.theta, args.size, args.steps, 0, keep_frames=True)
        fr[0].save(out / f"{name}.gif", save_all=True, append_images=fr[1:], duration=120, loop=0)
        fr[-1].resize((256, 256)).save(out / f"{name}_final.png")
        frames[name] = {"final_mean_activity": float(acts[-1])}
    summary = {
        "experiment": "plan37_dna_lenia", "fitness_kind": "morphology heuristic on real Lenia (not CLIP)",
        "args": vars(args), "elapsed_s": round(time.time() - t0, 1),
        "founder": founder.to_dict(), "best": best.to_dict(), "history": result["history"],
        "held_out_seed99": held_out, "frames": frames,
        "best_expression": express_persona(best), "lineage_records": len(store.lineage()),
        "best_ancestry_depth": len(store.ancestry(best.genome_id)),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(out), "founder_fit": result["history"][0]["best_fitness"] if founder.fitness is None else founder.fitness,
                      "best_fit": best.fitness, "held_out": held_out, "lineage": summary["lineage_records"],
                      "ancestry_depth": summary["best_ancestry_depth"], "elapsed_s": summary["elapsed_s"],
                      "best_genome": best.genome_id}, indent=1))


if __name__ == "__main__":
    main()
