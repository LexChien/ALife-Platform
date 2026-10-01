#!/usr/bin/env python3
"""Plan 37 N2/N4: evolve Lenia theta + persona traits with a real Lenia simulation.

Fitness runs on the real FFT Lenia substrate (research/asal_engine/substrates/lenia.py):
  --fitness v1 (round 1): alive + 4*std(final) - instability. Gamed by space-filling rings.
  --fitness v2 (round 2, default): localized-creature objective (dna/lenia_fitness.py):
     alive * stability * (1 + 2*localization + 2*contrast) - fill_penalty
  --clip-weight W (optional, needs torch + open_clip; Mac): adds
     W * [cos(img, POS) - cos(img, NEG)] with real OpenCLIP ViT-B-32 laion2b.
  --clip-eval: score founder/best with OpenCLIP afterwards (independent check, not selection).
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
from dna.lenia_fitness import center_of_mass, com_travel, fitness_v1, fitness_v2, morphology  # noqa: E402
from research.asal_engine.substrates.lenia import Lenia  # noqa: E402

BOUNDS = [(0.01, 0.95), (0.005, 0.3), (0.001, 0.2), (0.03, 0.4), (0.05, 1.0)]
# v2: dt >= 0.05 so 300 steps cover >= 15 Lenia time units (round-2 frozen-patch finding)
BOUNDS_V2 = [(0.01, 0.95), (0.005, 0.3), (0.05, 0.2), (0.03, 0.4), (0.05, 1.0)]


CLIP_POS = "a single isolated glowing microorganism on a black background"
CLIP_NEG = "a repeating texture pattern covering the whole image"
_CLIP = {}


def clip_model():
    if "m" not in _CLIP:
        from foundation_models.openclip_adapter import OpenCLIPAdapter
        m = OpenCLIPAdapter(device="cpu")
        _CLIP["m"] = m
        _CLIP["pos"] = m.txt_embed(CLIP_POS)
        _CLIP["neg"] = m.txt_embed(CLIP_NEG)
    return _CLIP


def clip_diff(img):
    c = clip_model()
    e = c["m"].img_embed(img.resize((224, 224)))
    return float(e @ c["pos"] - e @ c["neg"])


def simulate(theta, size, steps, seed, keep_frames=False):
    sim = Lenia(size=size)
    sim.reset(theta, seed=seed)
    sim.initial_grid = sim.grid.copy()
    acts, areas, frames, coms = [], [], [], []
    for i in range(steps):
        sim.step()
        acts.append(float(sim.grid.mean()))
        areas.append(float((sim.grid > 0.1).mean()))
        if keep_frames and i >= steps - steps // 3:
            coms.append(center_of_mass(sim.grid))
        if keep_frames and i % max(1, steps // 12) == 0:
            frames.append(sim.render())
    sim.tail_travel = com_travel(coms, size) if keep_frames else None
    if keep_frames:
        frames.append(sim.render())
    return sim, np.array(acts), np.array(areas), frames


def evaluate(theta, size, steps, seeds, kind="v2", clip_weight=0.0, detail=False):
    scores, details = [], []
    for seed in seeds:
        sim, acts, areas, _ = simulate(theta, size, steps, seed)
        if kind == "v1":
            sc, d = fitness_v1(sim.grid, acts), {}
        else:
            d = fitness_v2(sim.grid, areas[-max(3, steps // 3):], initial_grid=sim.initial_grid)
            sc = d["score"]
        if clip_weight:
            cd = clip_diff(sim.render())
            d["clip_diff"] = cd
            sc += clip_weight * cd
        d["seed"] = seed
        scores.append(sc)
        details.append(d)
    return (float(np.mean(scores)), details) if detail else float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fitness", choices=["v1", "v2"], default="v2")
    ap.add_argument("--clip-weight", type=float, default=0.0)
    ap.add_argument("--clip-eval", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--init-random", type=int, default=0,
                    help="extra founders sampled uniformly within theta bounds (seeded by --seed)")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--eval-seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    tag = args.tag or (f"_{args.fitness}" + ("_clip" if args.clip_weight else ""))
    out = ROOT / "runs/asal" / f"plan37_dna_lenia{tag}_{time.strftime('%Y%m%d-%H%M%S')}_s{args.seed}"
    fit = lambda th, sd: evaluate(th, args.size, args.steps, sd, args.fitness, args.clip_weight)  # noqa: E731
    store = GenomeStore(out / "genome_store")
    founder = Genome(theta=[0.15, 0.035, 0.1, 0.15, 0.5], substrate="lenia", meta={"origin": "lenia_default_theta"})
    import random as _random
    bounds = BOUNDS if args.fitness == "v1" else BOUNDS_V2
    rr = _random.Random(10_000 + args.seed)
    founders = [founder] + [Genome(theta=[rr.uniform(lo, hi) for lo, hi in bounds], substrate="lenia",
                                   meta={"origin": "random_init", "init_seed": 10_000 + args.seed})
                            for _ in range(args.init_random)]
    t0 = time.time()
    result = evolve(founders, lambda g: fit(g.theta, seeds),
                    generations=args.generations, population=max(args.population, len(founders)), seed=args.seed,
                    theta_sigma=0.03, trait_sigma=0.05, theta_bounds=BOUNDS if args.fitness == "v1" else BOUNDS_V2, on_genome=store.put)
    best = result["best"]
    store.put(best)
    store.set_current(best.genome_id)
    # held-out seed check (not used during selection)
    held = [99, 123, 777]
    held_out = {}
    for name, g in (("founder", founder), ("best", best)):
        sc, det = evaluate(g.theta, args.size, args.steps, held, args.fitness, args.clip_weight, detail=True)
        held_out[name] = {"score": sc, "per_seed": det}
    frames = {}
    from PIL import Image
    for name, g in (("founder", founder), ("best", best)):
        frames[name] = {}
        for sd in [0] + held:
            sim, acts, areas, fr = simulate(g.theta, args.size, args.steps, sd, keep_frames=True)
            if sd == 0:
                fr[0].save(out / f"{name}.gif", save_all=True, append_images=fr[1:], duration=120, loop=0)
                fr[-1].resize((256, 256), Image.NEAREST).save(out / f"{name}_final.png")
            # keyframe strip: 6 evenly spaced frames, upscaled x2
            idx = np.linspace(0, len(fr) - 1, 6).astype(int)
            strip = Image.new("RGB", (6 * args.size * 2, args.size * 2))
            for j, k in enumerate(idx):
                strip.paste(fr[k].resize((args.size * 2, args.size * 2), Image.NEAREST), (j * args.size * 2, 0))
            strip.save(out / f"{name}_keyframes_seed{sd}.png")
            mm = morphology(sim.grid)
            frames[name][f"seed{sd}"] = {"final_mean_activity": float(acts[-1]), "tail_com_travel_cells": sim.tail_travel, **mm}
    clip_eval = None
    if args.clip_eval:
        clip_eval = {"pos": CLIP_POS, "neg": CLIP_NEG}
        for name, g in (("founder", founder), ("best", best)):
            vals = [clip_diff(simulate(g.theta, args.size, args.steps, sd)[0].render()) for sd in [0] + held]
            clip_eval[name] = {"per_seed": vals, "mean": float(np.mean(vals))}
    summary = {
        "experiment": "plan37_dna_lenia", "fitness_kind": args.fitness + (f"+{args.clip_weight}*openclip_diff" if args.clip_weight else ""),
        "clip_eval": clip_eval,
        "args": vars(args), "elapsed_s": round(time.time() - t0, 1),
        "founder": founder.to_dict(), "best": best.to_dict(), "history": result["history"],
        "held_out_seed99": held_out, "frames": frames,
        "best_expression": express_persona(best), "lineage_records": len(store.lineage()),
        "best_ancestry_depth": len(store.ancestry(best.genome_id)),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(out), "founder_fit": result["history"][0]["best_fitness"] if founder.fitness is None else founder.fitness,
                      "best_fit": best.fitness, "held_out": {k: v["score"] for k, v in held_out.items()}, "lineage": summary["lineage_records"],
                      "ancestry_depth": summary["best_ancestry_depth"], "elapsed_s": summary["elapsed_s"],
                      "best_genome": best.genome_id, "best_theta": [round(x, 4) for x in best.theta],
                      "best_morph_seed0": frames["best"]["seed0"], "clip_eval": clip_eval}, indent=1))


if __name__ == "__main__":
    main()
