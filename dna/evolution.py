from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

from dna.genome import Genome, TRAIT_SPEC, _clip01

FitnessFn = Callable[[Genome], float]


def mutate(genome: Genome, rng: random.Random, theta_sigma: float = 0.1, trait_sigma: float = 0.05,
           theta_bounds: Optional[Sequence[tuple]] = None) -> Genome:
    theta = []
    for i, x in enumerate(genome.theta):
        y = x + rng.gauss(0.0, theta_sigma)
        if theta_bounds is not None:
            lo, hi = theta_bounds[i]
            y = min(max(y, lo), hi)
        theta.append(y)
    traits = {k: _clip01(v + rng.gauss(0.0, trait_sigma)) for k, v in genome.traits.items()}
    return Genome(theta=theta, traits=traits, parents=[genome.genome_id], generation=genome.generation + 1,
                  substrate=genome.substrate, meta={"op": "mutate"})


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    if len(a.theta) != len(b.theta) or a.substrate != b.substrate:
        raise ValueError("crossover requires matching substrate and theta dimension")
    theta = [x if rng.random() < 0.5 else y for x, y in zip(a.theta, b.theta)]
    traits = {k: (a.traits[k] if rng.random() < 0.5 else b.traits[k]) for k in TRAIT_SPEC}
    return Genome(theta=theta, traits=traits, parents=[a.genome_id, b.genome_id],
                  generation=max(a.generation, b.generation) + 1, substrate=a.substrate, meta={"op": "crossover"})


def evolve(founders: List[Genome], fitness_fn: FitnessFn, generations: int, population: int = 8,
           elite: int = 2, seed: int = 0, theta_sigma: float = 0.1, trait_sigma: float = 0.05,
           crossover_rate: float = 0.3, theta_bounds: Optional[Sequence[tuple]] = None,
           on_genome: Optional[Callable[[Genome], None]] = None) -> Dict[str, object]:
    """Seeded (mu+lambda)-style GA with elitism. Best fitness is non-decreasing."""
    if not founders:
        raise ValueError("need at least one founder")
    rng = random.Random(seed)

    def score(g: Genome) -> Genome:
        if g.fitness is None:
            g.fitness = float(fitness_fn(g))
        if on_genome is not None:
            on_genome(g)
        return g

    pop = [score(g) for g in founders]
    while len(pop) < population:
        pop.append(score(mutate(rng.choice(founders), rng, theta_sigma, trait_sigma, theta_bounds)))
    history = []
    for gen in range(generations):
        pop.sort(key=lambda g: g.fitness, reverse=True)
        history.append({"generation": gen, "best_fitness": pop[0].fitness, "best_genome": pop[0].genome_id,
                        "mean_fitness": sum(g.fitness for g in pop) / len(pop)})
        elites = pop[:max(1, elite)]
        children: List[Genome] = []
        while len(elites) + len(children) < population:
            if len(pop) > 1 and rng.random() < crossover_rate:
                p1, p2 = rng.sample(pop[: max(2, population // 2)], 2)
                child = mutate(crossover(p1, p2, rng), rng, theta_sigma * 0.5, trait_sigma * 0.5, theta_bounds)
                child.parents = [p1.genome_id, p2.genome_id]
            else:
                child = mutate(rng.choice(pop[: max(2, population // 2)]), rng, theta_sigma, trait_sigma, theta_bounds)
            children.append(score(child))
        pop = elites + children
    pop.sort(key=lambda g: g.fitness, reverse=True)
    history.append({"generation": generations, "best_fitness": pop[0].fitness, "best_genome": pop[0].genome_id,
                    "mean_fitness": sum(g.fitness for g in pop) / len(pop)})
    return {"best": pop[0], "population": pop, "history": history}
