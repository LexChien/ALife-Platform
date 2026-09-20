import numpy as np


def mutate(theta, sigma=0.03, bounds=None, rng=None):
    if rng is None:
        rng = np.random
    g = np.asarray(theta, dtype=float) + rng.randn(*np.asarray(theta).shape) * sigma
    if bounds is not None:
        low, high = bounds
        g = np.clip(g, low, high)
    return g


def evo_search(init_thetas, evaluate_fn, iters=6, pop=8, keep=8, sigma=0.03, bounds=None, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    pool = list(init_thetas)
    scores = [evaluate_fn(t) for t in pool]
    for gen in range(iters):
        children = [mutate(pool[rng.randint(len(pool))], sigma, bounds, rng=rng) for _ in range(pop)]
        child_scores = [evaluate_fn(c) for c in children]
        pool += children
        scores += child_scores
        order = np.argsort(scores)[::-1][:keep]
        pool = [pool[i] for i in order]
        scores = [scores[i] for i in order]
        if (gen + 1) % 2 == 0:
            print(f"[asal/evo] gen={gen+1} best={max(scores):.4f}")
    best_idx = int(np.argmax(scores))
    return pool, scores, pool[best_idx], scores[best_idx]
