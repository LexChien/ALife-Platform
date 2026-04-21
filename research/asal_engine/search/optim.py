import numpy as np

def mutate(theta, sigma=0.03, bounds=None):
    g = np.asarray(theta, dtype=float) + np.random.randn(*np.asarray(theta).shape) * sigma
    if bounds is not None:
        low, high = bounds
        g = np.clip(g, low, high)
    return g

def evo_search(init_thetas, evaluate_fn, iters=6, pop=8, keep=8, sigma=0.03, bounds=None):
    pool = list(init_thetas)
    scores = [evaluate_fn(t) for t in pool]
    for gen in range(iters):
        children = [mutate(pool[np.random.randint(len(pool))], sigma, bounds) for _ in range(pop)]
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
