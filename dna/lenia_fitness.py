"""Lenia fitness functions for DNA evolution (Plan 37 N4).

v1 (round 1, kept for reproducibility): alive + 4*std(final) - instability.
    Known flaw (log/2026-10-02/dna_genome_evolution_inheritance.md): the std term is
    maximised by space-filling ring textures, not by localized creatures.
v2 (round 2): localized-creature objective. A pattern scores only if it stays alive
    AND occupies a small, connected fraction of the lattice. Space filling is penalised.
All metrics are computed on the real FFT Lenia grid (no mock).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - scipy is installed on both machines
    ndimage = None

ACTIVE_THRESHOLD = 0.1
MIN_AREA = 0.01   # >= 1% of lattice (41 cells at 64x64): exclude single-pixel specks
MAX_AREA = 0.25   # a "creature" must leave most of the world empty
FILL_AREA = 0.30  # above this the pattern is treated as space filling


def morphology(grid: np.ndarray) -> Dict[str, float]:
    """Morphology of one Lenia state: active area, components, largest-component mass share."""
    g = np.asarray(grid, dtype=float)
    active = g > ACTIVE_THRESHOLD
    area = float(active.mean())
    if not active.any():
        return {"area": 0.0, "components": 0, "largest_share": 0.0, "contrast": 0.0, "mean": float(g.mean())}
    if ndimage is not None:
        labels, n = ndimage.label(active)
        masses = np.asarray(ndimage.sum(g, labels, range(1, n + 1)), dtype=float)
        # torus wrap: merge components that touch opposite edges
        merged = _merge_wrapped(labels, n, masses)
        share = float(merged.max() / merged.sum()) if merged.sum() > 0 else 0.0
        comps = int(len(merged))
    else:  # fallback: treat all active cells as one component
        share, comps = 1.0, 1
    contrast = float(g[active].std())
    return {"area": area, "components": comps, "largest_share": share, "contrast": contrast, "mean": float(g.mean())}


def _merge_wrapped(labels: np.ndarray, n: int, masses: np.ndarray) -> np.ndarray:
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_edge, b_edge in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
        for a, b in zip(a_edge, b_edge):
            if a and b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
    groups: Dict[int, float] = {}
    for lab in range(1, n + 1):
        r = find(lab)
        groups[r] = groups.get(r, 0.0) + float(masses[lab - 1])
    return np.array(list(groups.values()), dtype=float)


def fitness_v1(final_grid: np.ndarray, acts: Sequence[float]) -> float:
    acts = np.asarray(acts, dtype=float)
    tail = acts[-max(3, len(acts) // 3):]
    alive = float(np.mean((tail > 0.005) & (tail < 0.6)))
    structure = float(np.asarray(final_grid).std()) * 4.0
    instability = float(np.mean(np.abs(np.diff(tail)))) * 50.0
    return alive + structure - instability


def localization(m: Dict[str, float]) -> float:
    """1 for a single compact blob in [MIN_AREA, MAX_AREA]; 0 for empty / speck / space filling."""
    a = m["area"]
    if a < MIN_AREA or a > MAX_AREA:
        return 0.0
    emptiness = 1.0 - a / MAX_AREA
    return float(m["largest_share"] * (0.5 + 0.5 * emptiness))


MIN_TRANSFORM = 0.3  # final state must differ from the initial noise patch (anti "frozen patch")


def transformation(initial_grid: np.ndarray, final_grid: np.ndarray) -> float:
    """1 - Pearson correlation(initial, final). ~0 means the initial patch was merely frozen."""
    a = np.asarray(initial_grid, dtype=float).ravel()
    b = np.asarray(final_grid, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(1.0 - np.corrcoef(a, b)[0, 1])


def fitness_v2(final_grid: np.ndarray, tail_areas: Sequence[float], initial_grid=None) -> Dict[str, float]:
    """Localized-creature fitness. Returns components so reports can show *why* a genome scored.

    Round-2 finding: with tiny dt the initial disc stays frozen and looks "localized".
    If initial_grid is given, the score is scaled by min(1, transformation/MIN_TRANSFORM).
    """
    m = morphology(final_grid)
    transform = transformation(initial_grid, final_grid) if initial_grid is not None else 1.0
    dyn = min(1.0, transform / MIN_TRANSFORM)
    tail = np.asarray(tail_areas, dtype=float)
    alive = float(np.mean((tail >= MIN_AREA) & (tail <= MAX_AREA))) if len(tail) else 0.0
    if len(tail) > 1 and tail.mean() > 0:
        rel_change = float(np.mean(np.abs(np.diff(tail))) / tail.mean())
    else:
        rel_change = 1.0
    stability = max(0.0, 1.0 - 10.0 * rel_change)
    loc = localization(m)
    fill_penalty = 4.0 * max(0.0, m["area"] - FILL_AREA)
    score = dyn * alive * stability * (1.0 + 2.0 * loc + 2.0 * min(m["contrast"], 0.5)) - fill_penalty
    return {"score": float(score), "alive": alive, "stability": stability, "localization": loc,
            "transformation": transform, "dynamics": dyn,
            "fill_penalty": fill_penalty, **{f"m_{k}": v for k, v in m.items()}}


def center_of_mass(grid: np.ndarray):
    """Torus-aware centre of mass (circular mean per axis), in cells."""
    g = np.asarray(grid, dtype=float)
    n0, n1 = g.shape
    total = g.sum()
    if total <= 0:
        return None
    out = []
    for axis, n in ((0, n0), (1, n1)):
        w = g.sum(axis=1 - axis)
        ang = 2 * np.pi * np.arange(n) / n
        c = np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())
        out.append((c % (2 * np.pi)) * n / (2 * np.pi))
    return tuple(out)


def com_travel(coms, size: int) -> float:
    """Total torus distance travelled by the centre of mass (cells)."""
    pts = [c for c in coms if c is not None]
    dist = 0.0
    for a, b in zip(pts, pts[1:]):
        d = [min(abs(x - y), size - abs(x - y)) for x, y in zip(a, b)]
        dist += float(np.hypot(*d))
    return dist
