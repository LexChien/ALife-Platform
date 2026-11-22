"""L6 (threshold form for M3): division test and symmetric split."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .params import Params
from .state import CellState


def ready_to_divide(params: Params, state: CellState, birth_area_nm2: float) -> bool:
    area = state.surface_area_nm2(params)
    replicated = state.replicated_bp >= params["genome_bp"]
    return bool(area >= 2.0 * birth_area_nm2 and replicated)


def divide(state: CellState, rng: np.random.Generator) -> Tuple[CellState, CellState]:
    """Symmetric split: molecule counts are binomially partitioned, geometry halved."""
    d1 = state.copy()
    d2 = state.copy()
    for field in ("mrna", "protein"):
        parent = getattr(state, field)
        left = rng.binomial(parent.astype(np.int64), 0.5).astype(np.float64)
        setattr(d1, field, left)
        setattr(d2, field, parent - left)
    for d in (d1, d2):
        d.atp = state.atp / 2.0
        d.adp = state.adp / 2.0
        d.lipids = state.lipids / 2.0
        d.volume_um3 = state.volume_um3 / 2.0
        d.replicated_bp = 0.0
        d.generation = state.generation + 1
        d.birth_t_s = state.t_s
    return d1, d2
