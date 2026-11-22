"""L6 (threshold form for M3): division test and symmetric split."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .params import Params
from .state import CellState, founder_lipids


def adder_area_nm2(params: Params) -> float:
    """Membrane area added per cycle before division ("adder", added 2026-11-23).

    ZH：舊規則「面積達到自己出生面積的 2 倍」沒有大小恆定的回復力，每次分裂多出的一步脂質會被
    子代繼承而逐代累積（dt=10 s 時出生面積每代約 +0.027%，M6 祖先族群倍增時間 30 代內 6,315→6,383 s）。
    改用 adder：每週期加上固定面積 Δ 後分裂（Taheri-Araghi 2015 Curr Biol，PMC4323405：
    "maintain cell-size homeostasis by adding a constant size Δ"）。Δ = 創始細胞面積（引用的 400 nm 直徑），
    所以穩態新生大小 = 引用大小。此規則由大腸桿菌/枯草桿菌移植到 syn3A：ASSUMED-TRANSFER。
    EN: The old rule "divide when area reaches twice the own birth area" has no restoring force on
    size: the one-step lipid overshoot at division is inherited and accumulates (at dt=10 s birth area
    grew ~0.027% per generation; M6 ancestral doubling drifted 6,315->6,383 s in 30 generations).
    Adder: divide after adding a fixed area Δ per cycle (Taheri-Araghi 2015 Curr Biol, PMC4323405:
    "maintain cell-size homeostasis by adding a constant size Δ"). Δ = founder area (cited 400 nm
    diameter), so the steady newborn size equals the cited size. Transferred from E. coli / B. subtilis
    to syn3A: ASSUMED-TRANSFER. Author: Grok Bot ALife Project - Andreeva, 2026-11-23.
    """
    return founder_lipids(params) * params["area_per_lipid_nm2"] / 2.0


def ready_to_divide(params: Params, state: CellState, birth_area_nm2: float) -> bool:
    area = state.surface_area_nm2(params)
    replicated = state.replicated_bp >= params["genome_bp"]
    return bool(area >= birth_area_nm2 + adder_area_nm2(params) and replicated)


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
