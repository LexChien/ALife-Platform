"""L7 (partial, needed for M3): genome replication progress and its ATP cost."""
from __future__ import annotations

from typing import Dict

from .genome_map import GeneTable
from .params import Params
from .state import CellState
from .metabolism import atp_saturation


def step(params: Params, genes: GeneTable, state: CellState, dt: float) -> Dict[str, float]:
    total_bp = params["genome_bp"]
    if state.replicated_bp >= total_bp:
        return {"bp": 0.0, "atp_replication": 0.0, "fraction": 1.0}
    sat = atp_saturation(params, state.atp)
    bp = params["k_replication_bp_per_s"] * sat * dt
    bp = min(bp, total_bp - state.replicated_bp)
    state.replicated_bp += bp
    return {"bp": bp,
            "atp_replication": bp * params["atp_per_bp_replication"],
            "fraction": state.replicated_bp / total_bp}
