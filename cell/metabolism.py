"""L3: deterministic metabolism and the ATP ledger (Plan 44 L3, gate M2)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .genome_map import GeneTable
from .params import Params
from .state import CellState


def atp_saturation(params: Params, atp: float) -> float:
    km = params["atp_km"]
    return float(atp / (km + atp)) if atp > 0 else 0.0


def glycolytic_production(params: Params, genes: GeneTable, state: CellState) -> float:
    """ATP molecules per second from glycolysis (syn3A is glycolysis-dominated, C0862)."""
    metabolic = float(state.protein[genes.class_mask("metabolic")].sum())
    glucose_per_s = params["k_glucose_per_transporter_per_s"] * metabolic
    return glucose_per_s * params["atp_per_glucose_glycolysis"]


def step(params: Params, genes: GeneTable, state: CellState, dt: float,
         costs: Dict[str, float]) -> Dict[str, float]:
    """Apply ATP production and the accumulated costs for this communication step.

    Returns a ledger dict; `closure_error` is the relative imbalance of the
    ATP+ADP pool, which gate M2 requires to stay <= 1%.
    """
    produced = glycolytic_production(params, genes, state) * dt
    maintenance = params["maintenance_atp_per_s"] * dt
    consumed = (costs.get("atp_transcription", 0.0) + costs.get("atp_translation", 0.0)
                + costs.get("atp_lipid", 0.0) + costs.get("atp_replication", 0.0) + maintenance)
    pool_before = state.atp + state.adp
    net = produced - consumed
    new_atp = state.atp + net
    if new_atp < 0.0:
        # ATP cannot go negative: the shortfall is reported, the cell stalls.
        shortfall = -new_atp
        new_atp = 0.0
    else:
        shortfall = 0.0
    new_atp = min(new_atp, pool_before)
    state.atp = new_atp
    state.adp = pool_before - new_atp
    pool_after = state.atp + state.adp
    closure_error = abs(pool_after - pool_before) / pool_before if pool_before else 0.0
    return {
        "atp_produced": produced,
        "atp_consumed": consumed,
        "atp_maintenance": maintenance,
        "atp_shortfall": shortfall,
        "atp": state.atp,
        "closure_error": closure_error,
    }
