"""L4/L5: lipid synthesis, surface area and volume growth."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .genome_map import GeneTable
from .params import Params
from .state import CellState
from .metabolism import atp_saturation


def step(params: Params, genes: GeneTable, state: CellState, dt: float) -> Dict[str, float]:
    """Synthesise lipids and grow the volume with the same ATP-limited factor."""
    membrane_enzymes = float(state.protein[genes.class_mask("membrane")].sum())
    sat = atp_saturation(params, state.atp)
    n_lipid = params["k_lipid_per_enzyme_per_s"] * membrane_enzymes * sat * dt
    state.lipids += n_lipid
    # Volume tracks the enclosed material: keep the sphere-equivalent relation so
    # that SA/V stays in the band the whole-cell model maintains (C0862).
    area_nm2 = state.surface_area_nm2(params)
    radius_nm = (area_nm2 / (4.0 * np.pi)) ** 0.5
    state.volume_um3 = 4.0 / 3.0 * np.pi * (radius_nm / 1000.0) ** 3
    return {
        "n_lipid": n_lipid,
        "atp_lipid": n_lipid * params["atp_per_lipid"],
        "surface_area_um2": state.surface_area_um2(params),
        "sa_v": state.sa_v_ratio(params),
    }
