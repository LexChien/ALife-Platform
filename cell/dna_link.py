"""Link to the existing DNA module: express a Genome into cell model parameters.

ZH: 不改 dna/genome.py 既有語意；以 substrate="cell" 的 theta 當成速率縮放因子。
EN: Does not change dna/genome.py semantics; a substrate="cell" genome's theta is read
as multiplicative scalings of the free cell rates.
"""
from __future__ import annotations

from typing import Dict, List

from dna.genome import Genome

# theta -> parameter mapping (order is part of the genome contract)
THETA_KEYS: List[str] = [
    "k_lipid_per_enzyme_per_s",
    "k_glucose_per_transporter_per_s",
    "k_replication_bp_per_s",
    "maintenance_atp_per_s",
]
SCALE_RANGE = (0.5, 2.0)  # theta in [0,1] maps linearly onto this multiplier range


def express_cell(genome: Genome, base: Dict[str, float]) -> Dict[str, float]:
    """Return parameter overrides for a genome (theta entries beyond the mapping are ignored)."""
    lo, hi = SCALE_RANGE
    out: Dict[str, float] = {}
    for i, key in enumerate(THETA_KEYS):
        if i >= len(genome.theta):
            break
        x = min(max(float(genome.theta[i]), 0.0), 1.0)
        out[key] = float(base[key]) * (lo + (hi - lo) * x)
    return out


def neutral_theta() -> List[float]:
    """theta that reproduces the base parameters exactly (scale 1.0)."""
    lo, hi = SCALE_RANGE
    x = (1.0 - lo) / (hi - lo)
    return [x] * len(THETA_KEYS)
