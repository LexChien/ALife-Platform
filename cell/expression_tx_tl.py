"""L2: stochastic transcription / translation (CME, tau-leaping).

ZH: 以 Poisson tau-leaping 近似化學主方程；mRNA 低拷貝數必須隨機處理（C5382、C0862）。
EN: Poisson tau-leaping approximation of the CME for low-copy mRNA (C5382, C0862).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .genome_map import GeneTable
from .params import Params
from .state import CellState


def rates(params: Params, genes: GeneTable, state: CellState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (transcription, degradation, translation) propensities per gene, 1/s."""
    k_tx = params["tx_elongation_nt_per_s"] / genes.length_bp * genes.copies
    k_deg = math.log(2.0) / params["mrna_half_life_s"] * state.mrna
    # Translation: one ribosome finishes a protein in L_aa / v_aa seconds.
    aa = genes.aa_length()
    per_ribosome = params["tl_elongation_aa_per_s"] / aa
    ribosomes = params["n_ribosomes"] * params["ribosome_active_fraction"]
    total_mrna = float(state.mrna.sum())
    if total_mrna <= 0:
        share = np.zeros_like(state.mrna)
    else:
        share = state.mrna / total_mrna
    k_tl = per_ribosome * ribosomes * share
    return k_tx, k_deg, k_tl


def step(params: Params, genes: GeneTable, state: CellState, dt: float,
         rng: np.random.Generator, atp_factor: float = 1.0) -> dict:
    """Advance transcription/translation by dt seconds. Returns ATP costs."""
    k_tx, k_deg, k_tl = rates(params, genes, state)
    n_tx = rng.poisson(np.maximum(k_tx * atp_factor * dt, 0.0))
    n_deg = rng.poisson(np.maximum(k_deg * dt, 0.0))
    n_deg = np.minimum(n_deg, state.mrna + n_tx)
    n_tl = rng.poisson(np.maximum(k_tl * atp_factor * dt, 0.0))
    state.mrna = state.mrna + n_tx - n_deg
    state.protein = state.protein + n_tl
    nt_polymerised = float(np.sum(n_tx * genes.length_bp))
    aa_polymerised = float(np.sum(n_tl * genes.aa_length()))
    return {
        "n_transcripts": float(n_tx.sum()),
        "n_proteins": float(n_tl.sum()),
        "atp_transcription": nt_polymerised * params["atp_per_nt_transcription"],
        "atp_translation": aa_polymerised * params["atp_per_aa_translation"],
    }


def analytic_mrna_steady_state(params: Params, genes: GeneTable) -> np.ndarray:
    """Mean mRNA per gene at steady state: k_tx / k_deg (birth-death process)."""
    k_tx = params["tx_elongation_nt_per_s"] / genes.length_bp * genes.copies
    k_deg = math.log(2.0) / params["mrna_half_life_s"]
    return k_tx / k_deg
