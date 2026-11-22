"""L0: shared cell state and geometry (Plan 44 L0/L4)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .genome_map import GeneTable
from .params import Params

NM2_PER_UM2 = 1.0e6


@dataclass
class CellState:
    """One cell. Counts are molecule numbers; geometry is in nm^2 / um^3."""

    t_s: float
    mrna: np.ndarray
    protein: np.ndarray
    atp: float
    adp: float
    lipids: float
    replicated_bp: float
    volume_um3: float
    generation: int = 0
    birth_t_s: float = 0.0
    alive: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- geometry -------------------------------------------------------
    def surface_area_nm2(self, params: Params) -> float:
        return self.lipids * params["area_per_lipid_nm2"] / 2.0

    def surface_area_um2(self, params: Params) -> float:
        return self.surface_area_nm2(params) / NM2_PER_UM2

    def sa_v_ratio(self, params: Params) -> float:
        return self.surface_area_um2(params) / self.volume_um3

    def copy(self) -> "CellState":
        return CellState(t_s=self.t_s, mrna=self.mrna.copy(), protein=self.protein.copy(),
                         atp=self.atp, adp=self.adp, lipids=self.lipids,
                         replicated_bp=self.replicated_bp, volume_um3=self.volume_um3,
                         generation=self.generation, birth_t_s=self.birth_t_s,
                         alive=self.alive, extra=dict(self.extra))


def sphere_volume_um3(radius_nm: float) -> float:
    r_um = radius_nm / 1000.0
    return 4.0 / 3.0 * math.pi * r_um ** 3


def sphere_area_nm2(radius_nm: float) -> float:
    return 4.0 * math.pi * radius_nm ** 2


def founder_lipids(params: Params) -> float:
    """Lipid count of the founder (newborn) cell: two leaflets of a sphere of the cited radius."""
    return 2.0 * sphere_area_nm2(params["cell_radius_nm"]) / params["area_per_lipid_nm2"]


def balanced_protein(params: Params, genes: GeneTable) -> np.ndarray:
    """Per-gene protein count of a newborn cell in balanced growth (DERIVED, added 2026-11-23).

    ZH：核糖體數固定時，每個基因每週期新增的蛋白 = 平均轉譯速率 × 倍增時間；平衡生長下新生細胞
    的蛋白數必須等於這個量，分裂前才會剛好加倍（C0862：多數蛋白每週期變為 1.75–2.25 倍）。
    舊版把每個基因都設成 503（核糖體數，ASSUMED），造成第 0 代比穩態快、之後各代變慢。
    EN: With a fixed ribosome count, the protein added per gene per cycle is the mean translation
    rate times the doubling time; in balanced growth a newborn must hold exactly that amount so that
    it doubles by division (C0862: most proteins end the cycle at 1.75-2.25x). The old version set
    every gene to 503 (the ribosome count, ASSUMED), which made generation 0 faster than the steady
    state and later generations slower. Author: Grok Bot ALife Project - Andreeva, 2026-11-23.
    """
    k_tx = params["tx_elongation_nt_per_s"] / genes.length_bp * genes.copies
    mrna_ss = k_tx / (math.log(2.0) / params["mrna_half_life_s"])
    share = mrna_ss / mrna_ss.sum()
    ribosomes = params["n_ribosomes"] * params["ribosome_active_fraction"]
    rate = params["tl_elongation_aa_per_s"] / genes.aa_length() * ribosomes * share
    return np.round(rate * params["doubling_time_experimental_s"]).astype(np.float64)


def initial_state(params: Params, genes: GeneTable, seed: int = 0) -> CellState:
    """Initial syn3A-like cell.

    Lipid count is derived from the cited 400 nm cell diameter and the ASSUMED
    area per lipid; initial mRNA is the transcription/degradation steady state.
    """
    rng = np.random.default_rng(seed)
    radius_nm = params["cell_radius_nm"]
    area_nm2 = sphere_area_nm2(radius_nm)
    lipids = 2.0 * area_nm2 / params["area_per_lipid_nm2"]
    k_tx = params["tx_elongation_nt_per_s"] / genes.length_bp          # per gene per second
    k_deg = math.log(2.0) / params["mrna_half_life_s"]
    mrna_ss = k_tx * genes.copies / k_deg
    mrna = rng.poisson(mrna_ss).astype(np.float64)
    # Proteins: balanced-growth newborn counts (see balanced_protein).
    protein = balanced_protein(params, genes)
    pool = params["adenylate_pool"]
    f = params["atp_initial_fraction"]
    return CellState(t_s=0.0, mrna=mrna, protein=protein, atp=pool * f, adp=pool * (1.0 - f),
                     lipids=lipids, replicated_bp=0.0, volume_um3=sphere_volume_um3(radius_nm))
