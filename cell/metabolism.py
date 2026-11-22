"""L3: deterministic metabolism and the ATP ledger (Plan 44 L3, gate M2)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .genome_map import GeneTable
from .params import Params
from .state import CellState, founder_lipids


def atp_saturation(params: Params, atp: float) -> float:
    km = params["atp_km"]
    return float(atp / (km + atp)) if atp > 0 else 0.0


def glycolytic_production(params: Params, genes: GeneTable, state: CellState) -> float:
    """ATP molecules per second from glycolysis (syn3A is glycolysis-dominated, C0862)."""
    metabolic = float(state.protein[genes.class_mask("metabolic")].sum())
    glucose_per_s = params["k_glucose_per_transporter_per_s"] * metabolic
    # Glycolysis must invest ATP before it pays back (priming); with no ATP it cannot restart
    # (ASSUMED-TEXTBOOK mechanism, Km ESTIMATE-UNMEASURED). `glucose` in state.extra is the
    # external availability (1 = fed, 0 = starved), used by Plan 45 P7/P8.
    km_p = params["glycolysis_atp_priming_km"]
    prime = state.atp / (km_p + state.atp) if state.atp > 0 else 0.0
    glucose = float(state.extra.get("glucose", 1.0))
    return glucose_per_s * params["atp_per_glucose_glycolysis"] * prime * glucose


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
    salvaged = salvage_adenylate(params, state)
    return {
        "atp_produced": produced,
        "atp_consumed": consumed,
        "atp_maintenance": maintenance,
        "atp_shortfall": shortfall,
        "atp": state.atp,
        "closure_error": closure_error,
        "adenylate_salvaged": salvaged,
    }


def adenylate_target(params: Params, lipids: float) -> float:
    """ATP+ADP a cell of this size should hold: the founder pool per membrane lipid."""
    return params["adenylate_pool"] * lipids / founder_lipids(params)


def salvage_adenylate(params: Params, state: CellState) -> float:
    """Nucleotide salvage keeps the adenylate pool in step with cell size (added 2026-11-23).

    ZH：syn3A 沒有從頭合成核苷酸，只靠回收途徑（C0715）；C0862 的全細胞模擬中 ATP 在整個週期持續
    增加。舊版 ATP+ADP 總量固定，分裂時減半卻不再補，於是每代減半、譜系越來越慢。這裡把每個
    膜脂質對應的腺苷酸量維持在創始細胞的比例（以 ADP 形式補入，回收的 ATP 成本 ASSUMED 為 0）。
    是一個明確記帳的流入項，所以閉合誤差仍只量 ATP 帳本身。
    EN: syn3A has no de novo nucleotide synthesis, only salvage (C0715); in C0862's whole-cell runs
    ATP keeps rising across the cycle. The old version held ATP+ADP fixed, so it halved at every
    division and was never re-made, and lineages slowed every generation. Here the adenylate amount
    per membrane lipid is kept at the founder ratio (added as ADP; salvage ATP cost ASSUMED 0). It is
    an explicitly booked influx, so the closure error still measures the ATP ledger alone.
    Author: Grok Bot ALife Project - Andreeva, 2026-11-23.
    """
    deficit = adenylate_target(params, state.lipids) - (state.atp + state.adp)
    if deficit > 0.0:
        state.adp += deficit
        return deficit
    return 0.0
