"""L0 driver: hybrid CME + ODE loop with a fixed 1 s communication step (Plan 44 section 5)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import division, expression_tx_tl, membrane, metabolism, replication
from .genome_map import GeneTable, build_gene_table
from .params import Params, load_params
from .state import CellState, initial_state


@dataclass
class RunConfig:
    max_time_s: float = 20000.0
    dt_s: float = 1.0                 # communication step (C0862 uses 1 s)
    max_generations: int = 1
    seed: int = 0
    telemetry_every_s: float = 60.0
    overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunResult:
    doubling_times_s: List[float]
    telemetry: List[Dict[str, Any]]
    max_closure_error: float
    atp_shortfall_total: float
    wall_time_s: float
    finished_reason: str
    final_state: Optional[CellState] = None

    def summary(self) -> Dict[str, Any]:
        dt = np.asarray(self.doubling_times_s, dtype=float)
        return {
            "n_divisions": int(dt.size),
            "doubling_time_s": dt.tolist(),
            "doubling_time_median_s": float(np.median(dt)) if dt.size else None,
            "doubling_time_min_s": float(dt.min()) if dt.size else None,
            "doubling_time_max_s": float(dt.max()) if dt.size else None,
            "max_closure_error": self.max_closure_error,
            "atp_shortfall_total": self.atp_shortfall_total,
            "wall_time_s": self.wall_time_s,
            "finished_reason": self.finished_reason,
        }


def apply_overrides(params: Params, overrides: Dict[str, float]) -> Params:
    if not overrides:
        return params
    constants = {k: dict(v) for k, v in params.constants.items()}
    for key, value in overrides.items():
        if key not in constants:
            raise KeyError(f"unknown parameter override: {key}")
        constants[key] = dict(constants[key])
        constants[key]["value"] = float(value)
        constants[key]["source"] = constants[key]["source"] + " | OVERRIDDEN-AT-RUNTIME"
    return Params(meta=params.meta, constants=constants, gene_classes=params.gene_classes)


def run_single(params: Params, genes: GeneTable, config: RunConfig) -> RunResult:
    """Run one lineage (mother cell, keeping daughter 1) for up to max_generations."""
    params = apply_overrides(params, config.overrides)
    rng = np.random.default_rng(config.seed)
    state = initial_state(params, genes, seed=config.seed)
    birth_area = state.surface_area_nm2(params)
    doubling: List[float] = []
    telemetry: List[Dict[str, Any]] = []
    max_closure = 0.0
    shortfall_total = 0.0
    t0 = time.perf_counter()
    reason = "max_time"
    steps = int(config.max_time_s / config.dt_s)
    next_tel = 0.0
    for _ in range(steps):
        dt = config.dt_s
        costs: Dict[str, float] = {}
        costs.update(expression_tx_tl.step(params, genes, state, dt, rng))
        mem = membrane.step(params, genes, state, dt)
        rep = replication.step(params, genes, state, dt)
        costs["atp_lipid"] = mem["atp_lipid"]
        costs["atp_replication"] = rep["atp_replication"]
        ledger = metabolism.step(params, genes, state, dt, costs)
        max_closure = max(max_closure, ledger["closure_error"])
        shortfall_total += ledger["atp_shortfall"]
        state.t_s += dt
        if state.t_s >= next_tel:
            telemetry.append({
                "t_s": state.t_s, "generation": state.generation,
                "mrna_total": float(state.mrna.sum()), "protein_total": float(state.protein.sum()),
                "atp": state.atp, "lipids": state.lipids,
                "surface_area_um2": mem["surface_area_um2"], "sa_v": mem["sa_v"],
                "replication_fraction": rep["fraction"],
                "atp_produced_per_s": ledger["atp_produced"] / dt,
                "atp_consumed_per_s": ledger["atp_consumed"] / dt,
                "closure_error": ledger["closure_error"],
            })
            next_tel += config.telemetry_every_s
        if division.ready_to_divide(params, state, birth_area):
            doubling.append(state.t_s - state.birth_t_s)
            d1, _d2 = division.divide(state, rng)
            state = d1
            birth_area = state.surface_area_nm2(params)
            if state.generation >= config.max_generations:
                reason = "max_generations"
                break
    wall = time.perf_counter() - t0
    return RunResult(doubling_times_s=doubling, telemetry=telemetry, max_closure_error=max_closure,
                     atp_shortfall_total=shortfall_total, wall_time_s=wall,
                     finished_reason=reason, final_state=state)


def run_from_defaults(seed: int = 0, **kwargs) -> RunResult:
    params = load_params()
    genes = build_gene_table(params)
    return run_single(params, genes, RunConfig(seed=seed, **kwargs))


def write_telemetry(path: Path, result: RunResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in result.telemetry:
            fh.write(json.dumps(row) + "\n")
    return path
