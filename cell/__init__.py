"""Artificial cell process simulation (Plan 44, M0-M3).

ZH: 依 docs/PLAN/44_ARTIFICIAL_CELL_SIMULATION_PLAN.md 的 L0-L5 實作；參考生物 JCVI-syn3A。
EN: Implements layers L0-L5 of Plan 44 for the JCVI-syn3A reference organism.

Every rate constant lives in cell/params/*.json with a `source` field. Values tagged
ESTIMATE-UNMEASURED are NOT literature values; the run summary reports their share.
"""
from .params import Params, load_params  # noqa: F401
from .state import CellState  # noqa: F401
from .genome_map import GeneTable, build_gene_table  # noqa: F401

__all__ = ["Params", "load_params", "CellState", "GeneTable", "build_gene_table"]
