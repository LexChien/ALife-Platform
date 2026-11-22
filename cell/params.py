"""L0: parameter loading with mandatory provenance (Plan 44 section 4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

PARAM_DIR = Path(__file__).resolve().parent / "params"
UNSOURCED_TAGS = ("ESTIMATE-UNMEASURED",)
ASSUMED_TAGS = ("ASSUMED-TEXTBOOK", "DERIVED", "ASSUMED", "CALIBRATED")


@dataclass
class Params:
    """Flat parameter set plus the provenance of every value."""

    meta: Dict[str, Any]
    constants: Dict[str, Dict[str, Any]]
    gene_classes: Dict[str, Dict[str, Any]]

    def __getitem__(self, key: str) -> float:
        return float(self.constants[key]["value"])

    def source(self, key: str) -> str:
        return str(self.constants[key]["source"])

    def unsourced_keys(self) -> list:
        return sorted(k for k, v in self.constants.items()
                      if any(t in str(v["source"]) for t in UNSOURCED_TAGS))

    def assumed_keys(self) -> list:
        return sorted(k for k, v in self.constants.items()
                      if any(t in str(v["source"]) for t in ASSUMED_TAGS))

    def literature_keys(self) -> list:
        bad = set(self.unsourced_keys()) | set(self.assumed_keys())
        return sorted(k for k in self.constants if k not in bad)

    def provenance_report(self) -> Dict[str, Any]:
        n = len(self.constants)
        unsourced = self.unsourced_keys()
        assumed = self.assumed_keys()
        return {
            "n_constants": n,
            "n_literature": len(self.literature_keys()),
            "n_assumed": len(assumed),
            "n_estimate_unmeasured": len(unsourced),
            "fraction_estimate_unmeasured": (len(unsourced) / n) if n else 0.0,
            "estimate_unmeasured_keys": unsourced,
            "assumed_keys": assumed,
        }


def load_params(name: str = "syn3a_core") -> Params:
    path = PARAM_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    constants = data["constants"]
    for key, entry in constants.items():
        if "value" not in entry or "source" not in entry or not str(entry["source"]).strip():
            raise ValueError(f"parameter {key} has no source; Plan 44 section 4 forbids unsourced parameters")
    return Params(meta=data.get("meta", {}), constants=constants, gene_classes=data.get("gene_classes", {}))
