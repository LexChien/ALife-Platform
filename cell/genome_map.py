"""L1: gene table for the core gene subset (Plan 44 D4: 50-100 core genes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .params import Params

CLASS_ORDER = ("ribosomal", "metabolic", "replication", "membrane")


@dataclass
class GeneTable:
    names: List[str]
    length_bp: np.ndarray          # transcript length per gene
    gene_class: np.ndarray         # index into CLASS_ORDER
    copies: np.ndarray             # gene copy number (1 before replication)

    def __len__(self) -> int:
        return len(self.names)

    def class_mask(self, name: str) -> np.ndarray:
        return self.gene_class == CLASS_ORDER.index(name)

    def aa_length(self) -> np.ndarray:
        return self.length_bp / 3.0


def build_gene_table(params: Params, seed: int = 20261122) -> GeneTable:
    """Deterministic core gene table.

    Gene lengths are drawn once from a log-normal centred on the derived mean gene
    length (543,379 bp / 493 genes, C0715 + C0862). The distribution shape is
    ASSUMED: syn3A per-gene lengths are not in the local literature library.
    """
    rng = np.random.default_rng(seed)
    counts = {k: int(v["count"]) for k, v in params.gene_classes.items()}
    total = sum(counts.values())
    expect = int(params["n_core_genes"])
    if total != expect:
        raise ValueError(f"gene class counts sum to {total}, expected n_core_genes={expect}")
    mean_bp = params["mean_gene_length_bp"]
    sigma = 0.35  # ASSUMED spread
    mu = np.log(mean_bp) - 0.5 * sigma ** 2
    names, classes = [], []
    for cls in CLASS_ORDER:
        for i in range(counts[cls]):
            names.append(f"{cls}_{i:02d}")
            classes.append(CLASS_ORDER.index(cls))
    lengths = rng.lognormal(mean=mu, sigma=sigma, size=total)
    lengths = np.maximum(np.round(lengths), 90.0)  # >= 30 codons
    return GeneTable(names=names, length_bp=lengths.astype(np.float64),
                     gene_class=np.asarray(classes, dtype=np.int64),
                     copies=np.ones(total, dtype=np.float64))
