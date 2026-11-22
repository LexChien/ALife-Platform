"""L7 heredity on the CPU (Plan 44 M5, Plan 45 P3/P4): full division trees with dna.genome lineage.

ZH: 每個細胞以 NumPy 驅動器（與 cell/driver.py 相同模組）從「出生狀態」跑到可分裂，
分裂時分子數以二項分配給兩個子代；每個子代有一個 substrate="cell" 的 dna.genome.Genome，
突變只發生在被記錄的位點，因此任何子代與親代的關係都可由 genome_id 驗證。
EN: Each cell is advanced with the NumPy driver modules from its birth state to division;
molecules are binomially partitioned to both daughters; every daughter carries a dna.genome.Genome
(substrate="cell"), mutated only at recorded loci, so every parent/child relation is checkable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dna.genome import Genome
from dna.store import GenomeStore

from . import division, expression_tx_tl, membrane, metabolism, replication
from .dna_link import THETA_KEYS, express_cell, neutral_theta
from .driver import apply_overrides
from .genome_map import GeneTable, build_gene_table
from .params import Params, load_params
from .state import CellState, initial_state


def grow_until_division(params: Params, genes: GeneTable, state: CellState, rng: np.random.Generator,
                        dt: float = 1.0, max_time_s: float = 40000.0) -> Tuple[bool, float, float]:
    """Advance `state` in place until it is ready to divide. Returns (divided, doubling_s, max_closure)."""
    birth_area = state.surface_area_nm2(params)
    t_birth = state.t_s
    max_closure = 0.0
    steps = int(max_time_s / dt)
    for _ in range(steps):
        costs: Dict[str, float] = {}
        costs.update(expression_tx_tl.step(params, genes, state, dt, rng))
        mem = membrane.step(params, genes, state, dt)
        rep = replication.step(params, genes, state, dt)
        costs["atp_lipid"] = mem["atp_lipid"]
        costs["atp_replication"] = rep["atp_replication"]
        ledger = metabolism.step(params, genes, state, dt, costs)
        max_closure = max(max_closure, ledger["closure_error"])
        state.t_s += dt
        if division.ready_to_divide(params, state, birth_area):
            return True, state.t_s - t_birth, max_closure
    return False, float("nan"), max_closure


def mutate_genome(parent: Genome, rng: np.random.Generator, mu: float, sigma: float,
                  uid: str, parent_uid: str) -> Genome:
    theta = list(parent.theta)
    loci = []
    for i in range(len(theta)):
        if rng.random() < mu:
            theta[i] = float(min(max(theta[i] + rng.normal(0.0, sigma), 0.0), 1.0))
            loci.append(i)
    return Genome(theta=theta, parents=[parent.genome_id], generation=parent.generation + 1,
                  substrate="cell", meta={"op": "cell_division", "cell_uid": uid, "parent_uid": parent_uid,
                                          "mutated_loci": loci})


def run_tree(founder_index: int, depth: int, mu: float, sigma: float, seed: int) -> List[Dict[str, Any]]:
    """Simulate the complete division tree of one founder for `depth` generations of divisions."""
    base = load_params()
    genes = build_gene_table(base)
    rng = np.random.default_rng(seed)
    founder = Genome(theta=neutral_theta(), substrate="cell",
                     meta={"op": "founder", "cell_uid": f"F{founder_index}", "parent_uid": None, "mutated_loci": []})
    st0 = initial_state(base, genes, seed=seed)
    records: List[Dict[str, Any]] = []
    frontier = [(founder, st0)]
    for gen in range(depth + 1):
        nxt = []
        for g, st in frontier:
            params = apply_overrides(base, express_cell(g, {k: base[k] for k in THETA_KEYS}))
            mol_before = None
            divided, dbl, clos = grow_until_division(params, genes, st, rng)
            rec = {"cell_uid": g.meta["cell_uid"], "parent_uid": g.meta["parent_uid"], "generation": gen,
                   "genome": g.to_dict(), "divided": divided, "doubling_s": dbl, "max_closure": clos}
            if divided and gen < depth:
                mol_before = {"mrna": float(st.mrna.sum()), "protein": float(st.protein.sum()),
                              "lipids": st.lipids, "atp": st.atp, "adp": st.adp}
                d1, d2 = division.divide(st, rng)
                rec["partition"] = {"parent": mol_before,
                                    "d1": {"mrna": float(d1.mrna.sum()), "protein": float(d1.protein.sum()),
                                           "lipids": d1.lipids, "atp": d1.atp, "adp": d1.adp},
                                    "d2": {"mrna": float(d2.mrna.sum()), "protein": float(d2.protein.sum()),
                                           "lipids": d2.lipids, "atp": d2.atp, "adp": d2.adp}}
                for k, d in enumerate((d1, d2)):
                    uid = f"{g.meta['cell_uid']}.{k}"
                    child = mutate_genome(g, rng, mu, sigma, uid, g.meta["cell_uid"])
                    nxt.append((child, d))
            records.append(rec)
        frontier = nxt
        if not frontier:
            break
    return records


def _tree_worker(args):
    return run_tree(*args)


def verify_lineage(records: List[Dict[str, Any]], store_root: Path | None = None) -> Dict[str, Any]:
    """Gate M5 / L-A4 checks, recomputed from the raw records only."""
    by_uid = {r["cell_uid"]: r for r in records}
    n_ok_id = n_rel = n_rel_ok = 0
    errors: List[str] = []
    genomes: Dict[str, Genome] = {}
    for r in records:
        try:
            genomes[r["cell_uid"]] = Genome.from_dict(r["genome"])   # recomputes genome_id
            n_ok_id += 1
        except ValueError as exc:
            errors.append(f"{r['cell_uid']}: {exc}")
    events = trials = 0
    max_depth = 0
    for uid, r in by_uid.items():
        pu = r["parent_uid"]
        if pu is None:
            continue
        n_rel += 1
        parent, child = genomes.get(pu), genomes.get(uid)
        if parent is None or child is None:
            errors.append(f"{uid}: parent {pu} missing"); continue
        loci = child.meta.get("mutated_loci", [])
        diff = [i for i, (a, b) in enumerate(zip(parent.theta, child.theta)) if round(a, 8) != round(b, 8)]
        ok = child.parents == [parent.genome_id] and child.generation == parent.generation + 1 \
            and set(diff) <= set(loci)
        if not loci:
            ok = ok and child.genome_id == parent.genome_id
        n_rel_ok += int(ok)
        if not ok:
            errors.append(f"{uid}: relation check failed")
        events += len(loci); trials += len(child.theta)
        # depth by walking parent_uid back to a founder
        d, cur = 0, r
        while cur["parent_uid"] is not None:
            cur = by_uid[cur["parent_uid"]]; d += 1
        max_depth = max(max_depth, d)
    # molecule conservation at division
    cons_err = 0.0
    for r in records:
        p = r.get("partition")
        if p:
            for k in ("mrna", "protein", "lipids", "atp", "adp"):
                tot = p["parent"][k]
                if tot:
                    cons_err = max(cons_err, abs(p["d1"][k] + p["d2"][k] - tot) / tot)
    out = {"n_records": len(records), "n_genome_id_valid": n_ok_id, "n_relations": n_rel,
           "n_relations_valid": n_rel_ok, "max_depth_generations": max_depth,
           "mutation_events": events, "mutation_trials": trials,
           "mutation_rate_measured": (events / trials) if trials else None,
           "max_partition_conservation_error": cons_err, "errors": errors[:20]}
    if store_root is not None:
        store = GenomeStore(store_root)
        for r in sorted(records, key=lambda x: x["generation"]):
            g = genomes[r["cell_uid"]]
            if not store.path_for(g.genome_id).exists():   # first occurrence only (identical genomes share an id)
                store.put(g)
        leaves = [r for r in records if r["generation"] == max(x["generation"] for x in records)]
        chains_ok = 0
        for r in leaves:
            anc = store.ancestry(genomes[r["cell_uid"]].genome_id)
            root = r
            while root["parent_uid"] is not None:
                root = by_uid[root["parent_uid"]]
            chains_ok += int(genomes[root["cell_uid"]].genome_id in anc)
        out["store_leaf_chains_reaching_founder"] = chains_ok
        out["store_leaves_checked"] = len(leaves)
    return out
