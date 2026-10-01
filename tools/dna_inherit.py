#!/usr/bin/env python3
"""Plan 37 N3: inherit an evolved ASAL genome into the DigiClone genome store.

Child = theta from the ASAL donor (body) + traits crossed between the current
clone genome and the donor (seeded), parents=[clone_current, donor].
The new child becomes the clone store's current genome; gemma_web reads it on start.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dna import Genome, GenomeStore, TRAIT_SPEC, express_persona  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-store", required=True, help="e.g. runs/asal/plan37_dna_lenia_*/genome_store")
    ap.add_argument("--clone-store", default="runs/dna/clone_store")
    ap.add_argument("--seed", type=int, default=37)
    args = ap.parse_args()
    donor_store = GenomeStore(ROOT / args.donor_store)
    donor = donor_store.current()
    if donor is None:
        raise SystemExit("donor store has no current genome")
    clone_store = GenomeStore(ROOT / args.clone_store)
    current = clone_store.current() or Genome.founder(theta_dim=len(donor.theta), substrate="clone")
    clone_store.put(donor)  # keep the donor itself in the clone lineage
    rng = random.Random(args.seed)
    traits = {k: (current.traits[k] if rng.random() < 0.5 else donor.traits[k]) for k in TRAIT_SPEC}
    child = Genome(theta=list(donor.theta), traits=traits, parents=[current.genome_id, donor.genome_id],
                   generation=max(current.generation, donor.generation) + 1, substrate=f"clone<-{donor.substrate}",
                   fitness=donor.fitness, meta={"op": "inherit", "donor_store": args.donor_store})
    clone_store.put(child)
    clone_store.set_current(child.genome_id)
    print(json.dumps({"child": child.genome_id, "parents": child.parents, "generation": child.generation,
                      "expression": express_persona(child)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
