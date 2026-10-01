from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dna.genome import Genome


class GenomeStore:
    """Filesystem genome store: genomes/<id>.json, lineage.jsonl, current.json pointer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "genomes").mkdir(parents=True, exist_ok=True)

    def path_for(self, genome_id: str) -> Path:
        return self.root / "genomes" / f"{genome_id}.json"

    def put(self, genome: Genome) -> str:
        path = self.path_for(genome.genome_id)
        is_new = not path.exists()
        genome.save(path)
        if is_new:
            with (self.root / "lineage.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"genome_id": genome.genome_id, "parents": genome.parents,
                                     "generation": genome.generation, "fitness": genome.fitness,
                                     "op": genome.meta.get("op"), "recorded_at": datetime.now(timezone.utc).isoformat()},
                                    ensure_ascii=False) + "\n")
        return genome.genome_id

    def get(self, genome_id: str) -> Genome:
        return Genome.load(self.path_for(genome_id))

    def set_current(self, genome_id: str) -> None:
        if not self.path_for(genome_id).exists():
            raise KeyError(genome_id)
        (self.root / "current.json").write_text(json.dumps({"genome_id": genome_id}), encoding="utf-8")

    def current(self) -> Optional[Genome]:
        ptr = self.root / "current.json"
        if not ptr.exists():
            return None
        return self.get(json.loads(ptr.read_text(encoding="utf-8"))["genome_id"])

    def lineage(self) -> List[Dict]:
        path = self.root / "lineage.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def ancestry(self, genome_id: str) -> List[str]:
        """Breadth-first list of ancestor ids present in the store."""
        seen: List[str] = []
        frontier = [genome_id]
        while frontier:
            gid = frontier.pop(0)
            if gid in seen or not self.path_for(gid).exists():
                continue
            seen.append(gid)
            frontier.extend(self.get(gid).parents)
        return seen
