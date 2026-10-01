from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

GENOME_SCHEMA_VERSION = "1.0"

# Named heritable traits in [0, 1] (default, description). These are expressed
# into the DigiClone runtime by dna.expression.express_persona.
TRAIT_SPEC: Dict[str, tuple] = {
    "warmth": (0.6, "affective warmth of replies / 回覆溫度"),
    "empathy": (0.6, "weight on acknowledging user feelings / 同理"),
    "curiosity": (0.5, "tendency to ask follow-up questions / 好奇"),
    "verbosity": (0.4, "answer length / 冗長度"),
    "formality": (0.5, "register / 正式程度"),
    "playfulness": (0.3, "humour, sampling temperature / 玩心"),
    "stability": (0.7, "identity stability, resistance to drift / 穩定"),
}


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Genome:
    """A heritable genome: substrate parameters (theta) + named persona traits."""

    theta: List[float]
    traits: Dict[str, float] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    fitness: Optional[float] = None
    substrate: str = "generic"
    created_at: str = field(default_factory=_now)
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = GENOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.theta = [float(x) for x in self.theta]
        merged = {name: spec[0] for name, spec in TRAIT_SPEC.items()}
        for name, value in (self.traits or {}).items():
            if name not in TRAIT_SPEC:
                raise ValueError(f"unknown trait: {name}")
            merged[name] = _clip01(float(value))
        self.traits = merged

    @property
    def genome_id(self) -> str:
        """Content hash over heritable material only (theta, traits, substrate)."""
        payload = json.dumps(
            {"theta": [round(x, 8) for x in self.theta],
             "traits": {k: round(v, 8) for k, v in sorted(self.traits.items())},
             "substrate": self.substrate},
            sort_keys=True,
        )
        return "g-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "genome_id": self.genome_id,
            "substrate": self.substrate,
            "generation": self.generation,
            "theta": list(self.theta),
            "traits": dict(self.traits),
            "parents": list(self.parents),
            "fitness": self.fitness,
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        g = cls(
            theta=data["theta"], traits=data.get("traits", {}), parents=data.get("parents", []),
            generation=int(data.get("generation", 0)), fitness=data.get("fitness"),
            substrate=data.get("substrate", "generic"), created_at=data.get("created_at", _now()),
            meta=data.get("meta", {}), schema_version=data.get("schema_version", GENOME_SCHEMA_VERSION),
        )
        stored = data.get("genome_id")
        if stored and stored != g.genome_id:
            raise ValueError(f"genome_id mismatch: stored {stored} != computed {g.genome_id}")
        return g

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def founder(cls, theta_dim: int = 5, substrate: str = "generic", **traits: float) -> "Genome":
        return cls(theta=[0.0] * theta_dim, traits=traits, substrate=substrate, meta={"origin": "founder"})
