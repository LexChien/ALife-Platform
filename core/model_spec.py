from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    role: str
    runtime_backend: str
    artifact_format: str
    artifact_path: str
    hardware_target: str | None = None
    prompt_profile: str | None = None
    quantization: str | None = None
    lineage: dict[str, Any] | None = None

    @property
    def model_path(self) -> str:
        return self.artifact_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        required = (
            "model_id",
            "family",
            "role",
            "runtime_backend",
            "artifact_format",
            "artifact_path",
        )
        missing = [field for field in required if not data.get(field)]
        if missing:
            raise ValueError(f"ModelSpec missing required fields: {missing}")

        lineage = data.get("lineage") or {}
        if not isinstance(lineage, dict):
            raise ValueError("ModelSpec lineage must be a mapping.")

        return cls(
            model_id=str(data["model_id"]),
            family=str(data["family"]),
            role=str(data["role"]),
            runtime_backend=str(data["runtime_backend"]),
            artifact_format=str(data["artifact_format"]),
            artifact_path=str(data["artifact_path"]),
            hardware_target=data.get("hardware_target"),
            prompt_profile=data.get("prompt_profile"),
            quantization=data.get("quantization"),
            lineage=lineage or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "role": self.role,
            "runtime_backend": self.runtime_backend,
            "artifact_format": self.artifact_format,
            "artifact_path": self.artifact_path,
            "hardware_target": self.hardware_target,
            "prompt_profile": self.prompt_profile,
            "quantization": self.quantization,
            "lineage": self.lineage or {},
        }

