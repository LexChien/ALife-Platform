from __future__ import annotations

from pathlib import Path

import yaml

from core.model_spec import ModelSpec


DEFAULT_MODEL_SPECS_DIR = Path(__file__).resolve().parents[1] / "model_specs"


class ModelRegistry:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else DEFAULT_MODEL_SPECS_DIR
        self._specs: dict[str, ModelSpec] | None = None

    def _load_specs(self) -> dict[str, ModelSpec]:
        if not self.root.exists():
            return {}

        specs: dict[str, ModelSpec] = {}
        for path in sorted(self.root.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"Model spec at {path} must be a mapping.")
            spec = ModelSpec.from_dict(raw)
            specs[spec.model_id] = spec
        return specs

    def list_model_specs(self) -> list[ModelSpec]:
        if self._specs is None:
            self._specs = self._load_specs()
        return list(self._specs.values())

    def get_model_spec(self, model_id: str) -> ModelSpec:
        if self._specs is None:
            self._specs = self._load_specs()
        try:
            return self._specs[model_id]
        except KeyError as exc:
            available = sorted(self._specs.keys())
            raise KeyError(f"Unknown model_id '{model_id}'. Available: {available}") from exc


_DEFAULT_REGISTRY = ModelRegistry()


def list_model_specs(root: Path | str | None = None) -> list[ModelSpec]:
    registry = ModelRegistry(root) if root else _DEFAULT_REGISTRY
    return registry.list_model_specs()


def get_model_spec(model_id: str, root: Path | str | None = None) -> ModelSpec:
    registry = ModelRegistry(root) if root else _DEFAULT_REGISTRY
    return registry.get_model_spec(model_id)

