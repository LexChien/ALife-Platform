from __future__ import annotations

from copy import deepcopy

from core.model_registry import get_model_spec
from genai.llm.backends.dummy import DummyLLMAdapter
from genai.llm.backends.llama_cpp import LlamaCppAdapter


def _resolve_llm_config(cfg: dict) -> dict:
    llm_cfg = deepcopy(cfg.get("llm", {}))
    model_id = llm_cfg.get("model_id")
    if not model_id:
        return llm_cfg

    spec = get_model_spec(model_id)
    resolved = {
        "model_id": spec.model_id,
        "backend": spec.runtime_backend,
        "model_family": spec.family,
        "model_path": spec.model_path,
        "prompt_profile": spec.prompt_profile,
        "quantization": spec.quantization,
        "role": spec.role,
        "lineage": deepcopy(spec.lineage) if spec.lineage else {},
    }
    resolved.update(llm_cfg)
    resolved["model_id"] = spec.model_id
    resolved["backend"] = spec.runtime_backend
    resolved["model_family"] = spec.family
    resolved["model_path"] = spec.model_path
    if spec.prompt_profile and "prompt_profile" not in llm_cfg:
        resolved["prompt_profile"] = spec.prompt_profile
    if spec.quantization and "quantization" not in llm_cfg:
        resolved["quantization"] = spec.quantization
    resolved["role"] = spec.role
    resolved["lineage"] = deepcopy(spec.lineage) if spec.lineage else {}
    return resolved


def create_llm_adapter(cfg: dict):
    llm_cfg = _resolve_llm_config(cfg)
    backend = llm_cfg.get("backend", "dummy")
    if backend == "dummy":
        return DummyLLMAdapter(model_family=llm_cfg.get("model_family", "dummy"))
    if backend == "llama_cpp":
        return LlamaCppAdapter.from_config(llm_cfg)
    raise ValueError(f"Unsupported llm backend: {backend}")
