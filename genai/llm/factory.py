from __future__ import annotations

from genai.llm.backends.dummy import DummyLLMAdapter
from genai.llm.backends.llama_cpp import LlamaCppAdapter


def create_llm_adapter(cfg: dict):
    llm_cfg = cfg.get("llm", {})
    backend = llm_cfg.get("backend", "dummy")
    if backend == "dummy":
        return DummyLLMAdapter(model_family=llm_cfg.get("model_family", "dummy"))
    if backend == "llama_cpp":
        return LlamaCppAdapter.from_config(llm_cfg)
    raise ValueError(f"Unsupported llm backend: {backend}")
