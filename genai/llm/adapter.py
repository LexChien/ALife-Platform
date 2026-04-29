from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    prompt: str
    context: str | None = None
    system: str | None = None
    life_context: dict[str, Any] | None = None  # 新增：ASAL metrics
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    json_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model_family: str
    backend: str
    runtime: dict[str, Any]
    prompt_tokens_est: int | None = None
    completion_tokens_est: int | None = None
    raw: Any = None


class BaseLLMAdapter(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_family(self) -> str:
        ...

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        ...

    def build_prompt(self, request: LLMRequest) -> str:
        parts = []
        if request.system:
            parts.append(f"<|system|>\n{request.system}")
        if request.life_context:
            life_str = json.dumps(request.life_context, ensure_ascii=False)
            parts.append(f"<|life_context|>\n{life_str}")
        if request.context:
            parts.append(f"<|context|>\n{request.context}")
        parts.append(f"<|user|>\n{request.prompt}")
        return "\n\n".join(parts)
