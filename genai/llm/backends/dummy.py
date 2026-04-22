from __future__ import annotations

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse


class DummyLLMAdapter(BaseLLMAdapter):
    def __init__(self, model_family: str = "dummy"):
        self._model_family = model_family

    @property
    def backend_name(self) -> str:
        return "dummy"

    @property
    def model_family(self) -> str:
        return self._model_family

    def healthcheck(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "model_family": self.model_family,
            "mode": "dummy",
            "ok": True,
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = (
            f"[DummyLLM] prompt={request.prompt} "
            f"context={request.context or 'none'} "
            f"system={request.system or 'none'}"
        )
        return LLMResponse(
            text=text,
            model_family=self.model_family,
            backend=self.backend_name,
            runtime={
                "backend": self.backend_name,
                "model_family": self.model_family,
                "mode": "dummy",
            },
            raw=None,
        )
