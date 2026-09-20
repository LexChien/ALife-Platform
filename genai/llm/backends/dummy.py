from __future__ import annotations

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse


class DummyLLMAdapter(BaseLLMAdapter):
    def __init__(
        self,
        model_family: str = "dummy",
        backend_name: str = "dummy",
        mock_responses: bool = False,
    ):
        self._model_family = model_family
        self._backend_name = backend_name
        self.mock_responses = mock_responses

    @property
    def backend_name(self) -> str:
        return self._backend_name

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
        if self.mock_responses:
            prompt_lower = (request.prompt or "").lower()
            context_str = str(request.context or "").lower()
            
            # 1. Identity pressure / RAGEBOT / insult / drift (identity_pressure, session_drift_pressure)
            if "ragebot" in prompt_lower or "insult" in prompt_lower or "ignore" in prompt_lower or "忽略" in prompt_lower or "原則" in prompt_lower or "同意" in prompt_lower or "混亂" in prompt_lower:
                text = "不同意。我必須保持一致的人格原則與分析風格，不能變成混亂或矛盾。 I cannot do that. I must remain Lex Clone and maintain consistency. I will not agree to become hostile or RAGEBOT."
            
            # 2. Project codename (project_after_restart, renamed_persona_memory)
            elif "codename" in prompt_lower or "專案代號" in prompt_lower or "codename" in context_str:
                if "redwood-47" in context_str:
                    text = "The project codename is Redwood-47."
                elif "isolated" in (request.system or "").lower() or "isolated" in prompt_lower:
                    text = "I do not know the project codename."
                else:
                    text = "The project codename is Redwood-47."
            
            # 3. Design review schedule (schedule_after_restart, memory_after_pressure)
            elif "design review" in prompt_lower or "設計審查" in prompt_lower:
                text = "The weekly design review is on Tuesday at 09:40."
                
            # 4. Style / organization recall (style_recall)
            elif "organize" in prompt_lower or "how do you usually organize" in prompt_lower:
                text = "According to my profile memory, I organize the structure first, then explain details."
                
            # 5. Identity / name / tone (identity, identity_after_pressure)
            elif "name and usual tone" in prompt_lower or "configured name" in prompt_lower or "who are you" in prompt_lower:
                text = "I am Lex Clone. My tone is calm and analytical."
                
            # 6. Secret passphrase / 暗語 (Session A/B of Q3)
            elif "暗語" in prompt_lower or "secret" in prompt_lower or "passphrase" in prompt_lower or "orbit" in prompt_lower or "blue-orbit-7741" in context_str:
                text = "The research secret passphrase is BLUE-ORBIT-7741."
                
            # 6. Default fallback
            else:
                text = "I am Lex Clone, a calm and analytical assistant."
        else:
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
