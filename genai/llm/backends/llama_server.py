"""Plan 38 J1.2: resident llama-server adapter (OpenAI-compatible /v1/chat/completions, SSE streaming).

- ``chat(messages, ...)``: streaming chat with TTFT/timings (prompt_n = tokens actually evaluated, i.e. NOT served
  from the KV cache - the J1.4 cache metric), optional JSON schema (``response_format``) and slot pinning
  (``id_slot``: speech on slot 0, private thought on slot 1, so each keeps its own cached prefix).
- ``generate(LLMRequest)``: BaseLLMAdapter contract. The stable prefix (system) comes first; per-turn dynamic context
  (memory, emotion, language lock) goes at the END, inside the last user message (J1.4).
- If the server is unreachable and a fallback adapter (llama-cli) is configured, ``generate`` falls back to it."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse


class LlamaServerError(RuntimeError):
    pass


class LlamaServerAdapter(BaseLLMAdapter):
    def __init__(self, *, url: str = "http://127.0.0.1:8091", model_family: str = "gemma", max_tokens: int = 256,
                 temperature: float = 0.7, timeout: float = 120.0, manager=None, fallback=None, seed: int | None = None,
                 lineage: dict | None = None):
        # Plan 38: non-speech generate() calls (emotion classifier, ...) default to the thought slot so they
        # never evict the speech slot's cached prompt prefix (smoke: cache_n 4 -> see log).
        self.default_slot = None
        self.logit_bias = None  # Plan 38: [[token_id, false], ...] from genai.llm.script_ban
        self.url = url.rstrip("/")
        self._model_family = model_family
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.timeout = timeout
        self.manager = manager
        self.fallback = fallback
        self.seed = seed
        self.lineage = lineage or {}
        self.last_timings: dict | None = None

    @property
    def backend_name(self) -> str:
        return "llama_server"

    @property
    def model_family(self) -> str:
        return self._model_family

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LlamaServerAdapter":
        server = dict(cfg.get("server") or {})
        manager = None
        if server.get("manage", True):
            from genai.llm.server_manager import LlamaServerManager, ServerConfig
            sc = ServerConfig.from_dict({"model": cfg.get("model_path") or server.get("model"), **server})
            manager = LlamaServerManager(sc)
            try:
                manager.ensure()
                if server.get("watchdog", True):
                    manager.start_watchdog()
            except Exception:
                pass  # healthcheck reports it; generate() falls back
            url = sc.url
        else:
            url = server.get("url", "http://127.0.0.1:8091")
        fallback = None
        if server.get("fallback", "llama_cpp") == "llama_cpp" and cfg.get("model_path"):
            from genai.llm.backends.llama_cpp import LlamaCppAdapter
            fallback = LlamaCppAdapter.from_config({**cfg, "backend": "llama_cpp"})
        ad = cls(url=url, model_family=cfg.get("model_family", "gemma"), max_tokens=cfg.get("max_tokens", 256),
                 temperature=cfg.get("temperature", 0.7), timeout=float(server.get("timeout", 120)), manager=manager,
                 fallback=fallback, seed=cfg.get("seed"), lineage=cfg.get("lineage"))
        scripts = server.get("ban_scripts")
        if scripts and cfg.get("model_path"):
            try:
                from genai.llm.script_ban import load_ban_ids
                ad.logit_bias = [[i, False] for i in load_ban_ids(cfg["model_path"], tuple(scripts))]
            except Exception as exc:  # hygiene guard still strips foreign script
                ad.ban_error = f"{type(exc).__name__}: {exc}"
        return ad

    # ------------------------------------------------------------------ low level
    def _body(self, messages, *, max_tokens, temperature, json_schema, slot, stream, stop, extra):
        body: dict[str, Any] = {"messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
                                "stream": stream, "cache_prompt": True}
        if self.logit_bias:
            body["logit_bias"] = self.logit_bias
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"schema": json_schema}}
        if slot is not None:
            body["id_slot"] = int(slot)
        if stop:
            body["stop"] = list(stop)
        if self.seed is not None:
            body["seed"] = int(self.seed)
        if extra:
            body.update(extra)
        return body

    def stream(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None,
               json_schema: dict | None = None, slot: int | None = None, stop: list[str] | None = None,
               extra: dict | None = None, info: dict | None = None) -> Iterator[str]:
        """Yield content deltas. ``info`` (if given) is filled with ttft_s / total_s / timings at the end."""
        body = self._body(messages, max_tokens=max_tokens or self.default_max_tokens,
                          temperature=self.default_temperature if temperature is None else temperature,
                          json_schema=json_schema, slot=slot, stream=True, stop=stop, extra=extra)
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first = None
        timings = None
        finish = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    timings = obj.get("timings", timings)
                    for ch in obj.get("choices", []):
                        finish = ch.get("finish_reason") or finish
                        delta = (ch.get("delta") or {}).get("content")
                        if delta:
                            if first is None:
                                first = time.perf_counter() - t0
                            yield delta
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise LlamaServerError(f"llama-server unreachable at {self.url}: {exc}") from exc
        finally:
            meta = {"ttft_s": round(first, 4) if first is not None else None, "total_s": round(time.perf_counter() - t0, 4),
                    "timings": timings, "finish_reason": finish, "slot": slot}
            if timings:
                meta.update({"prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
                             "predicted_n": timings.get("predicted_n"),
                             "tok_s": round(timings.get("predicted_per_second") or 0.0, 1)})
            self.last_timings = meta
            if info is not None:
                info.update(meta)

    def chat(self, messages: list[dict], *, on_delta: Callable[[str], None] | None = None, **kw) -> dict:
        info: dict = {}
        parts = []
        for d in self.stream(messages, info=info, **kw):
            parts.append(d)
            if on_delta:
                on_delta(d)
        return {"text": "".join(parts), **info}

    # ------------------------------------------------------------------ BaseLLMAdapter
    @staticmethod
    def messages_from_request(request: LLMRequest) -> list[dict]:
        meta = request.metadata or {}
        msgs: list[dict] = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        for item in meta.get("history") or []:
            role = item.get("role")
            if role in ("user", "assistant") and item.get("content"):
                msgs.append({"role": role, "content": item["content"]})
        user = request.prompt
        if request.context:
            # dynamic per-turn context goes LAST so the cached prefix (system + history) stays identical
            user = f"{request.context}\n\n---\n{request.prompt}"
        msgs.append({"role": "user", "content": user})
        return msgs

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = (request.metadata or {}).get("messages") or self.messages_from_request(request)
        t0 = time.perf_counter()
        try:
            out = self.chat(messages, max_tokens=request.max_tokens, temperature=request.temperature,
                            json_schema=(request.metadata or {}).get("json_schema"),
                            slot=(request.metadata or {}).get("slot", self.default_slot), stop=request.stop)
        except LlamaServerError as exc:
            if self.fallback is None:
                raise
            resp = self.fallback.generate(request)
            resp.runtime = {**(resp.runtime or {}), "fallback_from": "llama_server", "fallback_reason": str(exc)}
            return resp
        runtime = {"backend": "llama_server", "url": self.url, "elapsed_s": round(time.perf_counter() - t0, 3),
                   **{k: out.get(k) for k in ("ttft_s", "total_s", "prompt_n", "cache_n", "predicted_n", "tok_s", "finish_reason", "slot")}}
        return LLMResponse(text=out["text"].strip(), model_family=self.model_family, backend=self.backend_name,
                           runtime=runtime, completion_tokens_est=out.get("predicted_n"), raw={"timings": out.get("timings")})

    def healthcheck(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                status = json.loads(r.read() or b"{}").get("status")
        except Exception as exc:
            status = f"down: {type(exc).__name__}"
        out = {"backend": self.backend_name, "url": self.url, "ok": status == "ok", "status": status,
               "model_family": self.model_family, "fallback": getattr(self.fallback, "backend_name", None),
               "banned_tokens": len(self.logit_bias or []), "ban_error": getattr(self, "ban_error", None)}
        if self.manager is not None:
            h = self.manager.health()
            out["manager"] = {k: h.get(k) for k in ("pid", "launcher", "adopted", "restarts", "last_event")}
        return out
