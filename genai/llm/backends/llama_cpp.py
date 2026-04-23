from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LlamaCppAdapter(BaseLLMAdapter):
    def __init__(
        self,
        model_family: str,
        model_path: str,
        context_length: int = 4096,
        temperature: float = 0.7,
        max_tokens: int = 256,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
        verbose: bool = False,
        driver: str = "auto",
        cli_path: str = "llama-completion",
    ):
        self._model_family = model_family
        self.model_path = model_path
        self.context_length = context_length
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.verbose = verbose
        self.driver = driver
        self.cli_path = cli_path
        self.subprocess_timeout = 120
        self._llm = None

    @property
    def backend_name(self) -> str:
        return "llama_cpp"

    @property
    def model_family(self) -> str:
        return self._model_family

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LlamaCppAdapter":
        return cls(
            model_family=cfg["model_family"],
            model_path=cfg["model_path"],
            context_length=cfg.get("context_length", 4096),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 256),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            chat_format=cfg.get("chat_format"),
            verbose=cfg.get("verbose", False),
            driver=cfg.get("driver", "auto"),
            cli_path=cfg.get("cli_path", "llama-completion"),
        )

    def _cli_resolved_path(self) -> str | None:
        return shutil.which(self.cli_path)

    def _binding_available(self) -> bool:
        return Llama is not None

    def _resolve_driver(self) -> str:
        if self.driver == "python":
            return "python"
        if self.driver == "subprocess":
            return "subprocess"
        if self._binding_available():
            return "python"
        if self._cli_resolved_path():
            return "subprocess"
        return "unavailable"

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        if not self._binding_available():
            raise RuntimeError("llama-cpp-python is not installed")
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_length,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format,
            verbose=self.verbose,
        )

    def build_prompt(self, request: LLMRequest) -> str:
        parts = []
        if request.system:
            parts.append(f"<system>\n{request.system}")
        if request.context:
            parts.append(f"<context>\n{request.context}")
        parts.append(f"<user>\n{request.prompt}")
        return "\n\n".join(parts)

    def _subprocess_prompt(self, request: LLMRequest) -> str:
        if request.context:
            return f"Context:\n{request.context}\n\nUser request:\n{request.prompt}"
        return request.prompt

    def _subprocess_system(self, request: LLMRequest) -> str:
        base = request.system.strip() if request.system else "You are a helpful assistant."
        suffix = (
            " Return only the final answer. "
            "Do not reveal chain-of-thought. "
            "Do not emit reasoning channels, hidden analysis, or thought tags."
        )
        return f"{base}{suffix}"

    def _reasoning_markers(self) -> tuple[str, ...]:
        return (
            "<|channel>thought",
            "Thinking Process:",
            "Here's a thinking process",
            "Analyze the Request:",
            "Deconstruct Key Terms:",
            "Brainstorm Core Concepts",
        )

    def _has_reasoning_leak(self, text: str) -> bool:
        stripped = text.strip()
        if any(marker in stripped for marker in self._reasoning_markers()):
            return True
        return bool(re.match(r"^1\.\s+\*\*Analyze", stripped))

    def _heuristic_extract_final_answer(self, text: str) -> str:
        cleaned = text.strip()
        for marker in ("Final Answer:", "Final Response:", "Answer:"):
            if marker in cleaned:
                tail = cleaned.split(marker, 1)[-1].strip()
                if tail:
                    return tail

        lines = [line.rstrip() for line in cleaned.splitlines()]
        kept: list[str] = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if started and kept and kept[-1] != "":
                    kept.append("")
                continue
            if any(marker in stripped for marker in self._reasoning_markers()):
                continue
            if re.match(r"^\d+\.\s+\*\*.*\*\*:?", stripped):
                continue
            if stripped.startswith("*   **") or stripped.startswith("- **"):
                continue
            if not started and re.match(r"^\d+\.\s+", stripped):
                continue
            started = True
            kept.append(stripped)

        candidate = "\n".join(kept).strip()
        return candidate or cleaned

    def _looks_fragmented(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return True
        bullet_like = sum(
            1
            for line in lines
            if line.startswith(("*", "-", "•"))
            or re.match(r"^\d+\.", line)
            or re.match(r"^[A-Z][A-Za-z ]+:\s+\S+", line)
        )
        prose_markers = sum(stripped.count(mark) for mark in (".", "。", "!", "！", "?", "？"))
        if bullet_like == len(lines) and len(lines) >= 2:
            return True
        if len(stripped) < 80 and prose_markers == 0:
            return True
        return False

    def _should_apply_fallback(self, original: str, candidate: str) -> tuple[bool, str | None]:
        original = original.strip()
        candidate = candidate.strip()
        if not candidate:
            return False, "empty_candidate"
        if candidate == original:
            return False, "no_change"
        if self._looks_fragmented(candidate) and not self._looks_fragmented(original):
            return False, "candidate_fragmented"
        if len(original) >= 160 and len(candidate) < max(80, int(len(original) * 0.4)):
            return False, "overcompressed"
        return True, None

    def _extractor_request(self, request: LLMRequest, draft: str) -> LLMRequest:
        return LLMRequest(
            prompt=(
                "Rewrite the following draft as the final answer only. "
                "Remove reasoning, planning, analysis, and chain-of-thought. "
                "Keep only the user-facing answer.\n\n"
                f"Original user request:\n{request.prompt}\n\n"
                f"Draft:\n{draft}"
            ),
            context=request.context,
            system=(
                "You are a response post-processor. "
                "Return only the cleaned final answer. "
                "Do not include reasoning, commentary, or meta text."
            ),
            max_tokens=min(
                request.max_tokens or self.default_max_tokens,
                128,
            ),
            temperature=min(
                request.temperature if request.temperature is not None else self.default_temperature,
                0.2,
            ),
            stop=request.stop,
            json_mode=request.json_mode,
            metadata={**request.metadata, "disable_reasoning_extractor": True},
        )

    def _generate_once(self, request: LLMRequest) -> tuple[str, Any, str]:
        final_prompt = self.build_prompt(request)
        max_tokens = request.max_tokens or self.default_max_tokens
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.default_temperature
        )
        driver = self._resolve_driver()
        if driver == "python":
            self._ensure_loaded()
            out = self._llm(
                final_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=request.stop,
            )
            text = out["choices"][0]["text"].strip()
        elif driver == "subprocess":
            cli_path = self._cli_resolved_path()
            if cli_path is None:
                raise RuntimeError("llama.cpp CLI runtime is not available")
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
            cli_prompt = self._subprocess_prompt(request)
            cmd = [
                cli_path,
                "-m",
                self.model_path,
                "-c",
                str(self.context_length),
                "-n",
                str(max_tokens),
                "--temp",
                str(temperature),
                "--simple-io",
                "--no-display-prompt",
                "--jinja",
                "-rea",
                "off",
                "--reasoning-budget",
                "0",
                "-st",
                "-p",
                cli_prompt,
            ]
            cmd.extend(["-sys", self._subprocess_system(request)])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.subprocess_timeout,
            )
            out = {"stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
            text = self._clean_subprocess_output(proc.stdout)
        else:
            raise RuntimeError(
                "No usable llama.cpp runtime found. Install llama-cpp-python or provide a llama.cpp CLI binary."
            )
        return text, out, driver

    def _clean_subprocess_output(self, text: str) -> str:
        cleaned = text.strip()
        if "\nmodel\n" in cleaned:
            cleaned = cleaned.rsplit("\nmodel\n", 1)[-1].strip()
        if cleaned.startswith("model\n"):
            cleaned = cleaned[len("model\n") :].strip()
        if cleaned.startswith("<|channel>final\n"):
            cleaned = cleaned[len("<|channel>final\n") :].strip()
        if "<|channel>thought\n" in cleaned and "<|channel>final\n" in cleaned:
            cleaned = cleaned.split("<|channel>final\n", 1)[-1].strip()
        elif cleaned.startswith("<|channel>thought\n"):
            cleaned = cleaned[len("<|channel>thought\n") :].strip()
            cleaned = re.sub(
                r"^Here's a thinking process that leads to the suggested concept:\s*",
                "",
                cleaned,
                count=1,
            ).strip()
        cleaned = re.sub(r"^<\|channel\>[a-z_]+\n", "", cleaned).strip()
        return cleaned

    def healthcheck(self) -> dict[str, object]:
        model_exists = Path(self.model_path).exists()
        binding_available = self._binding_available()
        cli_resolved_path = self._cli_resolved_path()
        driver = self._resolve_driver()
        return {
            "backend": self.backend_name,
            "model_family": self.model_family,
            "model_path": self.model_path,
            "exists": model_exists,
            "binding_available": binding_available,
            "cli_available": cli_resolved_path is not None,
            "cli_path": cli_resolved_path,
            "driver": driver,
            "ok": model_exists and driver in {"python", "subprocess"},
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        max_tokens = request.max_tokens or self.default_max_tokens
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.default_temperature
        )
        text, out, driver = self._generate_once(request)
        postprocess = {
            "reasoning_leak_detected": self._has_reasoning_leak(text),
            "extractor_applied": False,
            "extractor_error": None,
            "heuristic_fallback_applied": False,
            "heuristic_fallback_skipped": False,
            "heuristic_fallback_skip_reason": None,
            "heuristic_first_pass_applied": False,
        }
        if postprocess["reasoning_leak_detected"]:
            heuristic_candidate = self._heuristic_extract_final_answer(text)
            should_apply, skip_reason = self._should_apply_fallback(text, heuristic_candidate)
            if should_apply:
                text = heuristic_candidate
                postprocess["heuristic_fallback_applied"] = True
                postprocess["heuristic_first_pass_applied"] = True
                postprocess["reasoning_leak_detected"] = self._has_reasoning_leak(text)
            else:
                postprocess["heuristic_fallback_skipped"] = True
                postprocess["heuristic_fallback_skip_reason"] = skip_reason

        if postprocess["reasoning_leak_detected"] and not request.metadata.get("disable_reasoning_extractor"):
            extractor_request = self._extractor_request(request, text)
            try:
                extracted_text, extracted_out, _ = self._generate_once(extractor_request)
                if extracted_text:
                    text = extracted_text
                    out = {
                        "primary": out,
                        "extractor": extracted_out,
                    }
                    postprocess["extractor_applied"] = True
                    postprocess["reasoning_leak_detected"] = self._has_reasoning_leak(text)
                    if postprocess["reasoning_leak_detected"]:
                        fallback_text = self._heuristic_extract_final_answer(text)
                        should_apply, skip_reason = self._should_apply_fallback(text, fallback_text)
                        if should_apply:
                            text = fallback_text
                            postprocess["heuristic_fallback_applied"] = True
                            postprocess["reasoning_leak_detected"] = self._has_reasoning_leak(text)
                        else:
                            postprocess["heuristic_fallback_skipped"] = True
                            postprocess["heuristic_fallback_skip_reason"] = skip_reason
            except subprocess.TimeoutExpired as exc:
                postprocess["extractor_error"] = f"timeout: {exc.timeout}s"
            except subprocess.CalledProcessError as exc:
                postprocess["extractor_error"] = f"called_process_error: {exc.returncode}"
            except Exception as exc:
                postprocess["extractor_error"] = f"{type(exc).__name__}: {exc}"
        return LLMResponse(
            text=text,
            model_family=self.model_family,
            backend=self.backend_name,
            runtime={
                "model_path": self.model_path,
                "context_length": self.context_length,
                "n_gpu_layers": self.n_gpu_layers,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_format": self.chat_format,
                "driver": driver,
                "cli_path": self._cli_resolved_path(),
                "postprocess": postprocess,
            },
            raw=out,
        )
