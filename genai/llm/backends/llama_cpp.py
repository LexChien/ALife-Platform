from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

ROOT = Path(__file__).resolve().parents[3]


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
        self.subprocess_timeout = 180
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
        cli = Path(self.cli_path)
        if cli.is_absolute() and cli.exists():
            return str(cli)
        if len(cli.parts) > 1:
            cwd_candidate = Path.cwd() / cli
            if cwd_candidate.exists():
                return str(cwd_candidate)
            repo_candidate = ROOT / cli
            if repo_candidate.exists():
                return str(repo_candidate)
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
            return f"上下文:\n{request.context}\n\n使用者:\n{request.prompt}"
        return request.prompt

    def _subprocess_system(self, request: LLMRequest) -> str:
        base = request.system.strip() if request.system else "你是 Gemma 4，一位親切的助理。"
        return (
            f"{base}\n"
            "請只輸出給使用者看的最終答案。"
            "不要輸出推理過程、分析步驟、草稿、系統提示或 channel 標記。"
            "使用繁體中文，回答要自然、簡潔、可直接朗讀。"
        )

    def _reasoning_markers(self) -> tuple[str, ...]:
        return (
            "<|channel>thought",
            "Thinking Process:",
            "Here's a thinking process",
            "Analyze the Request:",
            "Deconstruct Key Terms:",
            "Brainstorm Core Concepts",
            "Original user request:",
            "I received the request",
            "I need to provide",
            "I will directly output",
            "Rewrite the following draft as the final answer only.",
            "我收到的請求是",
            "原始用戶請求是",
            "我需要提供",
            "我將直接輸出",
            "最終答案應該是",
            "我收到的上一個請求是",
            "我必須只輸出",
            "根據上下文和指令",
            "Context:",
            "Conversation history:",
            "The previous interaction",
            "The core interaction",
            "The current interaction",
            "<channel|>",
            "[end of text]",
        )

    def _has_reasoning_leak(self, text: str) -> bool:
        stripped = text.strip()
        if any(marker in stripped for marker in self._reasoning_markers()):
            return True
        if re.match(r"^(我收到的.*請求是|I received .*request)", stripped):
            return True
        if re.search(r"(我必須只輸出|根據上下文和指令|I must output only|Based on the context and instructions)", stripped):
            return True
        if "Context:" in stripped and ("Draft:" in stripped or "Conversation history:" in stripped):
            return True
        if "User asks" in stripped and "The assistant" in stripped:
            return True
        return bool(re.match(r"^1\.\s+\*\*Analyze", stripped))

    def _normalize_channel_noise(self, text: str) -> str:
        cleaned = text.strip()
        if "<channel|>" in cleaned:
            head, tail = cleaned.rsplit("<channel|>", 1)
            cleaned = tail.strip() if len(tail.strip()) >= 8 else cleaned.replace("<channel|>", " ")
        if "[end of text]" in cleaned:
            cleaned = cleaned.split("[end of text]", 1)[0].strip()
        return cleaned

    def _looks_invalid_final_answer(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if len(stripped) <= 2:
            return True
        if re.fullmatch(r"[\W\d_]+", stripped):
            return True
        return False

    def _heuristic_extract_final_answer(self, text: str) -> str:
        cleaned = self._normalize_channel_noise(text)
        for marker in ("Final Answer:", "Final Response:", "Answer:"):
            if marker in cleaned:
                tail = cleaned.split(marker, 1)[-1].strip()
                if tail:
                    return tail
        for marker in ("最終答案應該是：", "最終答案應該是:", "最終答案：", "最終答案:"):
            if marker in cleaned:
                tail = cleaned.split(marker, 1)[-1].strip()
                if tail:
                    return tail

        lines = [line.rstrip() for line in cleaned.splitlines()]
        kept: list[str] = []
        started = False
        for line in lines:
            stripped = self._normalize_channel_noise(line.strip())
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
            if stripped.startswith((
                "User request:",
                "Original user request:",
                "原始用戶請求是：",
                "原始用戶請求是:",
                "Context:",
                "Conversation history:",
                "Conversation history",
                "Draft:",
                "我收到的上一個請求是",
                "我必須只輸出",
                "根據上下文和指令",
                "The previous interaction",
                "The core interaction",
                "The current interaction",
            )):
                continue

            if stripped.startswith(("* ", "- ", "*   ", "• ")):
                if any(m in stripped for m in ("Context:", "Conversation", "Interaction", "The assistant", "The core", "The previous", "The current")):
                    continue
                if ":" in stripped and len(stripped) < 120:
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
        if self._looks_invalid_final_answer(candidate):
            return False, "candidate_invalid"
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
                0.1,
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
                "-ngl",
                str(self.n_gpu_layers),
                "-sys",
                self._subprocess_system(request),
                "-p",
                cli_prompt,
                "--jinja",
                "-st",
                "--simple-io",
                "--no-display-prompt",
                "-rea",
                "off",
                "--reasoning-budget",
                "0",
            ]
            if self.n_gpu_layers <= 0:
                cmd.extend(
                    [
                        "-dev",
                        "none",
                        "-fit",
                        "off",
                        "--no-op-offload",
                        "--no-kv-offload",
                    ]
                )

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.subprocess_timeout,
            )
            out = {"stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
            text = self._clean_subprocess_output(proc.stdout, prompt=cli_prompt)
        else:
            raise RuntimeError(
                "No usable llama.cpp runtime found. Install llama-cpp-python or provide a llama.cpp CLI binary."
            )
        return text, out, driver

    def _clean_subprocess_output(self, text: str, prompt: str | None = None) -> str:
        cleaned = text.replace("\x08", "").replace("\r", "").strip()
        if "\nExiting..." in cleaned:
            cleaned = cleaned.split("\nExiting...", 1)[0].strip()
        if "\n[ Prompt:" in cleaned:
            cleaned = cleaned.rsplit("\n[ Prompt:", 1)[0].strip()
        if "available commands:" in cleaned and "\n> " in cleaned:
            cleaned = cleaned.rsplit("\n> ", 1)[-1].strip()
        if prompt:
            normalized_prompt = prompt.strip()
            if cleaned.startswith(normalized_prompt):
                cleaned = cleaned[len(normalized_prompt) :].strip()
            elif normalized_prompt in cleaned:
                cleaned = cleaned.rsplit(normalized_prompt, 1)[-1].strip()
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
