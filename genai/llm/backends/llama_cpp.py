from __future__ import annotations

from pathlib import Path
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

    def _clean_subprocess_output(self, text: str) -> str:
        cleaned = text.strip()
        if "\nmodel\n" in cleaned:
            cleaned = cleaned.rsplit("\nmodel\n", 1)[-1].strip()
        if cleaned.startswith("model\n"):
            cleaned = cleaned[len("model\n") :].strip()
        if cleaned.startswith("<|channel>final\n"):
            cleaned = cleaned[len("<|channel>final\n") :].strip()
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
                "-st",
                "-p",
                cli_prompt,
            ]
            if request.system:
                cmd.extend(["-sys", request.system])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            out = {"stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
            text = self._clean_subprocess_output(proc.stdout)
        else:
            raise RuntimeError(
                "No usable llama.cpp runtime found. Install llama-cpp-python or provide a llama.cpp CLI binary."
            )
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
            },
            raw=out,
        )
