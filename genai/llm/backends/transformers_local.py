"""Local safetensors inference in a short-lived worker process.

This experimental backend serves the small Clone LoRA export; it does not load
Transformers or Torch in the caller and never downloads model weights.
"""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from genai.llm.adapter import BaseLLMAdapter, LLMRequest, LLMResponse


class TransformersLocalAdapter(BaseLLMAdapter):
    def __init__(self, model_path, model_family="smollm2", model_id=None,
                 prompt_profile=None, lineage=None, max_tokens=64, timeout=180, threads=2):
        self.model_path = str(Path(model_path).resolve())
        self._family = model_family
        self.model_id = model_id
        self.prompt_profile = prompt_profile
        self.lineage = lineage or {}
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.threads = int(threads)

    @property
    def backend_name(self):
        return "transformers_local"

    @property
    def model_family(self):
        return self._family

    @classmethod
    def from_config(cls, cfg):
        return cls(**{key: cfg[key] for key in ("model_path", "model_family", "model_id", "prompt_profile", "lineage",
                                               "max_tokens", "timeout", "threads") if key in cfg})

    def healthcheck(self):
        path = Path(self.model_path)
        missing = [name for name in ("config.json", "model.safetensors", "tokenizer_config.json") if not (path / name).is_file()]
        dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers")}
        return {"backend": self.backend_name, "model_family": self.model_family, "model_path": self.model_path,
                "missing_files": missing, "dependencies": dependencies, "driver": "subprocess",
                "ok": not missing and all(dependencies.values())}

    def _effective_system(self, request):
        system = request.system or ""
        if self.prompt_profile == "clone" and not request.metadata.get("disable_prompt_profile"):
            # Preserve a caller's complete Clone contract, including actual persona values.
            if not all(marker in system for marker in ("tone=", "principles=", "response=")):
                system += "\nUse the persona from the system message and this format: [Name] tone=[Tone] principles=[Principles] response=[Answer]"
        return system.strip()

    def build_prompt(self, request):
        return "System:\n" + self._effective_system(request) + "\nMemory:\n" + (request.context or "") + "\nQuestion:\n" + request.prompt

    def generate(self, request: LLMRequest):
        if request.json_mode:
            raise ValueError("transformers_local tiny Clone backend does not implement JSON mode")
        if request.temperature not in (None, 0, 0.0):
            raise ValueError("transformers_local tiny Clone backend supports deterministic temperature=0 only")
        health = self.healthcheck()
        if not health["ok"]:
            raise RuntimeError(f"Local Transformers model is unavailable: {health}")
        tokens = request.max_tokens if request.max_tokens is not None else self.max_tokens
        if not 1 <= tokens <= 512:
            raise ValueError("max_tokens must be between 1 and 512")
        payload = {"model_path": self.model_path, "threads": self.threads, "max_tokens": tokens,
                   "request": {**asdict(request), "system": self._effective_system(request)}}
        command = [sys.executable, "-m", "genai.llm.backends.transformers_worker"]
        completed = subprocess.run(command, input=json.dumps(payload), text=True,
                                   capture_output=True, check=True, timeout=self.timeout)
        result = json.loads(completed.stdout)
        text = result["text"]
        stop_applied = None
        for stop in request.stop or []:
            if stop and stop in text:
                text = text.split(stop, 1)[0]
                stop_applied = stop
        persisted_path = Path(self.model_path).parent / "lineage.json"
        persisted = json.loads(persisted_path.read_text()) if persisted_path.exists() else {}
        return LLMResponse(text=text, model_family=self.model_family, backend=self.backend_name,
            runtime={"model_id": self.model_id, "model_path": self.model_path, "driver": "subprocess",
                     "device": "cpu", "dtype": "float32", "temperature": 0.0, "max_tokens": tokens,
                     "prompt_profile": self.prompt_profile,
                     "prompt_profile_applied": self.prompt_profile == "clone" and not request.metadata.get("disable_prompt_profile", False),
                     "lineage": {**self.lineage, **persisted}, "worker": result["runtime"], "stop_applied": stop_applied},
            raw={"generation": result["text"]})
