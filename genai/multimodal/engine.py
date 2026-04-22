from pathlib import Path

from genai.llm.adapter import LLMRequest
from genai.llm.factory import create_llm_adapter
from genai.image.adapter import DummyImageAdapter
from genai.voice.adapter import DummyVoiceAdapter
from core.artifacts import save_image
from core.logger import save_json
import numpy as np

class GenAIEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm = create_llm_adapter(config)
        self.img = DummyImageAdapter()
        self.voice = DummyVoiceAdapter()

    def run(self):
        prompt = self.config["prompt"]
        context = self.config.get("context")
        system = self.config.get("system")
        llm_cfg = self.config.get("llm", {})
        health = self.llm.healthcheck()
        request = LLMRequest(
            prompt=prompt,
            context=context,
            system=system,
            max_tokens=llm_cfg.get("max_tokens"),
            temperature=llm_cfg.get("temperature"),
            stop=llm_cfg.get("stop"),
            json_mode=llm_cfg.get("json_mode", False),
        )
        response = self.llm.generate(request)
        text = response.text
        image = self.img.generate(prompt)
        audio = self.voice.synthesize(text)
        save_image(self.run_dir / "generated.png", np.asarray(image))
        output = {
            "text": text,
            "audio": audio,
            "prompt": prompt,
            "context": context,
            "system": system,
            "llm": {
                "backend": response.backend,
                "model_family": response.model_family,
                "runtime": response.runtime,
                "healthcheck": health,
            },
        }
        save_json(self.run_dir / "genai_output.json", output)
        return {
            "output": output,
            "summary": {
                "mode": "genai_multimodal_generation",
                "prompt": prompt,
                "has_context": context is not None,
                "text_length": len(text),
                "has_audio": bool(audio),
                "has_image": True,
                "llm_backend": response.backend,
                "llm_model_family": response.model_family,
                "llm_runtime": response.runtime,
                "llm_healthcheck": health,
            },
        }
