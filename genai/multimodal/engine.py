from genai.llm.adapter import DummyLLMAdapter
from genai.image.adapter import DummyImageAdapter
from genai.voice.adapter import DummyVoiceAdapter
from core.artifacts import save_image
from core.logger import save_json
import numpy as np

class GenAIEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.llm = DummyLLMAdapter()
        self.img = DummyImageAdapter()
        self.voice = DummyVoiceAdapter()

    def run(self):
        prompt = self.config["prompt"]
        context = self.config.get("context")
        text = self.llm.generate(prompt, context)
        image = self.img.generate(prompt)
        audio = self.voice.synthesize(text)
        save_image(self.run_dir / "generated.png", np.asarray(image))
        save_json(self.run_dir / "genai_output.json", {"text": text, "audio": audio, "prompt": prompt})
