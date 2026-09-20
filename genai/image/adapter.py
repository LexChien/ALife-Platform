import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_text(text: str) -> str:
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", errors="replace").decode("ascii")

class DummyImageAdapter:
    metadata = {"status": "stub", "backend": "dummy"}

    def generate(self, prompt, size=256):
        img = Image.new("RGB", (size, size), (24, 24, 24))
        d = ImageDraw.Draw(img)
        font = _load_font(16)
        d.rectangle([24, 24, size-24, size-24], outline=(160, 220, 255), width=3)
        text = prompt[:24]
        try:
            d.text((16, 16), text, fill=(230, 230, 230), font=font)
        except UnicodeEncodeError:
            d.text((16, 16), _safe_text(text), fill=(230, 230, 230), font=font)
        return img


class DiffusionImageAdapter:
    def __init__(self, **config):
        self.config = config
        self.metadata = {}

    def generate(self, prompt, size=256, output=None):
        if output is None:
            raise ValueError("A persistent image output path is required")
        if size < 64 or size > 1024 or size % 8:
            raise ValueError("Image size must be a multiple of 8 between 64 and 1024")
        request = {"config": self.config, "prompt": prompt, "size": size,
                   "output": str(Path(output).resolve())}
        proc = subprocess.run([sys.executable, "-m", "genai.image.diffusion_worker"],
                              input=json.dumps(request), text=True, capture_output=True,
                              timeout=int(self.config.get("timeout", 1800)))
        if proc.returncode:
            raise RuntimeError(f"Image generation failed: {proc.stderr[-4000:]}")
        self.metadata = json.loads(proc.stdout)
        with Image.open(output) as image:
            return image.convert("RGB")


def create_image_adapter(config):
    cfg = dict(config or {})
    backend = cfg.pop("backend", "dummy")
    cfg.pop("size", None)
    if backend == "dummy":
        return DummyImageAdapter()
    if backend == "diffusers":
        return DiffusionImageAdapter(**cfg)
    raise ValueError(f"Unknown image backend: {backend}")
