import hashlib
import numpy as np
from . import foundation_models

@foundation_models.register("tiny_vlm_stub")
class TinyVLMStub:
    """
    A platform-level placeholder for a future TinyVLM backend.
    Keeps the interface stable so ASAL / Digital Clone / GenAI can share it.
    """
    def __init__(self, dim=128):
        self.dim = dim

    @property
    def device(self):
        return "cpu"

    def img_embed(self, pil_img):
        arr = np.asarray(pil_img).astype(np.float32)
        rs = np.random.RandomState(int(arr.mean() * 1000) % (2**16))
        v = rs.randn(self.dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def txt_embed(self, text: str):
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rs = np.random.RandomState(h)
        v = rs.randn(self.dim)
        return v / (np.linalg.norm(v) + 1e-9)
