import hashlib
import numpy as np
from . import foundation_models

@foundation_models.register("random")
class RandomEmbedder:
    def __init__(self, dim=64, seed=0):
        self.dim = dim
        self.seed = seed

    @property
    def device(self):
        return "cpu"

    def img_embed(self, pil_img):
        arr = np.asarray(pil_img).astype(np.float32).reshape(-1)
        rs = np.random.RandomState((len(arr) + self.seed) % (2**16))
        proj = rs.randn(arr.size, self.dim) / max(arr.size, 1) ** 0.5
        v = arr @ proj
        return v / (np.linalg.norm(v) + 1e-9)

    def txt_embed(self, text: str):
        h = int(hashlib.sha256((text + str(self.seed)).encode("utf-8")).hexdigest(), 16) % (2**32)
        rs = np.random.RandomState(h)
        v = rs.randn(self.dim)
        return v / (np.linalg.norm(v) + 1e-9)
