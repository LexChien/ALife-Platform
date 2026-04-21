import numpy as np
from PIL import Image

class Lenia:
    def __init__(self, size=128):
        self.size = size
        self.reset([1.0, 1.0, 1.0, 1.0, 1.0])

    def reset(self, theta):
        self.theta = theta
        self.grid = np.random.rand(self.size, self.size)

    def step(self, substeps=1):
        for _ in range(substeps):
            # Stub Lenia logic
            self.grid = (self.grid + 0.01) % 1.0

    def render(self):
        img = np.stack([self.grid]*3, axis=-1)
        return Image.fromarray((img * 255).astype(np.uint8))

    def stats(self):
        return {"mean_activity": float(self.grid.mean())}
