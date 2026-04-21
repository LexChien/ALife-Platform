import numpy as np
from PIL import Image

class NCA:
    def __init__(self, size=128, channels=16):
        self.size = size
        self.channels = channels
        self.reset([1.0, 1.0, 1.0, 1.0, 1.0])

    def reset(self, theta):
        self.theta = theta
        self.grid = np.zeros((self.size, self.size, self.channels), dtype=np.float32)
        c = self.size // 2
        self.grid[c, c, :] = 1.0

    def step(self, substeps=1):
        for _ in range(substeps):
            # Stub NCA logic
            self.grid = np.clip(self.grid + 0.01 * np.random.randn(*self.grid.shape), 0, 1)

    def render(self):
        # Render the first 3 channels as RGB
        img = self.grid[..., :3]
        return Image.fromarray((img * 255).astype(np.uint8))

    def stats(self):
        return {"mean_cell_mass": float(self.grid.mean())}
