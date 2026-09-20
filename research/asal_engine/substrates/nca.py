"""Local neural CA baseline, with untrained weights and a diffusion/growth prior.

Sobel perception is based on https://distill.pub/2020/growing-ca/.
The default rule is executable, but it is not a pretrained organism generator.
"""
from pathlib import Path

import numpy as np
from PIL import Image


class NCA:
    def __init__(self, size=128, channels=16, hidden=24, rule_seed=0, weights_path=None):
        if size < 8 or channels < 4 or hidden < 1:
            raise ValueError("NCA requires size >= 8, channels >= 4, hidden >= 1")
        self.size, self.channels, self.hidden = int(size), int(channels), int(hidden)
        self.weights_loaded = weights_path is not None
        rules = np.random.RandomState(rule_seed)
        self.w1 = (rules.normal(size=(4 * channels, hidden)) / np.sqrt(4 * channels)).astype(np.float32)
        self.w2 = (rules.normal(size=(hidden, channels)) / np.sqrt(hidden)).astype(np.float32)
        if weights_path is not None:
            with np.load(Path(weights_path), allow_pickle=False) as weights:
                for name in ("w1", "w2"):
                    values = weights[name]
                    if values.shape != getattr(self, name).shape or not np.isfinite(values).all():
                        raise ValueError(f"Invalid NCA weight matrix {name}")
                    setattr(self, name, values.astype(np.float32))
        self.reset([0.16, 0.06, 0.01, 0.025, 0.8], seed=0)

    def reset(self, theta, seed=None):
        parameters = np.asarray(theta, dtype=float)
        if parameters.shape != (5,) or not np.isfinite(parameters).all():
            raise ValueError("NCA theta must contain five finite values")
        self.diffusion, self.growth, self.decay, self.neural_gain, self.fire_rate = np.clip(
            parameters, [0, 0, 0, 0, 0.01], [0.24, 0.3, 0.3, 0.2, 1]
        )
        self.rng = np.random.RandomState(seed)
        self.grid = np.zeros((self.size, self.size, self.channels), dtype=np.float32)
        yy, xx = np.mgrid[:self.size, :self.size]
        seed_mask = (xx - self.size // 2) ** 2 + (yy - self.size // 2) ** 2 <= 4
        self.grid[seed_mask] = self.rng.uniform(0.45, 0.65, (seed_mask.sum(), self.channels))
        self.grid[..., 3][seed_mask] = 0.8

    def perceive(self):
        """Identity, Sobel-x/y, and five-point Laplacian; periodic boundaries."""
        grid = self.grid
        north, south = np.roll(grid, 1, 0), np.roll(grid, -1, 0)
        east, west = np.roll(grid, -1, 1), np.roll(grid, 1, 1)
        sx = (np.roll(north, -1, 1) + 2 * east + np.roll(south, -1, 1)
              - np.roll(north, 1, 1) - 2 * west - np.roll(south, 1, 1)) / 8
        sy = (np.roll(south, 1, 1) + 2 * south + np.roll(south, -1, 1)
              - np.roll(north, 1, 1) - 2 * north - np.roll(north, -1, 1)) / 8
        laplacian = north + south + east + west - 4 * grid
        return np.concatenate((grid, sx, sy, laplacian), axis=-1), laplacian

    def step(self, substeps=1):
        for _ in range(substeps):
            perception, laplacian = self.perceive()
            hidden = np.tanh(perception @ self.w1)
            neural = np.tanh(hidden @ self.w2)
            # No additive spatial noise: an empty neighbourhood stays empty.
            update = (self.diffusion * laplacian
                      + self.growth * self.grid * (1 - self.grid)
                      - self.decay * self.grid + self.neural_gain * neural)
            fired = self.rng.rand(self.size, self.size, 1) < self.fire_rate
            self.grid = np.clip(self.grid + fired * update, 0, 1).astype(np.float32)

    def render(self):
        rgb = np.clip(self.grid[..., :3] * self.grid[..., 3:4], 0, 1)
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))

    def save_weights(self, path):
        np.savez_compressed(path, w1=self.w1, w2=self.w2)

    def stats(self):
        return {"mean_cell_mass": float(self.grid[..., 3].mean()),
                "active_cells": int((self.grid[..., 3] > 0.01).sum()),
                "rule": "local_sobel_mlp_with_diffusion_growth_prior",
                "weights_loaded": self.weights_loaded}
