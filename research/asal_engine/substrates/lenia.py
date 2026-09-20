"""Single-channel Lenia with FFT ring convolution and Gaussian growth.

Reference: Chan, Lenia — Biology of Artificial Life, arXiv:1812.05433.
The default random patch is not a curated persistent Lenia species.
"""
import numpy as np
from PIL import Image


class Lenia:
    def __init__(self, size=128):
        if size < 8:
            raise ValueError("Lenia size must be >= 8")
        self.size = int(size)
        self.reset([0.15, 0.035, 0.1, 0.15, 0.5], seed=0)

    def reset(self, theta, seed=None):
        parameters = np.asarray(theta, dtype=float)
        if parameters.shape != (5,) or not np.isfinite(parameters).all():
            raise ValueError("Lenia theta must contain five finite values")
        # mu, sigma, time step, radius / lattice size, initial density
        self.mu, self.sigma, self.dt, radius_fraction, self.density = np.clip(
            parameters, [0.01, 0.005, 0.001, 0.03, 0.05], [0.95, 0.3, 0.2, 0.4, 1]
        )
        radius = max(2.0, radius_fraction * self.size)
        yy, xx = np.mgrid[:self.size, :self.size]
        distance = np.hypot(xx - self.size // 2, yy - self.size // 2) / radius
        kernel = np.exp(-0.5 * ((distance - 0.5) / 0.15) ** 2) * (distance < 1)
        self.kernel = kernel / kernel.sum()
        self.kernel_fft = np.fft.rfft2(np.fft.ifftshift(self.kernel))
        rng = np.random.RandomState(seed)
        patch = np.hypot(xx - self.size // 2, yy - self.size // 2) < radius * 1.4
        self.grid = (patch * rng.uniform(0.4, 1, (self.size, self.size)) * self.density).astype(np.float64)

    def potential(self):
        return np.fft.irfft2(np.fft.rfft2(self.grid) * self.kernel_fft, s=self.grid.shape).real

    def growth(self, potential):
        return 2 * np.exp(-0.5 * ((potential - self.mu) / self.sigma) ** 2) - 1

    def step(self, substeps=1):
        for _ in range(substeps):
            self.grid = np.clip(self.grid + self.dt * self.growth(self.potential()), 0, 1)

    def render(self):
        rgb = np.stack((self.grid * 0.68, self.grid * 0.90, self.grid), axis=-1)
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))

    def stats(self):
        return {"mean_activity": float(self.grid.mean()), "mass": float(self.grid.sum()),
                "active_cells": int((self.grid > 0.01).sum()),
                "rule": "fft_ring_kernel_gaussian_growth"}
