"""Continuous overdamped particles under explicit phase control.

This is a controlled mechanics baseline, NOT autonomous cell reproduction.
Frames rasterize conserved particles: no phase-specific sprites, component
removal, particle respawning, or trajectory interpolation.
"""
import numpy as np
from PIL import Image


class ControlledCells:
    def __init__(self, num_particles=120, width=128, height=128, blur_sigma=2.8):
        if num_particles < 16 or min(width, height) < 32 or blur_sigma <= 0:
            raise ValueError("ControlledCells needs >=16 particles and >=32 pixel dimensions")
        self.num_particles, self.width, self.height = int(num_particles), int(width), int(height)
        self.blur_sigma = float(blur_sigma)
        self.phases = []
        self.center = np.array([width / 2, height / 2], dtype=float)
        fy = np.fft.fftfreq(self.height)[:, None]
        fx = np.fft.rfftfreq(self.width)[None, :]
        self.blur_fft = np.exp(-2 * np.pi ** 2 * self.blur_sigma ** 2 * (fx ** 2 + fy ** 2))
        self.reset([0.18, 0.8, 0.06, 25.0, 0.015], seed=0)

    def configure_narrative(self, total_steps, phases):
        self.phases = list(phases)
        self.total_steps = int(total_steps)

    def reset(self, theta, seed=None):
        values = np.asarray(theta, dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("ControlledCells theta needs five finite nonnegative values")
        self.mobility, self.repulsion, self.cohesion, self.split_distance, self.temperature = values
        self.rng = np.random.RandomState(seed)
        angle = self.rng.uniform(0, 2 * np.pi, self.num_particles)
        radius = np.sqrt(self.rng.uniform(0, 1, self.num_particles)) * 9
        self.positions = self.center + np.column_stack((np.cos(angle), np.sin(angle))) * radius[:, None]
        self.cohorts = np.ones(self.num_particles)
        self.cohorts[np.argsort(self.positions[:, 0])[:self.num_particles // 2]] = -1
        self.current_step = 0
        self.last_displacement = np.zeros_like(self.positions)

    def step(self, substeps=1):
        phase = next((p["name"] for p in self.phases
                      if p["frame_range"][0] <= self.current_step <= p["frame_range"][1]), "birth")
        offset = self.split_distance if phase == "split" else 0.0
        anchors = np.repeat(self.center[None, :], self.num_particles, axis=0)
        anchors[:, 0] += self.cohorts * offset
        # A frame advances one physical time unit; substeps refine integration.
        dt = 1.0 / substeps
        for _ in range(substeps):
            delta = self.positions[:, None, :] - self.positions[None, :, :]
            distance_squared = (delta * delta).sum(axis=-1)
            np.fill_diagonal(distance_squared, np.inf)
            neighbours = distance_squared < 36
            repulsive = (delta * (neighbours / np.maximum(distance_squared, 0.7))[..., None]).sum(axis=1)
            cohesive = np.zeros_like(self.positions)
            for cohort in (-1, 1):
                selected = self.cohorts == cohort
                cohesive[selected] = self.positions[selected].mean(axis=0) - self.positions[selected]
            force = (self.mobility * (anchors - self.positions)
                     + self.repulsion * repulsive + self.cohesion * cohesive)
            force += self.rng.normal(0, self.temperature, force.shape)
            speed = np.linalg.norm(force, axis=1, keepdims=True)
            displacement = force * np.minimum(1.0, 1.5 / np.maximum(speed, 1e-9)) * dt
            self.positions += displacement
            self.positions = np.clip(self.positions, [2, 2], [self.width - 3, self.height - 3])
            self.last_displacement = displacement
        self.current_step += 1

    def density(self):
        """Mass-preserving bilinear deposition and Gaussian convolution."""
        grid = np.zeros((self.height, self.width), dtype=float)
        integer = np.floor(self.positions).astype(int)
        fraction = self.positions - integer
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (fraction[:, 1] if dy else 1 - fraction[:, 1])
            np.add.at(grid, ((integer[:, 1] + dy) % self.height,
                             (integer[:, 0] + dx) % self.width), weight)
        return np.maximum(0, np.fft.irfft2(np.fft.rfft2(grid) * self.blur_fft, s=grid.shape))

    def render(self):
        # Fixed transfer function; dim particles are retained, never culled.
        body = 1 - np.exp(-self.density() * 6)
        rgb = np.stack((0.68 * body, 0.91 * body, body), axis=-1)
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))

    def stats(self):
        return {"particle_count": self.num_particles, "raster_mass": float(self.density().sum()),
                "mean_speed": float(np.linalg.norm(self.last_displacement, axis=1).mean()),
                "control": "explicit_phase_external_potential", "autonomous_reproduction": False}
