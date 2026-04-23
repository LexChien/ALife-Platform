import numpy as np
from PIL import Image


class Boids:
    def __init__(
        self,
        num_boids=100,
        width=128,
        height=128,
        keep_largest_component=True,
        narrative_controller=None,
    ):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        self.keep_largest_component = keep_largest_component
        self.narrative_controller = narrative_controller or {}
        self.total_steps = None
        self.phase_specs = []
        self.current_step = 0
        self._cohort_sign = None
        self.reset([1.0, 1.0, 1.0, 5.0, 2.0])

    def reset(self, theta):
        self.base_separation, self.base_alignment, self.base_cohesion, self.base_speed, self.base_view_radius = theta
        self.separation = self.base_separation
        self.alignment = self.base_alignment
        self.cohesion = self.base_cohesion
        self.speed = self.base_speed
        self.view_radius = self.base_view_radius
        seed_mode = self.narrative_controller.get("seed_mode", "uniform") if self.narrative_controller.get("enabled", False) else "uniform"
        if seed_mode == "center_cluster":
            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            seed_spread = float(self.narrative_controller.get("seed_spread", min(self.width, self.height) * 0.06))
            self.positions = center[None, :] + np.random.randn(self.num_boids, 2) * seed_spread
            self.positions[:, 0] %= self.width
            self.positions[:, 1] %= self.height
        else:
            self.positions = np.random.rand(self.num_boids, 2) * [self.width, self.height]
        angles = np.random.rand(self.num_boids) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.speed
        if seed_mode == "center_cluster":
            initial_speed_scale = float(self.narrative_controller.get("initial_speed_scale", 0.35))
            self.velocities *= initial_speed_scale
        half = self.num_boids // 2
        self._cohort_sign = np.ones((self.num_boids, 1), dtype=np.float32)
        self._cohort_sign[:half] = -1.0
        self.current_step = 0

    def configure_narrative(self, total_steps, phases):
        self.total_steps = int(total_steps)
        self.phase_specs = list(phases or [])

    def _active_phase_control(self):
        controller = self.narrative_controller or {}
        if not controller.get("enabled", False) or not self.phase_specs:
            return {}
        for phase_cfg in self.phase_specs:
            start, end = phase_cfg["frame_range"]
            if int(start) <= self.current_step <= int(end):
                name = phase_cfg["name"]
                for control in controller.get("phases", []):
                    if control.get("name") == name:
                        return control
                return {}
        return {}

    def step(self, substeps=1):
        for _ in range(substeps):
            phase_control = self._active_phase_control()
            separation_gain = self.base_separation * float(phase_control.get("separation_scale", 1.0))
            alignment_gain = self.base_alignment * float(phase_control.get("alignment_scale", 1.0))
            cohesion_gain = self.base_cohesion * float(phase_control.get("cohesion_scale", 1.0))
            speed = self.base_speed * float(phase_control.get("speed_scale", 1.0))
            view_radius = self.base_view_radius * float(phase_control.get("view_radius_scale", 1.0))
            center_pull = float(phase_control.get("center_pull", self.narrative_controller.get("base_center_pull", 0.0)))
            damping = float(phase_control.get("damping", self.narrative_controller.get("base_velocity_damping", 0.0)))
            split_push = float(phase_control.get("split_push", 0.0))
            cohort_pull = float(phase_control.get("cohort_pull", 0.0))
            split_offset = float(phase_control.get("split_offset", 0.0))
            split_axis = self.narrative_controller.get("split_axis", "horizontal")
            axis_vec = np.array([1.0, 0.0], dtype=np.float32) if split_axis == "horizontal" else np.array([0.0, 1.0], dtype=np.float32)

            dx = self.positions[:, 0:1] - self.positions[:, 0:1].T
            dy = self.positions[:, 1:2] - self.positions[:, 1:2].T

            dx = dx - self.width * np.round(dx / self.width)
            dy = dy - self.height * np.round(dy / self.height)

            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)

            mask = dist < view_radius
            counts = mask.sum(axis=1)[:, None]
            counts_safe = np.where(counts == 0, 1, counts)

            sep_x = np.where(mask & (dist > 0), dx / dist**2, 0).sum(axis=1)
            sep_y = np.where(mask & (dist > 0), dy / dist**2, 0).sum(axis=1)
            separation = np.stack([sep_x, sep_y], axis=1) * separation_gain

            align_x = np.where(mask, self.velocities[:, 0], 0).sum(axis=1)
            align_y = np.where(mask, self.velocities[:, 1], 0).sum(axis=1)
            alignment = np.stack([align_x, align_y], axis=1) / counts_safe - self.velocities
            alignment = np.where(counts > 0, alignment, 0) * alignment_gain

            local_center_x = np.where(mask, self.positions[:, 0], 0).sum(axis=1)
            local_center_y = np.where(mask, self.positions[:, 1], 0).sum(axis=1)
            local_center = np.stack([local_center_x, local_center_y], axis=1) / counts_safe

            coh_dx = local_center[:, 0] - self.positions[:, 0]
            coh_dy = local_center[:, 1] - self.positions[:, 1]
            coh_dx = coh_dx - self.width * np.round(coh_dx / self.width)
            coh_dy = coh_dy - self.height * np.round(coh_dy / self.height)

            cohesion = np.stack([coh_dx, coh_dy], axis=1)
            cohesion = np.where(counts > 0, cohesion, 0) * cohesion_gain

            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            to_center = center[None, :] - self.positions
            center_force = to_center * center_pull

            split_force = self._cohort_sign * axis_vec[None, :] * split_push
            cohort_anchor = center[None, :] + self._cohort_sign * axis_vec[None, :] * split_offset
            cohort_force = (cohort_anchor - self.positions) * cohort_pull

            self.velocities += separation + alignment + cohesion + center_force + split_force + cohort_force
            if damping > 0.0:
                self.velocities *= max(0.0, 1.0 - damping)

            speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
            speeds[speeds == 0] = 1.0
            self.velocities = (self.velocities / speeds) * speed

            self.positions += self.velocities
            self.positions[:, 0] %= self.width
            self.positions[:, 1] %= self.height
            self.current_step += 1

    def render(self):
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        density = np.zeros((self.height, self.width), dtype=np.float32)
        sigma = max(self.base_view_radius * 0.6, 1.5)
        norm = 2.0 * sigma * sigma

        for x, y in self.positions:
            dx = xx - x
            dy = yy - y
            density += np.exp(-(dx * dx + dy * dy) / norm)

        density = density / (density.max() + 1e-9)
        if self.keep_largest_component:
            density = self._keep_largest_component(density)
        gy, gx = np.gradient(density)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-9)

        body = np.clip(density ** 0.8, 0.0, 1.0)
        halo = np.clip(body + 0.55 * edge, 0.0, 1.0)
        shadow = np.clip(body - 0.22 * edge, 0.0, 1.0)
        img = np.stack([halo, body, shadow], axis=-1)
        return Image.fromarray((img * 255).astype(np.uint8))

    def _keep_largest_component(self, density: np.ndarray) -> np.ndarray:
        mask = density > max(0.18, float(density.mean() + 0.35 * density.std()))
        if not mask.any():
            return density

        visited = np.zeros_like(mask, dtype=bool)
        best_component: list[tuple[int, int]] = []
        h, w = mask.shape

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                component: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(component) > len(best_component):
                    best_component = component

        if not best_component:
            return density

        keep = np.zeros_like(density, dtype=np.float32)
        for y, x in best_component:
            keep[y, x] = density[y, x]

        yy, xx = np.mgrid[0:h, 0:w]
        coords = np.array(best_component, dtype=np.float32)
        cy, cx = coords.mean(axis=0)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        radius2 = max(len(best_component) * 0.35, 16.0)
        soft_mask = np.exp(-dist2 / (2.0 * radius2))
        filtered = np.maximum(keep, density * soft_mask * 0.35)
        return filtered / (filtered.max() + 1e-9)

    def stats(self):
        speeds = np.linalg.norm(self.velocities, axis=1)
        return {"avg_speed": float(speeds.mean())}
