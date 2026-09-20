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
        self._cohort_id = None
        self._role = None  # 0=normal tissue, 1=malignant
        self._conversion_events = 0
        self._rng = np.random.RandomState(0)
        self.reset([1.0, 1.0, 1.0, 5.0, 2.0])

    def reset(self, theta, seed=None):
        self.base_separation, self.base_alignment, self.base_cohesion, self.base_speed, self.base_view_radius = theta
        self.separation = self.base_separation
        self.alignment = self.base_alignment
        self.cohesion = self.base_cohesion
        self.speed = self.base_speed
        self.view_radius = self.base_view_radius
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        seed_mode = self.narrative_controller.get("seed_mode", "uniform") if self.narrative_controller.get("enabled", False) else "uniform"
        if seed_mode == "center_cluster":
            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            seed_spread = float(self.narrative_controller.get("seed_spread", min(self.width, self.height) * 0.06))
            self.positions = center[None, :] + rng.randn(self.num_boids, 2) * seed_spread
            self.positions[:, 0] %= self.width
            self.positions[:, 1] %= self.height
        else:
            self.positions = rng.rand(self.num_boids, 2) * [self.width, self.height]
        angles = rng.rand(self.num_boids) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.speed
        if seed_mode == "center_cluster":
            initial_speed_scale = float(self.narrative_controller.get("initial_speed_scale", 0.35))
            self.velocities *= initial_speed_scale
        max_cohorts = int(self.narrative_controller.get("num_cohorts", 2)) if self.narrative_controller.get("enabled", False) else 2
        max_cohorts = max(1, min(max_cohorts, self.num_boids))
        # Interleave IDs so each colony gets a similar particle count.
        cohort_ids = (np.arange(self.num_boids, dtype=np.int32) % max_cohorts).reshape(-1, 1)
        self._cohort_id = cohort_ids
        # Backward-compatible ±1 sign used by older 2-cohort horizontal tests.
        self._cohort_sign = np.where((cohort_ids % 2) == 0, -1.0, 1.0).astype(np.float32)
        self._conversion_events = 0
        self._rng = rng if hasattr(rng, 'rand') else np.random.RandomState(0)
        infection = self.narrative_controller.get("infection") or {}
        self._role = np.zeros((self.num_boids, 1), dtype=np.float32)
        if infection.get("enabled", False):
            # Wide tissue field: keep current positions; seed a small malignant focus near center.
            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            frac = float(infection.get("seed_malignant_fraction", 0.06))
            seed_count = max(1, min(self.num_boids - 1, int(round(self.num_boids * frac))))
            dist = np.linalg.norm(self.positions - center[None, :], axis=1)
            seed_idx = np.argsort(dist)[:seed_count]
            self._role[seed_idx] = 1.0
            # Optionally widen tissue: if seed_mode tissue_field, re-scatter normals more broadly.
            if infection.get("tissue_spread", None) is not None:
                spread = float(infection["tissue_spread"])
                self.positions = center[None, :] + self._rng.randn(self.num_boids, 2).astype(np.float32) * spread
                self.positions[:, 0] %= self.width
                self.positions[:, 1] %= self.height
                # re-pick malignant seed at center after respread
                dist = np.linalg.norm(self.positions - center[None, :], axis=1)
                self._role[:] = 0.0
                self._role[np.argsort(dist)[:seed_count]] = 1.0
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
    def _active_cohort_count(self, phase_control):
        controller = self.narrative_controller or {}
        raw = phase_control.get("num_cohorts", controller.get("num_cohorts", 2))
        max_available = int(self._cohort_id.max()) + 1 if self._cohort_id is not None else 2
        return max(1, min(int(raw), max_available, self.num_boids))

    def _cohort_labels(self, num_cohorts):
        if self._cohort_id is None or num_cohorts <= 1:
            return np.zeros(self.num_boids, dtype=np.int32)
        return (self._cohort_id[:, 0] % int(num_cohorts)).astype(np.int32)

    def _cohort_unit_dirs(self, num_cohorts, split_axis):
        n = int(num_cohorts)
        if n <= 1:
            return np.zeros((1, 2), dtype=np.float32)
        if n == 2 and split_axis == "horizontal":
            return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        if n == 2 and split_axis == "vertical":
            return np.array([[0.0, -1.0], [0.0, 1.0]], dtype=np.float32)
        angles = (2.0 * np.pi * np.arange(n, dtype=np.float32)) / float(n)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dirs / norms

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
            cohort_locality = bool(phase_control.get("cohort_locality", False))
            cross_cohort_repulsion = float(phase_control.get("cross_cohort_repulsion", 0.0))
            global_pull = float(phase_control.get("global_pull", 0.0))
            cohort_global_pull = float(phase_control.get("cohort_global_pull", 0.0))

            num_cohorts = self._active_cohort_count(phase_control)
            labels = self._cohort_labels(num_cohorts)
            same_cohort = labels[:, None] == labels[None, :]

            dx = self.positions[:, 0:1] - self.positions[:, 0:1].T
            dy = self.positions[:, 1:2] - self.positions[:, 1:2].T
            dx = dx - self.width * np.round(dx / self.width)
            dy = dy - self.height * np.round(dy / self.height)
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            mask = dist < view_radius
            local_mask = mask & same_cohort if cohort_locality else mask
            counts = local_mask.sum(axis=1)[:, None]
            counts_safe = np.where(counts == 0, 1, counts)

            sep_x = np.where(mask & (dist > 0), dx / dist**2, 0).sum(axis=1)
            sep_y = np.where(mask & (dist > 0), dy / dist**2, 0).sum(axis=1)
            separation = np.stack([sep_x, sep_y], axis=1) * separation_gain
            if cross_cohort_repulsion > 0.0 and num_cohorts > 1:
                cross_mask = mask & (~same_cohort)
                cross_sep_x = np.where(cross_mask & (dist > 0), dx / dist**2, 0).sum(axis=1)
                cross_sep_y = np.where(cross_mask & (dist > 0), dy / dist**2, 0).sum(axis=1)
                separation += np.stack([cross_sep_x, cross_sep_y], axis=1) * cross_cohort_repulsion

            align_x = np.where(local_mask, self.velocities[:, 0], 0).sum(axis=1)
            align_y = np.where(local_mask, self.velocities[:, 1], 0).sum(axis=1)
            alignment = np.stack([align_x, align_y], axis=1) / counts_safe - self.velocities
            alignment = np.where(counts > 0, alignment, 0) * alignment_gain

            local_center_x = np.where(local_mask, self.positions[:, 0], 0).sum(axis=1)
            local_center_y = np.where(local_mask, self.positions[:, 1], 0).sum(axis=1)
            local_center = np.stack([local_center_x, local_center_y], axis=1) / counts_safe
            coh_dx = local_center[:, 0] - self.positions[:, 0]
            coh_dy = local_center[:, 1] - self.positions[:, 1]
            coh_dx = coh_dx - self.width * np.round(coh_dx / self.width)
            coh_dy = coh_dy - self.height * np.round(coh_dy / self.height)
            cohesion = np.stack([coh_dx, coh_dy], axis=1)
            cohesion = np.where(counts > 0, cohesion, 0) * cohesion_gain

            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            center_force = (center[None, :] - self.positions) * center_pull
            global_center = self.positions.mean(axis=0, keepdims=True)
            global_force = (global_center - self.positions) * global_pull

            unit_dirs = self._cohort_unit_dirs(num_cohorts, split_axis)
            boid_dirs = unit_dirs[labels]
            cohort_centers = np.zeros_like(self.positions)
            for cohort in range(num_cohorts):
                selected = labels == cohort
                if np.any(selected):
                    cohort_centers[selected] = self.positions[selected].mean(axis=0, keepdims=True)
            cohort_global_force = (cohort_centers - self.positions) * cohort_global_pull
            split_force = boid_dirs * split_push
            cohort_anchor = center[None, :] + boid_dirs * split_offset
            cohort_force = (cohort_anchor - self.positions) * cohort_pull

            self.velocities += (
                separation
                + alignment
                + cohesion
                + center_force
                + global_force
                + split_force
                + cohort_force
                + cohort_global_force
            )
            speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
            speeds[speeds == 0] = 1.0
            self.velocities = (self.velocities / speeds) * speed

            if damping > 0.0:
                self.velocities *= max(0.0, 1.0 - damping)

            max_step = float(
                phase_control.get(
                    "max_step_displacement",
                    self.narrative_controller.get("max_step_displacement", 0.0),
                )
            )
            if max_step > 0.0:
                step_speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
                scale = np.minimum(1.0, max_step / np.maximum(step_speeds, 1e-8))
                self.velocities = self.velocities * scale

            self.positions += self.velocities
            self.positions[:, 0] %= self.width
            self.positions[:, 1] %= self.height
            self._apply_contact_conversion(phase_control)
            self.current_step += 1


    def _infection_enabled(self):
        infection = (self.narrative_controller or {}).get("infection") or {}
        return bool(infection.get("enabled", False))

    def _apply_contact_conversion(self, phase_control):
        """Convert nearby normal particles after contact with malignant ones."""
        if not self._infection_enabled() or self._role is None:
            return
        infection = self.narrative_controller.get("infection") or {}
        radius = float(phase_control.get("conversion_radius", infection.get("conversion_radius", 10.0)))
        rate = float(phase_control.get("conversion_rate", infection.get("conversion_rate", 0.0)))
        if rate <= 0.0 or radius <= 0.0:
            return
        malignant = self._role[:, 0] >= 0.5
        normal = ~malignant
        if not np.any(malignant) or not np.any(normal):
            return
        dx = self.positions[:, 0:1] - self.positions[:, 0:1].T
        dy = self.positions[:, 1:2] - self.positions[:, 1:2].T
        dx = dx - self.width * np.round(dx / self.width)
        dy = dy - self.height * np.round(dy / self.height)
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        # Min distance from each normal particle to any malignant particle.
        min_to_mal = dist[normal][:, malignant].min(axis=1)
        eligible_idx = np.flatnonzero(normal)[min_to_mal <= radius]
        if eligible_idx.size == 0:
            return
        draw = self._rng.rand(eligible_idx.size) < rate
        converted = eligible_idx[draw]
        if converted.size:
            self._role[converted] = 1.0
            self._conversion_events += int(converted.size)

    def render(self):
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        sigma = max(self.base_view_radius * 0.6, 1.5)
        norm = 2.0 * sigma * sigma

        def deposit(mask_roles):
            density = np.zeros((self.height, self.width), dtype=np.float32)
            pts = self.positions[mask_roles]
            for x, y in pts:
                dx = xx - x
                dy = yy - y
                density += np.exp(-(dx * dx + dy * dy) / norm)
            if density.max() > 0:
                density = density / (density.max() + 1e-9)
            if self.keep_largest_component and density.max() > 0:
                density = self._keep_largest_component(density)
            return density

        if self._infection_enabled() and self._role is not None:
            normal = self._role[:, 0] < 0.5
            malignant = ~normal
            # If a role is empty, keep a zero field.
            normal_d = deposit(normal) if np.any(normal) else np.zeros((self.height, self.width), dtype=np.float32)
            mal_d = deposit(malignant) if np.any(malignant) else np.zeros((self.height, self.width), dtype=np.float32)
            # Cool cyan/blue = normal tissue; warm orange/red = malignant converted cells.
            r = np.clip(0.15 * normal_d + 1.00 * mal_d, 0.0, 1.0)
            g = np.clip(0.55 * normal_d + 0.45 * mal_d, 0.0, 1.0)
            b = np.clip(0.95 * normal_d + 0.12 * mal_d, 0.0, 1.0)
            img = np.stack([r, g, b], axis=-1)
            return Image.fromarray((img * 255).astype(np.uint8))

        density = np.zeros((self.height, self.width), dtype=np.float32)
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
        out = {"avg_speed": float(speeds.mean())}
        if self._role is not None:
            malignant = float((self._role[:, 0] >= 0.5).mean())
            out.update({
                "malignant_fraction": malignant,
                "normal_fraction": 1.0 - malignant,
                "conversion_events": int(self._conversion_events),
                "infection_enabled": bool(self._infection_enabled()),
            })
        return out
