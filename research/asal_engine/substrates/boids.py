import numpy as np
from PIL import Image

class Boids:
    def __init__(self, num_boids=100, width=128, height=128):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        self.reset([1.0, 1.0, 1.0, 5.0, 2.0])

    def reset(self, theta):
        self.separation, self.alignment, self.cohesion, self.speed, self.view_radius = theta
        self.positions = np.random.rand(self.num_boids, 2) * [self.width, self.height]
        angles = np.random.rand(self.num_boids) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.speed

    def step(self, substeps=1):
        for _ in range(substeps):
            # Calculate pairwise distance vectors
            dx = self.positions[:, 0:1] - self.positions[:, 0:1].T
            dy = self.positions[:, 1:2] - self.positions[:, 1:2].T
            
            # Periodic boundaries for distance
            dx = dx - self.width * np.round(dx / self.width)
            dy = dy - self.height * np.round(dy / self.height)
            
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            mask = dist < self.view_radius
            counts = mask.sum(axis=1)[:, None]
            counts_safe = np.where(counts == 0, 1, counts)
            
            # Separation
            sep_x = np.where(mask & (dist > 0), dx / dist**2, 0).sum(axis=1)
            sep_y = np.where(mask & (dist > 0), dy / dist**2, 0).sum(axis=1)
            separation = np.stack([sep_x, sep_y], axis=1) * self.separation
            
            # Alignment
            align_x = np.where(mask, self.velocities[:, 0], 0).sum(axis=1)
            align_y = np.where(mask, self.velocities[:, 1], 0).sum(axis=1)
            alignment = np.stack([align_x, align_y], axis=1) / counts_safe - self.velocities
            alignment = np.where(counts > 0, alignment, 0) * self.alignment
            
            # Cohesion
            center_x = np.where(mask, self.positions[:, 0], 0).sum(axis=1)
            center_y = np.where(mask, self.positions[:, 1], 0).sum(axis=1)
            center = np.stack([center_x, center_y], axis=1) / counts_safe
            
            coh_dx = center[:, 0] - self.positions[:, 0]
            coh_dy = center[:, 1] - self.positions[:, 1]
            coh_dx = coh_dx - self.width * np.round(coh_dx / self.width)
            coh_dy = coh_dy - self.height * np.round(coh_dy / self.height)
            
            cohesion = np.stack([coh_dx, coh_dy], axis=1)
            cohesion = np.where(counts > 0, cohesion, 0) * self.cohesion
            
            # Update velocity
            self.velocities += separation + alignment + cohesion
            
            # Normalize and set speed
            speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
            speeds[speeds == 0] = 1.0
            self.velocities = (self.velocities / speeds) * self.speed
            
            # Update position
            self.positions += self.velocities
            self.positions[:, 0] %= self.width
            self.positions[:, 1] %= self.height

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Simple rendering
        for x, y in self.positions.astype(int):
            if 0 <= x < self.width and 0 <= y < self.height:
                img[y, x] = [255, 255, 255]
                
        return Image.fromarray(img)

    def stats(self):
        speeds = np.linalg.norm(self.velocities, axis=1)
        return {"avg_speed": float(speeds.mean())}
