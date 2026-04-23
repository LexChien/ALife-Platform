import numpy as np
from PIL import Image

class ReactionDiffusion:
    def __init__(self, size=128):
        self.size = size
        self.reset([0.16, 0.08, 0.06, 0.062, 1.0])

    def reset(self, theta):
        Du, Dv, F, k, _ = theta
        self.theta = [float(Du), float(Dv), float(F), float(k), 1.0]
        s = self.size
        self.U = np.ones((s, s), dtype=np.float32)
        self.V = np.zeros((s, s), dtype=np.float32)
        c = s // 2
        radius = max(s // 10, 3)
        yy, xx = np.mgrid[0:s, 0:s]
        dist2 = (yy - c) ** 2 + (xx - c) ** 2
        disc = dist2 <= radius ** 2
        halo = dist2 <= int((radius * 1.45) ** 2)
        self.V[disc] = 1.0
        self.U[disc] = 0.5
        self.V[halo & ~disc] = 0.35

    def step(self, substeps=2):
        Du, Dv, F, k, _ = self.theta
        for _ in range(substeps):
            U, V = self.U, self.V
            lapU = -U + 0.2*(np.roll(U,1,0)+np.roll(U,-1,0)+np.roll(U,1,1)+np.roll(U,-1,1))
            lapV = -V + 0.2*(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))
            uvv = U * V * V
            self.U = U + Du * lapU - uvv + F * (1-U)
            self.V = V + Dv * lapV + uvv - (F+k) * V
            self.U = np.clip(self.U, 0.0, 1.0)
            self.V = np.clip(self.V, 0.0, 1.0)

    def render(self):
        field = np.clip(self.V - 0.35 * self.U, -1.0, 1.0)
        x = (field - field.min()) / (np.ptp(field) + 1e-9)
        gy, gx = np.gradient(x)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-9)

        # Approximate a microscopy phase-contrast look with grayscale body and bright rim.
        body = np.clip(0.20 + 0.65 * x, 0.0, 1.0)
        halo = np.clip(body + 0.35 * edge, 0.0, 1.0)
        shadow = np.clip(body - 0.18 * edge, 0.0, 1.0)
        img = np.stack([halo, body, shadow], axis=-1)
        return Image.fromarray((img * 255).astype("uint8"))

    def stats(self):
        return {"mean_v": float(self.V.mean()), "std_v": float(self.V.std())}
