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
        r = s // 10
        c = s // 2
        self.V[c-r:c+r, c-r:c+r] = 1.0

    def step(self, substeps=2):
        Du, Dv, F, k, _ = self.theta
        for _ in range(substeps):
            U, V = self.U, self.V
            lapU = -U + 0.2*(np.roll(U,1,0)+np.roll(U,-1,0)+np.roll(U,1,1)+np.roll(U,-1,1))
            lapV = -V + 0.2*(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))
            uvv = U * V * V
            self.U = U + Du * lapU - uvv + F * (1-U)
            self.V = V + Dv * lapV + uvv - (F+k) * V

    def render(self):
        x = (self.V - self.V.min()) / (np.ptp(self.V) + 1e-9)
        img = np.stack([x, x*0.7, 1-x], axis=-1)
        return Image.fromarray((img * 255).astype("uint8"))

    def stats(self):
        return {"mean_v": float(self.V.mean()), "std_v": float(self.V.std())}
