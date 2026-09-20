import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class NCA:
    """
    Neural Cellular Automata Substrate.
    Optimized for spatial boundary control and dynamic stability.
    """
    def __init__(self, channel_n=16, width=128, height=128, fire_rate=0.5, device="cpu"):
        self.channel_n = channel_n
        self.w, self.h = width, height
        self.device = device
        self.fire_rate = fire_rate
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        self.filters = torch.stack([identity, sobel_x, sobel_y]).repeat(channel_n, 1, 1).unsqueeze(1).to(device)
        
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel_n, 1, bias=False)
        ).to(device)
        
        with torch.no_grad():
            self.update_net[-1].weight.fill_(0.0)
            
        self.state = None
        self.lyapunov_delta = 0.0
        self.clamped = False
        self.reset()

    def reset(self, theta=None):
        self.state = torch.zeros((1, self.channel_n, self.h, self.w), device=self.device)
        self.state[:, 3:, self.h//2, self.w//2] = 1.0 
        self.clamped = False

    def step(self, substeps=1):
        for _ in range(substeps):
            pre_energy = torch.sum(self.state**2).item()
            
            y = F.conv2d(self.state, self.filters, groups=self.channel_n, padding=1)
            dx = self.update_net(y)
            
            update_mask = torch.rand(self.state[:, :1, :, :].shape, device=self.device) < self.fire_rate
            alive = F.max_pool2d(self.state[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1
            
            self.state = self.state + dx * update_mask.float() * alive.float()
            
            # Global Pooling Constraint
            current_energy = torch.sum(self.state**2)
            if current_energy > 20000.0:
                self.state = self.state * (20000.0 / (current_energy + 1e-9)).sqrt()
                self.clamped = True
            else:
                self.clamped = False
            
            self.lyapunov_delta = torch.sum(self.state**2).item() - pre_energy

    def render(self):
        with torch.no_grad():
            rgb = self.state[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
            img = np.clip(rgb, 0, 1)
            return Image.fromarray((img * 255).astype(np.uint8))

    def stats(self):
        with torch.no_grad():
            alpha = self.state[0, 3, :, :]
            alive_mask = (alpha > 0.1).cpu().numpy().astype(np.uint8)
            energy = float(torch.sum(self.state**2))
            active_mass = float(np.sum(alive_mask))
            mass_ratio = active_mass / (self.w * self.h)
            
            # Refined Stability: Must not be clamped and must have reasonable mass
            is_stable = bool(abs(self.lyapunov_delta) < 50.0 and not self.clamped and mass_ratio < 0.8)
            
            return {
                "energy": energy,
                "lyapunov_delta": float(self.lyapunov_delta),
                "clamped": self.clamped,
                "is_stable": is_stable,
                "active_mass": active_mass,
                "mass_ratio": round(mass_ratio, 4),
                "num_components": self._count_components(alive_mask)
            }

    def _count_components(self, mask):
        try:
            from scipy.ndimage import label
            _, n = label(mask)
            return n
        except Exception: return 1 if np.any(mask) else 0
