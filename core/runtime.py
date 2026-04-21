class RuntimeProfile:
    def __init__(self, device="cpu", precision="fp32", backend="local"):
        self.device = device
        self.precision = precision
        self.backend = backend

    def to_dict(self):
        return {
            "device": self.device,
            "precision": self.precision,
            "backend": self.backend,
        }
