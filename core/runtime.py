import logging
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

class RuntimeManager:
    """
    Manages the execution hardware, tracking VRAM, and handling dynamic device switching.
    This replaces the naive RuntimeProfile stub.
    """
    def __init__(self, device="auto", precision="fp32", backend="local"):
        self.device_request = device
        self.precision = precision
        self.backend = backend
        self._device = self._resolve_device(device)
        logger.info(f"RuntimeManager initialized: Requested {device}, Resolved to {self._device}")

    def _resolve_device(self, device):
        if not HAS_TORCH:
            return "cpu"
            
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"Device {device} requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
        if device.startswith("mps") and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning(f"Device {device} requested but MPS is not available. Falling back to CPU.")
            return "cpu"
            
        return device

    @property
    def device(self):
        """Returns the fully resolved device string (e.g., 'cuda', 'cpu', 'mps')."""
        return self._device

    def get_memory_info(self):
        """
        Returns a dictionary of allocated and max allocated VRAM in MB.
        Only valid for CUDA devices.
        """
        if not HAS_TORCH or not self._device.startswith("cuda"):
            return None
            
        try:
            allocated = torch.cuda.memory_allocated(self._device) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
            return {
                "allocated_mb": round(allocated, 2),
                "max_allocated_mb": round(max_allocated, 2)
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return None

    def clear_memory(self):
        """Empties the CUDA cache if applicable."""
        if HAS_TORCH and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def offload_to_cpu(self, model):
        """
        Stub method for future local Large Language Model memory offloading.
        Safely moves a PyTorch model to CPU and clears GPU memory.
        """
        if HAS_TORCH and hasattr(model, "to"):
            model.to("cpu")
            self.clear_memory()

    def to_dict(self):
        """Used for experiment logging and summaries."""
        return {
            "device_request": self.device_request,
            "resolved_device": self.device,
            "precision": self.precision,
            "backend": self.backend,
        }
