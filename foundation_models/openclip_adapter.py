import numpy as np
from PIL import Image
from . import foundation_models

try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    open_clip = MagicMock()
    HAS_CLIP = False

@foundation_models.register("openclip")
class OpenCLIPAdapter:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="auto"):
        if not HAS_CLIP:
            raise ImportError("Please install torch and open_clip_torch to use OpenCLIPAdapter. Run: pip install torch open_clip_torch")

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self._device)
        self.model.eval()

    @property
    def device(self):
        return self._device

    @staticmethod
    def prepare_image(image, value_range="auto"):
        """Accept PIL or HxW/HxWx{1,3,4} arrays; floats use unit or byte range.

        Auto treats floating arrays with max <= 1 as unit range. Specify byte
        explicitly for dark floating arrays whose intended pixel range is 0..255.
        Invalid pixels fail instead of silently clipping or wrapping.
        """
        if value_range not in {"auto", "unit", "byte"}:
            raise ValueError("value_range must be auto, unit, or byte")
        if isinstance(image, Image.Image):
            return image if image.mode == "RGB" else image.convert("RGB")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a PIL image or NumPy array")
        if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] not in {1, 3, 4}):
            raise ValueError("image must have shape HxW or HxWx1/3/4")
        if not image.size or min(image.shape[:2]) == 0:
            raise ValueError("image must be nonempty")
        if image.dtype.kind not in "uif":
            raise TypeError("image pixels must be integers or floating point")
        if not np.isfinite(image).all() or image.min() < 0:
            raise ValueError("image pixels must be finite and nonnegative")
        unit = value_range == "unit" or (value_range == "auto" and image.dtype.kind == "f" and image.max() <= 1)
        maximum = 1 if unit else 255
        if image.max() > maximum:
            raise ValueError(f"image pixels exceed {maximum} for the selected value_range")
        pixels = np.rint(image.astype(np.float64) * (255 if unit else 1)).astype(np.uint8)
        if pixels.ndim == 3 and pixels.shape[2] == 1:
            pixels = pixels[..., 0]
        return Image.fromarray(pixels).convert("RGB")

    def img_embed(self, pil_img, value_range="auto"):
        import torch
        pil_img = self.prepare_image(pil_img, value_range=value_range)
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32)[0]

    def txt_embed(self, text: str):
        import torch
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)[0]
