import numpy as np
from PIL import Image
from . import foundation_models

try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

@foundation_models.register("openclip")
class OpenCLIPAdapter:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
        if not HAS_CLIP:
            raise ImportError("Please install torch and open_clip_torch to use OpenCLIPAdapter. Run: pip install torch open_clip_torch")
            
        self._device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self._device)
        self.model.eval()

    @property
    def device(self):
        return self._device

    def img_embed(self, pil_img):
        import torch
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
