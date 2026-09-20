import sys
from unittest.mock import MagicMock

# Gracefully handle import errors on platforms missing CUDA or other dependencies during testing
try:
    import torch
except ImportError:
    sys.modules["torch"] = MagicMock()

try:
    import open_clip
except ImportError:
    sys.modules["open_clip"] = MagicMock()

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class TestOpenCLIPAdapter(unittest.TestCase):

    @patch("foundation_models.openclip_adapter.open_clip")
    @patch("foundation_models.openclip_adapter.torch")
    def test_openclip_adapter_handles_numpy_and_pil(self, mock_torch, mock_open_clip):
        # 1. Setup mock open_clip objects
        mock_model = MagicMock()
        mock_preprocess = MagicMock(return_value=MagicMock())
        mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_open_clip.get_tokenizer.return_value = MagicMock()

        # Mock torch.cuda.is_available()
        mock_torch.cuda.is_available.return_value = False

        # Mock model.encode_image to return a tensor with norm
        mock_tensor = MagicMock()
        mock_tensor.norm.return_value = mock_tensor
        mock_tensor.__itruediv__.return_value = mock_tensor
        mock_model.encode_image.return_value = mock_tensor

        # Mock tensor conversion to numpy
        mock_features = np.ones((1, 512), dtype=np.float32)
        mock_tensor.cpu.return_value.numpy.return_value = mock_features

        # 2. Instantiate OpenCLIPAdapter
        # Temporarily mock HAS_CLIP as True so import and init succeed
        with patch("foundation_models.openclip_adapter.HAS_CLIP", True):
            from foundation_models.openclip_adapter import OpenCLIPAdapter
            adapter = OpenCLIPAdapter(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")

        # 3. Test with PIL Image
        pil_img = Image.new("RGB", (64, 64), color="red")
        embed_pil = adapter.img_embed(pil_img)

        # Verify preprocess was called with the PIL Image
        mock_preprocess.assert_called_with(pil_img)
        self.assertEqual(embed_pil.shape, (512,))

        # 4. Test with NumPy float32 array in range [0, 1]
        np_img_float = np.ones((64, 64, 3), dtype=np.float32)
        mock_preprocess.reset_mock()
        embed_np_float = adapter.img_embed(np_img_float)

        # Verify preprocess was called with a PIL Image
        called_arg = mock_preprocess.call_args[0][0]
        self.assertIsInstance(called_arg, Image.Image)

        # 5. Test with NumPy uint8 array
        np_img_uint8 = np.ones((64, 64, 3), dtype=np.uint8) * 255
        mock_preprocess.reset_mock()
        embed_np_uint8 = adapter.img_embed(np_img_uint8)

        called_arg_uint8 = mock_preprocess.call_args[0][0]
        self.assertIsInstance(called_arg_uint8, Image.Image)


if __name__ == "__main__":
    unittest.main()
