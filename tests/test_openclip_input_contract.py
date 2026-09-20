import unittest
import numpy as np
from PIL import Image
from foundation_models.openclip_adapter import OpenCLIPAdapter


class TestImageInputContract(unittest.TestCase):
    def test_all_float_precisions_preserve_unit_and_byte_pixels(self):
        original = np.arange(256, dtype=np.uint8).reshape(16, 16)
        expected = np.repeat(original[..., None], 3, axis=2)
        for dtype in (np.float16, np.float32, np.float64):
            for pixels in (original.astype(dtype) / 255, original.astype(dtype)):
                with self.subTest(dtype=dtype, maximum=pixels.max()):
                    converted = OpenCLIPAdapter.prepare_image(pixels)
                    np.testing.assert_array_equal(converted, expected)

    def test_dark_byte_floats_can_disambiguate_the_range(self):
        pixels = np.ones((2, 3, 3), dtype=np.float32)
        self.assertEqual(np.asarray(OpenCLIPAdapter.prepare_image(pixels)).max(), 255)
        self.assertEqual(np.asarray(OpenCLIPAdapter.prepare_image(pixels, "byte")).max(), 1)

    def test_supported_shapes_and_pil_modes_are_rgb(self):
        for shape in ((2, 3), (2, 3, 1), (2, 3, 3), (2, 3, 4)):
            self.assertEqual(OpenCLIPAdapter.prepare_image(np.zeros(shape, dtype=np.uint8)).mode, "RGB")
        self.assertEqual(OpenCLIPAdapter.prepare_image(Image.new("L", (3, 2))).mode, "RGB")

    def test_invalid_pixels_and_shapes_fail_instead_of_clipping(self):
        for bad in (np.full((2, 2), -1), np.full((2, 2), 256),
                    np.full((2, 2), np.nan), np.full((2, 2), np.inf),
                    np.zeros((3, 2, 2)), np.zeros((0, 2)), np.zeros((2,))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                OpenCLIPAdapter.prepare_image(bad)
        for bad in (np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=complex), [[1, 2]]):
            with self.assertRaises(TypeError):
                OpenCLIPAdapter.prepare_image(bad)


if __name__ == "__main__":
    unittest.main()
