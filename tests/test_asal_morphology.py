import unittest

import numpy as np

from research.asal_engine.morphology import analyze_frame


def _blank():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _draw_disk(img, cx, cy, radius, value=255):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[mask] = value
    return img


class TestASALMorphology(unittest.TestCase):
    def test_single_body_is_one_component(self):
        img = _draw_disk(_blank(), 32, 32, 10)
        stats = analyze_frame(img)
        self.assertEqual(stats["num_components"], 1)
        self.assertEqual(stats["dominant_num_components"], 1)
        self.assertGreater(stats["largest_area"], 200)
        self.assertGreater(stats["dominant_mass_ratio"], 0.95)

    def test_two_bodies_are_two_components(self):
        img = _draw_disk(_blank(), 20, 32, 8)
        img = _draw_disk(img, 44, 32, 8)
        stats = analyze_frame(img)
        self.assertEqual(stats["num_components"], 2)
        self.assertEqual(stats["dominant_num_components"], 2)
        self.assertGreater(stats["second_area"], 100)
        self.assertGreater(stats["centroid_distance"], 10.0)

    def test_many_fragments_exceed_two_components(self):
        img = _blank()
        for cx, cy in ((12, 12), (48, 12), (12, 48), (48, 48)):
            img = _draw_disk(img, cx, cy, 4)
        stats = analyze_frame(img)
        self.assertGreaterEqual(stats["num_components"], 4)

    def test_dominant_components_ignore_tiny_fragments(self):
        img = _draw_disk(_blank(), 20, 32, 8)
        img = _draw_disk(img, 44, 32, 8)
        for cx, cy in ((8, 8), (56, 8), (8, 56), (56, 56)):
            img = _draw_disk(img, cx, cy, 2)
        stats = analyze_frame(img)
        self.assertGreater(stats["num_components"], 2)
        self.assertEqual(stats["dominant_num_components"], 2)


if __name__ == "__main__":
    unittest.main()
