import unittest

import numpy as np

from research.asal_engine.narrative_scores import score_narrative_trajectory


def _blank():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _draw_disk(img, cx, cy, radius, value=255):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[mask] = value
    return img


def _single(cx=32):
    return _draw_disk(_blank(), cx, 32, 9)


def _double():
    img = _draw_disk(_blank(), 21, 32, 8)
    return _draw_disk(img, 43, 32, 8)


def _double_with_fragments():
    img = _double()
    img = _draw_disk(img, 8, 8, 2)
    img = _draw_disk(img, 56, 8, 2)
    img = _draw_disk(img, 8, 56, 2)
    return img


NARRATIVE_CFG = {
    "phases": [
        {"name": "birth", "frame_range": [0, 1], "target_components": 1, "weight": 0.3},
        {"name": "split", "frame_range": [2, 3], "target_components": 2, "weight": 0.4},
        {"name": "fusion", "frame_range": [4, 5], "target_components": 1, "weight": 0.3},
    ]
}


class TestASALNarrativeScores(unittest.TestCase):
    def test_valid_trajectory_scores_higher(self):
        valid_frames = [_single(), _single(31), _double(), _double(), _single(30), _single()]
        invalid_frames = [_single(), _single(), _single(), _single(), _single(), _single()]
        valid = score_narrative_trajectory(valid_frames, NARRATIVE_CFG)
        invalid = score_narrative_trajectory(invalid_frames, NARRATIVE_CFG)
        self.assertTrue(valid["phase_order_valid"])
        self.assertFalse(invalid["phase_order_valid"])
        self.assertGreater(valid["total_score"], invalid["total_score"])
        self.assertEqual(valid["actual_component_sequence"], [1, 2, 1])

    def test_dominant_sequence_ignores_small_fragments(self):
        frames = [_single(), _single(), _double_with_fragments(), _double_with_fragments(), _single(), _single()]
        result = score_narrative_trajectory(frames, NARRATIVE_CFG)
        self.assertEqual(result["actual_component_sequence"], [1, 2, 1])
        self.assertTrue(result["phase_order_valid"])


if __name__ == "__main__":
    unittest.main()
