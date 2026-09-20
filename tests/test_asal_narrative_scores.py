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

    def test_sustain_duration_penalty_applied(self):
        # 1. Sustained case: Split has 2 frames. Both match target 2. Sustain ratio = 2/2 = 1.0 (sustain_penalty = 1.0)
        cfg_sustained = {
            "phases": [
                {"name": "split", "frame_range": [0, 1], "target_components": 2, "weight": 1.0}
            ]
        }
        frames_sustained = [_double(), _double()]
        res_sustained = score_narrative_trajectory(frames_sustained, cfg_sustained)
        self.assertEqual(res_sustained["phases"][0]["sustain_ratio"], 1.0)
        self.assertAlmostEqual(res_sustained["phases"][0]["score"], res_sustained["total_score"], places=5)

        # 2. Transient case: Split has 5 frames. Only 1 matches target 2. Sustain ratio = 1/5 = 0.2 < 0.25 (sustain_penalty = 0.2 / 0.25 = 0.8)
        cfg_transient = {
            "phases": [
                {"name": "split", "frame_range": [0, 4], "target_components": 2, "weight": 1.0}
            ]
        }
        frames_transient = [_double(), _single(), _single(), _single(), _single()]
        res_transient = score_narrative_trajectory(frames_transient, cfg_transient)
        self.assertEqual(res_transient["phases"][0]["sustain_ratio"], 0.2)
        self.assertLess(res_transient["total_score"], res_sustained["total_score"])
        self.assertEqual(res_transient["phases"][0]["longest_qualified_run"], 1)
        self.assertFalse(res_transient["accepted"])
        self.assertTrue(res_sustained["accepted"])

    def test_fragmentation_coherence_score_applied(self):
        # Split phase score with very low fragmentation (clean double)
        cfg = {
            "phases": [
                {"name": "split", "frame_range": [0, 0], "target_components": 2, "weight": 1.0}
            ]
        }
        clean_res = score_narrative_trajectory([_double()], cfg)
        fragmented_res = score_narrative_trajectory([_double_with_fragments()], cfg)
        
        # Fragmented double must have a lower score than clean double due to coherence penalty
        self.assertGreater(clean_res["total_score"], fragmented_res["total_score"])

    def test_body_continuity_penalty_applied(self):
        cfg = {
            "phases": [
                {"name": "birth", "frame_range": [0, 1], "target_components": 1, "weight": 1.0}
            ]
        }
        # 1. Smoothly moving component: x=32 -> x=33 (jump = 1 pixel < threshold 12.8 pixels) -> continuity_score = 1.0
        smooth_frames = [_single(32), _single(33)]
        smooth_res = score_narrative_trajectory(smooth_frames, cfg)
        self.assertEqual(smooth_res["continuity_score"], 1.0)

        # 2. Teleporting component: x=20 -> x=50 (jump = 30 pixels > threshold 12.8 pixels) -> continuity_score < 1.0
        teleport_frames = [_single(20), _single(50)]
        teleport_res = score_narrative_trajectory(teleport_frames, cfg)
        self.assertLess(teleport_res["continuity_score"], 1.0)
        self.assertLess(teleport_res["total_score"], smooth_res["total_score"])

    def test_intermittent_flashes_do_not_equal_stable_split(self):
        cfg = {"phases": [{"name": "split", "frame_range": [0, 19], "target_components": 2}]}
        stable = score_narrative_trajectory([_double()] * 20, cfg)
        flashes = score_narrative_trajectory([_double(), _single(), _single(), _single()] * 5, cfg)
        self.assertTrue(stable["accepted"])
        self.assertFalse(flashes["accepted"])
        self.assertEqual(flashes["phases"][0]["sustain_ratio"], 0.25)
        self.assertEqual(flashes["phases"][0]["longest_qualified_run"], 1)
        self.assertLess(flashes["total_score"], stable["total_score"])

    def test_disappearance_and_reappearance_rejects_continuity(self):
        cfg = {"phases": [{"name": "birth", "frame_range": [0, 2], "target_components": 1}]}
        result = score_narrative_trajectory([_single(16), _blank(), _single(48)], cfg)
        self.assertTrue(result["phase_order_valid"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["continuity_score"], 0.0)
        self.assertIn("continuity:missing_body", result["failure_reasons"])

    def test_merger_retains_mass_but_vanishing_body_fails(self):
        left = _draw_disk(np.zeros((128, 128, 3), dtype=np.uint8), 48, 64, 9)
        pair = _draw_disk(left.copy(), 72, 64, 9)
        fused = _draw_disk(np.zeros_like(left), 60, 64, 13)
        vanished = score_narrative_trajectory([left, left, pair, pair, left, left], NARRATIVE_CFG)
        merged = score_narrative_trajectory([left, left, pair, pair, fused, fused], NARRATIVE_CFG)
        self.assertTrue(vanished["phase_order_valid"])
        self.assertFalse(vanished["accepted"])
        self.assertIn("continuity:body_mass_loss", vanished["failure_reasons"])
        self.assertTrue(merged["accepted"], merged["failure_reasons"])

    def test_small_fragments_are_included_in_quality_measurement(self):
        noisy = _double()
        for y in range(1, 64, 4):
            for x in range(1, 64, 4):
                if not np.any(noisy[max(0, y - 2):y + 3, max(0, x - 2):x + 3]):
                    noisy[y, x] = 255
        cfg = {"phases": [{"name": "split", "frame_range": [0, 3], "target_components": 2}]}
        clean = score_narrative_trajectory([_double()] * 4, cfg)
        fragmented = score_narrative_trajectory([noisy] * 4, cfg)
        self.assertGreater(fragmented["frame_stats"][0]["discarded_foreground_area"], 0)
        self.assertEqual(fragmented["actual_component_sequence"], [2])
        self.assertLess(fragmented["total_score"], clean["total_score"])
        self.assertFalse(fragmented["accepted"])

    def test_invalid_windows_and_empty_frames_fail_explicitly(self):
        with self.assertRaises(ValueError):
            score_narrative_trajectory([], NARRATIVE_CFG)
        for window in ([-1, 1], [0, 99], [2, 1]):
            with self.subTest(window=window), self.assertRaises(ValueError):
                score_narrative_trajectory([_single()] * 3, {"phases": [{"name": "birth", "frame_range": window}]})

    def test_duration_threshold_can_be_configured_without_changing_reward_type(self):
        cfg = {"phases": [{"name": "split", "frame_range": [0, 7], "target_components": 2}]}
        frames = [_double()] * 2 + [_single()] * 6
        loose = score_narrative_trajectory(frames, cfg)
        strict = score_narrative_trajectory(frames, {**cfg, "acceptance": {"min_consecutive_ratio": 0.75}})
        self.assertTrue(loose["accepted"], loose["failure_reasons"])
        self.assertFalse(strict["accepted"])
        self.assertLess(strict["total_score"], loose["total_score"])


if __name__ == "__main__":
    unittest.main()
