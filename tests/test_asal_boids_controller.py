import unittest

import numpy as np

from research.asal_engine.substrates.boids import Boids


class TestASALBoidsController(unittest.TestCase):
    def test_phase_control_changes_active_dynamics(self):
        controller = {
            "enabled": True,
            "split_axis": "horizontal",
            "base_center_pull": 0.0,
            "base_velocity_damping": 0.0,
            "phases": [
                {"name": "birth", "center_pull": 0.02},
                {"name": "split", "split_push": 0.05, "split_offset": 12.0, "cohort_pull": 0.02},
                {"name": "fusion", "center_pull": 0.03},
            ],
        }
        boids = Boids(num_boids=20, width=64, height=64, keep_largest_component=False, narrative_controller=controller)
        boids.configure_narrative(
            total_steps=9,
            phases=[
                {"name": "birth", "frame_range": [0, 2]},
                {"name": "split", "frame_range": [3, 5]},
                {"name": "fusion", "frame_range": [6, 8]},
            ],
        )
        boids.reset([0.8, 0.4, 1.1, 24.0, 4.0])
        self.assertEqual(boids._active_phase_control().get("center_pull"), 0.02)
        boids.current_step = 4
        self.assertEqual(boids._active_phase_control().get("split_push"), 0.05)
        boids.current_step = 7
        self.assertEqual(boids._active_phase_control().get("center_pull"), 0.03)

    def test_split_phase_creates_opposite_cohort_velocity(self):
        controller = {
            "enabled": True,
            "split_axis": "horizontal",
            "phases": [
                {"name": "birth"},
                {"name": "split", "split_push": 0.40, "split_offset": 0.0, "cohort_pull": 0.0},
            ],
        }
        phases = [
            {"name": "birth", "frame_range": [0, 4]},
            {"name": "split", "frame_range": [5, 11]},
        ]
        boids = Boids(num_boids=40, width=96, height=96, keep_largest_component=False, narrative_controller=controller)
        boids.configure_narrative(total_steps=12, phases=phases)
        boids.reset([0.0, 0.0, 0.0, 18.0, 5.0])
        rng = np.random.RandomState(7)
        boids.positions = 48.0 + rng.randn(40, 2) * 1.5
        boids.velocities = np.zeros((40, 2), dtype=np.float32)
        cohort_sign = boids._cohort_sign[:, 0]

        for _ in range(5):
            boids.step()
        boids.step()
        left_vx = float(boids.velocities[cohort_sign < 0][:, 0].mean())
        right_vx = float(boids.velocities[cohort_sign > 0][:, 0].mean())
        self.assertLess(left_vx, 0.0)
        self.assertGreater(right_vx, 0.0)


if __name__ == "__main__":
    unittest.main()
