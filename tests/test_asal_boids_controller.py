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

    def test_boids_damping_actually_reduces_speed(self):
        controller_no_damping = {
            "enabled": True,
            "base_velocity_damping": 0.0,
            "phases": [{"name": "birth", "damping": 0.0}]
        }
        boids_no_damping = Boids(num_boids=10, width=64, height=64, keep_largest_component=False, narrative_controller=controller_no_damping)
        boids_no_damping.configure_narrative(total_steps=5, phases=[{"name": "birth", "frame_range": [0, 4]}])
        boids_no_damping.reset([0.8, 0.4, 1.1, 10.0, 4.0], seed=42)
        boids_no_damping.step()
        speeds_no_damping = np.linalg.norm(boids_no_damping.velocities, axis=1)

        controller_with_damping = {
            "enabled": True,
            "base_velocity_damping": 0.5,
            "phases": [{"name": "birth", "damping": 0.5}]
        }
        boids_with_damping = Boids(num_boids=10, width=64, height=64, keep_largest_component=False, narrative_controller=controller_with_damping)
        boids_with_damping.configure_narrative(total_steps=5, phases=[{"name": "birth", "frame_range": [0, 4]}])
        boids_with_damping.reset([0.8, 0.4, 1.1, 10.0, 4.0], seed=42)
        boids_with_damping.step()
        speeds_with_damping = np.linalg.norm(boids_with_damping.velocities, axis=1)

        np.testing.assert_allclose(speeds_no_damping, 10.0, rtol=1e-5)
        np.testing.assert_allclose(speeds_with_damping, 5.0, rtol=1e-5)

    def test_boids_reset_with_seed_is_deterministic(self):
        boids = Boids(num_boids=10, width=64, height=64)
        boids.reset([0.8, 0.4, 1.1, 10.0, 4.0], seed=123)
        pos1 = boids.positions.copy()
        vel1 = boids.velocities.copy()
        boids.reset([0.8, 0.4, 1.1, 10.0, 4.0], seed=123)
        pos2 = boids.positions.copy()
        vel2 = boids.velocities.copy()
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(vel1, vel2)
        boids.reset([0.8, 0.4, 1.1, 10.0, 4.0], seed=456)
        pos3 = boids.positions.copy()
        self.assertFalse(np.array_equal(pos1, pos3))

    def test_max_step_displacement_caps_velocity(self):
        controller = {
            "enabled": True,
            "max_step_displacement": 1.5,
            "phases": [{"name": "birth", "damping": 0.0, "speed_scale": 1.0, "max_step_displacement": 1.5}],
        }
        boids = Boids(num_boids=8, width=64, height=64, keep_largest_component=False, narrative_controller=controller)
        boids.configure_narrative(total_steps=3, phases=[{"name": "birth", "frame_range": [0, 2]}])
        boids.reset([0.0, 0.0, 0.0, 10.0, 4.0], seed=3)
        boids.velocities[:] = 0.0
        boids.velocities[:, 0] = 10.0
        boids.step()
        speeds = np.linalg.norm(boids.velocities, axis=1)
        self.assertTrue(np.all(speeds <= 1.5 + 1e-5))

    def test_multicohort_phase_uses_four_labels(self):
        controller = {
            "enabled": True,
            "num_cohorts": 4,
            "split_axis": "horizontal",
            "phases": [
                {"name": "initiation", "num_cohorts": 1, "center_pull": 0.05},
                {"name": "metastasis", "num_cohorts": 4, "split_push": 0.2, "split_offset": 18.0, "cohort_pull": 0.1},
            ],
        }
        boids = Boids(num_boids=40, width=96, height=96, keep_largest_component=False, narrative_controller=controller)
        boids.configure_narrative(
            total_steps=8,
            phases=[
                {"name": "initiation", "frame_range": [0, 2]},
                {"name": "metastasis", "frame_range": [3, 7]},
            ],
        )
        boids.reset([0.0, 0.0, 0.0, 12.0, 5.0], seed=5)
        self.assertEqual(int(boids._cohort_id.max()) + 1, 4)
        self.assertEqual(boids._active_cohort_count({"num_cohorts": 1}), 1)
        self.assertEqual(len(np.unique(boids._cohort_labels(4))), 4)
        boids.current_step = 4
        ctrl = boids._active_phase_control()
        self.assertEqual(int(ctrl.get("num_cohorts")), 4)
        labels = boids._cohort_labels(4)
        centers_before = []
        for k in range(4):
            centers_before.append(boids.positions[labels == k].mean(axis=0))
        for _ in range(5):
            boids.step()
        centers_after = []
        for k in range(4):
            centers_after.append(boids.positions[labels == k].mean(axis=0))
        # Cohort centers should not all collapse to the same point.
        spread = np.linalg.norm(np.std(np.stack(centers_after), axis=0))
        self.assertGreater(spread, 1.0)

    def test_infection_contact_converts_neighbors(self):
        controller = {
            "enabled": True,
            "num_cohorts": 1,
            "infection": {
                "enabled": True,
                "seed_malignant_fraction": 0.1,
                "tissue_spread": 18.0,
                "conversion_radius": 20.0,
                "conversion_rate": 1.0,
            },
            "phases": [
                {"name": "healthy_tissue", "conversion_rate": 0.0},
                {"name": "contact_conversion", "conversion_rate": 1.0, "conversion_radius": 20.0},
            ],
        }
        boids = Boids(num_boids=40, width=96, height=96, keep_largest_component=False, narrative_controller=controller)
        boids.configure_narrative(
            total_steps=10,
            phases=[
                {"name": "healthy_tissue", "frame_range": [0, 2]},
                {"name": "contact_conversion", "frame_range": [3, 9]},
            ],
        )
        boids.reset([0.2, 0.2, 0.8, 6.0, 5.0], seed=9)
        start_mal = float((boids._role[:, 0] >= 0.5).mean())
        self.assertGreater(start_mal, 0.0)
        self.assertLess(start_mal, 0.35)
        boids.current_step = 3
        for _ in range(6):
            boids.step()
        end_mal = float((boids._role[:, 0] >= 0.5).mean())
        self.assertGreater(end_mal, start_mal)
        frame = np.asarray(boids.render())
        self.assertEqual(frame.ndim, 3)


if __name__ == "__main__":
    unittest.main()
