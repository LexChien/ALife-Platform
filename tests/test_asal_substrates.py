import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.asal_engine.narrative_scores import score_narrative_trajectory
from research.asal_engine.substrates.controlled_cells import ControlledCells
from research.asal_engine.substrates.lenia import Lenia
from research.asal_engine.substrates.nca import NCA


class TestLocalAutomata(unittest.TestCase):
    def test_nca_empty_state_and_local_causal_cone(self):
        nca = NCA(size=24, channels=4, hidden=6)
        nca.reset([0.16, 0.06, 0.01, 0.02, 1], seed=7)
        old = nca.grid.copy()
        nca.step()
        self.assertFalse(np.array_equal(old, nca.grid))
        self.assertTrue(np.all(nca.grid[:7] == 0))
        nca.grid.fill(0)
        nca.step(10)
        self.assertTrue(np.all(nca.grid == 0))

    def test_nca_shared_rule_translation_equivariance_and_weight_roundtrip(self):
        first = NCA(size=24, channels=4, hidden=6)
        first.reset([0.16, 0.06, 0.01, 0.02, 1], seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            first.save_weights(path)
            second = NCA(size=24, channels=4, hidden=6, rule_seed=99, weights_path=path)
            second.reset([0.16, 0.06, 0.01, 0.02, 1], seed=7)
            second.grid = np.roll(first.grid, (3, 5), axis=(0, 1))
            first.step(5)
            second.step(5)
            np.testing.assert_allclose(second.grid, np.roll(first.grid, (3, 5), axis=(0, 1)), atol=1e-7)

    def test_lenia_fft_matches_direct_periodic_convolution(self):
        model = Lenia(size=16)
        model.reset([0.15, 0.035, 0.1, 0.25, 0.5], seed=7)
        shifted = np.fft.ifftshift(model.kernel)
        direct = np.zeros_like(model.grid)
        for y, x in np.argwhere(shifted > 0):
            direct += shifted[y, x] * np.roll(model.grid, (y, x), axis=(0, 1))
        np.testing.assert_allclose(model.potential(), direct, atol=1e-12)
        expected = np.clip(model.grid + model.dt * model.growth(direct), 0, 1)
        model.step()
        np.testing.assert_allclose(model.grid, expected, atol=1e-12)
        model.step(50)
        self.assertTrue(np.isfinite(model.grid).all())
        self.assertTrue(((model.grid >= 0) & (model.grid <= 1)).all())


class TestControlledCells(unittest.TestCase):
    def test_particles_conserve_mass_and_never_teleport(self):
        model = ControlledCells()
        model.configure_narrative(4, [{"name": "split", "frame_range": [0, 3]}])
        model.reset([0.18, 0.8, 0.06, 25, 0.015], seed=11)
        for _ in range(4):
            before = model.positions.copy()
            model.step()
            self.assertLessEqual(np.linalg.norm(model.positions - before, axis=1).max(), 1.5 + 1e-9)
            self.assertAlmostEqual(model.density().sum(), model.num_particles, places=8)
        self.assertFalse(model.stats()["autonomous_reproduction"])

    def test_controlled_sequence_meets_unchanged_acceptance(self):
        phases = [{"name": name, "frame_range": [30 * i, 30 * i + 29]}
                  for i, name in enumerate(("birth", "split", "fusion"))]
        for seed in (3, 71):
            with self.subTest(seed=seed):
                model = ControlledCells()
                model.configure_narrative(90, phases)
                model.reset([0.18, 0.8, 0.06, 25, 0.015], seed=seed)
                frames = []
                for _ in range(90):
                    model.step()
                    frames.append(np.asarray(model.render()))
                result = score_narrative_trajectory(frames, {"phases": phases})
                self.assertTrue(result["accepted"], result["failure_reasons"])
                self.assertEqual(result["actual_component_sequence"], [1, 2, 1])
                self.assertGreaterEqual(min(p["qualified_sustain_ratio"] for p in result["phases"]), 0.65)


if __name__ == "__main__":
    unittest.main()
