import unittest

import numpy as np

from dna.lenia_fitness import FILL_AREA, center_of_mass, com_travel, fitness_v1, fitness_v2, localization, morphology


def blob(size=64, r=5, cx=32, cy=32):
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(xx - cx, yy - cy)
    return np.clip(1.0 - d / r, 0, 1) * (d < r)


def rings(size=64, period=6):
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(xx - size / 2, yy - size / 2)
    return 0.5 + 0.5 * np.sin(2 * np.pi * d / period)


class LeniaFitnessTests(unittest.TestCase):
    def test_single_blob_is_localized(self):
        m = morphology(blob(r=8))
        self.assertEqual(m["components"], 1)
        self.assertAlmostEqual(m["largest_share"], 1.0)
        self.assertGreater(localization(m), 0.5)

    def test_space_filling_rings_get_no_localization_and_penalty(self):
        g = rings()
        m = morphology(g)
        self.assertGreater(m["area"], FILL_AREA)
        self.assertEqual(localization(m), 0.0)
        r = fitness_v2(g, [m["area"]] * 10)
        self.assertGreater(r["fill_penalty"], 0.0)
        self.assertLessEqual(r["score"], 0.0)

    def test_v1_prefers_rings_but_v2_prefers_blob(self):
        """Regression for the round-1 gaming: v1 rewards texture, v2 rewards the creature."""
        b, g = blob(r=8), rings()
        acts_b, acts_g = [float(b.mean())] * 30, [float(g.mean())] * 30
        self.assertGreater(fitness_v1(g, acts_g), fitness_v1(b, acts_b))
        mb, mg = morphology(b), morphology(g)
        self.assertGreater(fitness_v2(b, [mb["area"]] * 10)["score"], fitness_v2(g, [mg["area"]] * 10)["score"])

    def test_empty_and_speck_score_zero(self):
        self.assertEqual(fitness_v2(np.zeros((64, 64)), [0.0] * 10)["score"], 0.0)
        speck = np.zeros((64, 64)); speck[10, 10] = 1.0
        self.assertEqual(fitness_v2(speck, [1 / 4096] * 10)["score"], 0.0)

    def test_two_blobs_less_localized_than_one(self):
        two = np.maximum(blob(r=6, cx=16, cy=16), blob(r=6, cx=48, cy=48))
        self.assertLess(localization(morphology(two)), localization(morphology(blob(r=6))))

    def test_blob_across_torus_edge_is_one_component(self):
        g = np.roll(blob(r=6), 32, axis=1)  # centre on the vertical wrap seam
        self.assertEqual(morphology(g)["components"], 1)

    def test_unstable_area_is_penalised(self):
        b = blob(r=8); a = morphology(b)["area"]
        stable = fitness_v2(b, [a] * 10)["score"]
        unstable = fitness_v2(b, [a * (1 + 0.5 * (i % 2)) for i in range(10)])["score"]
        self.assertLess(unstable, stable)

    def test_frozen_initial_patch_is_not_rewarded(self):
        rng = np.random.RandomState(0)
        init = blob(r=6) * rng.uniform(0.4, 1, (64, 64))
        a = morphology(init)["area"]
        frozen = fitness_v2(init * 0.999, [a] * 10, initial_grid=init)
        self.assertLess(frozen["transformation"], 0.01)
        self.assertLess(frozen["score"], 0.1)
        moved = np.roll(init, 20, axis=0)
        self.assertGreater(fitness_v2(moved, [a] * 10, initial_grid=init)["score"], 1.0)

    def test_com_travel_detects_motion_across_seam(self):
        frames = [np.roll(blob(r=5), k, axis=1) for k in range(0, 40, 4)]
        coms = [center_of_mass(f) for f in frames]
        self.assertAlmostEqual(com_travel(coms, 64), 36.0, delta=1.0)
        self.assertAlmostEqual(com_travel([center_of_mass(blob())] * 5, 64), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
