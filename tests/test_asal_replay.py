from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.asal_engine.engine import ASALEngine
from research.asal_engine.replay import verify_run
from research.asal_engine.scoring import resolve_config
from research.asal_engine.substrates.nca import NCA


def config(substrate="nca"):
    params = {"size": 24}
    low, high = [0.1, 0.04, 0.01, 0.01, 1.0], [0.25, 0.12, 0.1, 0.1, 1.0]
    if substrate == "boids":
        params = {"num_boids": 16, "width": 32, "height": 32, "keep_largest_component": False}
        low, high = [0, 0, 0, 8, 1], [1, 1, 1, 16, 3]
    return {
        "prompt": "a biological cell", "foundation_model": {"name": "morphology_judge_stub"},
        "substrate": {"name": substrate, "params": params},
        "runtime": {"steps": 6, "substeps": 1},
        "search": {"seed": 13, "iters": 1, "pop": 2, "keep": 2, "sigma": 0.03,
                   "theta_low": low, "theta_high": high},
    }


class TestSeedAndReplay(unittest.TestCase):
    def test_nca_uses_local_seed_and_does_not_change_global_rng(self):
        outputs = []
        for global_seed in (1, 999):
            np.random.seed(global_seed)
            expected = np.random.RandomState(global_seed).rand(4)
            nca = NCA(size=16)
            nca.reset([1] * 5, seed=7)
            nca.step(5)
            outputs.append(np.asarray(nca.render()))
            np.testing.assert_array_equal(np.random.rand(4), expected)
        np.testing.assert_array_equal(*outputs)
        nca.reset([1] * 5, seed=8)
        nca.step(5)
        self.assertFalse(np.array_equal(outputs[0], nca.render()))

    @patch("research.asal_engine.engine.save_mp4", side_effect=RuntimeError("no video encoder"))
    def test_search_and_lossless_replay_are_reproducible(self, video):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("nca", "reaction_diffusion", "boids", "lenia"):
                with self.subTest(substrate=name):
                    original = config(name)
                    before = deepcopy(original)
                    results = []
                    frames = []
                    for index, global_seed in enumerate((1, 99)):
                        np.random.seed(global_seed)
                        run = Path(tmp) / f"{name}-{index}"
                        results.append(ASALEngine(original, run).run())
                        with np.load(run / "trajectory.npz") as trajectory:
                            frames.append(trajectory["frames"])
                        replay = verify_run(run, resimulate=True)
                        self.assertTrue(replay["verified"], replay)
                        self.assertTrue(replay["resimulation_matches"], replay)
                        self.assertEqual(replay["absolute_difference"], 0.0)
                    self.assertEqual(original, before)
                    self.assertEqual(results[0]["best_score"], results[1]["best_score"])
                    self.assertEqual(results[0]["best_theta"], results[1]["best_theta"])
                    np.testing.assert_array_equal(*frames)
                    self.assertIsNone(results[0]["mp4"])

    @patch("research.asal_engine.engine.save_mp4", side_effect=RuntimeError("unavailable"))
    def test_narrative_artifacts_keep_acceptance_and_phase_details(self, video):
        cfg = config()
        cfg["narrative"] = {"enabled": True, "phases": [
            {"name": "birth", "frame_range": [0, 1]}, {"name": "split", "frame_range": [2, 3]},
            {"name": "fusion", "frame_range": [4, 5]},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            result = ASALEngine(cfg, run).run()
            summary = json.loads((run / "narrative_summary.json").read_text())
            self.assertEqual(result["narrative_accepted"], summary["accepted"])
            self.assertIn("continuity", summary)
            for phase in summary["phases"]:
                self.assertIn("longest_qualified_run", phase)
                self.assertIn("sustain_ratio", phase)
                self.assertIn("failure_reasons", phase)
            self.assertTrue(verify_run(run)["verified"])
            with (run / "trajectory.npz").open("ab") as stream:
                stream.write(b"corruption")
            broken = verify_run(run)
            self.assertFalse(broken["verified"])
            self.assertIn("trajectory.npz", broken["file_errors"])

    @patch("research.asal_engine.engine.save_mp4", side_effect=RuntimeError("unavailable"))
    def test_changed_source_is_reported_even_when_scores_match(self, video):
        with tempfile.TemporaryDirectory() as tmp:
            ASALEngine(config(), tmp).run()
            with patch("research.asal_engine.replay.source_hashes", return_value={}):
                replay = verify_run(tmp)
            self.assertTrue(replay["score_matches"])
            self.assertFalse(replay["verified"])
            self.assertTrue(replay["changed_sources"])

    def test_invalid_seed_is_rejected_and_missing_seed_is_resolved(self):
        cfg = config()
        del cfg["search"]["seed"]
        self.assertEqual(resolve_config(cfg)["search"]["seed"], 0)
        for seed in (None, -1, 2**32, True, 1.5):
            cfg["search"]["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                resolve_config(cfg)


if __name__ == "__main__":
    unittest.main()
