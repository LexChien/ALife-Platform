"""Tests for Plan 44 M4-M7 and Plan 45 code (Min division, heredity, population engine, report).

Author / 作者: Grok Bot ALife Project - Andreeva, 2026-11-23 (Asia/Taipei).
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cell import heredity, metabolism
from cell.genome_map import build_gene_table
from cell.min_system import MinParams, deterministic_profile, site_from_profile
from cell.params import load_params
from cell.state import initial_state

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


class TestMinSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mp = MinParams.from_params(load_params())

    def test_huang_parameters_loaded(self):
        self.assertEqual(self.mp.D_um2_s, 2.5)
        self.assertEqual(self.mp.s_de, 0.7)
        self.assertEqual(self.mp.mind_per_um, 1000.0)

    def test_oscillates_with_central_minimum_at_4um(self):
        r = deterministic_profile(self.mp, 4.0, dx=0.1, t_end_s=200.0, avg_from_s=80.0)
        self.assertIsNotNone(r["period_s"])
        self.assertTrue(30.0 <= r["period_s"] <= 52.0, r["period_s"])      # Huang 2003: ~40 s
        self.assertTrue(0.45 <= r["site_fraction"] <= 0.55, r["site_fraction"])
        self.assertAlmostEqual(r["mind_conservation"], 1.0, places=6)

    def test_site_from_profile_symmetric(self):
        x = np.linspace(-1, 1, 41)
        self.assertAlmostEqual(site_from_profile(x ** 2 + 1), 0.5, places=6)


class TestPrimingAndStarvation(unittest.TestCase):
    def test_no_atp_no_glycolysis(self):
        p = load_params(); g = build_gene_table(p); s = initial_state(p, g)
        self.assertGreater(metabolism.glycolytic_production(p, g, s), 0)
        s.atp = 0.0
        self.assertEqual(metabolism.glycolytic_production(p, g, s), 0.0)
        s.atp = 1e6; s.extra["glucose"] = 0.0
        self.assertEqual(metabolism.glycolytic_production(p, g, s), 0.0)


class TestHeredity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = heredity.run_tree(0, 1, 0.5, 0.05, seed=123)

    def test_tree_shape_and_verification(self):
        self.assertEqual(len(self.recs), 3)
        v = heredity.verify_lineage(self.recs)
        self.assertEqual(v["n_relations"], 2)
        self.assertEqual(v["n_relations_valid"], 2)
        self.assertEqual(v["max_depth_generations"], 1)
        self.assertLess(v["max_partition_conservation_error"], 1e-9)

    def test_tampering_is_detected(self):
        recs = json.loads(json.dumps(self.recs))
        child = [r for r in recs if r["parent_uid"] is not None][0]
        child["genome"]["theta"][0] = 0.99   # stored genome_id no longer matches
        v = heredity.verify_lineage(recs)
        self.assertLess(v["n_genome_id_valid"], v["n_records"])

    def test_genome_store_chain(self):
        with tempfile.TemporaryDirectory() as d:
            v = heredity.verify_lineage(self.recs, Path(d))
            self.assertEqual(v["store_leaf_chains_reaching_founder"], v["store_leaves_checked"])


@unittest.skipUnless(HAVE_TORCH, "torch not importable")
class TestCellBatchCPU(unittest.TestCase):
    def test_divide_into_conserves_and_death_is_irreversible(self):
        import torch
        from cell.population import CellBatch
        p = load_params(); g = build_gene_table(p)
        cb = CellBatch(p, g, 4, dt=1.0, device="cpu", seed=1, steps_per_graph=5, use_graph=False)
        cb.frozen[0] = 1.0; cb.ready_t[0] = 5.0
        before = (cb.mrna[0].sum().item(), cb.protein[0].sum().item(), cb.lipids[0].item())
        cb.divide_into(torch.tensor([0]), torch.tensor([1]))
        after = (cb.mrna[0].sum() + cb.mrna[1].sum()).item(), (cb.protein[0].sum() + cb.protein[1].sum()).item(), \
            (cb.lipids[0] + cb.lipids[1]).item()
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, delta=1e-3 * max(1.0, a))
        cb.glucose[2] = 0.0
        for _ in range(40):
            cb.advance()
        self.assertEqual(cb.dead[2].item(), 1.0)
        self.assertEqual(cb.dead[3].item(), 0.0)
        cb.glucose[2] = 1.0
        for _ in range(40):
            cb.advance()
        self.assertEqual(cb.post_death_max_atp[2].item(), 0.0)   # no revival


class TestReport(unittest.TestCase):
    def test_m4_gate_from_synthetic_jsonl(self):
        from cell.report import gate_m4
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with (run / "m4_divisions.jsonl").open("w") as fh:
                for gen in range(10):
                    for s in range(4):
                        f = 0.5 + 0.01 * (s - 1.5)
                        fh.write(json.dumps({"gen": gen, "slot": s, "was_ready": True, "volume_left_fraction": f,
                                             "min_period_s": 40.0, "mind_conservation": 1.0, "doubling_s": 6300.0,
                                             "sa_v_at_division_um_inv": 12.0}) + "\n")
            with (run / "m4_generations.jsonl").open("w") as fh:
                for gen in range(10):
                    fh.write(json.dumps({"gen": gen, "not_ready": 0, "nan_state": False, "dead": 0,
                                         "self_made_lipid_fraction_mean": 1 - 0.5 ** (gen + 1),
                                         "max_closure": 0.0}) + "\n")
            g = gate_m4(run)
            self.assertEqual(g["status"], "PASS")
            self.assertEqual(g["consecutive_generations_without_collapse"], 10)
            self.assertAlmostEqual(g["volume_deviation_max"], 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
