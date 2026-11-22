"""Plan 44 M0: unit tests for the artificial cell simulation package (cell/)."""
import math
import unittest

import numpy as np

from cell import build_gene_table, load_params
from cell import division, driver, expression_tx_tl, membrane, metabolism, replication
from cell.dna_link import THETA_KEYS, express_cell, neutral_theta
from cell.genome_map import CLASS_ORDER
from cell.state import initial_state
from dna.genome import Genome


class ParamsTest(unittest.TestCase):
    def test_every_parameter_has_a_source(self):
        p = load_params()
        for key, entry in p.constants.items():
            self.assertTrue(str(entry["source"]).strip(), f"{key} has an empty source")

    def test_provenance_report_counts(self):
        p = load_params()
        rep = p.provenance_report()
        self.assertEqual(rep["n_constants"],
                         rep["n_literature"] + rep["n_assumed"] + rep["n_estimate_unmeasured"])
        self.assertGreater(rep["n_literature"], 0)
        # Unmeasured estimates must be visible, not hidden.
        self.assertEqual(set(rep["estimate_unmeasured_keys"]), set(p.unsourced_keys()))

    def test_cited_constants_match_the_papers(self):
        p = load_params()
        self.assertEqual(p["genome_bp"], 543379)          # C0715
        self.assertEqual(p["n_genes_full"], 493)          # C0862
        self.assertEqual(p["n_ribosomes"], 503)           # C0862
        self.assertEqual(p["atp_per_bp_replication"], 1)  # C0862
        self.assertEqual(p["doubling_time_experimental_s"], 6300)  # 105 min


class GeneTableTest(unittest.TestCase):
    def test_core_gene_count_and_classes(self):
        p = load_params()
        g = build_gene_table(p)
        self.assertEqual(len(g), int(p["n_core_genes"]))
        self.assertEqual(sum(len(np.flatnonzero(g.class_mask(c))) for c in CLASS_ORDER), len(g))
        self.assertTrue(np.all(g.length_bp >= 90))

    def test_gene_table_is_deterministic(self):
        p = load_params()
        a, b = build_gene_table(p), build_gene_table(p)
        self.assertTrue(np.array_equal(a.length_bp, b.length_bp))


class ExpressionTest(unittest.TestCase):
    def test_mrna_steady_state_matches_analytic(self):
        p = load_params()
        g = build_gene_table(p)
        analytic = expression_tx_tl.analytic_mrna_steady_state(p, g).sum()
        res = driver.run_single(p, g, driver.RunConfig(max_time_s=2000.0, seed=3, max_generations=99,
                                                       telemetry_every_s=10.0))
        tail = [r["mrna_total"] for r in res.telemetry if r["t_s"] >= 1000.0]
        self.assertGreater(len(tail), 10)
        self.assertLess(abs(np.mean(tail) - analytic) / analytic, 0.05)

    def test_translation_requires_mrna(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=1)
        s.mrna[:] = 0.0
        rng = np.random.default_rng(0)
        before = s.protein.sum()
        costs = expression_tx_tl.step(p, g, s, 1.0, rng)
        self.assertEqual(costs["n_proteins"], 0.0)
        self.assertEqual(s.protein.sum(), before)


class AtpLedgerTest(unittest.TestCase):
    def test_pool_is_conserved(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=2)
        pool = s.atp + s.adp
        ledger = metabolism.step(p, g, s, 1.0, {"atp_translation": 1.0e5})
        self.assertLessEqual(ledger["closure_error"], 1e-9)
        self.assertAlmostEqual(s.atp + s.adp, pool, delta=pool * 1e-9)

    def test_replication_costs_one_atp_per_bp(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=4)
        out = replication.step(p, g, s, 1.0)
        self.assertGreater(out["bp"], 0.0)
        self.assertAlmostEqual(out["atp_replication"], out["bp"] * 1.0, places=9)

    def test_atp_never_goes_negative(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=5)
        ledger = metabolism.step(p, g, s, 1.0, {"atp_translation": 1.0e12})
        self.assertGreaterEqual(s.atp, 0.0)
        self.assertGreater(ledger["atp_shortfall"], 0.0)


class GeometryTest(unittest.TestCase):
    def test_area_follows_lipid_count(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=6)
        a0 = s.surface_area_nm2(p)
        s.lipids *= 2.0
        self.assertAlmostEqual(s.surface_area_nm2(p), 2.0 * a0, places=6)

    def test_membrane_step_grows_area_and_charges_atp(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=7)
        a0 = s.surface_area_um2(p)
        out = membrane.step(p, g, s, 1.0)
        self.assertGreater(out["n_lipid"], 0.0)
        self.assertGreater(s.surface_area_um2(p), a0)
        self.assertAlmostEqual(out["atp_lipid"], out["n_lipid"] * p["atp_per_lipid"], places=6)


class DivisionTest(unittest.TestCase):
    def test_division_conserves_molecules_and_halves_geometry(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=8)
        s.replicated_bp = p["genome_bp"]
        rng = np.random.default_rng(0)
        d1, d2 = division.divide(s, rng)
        self.assertTrue(np.array_equal(d1.mrna + d2.mrna, s.mrna))
        self.assertTrue(np.array_equal(d1.protein + d2.protein, s.protein))
        self.assertAlmostEqual(d1.lipids + d2.lipids, s.lipids, places=6)
        self.assertEqual(d1.generation, s.generation + 1)
        self.assertEqual(d1.replicated_bp, 0.0)

    def test_division_requires_replication_and_area(self):
        p = load_params()
        g = build_gene_table(p)
        s = initial_state(p, g, seed=9)
        area0 = s.surface_area_nm2(p)
        self.assertFalse(division.ready_to_divide(p, s, area0))
        s.lipids *= 2.0
        self.assertFalse(division.ready_to_divide(p, s, area0))  # genome not replicated
        s.replicated_bp = p["genome_bp"]
        self.assertTrue(division.ready_to_divide(p, s, area0))


class DnaLinkTest(unittest.TestCase):
    def test_neutral_theta_reproduces_base_parameters(self):
        p = load_params()
        base = {k: p[k] for k in THETA_KEYS}
        genome = Genome(theta=neutral_theta(), substrate="cell")
        out = express_cell(genome, base)
        for key in THETA_KEYS:
            self.assertAlmostEqual(out[key], base[key], places=6)

    def test_theta_scales_rates_and_genome_id_is_stable(self):
        p = load_params()
        base = {k: p[k] for k in THETA_KEYS}
        genome = Genome(theta=[1.0, 0.0, 0.5, 0.5], substrate="cell")
        out = express_cell(genome, base)
        self.assertAlmostEqual(out[THETA_KEYS[0]], base[THETA_KEYS[0]] * 2.0, places=6)
        self.assertAlmostEqual(out[THETA_KEYS[1]], base[THETA_KEYS[1]] * 0.5, places=6)
        self.assertEqual(genome.genome_id, Genome(theta=[1.0, 0.0, 0.5, 0.5], substrate="cell").genome_id)


class DriverTest(unittest.TestCase):
    def test_short_run_is_finite_and_tracks_time(self):
        p = load_params()
        g = build_gene_table(p)
        res = driver.run_single(p, g, driver.RunConfig(max_time_s=120.0, seed=10, telemetry_every_s=10.0))
        self.assertTrue(res.telemetry)
        for row in res.telemetry:
            for key, value in row.items():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value), f"{key} is not finite")
        self.assertLessEqual(res.max_closure_error, 0.01)

    def test_override_is_recorded_in_provenance(self):
        p = load_params()
        p2 = driver.apply_overrides(p, {"k_lipid_per_enzyme_per_s": 1.0})
        self.assertIn("OVERRIDDEN-AT-RUNTIME", p2.source("k_lipid_per_enzyme_per_s"))
        with self.assertRaises(KeyError):
            driver.apply_overrides(p, {"not_a_parameter": 1.0})


class GpuModuleTest(unittest.TestCase):
    def test_cuda_report_shape(self):
        from cell import gpu
        rep = gpu.cuda_report()
        self.assertIn("cuda_available", rep)

    def test_gpu_batch_matches_cpu_scale_on_cpu_device(self):
        from cell import gpu
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not installed in this venv")
        out = gpu.run_batch(batch=4, config=driver.RunConfig(max_time_s=60.0, seed=1), device="cpu")
        self.assertEqual(out["batch"], 4)
        self.assertLessEqual(out["max_closure_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
