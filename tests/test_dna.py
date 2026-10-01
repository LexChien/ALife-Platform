import random
import tempfile
import unittest
from pathlib import Path

from dna import Genome, GenomeStore, TRAIT_SPEC, crossover, evolve, express_persona, mutate


class GenomeTests(unittest.TestCase):
    def test_roundtrip_and_stable_id(self):
        g = Genome(theta=[0.1, 0.2, 0.3], traits={"warmth": 0.9}, substrate="lenia")
        with tempfile.TemporaryDirectory() as tmp:
            path = g.save(Path(tmp) / "g.json")
            back = Genome.load(path)
        self.assertEqual(back.genome_id, g.genome_id)
        self.assertEqual(back.traits["warmth"], 0.9)
        self.assertEqual(set(back.traits), set(TRAIT_SPEC))

    def test_id_ignores_non_heritable_fields(self):
        a = Genome(theta=[1.0], fitness=0.1)
        b = Genome(theta=[1.0], fitness=0.9, generation=5)
        self.assertEqual(a.genome_id, b.genome_id)
        self.assertNotEqual(a.genome_id, Genome(theta=[1.1]).genome_id)

    def test_tampered_file_rejected(self):
        data = Genome(theta=[0.5]).to_dict()
        data["theta"] = [0.6]
        with self.assertRaises(ValueError):
            Genome.from_dict(data)

    def test_unknown_trait_and_clipping(self):
        with self.assertRaises(ValueError):
            Genome(theta=[0.0], traits={"telepathy": 1.0})
        self.assertEqual(Genome(theta=[0.0], traits={"warmth": 3.0}).traits["warmth"], 1.0)


class EvolutionTests(unittest.TestCase):
    def test_mutate_is_seeded_and_records_parent(self):
        g = Genome(theta=[0.0, 0.0])
        a = mutate(g, random.Random(7))
        b = mutate(g, random.Random(7))
        self.assertEqual(a.genome_id, b.genome_id)
        self.assertEqual(a.parents, [g.genome_id])
        self.assertEqual(a.generation, 1)

    def test_crossover_mixes_parents(self):
        a = Genome(theta=[0.0] * 8, traits={k: 0.0 for k in TRAIT_SPEC})
        b = Genome(theta=[1.0] * 8, traits={k: 1.0 for k in TRAIT_SPEC})
        c = crossover(a, b, random.Random(3))
        self.assertEqual(sorted(c.parents), sorted([a.genome_id, b.genome_id]))
        self.assertTrue(set(c.theta) <= {0.0, 1.0})

    def test_evolve_monotone_best_and_improves(self):
        target = [0.3, -0.2, 0.5]
        fit = lambda g: -sum((x - t) ** 2 for x, t in zip(g.theta, target))
        out = evolve([Genome(theta=[0.0, 0.0, 0.0])], fit, generations=25, population=10, seed=1)
        bests = [h["best_fitness"] for h in out["history"]]
        self.assertTrue(all(b2 >= b1 for b1, b2 in zip(bests, bests[1:])))
        self.assertGreater(bests[-1], bests[0])
        again = evolve([Genome(theta=[0.0, 0.0, 0.0])], fit, generations=25, population=10, seed=1)
        self.assertEqual(out["best"].genome_id, again["best"].genome_id)


class StoreAndExpressionTests(unittest.TestCase):
    def test_store_lineage_and_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GenomeStore(tmp)
            root = Genome(theta=[0.0])
            child = mutate(root, random.Random(0))
            store.put(root)
            store.put(child)
            store.put(child)  # idempotent lineage
            store.set_current(child.genome_id)
            reopened = GenomeStore(tmp)
            self.assertEqual(reopened.current().genome_id, child.genome_id)
            self.assertEqual(len(reopened.lineage()), 2)
            self.assertEqual(reopened.ancestry(child.genome_id), [child.genome_id, root.genome_id])

    def test_expression_ranges_and_inheritance(self):
        cold = express_persona(Genome(theta=[0.0], traits={"warmth": 0.0, "empathy": 0.0, "playfulness": 0.0, "curiosity": 0.0, "verbosity": 0.0}))
        warm = express_persona(Genome(theta=[0.0], traits={"warmth": 1.0, "empathy": 1.0, "playfulness": 1.0, "curiosity": 1.0, "verbosity": 1.0}))
        self.assertLess(cold["temperature"], warm["temperature"])
        self.assertLess(cold["max_tokens"], warm["max_tokens"])
        self.assertTrue(0.3 <= cold["temperature"] <= 0.8 and 0.3 <= warm["temperature"] <= 0.8)
        self.assertTrue(any("feelings" in line for line in warm["guidance"]))
        self.assertFalse(any("feelings" in line for line in cold["guidance"]))
        self.assertIn("warm", warm["tone"])


if __name__ == "__main__":
    unittest.main()
