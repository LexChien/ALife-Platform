import unittest

from core.model_registry import get_model_spec, list_model_specs


class ModelRegistryTests(unittest.TestCase):
    def test_registry_lists_bootstrap_and_owned_specs(self):
        specs = list_model_specs()
        model_ids = {spec.model_id for spec in specs}
        self.assertIn("gemma_bootstrap_v1", model_ids)
        self.assertIn("clone_lora_v1", model_ids)
        self.assertIn("planner_lora_v1", model_ids)
        self.assertIn("judge_lora_v1", model_ids)

    def test_registry_returns_specific_spec(self):
        spec = get_model_spec("gemma_bootstrap_v1")
        self.assertEqual(spec.family, "gemma")
        self.assertEqual(spec.runtime_backend, "llama_cpp")

    def test_registry_fails_for_unknown_model_id(self):
        with self.assertRaises(KeyError):
            get_model_spec("missing_model")


if __name__ == "__main__":
    unittest.main()

