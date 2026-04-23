import unittest

from core.model_spec import ModelSpec


class ModelSpecTests(unittest.TestCase):
    def test_model_spec_roundtrip(self):
        raw = {
            "model_id": "gemma_bootstrap_v1",
            "family": "gemma",
            "role": "bootstrap",
            "runtime_backend": "llama_cpp",
            "artifact_format": "gguf",
            "artifact_path": "models/gemma/gemma.gguf",
            "hardware_target": "jetson_orin_nano_8gb",
            "prompt_profile": "default",
            "quantization": "q4_k_m",
            "lineage": {
                "base_model": "gemma-4-bootstrap",
                "tuning_type": "none",
                "dataset": "bootstrap_runtime_only",
                "version": "0.1",
            },
        }
        spec = ModelSpec.from_dict(raw)
        self.assertEqual(spec.model_id, "gemma_bootstrap_v1")
        self.assertEqual(spec.model_path, "models/gemma/gemma.gguf")
        self.assertEqual(spec.to_dict()["runtime_backend"], "llama_cpp")

    def test_model_spec_requires_core_fields(self):
        with self.assertRaises(ValueError):
            ModelSpec.from_dict({"model_id": "broken"})


if __name__ == "__main__":
    unittest.main()

