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

    def test_llama_cpp_adapter_consumes_prompt_profile_and_lineage(self):
        from genai.llm.backends.llama_cpp import LlamaCppAdapter
        from genai.llm.adapter import LLMRequest
        
        lineage = {"base_model": "test_base", "version": "1.0"}
        adapter = LlamaCppAdapter(
            model_family="gemma",
            model_path="dummy_path.gguf",
            prompt_profile="clone",
            lineage=lineage
        )
        
        self.assertEqual(adapter.prompt_profile, "clone")
        self.assertEqual(adapter.lineage, lineage)
        
        # Verify prompt building formats with clone profile rules
        req = LLMRequest(prompt="Hello", system="You are Lex.")
        prompt_built = adapter.build_prompt(req)
        self.assertIn("Format requirements: [Name] tone=[Tone]", prompt_built)
        self.assertIn("<system>\nYou are Lex.\nFormat requirements: [Name] tone=[Tone]", prompt_built)


if __name__ == "__main__":
    unittest.main()

