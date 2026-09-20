"""Runtime flags must honor the declared CPU and reproducibility configuration."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from genai.llm.adapter import LLMRequest
from genai.llm.backends.llama_cpp import LlamaCppAdapter


class LlamaRuntimeLimitsTests(unittest.TestCase):
    def test_subprocess_honors_cpu_batch_seed_and_repacking(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            adapter = LlamaCppAdapter.from_config({
                "model_family": "gemma", "model_path": str(model), "driver": "subprocess",
                "n_gpu_layers": 0, "batch_size": 64, "ubatch_size": 32,
                "seed": 42, "repack": False, "subprocess_timeout": 180,
            })
            with patch.object(adapter, "_cli_resolved_path", return_value="/test/llama-completion"), \
                    patch("genai.llm.backends.llama_cpp.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "Redwood-47 [end of text]\n", "")
                response = adapter.generate(LLMRequest(prompt="What is the code?"))
            argv = run.call_args.args[0]
            for key, expected in (("-ngl", "0"), ("-b", "64"), ("-ub", "32"), ("--seed", "42")):
                self.assertEqual(argv[argv.index(key) + 1], expected)
            self.assertIn("--no-repack", argv)
            self.assertEqual(run.call_args.kwargs["timeout"], 180)
            self.assertEqual(response.text, "Redwood-47")
            self.assertIn("[end of text]", response.raw["stdout"])
            self.assertFalse(response.runtime["repack"])
            self.assertEqual(response.runtime["seed"], 42)

    def test_cli_end_marker_is_removed_only_at_the_end(self):
        adapter = LlamaCppAdapter("gemma", "unused")
        self.assertEqual(adapter._clean_subprocess_output("The marker [end of text] is a sentinel."),
                         "The marker [end of text] is a sentinel.")
        self.assertEqual(adapter._clean_subprocess_output("Answer. [end of text]\n"), "Answer.")


if __name__ == "__main__":
    unittest.main()
