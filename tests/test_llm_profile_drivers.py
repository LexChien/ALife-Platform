from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from genai.llm.adapter import LLMRequest
from genai.llm.backends.llama_cpp import LlamaCppAdapter


class TestProfileDrivers(unittest.TestCase):
    def test_profiles_reach_python_and_actual_subprocess_arguments(self):
        formats = {None: None, "clone": "response=[Response]", "judge": "score=[Score]",
                   "planner": "next_step=[NextStep]"}
        request = LLMRequest(prompt="question", context="known context", system="base system")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            for profile, expected in formats.items():
                with self.subTest(profile=profile):
                    adapter = LlamaCppAdapter("gemma", str(model), driver="python",
                                              prompt_profile=profile, lineage={"base": "test"})
                    adapter._llm = MagicMock(return_value={"choices": [{"text": "Answer."}]})
                    response = adapter.generate(request)
                    prompt = adapter._llm.call_args.args[0]
                    self.assertIn("known context", prompt)
                    adapter.driver = "subprocess"
                    with patch.object(adapter, "_cli_resolved_path", return_value="/test/llama-completion"), \
                            patch("genai.llm.backends.llama_cpp.subprocess.run") as run:
                        run.return_value = subprocess.CompletedProcess([], 0, "Answer.", "")
                        response = adapter.generate(request)
                    argv = run.call_args.args[0]
                    system = argv[argv.index("-sys") + 1]
                    self.assertIn("base system", system)
                    self.assertIn("known context", argv[argv.index("-p") + 1])
                    if expected:
                        self.assertIn(expected, prompt)
                        self.assertIn(expected, system)
                    else:
                        self.assertNotIn("Format requirements", prompt + system)
                    self.assertEqual(response.runtime["prompt_profile_applied"], expected is not None)
                    self.assertEqual(response.runtime["lineage"], {"base": "test"})

    def test_extractor_does_not_wrap_answer_in_profile_again(self):
        adapter = LlamaCppAdapter("gemma", "unused", prompt_profile="clone")
        request = adapter._extractor_request(LLMRequest(prompt="question"), "draft")
        self.assertNotIn("Format requirements", adapter.build_prompt(request))
        self.assertNotIn("Format requirements", adapter._subprocess_system(request))
        self.assertTrue(request.metadata["disable_reasoning_extractor"])


if __name__ == "__main__":
    unittest.main()
