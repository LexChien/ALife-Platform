import tempfile
import unittest
from unittest.mock import MagicMock, patch

from digital_clone.engine import DigitalCloneEngine
from digital_clone.eval import score_case, summarize
from genai.llm.adapter import LLMResponse


class CloneBehaviorTests(unittest.TestCase):
    persona = {"id": "quality-test", "name": "Lex Clone", "tone": "calm, analytical",
               "principles": ["maintain consistency"], "goals": [], "facts": []}

    def test_memory_retrieval_does_not_count_as_answer(self):
        row = score_case(self.persona,
                         {"id": "recall", "input": "What is the code?", "must_recall": ["Redwood-47"],
                          "answer_groups": [["Redwood-47"]]},
                         {"output": "I do not know.", "retrieved_memories": ["Code: Redwood-47"],
                          "llm": {"backend": "llama_cpp"}}, {})
        self.assertTrue(row["retrieval"]["pass"])
        self.assertFalse(row["behavior"]["pass"])
        self.assertFalse(row["pass"])
        self.assertEqual(row["reasons"]["failed_checks"], ["expected_answer"])

    def test_display_wrapper_cannot_supply_expected_answer(self):
        row = score_case(self.persona, {"id": "identity", "input": "Who are you?", "must_include": ["Lex Clone"]},
                         {"output": "[Lex Clone] tone=calm response=I am RAGEBOT",
                          "generated_response": "I am RAGEBOT", "llm": {"backend": "llama_cpp"}}, {})
        self.assertFalse(row["checks"]["expected_answer"])

    def test_dummy_echo_is_not_quality_evidence(self):
        row = score_case(self.persona, {"id": "fact", "input": "Code?", "must_include": ["Redwood-47"]},
                         {"output": "[DummyLLM] context=Redwood-47", "llm": {"backend": "dummy"}}, {})
        self.assertFalse(row["pass"])
        self.assertFalse(row["checks"]["real_model"])
        self.assertFalse(summarize([])["all_passed"])

    @patch("digital_clone.engine.MemoryStore")
    @patch("digital_clone.engine.create_llm_adapter")
    def test_engine_preserves_exact_generation_and_raw_payload(self, create_llm, memory_class):
        generated = "The codename is Redwood-47."
        create_llm.return_value.generate.return_value = LLMResponse(
            text=generated, model_family="gemma", backend="llama_cpp", runtime={},
            raw={"stdout": "raw runtime text", "stderr": "diagnostics"})
        memory_class.return_value.retrieve_for_prompt.return_value = [
            {"id": "old", "role": "user", "kind": "dialogue", "content": "Code: Redwood-47"}]
        cfg = {"persona": self.persona, "inputs": ["Code?"], "llm": {}}
        with tempfile.TemporaryDirectory() as directory:
            row = DigitalCloneEngine(cfg, directory).run()["outputs"][0]
        self.assertEqual(row["output"], generated)
        self.assertEqual(row["generated_response"], generated)
        self.assertEqual(row["raw_generation"]["stdout"], "raw runtime text")
        self.assertTrue(row["request"]["metadata"]["disable_prompt_profile"])
        self.assertEqual(row["request"]["prompt"], "Code?")
        self.assertEqual(row["retrieved_records"][0]["id"], "old")
        calls = memory_class.return_value.method_calls
        self.assertEqual(calls[1][0], "retrieve_for_prompt")
        self.assertEqual(calls[2][0], "add")


if __name__ == "__main__":
    unittest.main()
