import unittest
from types import SimpleNamespace

from digital_clone.decision.policy import ClonePromptBuilder


class ClonePerspectiveTests(unittest.TestCase):
    def test_user_memories_are_labelled_and_second_person_rule_present(self):
        persona = SimpleNamespace(name="Lex Clone", tone="calm", principles=["p"], goals=[])
        built = ClonePromptBuilder().build(persona, [
            {"role": "user", "kind": "dialogue", "content": "我最喜歡的人工生命基質是 Lenia。"},
            {"role": "system", "kind": "profile_fact", "content": "fact"}], "我最喜歡哪一個？")
        self.assertIn('"speaker": "user (我 = the user)"', built["context"])
        self.assertIn('"speaker": "system"', built["context"])
        self.assertNotIn('"role":', built["context"])
        self.assertIn("second person", built["system"])
        self.assertIn("回答時請用「你」", built["system"])


if __name__ == "__main__":
    unittest.main()


class PersonaFactsInPromptTests(unittest.TestCase):
    def test_persona_facts_reach_system_prompt(self):
        from digital_clone.decision.policy import ClonePromptBuilder
        from digital_clone.persona.model import PersonaModel
        persona = PersonaModel(name="ALife Prototype", tone="calm", principles=["p"], goals=["g"],
                               facts=["Lenia 是連續型細胞自動機，不是對話層。"])
        built = ClonePromptBuilder().build(persona, [], "Lenia 是什麼？")
        self.assertIn("Lenia 是連續型細胞自動機", built["system"])
