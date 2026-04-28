import json
import tempfile
import unittest
from pathlib import Path

from genai.web.avatar import resolve_avatar_state
from genai.web.service import GemmaWebService


class GemmaWebServiceTests(unittest.TestCase):
    def test_health_payload_exposes_voice_and_avatar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GemmaWebService(
                config_path="configs/genai/genai_baseline.yaml",
                profile="cpu_smoke",
                host="127.0.0.1",
                port=8081,
                history_turns=4,
                run_base=tmpdir,
            )
            payload = service.health_payload()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["service"], "gemma_web")
            self.assertIn("voice", payload)
            self.assertIn("avatar", payload)
            self.assertEqual(payload["session_schema_version"], "1.0")

    def test_chat_rejects_empty_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GemmaWebService(
                config_path="configs/genai/genai_baseline.yaml",
                profile="cpu_smoke",
                host="127.0.0.1",
                port=8081,
                history_turns=4,
                run_base=tmpdir,
            )
            with self.assertRaises(ValueError):
                service.chat("session-a", "   ")

    def test_chat_writes_session_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GemmaWebService(
                config_path="configs/genai/genai_baseline.yaml",
                profile="cpu_smoke",
                host="127.0.0.1",
                port=8081,
                history_turns=4,
                run_base=tmpdir,
            )
            payload = service.chat("session-a", "你好")
            session_file = Path(service.sessions_dir) / "session-a.json"
            self.assertTrue(session_file.exists())
            saved = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["session_id"], "session-a")
            self.assertEqual(saved["schema_version"], "1.0")
            self.assertEqual(saved["message_count"], 2)
            self.assertEqual(saved["turn_count"], 1)
            self.assertEqual(saved["messages"][0]["role"], "user")
            self.assertEqual(saved["messages"][1]["role"], "assistant")

    def test_reset_increments_reset_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GemmaWebService(
                config_path="configs/genai/genai_baseline.yaml",
                profile="cpu_smoke",
                host="127.0.0.1",
                port=8081,
                history_turns=4,
                run_base=tmpdir,
            )
            service.chat("session-a", "你好")
            reset_payload = service.reset("session-a")
            self.assertEqual(reset_payload["session"]["reset_count"], 1)
            session_file = Path(service.sessions_dir) / "session-a.json"
            saved = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["message_count"], 0)
            self.assertEqual(saved["reset_count"], 1)


class AvatarStateTests(unittest.TestCase):
    def test_avatar_state_prefers_error(self):
        state = resolve_avatar_state(
            conversation_state="ready",
            mic_state="mic_ready",
            speech_state="speech_ready",
            has_error=True,
        )
        self.assertEqual(state, "error")

    def test_avatar_state_prefers_listening_over_ready(self):
        state = resolve_avatar_state(
            conversation_state="ready",
            mic_state="mic_listening",
            speech_state="speech_ready",
        )
        self.assertEqual(state, "listening")

    def test_avatar_state_speaking_after_reply(self):
        state = resolve_avatar_state(
            conversation_state="reply_received",
            mic_state="mic_ready",
            speech_state="speech_playing",
        )
        self.assertEqual(state, "speaking")


if __name__ == "__main__":
    unittest.main()
