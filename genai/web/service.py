from __future__ import annotations

from pathlib import Path
import threading

from core.config import load_config
from core.logger import iso_now, make_run_dir, save_json
from digital_clone.decision.policy import ClonePromptBuilder
from digital_clone.memory.store import MemoryStore
from digital_clone.persona.model import PersonaModel
from genai.llm.adapter import LLMRequest
from genai.llm.factory import create_llm_adapter
from genai.web.avatar import DEFAULT_AVATAR_CONFIG, DEFAULT_VOICE_CONFIG, merge_web_runtime_config
from genai.web.life_engine import LiveLifeManager
from genai.web.life_state import ASALProgressIndex
from genai.web.life_state_bridge import LifeStateBridge
from core.co_evolution import LifeMindFeedbackLoop
from digital_clone.consistency.evaluator import ConsistencyEvaluator
from genai.web.session_store import ChatSession, ChatSessionStore, SESSION_SCHEMA_VERSION
from genai.web.stt import MacOSSpeechTranscriber
from tools.chat_gemma import _build_turn_context, _strict_cleanup_with_retry


ROOT = Path(__file__).resolve().parents[2]


class GemmaWebService:
    def __init__(
        self,
        *,
        config_path: str,
        profile: str | None,
        host: str,
        port: int,
        history_turns: int,
        run_base: str = "runs/chat_gemma_web",
    ) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path, profile=profile)
        self.adapter = create_llm_adapter(self.cfg)
        self.host = host
        self.port = port
        self.history_turns = history_turns
        self._generate_lock = threading.Lock()
        self.base_context = self.cfg.get("context")
        self.system = self.cfg.get("system")
        self.life_cfg = self.cfg.get("life", {})
        self.life_enabled = bool(self.life_cfg.get("enabled", False))
        self.life_index = ASALProgressIndex(root=ROOT, config=self.life_cfg)
        self.life_bridge = LifeStateBridge(ROOT)
        
        self.run_dir = make_run_dir(run_base)
        self.live_life = None
        self.co_evolution = None
        
        if self.life_enabled:
            self.live_life = LiveLifeManager(self.life_cfg, self.run_dir / "live_engine")
            self.live_life.start()
            
            co_evo_cfg = self.cfg.get("co_evolution", {})
            if co_evo_cfg.get("enabled", True):
                self.co_evolution = LifeMindFeedbackLoop(
                    config=co_evo_cfg,
                    artifact_dir=self.run_dir / "co_evolution",
                    clone_evaluator=ConsistencyEvaluator(),
                    llm_adapter=self.adapter,
                    rollback_fn=self.live_life._apply_rollback if hasattr(self.live_life, "_apply_rollback") else None
                )

        self.voice_cfg = merge_web_runtime_config(self.cfg.get("voice"), DEFAULT_VOICE_CONFIG)
        self.avatar_cfg = merge_web_runtime_config(self.cfg.get("avatar"), DEFAULT_AVATAR_CONFIG)
        self.clone_persona = self._build_clone_persona()
        memory_cfg = self.life_cfg.get("memory", {}) if isinstance(self.life_cfg.get("memory"), dict) else {}
        self.clone_memory = MemoryStore(
            collection_name=memory_cfg.get("collection_name", "gemma_web_life"),
            use_vector_db=bool(memory_cfg.get("vector_db", False)),
            persist_directory=memory_cfg.get("persist_directory", ".chroma_db"),
        )
        self.clone_prompt_builder = ClonePromptBuilder()
        self.transcriber = None
        if self.voice_cfg.get("input_provider") == "macos_speech_via_upload":
            self.transcriber = MacOSSpeechTranscriber(locale=self.voice_cfg.get("recognition_lang", "zh-TW"))
        llm_cfg = self.cfg.get("llm", {})
        self.max_tokens = llm_cfg.get("max_tokens")
        self.temperature = llm_cfg.get("temperature")
        self.profile = self.cfg.get("_active_profile")
        self.store = ChatSessionStore()
        self.sessions_dir = self.run_dir / "sessions"
        self.transcriptions_dir = self.run_dir / "transcriptions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        self.server_meta = {
            "schema_version": "1.0",
            "started_at": iso_now(),
            "config_path": config_path,
            "host": host,
            "port": port,
            "profile": self.profile,
            "history_turns": history_turns,
            "session_schema_version": SESSION_SCHEMA_VERSION,
            "llm": {
                "backend": getattr(self.adapter, "backend_name", None),
                "model_family": getattr(self.adapter, "model_family", None),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            "voice": self.voice_cfg,
            "avatar": self.avatar_cfg,
            "life": self.life_payload(),
            "stt": self.transcriber.healthcheck() if self.transcriber else None,
        }
        save_json(self.run_dir / "server_meta.json", self.server_meta)

    def _build_clone_persona(self) -> PersonaModel:
        persona_cfg = self.life_cfg.get("persona") if isinstance(self.life_cfg.get("persona"), dict) else {}
        return PersonaModel(
            name=persona_cfg.get("name", "ALife Prototype"),
            tone=persona_cfg.get("tone", "calm, factual, precise"),
            principles=persona_cfg.get(
                "principles",
                [
                    "state current progress truthfully",
                    "separate verified behavior from prototype limits",
                    "use ASAL artifacts as the visible life state",
                ],
            ),
            goals=persona_cfg.get(
                "goals",
                [
                    "combine ASAL progress, clone memory, Gemma inference, and voice into one local prototype",
                    "answer project status questions from local evidence",
                ],
            ),
            facts=persona_cfg.get(
                "facts",
                [
                    "This prototype uses ASAL run artifacts as the visible artificial-life body.",
                    "The humanoid avatar is not considered fixed unless a real avatar asset exists.",
                    "Gemma web chat and voice are interaction layers, not proof that ASAL research is complete.",
                ],
            ),
        )

    def health_payload(self) -> dict:
        return {
            "ok": True,
            "service": "gemma_web",
            "host": self.host,
            "port": self.port,
            "profile": self.profile,
            "run_dir": str(self.run_dir),
            "session_schema_version": SESSION_SCHEMA_VERSION,
            "llm_healthcheck": self.adapter.healthcheck(),
            "voice": self.voice_cfg,
            "avatar": self.avatar_cfg,
            "life": self.life_payload(),
            "stt": self.transcriber.healthcheck() if self.transcriber else None,
        }

    def life_payload(self) -> dict:
        if not self.life_enabled:
            return {
                "ok": True,
                "enabled": False,
                "schema_version": "1.0",
                "summary": "life progress integration is disabled in this config",
            }
        payload = self.life_index.snapshot()
        payload["ok"] = True
        if self.live_life:
            stats = self.live_life.substrate.stats()
            payload["live_frame"] = self.live_life.get_latest_frame_b64()
            payload["live_state"] = self.live_life.current_state
            payload["energy"] = stats.get("energy", 0.0)
            payload["num_components"] = stats.get("num_components", 0)
            payload["clamped"] = stats.get("clamped", False)
            payload["is_stable"] = stats.get("is_stable", True)
            
            # Co-evolution specific metrics
            payload["life_likeness"] = (
                self.live_life.current_combined_score 
                if hasattr(self.live_life, "current_combined_score") 
                else 0.5
            )
            if self.co_evolution:
                payload["co_evolution_action"] = (
                    self.co_evolution.history[-1].action 
                    if hasattr(self.co_evolution, "history") and self.co_evolution.history 
                    else "CONTINUE"
                )
        return payload

    def read_life_artifact(self, run_id: str, asset: str) -> tuple[bytes, str]:
        path = self.life_index.resolve_artifact(run_id, asset)
        suffix = path.suffix.lower()
        content_type = {
            ".gif": "image/gif",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".mp4": "video/mp4",
            ".png": "image/png",
        }.get(suffix, "application/octet-stream")
        return path.read_bytes(), content_type

    def transcribe(self, session_id: str | None, audio_bytes: bytes, content_type: str) -> dict:
        if not self.transcriber:
            raise RuntimeError("transcriber is not configured")
        session = self.store.get_or_create(session_id)
        result = self.transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            content_type=content_type,
            outdir=self.transcriptions_dir,
        )
        payload = {
            "ok": True,
            "session_id": session.session_id,
            "transcript": result.transcript,
            "audio_artifacts": {
                "input": str(result.input_path),
                "normalized": str(result.normalized_path),
            },
        }
        save_json(self.transcriptions_dir / f"{session.session_id}_latest.json", payload)
        return payload

    def chat(self, session_id: str | None, message: str) -> dict:
        user_message = (message or "").strip()
        if not user_message:
            raise ValueError("message is required")

        if self.live_life:
            self.live_life.set_state("thinking", prompt=user_message)

        session = self.store.get_or_create(session_id)
        transcript = session.transcript_pairs()
        selected_transcript = transcript[-self.history_turns:] if self.history_turns > 0 else transcript
        turn_context = _build_turn_context(
            self.base_context,
            selected_transcript,
            self.history_turns,
        )
        request_system = self.system
        life_snapshot = None
        if self.life_enabled:
            life_snapshot = self.life_index.snapshot()
            memory_cfg = self.life_cfg.get("memory", {}) if isinstance(self.life_cfg.get("memory"), dict) else {}
            retrieved = self.clone_memory.retrieve_for_prompt(
                user_message,
                limit=int(memory_cfg.get("retrieval_limit", 5)),
            )
            built = self.clone_prompt_builder.build(
                self.clone_persona,
                retrieved,
                user_message,
                extra_context=self.life_index.context_text(life_snapshot),
            )
            context_parts = [turn_context]
            if built.get("context"):
                context_parts.append("Digital Clone / ASAL memory:\n" + built["context"])
            turn_context = "\n\n".join(part for part in context_parts if part)
            request_system = "\n".join(part for part in [self.system, built.get("system")] if part)

        request = LLMRequest(
            prompt=user_message,
            context=turn_context,
            system=request_system,
            life_context=self.life_bridge.get_narrative_context(life_snapshot) if self.life_enabled else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            metadata={"transcript": selected_transcript},
        )
        with self._generate_lock:
            if self.live_life:
                self.live_life.set_state("thinking")
            response = self.adapter.generate(request)
            if self.live_life:
                self.live_life.set_state("speaking")
            cleaned_text, cleanup_meta = _strict_cleanup_with_retry(
                self.adapter,
                request,
                response.text,
            )
        
        if self.live_life:
            self.live_life.set_state("idle")

        if self.co_evolution:
            # P1: Heart-Body Resonance Test via co-evolution loop
            asal_stats = self.live_life.substrate.stats() if self.live_life else {}
            # Simplified clone state for feedback
            clone_state = {
                "persona_drift": 0.0, # To be calculated via evaluator
                "persona": self.clone_persona.to_dict()
            }
            genai_output = {"input": user_message, "output": cleaned_text}
            self.co_evolution.step(asal_stats, clone_state, genai_output)

        session.append("user", user_message)
        session.append("assistant", cleaned_text)
        if self.life_enabled:
            self.clone_memory.add("user", user_message)
            self.clone_memory.add("assistant", cleaned_text)
        self._save_session(session, cleanup_meta, response.runtime, life_snapshot)
        return {
            "ok": True,
            "schema_version": "1.0",
            "session_id": session.session_id,
            "reply": cleaned_text,
            "life": life_snapshot,
            "cleanup": cleanup_meta,
            "runtime": response.runtime,
            "session": {
                "message_count": session.message_count,
                "turn_count": session.turn_count,
                "reset_count": session.reset_count,
            },
            "messages": session.to_dict()["messages"],
        }

    def reset(self, session_id: str | None) -> dict:
        session = self.store.reset(session_id)
        self._save_session(session, cleanup_meta=None, runtime=None, life_snapshot=None)
        return {
            "ok": True,
            "schema_version": "1.0",
            "session_id": session.session_id,
            "session": {
                "message_count": session.message_count,
                "turn_count": session.turn_count,
                "reset_count": session.reset_count,
            },
            "messages": [],
        }

    def _save_session(
        self,
        session: ChatSession,
        cleanup_meta: dict | None,
        runtime: dict | None,
        life_snapshot: dict | None,
    ) -> None:
        payload = session.to_dict()
        payload["config_path"] = self.config_path
        payload["profile"] = self.profile
        payload["cleanup"] = cleanup_meta
        payload["runtime"] = runtime
        payload["voice"] = self.voice_cfg
        payload["avatar"] = self.avatar_cfg
        payload["life"] = life_snapshot
        save_json(self.sessions_dir / f"{session.session_id}.json", payload)
