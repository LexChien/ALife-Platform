from __future__ import annotations

import logging
from pathlib import Path
import re
import threading
import time

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
from genai.web.session_store import ChatSession, ChatSessionStore, SESSION_SCHEMA_VERSION
from genai.web.stt import MacOSSpeechTranscriber
from genai.web.emotion import EmotionState, arousal_from_prosody, detect_text_emotion, modulation, prosody_features
from genai.web.emotion_llm import detect_emotion
from genai.web.voice import MLXWhisperTranscriber, MacSayTTS, WhisperTranscriber, decode_to_pcm, split_for_tts
from genai.llm.reasoning import has_prompt_echo_residue, has_reasoning_leak, looks_like_cli_banner, sanitize_reply
from dna import Genome, GenomeStore, express_persona
from tools.chat_gemma import _build_turn_context, _strict_cleanup_with_retry

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
_REMEMBER_RE = re.compile(r"(記住|記得|别忘|別忘|remember|don't forget)", re.I)


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
        
        self.run_dir = make_run_dir(run_base)
        self.live_life = None
        if self.life_enabled:
            self.live_life = LiveLifeManager(self.life_cfg, self.run_dir / "live_engine")
            self.live_life.start()

        self.voice_cfg = merge_web_runtime_config(self.cfg.get("voice"), DEFAULT_VOICE_CONFIG)
        self.avatar_cfg = merge_web_runtime_config(self.cfg.get("avatar"), DEFAULT_AVATAR_CONFIG)
        self.clone_persona = self._build_clone_persona()
        memory_cfg = self.life_cfg.get("memory", {}) if isinstance(self.life_cfg.get("memory"), dict) else {}
        # MemoryStore auto-enables Chroma when available; it does not take use_vector_db.
        # vector_db=true in config => require persistent Chroma (fail loud if missing).
        # Plan 37 R2: optional multilingual embedding (zh retrieval R@1 0.167 -> 0.833 with e5-small).
        base_collection = memory_cfg.get("collection_name", "gemma_web_life")
        persist_dir = memory_cfg.get("persist_directory", ".chroma_db")
        self.memory_embedding = None
        embedding_fn = None
        collection = base_collection
        if memory_cfg.get("embedding"):
            try:
                from digital_clone.memory.embeddings import PrefixedSentenceTransformerEF, collection_suffix
                embedding_fn = PrefixedSentenceTransformerEF(str(memory_cfg["embedding"]))
                collection = base_collection + collection_suffix(str(memory_cfg["embedding"]))
                self.memory_embedding = str(memory_cfg["embedding"])
            except Exception as exc:  # fail loud in health, keep the default embedding
                self.memory_embedding = f"unavailable: {type(exc).__name__}: {exc}"
        self.clone_memory = MemoryStore(
            collection_name=collection,
            persist_directory=persist_dir,
            require_persistence=bool(memory_cfg.get("vector_db", False)),
            embedding_function=embedding_fn,
        )
        self.memory_migrated = 0
        if embedding_fn is not None and self.clone_memory.use_vector_db:
            try:
                from digital_clone.memory.embeddings import migrate_collection
                self.memory_migrated = migrate_collection(persist_dir, base_collection, self.clone_memory.vector_store)
            except Exception as exc:
                self.memory_migrated = -1
                logger.warning("memory migration from %s failed: %s", base_collection, exc)
        self.clone_prompt_builder = ClonePromptBuilder()
        self.transcriber = None
        self.offline_stt = None
        input_provider = self.voice_cfg.get("input_provider")
        stt_backend = str(self.voice_cfg.get("whisper_backend", "faster_whisper"))
        if (input_provider in {"local_whisper_via_upload", "macos_speech_via_upload"} and stt_backend == "mlx"
                and MLXWhisperTranscriber.available()):
            self.offline_stt = MLXWhisperTranscriber(model_repo=str(self.voice_cfg.get("mlx_whisper_model",
                                                                                      "mlx-community/whisper-large-v3-mlx")))
            if self.voice_cfg.get("whisper_preload", False):
                self.offline_stt.preload()
        elif input_provider in {"local_whisper_via_upload", "macos_speech_via_upload"} and WhisperTranscriber.available():
            self.offline_stt = WhisperTranscriber(model_size=str(self.voice_cfg.get("whisper_model", "small")))
            if self.voice_cfg.get("whisper_preload", False):
                self.offline_stt.preload()
        if input_provider == "local_whisper_via_upload" and self.offline_stt is not None:
            self.transcriber = self.offline_stt
        elif input_provider in {"macos_speech_via_upload", "local_whisper_via_upload"}:
            self.transcriber = MacOSSpeechTranscriber(locale=self.voice_cfg.get("recognition_lang", "zh-TW"))
        self.tts = None
        if self.voice_cfg.get("output_provider") == "macos_say":
            candidate = MacSayTTS(voice=str(self.voice_cfg.get("tts_voice", "Meijia")))
            if candidate.healthcheck()["ok"]:
                self.tts = candidate
        # Plan 37 G4: per-session emotion state.
        self.emotion_cfg = self.cfg.get("emotion", {}) if isinstance(self.cfg.get("emotion"), dict) else {}
        self.emotion_enabled = bool(self.emotion_cfg.get("enabled", True))
        self.emotion_states: dict[str, EmotionState] = {}
        llm_cfg = self.cfg.get("llm", {})
        self.max_tokens = llm_cfg.get("max_tokens")
        self.temperature = llm_cfg.get("temperature")
        # Plan 37 G1/N3: DNA genome inherited into the clone persona. Must run AFTER the
        # llm defaults above, otherwise apply_sampling is silently overwritten (R2 fix).
        self._init_dna()
        self.profile = self.cfg.get("_active_profile")
        self.store = ChatSessionStore()
        self.sessions_dir = self.run_dir / "sessions"
        self.transcriptions_dir = self.run_dir / "transcriptions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        self.tts_dir = self.run_dir / "tts"
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        self._tts_pending: dict[str, float] = {}
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
            "dna": self.dna_payload(),
        }
        save_json(self.run_dir / "server_meta.json", self.server_meta)

    def _init_dna(self) -> None:
        dna_cfg = self.life_cfg.get("dna", {}) if isinstance(self.life_cfg.get("dna"), dict) else {}
        self.dna_cfg = dna_cfg
        self.dna_enabled = bool(dna_cfg.get("enabled", True))
        # Unconfigured => ephemeral per-run store (keeps tests/dummy configs out of the real clone lineage).
        store = dna_cfg.get("store")
        self.genome_store = GenomeStore(ROOT / store if store else self.run_dir / "dna_store")
        genome = self.genome_store.current()
        if genome is None:
            genome = Genome.founder(theta_dim=5, substrate="clone", **dna_cfg.get("founder_traits", {}))
            self.genome_store.put(genome)
            self.genome_store.set_current(genome.genome_id)
        self.genome = genome
        self.genome_expression = express_persona(genome)
        if self.dna_enabled and dna_cfg.get("apply_sampling", False):
            self.temperature = self.genome_expression["temperature"]
            self.max_tokens = self.genome_expression["max_tokens"]

    def dna_payload(self) -> dict:
        return {
            "enabled": self.dna_enabled,
            "genome": self.genome.to_dict(),
            "expression": self.genome_expression,
            "apply_sampling": bool(self.dna_cfg.get("apply_sampling", False)),
            "store": str(self.genome_store.root),
            "lineage_records": len(self.genome_store.lineage()),
            "ancestry": self.genome_store.ancestry(self.genome.genome_id)[:10],
        }

    def _dna_system_text(self) -> str:
        if not self.dna_enabled:
            return ""
        expr = self.genome_expression
        lines = [f"Inherited DNA genome {expr['genome_id']} (generation {expr['generation']}): tone {expr['tone']}."]
        lines.extend(expr.get("guidance", []))
        return " ".join(lines)

    def emotion_state(self, session_id: str) -> EmotionState:
        state = self.emotion_states.get(session_id)
        if state is None:
            state = EmotionState()
            self.emotion_states[session_id] = state
        return state

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
            "stt_offline": self.offline_stt.healthcheck() if self.offline_stt else {"provider": "faster_whisper", "installed": False, "ok": False},
            "tts": self.tts.healthcheck() if self.tts else {"provider": "browser_speech_synthesis", "server_side": False},
            "memory": {"embedding": self.memory_embedding or "chroma_default",
                       "collection": self.clone_memory.collection_name,
                       "vector_db": self.clone_memory.use_vector_db, "migrated": self.memory_migrated},
            "emotion": {"enabled": self.emotion_enabled, "detector": str(self.emotion_cfg.get("detector", "lexicon")) + "+prosody", "sessions": len(self.emotion_states)},
            "dna": {k: v for k, v in self.dna_payload().items() if k in ("enabled", "apply_sampling", "lineage_records")}
            | {"genome_id": self.genome.genome_id, "generation": self.genome.generation},
            "vlm": self.vlm_payload(),
        }

    def vlm_payload(self) -> dict:
        vlm = getattr(self.live_life, "vlm", None) if self.live_life else None
        return {
            "openclip_loaded": vlm is not None,
            "live_score_note": None if vlm is not None else "OpenCLIP unavailable: live semantic score stays -1 (not a measured score)",
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
            payload["live_frame"] = self.live_life.get_latest_frame_b64()
            payload["live_state"] = self.live_life.current_state
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

    def chat(self, session_id: str | None, message: str, *, voice_features: dict | None = None,
             want_tts: bool = False, source: str = "text") -> dict:
        user_message = (message or "").strip()
        if not user_message:
            raise ValueError("message is required")
        t_start = time.time()

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
            scope = session.session_id if memory_cfg.get("scope", "session") == "session" else None
            retrieved = self.clone_memory.retrieve_for_prompt(
                user_message,
                limit=int(memory_cfg.get("retrieval_limit", 5)),
                scope=scope,
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

        emotion_payload = None
        mod = None
        if self.emotion_enabled:
            observed = detect_emotion(self.adapter, user_message,
                                      mode=str(self.emotion_cfg.get("detector", "lexicon")),
                                      threshold=float(self.emotion_cfg.get("hybrid_threshold", 0.7)))
            voice_arousal = arousal_from_prosody(voice_features) if voice_features else None
            state = self.emotion_state(session.session_id).update(observed, voice_arousal=voice_arousal)
            mod = modulation(state, base_voice=self.genome_expression.get("voice") if self.dna_enabled else None)
            emotion_payload = {"observed": observed, "voice_features": voice_features, "voice_arousal": voice_arousal,
                               "state": state.to_dict(), "modulation": mod}
        extra_system = [self._dna_system_text(), mod["system_guidance"] if mod else ""]
        request_system = "\n".join(part for part in [request_system, *extra_system] if part)

        request = LLMRequest(
            prompt=user_message,
            context=turn_context,
            system=request_system,
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
        # Final product-path hygiene (Plan 37 G0.3 / C1).
        cleaned_text = sanitize_reply(cleaned_text)
        hygiene = {
            "reasoning_leak": has_reasoning_leak(cleaned_text),
            "cli_banner": looks_like_cli_banner(cleaned_text),
            "prompt_echo": has_prompt_echo_residue(cleaned_text),
        }
        llm_s = round(time.time() - t_start, 3)
        tts_payload = None
        if want_tts and self.tts is not None and cleaned_text.strip():
            voice = (mod or {}).get("tts", {"rate": 1.0, "pitch": 1.0})
            try:
                tts_payload = self._synthesize_chunked(cleaned_text, voice)
            except Exception as exc:
                tts_payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        session.append("user", user_message)
        session.append("assistant", cleaned_text)
        if self.life_enabled:
            # Explicit "remember ..." statements become global user facts; other dialogue stays session-scoped
            # so one session's emotional context does not leak into another (Plan 37 web eval 05:33 finding).
            kind = "user_fact" if _REMEMBER_RE.search(user_message) else "dialogue"
            self.clone_memory.add("user", user_message, kind=kind, scope=session.session_id)
            self.clone_memory.add("assistant", cleaned_text, scope=session.session_id)
        self._save_session(session, cleanup_meta, response.runtime, life_snapshot,
                           extra={"emotion": emotion_payload, "dna_genome_id": self.genome.genome_id,
                                  "hygiene": hygiene, "source": source, "tts": tts_payload})
        return {
            "ok": True,
            "schema_version": "1.0",
            "session_id": session.session_id,
            "reply": cleaned_text,
            "emotion": emotion_payload,
            "dna": {"genome_id": self.genome.genome_id, "generation": self.genome.generation, "tone": self.genome_expression["tone"]},
            "hygiene": hygiene,
            "tts": tts_payload,
            "timings": {"llm_s": llm_s, "tts_s": tts_payload.get("elapsed_s") if tts_payload else None},
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

    def voice_chat(self, session_id: str | None, audio_bytes: bytes, content_type: str) -> dict:
        """Plan 37 V3: audio -> offline STT -> emotion(prosody) -> Gemma -> TTS in one round trip."""
        t0 = time.time()
        stt = self.offline_stt or self.transcriber
        if stt is None:
            raise RuntimeError("no speech-to-text provider configured")
        session = self.store.get_or_create(session_id)
        result = stt.transcribe_bytes(audio_bytes=audio_bytes, content_type=content_type, outdir=self.transcriptions_dir)
        stt_s = round(time.time() - t0, 3)
        pcm = getattr(result, "pcm", None)
        if pcm is None:
            try:
                pcm = decode_to_pcm(Path(result.input_path))
            except Exception:
                pcm = None
        features = prosody_features(pcm) if pcm is not None else None
        transcript = (result.transcript or "").strip()
        if not transcript:
            return {"ok": False, "error": "empty_transcript", "session_id": session.session_id,
                    "stt_provider": getattr(stt, "provider", "macos_speech"), "voice_features": features,
                    "timings": {"stt_s": stt_s}}
        payload = self.chat(session.session_id, transcript, voice_features=features, want_tts=True, source="voice")
        payload["transcript"] = transcript
        payload["stt_provider"] = getattr(stt, "provider", "macos_speech")
        payload["timings"] = {**payload.get("timings", {}), "stt_s": stt_s, "total_s": round(time.time() - t0, 3)}
        save_json(self.transcriptions_dir / f"{session.session_id}_voice_latest.json",
                  {k: payload[k] for k in ("session_id", "transcript", "reply", "emotion", "tts", "timings", "stt_provider")})
        return payload

    def synthesize(self, session_id: str | None, text: str) -> dict:
        if self.tts is None:
            raise RuntimeError("server-side TTS not configured (browser speech synthesis is the fallback)")
        state = self.emotion_states.get(session_id or "") or EmotionState()
        mod = modulation(state, base_voice=self.genome_expression.get("voice") if self.dna_enabled else None)
        out = self.tts.synthesize(text, self.tts_dir, rate=mod["tts"]["rate"], pitch=mod["tts"]["pitch"])
        return {"ok": True, "url": f"/api/tts/{out['file']}", "emotion_label": mod["label"],
                **{k: out[k] for k in ("duration_s", "wpm", "pbas", "voice", "elapsed_s")}}

    def _synthesize_chunked(self, text: str, voice: dict) -> dict:
        """Plan 37 R2 V-latency: speak the first sentence chunk immediately; synthesize the rest in a
        background thread. Chunk URLs are known up front; GET waits for a pending chunk."""
        import uuid

        chunks = split_for_tts(text) if self.voice_cfg.get("tts_chunking", True) else [text]
        chunks = chunks or [text]
        stems = [uuid.uuid4().hex for _ in chunks]
        first = self.tts.synthesize(chunks[0], self.tts_dir, rate=voice["rate"], pitch=voice["pitch"], stem=stems[0])
        if len(chunks) > 1:
            for st in stems[1:]:
                self._tts_pending[st] = time.time()

            def _rest():
                for chunk, st in zip(chunks[1:], stems[1:]):
                    try:
                        self.tts.synthesize(chunk, self.tts_dir, rate=voice["rate"], pitch=voice["pitch"], stem=st)
                    except Exception as exc:  # recorded; GET will 404 after the wait
                        logger.warning("tts chunk failed: %s", exc)
                    finally:
                        self._tts_pending.pop(st, None)

            threading.Thread(target=_rest, daemon=True).start()
        return {"ok": True, "url": f"/api/tts/{first['file']}", "chunked": len(chunks) > 1,
                "chunks": [{"index": i, "url": f"/api/tts/{st}.wav", "chars": len(c)} for i, (c, st) in enumerate(zip(chunks, stems))],
                "first_chunk_s": first["elapsed_s"], "elapsed_s": first["elapsed_s"],
                **{k: first[k] for k in ("duration_s", "wpm", "pbas", "voice")}}

    def save_mic_check(self, payload: dict) -> dict:
        """Persist a browser mic self-check result (web/gemma_chat/mic_check.html) for the log."""
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        out_dir = self.run_dir / "mic_checks"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"mic_check_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.json"
        save_json(path, {"saved_at": iso_now(), "kind": "REAL_BROWSER_MIC_SELF_CHECK", **payload})
        return {"ok": True, "path": str(path)}

    def read_tts(self, name: str, wait_s: float = 30.0) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{32}\.wav", name or ""):
            raise FileNotFoundError(name)
        path = self.tts_dir / name
        deadline = time.time() + wait_s
        while not path.exists() and name[:-4] in self._tts_pending and time.time() < deadline:
            time.sleep(0.05)
        if not path.exists():
            raise FileNotFoundError(name)
        return path.read_bytes()

    def emotion_payload(self, session_id: str | None) -> dict:
        state = self.emotion_states.get(session_id or "")
        return {"ok": True, "session_id": session_id, "state": state.to_dict() if state else None}

    def reset(self, session_id: str | None) -> dict:
        session = self.store.reset(session_id)
        self.emotion_states.pop(session.session_id, None)
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
        extra: dict | None = None,
    ) -> None:
        payload = session.to_dict()
        if extra:
            payload.setdefault("turn_meta", [])
            prior = self.sessions_dir / f"{session.session_id}.json"
            if prior.exists():
                try:
                    import json as _json
                    payload["turn_meta"] = _json.loads(prior.read_text(encoding="utf-8")).get("turn_meta", [])
                except Exception:
                    payload["turn_meta"] = []
            payload["turn_meta"].append(extra)
        payload["config_path"] = self.config_path
        payload["profile"] = self.profile
        payload["cleanup"] = cleanup_meta
        payload["runtime"] = runtime
        payload["voice"] = self.voice_cfg
        payload["avatar"] = self.avatar_cfg
        payload["life"] = life_snapshot
        save_json(self.sessions_dir / f"{session.session_id}.json", payload)
