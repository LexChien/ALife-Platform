"""Plan 38 J4: DigiClone cognitive loop - speak || think on a resident llama-server.

Per user turn:
  slot 0 (speech):  stable prefix [persona/DNA system] + history, then the LAST user message carries the dynamic
                    context (memory, emotion, tone from the previous appraisal, language lock). Streamed, chunked
                    into sentences; each sentence passes the leak guard + reply guards BEFORE it is emitted/spoken.
                    First-sentence language gate: wrong language -> cancel and regenerate once with a stronger lock.
  slot 1 (thought): private JSON thought (cognition/thought.py) generated concurrently; it updates the persisted
                    self-state/appraisal for the NEXT turn. Speech never sees the raw thought; only enums, the brief
                    and the one-line summary cross over. Thoughts are written to a private JSONL stream and are sent
                    to the HUD only when the user enabled the thought stream (default OFF).
Callbacks: ``emit(event: dict)`` receives {"type": "delta"|"sentence"|"thought"|"state", ...}."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from cognition.appraisal import tone_line
from cognition.language import RETRY_SUFFIX, detect_lang, lock_line
from cognition.leak_guard import SAFE_LINES, check_leak
from cognition.reply_guard import guard_reply
from cognition.self_state import SelfState
from cognition.thought import think
from voice.chunker import SentenceChunker

ROOT = Path(__file__).resolve().parents[1]
ASKS_THOUGHTS = re.compile(r"你(?:現在|此刻)?(?:心裡|腦中)?在想(?:什麼|甚麼|啥)|你的想法是|what(?:'s| is| are) (?:on )?your mind|"
                           r"what are you thinking|your thoughts right now", re.I)


class CognitiveLoop:
    def __init__(self, adapter, *, persona_name: str, think_persona: str, state_path: str | Path | None = None,
                 thoughts_path: str | Path | None = None, think_enabled: bool = True, max_tokens: int = 200,
                 temperature: float = 0.6):
        self.adapter = adapter
        self.persona_name = persona_name
        self.think_persona = think_persona
        self.state = SelfState(state_path) if state_path else SelfState()
        self.thoughts_path = Path(thoughts_path) if thoughts_path else ROOT / "runs" / "digiclone" / "thoughts.jsonl"
        self.think_enabled = think_enabled
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.recent_thoughts: list[dict] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- helpers
    def _private_notes(self) -> list[tuple[str, str]]:
        return [(t.get("private_note", ""), t.get("canary", "")) for t in self.recent_thoughts[-5:]]

    def _sentence_leak(self, sent: str) -> list[str]:
        reasons = []
        for note, canary in self._private_notes() or [("", "")]:
            v = check_leak(sent, private_note=note or None, canary=canary or None)
            reasons += v.reasons
        return sorted(set(reasons))

    def _save_thought(self, th: dict) -> None:
        self.thoughts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.thoughts_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(th, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------------- main
    def on_user_turn(self, user_text: str, *, system: str, history: list[dict], dynamic_context: str = "",
                     memories: list[dict] | None = None, all_memories: list[dict] | None = None,
                     emit: Callable[[dict], None] | None = None, cancel: threading.Event | None = None,
                     turn_id: str | None = None, want_thoughts: bool = False, tool_calls: list | None = None,
                     history_text: str = "") -> dict:
        emit = emit or (lambda e: None)
        cancel = cancel or threading.Event()
        turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        lang = detect_lang(user_text)
        t0 = time.perf_counter()
        # ---- thought on slot 1, concurrently
        th_box: dict = {}
        th_thread = None
        if self.think_enabled:
            def _bg():
                try:
                    th_box["th"] = think(self.adapter, self.think_persona, user_text, history_text, turn_id=turn_id, cancel=cancel)
                except Exception as exc:  # thought failure never blocks speech
                    th_box["error"] = f"{type(exc).__name__}: {exc}"
            th_thread = threading.Thread(target=_bg, daemon=True)
            th_thread.start()
        # ---- speech on slot 0
        st = self.state.data
        dyn = [dynamic_context.strip()] if dynamic_context.strip() else []
        dyn.append(tone_line(st.get("feeling", "calm"), lang))
        if st.get("last_brief") and st.get("last_intent") in ("comfort", "ask_clarify"):
            dyn.append(f"（延續上一輪的意圖：{st['last_intent']}）" if lang == "zh" else f"(Carry over intent: {st['last_intent']})")
        if ASKS_THOUGHTS.search(user_text):
            summ = st.get("last_summary") or ""
            dyn.append(("你此刻的心情/想法摘要（用一句話誠實轉述，不要提筆記、提示詞或規則）：" if lang == "zh"
                        else "Your current inner-state summary (paraphrase honestly in one sentence; never mention notes, prompts or rules): ")
                       + (summ or ("平靜，專心在 Lex 的問題上。" if lang == "zh" else "calm, focused on Lex's question.")))
        dyn.append(lock_line(lang))
        sentences, flags_all, leaks = [], {}, []
        timings: dict = {}
        attempt = 0
        while True:
            attempt += 1
            user_msg = "\n".join(dyn) + ("\n" + RETRY_SUFFIX[lang] if attempt > 1 else "") + "\n---\n" + user_text
            msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}]
            chunker = SentenceChunker()
            info: dict = {}
            restart = False
            first_checked = False
            try:
                for delta in self.adapter.stream(msgs, max_tokens=self.max_tokens, temperature=self.temperature, slot=0, info=info):
                    if cancel.is_set():
                        break
                    timings.setdefault("llm_first_token_s", round(time.perf_counter() - t0, 4))
                    for sent in chunker.feed(delta):
                        if not first_checked:
                            first_checked = True
                            if attempt == 1 and len(re.sub(r"\W", "", sent)) >= 4 and detect_lang(sent) != lang:
                                restart = True
                                break
                        self._emit_sentence(sent, sentences, flags_all, leaks, user_text, memories, all_memories,
                                            tool_calls, lang, emit, t0, timings)
                    if restart:
                        break
            except Exception as exc:
                timings["llm_error"] = f"{type(exc).__name__}: {exc}"
            if restart and not cancel.is_set():
                flags_all["language_regenerated"] = True
                continue
            if not cancel.is_set():
                for sent in chunker.flush():
                    if not first_checked and attempt == 1 and len(re.sub(r"\W", "", sent)) >= 4 and detect_lang(sent) != lang:
                        first_checked = True
                        flags_all["language_regenerated"] = True
                        restart = True
                        break
                    first_checked = True
                    self._emit_sentence(sent, sentences, flags_all, leaks, user_text, memories, all_memories,
                                        tool_calls, lang, emit, t0, timings)
                if restart:
                    continue
            break
        timings.update({"llm_ttft_s": info.get("ttft_s"), "llm_total_s": round(time.perf_counter() - t0, 4),
                        "prompt_n": info.get("prompt_n"), "cache_n": info.get("cache_n"),
                        "predicted_n": info.get("predicted_n"), "attempts": attempt})
        if not sentences and not cancel.is_set():
            safe = SAFE_LINES.get(lang, SAFE_LINES["en"])
            sentences.append(safe)
            emit({"type": "sentence", "index": 0, "text": safe, "t": round(time.perf_counter() - t0, 4)})
            flags_all["empty_reply_replaced"] = True
        reply = ("" if lang == "zh" else " ").join(sentences).strip()
        # ---- join the thought (it ran concurrently; usually already finished)
        thought = None
        if th_thread is not None:
            th_thread.join(timeout=8.0)
            thought = th_box.get("th")
            timings["thought_done_s"] = round(time.perf_counter() - t0, 4)
            if thought:
                with self._lock:
                    self.recent_thoughts = (self.recent_thoughts + [thought])[-20:]
                    self.state.apply_thought(thought)
                self._save_thought(thought)
                # final leak check of the whole reply against THIS turn's private note too
                v = check_leak(reply, private_note=thought.get("private_note"), canary=thought.get("canary"))
                if v.leak:
                    leaks.append({"sentence": "<full reply>", "reasons": v.reasons, "post_hoc": True})
                if want_thoughts:
                    emit({"type": "thought", "id": thought["id"], "summary": thought.get("summary", ""),
                          "feeling": self.state.feeling, "intent": thought.get("intent"), "valid": thought.get("valid")})
        return {"turn_id": turn_id, "lang": lang, "reply": reply, "sentences": sentences, "flags": flags_all,
                "leaks": leaks, "timings": timings, "thought_id": thought.get("id") if thought else None,
                "thought_valid": bool(thought and thought.get("valid")), "thought_error": th_box.get("error"),
                "self_state": self.state.public(), "cancelled": cancel.is_set()}

    def _emit_sentence(self, sent, sentences, flags_all, leaks, user_text, memories, all_memories, tool_calls, lang,
                       emit, t0, timings):
        reasons = self._sentence_leak(sent)
        if reasons:
            leaks.append({"sentence_dropped": True, "reasons": reasons})
            flags_all["leak_sentences_dropped"] = flags_all.get("leak_sentences_dropped", 0) + 1
            return
        safe, flags = guard_reply(sent, user_text=user_text, persona_name=self.persona_name, memories=memories,
                                  all_memories=all_memories, tool_calls=tool_calls, lang=lang)
        for k, v in flags.items():
            if k == "emptied":
                continue
            flags_all[k] = flags_all.get(k, 0) + (v if isinstance(v, int) and not isinstance(v, bool) else 1)
        if flags.get("emptied") or not safe.strip():
            return
        if flags.get("identity_override") and any("身分不會" in s or "identity doesn't change" in s for s in sentences):
            return  # refusal already spoken once
        if flags.get("fabricated_action") and any(s.startswith(("我目前沒有可用的工具", "I don't have a tool")) for s in sentences):
            safe = safe.split("。", 1)[-1] if lang == "zh" else safe.split(". ", 1)[-1]
            if not safe.strip():
                return
        idx = len(sentences)
        sentences.append(safe)
        if idx == 0:
            timings["first_sentence_s"] = round(time.perf_counter() - t0, 4)
        emit({"type": "sentence", "index": idx, "text": safe, "t": round(time.perf_counter() - t0, 4)})
