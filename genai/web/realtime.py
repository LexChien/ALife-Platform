"""Plan 38 J1.3/J2: realtime voice session over WebSocket (ws://host:port+1/ws/session).

Client -> server
  * binary: 16 kHz mono Int16 PCM (any frame size; the AudioWorklet sends 20 ms)
  * {"type":"hello","session_id":..,"mode":"open|wake|ptt","thoughts":false}
  * {"type":"set","mode":..} / {"type":"set","thoughts":true}
  * {"type":"ptt","down":true|false}   push-to-talk (always available, any mode)
  * {"type":"text","text":".."}       typed turn through the same streaming path
  * {"type":"barge_in"}                client-side stop (e.g. Esc)
  * {"type":"playback","event":"started|ended|stopped","turn_id":..}
Server -> client events: state, vad, wake, transcript, ack, sentence, audio, emotion, thought, stop, done, trace, error.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from voice.trace import TurnTrace
from voice.turn_taking import TurnTaking

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "runs" / "digiclone" / "turn_traces.jsonl"
WAKE_WINDOW_S = 8.0


class VoiceSession:
    def __init__(self, service, send, *, vad=None, wake=None, stt=None, ack=None, barge_min_ms: int = 240,
                 barge_prob: float = 0.7, trace_path: Path | None = TRACE_PATH):
        self.service = service
        self.send = send
        self.session_id = None
        self.mode = "open"
        self.thoughts = bool(getattr(service, "cog_cfg", {}).get("thought_stream_default", False)) if service else False
        self.tt = TurnTaking(barge_min_ms=barge_min_ms)
        self.barge_prob = barge_prob
        self.vad = vad
        self.wake = wake
        self.stt = stt
        self.ack = ack
        self.awake_until = 0.0
        self.utt_start = None
        self.ptt_start = None
        self.turn_thread = None
        self.trace = None
        self.turn_id = None
        self.barge_t_onset = None
        self.lock = threading.Lock()
        self.stats = {"frames": 0, "turns": 0, "barge_ins": 0, "wakes": 0}
        self.trace_path = trace_path

    # ------------------------------------------------------------------ control
    def on_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.session_id = msg.get("session_id") or self.session_id
            self._set_opts(msg)
            self.send({"type": "state", "state": self.tt.state, "mode": self.mode, "thoughts": self.thoughts,
                       "wake": getattr(self.wake, "names", []), "vad": self.vad is not None, "stt": self.stt is not None})
        elif kind == "set":
            self._set_opts(msg)
            self.send({"type": "state", "state": self.tt.state, "mode": self.mode, "thoughts": self.thoughts})
        elif kind == "ptt":
            act = self.tt.on_ptt(bool(msg.get("down")))
            now = self._stream_t()
            if msg.get("down"):
                self.ptt_start = now
                if act == "barge_in":
                    self._barge("ptt")
                self.send({"type": "state", "state": self.tt.state, "why": "ptt_down"})
            elif act == "endpoint" and self.ptt_start is not None:
                self._endpoint(self.ptt_start, now, why="ptt_up")
                self.ptt_start = None
        elif kind == "text":
            text = str(msg.get("text") or "").strip()
            if text:
                self._start_turn(text=text, source="text")
        elif kind == "barge_in":
            if self.tt.state == "speaking" or self.turn_thread is not None:
                self.tt.cancel.set()
                self._barge("client")
        elif kind == "playback":
            ev = msg.get("event")
            if self.trace is not None and msg.get("turn_id") == self.turn_id:
                if ev == "started":
                    self.trace.mark("first_audio_played")
                elif ev == "stopped":
                    self.trace.mark("audio_stopped")
                    if self.barge_t_onset is not None:
                        self.trace.extra["barge_client_stop_ms"] = msg.get("latency_ms")
            if ev in ("ended", "stopped") and self.tt.state == "speaking" and self.turn_thread is None:
                self.tt.on_turn_done()
                self.send({"type": "state", "state": self.tt.state, "why": "playback_" + ev})

    def _set_opts(self, msg: dict) -> None:
        if msg.get("mode") in ("open", "wake", "ptt"):
            self.mode = msg["mode"]
        if "thoughts" in msg:
            self.thoughts = bool(msg["thoughts"])

    # ------------------------------------------------------------------ audio
    def _stream_t(self) -> float:
        return (self.vad.n / 16000.0) if self.vad is not None else time.time()

    def on_audio(self, data: bytes) -> None:
        pcm = np.frombuffer(data, dtype=np.int16)
        if not len(pcm):
            return
        self.stats["frames"] += 1
        if self.wake is not None and self.mode == "wake" and time.time() > self.awake_until:
            hit = self.wake.push(pcm)
            if hit:
                self.stats["wakes"] += 1
                self.awake_until = time.time() + WAKE_WINDOW_S
                self.send({"type": "wake", "keyword": hit, "window_s": WAKE_WINDOW_S})
                if self.ack is not None:
                    url = self.ack.pick("en" if hit == "hey_jarvis" else "zh")
                    if url:
                        self.send({"type": "ack", "url": f"/api/tts/{url}", "why": "wake"})
        if self.vad is None:
            return
        events = self.vad.push(pcm)
        prob = self.vad.last_prob
        for ev in events:
            if ev.kind == "speech_start":
                self.utt_start = ev.start_t
                self.tt.on_speech_start(time.time())
                if self.tt.state == "speaking":
                    self.barge_t_onset = time.time() - max(0.0, ev.t - ev.start_t - 0.2)
                self.send({"type": "vad", "event": "speech_start", "t": round(ev.t, 3)})
            elif ev.kind == "speech_end":
                self.send({"type": "vad", "event": "speech_end", "t": round(ev.t, 3)})
                act = self.tt.on_speech_end()
                if act == "endpoint" and self._listening_allowed() and self.utt_start is not None:
                    self._endpoint(self.utt_start, ev.t, why="vad")
                elif act == "endpoint":
                    self.tt.on_turn_done()
                self.utt_start = None
        if self.vad.ep.in_speech and self.tt.state == "speaking" and prob >= self.barge_prob:
            if self.tt.on_speech_continue(time.time()) == "barge_in":
                self._barge("vad")
        if self.stats["frames"] % 500 == 0:
            self.vad.trim(30.0)

    def _listening_allowed(self) -> bool:
        if self.mode == "open":
            return True
        if self.mode == "wake":
            return time.time() <= self.awake_until
        return False  # ptt mode: only the ptt_up endpoint starts a turn

    def _barge(self, why: str) -> None:
        self.stats["barge_ins"] += 1
        lat = None
        if self.barge_t_onset is not None:
            lat = round((time.time() - self.barge_t_onset) * 1000)
        if self.trace is not None:
            self.trace.mark("barge_in")
            self.trace.extra["barge_detect_ms"] = lat
        self.send({"type": "stop", "why": why, "turn_id": self.turn_id, "detect_ms": lat})
        self.barge_t_onset = None

    def _endpoint(self, start_t: float, end_t: float, why: str) -> None:
        seg = self.vad.segment(start_t, end_t) if self.vad is not None else np.zeros(0, dtype=np.float32)
        if len(seg) < 16000 * 0.25:
            self.tt.on_turn_done()
            self.send({"type": "state", "state": self.tt.state, "why": "too_short"})
            return
        if self.mode == "wake":
            self.awake_until = time.time() + WAKE_WINDOW_S  # follow-ups need no new wake word
        self._start_turn(audio=seg, source="voice", why=why)

    # ------------------------------------------------------------------ turn
    def _start_turn(self, *, text: str | None = None, audio=None, source: str, why: str = "") -> None:
        if self.turn_thread is not None and self.turn_thread.is_alive():
            self.tt.cancel.set()
            self.turn_thread.join(timeout=3)
        cancel = self.tt.new_turn()
        self.turn_id = f"t-{uuid.uuid4().hex[:8]}"
        self.trace = TurnTrace(self.turn_id, source=source)
        self.trace.mark("speech_end")
        self.tt._set("thinking", why or source)
        self.send({"type": "state", "state": "thinking", "turn_id": self.turn_id})
        th = threading.Thread(target=self._run_turn, args=(text, audio, source, cancel, self.trace, self.turn_id), daemon=True)
        self.turn_thread = th
        th.start()

    def _run_turn(self, text, audio, source, cancel, trace, turn_id) -> None:
        try:
            lang_hint = "zh"
            if audio is not None:
                if self.ack is not None and self.service is not None and self.service.voice_cfg.get("ack", True):
                    url = self.ack.pick("zh")
                    if url:
                        trace.mark("ack_audio")
                        self.send({"type": "ack", "url": f"/api/tts/{url}", "turn_id": turn_id})
                if self.stt is None:
                    self.send({"type": "error", "error": "stt unavailable"})
                    return
                r = self.stt.transcribe(audio)
                trace.mark("stt_done")
                trace.extra["stt"] = {k: r.get(k) for k in ("language", "elapsed_s", "audio_s")}
                text = (r.get("text") or "").strip()
                lang_hint = r.get("language") or lang_hint
                self.send({"type": "transcript", "text": text, "language": r.get("language"), "turn_id": turn_id})
                if not text:
                    return
            started = {"audio": False}

            def emit(ev):
                if cancel.is_set() and ev.get("type") in ("sentence", "audio"):
                    return
                if ev.get("type") == "audio" and not started["audio"]:
                    started["audio"] = True
                    self.tt.on_first_audio()
                    self.send({"type": "state", "state": "speaking", "turn_id": turn_id})
                if ev.get("type") == "thought" and not self.thoughts:
                    return
                self.send({**ev, "turn_id": turn_id})
            payload = self.service.chat_stream(self.session_id, text, want_tts=True, source=source, emit=emit,
                                               cancel=cancel, want_thoughts=True, trace=trace)
            self.session_id = payload.get("session_id") or self.session_id
            trace.mark("turn_done")
            trace.extra.update({"timings": payload.get("timings"), "cancelled": payload.get("cancelled"),
                                "lang_hint": lang_hint})
            self.stats["turns"] += 1
        except Exception as exc:
            logger.exception("turn failed")
            self.send({"type": "error", "error": f"{type(exc).__name__}: {exc}", "turn_id": turn_id})
        finally:
            try:
                if self.trace_path is not None:
                    trace.save(self.trace_path)
            except Exception:
                pass
            self.send({"type": "trace", **trace.to_dict()})
            if self.tt.state != "speaking":
                self.tt.on_turn_done()
                self.send({"type": "state", "state": self.tt.state, "turn_id": turn_id})
            if self.turn_thread is threading.current_thread():
                self.turn_thread = None


class RealtimeServer:
    """websockets server in a background thread; one VoiceSession per connection."""

    def __init__(self, service, host: str = "127.0.0.1", port: int = 8081):
        self.service = service
        self.host, self.port = host, port
        self.thread = None
        self.loop = None
        self.error = None
        self.connections = 0
        self.ack = None
        self._vad_ok = None

    def _make_session(self, send) -> VoiceSession:
        vad = wake = None
        try:
            from voice.vad import SileroVAD, StreamingVAD
            if SileroVAD.available():
                vad = StreamingVAD(min_silence_ms=int(self.service.voice_cfg.get("endpoint_silence_ms", 400)))
        except Exception as exc:
            logger.warning("vad unavailable: %s", exc)
        try:
            from voice.wake import WakeDetector
            wake = WakeDetector()
        except Exception as exc:
            logger.warning("wake unavailable: %s", exc)
        return VoiceSession(self.service, send, vad=vad, wake=wake, stt=getattr(self.service, "stream_stt", None),
                            ack=self.ack)

    def warm(self) -> None:
        tts = getattr(self.service, "tts", None)
        if tts is not None:
            try:
                from voice.tts_stream import AckCache
                ack = AckCache(tts, self.service.tts_dir)
                ack.warm()
                self.ack = ack
            except Exception as exc:
                logger.warning("ack warm failed: %s", exc)

    async def _handler(self, ws):
        path = getattr(getattr(ws, "request", None), "path", "/ws/session")
        if not str(path).startswith("/ws/session"):
            await ws.close(1008, "unknown path")
            return
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()

        def send(ev: dict):
            loop.call_soon_threadsafe(out.put_nowait, json.dumps(ev, ensure_ascii=False, default=str))

        async def pump():
            while True:
                msg = await out.get()
                await ws.send(msg)
        self.connections += 1
        sess = await loop.run_in_executor(None, self._make_session, send)
        pump_task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    sess.on_audio(bytes(msg))
                else:
                    try:
                        sess.on_control(json.loads(msg))
                    except json.JSONDecodeError:
                        send({"type": "error", "error": "bad json"})
        except Exception as exc:
            logger.info("ws closed: %s", exc)
        finally:
            sess.tt.cancel.set()
            pump_task.cancel()
            self.connections -= 1

    def start(self) -> bool:
        try:
            import websockets  # noqa: F401
        except Exception as exc:
            self.error = f"websockets missing: {exc}"
            return False
        ready = threading.Event()

        def run():
            import websockets
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def main():
                try:
                    async with websockets.serve(self._handler, self.host, self.port, max_size=2 ** 22):
                        ready.set()
                        await asyncio.Future()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    ready.set()
            self.loop.run_until_complete(main())
        self.thread = threading.Thread(target=run, daemon=True, name="realtime-ws")
        self.thread.start()
        ready.wait(10)
        return self.error is None

    def health(self) -> dict:
        return {"port": self.port, "running": self.thread is not None and self.thread.is_alive() and self.error is None,
                "error": self.error, "connections": self.connections,
                "ack": {k: len(v) for k, v in (self.ack.files.items() if self.ack else [])}}
