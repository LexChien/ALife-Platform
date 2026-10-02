"""Plan 38 J2.5: turn-taking state machine with barge-in.

States: idle -> listening (speech_start) -> thinking (speech_end/endpoint) -> speaking (first audio) -> idle.
Barge-in: speech detected while ``speaking`` for >= barge_min_ms -> ``stop`` action (client stops audio, server
cancels generation + TTS). Push-to-talk always works: ptt_down forces listening, ptt_up forces the endpoint."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TurnTaking:
    barge_min_ms: int = 240
    state: str = "idle"
    speaking_since: float | None = None
    barge_cand: float | None = None
    log: list = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    ptt: bool = False

    def _set(self, st: str, why: str):
        if st != self.state:
            self.log.append({"t": round(time.time(), 3), "from": self.state, "to": st, "why": why})
            self.log = self.log[-200:]
            self.state = st

    def on_speech_start(self, now: float | None = None) -> str | None:
        now = now or time.time()
        if self.state == "speaking":
            self.barge_cand = now
            return None
        if self.state in ("idle", "listening"):
            self._set("listening", "vad_start")
        return None

    def on_speech_continue(self, now: float | None = None) -> str | None:
        """Call while VAD says speech; returns 'barge_in' once the barge-in criterion is met."""
        now = now or time.time()
        if self.state == "speaking" and self.barge_cand is not None and (now - self.barge_cand) * 1000 >= self.barge_min_ms:
            self.barge_cand = None
            self.cancel.set()
            self._set("listening", "barge_in")
            return "barge_in"
        return None

    def on_speech_end(self) -> str | None:
        self.barge_cand = None
        if self.state == "listening" and not self.ptt:
            self._set("thinking", "endpoint")
            return "endpoint"
        return None

    def on_ptt(self, down: bool) -> str | None:
        self.ptt = down
        if down:
            if self.state == "speaking":
                self.cancel.set()
                self._set("listening", "ptt_barge_in")
                return "barge_in"
            self._set("listening", "ptt_down")
            return None
        if self.state == "listening":
            self._set("thinking", "ptt_up")
            return "endpoint"
        return None

    def on_first_audio(self):
        self.cancel.clear()
        self.speaking_since = time.time()
        self._set("speaking", "first_audio")

    def on_turn_done(self):
        self._set("idle", "done")
        self.speaking_since = None

    def new_turn(self) -> threading.Event:
        self.cancel = threading.Event()
        return self.cancel
