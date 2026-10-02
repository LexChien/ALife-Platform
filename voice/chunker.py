"""Plan 38 J1.5: streaming sentence chunker for LLM deltas -> TTS.

Feeds token deltas; emits a sentence as soon as it ends (。！？!?；; newline, or '.'/'!'/'?' + space in English).
The FIRST chunk may also be cut at a comma once it is long enough, so the first audio starts early; tiny fragments
(e.g. a lone "嗯") are merged with the next sentence unless they are the very first acknowledgement."""
from __future__ import annotations

import re

_END_ZH = "。！？；\n"
_END_EN = re.compile(r"[.!?;](?=\s)|[.!?](?=$)")
_SOFT = "，,、："


class SentenceChunker:
    def __init__(self, min_chars: int = 4, first_soft_max: int = 22, soft_max: int = 60):
        self.buf = ""
        self.min_chars = min_chars
        self.first_soft_max = first_soft_max
        self.soft_max = soft_max
        self.emitted = 0

    def _hard_end(self) -> int | None:
        b = self.buf
        for i, ch in enumerate(b):
            if ch in _END_ZH:
                return i + 1
            if ch in ".!?;" and (i + 1 < len(b) and b[i + 1].isspace()):
                if ch == "." and re.search(r"\b(?:Mr|Mrs|Dr|St|vs|e\.g|i\.e)$", b[:i]):
                    continue
                return i + 1
        return None

    def _cut_index(self) -> int | None:
        b = self.buf
        hard = self._hard_end()
        limit = self.first_soft_max if self.emitted == 0 else self.soft_max
        if hard is not None and hard <= limit:
            return hard
        if len(b) >= limit:
            cut = max(b.rfind(c, 0, limit) for c in _SOFT)
            if cut >= self.min_chars:
                return cut + 1
        return hard

    def feed(self, delta: str) -> list[str]:
        self.buf += delta or ""
        out = []
        while True:
            idx = self._cut_index()
            if idx is None:
                break
            piece, rest = self.buf[:idx].strip(), self.buf[idx:]
            if len(re.sub(r"\W", "", piece)) < self.min_chars and rest.strip() and self.emitted > 0:
                # merge tiny fragment with the following text
                nxt = re.search(r"[。！？；\n.!?]", rest)
                if not nxt:
                    break
                piece = (piece + rest[:nxt.end()]).strip()
                rest = rest[nxt.end():]
            self.buf = rest
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece = self.buf.strip()
        self.buf = ""
        if piece:
            self.emitted += 1
            return [piece]
        return []


def chunk_text(text: str, **kw) -> list[str]:
    c = SentenceChunker(**kw)
    return c.feed(text) + c.flush()
