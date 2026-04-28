from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid

from core.logger import iso_now


SESSION_SCHEMA_VERSION = "1.0"


@dataclass
class ChatMessage:
    role: str
    text: str
    created_at: str = field(default_factory=iso_now)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class ChatSession:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    reset_count: int = 0

    def append(self, role: str, text: str) -> None:
        self.messages.append(ChatMessage(role=role, text=text))
        self.updated_at = iso_now()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def turn_count(self) -> int:
        assistant_count = sum(1 for item in self.messages if item.role == "assistant")
        user_count = sum(1 for item in self.messages if item.role == "user")
        return min(user_count, assistant_count)

    def transcript_pairs(self) -> list[tuple[str, str]]:
        return [(item.role, item.text) for item in self.messages]

    def to_dict(self) -> dict:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reset_count": self.reset_count,
            "message_count": self.message_count,
            "turn_count": self.turn_count,
            "messages": [item.to_dict() for item in self.messages],
        }


class ChatSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> ChatSession:
        normalized = self._normalize_session_id(session_id)
        with self._lock:
            session = self._sessions.get(normalized)
            if session is None:
                session = ChatSession(session_id=normalized)
                self._sessions[normalized] = session
            return session

    def reset(self, session_id: str | None) -> ChatSession:
        normalized = self._normalize_session_id(session_id)
        with self._lock:
            previous = self._sessions.get(normalized)
            reset_count = previous.reset_count + 1 if previous else 1
            session = ChatSession(session_id=normalized, reset_count=reset_count)
            self._sessions[normalized] = session
            return session

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str:
        return (session_id or "").strip() or str(uuid.uuid4())
