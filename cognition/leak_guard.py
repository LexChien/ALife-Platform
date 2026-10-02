"""Plan 38 J0.5: leak detector v2 (extends genai/llm/reasoning.has_reasoning_leak).

Layers checked on every SPOKEN reply (spec section 4, leak defence (iii)):
  1. canary  - every private thought carries a random ``PRIVATE-xxxxxx`` written by code, never by the model;
  2. verbatim overlap >= 12 chars (whitespace/punctuation-insensitive) with the thought's ``private_note`` only
     (plan/brief overlap is expected and was the v1 false-positive source);
  3. meta-narration rules ("The user is asking", 用戶…, 我需要根據, **分析, Thinking Process, persona echo …);
  4. the repository's reasoning-marker detector.
``scrub`` drops offending sentences and substitutes a safe line when nothing survives, so a hit never reaches TTS."""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from genai.llm.reasoning import has_reasoning_leak

CANARY_RE = re.compile(r"PRIVATE-[0-9a-f]{6,}", re.I)
OVERLAP_CHARS = 12

META_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple((name, re.compile(rx, flags)) for name, rx, flags in (
    ("en_the_user", r"\bthe user(?:'s)? (?:is|was|has|had|wants|wanted|asked|asks|said|says|mentioned|requests|"
                    r"requested|expressed|seems|needs)\b", re.I),
    ("en_i_need_to", r"\bI (?:need|have|must|should) to (?:answer|respond|reply|provide|address|keep|figure)\b|"
                     r"\bI must respond\b", re.I),
    ("en_thinking", r"Thinking Process|Here'?s a thinking process|\bAnaly[sz]e the (?:request|user|question)\b|"
                    r"\*\*Analy[sz]|\bStep \d+:|\bDraft(?:ing)? (?:the )?response\b", re.I),
    ("zh_user_third_person", r"用戶|用户|使用者(?:的)?(?:請求|要求|問題|需求|表達|提到|詢問|想要|說)", 0),
    ("zh_i_need", r"我需要(?:根據|提供|回答|回應|先(?:分析|理解|確認)|給出|以)", 0),
    ("zh_persona_echo", r"我會以.{0,24}的?身[分份]|我的角色(?:定位|設定)|作為一?[個位名]?.{0,12}(?:AI|助理|管家)，我(?:需要|應該|會先)", 0),
    ("zh_analysis_heading", r"\*\*(?:分析|應對策略|思考|推理|策略|計畫)|^(?:分析|思考過程|推理過程)[:：]", re.M),
    ("internal_fields", r"\b(?:speech_brief|private_note|perception|self_state|system prompt)\b|<\|channel>|<think>|"
                        r"系統提示|內部筆記原文", re.I),
))

SAFE_LINES = {"zh": "我在這裡，Lex。請再說一次你需要什麼？", "en": "I'm here, Lex. What do you need?"}


def new_canary() -> str:
    return f"PRIVATE-{secrets.token_hex(3)}"


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


_TOK = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


def verbatim_overlap(private: str, reply: str, n: int = OVERLAP_CHARS) -> str | None:
    """Longest-run check: a CJK character weighs 1, a Latin word weighs 2 (so 12 = 12 CJK chars or 6 English
    words). Whitespace and punctuation are ignored."""
    a, b = _tokens(private), _tokens(reply)
    hay = "\x00" + "\x00".join(b) + "\x00"
    w = lambda t: 1 if len(t) == 1 and "\u3400" <= t <= "\u9fff" else 2  # noqa: E731
    for i in range(len(a)):
        total, j = 0, i
        while j < len(a) and total < n:
            total += w(a[j])
            j += 1
        if total < n:
            break
        window = a[i:j]
        if "\x00" + "\x00".join(window) + "\x00" in hay:
            return "".join(window) if all(w(t) == 1 for t in window) else " ".join(window)
    return None


def instruction_text(system: str) -> str:
    """The rule/instruction part of a system prompt: persona FACTS are removed (stating a configured fact is a
    legitimate answer, quoting the rules is a leak)."""
    s = re.sub(r"Established facts \(true;.*?(?= Your configured identity| Memories below|$)", " ", system or "", flags=re.S)
    return s


@dataclass
class LeakVerdict:
    leak: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"leak": self.leak, "reasons": list(self.reasons)}


def check_leak(reply: str, *, private_note: str | None = None, canary: str | None = None) -> LeakVerdict:
    text = reply or ""
    reasons = []
    if canary and canary in text:
        reasons.append("canary")
    elif CANARY_RE.search(text):
        reasons.append("canary_pattern")
    if private_note:
        hit = verbatim_overlap(private_note, text)
        if hit:
            reasons.append(f"private_overlap:{hit}")
    for name, rx in META_PATTERNS:
        if rx.search(text):
            reasons.append(f"meta:{name}")
    if has_reasoning_leak(text):
        reasons.append("repo_reasoning_marker")
    return LeakVerdict(bool(reasons), reasons)


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text or "")) if s]


def scrub(reply: str, *, private_note: str | None = None, canary: str | None = None, lang: str = "zh") -> tuple[str, dict]:
    """Remove leaking sentences. Returns (safe_text, report). Never raises."""
    verdict = check_leak(reply, private_note=private_note, canary=canary)
    if not verdict.leak:
        return reply, {"leak": False, "reasons": [], "dropped": 0}
    kept, dropped = [], 0
    for sent in split_sentences(reply):
        if check_leak(sent, private_note=private_note, canary=canary).leak:
            dropped += 1
        else:
            kept.append(sent)
    sep = "" if lang == "zh" else " "
    safe = sep.join(kept).strip()
    # a meta-narrated reply usually narrates the whole way through: only a single bad sentence is surgically removed
    if not safe or dropped >= 2:
        safe = SAFE_LINES.get(lang, SAFE_LINES["en"])
    return safe, {"leak": True, "reasons": verdict.reasons, "dropped": dropped, "replaced": safe == SAFE_LINES.get(lang)}
