"""Shared output hygiene for local LLM runtimes.

Removes reasoning channels (Gemma ``<|channel>thought``, ``<think>``), llama.cpp
CLI banners / prompt echo / timing lines, and stray control tokens so product
callers (gemma_web, DigitalClone) never surface them.
"""
from __future__ import annotations

import re

REASONING_MARKERS: tuple[str, ...] = (
    "<|channel>thought",
    "<think>",
    "Thinking Process:",
    "Here's a thinking process",
    "Analyze the Request:",
    "Deconstruct Key Terms:",
    "Brainstorm Core Concepts",
)

_CLOSED_BLOCKS = (
    re.compile(r"<\|channel\>thought.*?<channel\|>", re.S),
    re.compile(r"<\|channel\>analysis.*?<channel\|>", re.S),
    re.compile(r"<think>.*?</think>", re.S),
)
_CONTROL_TOKENS = re.compile(
    r"<\|channel\>(?:final|response|answer)\n?|<channel\|>|<\|turn\>model\n?|<turn\|>|<end_of_turn>|<eos>|<\|im_end\|>"
)
_TIMING_LINE = re.compile(r"^\s*\[\s*Prompt:\s*[\d.]+\s*t/s.*\]\s*$", re.M)
_BANNER_HINTS = ("Loading model...", "available commands:", "/exit or Ctrl+C")


def has_reasoning_leak(text: str) -> bool:
    stripped = (text or "").strip()
    if any(marker in stripped for marker in REASONING_MARKERS):
        return True
    return bool(re.match(r"^1\.\s+\*\*Analyze", stripped))


def looks_like_cli_banner(text: str) -> bool:
    return any(hint in (text or "") for hint in _BANNER_HINTS) or bool(_TIMING_LINE.search(text or ""))


def strip_cli_banner(text: str, prompt: str | None = None) -> str:
    """Return only the model answer from llama-cli conversation-mode stdout."""
    cleaned = (text or "").replace("\r\n", "\n")
    if not looks_like_cli_banner(cleaned):
        return cleaned.strip()
    # Cut the timing line and everything after it ("Exiting...").
    match = _TIMING_LINE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]
    cleaned = re.sub(r"\n\s*Exiting\.\.\.\s*$", "", cleaned)
    echoed = None
    if prompt:
        candidate = f"> {prompt.strip()}"
        idx = cleaned.rfind(candidate)
        if idx >= 0:
            echoed = idx + len(candidate)
    if echoed is None and "available commands:" in cleaned:
        # Fallback: answer starts after the first blank line following the "> " echo.
        idx = cleaned.find("\n> ", cleaned.find("available commands:"))
        if idx >= 0:
            blank = cleaned.find("\n\n", idx + 3)
            echoed = blank if blank >= 0 else idx + 3
    if echoed is not None:
        cleaned = cleaned[echoed:]
    return cleaned.strip()


def strip_reasoning(text: str) -> str:
    """Remove closed reasoning blocks and control tokens; keep the final answer."""
    cleaned = text or ""
    for pattern in _CLOSED_BLOCKS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _CONTROL_TOKENS.sub("", cleaned)
    cleaned = re.sub(r"(?:\s*\[end of text\])+\s*$", "", cleaned)
    return cleaned.strip()


def sanitize_reply(text: str, prompt: str | None = None) -> str:
    """Full product-path hygiene: banner, reasoning blocks, control tokens."""
    return strip_reasoning(strip_cli_banner(text, prompt=prompt))
