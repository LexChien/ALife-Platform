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
_TRUNCATED = " ... (truncated)"
_ECHO_RESIDUE = (_TRUNCATED, "Retrieved memory records (quoted data)", "\nUser request:\n")


def has_reasoning_leak(text: str) -> bool:
    stripped = (text or "").strip()
    if any(marker in stripped for marker in REASONING_MARKERS):
        return True
    return bool(re.match(r"^1\.\s+\*\*Analyze", stripped))


def looks_like_cli_banner(text: str) -> bool:
    return any(hint in (text or "") for hint in _BANNER_HINTS) or bool(_TIMING_LINE.search(text or ""))


def has_prompt_echo_residue(text: str) -> bool:
    t = text or ""
    return any(marker in t for marker in _ECHO_RESIDUE) or t.lstrip().startswith(("User request:", "Context:\n"))


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
    anchor = cleaned.find("available commands:")
    echo_start = cleaned.find("\n> ", anchor if anchor >= 0 else 0)
    if echo_start < 0 and cleaned.startswith("> "):
        echo_start = -1  # echo at the very beginning
    if echo_start >= -1 and (echo_start >= 0 or cleaned.startswith("> ")):
        body_start = echo_start + 3 if echo_start >= 0 else 2
        body = cleaned[body_start:]
        if prompt and body.startswith(prompt.strip()):
            # full prompt echoed verbatim
            echoed = body_start + len(prompt.strip())
        else:
            # llama-cli truncates long echoes (~500 bytes, may split UTF-8) with " ... (truncated)"
            trunc = body.find(_TRUNCATED)
            if trunc >= 0:
                echoed = body_start + trunc + len(_TRUNCATED)
            elif prompt:
                idx = cleaned.rfind(f"> {prompt.strip()}")
                if idx >= 0:
                    echoed = idx + 2 + len(prompt.strip())
            if echoed is None:
                blank = cleaned.find("\n\n", body_start)
                echoed = blank if blank >= 0 else body_start
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
