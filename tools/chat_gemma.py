#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logger import iso_now, make_run_dir, save_json
from genai.llm.adapter import LLMRequest
from genai.llm.factory import create_llm_adapter


def _contains_reasoning(text: str) -> bool:
    markers = (
        "<|channel>thought",
        "Thinking Process:",
        "Here's a thinking process",
        "Analyze the Request:",
        "Deconstruct Key Terms:",
        "Brainstorm Core Concepts",
        "Original user request:",
        "I received the request",
        "I need to provide",
        "I will directly output",
        "Rewrite the following draft as the final answer only.",
        "我收到的請求是",
        "原始用戶請求是",
        "我需要提供",
        "我將直接輸出",
        "最終答案應該是",
        "Context:",
        "上下文:",
        "使用者:",
        "Conversation history:",
        "Digital Clone / ASAL memory:",
        "人工生命原型狀態上下文",
        "The previous interaction",
        "The core interaction",
        "The current interaction",
    )
    stripped = text.strip()
    pattern_hits = (
        re.match(r"^(我收到的.*請求是|I received .*request)", stripped),
        re.search(r"(<channel\|>|\[end of text\])", stripped),
        re.search(r"(我必須只輸出|根據上下文和指令|I must output only|Based on the context and instructions)", stripped),
        ("Context:" in stripped and "Conversation history:" in stripped),
        ("上下文:" in stripped and ("使用者:" in stripped or "Digital Clone / ASAL memory:" in stripped)),
        ("User asks" in stripped and "The assistant" in stripped),
    )
    return (
        any(marker in stripped for marker in markers)
        or bool(re.match(r"^1\.\s+\*\*Analyze", stripped))
        or any(bool(hit) for hit in pattern_hits)
    )


def _normalize_channel_noise(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\_", "_")
    if "<channel|>" in cleaned:
        head, tail = cleaned.rsplit("<channel|>", 1)
        cleaned = tail.strip() if len(tail.strip()) >= 8 else cleaned.replace("<channel|>", " ")
    if "[end of text]" in cleaned:
        cleaned = cleaned.split("[end of text]", 1)[0].strip()
    return cleaned


def _looks_invalid_final_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if re.fullmatch(r"[\W\d_]+", stripped):
        return True
    return False


def _strict_local_cleanup(text: str) -> str:
    cleaned = _normalize_channel_noise(text)
    for marker in ("Final Answer:", "Final Response:", "Answer:"):
        if marker in cleaned:
            tail = cleaned.split(marker, 1)[-1].strip()
            if tail:
                cleaned = tail
                break
    for marker in ("最終答案應該是：", "最終答案應該是:", "最終答案：", "最終答案:"):
        if marker in cleaned:
            tail = cleaned.split(marker, 1)[-1].strip()
            if tail:
                cleaned = tail
                break

    lines = [line.rstrip() for line in cleaned.splitlines()]
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "<channel|>" in stripped:
            stripped = stripped.rsplit("<channel|>", 1)[-1].strip()
        if "[end of text]" in stripped:
            stripped = stripped.split("[end of text]", 1)[0].strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if stripped in {"Thinking Process:", "Final Answer:", "Final Response:"}:
            continue
        if any(
            marker in stripped
            for marker in (
                "<|channel>thought",
                "Here's a thinking process",
                "Analyze the Request:",
                "Deconstruct Key Terms:",
                "Brainstorm Core Concepts",
                "Original user request:",
                "I received the request",
                "I need to provide",
                "I will directly output",
                "Rewrite the following draft as the final answer only.",
                "我收到的請求是",
                "原始用戶請求是",
                "我需要提供",
                "我將直接輸出",
                "最終答案應該是",
                "我收到的上一個請求是",
                "我必須只輸出",
                "根據上下文和指令",
                "上下文:",
                "使用者:",
                "Local ALife prototype baseline",
                "Digital Clone / ASAL memory:",
                "人工生命原型狀態上下文",
                "這些是本機專案事實",
                "ASAL 進度摘要：",
                "最新 ASAL run：",
                "TASKS.md：",
                "已接上的能力：",
                "明確限制/待辦：",
                "對話策略：",
                "[end of text]",
                "<channel|>",
                "The previous interaction",
                "The core interaction",
                "The current interaction",
                "The assistant initially struggled",
                "Leading to a request",
                "The frustration stems",
            )
        ):
            continue
        if re.match(r"^\d+\.\s+\*\*.*\*\*:?", stripped):
            continue
        if re.match(r"^\d+\.\s+", stripped):
            continue
        if stripped.startswith((
            "User request:",
            "Original user request:",
            "原始用戶請求是：",
            "原始用戶請求是:",
            "Context:",
            "上下文:",
            "使用者:",
            "Local ALife prototype baseline",
            "Digital Clone / ASAL memory:",
            "人工生命原型狀態上下文",
            "這些是本機專案事實",
            "- ASAL 進度摘要：",
            "- 最新 ASAL run：",
            "- TASKS.md：",
            "- 已接上的能力：",
            "- 明確限制/待辦：",
            "- 對話策略：",
            "Conversation history:",
            "Conversation history",
            "Draft:",
            "我收到的上一個請求是",
            "我必須只輸出",
            "根據上下文和指令",
        )):
            continue
        if stripped.startswith(("* ", "- ", "*   ", "• ")):
            if any(
                m in stripped
                for m in (
                    "Context:",
                    "上下文:",
                    "Digital Clone / ASAL memory:",
                    "人工生命原型狀態上下文",
                    "ASAL 進度摘要：",
                    "Conversation",
                    "Interaction",
                    "The assistant",
                    "The core",
                    "The previous",
                    "The current",
                )
            ):
                continue
            if ":" in stripped and len(stripped) < 120:
                continue
        kept.append(stripped)

    candidate = "\n".join(kept).strip()
    return candidate or cleaned


def _build_direct_answer_retry_request(request: LLMRequest) -> LLMRequest:
    return LLMRequest(
        prompt=request.prompt,
        context=request.context,
        system=(
            "You are a direct-answer assistant. "
            "Return only the final answer for the user. "
            "Do not include thinking process, analysis, numbered steps, or meta commentary. "
            "Never output the literal strings 'Thinking Process:' or '<|channel>thought'. "
            "If you include any analysis, the response is invalid. "
            "Use Traditional Chinese when the user writes in Chinese."
        ),
        max_tokens=min(request.max_tokens or 128, 96),
        temperature=0.0,
        stop=request.stop,
        json_mode=False,
        metadata={**request.metadata, "disable_reasoning_extractor": True},
    )


def _build_direct_answer_rescue_request(request: LLMRequest) -> LLMRequest:
    return LLMRequest(
        prompt=(
            "Return one direct final answer to the user request below. "
            "Do not show analysis, plan, numbered steps, or chain-of-thought. "
            "Output only the answer text.\n\n"
            f"User request:\n{request.prompt}"
        ),
        context=request.context,
        system=(
            "You are in answer-rescue mode. "
            "Your entire output must be a clean final answer only. "
            "No analysis. No thinking process. No bullet plan. No meta text. "
            "Never output 'Thinking Process:' or '<|channel>thought'. "
            "If the user writes in Chinese, answer in Traditional Chinese."
        ),
        max_tokens=min(request.max_tokens or 128, 80),
        temperature=0.0,
        stop=["Thinking Process:", "<|channel>thought"],
        json_mode=False,
        metadata={**request.metadata, "disable_reasoning_extractor": True},
    )


def _fallback_user_safe_answer(request: LLMRequest) -> str:
    prompt = (request.prompt or "").strip()
    chinese_hint = bool(re.search(r"[\u4e00-\u9fff]", prompt))
    if chinese_hint:
        return "抱歉，我剛才沒有成功整理出乾淨回覆，請再問一次，我會直接回答。"
    return "Sorry, I could not produce a clean direct answer just now. Please ask again and I will answer directly."


def _strict_cleanup_with_retry(adapter, request: LLMRequest, text: str) -> tuple[str, dict]:
    meta = {
        "reasoning_detected_in_tool": _contains_reasoning(text),
        "strict_cleanup_retry": False,
        "strict_cleanup_retry_error": None,
        "strict_cleanup_local_applied": False,
        "strict_cleanup_retry_strategy": None,
        "strict_cleanup_rescue": False,
        "strict_cleanup_rescue_error": None,
        "strict_cleanup_final_fallback": False,
    }
    final_text = text
    if not meta["reasoning_detected_in_tool"]:
        return final_text, meta

    local_candidate = _strict_local_cleanup(final_text)
    if (
        local_candidate
        and local_candidate != final_text
        and not _contains_reasoning(local_candidate)
        and not _looks_invalid_final_answer(local_candidate)
    ):
        final_text = local_candidate
        meta["strict_cleanup_local_applied"] = True
        return final_text, meta

    meta["strict_cleanup_retry"] = True
    meta["strict_cleanup_retry_strategy"] = "direct_answer_retry"
    retry_request = _build_direct_answer_retry_request(request)
    try:
        retry_response = adapter.generate(retry_request)
        retry_text = retry_response.text.strip()
        retry_candidate = _strict_local_cleanup(retry_text)
        if retry_candidate and not _looks_invalid_final_answer(retry_candidate):
            final_text = retry_candidate
            if retry_candidate != retry_text:
                meta["strict_cleanup_local_applied"] = True
    except Exception as exc:
        meta["strict_cleanup_retry_error"] = f"{type(exc).__name__}: {exc}"

    if not _contains_reasoning(final_text):
        return final_text, meta

    meta["strict_cleanup_rescue"] = True
    meta["strict_cleanup_retry_strategy"] = "direct_answer_rescue"
    rescue_request = _build_direct_answer_rescue_request(request)
    try:
        rescue_response = adapter.generate(rescue_request)
        rescue_text = rescue_response.text.strip()
        rescue_candidate = _strict_local_cleanup(rescue_text)
        if rescue_candidate and not _looks_invalid_final_answer(rescue_candidate):
            final_text = rescue_candidate
            if rescue_candidate != rescue_text:
                meta["strict_cleanup_local_applied"] = True
    except Exception as exc:
        meta["strict_cleanup_rescue_error"] = f"{type(exc).__name__}: {exc}"

    if _contains_reasoning(final_text):
        final_text = _fallback_user_safe_answer(request)
        meta["strict_cleanup_final_fallback"] = True

    return final_text, meta


def _build_turn_context(base_context: str | None, transcript: list[tuple[str, str]], history_turns: int) -> str | None:
    parts: list[str] = []
    if base_context:
        parts.append(base_context.strip())
    if transcript:
        selected = transcript[-history_turns:] if history_turns > 0 else transcript
        history_lines = []
        for role, text in selected:
            label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{label}: {text}")
        parts.append("Conversation history:\n" + "\n".join(history_lines))
    combined = "\n\n".join(part for part in parts if part.strip())
    return combined or None


def _run_single_turn(adapter, request: LLMRequest, save_run: bool) -> str:
    health = adapter.healthcheck()
    response = adapter.generate(request)
    cleaned_text, cleanup_meta = _strict_cleanup_with_retry(adapter, request, response.text)
    print(cleaned_text)

    if save_run:
        run_dir = make_run_dir("runs/chat_gemma")
        payload = {
            "timestamp": iso_now(),
            "prompt": request.prompt,
            "context": request.context,
            "system": request.system,
            "text": cleaned_text,
            "raw_text": response.text,
            "chat_cleanup": cleanup_meta,
            "llm": {
                "backend": response.backend,
                "model_family": response.model_family,
                "runtime": response.runtime,
                "healthcheck": health,
            },
        }
        save_json(Path(run_dir) / "chat_output.json", payload)
        print(f"\nSaved: {run_dir}")
    return cleaned_text


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Single-turn or interactive chat entrypoint for local Gemma / llama.cpp without image or audio generation."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--prompt")
    ap.add_argument("--context")
    ap.add_argument("--system")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--save-run", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--history-turns", type=int, default=4)
    args = ap.parse_args()

    if not args.interactive and not args.prompt:
        ap.error("--prompt is required unless --interactive is used")

    cfg = load_config(args.config, profile=args.profile)
    adapter = create_llm_adapter(cfg)
    base_context = args.context if args.context is not None else cfg.get("context")
    system = args.system if args.system is not None else cfg.get("system")
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.get("llm", {}).get("max_tokens")
    temperature = args.temperature if args.temperature is not None else cfg.get("llm", {}).get("temperature")

    if args.interactive:
        transcript: list[tuple[str, str]] = []
        print("Interactive chat started. Type /exit to quit.")
        while True:
            try:
                user_prompt = input("You> ").strip()
            except EOFError:
                print()
                break
            if not user_prompt:
                continue
            if user_prompt in {"/exit", "/quit"}:
                break

            selected_transcript = transcript[-args.history_turns:] if args.history_turns > 0 else transcript

            request = LLMRequest(
                prompt=user_prompt,
                context=base_context,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={"transcript": selected_transcript},
            )
            print("Gemma> ", end="")
            cleaned_text = _run_single_turn(adapter, request, save_run=args.save_run)
            transcript.append(("user", user_prompt))
            transcript.append(("assistant", cleaned_text))
        return 0

    request = LLMRequest(
        prompt=args.prompt,
        context=base_context,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _run_single_turn(adapter, request, save_run=args.save_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
