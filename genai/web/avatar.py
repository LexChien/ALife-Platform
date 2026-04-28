from __future__ import annotations

from copy import deepcopy


VALID_AVATAR_STATES = ("idle", "listening", "thinking", "speaking", "error")

DEFAULT_VOICE_CONFIG = {
    "input_provider": "macos_speech_via_upload",
    "output_provider": "browser_speech_synthesis",
    "recognition_lang": "zh-TW",
    "synthesis_lang": "zh-TW",
    "auto_submit": True,
    "auto_speak": True,
}

DEFAULT_AVATAR_CONFIG = {
    "profile": "orb",
    "display_name": "Gemma 4",
    "enable_visual_state": True,
    "states": {state: state for state in VALID_AVATAR_STATES},
}


def merge_web_runtime_config(cfg: dict | None, defaults: dict) -> dict:
    merged = deepcopy(defaults)
    if not isinstance(cfg, dict):
        return merged
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def resolve_avatar_state(
    *,
    conversation_state: str,
    mic_state: str,
    speech_state: str,
    has_error: bool = False,
) -> str:
    if has_error:
        return "error"
    if mic_state == "mic_listening":
        return "listening"
    if conversation_state in {"submitting", "awaiting_response"}:
        return "thinking"
    if speech_state == "speech_playing":
        return "speaking"
    return "idle"
