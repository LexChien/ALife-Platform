import os
from copy import deepcopy
from pathlib import Path

import yaml


_RESERVED_KEYS = {"defaults", "profiles", "active_profile"}


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str, profile: str | None = None) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping.")

    selected_profile = (
        profile
        or os.environ.get("ALIFE_PROFILE")
        or raw.get("active_profile")
    )
    defaults = raw.get("defaults", {})
    profiles = raw.get("profiles", {})
    body = {key: value for key, value in raw.items() if key not in _RESERVED_KEYS}

    merged = _deep_merge(defaults if isinstance(defaults, dict) else {}, body)
    if selected_profile:
        if selected_profile not in profiles:
            raise ValueError(
                f"Unknown config profile '{selected_profile}' for {path}. "
                f"Available: {list(profiles.keys())}"
            )
        profile_payload = profiles[selected_profile]
        if not isinstance(profile_payload, dict):
            raise ValueError(
                f"Config profile '{selected_profile}' in {path} must be a mapping."
            )
        merged = _deep_merge(merged, profile_payload)
        merged["_active_profile"] = selected_profile

    return merged
