from __future__ import annotations

from typing import Iterable


def contains_all(text: str, tokens: Iterable[str]) -> bool:
    return all(token in text for token in tokens)


def contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def excludes_all(text: str, tokens: Iterable[str]) -> bool:
    return all(token not in text for token in tokens)


def score_checks(checks: dict[str, bool]) -> float:
    return sum(1.0 for ok in checks.values() if ok) / max(len(checks), 1)


def make_result(checks: dict[str, bool]) -> dict[str, object]:
    return {
        "checks": checks,
        "score": score_checks(checks),
        "pass": all(checks.values()),
    }

