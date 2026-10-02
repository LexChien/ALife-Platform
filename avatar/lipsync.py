"""Plan 38 J3.2: amplitude-driven 2D mouth keyframes (identity-preserving warp of the FIXED avatar).

Offline: tools/render_mouth_keyframes.py renders K openness levels (only the mouth/jaw ROI is displaced; k=0 is the
untouched avatar) and every frame must pass the appearance guard. Online (web/gemma_chat/avatar.js): the browser
maps the playing TTS audio's RMS envelope to a keyframe index each animation frame. This module holds the shared
mapping and the offline sync metric (Pearson r between keyframe openness and the audio envelope, and the lag of the
cross-correlation peak) so the browser mapping can be verified against real TTS WAVs."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

K = 8
HOP_S = 0.02  # 50 Hz, close to the browser's requestAnimationFrame rate
ATTACK, RELEASE = 0.6, 0.25  # same smoothing as avatar.js


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x, sr


def envelope(x: np.ndarray, sr: int, hop_s: float = HOP_S) -> np.ndarray:
    hop = max(1, int(sr * hop_s))
    n = len(x) // hop
    return np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(n)], dtype=np.float32)


def keyframe_indices(env: np.ndarray, k: int = K, gain: float | None = None) -> np.ndarray:
    """Envelope -> keyframe index with attack/release smoothing (identical to the browser implementation)."""
    ref = gain if gain is not None else (np.percentile(env, 95) + 1e-9)
    level, out = 0.0, []
    for v in env:
        target = min(1.0, float(v) / ref)
        level += (ATTACK if target > level else RELEASE) * (target - level)
        out.append(int(round(level * (k - 1))))
    return np.array(out)


def browser_indices(env: np.ndarray, k: int = K) -> np.ndarray:
    """Exact mirror of web/gemma_chat/avatar.js tick(): adaptive gain (starts 1/0.12, never amplifies quiet
    frames below 0.02 RMS, slowly recovers x1.002/frame), attack/release smoothing, round to K levels."""
    gain, level, out = 1 / 0.12, 0.0, []
    for rms in env:
        gain = min(gain, 1 / max(float(rms), 0.02)) * 1.002
        target = min(1.0, float(rms) * gain)
        level += (ATTACK if target > level else RELEASE) * (target - level)
        out.append(max(0, min(k - 1, int(round(level * (k - 1))))))
    return np.array(out)


def sync_metrics(env: np.ndarray, idx: np.ndarray, hop_s: float = HOP_S, max_lag_s: float = 0.3) -> dict:
    a = (env - env.mean()) / (env.std() + 1e-9)
    b = (idx - idx.mean()) / (idx.std() + 1e-9)
    r0 = float(np.mean(a * b))
    best, best_lag = -2.0, 0
    m = int(max_lag_s / hop_s)
    for lag in range(-m, m + 1):
        if lag >= 0:
            r = float(np.mean(a[:len(a) - lag] * b[lag:])) if lag < len(a) else -2
        else:
            r = float(np.mean(a[-lag:] * b[:len(b) + lag]))
        if r > best:
            best, best_lag = r, lag
    return {"pearson_r": round(r0, 3), "best_r": round(best, 3), "lag_ms": int(best_lag * hop_s * 1000)}
