"""Plan 40 T3.4/T4.3: switchable DigiClone profiles (persona + speech style + TTS voice).

The original DigiClone persona (``life.persona`` + ``system`` + ``cognition.speech_rules`` + ``voice.tts_voice``) is
always available as profile ``digiclone`` and is never modified. Extra profiles (e.g. ``yaying``) come from
``clone_profiles`` in the web config; their persona/tone text lives in a PRIVATE machine-local YAML
(``persona_file``, e.g. runs/yaying_clone/persona_yaying.yaml, derived from transcripts, never committed). A profile
whose persona file is missing is listed as unavailable and cannot be activated (the base profile stays active).

Config shape::

  clone_profiles:
    default: yaying            # or digiclone; env GEMMA_WEB_CLONE_PROFILE overrides
    profiles:
      yaying: {label: 雅英, persona_file: runs/yaying_clone/persona_yaying.yaml, tts_voice: 雅英}
  voices:                      # extra server-side voices beyond macOS say voices
    雅英: {provider: clone_tts, engine: f5, python: ~/yaying_cache/venv_f5/bin/python,
           ref_wav: runs/yaying_clone/tts_ref/ref.wav, ref_text_file: runs/yaying_clone/tts_ref/ref.txt}
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = "digiclone"


def _resolve(path: str | os.PathLike | None, root: Path = ROOT) -> Path | None:
    if not path:
        return None
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else root / p


class CloneProfiles:
    def __init__(self, cfg: dict, root: Path = ROOT):
        self.root = root
        cfg = cfg or {}
        voice_cfg = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
        life = cfg.get("life") if isinstance(cfg.get("life"), dict) else {}
        cog = cfg.get("cognition") if isinstance(cfg.get("cognition"), dict) else {}
        self.base = {
            "id": BASE, "label": "DigiClone", "available": True,
            "persona": copy.deepcopy(life.get("persona") or {}),
            "system": cfg.get("system"),
            "speech_rules": cog.get("speech_rules"),
            "tts_voice": voice_cfg.get("tts_voice", "Meijia"),
        }
        block = cfg.get("clone_profiles") if isinstance(cfg.get("clone_profiles"), dict) else {}
        self.profiles: dict[str, dict] = {BASE: self.base}
        for pid, spec in (block.get("profiles") or {}).items():
            if pid == BASE or not isinstance(spec, dict):
                continue
            self.profiles[pid] = self._load(pid, spec)
        self.voices: dict[str, dict] = {k: dict(v) for k, v in (cfg.get("voices") or {}).items() if isinstance(v, dict)}
        want = os.environ.get("GEMMA_WEB_CLONE_PROFILE") or block.get("default") or BASE
        self.active = want if self.profiles.get(want, {}).get("available") else BASE
        self.requested_default = want

    def _load(self, pid: str, spec: dict) -> dict:
        prof = {"id": pid, "label": spec.get("label", pid), "tts_voice": spec.get("tts_voice", self.base["tts_voice"]),
                "persona_file": spec.get("persona_file"), "available": False, "error": None}
        path = _resolve(spec.get("persona_file"), self.root)
        data = dict(spec.get("inline") or {})
        if path is not None:
            if path.exists():
                try:
                    data.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
                except Exception as exc:  # malformed private file -> unavailable, base keeps working
                    prof["error"] = f"{type(exc).__name__}: {exc}"
                    return prof
            elif not data:
                prof["error"] = f"persona_file missing: {spec.get('persona_file')}"
                return prof
        persona = copy.deepcopy(self.base["persona"])
        persona.update(data.get("persona") or {})
        prof.update({"persona": persona, "system": data.get("system", self.base["system"]),
                     "speech_rules": "\n".join(p for p in [self.base["speech_rules"], data.get("speech_style")] if p),
                     "tone_prompt": data.get("speech_style"), "available": True,
                     "tts_voice": data.get("tts_voice", prof["tts_voice"])})
        return prof

    def get(self, pid: str | None = None) -> dict:
        return self.profiles[pid or self.active]

    def switch(self, pid: str) -> dict:
        if pid not in self.profiles:
            raise KeyError(f"unknown profile {pid!r}; have {sorted(self.profiles)}")
        if not self.profiles[pid].get("available"):
            raise ValueError(f"profile {pid!r} unavailable: {self.profiles[pid].get('error')}")
        self.active = pid
        return self.profiles[pid]

    def voice_spec(self, name: str) -> dict | None:
        return self.voices.get(name)

    def public(self) -> dict:
        return {"active": self.active, "requested_default": self.requested_default,
                "profiles": {k: {"label": v.get("label"), "available": bool(v.get("available")), "tts_voice": v.get("tts_voice"),
                                 "error": v.get("error")} for k, v in self.profiles.items()},
                "voices": sorted(set(self.voices) | {self.base["tts_voice"]})}


def build_voice(spec: dict, fallback=None, root: Path = ROOT, name: str = ""):
    """Instantiate an extra server-side voice from ``voices:`` config (currently provider clone_tts)."""
    if spec.get("provider") != "clone_tts":
        raise ValueError(f"unsupported voice provider {spec.get('provider')!r}")
    from voice.clone_tts import CloneTTS
    ref_text = spec.get("ref_text") or ""
    tf = _resolve(spec.get("ref_text_file"), root)
    if not ref_text and tf is not None and tf.exists():
        ref_text = tf.read_text(encoding="utf-8").strip()
    return CloneTTS(voice=name or spec.get("label", "clone"), engine=spec.get("engine", "f5"), python=spec["python"],
                    ref_wav=str(_resolve(spec["ref_wav"], root)), ref_text=ref_text, device=spec.get("device", "mps"),
                    nfe=int(spec.get("nfe", 16)), fallback=fallback, extra_args=spec.get("extra_args"))
