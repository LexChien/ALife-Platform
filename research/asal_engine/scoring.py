"""Shared scoring formula for search, saved trajectories, and replay."""
import math
from copy import deepcopy

import numpy as np

from foundation_models import foundation_models
from .narrative_scores import _acceptance_config, score_narrative_trajectory
from .scores import supervised_target_score


def resolve_config(config):
    cfg = deepcopy(config)
    search = cfg["search"]
    seed = search.setdefault("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("search.seed must be an integer in [0, 2**32)")
    for group, names in (("runtime", ("steps", "substeps")), ("search", ("pop", "keep"))):
        for name in names:
            value = cfg[group][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{group}.{name} must be a positive integer")
    iters = search["iters"]
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 0:
        raise ValueError("search.iters must be a nonnegative integer")
    if not math.isfinite(search["sigma"]) or search["sigma"] < 0:
        raise ValueError("search.sigma must be finite and nonnegative")
    low, high = (np.asarray(search[key], dtype=float) for key in ("theta_low", "theta_high"))
    if low.ndim != 1 or not low.size or low.shape != high.shape:
        raise ValueError("theta bounds must be nonempty arrays of equal length")
    if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
        raise ValueError("theta bounds must be finite with low <= high")
    narrative = cfg.get("narrative", {})
    if narrative.get("enabled", False):
        narrative["acceptance"] = _acceptance_config(narrative.get("acceptance"))
    return cfg


class TrajectoryScorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.narrative_cfg = cfg.get("narrative", {})
        model_cfg = cfg["foundation_model"]
        self.fm = foundation_models.create(model_cfg["name"], **model_cfg.get("params", {}))
        self.text = self.fm.txt_embed(cfg["prompt"])
        self.morphology_fm = None
        self.morphology_text = None
        morph = cfg.get("morphology_judge", {})
        self.morphology_weight = float(morph.get("weight", 0.0)) if morph.get("enabled", False) else 0.0
        if not math.isfinite(self.morphology_weight) or not 0 <= self.morphology_weight <= 1:
            raise ValueError("morphology_judge.weight must be in [0, 1]")
        if morph.get("enabled", False):
            self.morphology_fm = foundation_models.create(morph["name"], **morph.get("params", {}))
            self.morphology_text = self.morphology_fm.txt_embed(cfg["prompt"])
        # Capture resolved execution devices, not the ambiguous 'auto' setting.
        for settings, model in ((model_cfg, self.fm), (morph, self.morphology_fm)):
            if settings.get("name") == "openclip" and model is not None:
                settings.setdefault("params", {})["device"] = model.device

    def score(self, frames):
        if len(frames) == 0:
            raise ValueError("Cannot score an empty trajectory")
        image = frames[-1]
        semantic = supervised_target_score([self.fm.img_embed(image)], self.text)
        morphology = None
        if self.morphology_fm is not None and self.morphology_weight > 0:
            morphology = supervised_target_score([self.morphology_fm.img_embed(image)], self.morphology_text)
        narrative = None
        if self.narrative_cfg.get("enabled", False):
            narrative = score_narrative_trajectory(frames, self.narrative_cfg)
        components = {"semantic": float(semantic), "morphology": morphology,
                      "narrative": narrative["total_score"] if narrative else None}
        weights = {"semantic": 1.0, "morphology": 0.0, "narrative": 0.0}
        if narrative is not None:
            supplied = {
                "semantic": self.narrative_cfg.get("semantic_weight", 0.2),
                "morphology": self.narrative_cfg.get("morphology_weight", 0.2 if morphology is not None else 0.0),
                "narrative": self.narrative_cfg.get("weight", 0.6),
            }
            if any(not math.isfinite(float(w)) or float(w) < 0 for w in supplied.values()):
                raise ValueError("Narrative score weights must be finite and nonnegative")
            active = {key: float(value) if components[key] is not None else 0.0 for key, value in supplied.items()}
            total = sum(active.values())
            if total > 0:
                weights = {key: value / total for key, value in active.items()}
            elif morphology is not None:
                weights.update(semantic=1 - self.morphology_weight, morphology=self.morphology_weight)
        elif morphology is not None:
            weights.update(semantic=1 - self.morphology_weight, morphology=self.morphology_weight)
        combined = float(sum(weights[key] * value for key, value in components.items() if value is not None))
        if not all(math.isfinite(value) for value in [combined, *[v for v in components.values() if v is not None]]):
            raise ValueError("Scoring produced a nonfinite value")
        return {"components": components, "weights": weights, "combined": combined}, narrative
