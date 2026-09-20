"""Narrative rewards and explicit geometric acceptance criteria."""
import math

import numpy as np

from .morphology import analyze_frame


DEFAULT_ACCEPTANCE = {
    "min_sustain_ratio": 0.25,
    "min_consecutive_ratio": 0.25,
    "min_consecutive_frames": 2,
    "max_fragment_fraction": 0.35,
    "max_centroid_jump_fraction": 0.20,
    "min_mass_retention": 0.60,
    "min_total_score": 0.50,
}


def _component_match_score(num_components: int, target_components: int) -> float:
    return max(0.0, 1.0 - abs(num_components - target_components) / max(target_components, 1))


def _normalize(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0)) if high > low else 0.0


def _foreground_area(stats):
    return float(stats.get("raw_foreground_area", sum(c["area"] for c in stats.get("components", []))))


def _mass_share(stats, component_count):
    area = _foreground_area(stats)
    selected = sum(c["area"] for c in stats.get("components", [])[:component_count])
    return float(selected / area) if area else 0.0


def score_birth_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 1)
    mass_score = _normalize(stats["foreground_fraction"], 0.01, 0.18)
    dominance_score = _normalize(_mass_share(stats, 1), 0.55, 0.95)
    circularity_score = _normalize(stats["largest_circularity_proxy"], 0.3, 1.0)
    return float(0.35 * component_score + 0.25 * mass_score + 0.25 * dominance_score + 0.15 * circularity_score)


def score_split_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 2)
    area_sum = stats["largest_area"] + stats["second_area"]
    area_balance = 1.0 - abs(stats["largest_area"] - stats["second_area"]) / area_sum if area_sum else 0.0
    separation_score = _normalize(stats["centroid_distance"] or 0.0, 6.0, 48.0)
    mass_score = _normalize(stats["foreground_fraction"], 0.015, 0.22)
    coherence_score = _normalize(_mass_share(stats, 2), 0.55, 0.95)
    return float(0.3 * component_score + 0.2 * area_balance + 0.2 * coherence_score + 0.15 * separation_score + 0.15 * mass_score)


def score_fusion_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 1)
    dominance_score = _normalize(_mass_share(stats, 1), 0.65, 0.98)
    mass_score = _normalize(stats["foreground_fraction"], 0.015, 0.25)
    circularity_score = _normalize(stats["largest_circularity_proxy"], 0.25, 1.0)
    return float(0.4 * component_score + 0.25 * dominance_score + 0.2 * mass_score + 0.15 * circularity_score)


def score_invasion_phase(stats: dict) -> float:
    """Like split, but prefer stronger spatial separation (invasive phenotype metaphor)."""
    base = score_split_phase(stats)
    separation_score = _normalize(stats["centroid_distance"] or 0.0, 18.0, 64.0)
    return float(0.7 * base + 0.3 * separation_score)


def _pairwise_centroid_span(stats: dict, target: int) -> float:
    comps = list(stats.get("dominant_components") or stats.get("components") or [])[:target]
    if len(comps) < 2:
        return float(stats.get("centroid_distance") or 0.0)
    cents = np.asarray([c["centroid"] for c in comps], dtype=float)
    total = 0.0
    count = 0
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            total += float(np.linalg.norm(cents[i] - cents[j]))
            count += 1
    return total / count if count else 0.0


def _area_balance(stats: dict, target: int) -> float:
    comps = list(stats.get("dominant_components") or stats.get("components") or [])[:target]
    if not comps:
        return 0.0
    areas = np.asarray([c["area"] for c in comps], dtype=float)
    if areas.sum() <= 0:
        return 0.0
    ideal = areas.sum() / max(len(areas), 1)
    return float(1.0 - np.mean(np.abs(areas - ideal)) / ideal)


def score_multi_colony_phase(stats: dict, target: int, sep_low: float, sep_high: float) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], target)
    balance = _area_balance(stats, target)
    separation_score = _normalize(_pairwise_centroid_span(stats, target), sep_low, sep_high)
    coherence_score = _normalize(_mass_share(stats, target), 0.50, 0.95)
    mass_score = _normalize(stats["foreground_fraction"], 0.015, 0.28)
    return float(
        0.34 * component_score
        + 0.18 * balance
        + 0.18 * coherence_score
        + 0.20 * separation_score
        + 0.10 * mass_score
    )


def score_colony_expansion_phase(stats: dict) -> float:
    """Three coherent colonies (mid proliferation metaphor)."""
    return score_multi_colony_phase(stats, target=3, sep_low=10.0, sep_high=56.0)


def score_metastasis_phase(stats: dict) -> float:
    """Four dispersed colonies (late proliferation / metastatic seeding metaphor)."""
    return score_multi_colony_phase(stats, target=4, sep_low=14.0, sep_high=70.0)


# Canonical phases + cancer-progression narrative aliases (visual metaphor, not biomedical model).
_PHASE_SCORERS = {
    "birth": score_birth_phase,
    "split": score_split_phase,
    "fusion": score_fusion_phase,
    "initiation": score_birth_phase,
    "proliferation": score_split_phase,
    "invasion": score_invasion_phase,
    "colony_expansion": score_colony_expansion_phase,
    "metastasis": score_metastasis_phase,
}
_PHASE_TARGETS = {
    "birth": 1,
    "split": 2,
    "fusion": 1,
    "initiation": 1,
    "proliferation": 2,
    "invasion": 2,
    "colony_expansion": 3,
    "metastasis": 4,
}



def malignant_fraction_from_frame(frame) -> float:
    """Infer malignant vs normal mass from dual-color infection render (warm vs cool)."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    rgb = arr[..., :3].astype(np.float32)
    warm = (rgb[..., 0] > rgb[..., 2] + 12.0) & (rgb[..., 0] > 28.0)
    cool = (rgb[..., 2] > rgb[..., 0] + 12.0) & (rgb[..., 2] > 28.0)
    body = warm | cool
    denom = float(body.sum())
    if denom <= 0:
        return 0.0
    return float(warm.sum() / denom)


def score_infection_fraction(fraction: float, low: float, high: float) -> float:
    """Peak score inside [low, high]; soft falloff outside."""
    mid = 0.5 * (low + high)
    half = max(0.5 * (high - low), 1e-6)
    return float(np.clip(1.0 - abs(fraction - mid) / (half * 1.35), 0.0, 1.0))


# Infection-conversion narrative (visual metaphor of malignant contact conversion).
# Values are target malignant_fraction bands, not component counts.
_INFECTION_FRACTION_TARGETS = {
    "healthy_tissue": (0.0, 0.10),
    "malignant_seeding": (0.08, 0.25),
    "contact_conversion": (0.28, 0.58),
    "tissue_takeover": (0.60, 0.95),
}


def _acceptance_config(config):
    supplied = config or {}
    unknown = set(supplied) - set(DEFAULT_ACCEPTANCE)
    if unknown:
        raise ValueError(f"Unknown narrative acceptance settings: {sorted(unknown)}")
    result = {**DEFAULT_ACCEPTANCE, **supplied}
    for name, value in result.items():
        if name == "min_consecutive_frames":
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError("min_consecutive_frames must be an integer >= 2")
        else:
            allow_zero = name in {"max_fragment_fraction", "min_total_score"}
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if not valid or not (0 <= value <= 1 if allow_zero else 0 < value <= 1):
                raise ValueError(f"Invalid narrative acceptance setting: {name}={value}")
    return result


def _longest_run(matches):
    longest = current = 0
    for matches_target in matches:
        current = current + 1 if matches_target else 0
        longest = max(longest, current)
    return longest


def _centroids(stats):
    values = [c["centroid"] for c in stats.get("dominant_components", [])]
    if not values and stats.get("largest_centroid") is not None:
        values = [stats["largest_centroid"]]
    return np.asarray(values, dtype=float).reshape(-1, 2)


def _body_continuity(frame_stats, width, height, policy):
    """Bidirectional matching; missing bodies cannot bridge a gap."""
    threshold = policy["max_centroid_jump_fraction"] * min(width, height)
    max_jump = 0.0
    minimum_mass_ratio = 1.0
    seen_body = False
    events = []
    for index, stats in enumerate(frame_stats):
        current = _centroids(stats)
        if not len(current):
            if seen_body:
                events.append({"frame_index": index, "reason": "missing_body"})
            continue
        seen_body = True
        if index == 0:
            continue
        previous_stats = frame_stats[index - 1]
        previous = _centroids(previous_stats)
        if not len(previous):
            continue  # A missing_body event already invalidates later reappearance.
        distances = current[:, None, :] - previous[None, :, :]
        dimensions = np.asarray([width, height], dtype=float)
        distances -= dimensions * np.round(distances / dimensions)
        distances = np.linalg.norm(distances, axis=-1)
        jump = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
        max_jump = max(max_jump, jump)
        if jump > threshold:
            events.append({"frame_index": index, "reason": "centroid_jump", "distance": jump})
        # Speckles cannot conceal loss of tracked bodies.
        old_mass = sum(c["area"] for c in previous_stats.get("dominant_components", []))
        new_mass = sum(c["area"] for c in stats.get("dominant_components", []))
        if old_mass > 0:
            ratio = float(new_mass / old_mass)
            minimum_mass_ratio = min(minimum_mass_ratio, ratio)
            if ratio < policy["min_mass_retention"]:
                events.append({"frame_index": index, "reason": "body_mass_loss", "retained_ratio": ratio})
    if not seen_body:
        events.append({"frame_index": None, "reason": "no_body"})
    score = max(0.0, min(1.0, 2.0 - max_jump / threshold))
    score *= min(1.0, minimum_mass_ratio / policy["min_mass_retention"])
    if any(event["reason"] in {"missing_body", "no_body"} for event in events):
        score = 0.0
    return {
        "score": float(score), "accepted": not events, "events": events,
        "max_centroid_jump": max_jump, "max_allowed_jump": threshold,
        "minimum_mass_retention": minimum_mass_ratio,
    }


def _compute_body_continuity_score(frame_stats, width=128, height=128) -> float:
    return _body_continuity(frame_stats, width, height, DEFAULT_ACCEPTANCE)["score"]


def score_narrative_trajectory(frames, narrative_cfg: dict) -> dict:
    frame_arrays = [np.asarray(frame) for frame in frames]
    phases = narrative_cfg.get("phases", [])
    if not frame_arrays or not phases:
        raise ValueError("Narrative scoring requires nonempty frames and phases")
    if any(frame.shape != frame_arrays[0].shape for frame in frame_arrays):
        raise ValueError("Narrative frames must have identical shapes")
    policy = _acceptance_config(narrative_cfg.get("acceptance"))
    frame_stats = [analyze_frame(frame) for frame in frame_arrays]
    phase_results = []
    weighted_total = total_weight = 0.0
    previous_end = -1
    malignant_fractions = [malignant_fraction_from_frame(frame) for frame in frame_arrays]
    infection_mode = any(phase["name"] in _INFECTION_FRACTION_TARGETS for phase in phases)
    for phase in phases:
        name = phase["name"]
        is_infection = name in _INFECTION_FRACTION_TARGETS
        if (not is_infection) and name not in _PHASE_SCORERS:
            raise ValueError(f"Unsupported narrative phase: {name}")
        start, end = phase["frame_range"]
        if (not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start <= end < len(frame_stats) or start <= previous_end):
            raise ValueError("Phase ranges must be ordered, disjoint, and within the trajectory")
        previous_end = end
        weight = float(phase.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Phase weights must be positive and finite")
        indices = list(range(start, end + 1))
        duration = len(indices)
        required = max(policy["min_consecutive_frames"], math.ceil(duration * policy["min_consecutive_ratio"]))

        if is_infection:
            low, high = phase.get("target_fraction_range", _INFECTION_FRACTION_TARGETS[name])
            low, high = float(low), float(high)
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(f"Invalid target_fraction_range for {name}")
            fracs = [malignant_fractions[i] for i in indices]
            matches = [low <= f <= high for f in fracs]
            # Soft qualify: within expanded band
            pad = max(0.04, 0.25 * (high - low))
            qualified = [(low - pad) <= f <= (high + pad) for f in fracs]
            candidates = [i for i, good in zip(indices, matches) if good] or indices
            best_index = max(candidates, key=lambda i: score_infection_fraction(malignant_fractions[i], low, high))
            best_score = score_infection_fraction(malignant_fractions[best_index], low, high)
            sustain_ratio = sum(matches) / duration
            qualified_ratio = sum(qualified) / duration
            longest = _longest_run(qualified)
            penalty = min(1.0, max(qualified_ratio / max(policy["min_sustain_ratio"], 1e-6), 1e-6), max(longest / required, 1e-6))
            # Infection arcs tolerate brief misses if overall fraction trend is right.
            penalty = max(penalty, 0.35 * best_score)
            score = float(max(best_score, 0.0) * min(1.0, penalty))
            reasons = []
            if not any(qualified):
                reasons.append("malignant_fraction_band_not_observed")
            if qualified_ratio < policy["min_sustain_ratio"] * 0.6:
                reasons.append("insufficient_infection_coverage")
            phase_results.append({
                "name": name, "frame_index": best_index, "frame_range": [start, end],
                "target_components": None,
                "target_fraction_range": [low, high],
                "malignant_fraction": float(malignant_fractions[best_index]),
                "score": score, "sustain_ratio": sustain_ratio,
                "qualified_sustain_ratio": qualified_ratio, "longest_target_run": _longest_run(matches),
                "longest_qualified_run": longest, "required_consecutive_frames": required,
                "fragment_fraction": 0.0,
                "accepted": not reasons, "failure_reasons": reasons, "stats": frame_stats[best_index],
            })
        else:
            target = phase.get("target_components", _PHASE_TARGETS[name])
            if target != _PHASE_TARGETS[name]:
                raise ValueError(f"Phase {name} requires target_components={_PHASE_TARGETS[name]}")
            matches = [frame_stats[i]["dominant_num_components"] == target for i in indices]
            fragments = [max(0.0, 1.0 - _mass_share(frame_stats[i], target)) for i in indices]
            qualified = [match and fraction <= policy["max_fragment_fraction"] for match, fraction in zip(matches, fragments)]
            candidates = [i for i, good in zip(indices, qualified) if good]
            candidates = candidates or [i for i, match in zip(indices, matches) if match] or indices
            best_index = max(candidates, key=lambda i: _PHASE_SCORERS[name](frame_stats[i]))
            best_score = _PHASE_SCORERS[name](frame_stats[best_index])
            sustain_ratio = sum(matches) / duration
            qualified_ratio = sum(qualified) / duration
            longest = _longest_run(qualified)
            penalty = min(1.0, qualified_ratio / policy["min_sustain_ratio"], longest / required)
            score = float(max(best_score, 0.0) * penalty)
            reasons = []
            if not any(matches):
                reasons.append("target_count_not_observed")
            if qualified_ratio < policy["min_sustain_ratio"]:
                reasons.append("insufficient_qualified_coverage")
            if longest < required:
                reasons.append("insufficient_consecutive_frames")
            if any(match and not good for match, good in zip(matches, qualified)):
                if qualified_ratio < policy["min_sustain_ratio"] or longest < required:
                    reasons.append("excess_fragmentation")
            phase_results.append({
                "name": name, "frame_index": best_index, "frame_range": [start, end],
                "target_components": target, "score": score, "sustain_ratio": sustain_ratio,
                "qualified_sustain_ratio": qualified_ratio, "longest_target_run": _longest_run(matches),
                "longest_qualified_run": longest, "required_consecutive_frames": required,
                "fragment_fraction": fragments[best_index - start],
                "accepted": not reasons, "failure_reasons": reasons, "stats": frame_stats[best_index],
            })
        weighted_total += weight * score
        total_weight += weight

    if infection_mode:
        actual = [round(float(phase.get("malignant_fraction", 0.0)), 3) for phase in phase_results]
        expected = [
            round(0.5 * (phase["target_fraction_range"][0] + phase["target_fraction_range"][1]), 3)
            for phase in phase_results
        ]
        # Valid if malignant fraction is non-decreasing across phase keyframes.
        phase_order_valid = all(actual[i] <= actual[i + 1] + 1e-6 for i in range(len(actual) - 1))
        # And each keyframe roughly inside its band (soft).
        for phase, frac in zip(phase_results, actual):
            low, high = phase["target_fraction_range"]
            pad = max(0.06, 0.3 * (high - low))
            if not ((low - pad) <= frac <= (high + pad)):
                phase_order_valid = False
                break
    else:
        actual = [phase["stats"]["dominant_num_components"] for phase in phase_results]
        expected = [phase["target_components"] for phase in phase_results]
        phase_order_valid = actual == expected
    height, width = frame_arrays[0].shape[:2]
    continuity = _body_continuity(frame_stats, width, height, policy)
    cont_score = continuity["score"]
    if infection_mode:
        # Dual-color infection fields can flicker mask topology; keep a floor.
        cont_score = max(cont_score, 0.55)
    total_score = weighted_total / total_weight * cont_score
    if not phase_order_valid:
        total_score *= 0.35
    reasons = [f"{phase['name']}:{reason}" for phase in phase_results for reason in phase["failure_reasons"]]
    if not phase_order_valid:
        reasons.append("phase_order_invalid")
    if infection_mode:
        # Dual-color infection renders often flicker grayscale morphology mass; do not
        # reject a valid malignant-fraction arc solely on continuity topology events.
        severe = {"no_body", "missing_body"}
        reasons.extend(
            f"continuity:{reason}"
            for reason in dict.fromkeys(event["reason"] for event in continuity["events"])
            if reason in severe
        )
    else:
        reasons.extend(f"continuity:{reason}" for reason in dict.fromkeys(event["reason"] for event in continuity["events"]))
    if total_score < policy["min_total_score"]:
        reasons.append("score_below_threshold")
    result = {
        "schema_version": "2.0", "total_score": float(total_score),
        "accepted": not reasons, "failure_reasons": reasons, "acceptance": policy,
        "phase_order_valid": phase_order_valid, "expected_component_sequence": expected,
        "actual_component_sequence": actual, "phases": phase_results, "frame_stats": frame_stats,
        "continuity_score": continuity["score"], "continuity": continuity,
    }
    if infection_mode:
        result["narrative_mode"] = "infection_conversion"
        result["malignant_fraction_series"] = malignant_fractions
        result["expected_fraction_mids"] = expected
        result["actual_fraction_keyframes"] = actual
    return result
