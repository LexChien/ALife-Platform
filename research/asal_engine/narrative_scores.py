import numpy as np

from .morphology import analyze_frame


def _component_match_score(num_components: int, target_components: int) -> float:
    return max(0.0, 1.0 - abs(num_components - target_components) / max(target_components, 1))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def score_birth_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 1)
    mass_score = _normalize(stats["foreground_fraction"], 0.01, 0.18)
    dominance_score = _normalize(stats["dominant_mass_ratio"], 0.55, 0.95)
    circularity_score = _normalize(stats["largest_circularity_proxy"], 0.3, 1.0)
    return float(0.35 * component_score + 0.25 * mass_score + 0.25 * dominance_score + 0.15 * circularity_score)


def score_split_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 2)
    area_sum = stats["largest_area"] + stats["second_area"]
    area_balance = 0.0
    if area_sum > 0:
        area_balance = 1.0 - abs(stats["largest_area"] - stats["second_area"]) / area_sum
    separation_score = _normalize(stats["centroid_distance"] or 0.0, 6.0, 48.0)
    mass_score = _normalize(stats["foreground_fraction"], 0.015, 0.22)
    return float(0.4 * component_score + 0.25 * area_balance + 0.2 * separation_score + 0.15 * mass_score)


def score_fusion_phase(stats: dict) -> float:
    component_score = _component_match_score(stats["dominant_num_components"], 1)
    dominance_score = _normalize(stats["dominant_mass_ratio"], 0.65, 0.98)
    mass_score = _normalize(stats["foreground_fraction"], 0.015, 0.25)
    circularity_score = _normalize(stats["largest_circularity_proxy"], 0.25, 1.0)
    return float(0.4 * component_score + 0.25 * dominance_score + 0.2 * mass_score + 0.15 * circularity_score)


_PHASE_SCORERS = {
    "birth": score_birth_phase,
    "split": score_split_phase,
    "fusion": score_fusion_phase,
}


def _frame_slice(frame_count: int, frame_range):
    start, end = int(frame_range[0]), int(frame_range[1])
    start = max(0, min(start, frame_count - 1))
    end = max(start, min(end, frame_count - 1))
    return start, end


def score_narrative_trajectory(frames, narrative_cfg: dict) -> dict:
    frame_arrays = [np.asarray(frame) for frame in frames]
    frame_stats = [analyze_frame(frame) for frame in frame_arrays]
    phases = narrative_cfg.get("phases", [])

    phase_results = []
    weighted_total = 0.0
    total_weight = 0.0
    selected_indices = []
    expected_sequence = []

    for phase in phases:
        name = phase["name"]
        scorer = _PHASE_SCORERS[name]
        start, end = _frame_slice(len(frame_stats), phase["frame_range"])
        target = int(phase.get("target_components", 1))
        expected_sequence.append(target)
        best_score = -1.0
        best_index = start
        best_stats = frame_stats[start]
        for idx in range(start, end + 1):
            stats = frame_stats[idx]
            score = scorer(stats)
            if score > best_score:
                best_score = score
                best_index = idx
                best_stats = stats
        weight = float(phase.get("weight", 1.0))
        total_weight += weight
        weighted_total += weight * max(best_score, 0.0)
        selected_indices.append(best_index)
        phase_results.append(
            {
                "name": name,
                "frame_index": int(best_index),
                "frame_range": [start, end],
                "target_components": target,
                "score": float(max(best_score, 0.0)),
                "stats": best_stats,
            }
        )

    actual_sequence = [item["stats"]["dominant_num_components"] for item in phase_results]
    phase_order_valid = actual_sequence == expected_sequence and selected_indices == sorted(selected_indices)
    total_score = float(weighted_total / total_weight) if total_weight > 0 else 0.0
    if not phase_order_valid:
        total_score *= 0.35

    return {
        "total_score": total_score,
        "phase_order_valid": phase_order_valid,
        "expected_component_sequence": expected_sequence,
        "actual_component_sequence": actual_sequence,
        "phases": phase_results,
        "frame_stats": frame_stats,
    }
