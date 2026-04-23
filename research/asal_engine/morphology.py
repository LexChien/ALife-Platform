import math
from collections import deque

import numpy as np


def _to_gray(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported frame shape for morphology analysis: {arr.shape}")
    gray = gray - gray.min()
    maxv = gray.max()
    return gray / (maxv + 1e-9)


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    threshold = max(0.22, float(gray.mean() + 0.55 * gray.std()))
    mask = gray >= threshold
    if mask.any():
        return mask
    fallback = gray >= max(0.12, float(gray.mean()))
    return fallback


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            coords = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                    (cy - 1, cx - 1),
                    (cy - 1, cx + 1),
                    (cy + 1, cx - 1),
                    (cy + 1, cx + 1),
                ):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(coords)
    return components


def _component_stats(coords):
    pts = np.asarray(coords, dtype=np.float32)
    area = float(len(coords))
    centroid_yx = pts.mean(axis=0)
    yy = pts[:, 0] - centroid_yx[0]
    xx = pts[:, 1] - centroid_yx[1]
    radius = float(np.sqrt((xx * xx + yy * yy).mean() + 1e-9))
    circularity_proxy = float(area / (math.pi * radius * radius + 1e-9))
    return {
        "area": area,
        "centroid": [float(centroid_yx[1]), float(centroid_yx[0])],
        "radius": radius,
        "circularity_proxy": circularity_proxy,
    }


def analyze_frame(
    frame,
    min_component_area: int = 12,
    dominant_component_ratio: float = 0.30,
    dominant_component_min_area: int = 48,
) -> dict:
    gray = _to_gray(frame)
    mask = _foreground_mask(gray)
    raw_components = _connected_components(mask)
    components = [coords for coords in raw_components if len(coords) >= min_component_area]
    stats = sorted((_component_stats(coords) for coords in components), key=lambda item: item["area"], reverse=True)

    total_foreground = float(sum(item["area"] for item in stats))
    largest = stats[0] if stats else None
    second = stats[1] if len(stats) > 1 else None
    dominant_threshold = 0.0
    if largest is not None:
        dominant_threshold = max(float(dominant_component_min_area), float(largest["area"]) * float(dominant_component_ratio))
    dominant_components = [item for item in stats if item["area"] >= dominant_threshold] if largest is not None else []
    centroid_distance = None
    first_dominant = dominant_components[0] if dominant_components else largest
    second_dominant = dominant_components[1] if len(dominant_components) > 1 else second
    if first_dominant is not None and second_dominant is not None:
        dx = first_dominant["centroid"][0] - second_dominant["centroid"][0]
        dy = first_dominant["centroid"][1] - second_dominant["centroid"][1]
        centroid_distance = float(math.sqrt(dx * dx + dy * dy))

    return {
        "num_components": len(stats),
        "dominant_num_components": len(dominant_components),
        "largest_area": float(largest["area"]) if largest else 0.0,
        "second_area": float(second["area"]) if second else 0.0,
        "dominant_mass_ratio": float((largest["area"] / total_foreground) if largest and total_foreground > 0 else 0.0),
        "largest_centroid": largest["centroid"] if largest else None,
        "second_centroid": second["centroid"] if second else None,
        "centroid_distance": centroid_distance,
        "foreground_fraction": float(total_foreground / mask.size),
        "largest_radius": float(largest["radius"]) if largest else 0.0,
        "largest_circularity_proxy": float(largest["circularity_proxy"]) if largest else 0.0,
        "dominant_component_threshold": float(dominant_threshold),
        "dominant_components": dominant_components,
        "components": stats,
    }
