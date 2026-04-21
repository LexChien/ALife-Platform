import re

import numpy as np

from . import foundation_models


@foundation_models.register("morphology_judge_stub")
class MorphologyJudgeStub:
    """
    Deterministic hand-built judge that maps text prompts and rendered images
    into the same morphology-oriented feature space.

    It is still a stub, but it is more meaningful than a random projection
    because image features correspond to visible structure and prompt features
    correspond to requested morphology cues.
    """

    def __init__(self):
        self.dim = 8

    @property
    def device(self):
        return "cpu"

    def txt_embed(self, text: str):
        lowered = text.lower()
        vec = np.array(
            [
                self._has_any(lowered, ["cell", "biological", "organism"]),
                self._has_any(lowered, ["microscopy", "phase contrast", "microscope"]),
                self._has_any(lowered, ["round", "circular", "cell"]),
                self._has_any(lowered, ["contrast", "phase"]),
                self._has_any(lowered, ["blue", "cyan", "membrane", "cell"]),
                self._has_any(lowered, ["center", "centered", "single"]),
                self._has_any(lowered, ["texture", "pattern", "structure"]),
                1.0,
            ],
            dtype=np.float32,
        )
        return vec / (np.linalg.norm(vec) + 1e-9)

    def img_embed(self, pil_img):
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        h, w, _ = arr.shape

        gray = arr.mean(axis=-1)
        blue_dominance = float(np.mean(arr[..., 2] - 0.5 * (arr[..., 0] + arr[..., 1])))
        contrast = float(gray.std())
        occupancy = float(np.mean(gray > gray.mean()))

        ys, xs = np.mgrid[0:h, 0:w]
        weights = gray + 1e-6
        cy = float((ys * weights).sum() / weights.sum() / max(h - 1, 1))
        cx = float((xs * weights).sum() / weights.sum() / max(w - 1, 1))
        center_bias = 1.0 - min(np.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2) / 0.71, 1.0)

        left = gray[:, : w // 2]
        right = np.fliplr(gray[:, w - left.shape[1] :])
        symmetry = 1.0 - float(np.mean(np.abs(left - right)))

        r = np.sqrt((ys / max(h - 1, 1) - 0.5) ** 2 + (xs / max(w - 1, 1) - 0.5) ** 2)
        radial_profile = []
        for lo, hi in [(0.0, 0.12), (0.12, 0.24), (0.24, 0.36), (0.36, 0.50)]:
            mask = (r >= lo) & (r < hi)
            radial_profile.append(float(gray[mask].mean()) if np.any(mask) else 0.0)
        central_peak = radial_profile[0] - radial_profile[-1]

        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        texture = float((gx + gy) * 0.5)

        vec = np.array(
            [
                occupancy,
                contrast,
                max(symmetry, 0.0),
                max(center_bias, 0.0),
                max(blue_dominance + 0.5, 0.0),
                max(central_peak + 0.5, 0.0),
                texture,
                1.0,
            ],
            dtype=np.float32,
        )
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _has_any(self, text: str, keywords: list[str]) -> float:
        return 1.0 if any(re.search(re.escape(word), text) for word in keywords) else 0.0
