from __future__ import annotations

import numpy as np


ORIGIN_MODES: tuple[tuple[str, str], ...] = (
    ("Camera origin (0, 0, 0)", "camera"),
    ("Manual (drag cyan handle)", "manual"),
    ("Selection bounding-box center", "center"),
    ("Minimum X point", "min_x"),
    ("Maximum X point", "max_x"),
    ("Minimum Y point (visual top)", "min_y"),
    ("Maximum Y point (visual bottom)", "max_y"),
    ("Minimum Z point", "min_z"),
    ("Maximum Z point", "max_z"),
)

VALID_ORIGIN_MODES = frozenset(key for _, key in ORIGIN_MODES)


def calculate_origin_anchor(
    points: np.ndarray,
    mode: str,
    manual_origin_m: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate an export origin in camera coordinates for the requested mode."""
    if mode not in VALID_ORIGIN_MODES:
        raise ValueError(f"Unknown origin mode: {mode}")
    if mode == "camera":
        return np.zeros(3, dtype=np.float64)
    if mode == "manual":
        origin = np.zeros(3) if manual_origin_m is None else np.asarray(manual_origin_m)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("manual_origin_m must contain three finite coordinates")
        return origin.astype(np.float64, copy=True)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("Cannot calculate an automatic origin from an empty point cloud")
    if mode == "center":
        return (points.min(axis=0) + points.max(axis=0)) * 0.5

    axis = {"x": 0, "y": 1, "z": 2}[mode[-1]]
    index = np.argmin(points[:, axis]) if mode.startswith("min") else np.argmax(points[:, axis])
    return points[index].copy()
