from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = "unknown"
    coeffs: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "model": self.model,
            "coeffs": list(self.coeffs),
        }


@dataclass(slots=True)
class CaptureFrame:
    """One RGB-D, point-cloud, or triangle-mesh frame with coordinates in metres."""

    xyz: np.ndarray  # H x W x 3, float32, metres
    rgb: np.ndarray  # H x W x 3, uint8, RGB order
    depth_mm: np.ndarray  # H x W, uint16
    intrinsics: CameraIntrinsics
    timestamp_ms: float
    source: str
    organized: bool = True
    zero_is_invalid: bool = False
    source_faces: np.ndarray | None = None
    coordinate_system: str = "+X right, +Y down, +Z forward (Intel RealSense)"
    source_units: str = "metres"
    processing_metadata: dict[str, Any] | None = None

    @property
    def valid_mask(self) -> np.ndarray:
        valid = np.isfinite(self.xyz).all(axis=2)
        if self.zero_is_invalid:
            valid &= np.any(np.abs(self.xyz) > 1e-12, axis=2)
        return valid

    @property
    def valid_points(self) -> np.ndarray:
        return self.xyz[self.valid_mask]

    @property
    def valid_colors(self) -> np.ndarray:
        return self.rgb[self.valid_mask]
