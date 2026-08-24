from __future__ import annotations

import numpy as np


def orthonormalize_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest right-handed orthonormal 3 x 3 matrix."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Rotation must be a finite 3 x 3 matrix")
    left, _, right = np.linalg.svd(value)
    result = left @ right
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


def euler_xyz_degrees_to_matrix(angles_degrees: np.ndarray) -> np.ndarray:
    """Build local export axes using intrinsic XYZ angles in degrees."""
    angles = np.asarray(angles_degrees, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("Euler angles must contain three finite values")
    x, y, z = np.deg2rad(angles)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to intrinsic XYZ angles in degrees."""
    value = orthonormalize_rotation(matrix)
    sy = -value[2, 0]
    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    cy = np.cos(y)
    if abs(cy) > 1e-8:
        x = np.arctan2(value[2, 1], value[2, 2])
        z = np.arctan2(value[1, 0], value[0, 0])
    else:
        x = np.arctan2(-value[1, 2], value[1, 1])
        z = 0.0
    return np.rad2deg([x, y, z])
