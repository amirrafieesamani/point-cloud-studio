from __future__ import annotations

import unittest

import numpy as np
import pyvista as pv

from realsense_cropper.models import CameraIntrinsics, CaptureFrame
from realsense_cropper.normals import estimate_normals


def make_frame(points: np.ndarray, faces: np.ndarray | None = None) -> CaptureFrame:
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim == 2:
        xyz = xyz[None, :, :]
    height, width = xyz.shape[:2]
    return CaptureFrame(
        xyz=xyz,
        rgb=np.full((height, width, 3), 180, dtype=np.uint8),
        depth_mm=np.zeros((height, width), dtype=np.uint16),
        intrinsics=CameraIntrinsics(width, height, 1.0, 1.0, 0.0, 0.0),
        timestamp_ms=0.0,
        source="normal-test",
        organized=height > 1,
        source_faces=faces,
    )


class NormalTests(unittest.TestCase):
    def test_organized_normals_can_point_toward_or_away_from_view(self) -> None:
        y, x = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01]
        frame = make_frame(np.dstack((x, y, np.ones_like(x))))
        toward = estimate_normals(
            frame,
            method="organized",
            orientation="toward_view",
            viewpoint=np.array([0.0, 0.0, 0.0]),
            max_edge_m=0.1,
        )
        away = estimate_normals(
            frame,
            method="organized",
            orientation="away_view",
            viewpoint=np.array([0.0, 0.0, 0.0]),
            max_edge_m=0.1,
        )
        self.assertLess(float(np.nanmean(toward.normals[..., 2])), -0.99)
        self.assertGreater(float(np.nanmean(away.normals[..., 2])), 0.99)

    def test_closed_ring_mesh_normals_are_oriented_outward(self) -> None:
        ring_radius = 1.0
        torus = pv.ParametricTorus(ringradius=ring_radius, crosssectionradius=0.25)
        frame = make_frame(torus.points, torus.regular_faces)
        result = estimate_normals(frame, method="mesh", orientation="outward")
        points = frame.xyz.reshape(-1, 3)
        radial = points[:, :2].copy()
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        tube_centres = np.column_stack(
            (radial[:, 0] * ring_radius, radial[:, 1] * ring_radius, np.zeros(len(points)))
        )
        expected_outward = points - tube_centres
        dots = np.einsum("ij,ij->i", result.normals.reshape(-1, 3), expected_outward)
        self.assertGreater(float(np.mean(dots > 0.0)), 0.99)


if __name__ == "__main__":
    unittest.main()
