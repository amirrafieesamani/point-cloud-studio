from __future__ import annotations

import unittest

import numpy as np

from realsense_cropper.transforms import (
    euler_xyz_degrees_to_matrix,
    matrix_to_euler_xyz_degrees,
    orthonormalize_rotation,
)


class TransformTests(unittest.TestCase):
    def test_euler_round_trip_preserves_rotation(self) -> None:
        rotation = euler_xyz_degrees_to_matrix(np.array([20.0, -35.0, 70.0]))
        recovered = euler_xyz_degrees_to_matrix(matrix_to_euler_xyz_degrees(rotation))
        np.testing.assert_allclose(recovered, rotation, atol=1e-10)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_orthonormalization_removes_scale_and_keeps_right_handedness(self) -> None:
        source = np.array([[1.0, 0.02, 0.0], [0.0, 2.0, 0.03], [0.01, 0.0, -0.5]])
        rotation = orthonormalize_rotation(source)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertGreater(float(np.linalg.det(rotation)), 0.999999)


if __name__ == "__main__":
    unittest.main()
