from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from realsense_cropper.photoneo import _convert_frame


class PhotoneoConversionTests(unittest.TestCase):
    def test_organized_frame_converts_mm_and_texture(self) -> None:
        frame = SimpleNamespace(
            PointCloud=np.array(
                [[[0, 0, 0], [100, 0, 500]], [[0, 100, 600], [100, 100, 700]]],
                dtype=np.float32,
            ),
            Texture=np.array([[0, 64], [128, 255]], dtype=np.uint8),
            DepthMap=np.array([[0, 500], [600, 700]], dtype=np.float32),
        )
        converted = _convert_frame(frame, "Mock Photoneo")
        self.assertTrue(converted.organized)
        self.assertEqual(converted.valid_mask.sum(), 3)
        np.testing.assert_allclose(converted.xyz[1, 1], [0.1, 0.1, 0.7], atol=1e-7)
        np.testing.assert_array_equal(converted.rgb[0, 1], [64, 64, 64])
        self.assertEqual(converted.depth_mm[1, 1], 700)

    def test_missing_point_cloud_has_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no PointCloud"):
            _convert_frame(SimpleNamespace(PointCloud=None), "Mock")


if __name__ == "__main__":
    unittest.main()
