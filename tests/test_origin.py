from __future__ import annotations

import unittest

import numpy as np

from realsense_cropper.origin import calculate_origin_anchor


class OriginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array(
            [
                [-4.0, 2.0, 3.0],
                [5.0, -6.0, 7.0],
                [8.0, 9.0, -10.0],
                [1.0, 12.0, 13.0],
            ]
        )

    def test_camera_and_manual_modes(self) -> None:
        np.testing.assert_array_equal(calculate_origin_anchor(self.points, "camera"), [0, 0, 0])
        np.testing.assert_array_equal(
            calculate_origin_anchor(self.points, "manual", [0.1, 0.2, 0.3]),
            [0.1, 0.2, 0.3],
        )

    def test_selection_center_uses_bounding_box_center(self) -> None:
        expected = (self.points.min(axis=0) + self.points.max(axis=0)) / 2
        np.testing.assert_array_equal(calculate_origin_anchor(self.points, "center"), expected)

    def test_all_six_extrema_return_the_full_extreme_point(self) -> None:
        expected_indices = {
            "min_x": 0,
            "max_x": 2,
            "min_y": 1,
            "max_y": 3,
            "min_z": 2,
            "max_z": 3,
        }
        for mode, index in expected_indices.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(
                    calculate_origin_anchor(self.points, mode), self.points[index]
                )

    def test_invalid_mode_and_empty_automatic_input_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_origin_anchor(self.points, "not-a-mode")
        with self.assertRaises(ValueError):
            calculate_origin_anchor(np.empty((0, 3)), "max_z")


if __name__ == "__main__":
    unittest.main()
