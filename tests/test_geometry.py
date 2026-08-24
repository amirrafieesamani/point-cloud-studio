from __future__ import annotations

import unittest

import numpy as np

from realsense_cropper.geometry import (
    build_indexed_mesh,
    build_organized_mesh,
    finite_bounds,
    points_inside_planes,
)


def plane(width: int, height: int, spacing: float = 0.01) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.float32)
    return np.dstack((cols * spacing, rows * spacing, np.ones_like(cols)))


class GeometryTests(unittest.TestCase):
    def test_points_inside_axis_aligned_planes(self) -> None:
        origins = np.array(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            dtype=float,
        )
        normals = np.array(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            dtype=float,
        )
        points = np.array([[0, 0, 0], [0.9, -0.9, 0.9], [1.1, 0, 0]])
        actual = points_inside_planes(points, origins, normals)
        np.testing.assert_array_equal(actual, [True, True, False])

    def test_full_grid_produces_two_triangles_per_quad(self) -> None:
        xyz = plane(5, 4)
        vertices, triangles = build_organized_mesh(
            xyz, np.ones((4, 5), dtype=bool), max_edge_m=0.02
        )
        self.assertEqual(vertices.shape, (20, 3))
        self.assertEqual(triangles.shape, (2 * 3 * 4, 3))

    def test_selection_hole_does_not_create_triangles_through_hole(self) -> None:
        xyz = plane(3, 3)
        selected = np.ones((3, 3), dtype=bool)
        selected[1, 1] = False
        vertices, triangles = build_organized_mesh(xyz, selected, max_edge_m=0.02)
        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(triangles), 2)

    def test_depth_discontinuity_is_not_bridged(self) -> None:
        xyz = plane(3, 2)
        xyz[:, 2:, 2] = 1.5
        _, triangles = build_organized_mesh(
            xyz, np.ones((2, 3), dtype=bool), max_edge_m=0.05
        )
        self.assertEqual(len(triangles), 2)

    def test_invalid_shapes_and_edge_limit(self) -> None:
        with self.assertRaises(ValueError):
            build_organized_mesh(np.zeros((3, 3)), np.ones((3, 3), bool), 0.1)
        with self.assertRaises(ValueError):
            build_organized_mesh(plane(2, 2), np.ones((2, 2), bool), 0)

    def test_finite_bounds_rejects_empty_and_orders_axes(self) -> None:
        with self.assertRaises(ValueError):
            finite_bounds(np.empty((0, 3)))
        bounds = finite_bounds(np.array([[0, 0, 1], [1, 2, 3]], dtype=float))
        self.assertLess(bounds[0], bounds[1])
        self.assertLess(bounds[2], bounds[3])
        self.assertLess(bounds[4], bounds[5])

    def test_indexed_mesh_crop_compacts_selected_vertices(self) -> None:
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 2, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]])
        vertices, triangles = build_indexed_mesh(
            points, faces, np.array([True, True, True, True, False])
        )
        self.assertEqual(vertices.shape, (4, 3))
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3]])


if __name__ == "__main__":
    unittest.main()
