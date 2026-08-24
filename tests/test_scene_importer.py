from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from realsense_cropper.exporter import write_binary_pcd, write_binary_ply, write_binary_stl
from realsense_cropper.scene_importer import attach_rgb_image, load_3d_scene


class SceneImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.points = np.array(
            [[0.0, 0.0, 0.5], [0.1, 0.0, 0.6], [0.0, 0.1, 0.7], [0.1, 0.1, 0.8]],
            dtype=np.float32,
        )
        self.colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 150, 100]], dtype=np.uint8
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_binary_ply_and_pcd_keep_points_and_rgb(self) -> None:
        ply = self.folder / "sample.ply"
        pcd = self.folder / "sample.pcd"
        write_binary_ply(ply, self.points, self.colors)
        write_binary_pcd(pcd, self.points, self.colors)

        for path in (ply, pcd):
            with self.subTest(extension=path.suffix):
                frame = load_3d_scene(path, unit_mode="metres")
                np.testing.assert_allclose(frame.xyz.reshape(-1, 3), self.points, atol=1e-6)
                np.testing.assert_array_equal(frame.rgb.reshape(-1, 3), self.colors)
                self.assertEqual(frame.valid_mask.sum(), 4)

    def test_photoneo_ptx_is_organized_and_converted_from_mm(self) -> None:
        ptx = self.folder / "photoneo.ptx"
        header = "\n".join(
            [
                "2", "2", "0 0 0", "1 0 0", "0 1 0", "0 0 1",
                "1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1",
            ]
        )
        rows = "\n".join(
            [
                "0 0 0 0 0 0 0",
                "100 0 500 1 255 0 0",
                "0 100 600 1 0 255 0",
                "100 100 700 1 0 0 255",
            ]
        )
        ptx.write_text(header + "\n" + rows + "\n", encoding="utf-8")

        frame = load_3d_scene(ptx, unit_mode="metres")
        self.assertTrue(frame.organized)
        self.assertEqual(frame.xyz.shape, (2, 2, 3))
        self.assertEqual(frame.valid_mask.sum(), 3)
        np.testing.assert_allclose(frame.xyz[1, 1], [0.1, 0.1, 0.7], atol=1e-6)
        np.testing.assert_array_equal(frame.rgb[1, 1], [0, 0, 255])
        self.assertEqual(frame.source_units, "millimetres")

    def test_xyz_csv_and_companion_rgb_image(self) -> None:
        csv_path = self.folder / "points.csv"
        np.savetxt(csv_path, np.column_stack((self.points, self.colors)), delimiter=",")
        frame = load_3d_scene(csv_path, unit_mode="metres")
        self.assertFalse(frame.organized)

        expected = np.array(
            [[[11, 22, 33], [44, 55, 66]], [[77, 88, 99], [111, 122, 133]]],
            dtype=np.uint8,
        )
        image_path = self.folder / "rgb.png"
        self.assertTrue(cv2.imwrite(str(image_path), expected[:, :, ::-1]))
        attach_rgb_image(frame, image_path)
        self.assertTrue(frame.organized)
        np.testing.assert_array_equal(frame.rgb, expected)

    def test_stl_import_preserves_triangle_topology_and_units(self) -> None:
        vertices_mm = np.array(
            [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=np.float32
        )
        triangles = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        stl = self.folder / "mesh.stl"
        write_binary_stl(stl, vertices_mm, triangles)
        frame = load_3d_scene(stl, unit_mode="millimetres")
        self.assertFalse(frame.organized)
        self.assertIsNotNone(frame.source_faces)
        self.assertGreaterEqual(len(frame.source_faces), 2)
        self.assertAlmostEqual(float(np.max(frame.xyz)), 0.1, places=5)

    def test_invalid_extension_is_rejected(self) -> None:
        path = self.folder / "data.bin"
        path.write_bytes(b"not a point cloud")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_3d_scene(path)


if __name__ == "__main__":
    unittest.main()
