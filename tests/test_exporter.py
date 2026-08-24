from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realsense_cropper.capture import create_demo_frame
from realsense_cropper.exporter import (
    export_selection,
    write_binary_pcd,
    write_binary_ply,
    write_binary_stl,
)
from realsense_cropper.geometry import build_organized_mesh
from realsense_cropper.scene_importer import load_3d_scene


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_point_cloud_writers(self) -> None:
        points = np.array([[0, 0, 1], [0.1, 0.2, 1.2]], dtype=np.float32)
        colors = np.array([[255, 0, 1], [2, 128, 254]], dtype=np.uint8)
        ply = self.tmp_path / "cloud.ply"
        pcd = self.tmp_path / "cloud.pcd"
        write_binary_ply(ply, points, colors)
        write_binary_pcd(pcd, points, colors)
        self.assertTrue(ply.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0"))
        self.assertIn(b"element vertex 2", ply.read_bytes()[:300])
        self.assertTrue(pcd.read_bytes().startswith(b"# .PCD v0.7"))
        self.assertIn(b"POINTS 2\nDATA binary", pcd.read_bytes()[:500])

    def test_point_cloud_writers_include_normals(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        colors = np.array([[10, 20, 30]], dtype=np.uint8)
        normals = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        ply = self.tmp_path / "normal_cloud.ply"
        pcd = self.tmp_path / "normal_cloud.pcd"
        write_binary_ply(ply, points, colors, normals)
        write_binary_pcd(pcd, points, colors, normals)
        self.assertIn(b"property float nx", ply.read_bytes()[:500])
        self.assertIn(b"FIELDS x y z normal_x normal_y normal_z rgb", pcd.read_bytes()[:500])

    def test_binary_stl_header_and_triangle_count(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]])
        path = self.tmp_path / "mesh.stl"
        write_binary_stl(path, vertices, triangles)
        data = path.read_bytes()
        self.assertEqual(len(data), 84 + 50)
        self.assertEqual(struct.unpack("<I", data[80:84])[0], 1)

    def test_complete_demo_export(self) -> None:
        frame = create_demo_frame(48, 36)
        selected = frame.valid_mask.copy()
        box = {"plane_origins": [[0, 0, 0]], "plane_normals": [[0, 0, 1]]}
        result = export_selection(self.tmp_path, frame, selected, box, 0.1)
        self.assertGreater(result.point_count, 10)
        self.assertGreater(result.triangle_count, 10)
        for name in (
            "selected_cloud.ply",
            "selected_cloud.pcd",
            "selected_mesh.stl",
            "metadata.json",
        ):
            self.assertTrue((result.folder / name).is_file())
        metadata = json.loads((result.folder / "metadata.json").read_text("utf-8"))
        self.assertEqual(metadata["point_cloud_units"], "metres")
        self.assertEqual(metadata["stl_units_convention"], "millimetres")
        self.assertEqual(metadata["point_count"], result.point_count)

    def test_export_origin_is_subtracted_from_ply_pcd_and_stl(self) -> None:
        frame = create_demo_frame(32, 24)
        selected = frame.valid_mask.copy()
        first_camera_point = frame.xyz[selected][0].astype(np.float64)
        box = {"plane_origins": [], "plane_normals": []}
        result = export_selection(
            self.tmp_path,
            frame,
            selected,
            box,
            0.2,
            export_origin_m=first_camera_point,
            origin_mode="manual",
        )

        ply_data = (result.folder / "selected_cloud.ply").read_bytes()
        ply_start = ply_data.index(b"end_header\n") + len(b"end_header\n")
        first_ply_xyz = np.frombuffer(ply_data, dtype="<f4", count=3, offset=ply_start)
        np.testing.assert_allclose(first_ply_xyz, [0, 0, 0], atol=1e-7)

        pcd_data = (result.folder / "selected_cloud.pcd").read_bytes()
        pcd_start = pcd_data.index(b"DATA binary\n") + len(b"DATA binary\n")
        first_pcd_xyz = np.frombuffer(pcd_data, dtype="<f4", count=3, offset=pcd_start)
        np.testing.assert_allclose(first_pcd_xyz, [0, 0, 0], atol=1e-7)

        metadata = json.loads((result.folder / "metadata.json").read_text("utf-8"))
        origin_metadata = metadata["export_coordinate_origin"]
        self.assertEqual(origin_metadata["mode"], "manual")
        np.testing.assert_allclose(origin_metadata["camera_space_m"], first_camera_point)

        stl_data = (result.folder / "selected_mesh.stl").read_bytes()
        first_stl_vertex = np.frombuffer(stl_data, dtype="<f4", count=3, offset=96)
        vertices, triangles = build_organized_mesh(frame.xyz, selected, 0.2)
        expected_first_vertex_mm = (vertices[triangles[0, 0]] - first_camera_point) * 1000.0
        np.testing.assert_allclose(first_stl_vertex, expected_first_vertex_mm, atol=1e-4)

    def test_export_rotation_is_applied_to_points_and_normals(self) -> None:
        frame = create_demo_frame(16, 12)
        selected = frame.valid_mask.copy()
        normals = np.zeros_like(frame.xyz)
        normals[..., 0] = 1.0
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = export_selection(
            self.tmp_path,
            frame,
            selected,
            {"plane_origins": [], "plane_normals": []},
            0.2,
            export_rotation=rotation,
            normals=normals,
            normal_metadata={"method": "test", "orientation": "test"},
        )
        data = (result.folder / "selected_cloud.ply").read_bytes()
        start = data.index(b"end_header\n") + len(b"end_header\n")
        first = np.frombuffer(data, dtype="<f4", count=6, offset=start)
        expected_point = frame.xyz[selected][0] @ rotation
        np.testing.assert_allclose(first[:3], expected_point, atol=1e-7)
        np.testing.assert_allclose(first[3:], [0.0, -1.0, 0.0], atol=1e-7)
        imported_ply = load_3d_scene(result.folder / "selected_cloud.ply", unit_mode="metres")
        imported_pcd = load_3d_scene(result.folder / "selected_cloud.pcd", unit_mode="metres")
        np.testing.assert_allclose(imported_ply.xyz.reshape(-1, 3)[0], expected_point, atol=1e-7)
        np.testing.assert_allclose(imported_pcd.xyz.reshape(-1, 3)[0], expected_point, atol=1e-7)

    def test_imported_mesh_export_keeps_selected_source_faces(self) -> None:
        source = self.tmp_path / "source.stl"
        vertices_mm = np.array(
            [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=np.float32
        )
        write_binary_stl(source, vertices_mm, np.array([[0, 1, 2], [0, 1, 3]]))
        frame = load_3d_scene(source, unit_mode="millimetres")
        result = export_selection(
            self.tmp_path, frame, frame.valid_mask, {"plane_origins": [], "plane_normals": []}, 0.1
        )
        self.assertEqual(result.triangle_count, len(frame.source_faces))
        metadata = json.loads((result.folder / "metadata.json").read_text("utf-8"))
        self.assertTrue(metadata["input"]["contained_triangle_mesh"])


if __name__ == "__main__":
    unittest.main()
