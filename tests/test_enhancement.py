from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from realsense_cropper.enhancement import EnhancementProcessor, EnhancementSettings
from realsense_cropper.exporter import export_selection
from realsense_cropper.models import CameraIntrinsics, CaptureFrame


def planar_frame(
    *, bump_mm: float = 10.0, hole: bool = False, outlier: bool = False
) -> CaptureFrame:
    height, width = 40, 50
    fx = fy = 100.0
    ppx, ppy = (width - 1) / 2, (height - 1) / 2
    rows, cols = np.indices((height, width), dtype=np.float32)
    z = np.full((height, width), 0.7, dtype=np.float32)
    z[16:24, 21:29] -= bump_mm / 1000.0
    x = (cols - ppx) / fx * z
    y = (rows - ppy) / fy * z
    xyz = np.dstack((x, y, z)).astype(np.float32)
    if hole:
        xyz[10, 10] = np.nan
        z[10, 10] = np.nan
    if outlier:
        xyz[5, 5] = (2.0, 2.0, 2.0)
        z[5, 5] = 2.0
    rgb = np.full((height, width, 3), (120, 140, 160), dtype=np.uint8)
    return CaptureFrame(
        xyz=xyz,
        rgb=rgb,
        depth_mm=np.nan_to_num(z * 1000.0, nan=0.0).astype(np.uint16),
        intrinsics=CameraIntrinsics(width, height, fx, fy, ppx, ppy, "test", ()),
        timestamp_ms=1.0,
        source="Planar test",
    )


def set_depth(frame: CaptureFrame, depth_m: np.ndarray) -> None:
    rows, cols = np.indices(depth_m.shape, dtype=np.float32)
    intrinsics = frame.intrinsics
    frame.xyz[:, :, 0] = (cols - intrinsics.ppx) / intrinsics.fx * depth_m
    frame.xyz[:, :, 1] = (rows - intrinsics.ppy) / intrinsics.fy * depth_m
    frame.xyz[:, :, 2] = depth_m
    frame.depth_mm = np.nan_to_num(depth_m * 1000.0, nan=0.0).astype(np.uint16)


class EnhancementTests(unittest.TestCase):
    def test_disabled_pipeline_is_an_independent_raw_copy(self) -> None:
        raw = planar_frame()
        original = raw.xyz.copy()
        result = EnhancementProcessor().process(raw, EnhancementSettings(enabled=False))
        np.testing.assert_array_equal(result.frame.xyz, raw.xyz)
        result.frame.xyz[0, 0] = 99
        np.testing.assert_array_equal(raw.xyz, original)

    def test_hole_fill_recovers_small_enclosed_gap(self) -> None:
        raw = planar_frame(hole=True)
        settings = EnhancementSettings(
            enabled=True, hole_fill_enabled=True, hole_radius=1, hole_iterations=1
        )
        result = EnhancementProcessor().process(raw, settings)
        self.assertTrue(np.isfinite(result.frame.xyz[10, 10]).all())
        self.assertAlmostEqual(float(result.frame.xyz[10, 10, 2]), 0.7, places=4)
        self.assertFalse(np.isfinite(raw.xyz[10, 10]).all())

    def test_plane_height_and_display_exaggeration_keep_true_scale(self) -> None:
        raw = planar_frame(bump_mm=10.0)
        settings = EnhancementSettings(
            enabled=True,
            plane_align_enabled=True,
            plane_threshold_mm=3.0,
            height_color_enabled=True,
            height_min_mm=-2.0,
            height_max_mm=15.0,
            vertical_exaggeration=3.0,
        )
        result = EnhancementProcessor().process(raw, settings)
        background_z = float(np.median(result.frame.xyz[:10, :, 2]))
        bump_z = float(np.median(result.frame.xyz[17:23, 22:28, 2]))
        display_bump_z = float(np.median(result.display_xyz[17:23, 22:28, 2]))
        self.assertAlmostEqual(background_z, 0.0, places=4)
        self.assertAlmostEqual(bump_z, 0.010, places=3)
        self.assertAlmostEqual(display_bump_z, 0.030, places=3)
        self.assertFalse(np.array_equal(result.display_rgb, raw.rgb))
        self.assertTrue(result.frame.processing_metadata["vertical_exaggeration_is_display_only"])

    def test_temporal_alpha_smooths_depth_change(self) -> None:
        processor = EnhancementProcessor()
        settings = EnhancementSettings(
            enabled=True, temporal_enabled=True, temporal_alpha=0.25
        )
        first = planar_frame(bump_mm=0)
        second = planar_frame(bump_mm=0)
        second.xyz[:, :, 2] += 0.04
        processor.process(first, settings)
        result = processor.process(second, settings)
        self.assertAlmostEqual(float(np.median(result.frame.xyz[:, :, 2])), 0.71, places=3)

    def test_statistical_filter_removes_isolated_point(self) -> None:
        raw = planar_frame(outlier=True)
        settings = EnhancementSettings(
            enabled=True,
            outlier_enabled=True,
            outlier_neighbors=12,
            outlier_std_ratio=1.0,
        )
        result = EnhancementProcessor().process(raw, settings)
        self.assertFalse(np.isfinite(result.frame.xyz[5, 5]).all())
        self.assertGreater(result.frame.valid_mask.sum(), 1800)

    def test_display_exaggeration_is_not_written_to_export_geometry(self) -> None:
        raw = planar_frame(bump_mm=10.0)
        settings = EnhancementSettings(
            enabled=True,
            plane_align_enabled=True,
            plane_threshold_mm=3.0,
            vertical_exaggeration=3.0,
        )
        enhanced = EnhancementProcessor().process(raw, settings)
        with tempfile.TemporaryDirectory() as temporary:
            exported = export_selection(
                Path(temporary), enhanced.frame, enhanced.frame.valid_mask,
                {"plane_origins": [], "plane_normals": []}, max_edge_m=0.1,
            )
            ply = (exported.folder / "selected_cloud.ply").read_bytes()
            start = ply.index(b"end_header\n") + len(b"end_header\n")
            records = np.frombuffer(
                ply,
                dtype=[
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ],
                offset=start,
            )
            self.assertAlmostEqual(float(np.max(records["z"])), 0.010, places=3)
            metadata = json.loads((exported.folder / "metadata.json").read_text("utf-8"))
            self.assertEqual(
                metadata["processing"]["enhancement_settings"]["vertical_exaggeration"],
                3.0,
            )
            self.assertTrue(
                metadata["processing"]["vertical_exaggeration_is_display_only"]
            )

    def test_empty_table_reference_removes_wave_and_detects_10mm_object(self) -> None:
        background = planar_frame(bump_mm=0)
        current = planar_frame(bump_mm=0)
        rows, cols = np.indices(background.depth_mm.shape, dtype=np.float32)
        measured_wave = 0.004 * np.sin(cols * 0.42) * np.cos(rows * 0.35)
        background_depth = 0.7 + measured_wave
        current_depth = background_depth.copy()
        current_depth[15:24, 20:30] -= 0.010
        set_depth(background, background_depth)
        set_depth(current, current_depth)
        current.rgb[15:24, 20:30] = (15, 15, 15)

        processor = EnhancementProcessor()
        processor.set_background_from_frame(background)
        settings = EnhancementSettings(
            enabled=True,
            background_enabled=True,
            plane_align_enabled=True,
            floor_snap_enabled=True,
            floor_snap_mm=2.5,
            object_detection_enabled=True,
            object_threshold_mm=4.0,
            object_min_pixels=12,
            rgb_assist_enabled=True,
            rgb_difference_threshold=25.0,
            height_color_enabled=True,
        )
        result = processor.process(current, settings)
        floor_mask = np.ones(current.depth_mm.shape, dtype=bool)
        floor_mask[15:24, 20:30] = False
        np.testing.assert_allclose(result.frame.xyz[:, :, 2][floor_mask], 0.0, atol=1e-6)
        self.assertAlmostEqual(
            float(np.median(result.frame.xyz[16:23, 21:29, 2])), 0.010, places=3
        )
        self.assertEqual(result.object_count, 1)
        self.assertTrue(result.object_mask[15:24, 20:30].any())
        self.assertTrue(result.rgb_candidate_mask[15:24, 20:30].any())
        self.assertLess(result.floor_noise_sigma_mm, 0.05)

    def test_rgb_assist_marks_black_object_when_depth_is_invalid(self) -> None:
        background = planar_frame(bump_mm=0)
        current = planar_frame(bump_mm=0)
        current.xyz[10:16, 10:18] = np.nan
        current.depth_mm[10:16, 10:18] = 0
        current.rgb[10:16, 10:18] = (0, 0, 0)
        processor = EnhancementProcessor()
        processor.set_background_from_frame(background)
        settings = EnhancementSettings(
            enabled=True,
            background_enabled=True,
            object_detection_enabled=True,
            object_threshold_mm=4.0,
            object_min_pixels=10,
            rgb_assist_enabled=True,
            rgb_difference_threshold=25.0,
        )
        result = processor.process(current, settings)
        self.assertFalse(result.object_mask[10:16, 10:18].any())
        self.assertTrue(result.rgb_candidate_mask[10:16, 10:18].all())

    def test_background_reference_round_trip(self) -> None:
        processor = EnhancementProcessor()
        processor.set_background_from_frame(planar_frame(bump_mm=0))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty_table.npz"
            processor.save_background(path)
            restored = EnhancementProcessor()
            restored.load_background(path)
            self.assertIsNotNone(restored.background_reference)
            np.testing.assert_allclose(
                restored.background_reference.frame.xyz,
                processor.background_reference.frame.xyz,
            )

    def test_multi_frame_background_capture_uses_depth_median(self) -> None:
        processor = EnhancementProcessor()
        processor.begin_background_capture(3)
        depths = (0.699, 0.701, 1.2)
        for index, depth in enumerate(depths):
            frame = planar_frame(bump_mm=0)
            set_depth(frame, np.full(frame.depth_mm.shape, depth, dtype=np.float32))
            current, target, complete = processor.add_background_frame(frame)
            self.assertEqual(current, index + 1)
            self.assertEqual(target, 3)
        self.assertTrue(complete)
        self.assertFalse(processor.background_capture_active)
        self.assertEqual(processor.background_reference.captured_frames, 3)
        self.assertAlmostEqual(
            float(np.median(processor.background_reference.frame.xyz[:, :, 2])),
            0.701,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
