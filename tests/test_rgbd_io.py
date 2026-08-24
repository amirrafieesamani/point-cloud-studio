from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from realsense_cropper.capture import create_demo_frame
from realsense_cropper.rgbd_io import load_rgbd_frame, save_rgbd_frame


class RgbdIoTests(unittest.TestCase):
    def test_round_trip_preserves_all_arrays_and_intrinsics(self) -> None:
        source = create_demo_frame(64, 48)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample_rgbd.npz"
            save_rgbd_frame(path, source)
            loaded = load_rgbd_frame(path)
        np.testing.assert_equal(loaded.xyz, source.xyz)
        np.testing.assert_array_equal(loaded.rgb, source.rgb)
        np.testing.assert_array_equal(loaded.depth_mm, source.depth_mm)
        self.assertEqual(loaded.intrinsics.as_dict(), source.intrinsics.as_dict())
        self.assertIn("sample_rgbd.npz", loaded.source)

    def test_invalid_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            np.savez_compressed(path, xyz=np.zeros((2, 2, 3), np.float32))
            with self.assertRaises(ValueError):
                load_rgbd_frame(path)


if __name__ == "__main__":
    unittest.main()
