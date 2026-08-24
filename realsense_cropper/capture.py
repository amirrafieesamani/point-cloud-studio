from __future__ import annotations

import time

import numpy as np

from .models import CameraIntrinsics, CaptureFrame


class RealSenseSession:
    """Non-blocking RealSense capture with depth aligned to the color stream."""

    def __init__(self) -> None:
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self.device_name = ""
        self.serial = ""

    @property
    def running(self) -> bool:
        return self._pipeline is not None

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        if self.running:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install requirements.txt first."
            ) from exc

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            profile = pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(
                "Could not start the RealSense camera. Check USB 3, Intel RealSense "
                "drivers, and whether another program is using the camera."
            ) from exc

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self.device_name = device.get_info(rs.camera_info.name)
        self.serial = device.get_info(rs.camera_info.serial_number)
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._align = None
        if pipeline is not None:
            pipeline.stop()

    def poll(self) -> CaptureFrame | None:
        if self._pipeline is None or self._align is None:
            return None
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        bgr = np.asanyarray(color_frame.get_data()).copy()
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        depth_m = depth_raw.astype(np.float32) * self._depth_scale

        profile = depth_frame.profile.as_video_stream_profile()
        intr = profile.intrinsics
        height, width = depth_raw.shape
        rows, cols = np.indices((height, width), dtype=np.float32)
        z = depth_m
        x = (cols - float(intr.ppx)) / float(intr.fx) * z
        y = (rows - float(intr.ppy)) / float(intr.fy) * z
        xyz = np.dstack((x, y, z)).astype(np.float32, copy=False)
        xyz[z <= 0] = np.nan

        intrinsics = CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
            model=str(intr.model),
            coeffs=tuple(float(value) for value in intr.coeffs),
        )
        return CaptureFrame(
            xyz=xyz,
            rgb=rgb,
            depth_mm=np.rint(depth_m * 1000.0).astype(np.uint16),
            intrinsics=intrinsics,
            timestamp_ms=float(depth_frame.get_timestamp()),
            source=f"{self.device_name} ({self.serial})",
        )


def create_demo_frame(width: int = 320, height: int = 240) -> CaptureFrame:
    """Generate an organized synthetic scene for UI and export testing."""
    fx = fy = 310.0
    ppx = (width - 1) / 2.0
    ppy = (height - 1) / 2.0
    rows, cols = np.indices((height, width), dtype=np.float32)
    xn = (cols - ppx) / fx
    yn = (rows - ppy) / fy

    z = 0.72 + 0.035 * np.sin(xn * 19.0) * np.cos(yn * 16.0)
    object_mask = (xn / 0.36) ** 2 + (yn / 0.29) ** 2 < 1.0
    bump = 0.11 * np.exp(-((xn / 0.16) ** 2 + (yn / 0.13) ** 2))
    z = z - bump
    z[~object_mask] = np.nan
    x = xn * z
    y = yn * z
    xyz = np.dstack((x, y, z)).astype(np.float32)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(80 + 175 * (cols / max(width - 1, 1)), 0, 255)
    rgb[:, :, 1] = np.clip(190 - 100 * np.abs(yn), 0, 255)
    rgb[:, :, 2] = np.clip(220 - 140 * (rows / max(height - 1, 1)), 0, 255)
    rgb[~object_mask] = (22, 26, 33)
    depth_mm = np.nan_to_num(z * 1000.0, nan=0.0).astype(np.uint16)

    return CaptureFrame(
        xyz=xyz,
        rgb=rgb,
        depth_mm=depth_mm,
        intrinsics=CameraIntrinsics(width, height, fx, fy, ppx, ppy, "demo", ()),
        timestamp_ms=time.time() * 1000.0,
        source="Synthetic demo",
    )
