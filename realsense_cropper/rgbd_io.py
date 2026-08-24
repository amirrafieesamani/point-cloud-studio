from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, CaptureFrame


def save_rgbd_frame(path: Path, frame: CaptureFrame) -> None:
    """Save an organized RGB-D frame in a portable, lossless NPZ container."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intr = frame.intrinsics
    np.savez_compressed(
        path,
        xyz=np.asarray(frame.xyz, dtype=np.float32),
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth_mm=np.asarray(frame.depth_mm, dtype=np.uint16),
        intr_width=np.int32(intr.width),
        intr_height=np.int32(intr.height),
        intr_fx=np.float64(intr.fx),
        intr_fy=np.float64(intr.fy),
        intr_ppx=np.float64(intr.ppx),
        intr_ppy=np.float64(intr.ppy),
        intr_model=np.str_(intr.model),
        intr_coeffs=np.asarray(intr.coeffs, dtype=np.float64),
        timestamp_ms=np.float64(frame.timestamp_ms),
        source=np.str_(frame.source),
    )


def load_rgbd_frame(path: Path) -> CaptureFrame:
    """Load and strictly validate an RGB-D NPZ created by :func:`save_rgbd_frame`."""
    path = Path(path)
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read RGB-D file: {path}") from exc

    required = {
        "xyz",
        "rgb",
        "depth_mm",
        "intr_width",
        "intr_height",
        "intr_fx",
        "intr_fy",
        "intr_ppx",
        "intr_ppy",
        "intr_model",
        "intr_coeffs",
        "timestamp_ms",
        "source",
    }
    try:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"RGB-D file is missing fields: {', '.join(sorted(missing))}")
        xyz = np.asarray(archive["xyz"], dtype=np.float32)
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        depth_mm = np.asarray(archive["depth_mm"], dtype=np.uint16)
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError("xyz must have shape H x W x 3")
        if rgb.shape != xyz.shape:
            raise ValueError("rgb must have the same H x W x 3 shape as xyz")
        if depth_mm.shape != xyz.shape[:2]:
            raise ValueError("depth_mm must have shape H x W")
        height, width = depth_mm.shape
        if int(archive["intr_width"]) != width or int(archive["intr_height"]) != height:
            raise ValueError("Stored intrinsics dimensions do not match the arrays")
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(archive["intr_fx"]),
            fy=float(archive["intr_fy"]),
            ppx=float(archive["intr_ppx"]),
            ppy=float(archive["intr_ppy"]),
            model=str(archive["intr_model"]),
            coeffs=tuple(float(value) for value in archive["intr_coeffs"]),
        )
        return CaptureFrame(
            xyz=np.ascontiguousarray(xyz),
            rgb=np.ascontiguousarray(rgb),
            depth_mm=np.ascontiguousarray(depth_mm),
            intrinsics=intrinsics,
            timestamp_ms=float(archive["timestamp_ms"]),
            source=f"Imported RGB-D: {path.name} ({str(archive['source'])})",
        )
    finally:
        archive.close()
