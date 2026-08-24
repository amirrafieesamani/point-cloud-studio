from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, CaptureFrame


PHOTONEO_INSTALL_HELP = (
    "Photoneo native input requires PhoXi Control 1.17+ and its optional phoxi-api Python "
    "package. Set PHOXI_CONTROL_PATH to the PhoXi Control installation directory. "
    "Alternatively, open the PRAW/PMRAW file in PhoXi Control and export an organized "
    "PLY or PTX file, then import that file here."
)


def _import_api():
    try:
        import phoxi_api
    except ImportError as exc:
        raise RuntimeError(PHOTONEO_INSTALL_HELP) from exc
    return phoxi_api


def _texture_to_rgb(texture: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if texture is None:
        return np.full((height, width, 3), 190, dtype=np.uint8)
    values = np.asarray(texture)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.shape[:2] != shape:
        return np.full((height, width, 3), 190, dtype=np.uint8)
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    values = values[:, :, :3].astype(np.float64)
    maximum = float(np.nanmax(values, initial=0))
    if maximum > 255:
        values *= 255.0 / maximum
    elif maximum <= 1.0:
        values *= 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _convert_frame(frame: object, source: str) -> CaptureFrame:
    component = getattr(frame, "PointCloud", None)
    if component is None:
        raise RuntimeError("Photoneo frame contains no PointCloud component")
    points = np.asarray(component)
    if points.size == 0 or points.ndim < 2 or points.shape[-1] != 3:
        raise RuntimeError("Photoneo frame contains no PointCloud component")
    organized = points.ndim == 3
    if organized:
        height, width = points.shape[:2]
        xyz_mm = points
    else:
        xyz_mm = points.reshape(1, -1, 3)
        height, width = 1, xyz_mm.shape[1]
    texture = getattr(frame, "Texture", None)
    if texture is None:
        texture = getattr(frame, "ColorCameraImage", None)
    rgb = _texture_to_rgb(texture, (height, width))
    xyz = (xyz_mm.astype(np.float64) * 0.001).astype(np.float32)
    depth_component = getattr(frame, "DepthMap", None)
    if depth_component is not None and np.asarray(depth_component).shape[:2] == (height, width):
        depth_mm = np.clip(np.asarray(depth_component), 0, 65535).astype(np.uint16)
    else:
        depth_mm = np.clip(np.abs(xyz[:, :, 2]) * 1000.0, 0, 65535).astype(np.uint16)
    return CaptureFrame(
        xyz=xyz,
        rgb=rgb,
        depth_mm=depth_mm,
        intrinsics=CameraIntrinsics(width, height, 1.0, 1.0, 0.0, 0.0, "Photoneo", ()),
        timestamp_ms=time.time() * 1000.0,
        source=source,
        organized=organized,
        zero_is_invalid=True,
        coordinate_system="Photoneo source coordinate space; values converted from millimetres to metres",
        source_units="millimetres",
    )


def _capture_from_device(control: object, device_id: str) -> CaptureFrame:
    phoxi_api = _import_api()
    with control.connect(device_id, timeout_ms=15000) as device:
        frame_settings = device.frame_settings()
        if frame_settings is not None:
            for component in ("PointCloud", "DepthMap", "Texture", "ColorCameraImage"):
                setting = getattr(frame_settings, component, None)
                if setting is not None and (
                    not hasattr(setting, "can_set") or setting.can_set()
                ):
                    setting.value = True
        device.set_trigger_mode(phoxi_api.TriggerMode.SOFTWARE)
        if not device.is_acquiring():
            device.start_acquisition()
        frame_id = device.trigger_frame(wait_accept=True, wait_grabbing_end=True)
        frame = device.get_frame(frame_id, timeout_ms=30000)
        return _convert_frame(frame, f"Photoneo PhoXi API: {device_id}")


def capture_photoneo(device_id: str = "") -> CaptureFrame:
    phoxi_api = _import_api()
    control = phoxi_api.PhoXiControl()
    if not control.is_phoxicontrol_running():
        raise RuntimeError("PhoXi Control must be running before direct Photoneo capture. " + PHOTONEO_INSTALL_HELP)
    if not device_id.strip():
        devices = control.get_device_list(refresh=True)
        if len(devices) != 1:
            raise RuntimeError(
                "Enter the Photoneo device ID. PhoXi Control reported "
                f"{len(devices)} discovered devices: {devices}"
            )
        record = devices[0]
        for key in ("HWIdentification", "DeviceID", "ID", "Name"):
            if record.get(key):
                device_id = str(record[key])
                break
        if not device_id:
            raise RuntimeError(f"Could not determine the device ID from: {record}")
    return _capture_from_device(control, device_id.strip())


def load_photoneo_raw(path: Path) -> CaptureFrame:
    phoxi_api = _import_api()
    control = phoxi_api.PhoXiControl()
    if not control.is_phoxicontrol_running():
        raise RuntimeError("PhoXi Control must be running to attach a PRAW/PMRAW File Camera. " + PHOTONEO_INSTALL_HELP)
    camera_name = f"CropStudio-{path.stem}-{int(time.time())}"
    device_id = control.attach_file_camera(camera_name, str(path))
    try:
        frame = _capture_from_device(control, device_id)
        frame.source = f"Photoneo RAW File Camera: {path.name}"
        return frame
    finally:
        control.detach_file_camera(device_id)
