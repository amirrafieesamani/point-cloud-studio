from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, CaptureFrame
from .rgbd_io import load_rgbd_frame


SUPPORTED_3D_EXTENSIONS = {
    ".npz",
    ".praw",
    ".pmraw",
    ".ply",
    ".pcd",
    ".ptx",
    ".xyz",
    ".txt",
    ".csv",
    ".stl",
    ".obj",
    ".off",
    ".vtk",
    ".vtp",
}


def _scale_to_metres(points: np.ndarray, unit_mode: str, extension: str) -> tuple[np.ndarray, str]:
    if unit_mode not in {"auto", "metres", "millimetres"}:
        raise ValueError(f"Unknown input unit mode: {unit_mode}")
    finite = points[np.isfinite(points).all(axis=1)]
    if unit_mode == "auto":
        if extension == ".ptx":
            unit_mode = "millimetres"
        elif len(finite):
            span = np.ptp(finite, axis=0)
            unit_mode = "millimetres" if float(np.max(span)) > 10.0 else "metres"
        else:
            unit_mode = "metres"
    scale = 0.001 if unit_mode == "millimetres" else 1.0
    return (points.astype(np.float64) * scale).astype(np.float32), unit_mode


def _normalise_colors(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1 and len(values) == count:
        if values.dtype.kind == "f":
            packed = values.astype("<f4", copy=False).view("<u4")
        else:
            packed = values.astype(np.uint32, copy=False)
        return np.column_stack(
            ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
        ).astype(np.uint8)
    if values.ndim == 2 and values.shape[0] == count and values.shape[1] >= 3:
        values = values[:, :3]
        if values.dtype.kind == "f" and np.nanmax(values, initial=0) <= 1.0:
            values = values * 255.0
        return np.clip(values, 0, 255).astype(np.uint8)
    raise ValueError("Color data does not match the point count")


def _default_colors(count: int) -> np.ndarray:
    return np.full((count, 3), (190, 198, 210), dtype=np.uint8)


def _make_frame(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
    organized: bool,
    source_units: str,
    faces: np.ndarray | None = None,
    zero_is_invalid: bool = False,
) -> CaptureFrame:
    if len(points) != width * height:
        raise ValueError("Point count does not match the declared width and height")
    xyz = np.ascontiguousarray(points.reshape(height, width, 3), dtype=np.float32)
    rgb = np.ascontiguousarray(colors.reshape(height, width, 3), dtype=np.uint8)
    z_mm = np.clip(np.nan_to_num(np.abs(xyz[:, :, 2]) * 1000.0), 0, 65535).astype(np.uint16)
    return CaptureFrame(
        xyz=xyz,
        rgb=rgb,
        depth_mm=z_mm,
        intrinsics=CameraIntrinsics(width, height, 1.0, 1.0, 0.0, 0.0, "imported", ()),
        timestamp_ms=path.stat().st_mtime * 1000.0,
        source=f"Imported {path.suffix.upper()}: {path.name}",
        organized=organized,
        zero_is_invalid=zero_is_invalid,
        source_faces=faces,
        coordinate_system=f"Axes as stored in {path.name}; coordinates converted to metres",
        source_units=source_units,
    )


def _load_ptx(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first = stream.readline().strip()
        second = stream.readline().strip()
    try:
        # PhoXi Control writes image height first and width second (e.g. 1544, 2064).
        height, width = int(first), int(second)
    except ValueError as exc:
        raise ValueError("Invalid PTX header: expected height and width on the first two lines") from exc
    data = np.loadtxt(path, skiprows=10)
    if data.ndim == 1:
        data = data[None, :]
    if len(data) != width * height or data.shape[1] < 3:
        raise ValueError(
            f"PTX declares {width} x {height} points but contains {len(data)} point rows"
        )
    points = data[:, :3]
    if data.shape[1] >= 7:
        colors = _normalise_colors(data[:, 4:7], len(points))
    elif data.shape[1] >= 4:
        intensity = data[:, 3]
        intensity = np.clip(intensity / max(float(np.max(intensity)), 1.0) * 255.0, 0, 255)
        colors = np.repeat(intensity[:, None], 3, axis=1).astype(np.uint8)
    else:
        colors = _default_colors(len(points))
    return points, colors, width, height


def _pcd_dtype(fields: list[str], sizes: list[int], types: list[str], counts: list[int]) -> np.dtype:
    descriptors: list[tuple] = []
    type_map = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
    }
    for field, size, kind, count in zip(fields, sizes, types, counts, strict=True):
        code = type_map.get((kind.upper(), size))
        if code is None:
            raise ValueError(f"Unsupported PCD field type: {kind}{size}")
        descriptors.append((field, code) if count == 1 else (field, code, (count,)))
    return np.dtype(descriptors)


def _load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    raw = path.read_bytes()
    match = re.search(br"(?m)^DATA\s+(ascii|binary)\s*$", raw)
    if match is None:
        raise ValueError("PCD header has no supported DATA ascii/binary declaration")
    data_mode = match.group(1).decode("ascii")
    data_start = raw.find(b"\n", match.start()) + 1
    header: dict[str, list[str]] = {}
    for line in raw[:data_start].decode("ascii", errors="strict").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]
    fields = header.get("FIELDS") or header.get("FIELD")
    if not fields or not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD must contain x, y, and z fields")
    sizes = [int(value) for value in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    width = int(header.get("WIDTH", [header.get("POINTS", ["0"])[0]])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    point_count = int(header.get("POINTS", [str(width * height)])[0])
    dtype = _pcd_dtype(fields, sizes, types, counts)
    if data_mode == "binary":
        records = np.frombuffer(raw, dtype=dtype, count=point_count, offset=data_start)
    else:
        matrix = np.loadtxt(path, comments="#", skiprows=len(raw[:data_start].splitlines()))
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        records = np.empty(len(matrix), dtype=dtype)
        column = 0
        for field, count in zip(fields, counts, strict=True):
            records[field] = matrix[:, column] if count == 1 else matrix[:, column:column + count]
            column += count
    points = np.column_stack((records["x"], records["y"], records["z"]))
    color_field = "rgb" if "rgb" in fields else "rgba" if "rgba" in fields else None
    colors = _normalise_colors(records[color_field], len(points)) if color_field else _default_colors(len(points))
    organized = height > 1 and width * height == len(points)
    if not organized:
        width, height = len(points), 1
    return points, colors, width, height, organized


def _load_xyz_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    data = np.genfromtxt(path, delimiter=delimiter, comments="#")
    data = data[None, :] if data.ndim == 1 else data
    data = data[np.isfinite(data[:, :3]).all(axis=1)]
    if data.shape[1] < 3:
        raise ValueError("Text point clouds require at least X, Y, and Z columns")
    points = data[:, :3]
    colors = _normalise_colors(data[:, 3:6], len(points)) if data.shape[1] >= 6 else _default_colors(len(points))
    return points, colors


def _load_with_pyvista(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import pyvista as pv

    data = pv.read(path)
    if isinstance(data, pv.MultiBlock):
        data = data.combine(merge_points=True)
    if isinstance(data, pv.PolyData):
        # Triangulating a vertex-only PLY removes all vertices, so preserve pure point clouds.
        surface = data.clean() if len(data.faces) == 0 else data.triangulate().clean()
    else:
        surface = data.extract_surface().triangulate().clean()
    points = np.asarray(surface.points, dtype=np.float32)
    colors = None
    for name in surface.point_data.keys():
        values = np.asarray(surface.point_data[name])
        if name.lower() in {"rgb", "rgba", "color", "colors", "texture"}:
            try:
                colors = _normalise_colors(values, len(points))
                break
            except ValueError:
                continue
    if colors is None:
        colors = _default_colors(len(points))
    faces = None
    if surface.n_cells and len(surface.faces):
        face_data = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
        faces = face_data[:, 1:4]
    return points, colors, faces


def load_3d_scene(path: Path, unit_mode: str = "auto") -> CaptureFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_3D_EXTENSIONS:
        raise ValueError(f"Unsupported 3D input format: {extension}")
    if extension == ".npz":
        return load_rgbd_frame(path)
    if extension in {".praw", ".pmraw"}:
        from .photoneo import load_photoneo_raw

        return load_photoneo_raw(path)
    if extension == ".ptx":
        points, colors, width, height = _load_ptx(path)
        points, chosen_units = _scale_to_metres(points, "millimetres", extension)
        return _make_frame(
            path, points, colors, width=width, height=height, organized=True,
            source_units=chosen_units, zero_is_invalid=True,
        )
    if extension == ".pcd":
        points, colors, width, height, organized = _load_pcd(path)
        points, chosen_units = _scale_to_metres(points, unit_mode, extension)
        return _make_frame(
            path, points, colors, width=width, height=height, organized=organized,
            source_units=chosen_units, zero_is_invalid=organized,
        )
    if extension in {".xyz", ".txt", ".csv"}:
        points, colors = _load_xyz_text(path)
        points, chosen_units = _scale_to_metres(points, unit_mode, extension)
        return _make_frame(
            path, points, colors, width=len(points), height=1, organized=False,
            source_units=chosen_units,
        )
    points, colors, faces = _load_with_pyvista(path)
    points, chosen_units = _scale_to_metres(points, unit_mode, extension)
    return _make_frame(
        path, points, colors, width=len(points), height=1, organized=False,
        source_units=chosen_units, faces=faces,
    )


def attach_rgb_image(frame: CaptureFrame, path: Path) -> None:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read RGB image: {path}")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    height, width = rgb.shape[:2]
    if frame.xyz.shape[:2] == (height, width):
        frame.rgb = rgb
        return
    if height * width == frame.xyz.shape[0] * frame.xyz.shape[1]:
        if frame.source_faces is None:
            frame.xyz = frame.xyz.reshape(height, width, 3)
            frame.depth_mm = frame.depth_mm.reshape(height, width)
            frame.rgb = rgb
            frame.organized = True
            frame.intrinsics.width = width
            frame.intrinsics.height = height
        else:
            # A mesh remains unorganized; the image is interpreted as one RGB triplet per vertex.
            frame.rgb = np.ascontiguousarray(rgb.reshape(1, -1, 3))
        return
    raise ValueError(
        f"Image resolution {width} x {height} does not match the point topology "
        f"({frame.xyz.shape[1]} x {frame.xyz.shape[0]} or {frame.xyz.size // 3} total points)."
    )
