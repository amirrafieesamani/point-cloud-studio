from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import build_indexed_mesh, build_organized_mesh, reconstruct_unorganized_mesh
from .models import CaptureFrame


@dataclass(slots=True)
class ExportResult:
    folder: Path
    point_count: int
    triangle_count: int


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape N x 3")
    if normals is not None:
        normals = np.asarray(normals, dtype="<f4")
        if normals.shape != points.shape:
            raise ValueError("normals must have shape N x 3")
    normal_header = ""
    if normals is not None:
        normal_header = "property float nx\nproperty float ny\nproperty float nz\n"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{normal_header}"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields.extend([("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
    fields.extend([("r", "u1"), ("g", "u1"), ("b", "u1")])
    records = np.empty(len(points), dtype=fields)
    records["x"], records["y"], records["z"] = points.T
    if normals is not None:
        records["nx"], records["ny"], records["nz"] = normals.T
    records["r"], records["g"], records["b"] = colors.T
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)


def write_binary_pcd(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape N x 3")
    if normals is not None:
        normals = np.asarray(normals, dtype="<f4")
        if normals.shape != points.shape:
            raise ValueError("normals must have shape N x 3")
    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields.extend([("normal_x", "<f4"), ("normal_y", "<f4"), ("normal_z", "<f4")])
    fields.append(("rgb", "<u4"))
    records = np.empty(len(points), dtype=fields)
    records["x"], records["y"], records["z"] = points.T
    if normals is not None:
        records["normal_x"], records["normal_y"], records["normal_z"] = normals.T
    records["rgb"] = (
        colors[:, 0].astype(np.uint32) << 16
        | colors[:, 1].astype(np.uint32) << 8
        | colors[:, 2].astype(np.uint32)
    )
    normal_fields = " normal_x normal_y normal_z" if normals is not None else ""
    normal_layout = " 4 4 4" if normals is not None else ""
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS x y z{normal_fields} rgb\n"
        f"SIZE 4 4 4{normal_layout} 4\n"
        f"TYPE F F F{' F F F' if normals is not None else ''} U\n"
        f"COUNT 1 1 1{' 1 1 1' if normals is not None else ''} 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)


def write_binary_stl(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape M x 3")
    tri = vertices[triangles] if len(triangles) else np.empty((0, 3, 3), np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    normals[~good] = 0
    records = np.zeros(
        len(tri),
        dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")],
    )
    records["normal"] = normals
    records["vertices"] = tri
    with path.open("wb") as stream:
        stream.write(b"Multi-Camera 3D Crop Studio".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(tri)))
        records.tofile(stream)


def export_selection(
    output_root: Path,
    frame: CaptureFrame,
    selected_mask: np.ndarray,
    box_definition: dict[str, Any],
    max_edge_m: float,
    export_origin_m: np.ndarray | None = None,
    origin_mode: str = "camera",
    export_rotation: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    normal_metadata: dict[str, Any] | None = None,
) -> ExportResult:
    selected_mask = selected_mask.astype(bool, copy=False) & frame.valid_mask
    camera_points = frame.xyz[selected_mask]
    colors = frame.rgb[selected_mask]
    if len(camera_points) < 3:
        raise ValueError("The box contains fewer than three valid points")
    origin = (
        np.zeros(3, dtype=np.float64)
        if export_origin_m is None
        else np.asarray(export_origin_m, dtype=np.float64)
    )
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("export_origin_m must contain three finite coordinates")
    rotation = np.eye(3, dtype=np.float64) if export_rotation is None else np.asarray(
        export_rotation, dtype=np.float64
    )
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("export_rotation must be a finite 3 x 3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.0:
        raise ValueError("export_rotation must be a right-handed orthonormal matrix")
    points = ((camera_points.astype(np.float64) - origin) @ rotation).astype(np.float32)
    selected_normals: np.ndarray | None = None
    if normals is not None:
        source_normals = np.asarray(normals, dtype=np.float64)
        if source_normals.shape != frame.xyz.shape:
            raise ValueError("normals must have the same shape as frame.xyz")
        selected_normals = source_normals[selected_mask] @ rotation
        lengths = np.linalg.norm(selected_normals, axis=1)
        valid_normals = np.isfinite(selected_normals).all(axis=1) & (lengths > 1e-10)
        selected_normals[valid_normals] /= lengths[valid_normals, None]
        selected_normals[~valid_normals] = 0.0
        selected_normals = selected_normals.astype(np.float32)

    stamp = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")[:-3]
    root = Path(output_root).expanduser().resolve()
    folder = root / stamp
    sequence = 1
    while folder.exists():
        folder = root / f"{stamp}_{sequence:02d}"
        sequence += 1
    folder.mkdir(parents=True, exist_ok=False)

    if frame.source_faces is not None:
        camera_vertices, triangles = build_indexed_mesh(
            frame.xyz.reshape(-1, 3), frame.source_faces, selected_mask.reshape(-1)
        )
    elif frame.organized and frame.xyz.shape[0] > 1:
        camera_vertices, triangles = build_organized_mesh(frame.xyz, selected_mask, max_edge_m)
    else:
        camera_vertices, triangles = reconstruct_unorganized_mesh(camera_points)
    vertices = ((camera_vertices.astype(np.float64) - origin) @ rotation).astype(np.float32)
    write_binary_ply(folder / "selected_cloud.ply", points, colors, selected_normals)
    write_binary_pcd(folder / "selected_cloud.pcd", points, colors, selected_normals)
    # STL has no unit field; millimetres are the most interoperable convention in CAD/slicers.
    write_binary_stl(folder / "selected_mesh.stl", vertices * 1000.0, triangles)

    try:
        import cv2

        cv2.imwrite(str(folder / "rgb.png"), frame.rgb[:, :, ::-1])
        cv2.imwrite(str(folder / "depth_mm_16bit.png"), frame.depth_mm)
        preview = np.zeros_like(frame.rgb)
        preview[selected_mask] = frame.rgb[selected_mask]
        cv2.imwrite(str(folder / "selection_preview.png"), preview[:, :, ::-1])
    except ImportError:
        pass

    metadata: dict[str, Any] = {
        "format_version": 2,
        "point_cloud_units": "metres",
        "stl_units_convention": "millimetres",
        "coordinate_axes": frame.coordinate_system,
        "input": {
            "organized": frame.organized,
            "source_units_before_conversion": frame.source_units,
            "contained_triangle_mesh": frame.source_faces is not None,
        },
        "processing": frame.processing_metadata,
        "export_coordinate_origin": {
            "mode": origin_mode,
            "camera_space_m": origin.tolist(),
            "axes_in_source_coordinates": {
                "x": rotation[:, 0].tolist(),
                "y": rotation[:, 1].tolist(),
                "z": rotation[:, 2].tolist(),
            },
            "transform": "Each export coordinate is dot(source_point - source_space_origin, corresponding source-space axis)",
            "applied_to": ["selected_cloud.ply", "selected_cloud.pcd", "selected_mesh.stl"],
        },
        "normals": normal_metadata if selected_normals is not None else None,
        "source": frame.source,
        "camera_timestamp_ms": frame.timestamp_ms,
        "intrinsics": frame.intrinsics.as_dict(),
        "point_count": int(len(points)),
        "triangle_count": int(len(triangles)),
        "mesh_max_edge_m": float(max_edge_m),
        "bounding_box": box_definition,
        "note": "STL stores geometry only and has no colour or explicit unit field. Vertices were converted to millimetres. Organized RGB-D inputs use their pixel topology, imported meshes keep fully selected source triangles, and unorganized point clouds use approximate surface reconstruction. The result may be open at occlusions or crop borders.",
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ExportResult(folder, len(points), len(triangles))
