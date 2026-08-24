from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CaptureFrame


NORMAL_METHODS = (
    ("Automatic", "auto"),
    ("Organized cross-product", "organized"),
    ("PCA neighbourhood", "pca"),
    ("Imported mesh topology", "mesh"),
)

NORMAL_ORIENTATIONS = (
    ("Outward from object", "outward"),
    ("Toward current 3D view", "toward_view"),
    ("Away from current 3D view", "away_view"),
    ("Consistent estimated direction", "consistent"),
    ("Flip estimated direction", "flip"),
)


@dataclass(slots=True)
class NormalResult:
    normals: np.ndarray
    valid_mask: np.ndarray
    method: str
    orientation: str


def _normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (lengths > 1e-10)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    output[valid] = (values[valid] / lengths[valid, None]).astype(np.float32)
    return output, valid


def _organized_normals(frame: CaptureFrame, max_edge_m: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(frame.xyz, dtype=np.float64)
    valid = frame.valid_mask
    if xyz.shape[0] < 3 or xyz.shape[1] < 3:
        return np.full_like(xyz, np.nan, dtype=np.float32), np.zeros(valid.shape, bool)
    horizontal = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    vertical = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    central_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
    )
    if max_edge_m > 0:
        central_valid &= (
            (np.linalg.norm(horizontal, axis=2) <= max_edge_m * 2.0)
            & (np.linalg.norm(vertical, axis=2) <= max_edge_m * 2.0)
        )
    centre_normals, normal_valid = _normalize(np.cross(horizontal, vertical))
    output = np.full(xyz.shape, np.nan, dtype=np.float32)
    output_valid = np.zeros(valid.shape, dtype=bool)
    accepted = central_valid & normal_valid
    output[1:-1, 1:-1][accepted] = centre_normals[accepted]
    output_valid[1:-1, 1:-1] = accepted
    return output, output_valid


def _mesh_normals(frame: CaptureFrame, auto_orient: bool) -> tuple[np.ndarray, np.ndarray]:
    if frame.source_faces is None or not len(frame.source_faces):
        raise ValueError("The current source has no imported triangle topology")
    import pyvista as pv

    points = frame.xyz.reshape(-1, 3)
    prefix = np.full((len(frame.source_faces), 1), 3, dtype=np.int64)
    mesh = pv.PolyData(points, np.hstack((prefix, frame.source_faces)).ravel())
    computed = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=auto_orient,
        non_manifold_traversal=True,
        inplace=False,
    )
    normals, valid = _normalize(np.asarray(computed.point_data["Normals"]))
    return normals.reshape(frame.xyz.shape), valid.reshape(frame.valid_mask.shape)


def _pca_normals(frame: CaptureFrame, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    import pyvista as pv
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersPoints import vtkPCANormalEstimation

    source_valid = frame.valid_mask
    points = frame.xyz[source_valid]
    output = np.full(frame.xyz.shape, np.nan, dtype=np.float32)
    output_valid = np.zeros(source_valid.shape, dtype=bool)
    if len(points) < 4:
        return output, output_valid
    cloud = pv.PolyData(points)
    estimator = vtkPCANormalEstimation()
    estimator.SetInputData(cloud)
    estimator.SetSampleSize(min(neighbors, max(3, len(points) - 1)))
    estimator.SetNormalOrientationToGraphTraversal()
    estimator.FlipNormalsOff()
    estimator.Update()
    array = estimator.GetOutput().GetPointData().GetNormals()
    if array is None:
        return output, output_valid
    normals, valid = _normalize(vtk_to_numpy(array))
    output[source_valid] = normals
    output_valid[source_valid] = valid
    return output, output_valid


def _orient_normals(
    frame: CaptureFrame,
    normals: np.ndarray,
    valid: np.ndarray,
    orientation: str,
    selected_mask: np.ndarray | None,
    viewpoint: np.ndarray | None,
    mesh_was_auto_oriented: bool,
) -> np.ndarray:
    result = normals.copy()
    active = valid.copy()
    if selected_mask is not None:
        active &= selected_mask
    if not active.any() or orientation == "consistent":
        return result
    if orientation == "flip":
        result[active] *= -1.0
        return result
    points = frame.xyz
    if orientation == "outward" and mesh_was_auto_oriented:
        return result
    if orientation == "outward":
        centre_points = points[active]
        centre = (np.min(centre_points, axis=0) + np.max(centre_points, axis=0)) * 0.5
        direction = points[active] - centre
    elif orientation in {"toward_view", "away_view"}:
        if viewpoint is None:
            raise ValueError("A 3D viewpoint is required for view-oriented normals")
        direction = np.asarray(viewpoint, dtype=np.float64) - points[active]
        if orientation == "away_view":
            direction *= -1.0
    else:
        raise ValueError(f"Unknown normal orientation: {orientation}")
    flip = np.einsum("ij,ij->i", result[active], direction) < 0
    active_indices = np.flatnonzero(active)
    flattened = result.reshape(-1, 3)
    flattened[active_indices[flip]] *= -1.0
    return result


def estimate_normals(
    frame: CaptureFrame,
    *,
    method: str = "auto",
    orientation: str = "outward",
    neighbors: int = 24,
    max_edge_m: float = 0.02,
    selected_mask: np.ndarray | None = None,
    viewpoint: np.ndarray | None = None,
) -> NormalResult:
    if method not in {key for _, key in NORMAL_METHODS}:
        raise ValueError(f"Unknown normal method: {method}")
    if orientation not in {key for _, key in NORMAL_ORIENTATIONS}:
        raise ValueError(f"Unknown normal orientation: {orientation}")
    if method == "auto":
        if frame.source_faces is not None and len(frame.source_faces):
            chosen = "mesh"
        elif frame.organized and frame.xyz.shape[0] > 2 and frame.xyz.shape[1] > 2:
            chosen = "organized"
        else:
            chosen = "pca"
    else:
        chosen = method
    mesh_auto_oriented = chosen == "mesh" and orientation == "outward"
    if chosen == "mesh":
        normals, valid = _mesh_normals(frame, auto_orient=mesh_auto_oriented)
    elif chosen == "organized":
        if not frame.organized:
            raise ValueError("Organized normal estimation requires an organized point cloud")
        normals, valid = _organized_normals(frame, max_edge_m)
    else:
        normals, valid = _pca_normals(frame, neighbors)
    normals = _orient_normals(
        frame,
        normals,
        valid,
        orientation,
        selected_mask,
        viewpoint,
        mesh_auto_oriented,
    )
    return NormalResult(normals, valid, chosen, orientation)
