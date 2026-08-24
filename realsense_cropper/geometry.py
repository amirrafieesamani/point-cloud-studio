from __future__ import annotations

import numpy as np


def points_inside_planes(
    points: np.ndarray,
    origins: np.ndarray,
    normals: np.ndarray,
    tolerance_m: float = 1e-7,
) -> np.ndarray:
    """Return points inside a convex volume described by its outward-facing planes."""
    points = np.asarray(points, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    if origins.shape != normals.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins and normals must have matching K x 3 shapes")
    if len(origins) < 4:
        raise ValueError("at least four planes are required")

    # VTK box normals are outward, but orienting them here makes the file format and
    # future widget implementations deterministic too.
    centre = origins.mean(axis=0)
    normals = normals.copy()
    inward = np.einsum("ij,ij->i", normals, origins - centre) < 0
    normals[inward] *= -1.0
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("plane normal cannot be zero")
    normals /= lengths[:, None]

    result = np.empty(len(points), dtype=bool)
    chunk_size = 200_000
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        signed = np.einsum(
            "nkj,kj->nk", points[start:stop, None, :] - origins[None, :, :], normals
        )
        result[start:stop] = np.all(signed <= tolerance_m, axis=1)
    return result


def build_organized_mesh(
    xyz: np.ndarray,
    selected_mask: np.ndarray,
    max_edge_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate neighbouring selected RGB-D pixels without bridging depth jumps.

    Returns compact vertices (N, 3) and zero-based triangle indices (M, 3).
    The resulting mesh is the observed surface; it is not artificially made watertight.
    """
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("xyz must have shape H x W x 3")
    if selected_mask.shape != xyz.shape[:2]:
        raise ValueError("selected_mask must match the first two xyz dimensions")
    if max_edge_m <= 0:
        raise ValueError("max_edge_m must be positive")

    mask = selected_mask.astype(bool, copy=False)
    mask = mask & np.isfinite(xyz).all(axis=2)

    index_map = np.full(mask.shape, -1, dtype=np.int64)
    vertices = np.asarray(xyz[mask], dtype=np.float32)
    index_map[mask] = np.arange(vertices.shape[0], dtype=np.int64)

    if vertices.shape[0] < 3 or min(mask.shape) < 2:
        return vertices, np.empty((0, 3), dtype=np.int64)

    i00 = index_map[:-1, :-1]
    i01 = index_map[:-1, 1:]
    i10 = index_map[1:, :-1]
    i11 = index_map[1:, 1:]

    t1_valid = (i00 >= 0) & (i01 >= 0) & (i10 >= 0)
    t2_valid = (i01 >= 0) & (i11 >= 0) & (i10 >= 0)
    t1 = np.column_stack((i00[t1_valid], i01[t1_valid], i10[t1_valid]))
    t2 = np.column_stack((i01[t2_valid], i11[t2_valid], i10[t2_valid]))
    triangles = np.vstack((t1, t2)).astype(np.int64, copy=False)

    if triangles.size == 0:
        return vertices, triangles.reshape(0, 3)

    tri_points = vertices[triangles]
    e01 = np.linalg.norm(tri_points[:, 0] - tri_points[:, 1], axis=1)
    e12 = np.linalg.norm(tri_points[:, 1] - tri_points[:, 2], axis=1)
    e20 = np.linalg.norm(tri_points[:, 2] - tri_points[:, 0], axis=1)
    keep = np.maximum.reduce((e01, e12, e20)) <= max_edge_m
    return vertices, triangles[keep]


def build_indexed_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    selected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an indexed triangle mesh and compact its vertex indices."""
    points = np.asarray(points, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    selected_mask = np.asarray(selected_mask, dtype=bool).copy()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape M x 3")
    if selected_mask.shape != (len(points),):
        raise ValueError("selected_mask must contain one value per point")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index")
    selected_mask &= np.isfinite(points).all(axis=1)
    keep_faces = np.all(selected_mask[faces], axis=1) if len(faces) else np.zeros(0, bool)
    kept = faces[keep_faces]
    index_map = np.full(len(points), -1, dtype=np.int64)
    vertices = points[selected_mask]
    index_map[selected_mask] = np.arange(len(vertices))
    triangles = index_map[kept] if len(kept) else np.empty((0, 3), dtype=np.int64)
    return vertices, triangles


def reconstruct_unorganized_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create an approximate surface for an unorganized point cloud using VTK."""
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return points, np.empty((0, 3), dtype=np.int64)
    # Surface reconstruction becomes impractical for multi-million-point clouds. A stable,
    # deterministic sample is sufficient for STL generation while PLY/PCD keep every point.
    if len(points) > 150_000:
        indices = np.linspace(0, len(points) - 1, 150_000, dtype=np.int64)
        points = points[indices]
    import pyvista as pv

    cloud = pv.PolyData(points).clean(point_merging=True)
    try:
        surface = cloud.reconstruct_surface(nbr_sz=min(20, max(5, len(points) - 1)))
        surface = surface.triangulate().clean()
    except Exception:
        surface = cloud.delaunay_2d().triangulate().clean()
    if not surface.n_cells:
        return np.asarray(surface.points, dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    return np.asarray(surface.points, dtype=np.float32), faces[:, 1:4]


def finite_bounds(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("Point cloud has no finite points")
    low = np.percentile(points, 2.0, axis=0)
    high = np.percentile(points, 98.0, axis=0)
    span = np.maximum(high - low, 0.02)
    low -= span * 0.03
    high += span * 0.03
    return (low[0], high[0], low[1], high[1], low[2], high[2])
