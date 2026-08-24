from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, CaptureFrame


@dataclass(slots=True)
class EnhancementSettings:
    enabled: bool = False
    multi_frame_enabled: bool = False
    multi_frame_count: int = 12
    temporal_enabled: bool = False
    temporal_alpha: float = 0.35
    hole_fill_enabled: bool = False
    hole_radius: int = 1
    hole_iterations: int = 1
    spatial_enabled: bool = False
    spatial_diameter: int = 5
    spatial_sigma_mm: float = 8.0
    spatial_strength: float = 0.65
    outlier_enabled: bool = False
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 1.5
    plane_align_enabled: bool = False
    plane_threshold_mm: float = 4.0
    height_color_enabled: bool = False
    height_min_mm: float = -2.0
    height_max_mm: float = 20.0
    height_color_blend: float = 0.85
    vertical_exaggeration: float = 1.0
    background_enabled: bool = False
    floor_snap_enabled: bool = False
    floor_snap_mm: float = 2.5
    object_detection_enabled: bool = False
    object_threshold_mm: float = 4.0
    object_min_pixels: int = 12
    rgb_assist_enabled: bool = False
    rgb_difference_threshold: float = 35.0

    def as_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)


@dataclass(slots=True)
class EnhancementResult:
    frame: CaptureFrame
    display_xyz: np.ndarray
    display_rgb: np.ndarray
    height_mm: np.ndarray | None
    display_transform: np.ndarray
    inverse_display_transform: np.ndarray
    plane_origin_m: np.ndarray | None
    plane_normal: np.ndarray | None
    object_mask: np.ndarray | None = None
    rgb_candidate_mask: np.ndarray | None = None
    object_count: int = 0
    floor_noise_sigma_mm: float | None = None
    background_active: bool = False


@dataclass(slots=True)
class BackgroundReference:
    frame: CaptureFrame
    captured_frames: int


def _component_filter(mask: np.ndarray, minimum_pixels: int) -> tuple[np.ndarray, int]:
    import cv2

    values = np.asarray(mask, dtype=np.uint8)
    if not values.any():
        return np.zeros_like(mask, dtype=bool), 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    values = cv2.morphologyEx(values, cv2.MORPH_CLOSE, kernel)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(values, connectivity=8)
    kept = np.zeros_like(values, dtype=bool)
    object_count = 0
    for label in range(1, count):
        if int(statistics[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
            kept |= labels == label
            object_count += 1
    return kept, object_count


def _clone_intrinsics(value: CameraIntrinsics) -> CameraIntrinsics:
    return CameraIntrinsics(
        value.width, value.height, value.fx, value.fy, value.ppx, value.ppy,
        value.model, tuple(value.coeffs),
    )


def clone_frame(frame: CaptureFrame) -> CaptureFrame:
    return CaptureFrame(
        xyz=np.ascontiguousarray(frame.xyz.copy()),
        rgb=np.ascontiguousarray(frame.rgb.copy()),
        depth_mm=np.ascontiguousarray(frame.depth_mm.copy()),
        intrinsics=_clone_intrinsics(frame.intrinsics),
        timestamp_ms=frame.timestamp_ms,
        source=frame.source,
        organized=frame.organized,
        zero_is_invalid=frame.zero_is_invalid,
        source_faces=None if frame.source_faces is None else frame.source_faces.copy(),
        coordinate_system=frame.coordinate_system,
        source_units=frame.source_units,
        processing_metadata=None if frame.processing_metadata is None else dict(frame.processing_metadata),
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    result = values @ transform[:3, :3].T + transform[:3, 3]
    return result.astype(np.float32)


def _fit_reference_plane(
    points: np.ndarray, threshold_m: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 20:
        raise ValueError("At least 20 valid points are required to estimate a reference plane")
    if len(values) > 50_000:
        indices = np.linspace(0, len(values) - 1, 50_000, dtype=np.int64)
        sample = values[indices]
    else:
        sample = values
    working = sample
    for _ in range(5):
        origin = np.median(working, axis=0)
        _, _, vectors = np.linalg.svd(working - origin, full_matrices=False)
        normal = vectors[-1]
        residual = np.abs((sample - origin) @ normal)
        keep = residual <= threshold_m
        if keep.sum() < max(20, int(len(sample) * 0.25)):
            cutoff = np.percentile(residual, 65.0)
            keep = residual <= cutoff
        updated = sample[keep]
        if len(updated) == len(working):
            working = updated
            break
        working = updated
    origin = np.mean(working, axis=0)
    _, _, vectors = np.linalg.svd(working - origin, full_matrices=False)
    normal = vectors[-1]
    # Camera-space +Z points away from the camera; orient the height normal toward it.
    if float(np.dot(normal, (0.0, 0.0, -1.0))) < 0:
        normal = -normal
    return origin.astype(np.float64), normal.astype(np.float64)


def _plane_alignment_transform(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    x_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis -= normal * float(np.dot(x_axis, normal))
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        x_axis -= normal * float(np.dot(x_axis, normal))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.vstack((x_axis, y_axis, normal))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ origin)
    return transform


def _vertical_scale_transform(
    origin: np.ndarray, direction: np.ndarray, factor: float
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    outer = np.outer(direction, direction)
    transform[:3, :3] += (factor - 1.0) * outer
    transform[:3, 3] = -(factor - 1.0) * (outer @ origin)
    return transform


def _replace_depth(points: np.ndarray, new_z: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    result = points.copy()
    old_z = points[:, :, 2]
    old_valid = np.isfinite(points).all(axis=2) & (np.abs(old_z) > 1e-12)
    new_valid = np.isfinite(new_z) & (np.abs(new_z) > 1e-12)
    common = old_valid & new_valid
    ratio = np.ones_like(new_z, dtype=np.float32)
    ratio[common] = new_z[common] / old_z[common]
    result[:, :, 0][common] *= ratio[common]
    result[:, :, 1][common] *= ratio[common]
    result[:, :, 2][common] = new_z[common]
    added = new_valid & ~old_valid
    if added.any() and intrinsics.fx > 1.0 and intrinsics.fy > 1.0:
        rows, cols = np.indices(new_z.shape, dtype=np.float32)
        result[:, :, 0][added] = (
            (cols[added] - intrinsics.ppx) / intrinsics.fx * new_z[added]
        )
        result[:, :, 1][added] = (
            (rows[added] - intrinsics.ppy) / intrinsics.fy * new_z[added]
        )
        result[:, :, 2][added] = new_z[added]
    result[~new_valid] = np.nan
    return result


def _fill_holes(points: np.ndarray, radius: int, iterations: int) -> np.ndarray:
    import cv2

    result = points.copy()
    kernel_size = radius * 2 + 1
    area = kernel_size * kernel_size
    for _ in range(iterations):
        valid = np.isfinite(result).all(axis=2)
        count = cv2.boxFilter(
            valid.astype(np.float32), -1, (kernel_size, kernel_size), normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        fill = ~valid & (count >= max(4.0, area * 0.72))
        if not fill.any():
            break
        safe_count = np.maximum(count, 1.0)
        for axis in range(3):
            values = np.where(valid, result[:, :, axis], 0.0).astype(np.float32)
            summed = cv2.boxFilter(
                values, -1, (kernel_size, kernel_size), normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            result[:, :, axis][fill] = summed[fill] / safe_count[fill]
    return result


def _spatial_smooth(
    points: np.ndarray, diameter: int, sigma_mm: float, strength: float
) -> np.ndarray:
    import cv2

    valid = np.isfinite(points).all(axis=2)
    if not valid.any():
        return points
    z = np.where(valid, points[:, :, 2], 0.0).astype(np.float32)
    missing = (~valid).astype(np.uint8)
    support = cv2.inpaint(z, missing, max(1, diameter // 2), cv2.INPAINT_NS)
    filtered = cv2.bilateralFilter(
        support, diameter, sigmaColor=max(sigma_mm / 1000.0, 1e-6),
        sigmaSpace=max(diameter / 2.0, 1.0),
    )
    blended = z.copy()
    blended[valid] = z[valid] * (1.0 - strength) + filtered[valid] * strength
    blended[~valid] = np.nan
    return _replace_depth(points, blended, CameraIntrinsics(0, 0, 0, 0, 0, 0))


def _statistical_inlier_mask(
    points: np.ndarray, neighbors: int, std_ratio: float
) -> np.ndarray:
    import pyvista as pv
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersPoints import vtkStatisticalOutlierRemoval

    cloud = pv.PolyData(np.asarray(points, dtype=np.float32))
    cloud["OriginalPointId"] = np.arange(len(points), dtype=np.int64)
    algorithm = vtkStatisticalOutlierRemoval()
    algorithm.SetInputData(cloud)
    algorithm.SetSampleSize(min(neighbors, max(2, len(points) - 1)))
    algorithm.SetStandardDeviationFactor(std_ratio)
    algorithm.GenerateOutliersOff()
    algorithm.Update()
    array = algorithm.GetOutput().GetPointData().GetArray("OriginalPointId")
    if array is None:
        return np.ones(len(points), dtype=bool)
    kept_ids = vtk_to_numpy(array).astype(np.int64, copy=False)
    mask = np.zeros(len(points), dtype=bool)
    mask[kept_ids] = True
    return mask


def _height_colors(
    rgb: np.ndarray,
    height_mm: np.ndarray,
    valid: np.ndarray,
    minimum_mm: float,
    maximum_mm: float,
    blend: float,
) -> np.ndarray:
    import cv2

    safe_height = np.nan_to_num(height_mm, nan=minimum_mm)
    scaled = np.clip(
        (safe_height - minimum_mm) / max(maximum_mm - minimum_mm, 1e-6) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    mapped = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1]
    result = rgb.astype(np.float32)
    result[valid] = result[valid] * (1.0 - blend) + mapped[valid] * blend
    return np.clip(result, 0, 255).astype(np.uint8)


class EnhancementProcessor:
    def __init__(self) -> None:
        self._depth_history: list[np.ndarray] = []
        self._temporal_depth: np.ndarray | None = None
        self._history_shape: tuple[int, int] | None = None
        self._background_reference: BackgroundReference | None = None
        self._background_target = 0
        self._background_depth_samples: list[np.ndarray] = []
        self._background_rgb_sum: np.ndarray | None = None
        self._background_template: CaptureFrame | None = None

    def reset(self) -> None:
        self._depth_history.clear()
        self._temporal_depth = None
        self._history_shape = None

    def reset_temporal(self) -> None:
        self._temporal_depth = None

    @property
    def background_reference(self) -> BackgroundReference | None:
        return self._background_reference

    @property
    def background_capture_active(self) -> bool:
        return self._background_target > 0

    def begin_background_capture(self, target_frames: int) -> None:
        if target_frames < 1:
            raise ValueError("Background capture requires at least one frame")
        self._background_target = int(target_frames)
        self._background_depth_samples = []
        self._background_rgb_sum = None
        self._background_template = None

    def cancel_background_capture(self) -> None:
        self._background_target = 0
        self._background_depth_samples = []
        self._background_rgb_sum = None
        self._background_template = None

    def add_background_frame(self, frame: CaptureFrame) -> tuple[int, int, bool]:
        if not self.background_capture_active:
            raise RuntimeError("Background capture has not been started")
        if not frame.organized or frame.xyz.shape[0] <= 1:
            raise ValueError("Background calibration requires an organized point cloud")
        if self._background_template is not None and frame.xyz.shape != self._background_template.xyz.shape:
            raise ValueError("Background frame resolution changed during capture")
        if self._background_template is None:
            self._background_template = clone_frame(frame)
            self._background_rgb_sum = np.zeros_like(frame.rgb, dtype=np.float64)
        depth = np.where(frame.valid_mask, frame.xyz[:, :, 2], np.nan).astype(np.float32)
        self._background_depth_samples.append(depth)
        assert self._background_rgb_sum is not None
        self._background_rgb_sum += frame.rgb
        current = len(self._background_depth_samples)
        complete = current >= self._background_target
        if complete:
            with np.errstate(invalid="ignore"):
                median_depth = np.nanmedian(np.stack(self._background_depth_samples), axis=0)
            template = self._background_template
            assert template is not None
            template.xyz = _replace_depth(template.xyz, median_depth, template.intrinsics)
            template.depth_mm = np.clip(
                np.nan_to_num(np.abs(median_depth) * 1000.0), 0, 65535
            ).astype(np.uint16)
            template.rgb = np.clip(
                self._background_rgb_sum / current, 0, 255
            ).astype(np.uint8)
            template.source = f"Empty-table background ({current} frames)"
            self._background_reference = BackgroundReference(template, current)
            target = self._background_target
            self.cancel_background_capture()
            return current, target, True
        return current, self._background_target, False

    def set_background_from_frame(self, frame: CaptureFrame) -> None:
        if not frame.organized or frame.xyz.shape[0] <= 1:
            raise ValueError("Background calibration requires an organized point cloud")
        reference = clone_frame(frame)
        reference.source = "Empty-table background (single frame)"
        self._background_reference = BackgroundReference(reference, 1)

    def clear_background(self) -> None:
        self._background_reference = None
        self.cancel_background_capture()

    def save_background(self, path: Path) -> None:
        if self._background_reference is None:
            raise ValueError("No empty-table background has been captured")
        from .rgbd_io import save_rgbd_frame

        save_rgbd_frame(path, self._background_reference.frame)

    def load_background(self, path: Path) -> None:
        from .rgbd_io import load_rgbd_frame

        frame = load_rgbd_frame(path)
        if not frame.organized or frame.xyz.shape[0] <= 1:
            raise ValueError("The background file must contain organized RGB-D data")
        frame.source = f"Loaded empty-table background: {Path(path).name}"
        self._background_reference = BackgroundReference(frame, 1)

    def process(
        self,
        raw_frame: CaptureFrame,
        settings: EnhancementSettings,
        *,
        update_history: bool = True,
    ) -> EnhancementResult:
        frame = clone_frame(raw_frame)
        identity = np.eye(4, dtype=np.float64)
        if not settings.enabled:
            return EnhancementResult(
                frame, frame.xyz.copy(), frame.rgb.copy(), None, identity, identity,
                None, None,
            )

        points = frame.xyz.copy()
        organized = frame.organized and frame.xyz.shape[0] > 1
        background_points: np.ndarray | None = None
        background_rgb: np.ndarray | None = None
        background_active = False
        if settings.background_enabled:
            if self._background_reference is None:
                raise ValueError(
                    "Background correction is enabled, but no empty-table reference exists. "
                    "Capture or load a background first."
                )
            reference_frame = self._background_reference.frame
            if reference_frame.xyz.shape != frame.xyz.shape:
                raise ValueError(
                    "Background resolution does not match the current frame. Capture a new "
                    "empty-table reference without changing the camera resolution."
                )
            background_points = reference_frame.xyz.copy()
            background_rgb = reference_frame.rgb.copy()
            background_active = True
        if organized:
            raw_z = np.where(frame.valid_mask, points[:, :, 2], np.nan).astype(np.float32)
            if self._history_shape != raw_z.shape:
                self.reset()
                self._history_shape = raw_z.shape
            if update_history:
                self._depth_history.append(raw_z.copy())
            self._depth_history = self._depth_history[-max(2, settings.multi_frame_count):]
            if settings.multi_frame_enabled and self._depth_history:
                with np.errstate(invalid="ignore"):
                    averaged = np.nanmedian(np.stack(self._depth_history), axis=0)
                points = _replace_depth(points, averaged, frame.intrinsics)
            if settings.hole_fill_enabled:
                points = _fill_holes(points, settings.hole_radius, settings.hole_iterations)
            if settings.spatial_enabled:
                points = _spatial_smooth(
                    points, settings.spatial_diameter, settings.spatial_sigma_mm,
                    settings.spatial_strength,
                )
            if settings.temporal_enabled:
                current = points[:, :, 2]
                current_valid = np.isfinite(current)
                if self._temporal_depth is None or self._temporal_depth.shape != current.shape:
                    self._temporal_depth = current.copy()
                else:
                    previous_valid = np.isfinite(self._temporal_depth)
                    common = current_valid & previous_valid
                    alpha = settings.temporal_alpha
                    self._temporal_depth[common] = (
                        alpha * current[common] + (1.0 - alpha) * self._temporal_depth[common]
                    )
                    self._temporal_depth[current_valid & ~previous_valid] = current[
                        current_valid & ~previous_valid
                    ]
                    self._temporal_depth[~current_valid] = np.nan
                points = _replace_depth(points, self._temporal_depth, frame.intrinsics)
            if background_points is not None:
                if settings.hole_fill_enabled:
                    background_points = _fill_holes(
                        background_points, settings.hole_radius, settings.hole_iterations
                    )
                if settings.spatial_enabled:
                    background_points = _spatial_smooth(
                        background_points,
                        settings.spatial_diameter,
                        settings.spatial_sigma_mm,
                        settings.spatial_strength,
                    )

        valid = np.isfinite(points).all(axis=2)
        if frame.zero_is_invalid:
            valid &= np.any(np.abs(points) > 1e-12, axis=2)
        if settings.outlier_enabled and valid.sum() >= 20:
            inliers = _statistical_inlier_mask(
                points[valid], settings.outlier_neighbors, settings.outlier_std_ratio
            )
            updated_valid = np.zeros_like(valid)
            updated_valid[valid] = inliers
            points[~updated_valid] = np.nan
            valid = updated_valid

        plane_origin = None
        plane_normal = None
        height_mm = None
        object_mask = None
        rgb_candidate_mask = None
        object_count = 0
        rgb_candidate_count = 0
        floor_noise_sigma_mm = None
        needs_plane = (
            settings.plane_align_enabled
            or settings.height_color_enabled
            or settings.floor_snap_enabled
            or settings.object_detection_enabled
            or background_active
            or abs(settings.vertical_exaggeration - 1.0) > 1e-9
        )
        background_valid = (
            np.isfinite(background_points).all(axis=2)
            if background_points is not None
            else np.zeros_like(valid)
        )
        plane_source = (
            background_points[background_valid]
            if background_points is not None and background_valid.sum() >= 20
            else points[valid]
        )
        if needs_plane and len(plane_source) >= 20:
            plane_origin, plane_normal = _fit_reference_plane(
                plane_source, settings.plane_threshold_mm / 1000.0
            )
            signed_height = np.full(valid.shape, np.nan, dtype=np.float32)
            signed_height[valid] = (
                (points[valid].astype(np.float64) - plane_origin) @ plane_normal * 1000.0
            ).astype(np.float32)
            height_mm = signed_height

        align_geometry = (
            settings.plane_align_enabled
            or settings.floor_snap_enabled
            or background_active
        )
        if align_geometry and plane_origin is not None and plane_normal is not None:
            alignment = _plane_alignment_transform(plane_origin, plane_normal)
            points[valid] = transform_points(points[valid], alignment)
            if background_points is not None:
                background_points[background_valid] = transform_points(
                    background_points[background_valid], alignment
                )
            plane_origin = np.zeros(3, dtype=np.float64)
            plane_normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
            height_mm = np.full(valid.shape, np.nan, dtype=np.float32)
            height_mm[valid] = points[:, :, 2][valid] * 1000.0
            frame.coordinate_system = (
                "Reference-plane coordinates in metres: +Z is height toward the camera"
            )

        if background_active and background_points is not None and height_mm is not None:
            common = valid & background_valid
            background_height_mm = background_points[:, :, 2] * 1000.0
            height_mm[common] = (
                points[:, :, 2][common] - background_points[:, :, 2][common]
            ) * 1000.0
            # Express Z as height relative to the locally measured empty table. This cancels
            # repeatable stereo corrugation while retaining X/Y and true object height.
            points[:, :, 2][common] = height_mm[common] / 1000.0
            residual = height_mm[common]
            if len(residual):
                center = float(np.median(residual))
                floor_noise_sigma_mm = float(
                    1.4826 * np.median(np.abs(residual - center))
                )

        if settings.floor_snap_enabled and height_mm is not None:
            floor = valid & np.isfinite(height_mm) & (
                np.abs(height_mm) <= settings.floor_snap_mm
            )
            points[:, :, 2][floor] = 0.0
            height_mm[floor] = 0.0

        if settings.object_detection_enabled and height_mm is not None:
            candidates = valid & np.isfinite(height_mm) & (
                height_mm >= settings.object_threshold_mm
            )
            if organized:
                object_mask, object_count = _component_filter(
                    candidates, settings.object_min_pixels
                )
            else:
                object_mask = candidates
                object_count = int(candidates.any())

        if (
            settings.rgb_assist_enabled
            and background_rgb is not None
            and background_rgb.shape == frame.rgb.shape
        ):
            difference = np.mean(
                np.abs(frame.rgb.astype(np.float32) - background_rgb.astype(np.float32)),
                axis=2,
            )
            rgb_candidates = difference >= settings.rgb_difference_threshold
            rgb_candidate_mask, rgb_candidate_count = _component_filter(
                rgb_candidates, settings.object_min_pixels
            )

        frame.xyz = np.ascontiguousarray(points, dtype=np.float32)
        frame.depth_mm = np.clip(
            np.nan_to_num(np.abs(points[:, :, 2]) * 1000.0), 0, 65535
        ).astype(np.uint16)
        frame.processing_metadata = {
            "enhancement_settings": settings.as_dict(),
            "raw_source": raw_frame.source,
            "reference_plane_origin_m": None if plane_origin is None else plane_origin.tolist(),
            "reference_plane_normal": None if plane_normal is None else plane_normal.tolist(),
            "vertical_exaggeration_is_display_only": True,
            "background_reference_active": background_active,
            "background_reference_frames": (
                self._background_reference.captured_frames
                if background_active and self._background_reference is not None
                else 0
            ),
            "floor_noise_sigma_mm": floor_noise_sigma_mm,
            "detected_depth_objects": object_count,
            "detected_rgb_candidates": rgb_candidate_count,
        }

        display_rgb = frame.rgb.copy()
        if settings.height_color_enabled and height_mm is not None:
            display_rgb = _height_colors(
                display_rgb, height_mm, valid, settings.height_min_mm,
                settings.height_max_mm, settings.height_color_blend,
            )

        display_transform = np.eye(4, dtype=np.float64)
        if (
            abs(settings.vertical_exaggeration - 1.0) > 1e-9
            and plane_origin is not None
            and plane_normal is not None
        ):
            display_transform = _vertical_scale_transform(
                plane_origin, plane_normal, settings.vertical_exaggeration
            )
        display_xyz = frame.xyz.copy()
        display_xyz[valid] = transform_points(frame.xyz[valid], display_transform)
        return EnhancementResult(
            frame=frame,
            display_xyz=display_xyz,
            display_rgb=display_rgb,
            height_mm=height_mm,
            display_transform=display_transform,
            inverse_display_transform=np.linalg.inv(display_transform),
            plane_origin_m=plane_origin,
            plane_normal=plane_normal,
            object_mask=object_mask,
            rgb_candidate_mask=rgb_candidate_mask,
            object_count=object_count,
            floor_noise_sigma_mm=floor_noise_sigma_mm,
            background_active=background_active,
        )
