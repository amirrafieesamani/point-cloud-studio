from __future__ import annotations

import json
import os
import struct
import sys
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realsense_cropper.capture import create_demo_frame
from realsense_cropper.exporter import export_selection, write_binary_pcd, write_binary_ply
from realsense_cropper.enhancement import clone_frame
from realsense_cropper.geometry import points_inside_planes
from realsense_cropper.main_window import MainWindow
from realsense_cropper.origin import calculate_origin_anchor
from realsense_cropper.rgbd_io import load_rgbd_frame, save_rgbd_frame
from realsense_cropper.scene_importer import load_3d_scene


ARTIFACTS = PROJECT_ROOT / "verification_artifacts"


def drag_box_face_handle(window: MainWindow, point_index: int, pixels: int = 60) -> None:
    poly = pv.PolyData()
    window._box_widget.GetPolyData(poly)
    renderer = window.plotter.renderer

    def project(point: np.ndarray) -> np.ndarray:
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        return np.asarray(renderer.GetDisplayPoint()[:2], dtype=np.float64)

    handle_display = project(poly.points[point_index])
    centre_display = project(poly.points[14])
    direction = handle_display - centre_display
    direction /= max(float(np.linalg.norm(direction)), 1.0)
    target_display = handle_display + direction * pixels
    widget = window.plotter.interactor
    render_size = np.asarray(window.plotter.render_window.GetSize(), dtype=np.float64)
    widget_size = np.array([widget.width(), widget.height()], dtype=np.float64)

    def to_qt(display: np.ndarray) -> QPoint:
        logical = np.array(
            [display[0] * widget_size[0] / render_size[0],
             (render_size[1] - display[1]) * widget_size[1] / render_size[1]]
        )
        return QPoint(int(round(logical[0])), int(round(logical[1])))

    QTest.mouseMove(widget, to_qt(handle_display), delay=10)
    QTest.mousePress(widget, Qt.LeftButton, pos=to_qt(handle_display), delay=10)
    QTest.mouseMove(widget, to_qt(target_display), delay=50)
    QTest.mouseRelease(widget, Qt.LeftButton, pos=to_qt(target_display), delay=10)
    QApplication.processEvents()
    assert window._crop_resize_widget._selected_index is None


def validate_pcd(path: Path, expected_points: int) -> None:
    data = path.read_bytes()
    marker = b"DATA binary\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii")
    assert f"POINTS {expected_points}\n" in header
    assert len(data) - header_end == expected_points * 16


def run_data_pipeline() -> tuple[Path, dict[str, int | str]]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame = create_demo_frame(320, 240)
    sample_path = ARTIFACTS / "sample_rgbd.npz"
    save_rgbd_frame(sample_path, frame)
    imported = load_rgbd_frame(sample_path)
    np.testing.assert_equal(imported.xyz, frame.xyz)
    np.testing.assert_array_equal(imported.rgb, frame.rgb)

    cv2.imwrite(str(ARTIFACTS / "rgb_input.png"), imported.rgb[:, :, ::-1])

    sample_ply = ARTIFACTS / "sample_colored_cloud.ply"
    sample_pcd = ARTIFACTS / "sample_colored_cloud.pcd"
    write_binary_ply(sample_ply, imported.valid_points, imported.valid_colors)
    write_binary_pcd(sample_pcd, imported.valid_points, imported.valid_colors)
    imported_ply = load_3d_scene(sample_ply, unit_mode="metres")
    imported_pcd = load_3d_scene(sample_pcd, unit_mode="metres")
    assert len(imported_ply.valid_points) == len(imported.valid_points)
    assert len(imported_pcd.valid_points) == len(imported.valid_points)
    np.testing.assert_array_equal(imported_ply.valid_colors, imported.valid_colors)
    np.testing.assert_array_equal(imported_pcd.valid_colors, imported.valid_colors)
    depth = imported.depth_mm.astype(np.float32)
    depth_scaled = np.clip(depth / 1200.0 * 255.0, 0, 255).astype(np.uint8)
    depth_preview = cv2.applyColorMap(255 - depth_scaled, cv2.COLORMAP_TURBO)
    depth_preview[depth == 0] = 0
    cv2.imwrite(str(ARTIFACTS / "depth_input.png"), depth_preview)

    valid_points = imported.valid_points
    low = np.percentile(valid_points, 18, axis=0)
    high = np.percentile(valid_points, 82, axis=0)
    origins = np.array(
        [
            [low[0], 0, 0],
            [high[0], 0, 0],
            [0, low[1], 0],
            [0, high[1], 0],
            [0, 0, low[2]],
            [0, 0, high[2]],
        ],
        dtype=float,
    )
    normals = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=float,
    )
    selected_valid = points_inside_planes(valid_points, origins, normals)
    selected_mask = np.zeros(imported.valid_mask.shape, dtype=bool)
    selected_mask[imported.valid_mask] = selected_valid
    export_origin = calculate_origin_anchor(imported.xyz[selected_mask], "min_z")
    result = export_selection(
        ARTIFACTS / "exports",
        imported,
        selected_mask,
        {"plane_origins": origins.tolist(), "plane_normals": normals.tolist()},
        max_edge_m=0.02,
        export_origin_m=export_origin,
        origin_mode="min_z",
    )

    ply = pv.read(result.folder / "selected_cloud.ply")
    stl = pv.read(result.folder / "selected_mesh.stl")
    assert ply.n_points == result.point_count
    assert stl.n_cells == result.triangle_count
    assert "RGB" in ply.point_data
    assert np.asarray(ply["RGB"]).shape == (result.point_count, 3)
    assert np.linalg.norm(np.asarray(ply.points), axis=1).min() < 1e-6
    validate_pcd(result.folder / "selected_cloud.pcd", result.point_count)

    stl_header = (result.folder / "selected_mesh.stl").read_bytes()[:84]
    assert struct.unpack("<I", stl_header[80:84])[0] == result.triangle_count
    metadata = json.loads((result.folder / "metadata.json").read_text("utf-8"))
    assert metadata["point_count"] == result.point_count
    assert metadata["triangle_count"] == result.triangle_count

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.set_background("#15191f")
    plotter.add_points(
        ply,
        scalars="RGB",
        rgb=True,
        point_size=4,
        render_points_as_spheres=False,
    )
    plotter.add_axes()
    plotter.show_grid(color="#58606b")
    plotter.view_isometric()
    plotter.camera.zoom(1.25)
    plotter.screenshot(ARTIFACTS / "point_cloud_import.png")
    plotter.close()

    report: dict[str, int | str] = {
        "sample": str(sample_path),
        "export_folder": str(result.folder),
        "input_valid_points": int(imported.valid_mask.sum()),
        "selected_points": int(result.point_count),
        "mesh_triangles": int(result.triangle_count),
        "ply_reimport_points": int(ply.n_points),
        "stl_reimport_triangles": int(stl.n_cells),
        "pipeline_origin_mode": "min_z",
        "pipeline_origin_camera_m": [float(value) for value in export_origin],
        "generic_ply_import_points": int(len(imported_ply.valid_points)),
        "generic_pcd_import_points": int(len(imported_pcd.valid_points)),
    }
    return result.folder, report


def capture_gui(report: dict[str, int | str]) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    imported = load_rgbd_frame(ARTIFACTS / "sample_rgbd.npz")
    window._prepare_new_source()
    window._accept_frame(imported, reset_box=True)
    window.freeze_button.setChecked(True)
    window._set_frozen(True)
    crop_transform_index = window.interaction_tool_combo.findData("crop_transform")
    crop_resize_index = window.interaction_tool_combo.findData("crop_resize")
    export_frame_index = window.interaction_tool_combo.findData("export_frame")
    window.interaction_tool_buttons["crop_transform"].click()
    assert window.interaction_tool_combo.currentIndex() == crop_transform_index
    assert window._box_widget.GetEnabled() == 0
    assert any(actor.GetVisibility() for actor in window._crop_affine_widget._arrows)
    assert not any(actor.GetVisibility() for actor in window._origin_widget._arrows)
    box_before = pv.PolyData()
    window._box_widget.GetPolyData(box_before)
    box_before_center = np.asarray(box_before.center).copy()
    crop_matrix = np.eye(4)
    crop_matrix[0, 3] = 0.005
    window._on_crop_gizmo_released(crop_matrix)
    box_after = pv.PolyData()
    window._box_widget.GetPolyData(box_after)
    np.testing.assert_allclose(
        np.asarray(box_after.center) - box_before_center, [0.005, 0.0, 0.0], atol=1e-6
    )
    window.interaction_tool_buttons["crop_resize"].click()
    assert window.interaction_tool_combo.currentIndex() == crop_resize_index
    assert window._box_widget.GetEnabled() == 0
    assert window._crop_resize_widget is not None
    assert window._crop_resize_widget._enabled
    assert all(actor.GetVisibility() for actor in window._crop_resize_widget.handle_actors)
    assert not any(actor.GetVisibility() for actor in window._crop_affine_widget._arrows)
    resize_before = pv.PolyData()
    window._box_widget.GetPolyData(resize_before)
    resize_before_length = resize_before.length
    mouse_resize_deltas: dict[int, float] = {}
    for point_index in range(8, 14):
        before_handle_drag = pv.PolyData()
        window._box_widget.GetPolyData(before_handle_drag)
        before_length = before_handle_drag.length
        drag_box_face_handle(window, point_index=point_index)
        app.processEvents()
        after_handle_drag = pv.PolyData()
        window._box_widget.GetPolyData(after_handle_drag)
        mouse_resize_deltas[point_index] = float(after_handle_drag.length - before_length)
    assert all(delta > 1e-5 for delta in mouse_resize_deltas.values())
    report["gui_all_six_resize_handles_verified"] = True
    resize_after = pv.PolyData()
    window._box_widget.GetPolyData(resize_after)
    assert resize_after.length > resize_before_length
    app.processEvents()
    window.plotter.render()
    window.plotter.screenshot(ARTIFACTS / "interactive_crop_resize.png")
    window.interaction_tool_buttons["crop_transform"].click()
    app.processEvents()
    window.plotter.render()
    window.plotter.screenshot(ARTIFACTS / "interactive_crop_transform.png")
    max_z_index = window.origin_mode_combo.findData("max_z")
    window.origin_mode_combo.setCurrentIndex(max_z_index)
    window.statusBar().showMessage(
        "Imported sample_rgbd.npz; automatic Maximum Z export origin is active."
    )

    def capture_and_exit() -> None:
        app.processEvents()
        window.plotter.render()
        screen = app.primaryScreen()
        auto_screenshot = screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        if not auto_screenshot.save(str(ARTIFACTS / "interactive_application_auto_origin.png")):
            raise RuntimeError("Qt could not save the application screenshot")
        report["gui_auto_origin_mode"] = window._current_origin_mode()
        report["gui_auto_origin_camera_m"] = [float(value) for value in window.export_origin_m]

        manual_index = window.origin_mode_combo.findData("manual")
        window.origin_mode_combo.setCurrentIndex(manual_index)
        assert window.interaction_tool_combo.currentIndex() == export_frame_index
        assert window.interaction_tool_buttons["export_frame"].isChecked()
        assert window._box_widget.GetEnabled() == 0
        assert any(actor.GetVisibility() for actor in window._origin_widget._arrows)
        assert not any(actor.GetVisibility() for actor in window._crop_affine_widget._arrows)
        window.origin_spins[0].setValue(window.origin_spins[0].value() + 10.0)
        simulated_drag_target = window.export_origin_m + np.array([0.0, 0.005, 0.0])
        window._set_export_origin(simulated_drag_target, move_widget=False)
        window._render_origin_controls(manual=True, rebuild_widget=True)
        np.testing.assert_allclose(window.export_origin_m, simulated_drag_target, atol=1e-9)
        assert window._origin_widget is not None
        display_origin = window._true_to_display_point(window.export_origin_m)
        angle = np.deg2rad(12.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        gizmo_matrix = np.eye(4)
        gizmo_matrix[:3, :3] = rotation
        gizmo_matrix[:3, 3] = display_origin - rotation @ display_origin + [0.0, 0.002, 0.0]
        origin_callback_started = perf_counter()
        window._on_origin_gizmo_released(gizmo_matrix)
        report["gui_origin_release_callback_ms"] = round(
            (perf_counter() - origin_callback_started) * 1000.0, 2
        )
        np.testing.assert_allclose(window.export_rotation.T @ window.export_rotation, np.eye(3), atol=1e-7)
        assert not np.allclose(window.export_rotation, np.eye(3))
        window.show_normals_check.setChecked(True)
        toward_view_index = window.normal_orientation_combo.findData("toward_view")
        window.normal_orientation_combo.setCurrentIndex(toward_view_index)
        window.normal_count_spin.setValue(250)
        window.normal_length_spin.setValue(8.0)
        window._update_normals()
        assert window._normal_result is not None
        window.statusBar().showMessage(
            "Manual export frame is active; use arrows to translate and rings to rotate."
        )
        app.processEvents()
        window.plotter.render()
        window.plotter.screenshot(ARTIFACTS / "interactive_3d_view.png")
        manual_screenshot = screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        if not manual_screenshot.save(str(ARTIFACTS / "interactive_application.png")):
            raise RuntimeError("Qt could not save the manual-origin application screenshot")
        report["gui_title"] = window.windowTitle()
        report["gui_selected_label"] = window.selection_label.text()
        report["gui_manual_origin_camera_m"] = [float(value) for value in window.export_origin_m]
        report["gui_manual_drag_callback_verified"] = True
        report["gui_manual_widget_created"] = True
        report["gui_crop_translation_callback_verified"] = True
        report["gui_crop_resize_callback_verified"] = True
        report["gui_exclusive_interaction_modes_verified"] = True
        report["gui_origin_rotation_callback_verified"] = True
        report["gui_normal_method"] = window._normal_result.method
        report["gui_normal_orientation"] = window._normal_result.orientation

        imported_ply = load_3d_scene(
            ARTIFACTS / "sample_colored_cloud.ply", unit_mode="metres"
        )
        window._prepare_new_source()
        window._accept_frame(imported_ply, reset_box=True)
        window.statusBar().showMessage(
            "Imported colored PLY point cloud; projected RGB/depth previews are active."
        )
        app.processEvents()
        window.plotter.render()
        ply_screenshot = screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        if not ply_screenshot.save(str(ARTIFACTS / "interactive_application_ply_import.png")):
            raise RuntimeError("Qt could not save the PLY-import application screenshot")
        report["gui_ply_import_points"] = int(len(imported_ply.valid_points))
        report["gui_ply_preview_verified"] = True

        background = load_rgbd_frame(ARTIFACTS / "sample_rgbd.npz")
        organized = clone_frame(background)
        rows, cols = np.indices(organized.depth_mm.shape)
        raised = (
            ((cols - 125) ** 2 + (rows - 115) ** 2 < 11 ** 2)
            | ((cols - 175) ** 2 + (rows - 105) ** 2 < 9 ** 2)
        ) & organized.valid_mask
        old_z = organized.xyz[:, :, 2].copy()
        new_z = old_z.copy()
        new_z[raised] -= 0.010
        scale = np.ones_like(new_z)
        scale[raised] = new_z[raised] / old_z[raised]
        organized.xyz[:, :, 0][raised] *= scale[raised]
        organized.xyz[:, :, 1][raised] *= scale[raised]
        organized.xyz[:, :, 2][raised] = new_z[raised]
        organized.depth_mm[raised] = np.rint(new_z[raised] * 1000.0).astype(np.uint16)
        organized.rgb[raised] = (8, 8, 8)
        rgb_only = ((cols - 155) ** 2 + (rows - 145) ** 2 < 8 ** 2) & organized.valid_mask
        organized.rgb[rgb_only] = (0, 0, 0)
        organized.xyz[rgb_only] = np.nan
        organized.depth_mm[rgb_only] = 0
        window.enhancement_processor.set_background_from_frame(background)
        window._prepare_new_source()
        window._accept_frame(organized, reset_box=True)
        window._set_recommended_enhancement()
        window._show_top_view()
        window.controls_scroll.verticalScrollBar().setValue(520)
        window.statusBar().showMessage(
            "Recommended 10 mm enhancement preset; every parameter remains editable."
        )
        app.processEvents()
        window.plotter.render()
        enhancement_screenshot = (
            screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        )
        if not enhancement_screenshot.save(
            str(ARTIFACTS / "interactive_application_enhancement_gui.png")
        ):
            raise RuntimeError("Qt could not save the enhancement GUI screenshot")
        report["enhancement_valid_points"] = int(window.frame.valid_mask.sum())
        report["enhancement_height_map_created"] = window._height_mm is not None
        report["enhancement_vertical_exaggeration"] = window.vertical_exaggeration.value()
        report["enhancement_raw_frame_preserved"] = window.raw_frame is organized
        report["background_reference_active"] = bool(
            window.frame.processing_metadata["background_reference_active"]
        )
        report["background_detected_depth_objects"] = window._detected_object_count
        report["background_rgb_candidates_present"] = bool(
            window._rgb_candidate_mask is not None and window._rgb_candidate_mask.any()
        )
        window.controls_scroll.verticalScrollBar().setValue(0)
        app.processEvents()
        overview_screenshot = (
            screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        )
        if not overview_screenshot.save(
            str(ARTIFACTS / "interactive_application_background_overview.png")
        ):
            raise RuntimeError("Qt could not save the background overview screenshot")
        window.controls_scroll.verticalScrollBar().setValue(870)
        app.processEvents()
        background_screenshot = (
            screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
        )
        if not background_screenshot.save(
            str(ARTIFACTS / "interactive_application_background_detection.png")
        ):
            raise RuntimeError("Qt could not save the background-detection screenshot")
        (ARTIFACTS / "verification_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        window.close()
        app.quit()

    QTimer.singleShot(1800, capture_and_exit)
    exit_code = app.exec()
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    _, report = run_data_pipeline()
    capture_gui(report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
