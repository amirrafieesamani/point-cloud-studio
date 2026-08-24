from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .capture import RealSenseSession, create_demo_frame
from .enhancement import EnhancementProcessor, EnhancementResult, EnhancementSettings, transform_points
from .exporter import export_selection
from .geometry import finite_bounds, points_inside_planes
from .models import CaptureFrame
from .normals import NORMAL_METHODS, NORMAL_ORIENTATIONS, NormalResult, estimate_normals
from .origin import ORIGIN_MODES, calculate_origin_anchor
from .photoneo import capture_photoneo
from .resize_widget import FaceResizeWidget
from .scene_importer import SUPPORTED_3D_EXTENSIONS, attach_rgb_image, load_3d_scene
from .transforms import (
    euler_xyz_degrees_to_matrix,
    matrix_to_euler_xyz_degrees,
    orthonormalize_rotation,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Camera 3D Crop Studio")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)

        self.session = RealSenseSession()
        self.enhancement_processor = EnhancementProcessor()
        self.raw_frame: CaptureFrame | None = None
        self.frame: CaptureFrame | None = None
        self._display_xyz: np.ndarray | None = None
        self._display_rgb: np.ndarray | None = None
        self._height_mm: np.ndarray | None = None
        self._object_mask: np.ndarray | None = None
        self._rgb_candidate_mask: np.ndarray | None = None
        self._detected_object_count = 0
        self._display_transform = np.eye(4, dtype=np.float64)
        self._inverse_display_transform = np.eye(4, dtype=np.float64)
        self.box_origins: np.ndarray | None = None
        self.box_normals: np.ndarray | None = None
        self.frozen = False
        self._camera_was_reset = False
        self._last_selection_mask: np.ndarray | None = None
        self.export_origin_m = np.zeros(3, dtype=np.float64)
        self.export_rotation = np.eye(3, dtype=np.float64)
        self._origin_internal_update = False
        self._origin_widget: object | None = None
        self._origin_proxy_actor: object | None = None
        self._origin_gizmo_base: tuple[np.ndarray, np.ndarray] | None = None
        self._origin_axis_length_m = 0.03
        self._box_widget: object | None = None
        self._crop_proxy_actor: object | None = None
        self._crop_affine_widget: object | None = None
        self._crop_gizmo_internal_update = False
        self._crop_resize_widget: FaceResizeWidget | None = None
        self._normal_result: NormalResult | None = None
        self._cloud_bounds: tuple[float, float, float, float, float, float] | None = None
        self._preview_clock = QElapsedTimer()
        self._preview_clock.start()

        self._build_ui()
        self._build_actions()

        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._poll_camera)
        self.timer.start()

        self.statusBar().showMessage(
            "Use Demo Mode, connect a RealSense, capture from Photoneo, or open a 3D file."
        )

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        controls = QWidget()
        controls.setLayoutDirection(Qt.LeftToRight)
        controls.setMinimumWidth(390)
        controls.setMaximumWidth(500)
        controls_layout = QVBoxLayout(controls)

        camera_box = QGroupBox("Input source")
        camera_layout = QVBoxLayout(camera_box)
        row = QHBoxLayout()
        self.connect_button = QPushButton("Connect RealSense")
        self.connect_button.clicked.connect(self._toggle_camera)
        self.demo_button = QPushButton("Demo mode")
        self.demo_button.clicked.connect(self._start_demo)
        row.addWidget(self.connect_button)
        row.addWidget(self.demo_button)
        camera_layout.addLayout(row)
        file_row = QHBoxLayout()
        self.load_3d_button = QPushButton("Open 3D / RGB-D file...")
        self.load_3d_button.setToolTip(
            "Open NPZ, PRAW/PMRAW, PLY, PCD, PTX, XYZ, TXT, CSV, STL, OBJ, OFF, VTK, or VTP."
        )
        self.load_3d_button.clicked.connect(self._load_3d_file)
        self.attach_rgb_button = QPushButton("Attach RGB...")
        self.attach_rgb_button.setToolTip(
            "Attach a PNG, JPEG, BMP, or TIFF image whose pixels match the point topology."
        )
        self.attach_rgb_button.clicked.connect(self._attach_rgb_file)
        file_row.addWidget(self.load_3d_button, 2)
        file_row.addWidget(self.attach_rgb_button, 1)
        camera_layout.addLayout(file_row)

        photoneo_row = QHBoxLayout()
        self.photoneo_id_edit = QLineEdit()
        self.photoneo_id_edit.setPlaceholderText("Photoneo device ID (optional)")
        self.photoneo_button = QPushButton("Capture Photoneo")
        self.photoneo_button.setToolTip(
            "Capture one frame through PhoXi Control 1.17+ and the optional phoxi-api package."
        )
        self.photoneo_button.clicked.connect(self._capture_photoneo_frame)
        photoneo_row.addWidget(self.photoneo_id_edit, 2)
        photoneo_row.addWidget(self.photoneo_button, 1)
        camera_layout.addLayout(photoneo_row)

        settings_form = QFormLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640 × 480 @ 30", "848 × 480 @ 30"])
        settings_form.addRow("Resolution:", self.resolution_combo)
        self.input_units_combo = QComboBox()
        self.input_units_combo.addItem("Auto-detect", "auto")
        self.input_units_combo.addItem("Metres", "metres")
        self.input_units_combo.addItem("Millimetres", "millimetres")
        self.input_units_combo.setToolTip(
            "Applied to generic 3D files. Photoneo PTX/PRAW data is always interpreted as millimetres."
        )
        settings_form.addRow("File units:", self.input_units_combo)
        self.display_step = QSpinBox()
        self.display_step.setRange(1, 8)
        self.display_step.setValue(2)
        self.display_step.setToolTip(
            "Reduces display load only; exported files keep the full resolution."
        )
        self.display_step.valueChanged.connect(lambda _value: self._refresh_current_frame())
        settings_form.addRow("Display stride:", self.display_step)
        self.max_depth = QDoubleSpinBox()
        self.max_depth.setRange(0.2, 10.0)
        self.max_depth.setValue(2.0)
        self.max_depth.setSingleStep(0.1)
        self.max_depth.setSuffix(" m")
        self.max_depth.valueChanged.connect(lambda _value: self._refresh_images())
        settings_form.addRow("Depth display limit:", self.max_depth)
        camera_layout.addLayout(settings_form)

        row = QHBoxLayout()
        self.freeze_button = QPushButton("Freeze frame")
        self.freeze_button.setCheckable(True)
        self.freeze_button.clicked.connect(self._set_frozen)
        self.reset_box_button = QPushButton("Reset crop box")
        self.reset_box_button.clicked.connect(self._reset_box)
        row.addWidget(self.freeze_button)
        row.addWidget(self.reset_box_button)
        camera_layout.addLayout(row)

        manipulation_row = QHBoxLayout()
        self.interaction_button_group = QButtonGroup(self)
        self.interaction_button_group.setExclusive(True)
        self.interaction_tool_buttons: dict[str, QPushButton] = {}
        for label, mode in (
            ("Crop move / rotate", "crop_transform"),
            ("Crop resize", "crop_resize"),
            ("Origin move / rotate", "export_frame"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, selected_mode=mode: (
                    self._set_interaction_tool(selected_mode) if checked else None
                )
            )
            self.interaction_button_group.addButton(button)
            self.interaction_tool_buttons[mode] = button
            manipulation_row.addWidget(button)
        self.interaction_tool_buttons["crop_transform"].setChecked(True)
        camera_layout.addLayout(manipulation_row)
        self.interaction_tool_combo = QComboBox(self)
        self.interaction_tool_combo.addItem("Crop: translate / rotate", "crop_transform")
        self.interaction_tool_combo.addItem("Crop: resize with face handles", "crop_resize")
        self.interaction_tool_combo.addItem("Export frame: translate / rotate", "export_frame")
        self.interaction_tool_combo.currentIndexChanged.connect(
            lambda _index: self._on_interaction_tool_changed()
        )
        self.interaction_tool_combo.hide()
        controls_layout.addWidget(camera_box)

        self.rgb_label = self._make_image_label("RGB")
        self.depth_label = self._make_image_label("Depth")
        image_row = QHBoxLayout()
        image_row.addWidget(self.rgb_label)
        image_row.addWidget(self.depth_label)
        controls_layout.addLayout(image_row)

        enhancement_box = QGroupBox("Point-cloud enhancement")
        enhancement_layout = QVBoxLayout(enhancement_box)
        self.enhancement_enabled = QCheckBox("Enable reversible enhancement")
        self.enhancement_enabled.setToolTip(
            "All filters are recalculated from the untouched raw frame."
        )
        enhancement_layout.addWidget(self.enhancement_enabled)

        enhancement_form = QFormLayout()
        self.average_check = QCheckBox("Multi-frame median")
        self.average_frames = QSpinBox()
        self.average_frames.setRange(2, 60)
        self.average_frames.setValue(12)
        self.average_frames.setSuffix(" frames")
        self.average_frames.setToolTip("Live camera only; the scene must remain stationary.")
        enhancement_form.addRow(self.average_check, self.average_frames)

        self.temporal_check = QCheckBox("Temporal smoothing")
        self.temporal_alpha = QDoubleSpinBox()
        self.temporal_alpha.setRange(0.05, 1.0)
        self.temporal_alpha.setSingleStep(0.05)
        self.temporal_alpha.setValue(0.35)
        self.temporal_alpha.setToolTip("Lower values are smoother but react more slowly.")
        enhancement_form.addRow(self.temporal_check, self.temporal_alpha)

        self.hole_fill_check = QCheckBox("Hole filling")
        hole_row = QWidget()
        hole_layout = QHBoxLayout(hole_row)
        hole_layout.setContentsMargins(0, 0, 0, 0)
        self.hole_radius = QSpinBox()
        self.hole_radius.setRange(1, 4)
        self.hole_radius.setValue(1)
        self.hole_radius.setPrefix("Radius ")
        self.hole_radius.setSuffix(" px")
        self.hole_iterations = QSpinBox()
        self.hole_iterations.setRange(1, 4)
        self.hole_iterations.setValue(1)
        self.hole_iterations.setPrefix("Passes ")
        hole_layout.addWidget(self.hole_radius)
        hole_layout.addWidget(self.hole_iterations)
        enhancement_form.addRow(self.hole_fill_check, hole_row)

        self.spatial_check = QCheckBox("Spatial bilateral")
        spatial_row = QWidget()
        spatial_layout = QHBoxLayout(spatial_row)
        spatial_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_diameter = QComboBox()
        for value in (3, 5, 7, 9):
            self.spatial_diameter.addItem(f"{value} px", value)
        self.spatial_diameter.setCurrentIndex(1)
        self.spatial_sigma = QDoubleSpinBox()
        self.spatial_sigma.setRange(0.5, 50.0)
        self.spatial_sigma.setValue(8.0)
        self.spatial_sigma.setSuffix(" mm")
        spatial_layout.addWidget(self.spatial_diameter)
        spatial_layout.addWidget(self.spatial_sigma)
        enhancement_form.addRow(self.spatial_check, spatial_row)
        self.spatial_strength = QDoubleSpinBox()
        self.spatial_strength.setRange(0.0, 1.0)
        self.spatial_strength.setSingleStep(0.05)
        self.spatial_strength.setValue(0.65)
        enhancement_form.addRow("Spatial strength:", self.spatial_strength)

        self.outlier_check = QCheckBox("Statistical outliers")
        outlier_row = QWidget()
        outlier_layout = QHBoxLayout(outlier_row)
        outlier_layout.setContentsMargins(0, 0, 0, 0)
        self.outlier_neighbors = QSpinBox()
        self.outlier_neighbors.setRange(5, 100)
        self.outlier_neighbors.setValue(20)
        self.outlier_neighbors.setPrefix("K ")
        self.outlier_std = QDoubleSpinBox()
        self.outlier_std.setRange(0.1, 5.0)
        self.outlier_std.setSingleStep(0.1)
        self.outlier_std.setValue(1.5)
        self.outlier_std.setPrefix("Std ")
        outlier_layout.addWidget(self.outlier_neighbors)
        outlier_layout.addWidget(self.outlier_std)
        enhancement_form.addRow(self.outlier_check, outlier_row)

        self.plane_align_check = QCheckBox("Flatten reference plane")
        self.plane_threshold = QDoubleSpinBox()
        self.plane_threshold.setRange(0.5, 30.0)
        self.plane_threshold.setValue(4.0)
        self.plane_threshold.setSuffix(" mm tolerance")
        enhancement_form.addRow(self.plane_align_check, self.plane_threshold)

        self.height_color_check = QCheckBox("Height color map")
        height_row = QWidget()
        height_layout = QHBoxLayout(height_row)
        height_layout.setContentsMargins(0, 0, 0, 0)
        self.height_min = QDoubleSpinBox()
        self.height_min.setRange(-100.0, 100.0)
        self.height_min.setValue(-2.0)
        self.height_min.setSuffix(" mm min")
        self.height_max = QDoubleSpinBox()
        self.height_max.setRange(-100.0, 200.0)
        self.height_max.setValue(20.0)
        self.height_max.setSuffix(" mm max")
        height_layout.addWidget(self.height_min)
        height_layout.addWidget(self.height_max)
        enhancement_form.addRow(self.height_color_check, height_row)
        self.height_blend = QDoubleSpinBox()
        self.height_blend.setRange(0.0, 1.0)
        self.height_blend.setSingleStep(0.05)
        self.height_blend.setValue(0.85)
        enhancement_form.addRow("Height-color blend:", self.height_blend)

        self.vertical_exaggeration = QDoubleSpinBox()
        self.vertical_exaggeration.setRange(1.0, 10.0)
        self.vertical_exaggeration.setSingleStep(0.5)
        self.vertical_exaggeration.setValue(1.0)
        self.vertical_exaggeration.setSuffix("× display only")
        enhancement_form.addRow("Vertical exaggeration:", self.vertical_exaggeration)
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(1.0, 12.0)
        self.point_size_spin.setValue(3.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.valueChanged.connect(lambda _value: self._refresh_current_frame())
        enhancement_form.addRow("3D point size:", self.point_size_spin)

        self.background_check = QCheckBox("Use empty-table reference")
        self.background_frames = QSpinBox()
        self.background_frames.setRange(1, 120)
        self.background_frames.setValue(60)
        self.background_frames.setSuffix(" frames")
        enhancement_form.addRow(self.background_check, self.background_frames)

        self.floor_snap_check = QCheckBox("Snap measured floor to zero")
        self.floor_snap = QDoubleSpinBox()
        self.floor_snap.setRange(0.2, 10.0)
        self.floor_snap.setValue(2.5)
        self.floor_snap.setSuffix(" mm band")
        enhancement_form.addRow(self.floor_snap_check, self.floor_snap)

        self.object_detection_check = QCheckBox("Detect raised objects")
        object_row = QWidget()
        object_layout = QHBoxLayout(object_row)
        object_layout.setContentsMargins(0, 0, 0, 0)
        self.object_threshold = QDoubleSpinBox()
        self.object_threshold.setRange(0.5, 100.0)
        self.object_threshold.setValue(4.0)
        self.object_threshold.setSuffix(" mm")
        self.object_min_pixels = QSpinBox()
        self.object_min_pixels.setRange(1, 10000)
        self.object_min_pixels.setValue(12)
        self.object_min_pixels.setSuffix(" px min")
        object_layout.addWidget(self.object_threshold)
        object_layout.addWidget(self.object_min_pixels)
        enhancement_form.addRow(self.object_detection_check, object_row)

        self.rgb_assist_check = QCheckBox("RGB change assist")
        self.rgb_difference = QDoubleSpinBox()
        self.rgb_difference.setRange(1.0, 255.0)
        self.rgb_difference.setValue(35.0)
        self.rgb_difference.setSuffix(" intensity")
        self.rgb_difference.setToolTip(
            "Marks 2D changes from the empty-table RGB image, including dark objects with invalid depth."
        )
        enhancement_form.addRow(self.rgb_assist_check, self.rgb_difference)

        self.show_detected_check = QCheckBox("Show detected-object overlay")
        self.show_detected_check.setChecked(True)
        self.show_detected_check.toggled.connect(
            lambda _checked: self._refresh_current_frame()
        )
        enhancement_form.addRow("Detection overlay:", self.show_detected_check)
        enhancement_layout.addLayout(enhancement_form)

        background_buttons = QHBoxLayout()
        self.capture_background_button = QPushButton("Capture empty table")
        self.capture_background_button.clicked.connect(self._capture_empty_table)
        self.save_background_button = QPushButton("Save reference...")
        self.save_background_button.clicked.connect(self._save_background_reference)
        self.load_background_button = QPushButton("Load reference...")
        self.load_background_button.clicked.connect(self._load_background_reference)
        background_buttons.addWidget(self.capture_background_button)
        background_buttons.addWidget(self.save_background_button)
        background_buttons.addWidget(self.load_background_button)
        enhancement_layout.addLayout(background_buttons)
        clear_background_button = QPushButton("Clear empty-table reference")
        clear_background_button.clicked.connect(self._clear_background_reference)
        enhancement_layout.addWidget(clear_background_button)
        self.background_status = QLabel("No empty-table reference is loaded.")
        self.background_status.setWordWrap(True)
        self.background_status.setStyleSheet("color: #666;")
        enhancement_layout.addWidget(self.background_status)

        enhancement_buttons = QHBoxLayout()
        self.recommended_button = QPushButton("10 mm preset")
        self.recommended_button.clicked.connect(self._set_recommended_enhancement)
        self.apply_enhancement_button = QPushButton("Apply")
        self.apply_enhancement_button.clicked.connect(self._apply_enhancements)
        self.reset_enhancement_button = QPushButton("Reset to raw")
        self.reset_enhancement_button.clicked.connect(self._reset_enhancements)
        enhancement_buttons.addWidget(self.recommended_button)
        enhancement_buttons.addWidget(self.apply_enhancement_button)
        enhancement_buttons.addWidget(self.reset_enhancement_button)
        enhancement_layout.addLayout(enhancement_buttons)
        view_buttons = QHBoxLayout()
        top_view_button = QPushButton("Top view")
        top_view_button.clicked.connect(self._show_top_view)
        clear_history_button = QPushButton("Clear temporal history")
        clear_history_button.clicked.connect(self._clear_enhancement_history)
        view_buttons.addWidget(top_view_button)
        view_buttons.addWidget(clear_history_button)
        enhancement_layout.addLayout(view_buttons)
        self.enhancement_status = QLabel("Raw data is active.")
        self.enhancement_status.setWordWrap(True)
        self.enhancement_status.setStyleSheet("color: #666;")
        enhancement_layout.addWidget(self.enhancement_status)
        normals_box = QGroupBox("Surface normals")
        normals_layout = QVBoxLayout(normals_box)
        self.show_normals_check = QCheckBox("Show normal direction arrows")
        self.show_normals_check.toggled.connect(lambda _checked: self._update_normals())
        normals_layout.addWidget(self.show_normals_check)
        normals_form = QFormLayout()
        self.normal_method_combo = QComboBox()
        for label, key in NORMAL_METHODS:
            self.normal_method_combo.addItem(label, key)
        normals_form.addRow("Estimation:", self.normal_method_combo)
        self.normal_orientation_combo = QComboBox()
        for label, key in NORMAL_ORIENTATIONS:
            self.normal_orientation_combo.addItem(label, key)
        self.normal_orientation_combo.setToolTip(
            "Closed imported meshes use topology for true outside orientation. "
            "Point clouds use the selected object centre or the current camera view."
        )
        normals_form.addRow("Direction:", self.normal_orientation_combo)
        self.normal_neighbors_spin = QSpinBox()
        self.normal_neighbors_spin.setRange(5, 100)
        self.normal_neighbors_spin.setValue(24)
        normals_form.addRow("PCA neighbours:", self.normal_neighbors_spin)
        self.normal_length_spin = QDoubleSpinBox()
        self.normal_length_spin.setRange(0.1, 1000.0)
        self.normal_length_spin.setValue(10.0)
        self.normal_length_spin.setSuffix(" mm")
        normals_form.addRow("Arrow length:", self.normal_length_spin)
        self.normal_count_spin = QSpinBox()
        self.normal_count_spin.setRange(10, 10000)
        self.normal_count_spin.setValue(800)
        normals_form.addRow("Maximum arrows:", self.normal_count_spin)
        normals_layout.addLayout(normals_form)
        self.normals_selected_only_check = QCheckBox("Use selected crop only")
        self.normals_selected_only_check.setChecked(True)
        self.export_normals_check = QCheckBox("Write normals to PLY and PCD")
        self.export_normals_check.setChecked(True)
        normals_layout.addWidget(self.normals_selected_only_check)
        normals_layout.addWidget(self.export_normals_check)
        recompute_normals_button = QPushButton("Recompute and display normals")
        recompute_normals_button.clicked.connect(self._update_normals)
        normals_layout.addWidget(recompute_normals_button)
        self.normal_status = QLabel("Normals have not been computed.")
        self.normal_status.setWordWrap(True)
        self.normal_status.setStyleSheet("color: #666;")
        normals_layout.addWidget(self.normal_status)
        self.normal_method_combo.currentIndexChanged.connect(
            lambda _index: self._update_normals_if_visible()
        )
        self.normal_orientation_combo.currentIndexChanged.connect(
            lambda _index: self._update_normals_if_visible()
        )
        self.normal_neighbors_spin.editingFinished.connect(self._update_normals_if_visible)
        self.normal_length_spin.editingFinished.connect(self._update_normals_if_visible)
        self.normal_count_spin.editingFinished.connect(self._update_normals_if_visible)
        self.normals_selected_only_check.toggled.connect(
            lambda _checked: self._update_normals_if_visible()
        )
        controls_layout.addWidget(normals_box)
        controls_layout.addWidget(enhancement_box)

        origin_box = QGroupBox("Export coordinate origin")
        origin_layout = QVBoxLayout(origin_box)
        self.origin_mode_combo = QComboBox()
        for label, key in ORIGIN_MODES:
            self.origin_mode_combo.addItem(label, key)
        origin_layout.addWidget(self.origin_mode_combo)
        origin_form = QFormLayout()
        self.origin_spins: list[QDoubleSpinBox] = []
        for axis in "XYZ":
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setSuffix(" mm")
            spin.valueChanged.connect(lambda _value: self._on_origin_coordinate_edited())
            self.origin_spins.append(spin)
            origin_form.addRow(f"{axis}:", spin)
        self.origin_rotation_spins: list[QDoubleSpinBox] = []
        for axis in "XYZ":
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix("°")
            spin.valueChanged.connect(lambda _value: self._on_origin_rotation_edited())
            self.origin_rotation_spins.append(spin)
            origin_form.addRow(f"R{axis}:", spin)
        origin_layout.addLayout(origin_form)
        reset_orientation_button = QPushButton("Reset export orientation")
        reset_orientation_button.clicked.connect(self._reset_export_orientation)
        origin_layout.addWidget(reset_orientation_button)
        origin_note = QLabel(
            "Automatic modes use the currently selected red points. Choose Manual or edit "
            "a coordinate to fine-tune it. Drag the RGB arrows for translation and the "
            "matching coloured rings for rotation. The full coordinate frame is applied "
            "to PLY, PCD, and STL."
        )
        origin_note.setWordWrap(True)
        origin_note.setStyleSheet("color: #666;")
        origin_layout.addWidget(origin_note)
        self.origin_mode_combo.currentIndexChanged.connect(
            lambda _index: self._on_origin_mode_changed()
        )
        controls_layout.addWidget(origin_box)

        output_box = QGroupBox("Selection export")
        output_layout = QVBoxLayout(output_box)
        folder_row = QHBoxLayout()
        self.output_edit = QLineEdit(str((Path.cwd() / "outputs").resolve()))
        self.output_edit.setLayoutDirection(Qt.LeftToRight)
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_output_folder)
        folder_row.addWidget(self.output_edit, 1)
        folder_row.addWidget(browse)
        output_layout.addLayout(folder_row)

        export_form = QFormLayout()
        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setRange(1.0, 100.0)
        self.edge_spin.setValue(20.0)
        self.edge_spin.setSuffix(" mm")
        self.edge_spin.setToolTip(
            "Triangles crossing a larger depth discontinuity are removed."
        )
        export_form.addRow("Maximum mesh edge:", self.edge_spin)
        self.selection_label = QLabel("Selected points: --")
        export_form.addRow("Status:", self.selection_label)
        self.selection_preview_check = QCheckBox("Show red selection preview")
        self.selection_preview_check.setChecked(True)
        self.selection_preview_check.toggled.connect(
            lambda _checked: self._update_selection_preview()
        )
        export_form.addRow("Overlay:", self.selection_preview_check)
        output_layout.addLayout(export_form)

        self.save_button = QPushButton("Confirm and save PLY + PCD + STL")
        self.save_button.setMinimumHeight(42)
        self.save_button.clicked.connect(self._save_selection)
        output_layout.addWidget(self.save_button)
        controls_layout.addWidget(output_box)

        help_label = QLabel(
            "Select exactly one Active 3D manipulation tool. Use Crop translate / rotate "
            "for the RGB arrows and rings, Crop resize for the six yellow face spheres, or "
            "Export frame for the coordinate-frame arrows and rings. "
            "Drag empty space with the left mouse button to orbit, use Shift+left drag "
            "to pan, and use the wheel to zoom. Freeze the frame before a final selection."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; padding: 4px;")
        controls_layout.addWidget(help_label)
        controls_layout.addStretch(1)

        view_frame = QFrame()
        view_layout = QVBoxLayout(view_frame)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(view_frame)
        view_layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#15191f")
        self.plotter.add_axes(line_width=2)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setMinimumWidth(410)
        self.controls_scroll.setMaximumWidth(530)
        self.controls_scroll.setWidget(controls)
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(view_frame)
        splitter.setSizes([430, 1070])

    @staticmethod
    def _make_image_label(title: str) -> QLabel:
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(175, 150)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet("background: #111; color: #aaa; border: 1px solid #444;")
        return label

    def _build_actions(self) -> None:
        save_action = QAction("Save selection", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_selection)
        self.addAction(save_action)
        freeze_action = QAction("Freeze", self)
        freeze_action.setShortcut(Qt.Key_Space)
        freeze_action.triggered.connect(lambda: self.freeze_button.click())
        self.addAction(freeze_action)

    def _selected_resolution(self) -> tuple[int, int, int]:
        if self.resolution_combo.currentIndex() == 1:
            return 848, 480, 30
        return 640, 480, 30

    def _current_enhancement_settings(self) -> EnhancementSettings:
        minimum = self.height_min.value()
        maximum = self.height_max.value()
        if maximum <= minimum:
            maximum = minimum + 0.1
        return EnhancementSettings(
            enabled=self.enhancement_enabled.isChecked(),
            multi_frame_enabled=self.average_check.isChecked(),
            multi_frame_count=self.average_frames.value(),
            temporal_enabled=self.temporal_check.isChecked(),
            temporal_alpha=self.temporal_alpha.value(),
            hole_fill_enabled=self.hole_fill_check.isChecked(),
            hole_radius=self.hole_radius.value(),
            hole_iterations=self.hole_iterations.value(),
            spatial_enabled=self.spatial_check.isChecked(),
            spatial_diameter=int(self.spatial_diameter.currentData()),
            spatial_sigma_mm=self.spatial_sigma.value(),
            spatial_strength=self.spatial_strength.value(),
            outlier_enabled=self.outlier_check.isChecked(),
            outlier_neighbors=self.outlier_neighbors.value(),
            outlier_std_ratio=self.outlier_std.value(),
            plane_align_enabled=self.plane_align_check.isChecked(),
            plane_threshold_mm=self.plane_threshold.value(),
            height_color_enabled=self.height_color_check.isChecked(),
            height_min_mm=minimum,
            height_max_mm=maximum,
            height_color_blend=self.height_blend.value(),
            vertical_exaggeration=self.vertical_exaggeration.value(),
            background_enabled=self.background_check.isChecked(),
            floor_snap_enabled=self.floor_snap_check.isChecked(),
            floor_snap_mm=self.floor_snap.value(),
            object_detection_enabled=self.object_detection_check.isChecked(),
            object_threshold_mm=self.object_threshold.value(),
            object_min_pixels=self.object_min_pixels.value(),
            rgb_assist_enabled=self.rgb_assist_check.isChecked(),
            rgb_difference_threshold=self.rgb_difference.value(),
        )

    def _set_recommended_enhancement(self) -> None:
        self.enhancement_enabled.setChecked(True)
        self.average_check.setChecked(True)
        self.average_frames.setValue(40)
        self.temporal_check.setChecked(True)
        self.temporal_alpha.setValue(0.30)
        self.hole_fill_check.setChecked(False)
        self.hole_radius.setValue(1)
        self.hole_iterations.setValue(1)
        self.spatial_check.setChecked(True)
        self.spatial_diameter.setCurrentIndex(self.spatial_diameter.findData(5))
        self.spatial_sigma.setValue(3.0)
        self.spatial_strength.setValue(0.40)
        self.outlier_check.setChecked(False)
        self.outlier_neighbors.setValue(20)
        self.outlier_std.setValue(1.5)
        self.plane_align_check.setChecked(True)
        self.plane_threshold.setValue(3.0)
        self.height_color_check.setChecked(True)
        self.height_min.setValue(-2.0)
        self.height_max.setValue(15.0)
        self.height_blend.setValue(0.90)
        self.vertical_exaggeration.setValue(2.0)
        self.selection_preview_check.setChecked(False)
        has_background = self.enhancement_processor.background_reference is not None
        self.background_check.setChecked(has_background)
        self.floor_snap_check.setChecked(True)
        self.floor_snap.setValue(2.5)
        self.object_detection_check.setChecked(True)
        self.object_threshold.setValue(4.0)
        self.object_min_pixels.setValue(12)
        self.rgb_assist_check.setChecked(has_background)
        if has_background:
            captured = self.enhancement_processor.background_reference.captured_frames
            self.background_status.setText(
                f"Empty-table reference is ready ({captured} frame"
                f"{'s' if captured != 1 else ''})."
            )
        self._apply_enhancements()

    def _apply_enhancements(self) -> None:
        if self.raw_frame is None:
            self.enhancement_status.setText("Open or capture a frame first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.enhancement_processor.reset_temporal()
            result = self.enhancement_processor.process(
                self.raw_frame, self._current_enhancement_settings(), update_history=False
            )
            self._accept_enhancement_result(result, reset_box=True)
            settings = self._current_enhancement_settings()
            active = [
                label for enabled, label in (
                    (settings.multi_frame_enabled, "multi-frame"),
                    (settings.temporal_enabled, "temporal"),
                    (settings.hole_fill_enabled, "hole fill"),
                    (settings.spatial_enabled, "spatial"),
                    (settings.outlier_enabled, "outliers"),
                    (settings.plane_align_enabled, "plane alignment"),
                    (settings.height_color_enabled, "height colors"),
                    (settings.background_enabled, "empty-table correction"),
                    (settings.floor_snap_enabled, "floor snap"),
                    (settings.object_detection_enabled, "object detection"),
                    (settings.rgb_assist_enabled, "RGB assist"),
                ) if enabled
            ]
            self.enhancement_status.setText(
                "Active: " + (", ".join(active) if settings.enabled and active else "raw geometry")
                + f". Valid points: {self.frame.valid_mask.sum():,}."
                + (
                    f" Floor noise sigma: {result.floor_noise_sigma_mm:.2f} mm."
                    if result.floor_noise_sigma_mm is not None
                    else ""
                )
                + (
                    f" Detected objects: {result.object_count}."
                    if result.object_mask is not None
                    else ""
                )
            )
        except Exception as exc:
            QMessageBox.critical(self, "Enhancement error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _reset_enhancements(self) -> None:
        self.enhancement_enabled.setChecked(False)
        self.enhancement_processor.reset()
        self._apply_enhancements()
        self.enhancement_status.setText("Raw data is active; no enhancement is applied.")

    def _clear_enhancement_history(self) -> None:
        self.enhancement_processor.reset()
        self.enhancement_status.setText("Multi-frame and temporal history was cleared.")

    def _capture_empty_table(self) -> None:
        if self.enhancement_processor.background_capture_active:
            self.enhancement_processor.cancel_background_capture()
            self.capture_background_button.setText("Capture empty table")
            self.background_status.setText("Empty-table capture was cancelled.")
            return
        if self.raw_frame is None:
            QMessageBox.information(
                self, "No frame", "Connect a camera or open an organized RGB-D frame first."
            )
            return
        try:
            if self.session.running:
                target = self.background_frames.value()
                self.enhancement_processor.begin_background_capture(target)
                self.freeze_button.setChecked(False)
                self._set_frozen(False)
                self.capture_background_button.setText("Cancel background capture")
                self.background_status.setText(
                    f"Capturing empty table: 0 / {target}. Keep the camera and table still."
                )
            else:
                self.enhancement_processor.set_background_from_frame(self.raw_frame)
                self._activate_background_controls(1)
        except Exception as exc:
            QMessageBox.critical(self, "Background capture error", str(exc))

    def _activate_background_controls(self, captured_frames: int) -> None:
        self.enhancement_enabled.setChecked(True)
        self.background_check.setChecked(True)
        self.plane_align_check.setChecked(True)
        self.floor_snap_check.setChecked(True)
        self.object_detection_check.setChecked(True)
        self.height_color_check.setChecked(True)
        self.rgb_assist_check.setChecked(True)
        self.selection_preview_check.setChecked(False)
        self.capture_background_button.setText("Capture empty table")
        self.background_status.setText(
            f"Empty-table reference is ready ({captured_frames} frame"
            f"{'s' if captured_frames != 1 else ''})."
        )
        self._apply_enhancements()

    def _save_background_reference(self) -> None:
        if self.enhancement_processor.background_reference is None:
            QMessageBox.information(
                self, "No reference", "Capture or load an empty-table reference first."
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save empty-table reference",
            str((Path.cwd() / "empty_table_reference.npz").resolve()),
            "RGB-D reference (*.npz)",
        )
        if not filename:
            return
        try:
            path = Path(filename)
            if path.suffix.lower() != ".npz":
                path = path.with_suffix(".npz")
            self.enhancement_processor.save_background(path)
            self.background_status.setText(f"Saved empty-table reference: {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Reference save error", str(exc))

    def _load_background_reference(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load empty-table reference",
            str(Path.cwd()),
            "RGB-D reference (*.npz);;All files (*)",
        )
        if not filename:
            return
        try:
            self.enhancement_processor.load_background(Path(filename))
            self._activate_background_controls(1)
            self.background_status.setText(
                f"Loaded empty-table reference: {Path(filename).name}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Reference load error", str(exc))

    def _clear_background_reference(self) -> None:
        self.enhancement_processor.clear_background()
        self.background_check.setChecked(False)
        self.rgb_assist_check.setChecked(False)
        self.capture_background_button.setText("Capture empty table")
        self.background_status.setText("No empty-table reference is loaded.")
        if self.raw_frame is not None:
            self._apply_enhancements()

    def _show_top_view(self) -> None:
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self.plotter.render()

    def _toggle_camera(self) -> None:
        if self.session.running:
            self._stop_camera()
            return
        try:
            width, height, fps = self._selected_resolution()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.session.start(width, height, fps)
            self._prepare_new_source()
            self.freeze_button.setChecked(False)
            self._set_frozen(False)
            self.connect_button.setText("Disconnect RealSense")
            self.demo_button.setEnabled(False)
            self.resolution_combo.setEnabled(False)
            self.statusBar().showMessage(
                f"Connected: {self.session.device_name} | S/N {self.session.serial}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Camera error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _stop_camera(self) -> None:
        try:
            self.session.stop()
        finally:
            self.connect_button.setText("Connect RealSense")
            self.demo_button.setEnabled(True)
            self.resolution_combo.setEnabled(True)
            self.statusBar().showMessage("Camera disconnected.")

    def _start_demo(self) -> None:
        if self.session.running:
            self._stop_camera()
        self._prepare_new_source()
        self._accept_frame(create_demo_frame(), reset_box=True)
        self.freeze_button.setChecked(True)
        self._set_frozen(True)
        self.statusBar().showMessage(
            "Demo mode is active; crop selection and export are ready to test."
        )

    def _load_3d_file(self) -> None:
        extensions = " ".join(f"*{extension}" for extension in sorted(SUPPORTED_3D_EXTENSIONS))
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D or RGB-D data",
            str(Path.cwd()),
            f"Supported 3D files ({extensions});;All files (*)",
        )
        if not filename:
            return
        try:
            if self.session.running:
                self._stop_camera()
            unit_mode = str(self.input_units_combo.currentData())
            frame = load_3d_scene(Path(filename), unit_mode=unit_mode)
            self._prepare_new_source()
            self._accept_frame(frame, reset_box=True)
            self.freeze_button.setChecked(True)
            self._set_frozen(True)
            self.statusBar().showMessage(
                f"Loaded {Path(filename).name}: {frame.valid_mask.sum():,} valid points "
                f"({frame.source_units} input converted to metres)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "3D import error", str(exc))

    def _capture_photoneo_frame(self) -> None:
        try:
            if self.session.running:
                self._stop_camera()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            frame = capture_photoneo(self.photoneo_id_edit.text())
            self._prepare_new_source()
            self._accept_frame(frame, reset_box=True)
            self.freeze_button.setChecked(True)
            self._set_frozen(True)
            self.statusBar().showMessage(
                f"Captured Photoneo frame: {frame.valid_mask.sum():,} valid points"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Photoneo capture error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _attach_rgb_file(self) -> None:
        if self.raw_frame is None:
            QMessageBox.information(self, "No 3D data", "Open or capture 3D data first.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Attach companion RGB image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not filename:
            return
        try:
            attach_rgb_image(self.raw_frame, Path(filename))
            self._apply_enhancements()
            self.statusBar().showMessage(f"Attached RGB image: {Path(filename).name}")
        except Exception as exc:
            QMessageBox.critical(self, "RGB image error", str(exc))

    def _prepare_new_source(self) -> None:
        self.raw_frame = None
        self.enhancement_processor.reset()
        self.enhancement_processor.cancel_background_capture()
        self._display_xyz = None
        self._display_rgb = None
        self._height_mm = None
        self._object_mask = None
        self._rgb_candidate_mask = None
        self._detected_object_count = 0
        self._display_transform = np.eye(4, dtype=np.float64)
        self._inverse_display_transform = np.eye(4, dtype=np.float64)
        self.box_origins = None
        self.box_normals = None
        self._camera_was_reset = False
        self._last_selection_mask = None
        self._cloud_bounds = None
        self.export_origin_m = np.zeros(3, dtype=np.float64)
        self.export_rotation = np.eye(3, dtype=np.float64)
        self._normal_result = None
        self._origin_internal_update = True
        for spin in self.origin_spins:
            spin.setValue(0.0)
        for spin in self.origin_rotation_spins:
            spin.setValue(0.0)
        self._origin_internal_update = False
        if self._origin_widget is not None and hasattr(self._origin_widget, "remove"):
            self._origin_widget.remove()
        self._origin_widget = None
        if self._crop_affine_widget is not None and hasattr(self._crop_affine_widget, "remove"):
            self._crop_affine_widget.remove()
        if self._crop_resize_widget is not None:
            self._crop_resize_widget.remove()
        self._crop_resize_widget = None
        self._crop_affine_widget = None
        self._box_widget = None
        self._crop_proxy_actor = None
        for name in (
            "source-mesh", "selection-preview", "detected-depth-objects",
            "detected-rgb-candidates", "export-origin-marker",
            "export-axis-x", "export-axis-y", "export-axis-z", "crop-gizmo-proxy",
            "surface-normals",
        ):
            self.plotter.remove_actor(name, render=False)
        self.plotter.clear_box_widgets()
        self.plotter.remove_bounds_axes()


    def _set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        self.freeze_button.setText("Resume stream" if self.frozen else "Freeze frame")
        if self.frame is not None:
            message = "Frame frozen." if self.frozen else "Live stream resumed."
            self.statusBar().showMessage(message)

    def _poll_camera(self) -> None:
        if self.frozen or not self.session.running:
            return
        try:
            frame = self.session.poll()
            if frame is not None:
                if self.enhancement_processor.background_capture_active:
                    current, target, complete = self.enhancement_processor.add_background_frame(
                        frame
                    )
                    self.background_status.setText(
                        f"Capturing empty table: {current} / {target}. "
                        "Keep the camera and table still."
                    )
                    if complete:
                        self._activate_background_controls(current)
                self._accept_frame(frame, reset_box=self.box_origins is None)
        except Exception as exc:
            self._stop_camera()
            QMessageBox.critical(self, "Frame capture error", str(exc))

    def _accept_frame(self, frame: CaptureFrame, reset_box: bool = False) -> None:
        self.raw_frame = frame
        result = self.enhancement_processor.process(
            frame,
            self._current_enhancement_settings(),
            update_history=self.session.running and not self.frozen,
        )
        self._accept_enhancement_result(result, reset_box=reset_box)

    def _accept_enhancement_result(
        self, result: EnhancementResult, reset_box: bool = False
    ) -> None:
        self.frame = result.frame
        self._display_xyz = result.display_xyz
        self._display_rgb = result.display_rgb
        self._height_mm = result.height_mm
        self._object_mask = result.object_mask
        self._rgb_candidate_mask = result.rgb_candidate_mask
        self._detected_object_count = result.object_count
        self._display_transform = result.display_transform
        self._inverse_display_transform = result.inverse_display_transform
        self._update_images(result.frame)
        self._update_cloud(result.frame)
        self._update_detected_object_preview()
        if reset_box or self.box_origins is None:
            self._reset_box()
        elif self._preview_clock.elapsed() > 180:
            self._update_selection_preview()
            self._preview_clock.restart()

    def _refresh_images(self) -> None:
        if self.frame is not None:
            self._update_images(self.frame)

    def _refresh_current_frame(self) -> None:
        if self.frame is not None:
            self._update_cloud(self.frame)
            self._update_images(self.frame)
            self._update_detected_object_preview()
            self._update_selection_preview()

    def _update_images(self, frame: CaptureFrame) -> None:
        if not frame.organized or frame.xyz.shape[0] == 1:
            rgb_preview, depth_preview = self._project_unorganized_previews(frame)
            self._set_rgb_pixmap(self.rgb_label, rgb_preview)
            self._set_rgb_pixmap(self.depth_label, depth_preview)
            return
        self._set_rgb_pixmap(self.rgb_label, self._annotated_rgb(frame.rgb))
        if self.height_color_check.isChecked() and self._height_mm is not None:
            minimum = self.height_min.value()
            maximum = max(self.height_max.value(), minimum + 0.1)
            valid_height = np.isfinite(self._height_mm)
            scaled_height = np.clip(
                (np.nan_to_num(self._height_mm, nan=minimum) - minimum)
                / (maximum - minimum)
                * 255.0,
                0,
                255,
            ).astype(np.uint8)
            try:
                import cv2

                height_bgr = cv2.applyColorMap(scaled_height, cv2.COLORMAP_TURBO)
                height_rgb = height_bgr[:, :, ::-1].copy()
            except ImportError:
                height_rgb = np.repeat(scaled_height[:, :, None], 3, axis=2)
            height_rgb[~valid_height] = 0
            self._set_rgb_pixmap(self.depth_label, height_rgb)
            return
        depth = frame.depth_mm.astype(np.float32)
        max_mm = self.max_depth.value() * 1000.0
        gray = np.clip(depth / max_mm * 255.0, 0, 255).astype(np.uint8)
        try:
            import cv2

            depth_bgr = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
            depth_rgb = depth_bgr[:, :, ::-1].copy()
        except ImportError:
            depth_rgb = np.repeat(gray[:, :, None], 3, axis=2)
        depth_rgb[depth <= 0] = 0
        self._set_rgb_pixmap(self.depth_label, depth_rgb)

    def _annotated_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if not self.show_detected_check.isChecked():
            return rgb
        if self._object_mask is None and self._rgb_candidate_mask is None:
            return rgb
        try:
            import cv2
        except ImportError:
            return rgb
        result = rgb.copy()
        depth_mask = (
            np.zeros(rgb.shape[:2], dtype=bool)
            if self._object_mask is None
            else self._object_mask
        )
        rgb_mask = (
            np.zeros(rgb.shape[:2], dtype=bool)
            if self._rgb_candidate_mask is None
            else self._rgb_candidate_mask & ~depth_mask
        )
        for mask, color, label in (
            (depth_mask, (0, 255, 0), "3D"),
            (rgb_mask, (255, 0, 255), "RGB"),
        ):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    result,
                    label,
                    (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return result

    def _update_detected_object_preview(self) -> None:
        self.plotter.remove_actor("detected-depth-objects", render=False)
        self.plotter.remove_actor("detected-rgb-candidates", render=False)
        if (
            self.frame is None
            or not self.show_detected_check.isChecked()
            or self._display_xyz is None
        ):
            return
        if self._object_mask is not None:
            mask = self._object_mask & self.frame.valid_mask
            if mask.any():
                self.plotter.add_points(
                    pv.PolyData(self._display_xyz[mask]),
                    name="detected-depth-objects",
                    color="#30d158",
                    point_size=self.point_size_spin.value() + 2.0,
                    render_points_as_spheres=False,
                    reset_camera=False,
                )
        if self._rgb_candidate_mask is not None:
            mask = self._rgb_candidate_mask & self.frame.valid_mask
            if self._object_mask is not None:
                mask &= ~self._object_mask
            if mask.any():
                self.plotter.add_points(
                    pv.PolyData(self._display_xyz[mask]),
                    name="detected-rgb-candidates",
                    color="#ff2dff",
                    point_size=self.point_size_spin.value() + 1.0,
                    render_points_as_spheres=False,
                    reset_camera=False,
                )

    def _project_unorganized_previews(
        self, frame: CaptureFrame, width: int = 480, height: int = 360
    ) -> tuple[np.ndarray, np.ndarray]:
        source_xyz = self._display_xyz if self._display_xyz is not None else frame.xyz
        source_rgb = self._display_rgb if self._display_rgb is not None else frame.rgb
        points = source_xyz[frame.valid_mask]
        colors = source_rgb[frame.valid_mask]
        rgb_canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        depth_canvas = np.full_like(rgb_canvas, 18)
        if not len(points):
            return rgb_canvas, depth_canvas
        x, y, z = points.T
        x_low, x_high = np.percentile(x, (1, 99))
        y_low, y_high = np.percentile(y, (1, 99))
        if x_high <= x_low:
            x_low, x_high = float(x.min()), float(x.max()) + 1.0
        if y_high <= y_low:
            y_low, y_high = float(y.min()), float(y.max()) + 1.0
        u = np.clip((x - x_low) / (x_high - x_low) * (width - 1), 0, width - 1).astype(int)
        v = np.clip((y - y_low) / (y_high - y_low) * (height - 1), 0, height - 1).astype(int)
        order = np.argsort(z)[::-1]
        rgb_canvas[v[order], u[order]] = colors[order]
        z_low, z_high = np.percentile(z, (2, 98))
        scaled = np.clip((z - z_low) / max(float(z_high - z_low), 1e-9) * 255, 0, 255).astype(np.uint8)
        try:
            import cv2

            depth_colors = cv2.applyColorMap(
                (255 - scaled).reshape(-1, 1), cv2.COLORMAP_TURBO
            )[:, 0, ::-1]
            kernel = np.ones((3, 3), dtype=np.uint8)
            rgb_canvas = cv2.dilate(rgb_canvas, kernel)
        except ImportError:
            depth_colors = np.repeat(scaled[:, None], 3, axis=1)
        depth_canvas[v[order], u[order]] = depth_colors[order]
        try:
            depth_canvas = cv2.dilate(depth_canvas, kernel)
        except (ImportError, UnboundLocalError):
            pass
        return rgb_canvas, depth_canvas

    @staticmethod
    def _set_rgb_pixmap(label: QLabel, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def _display_arrays(self, frame: CaptureFrame) -> tuple[np.ndarray, np.ndarray]:
        step = self.display_step.value()
        source_xyz = self._display_xyz if self._display_xyz is not None else frame.xyz
        source_rgb = self._display_rgb if self._display_rgb is not None else frame.rgb
        xyz = source_xyz[::step, ::step]
        rgb = source_rgb[::step, ::step]
        valid = frame.valid_mask[::step, ::step]
        return xyz[valid], rgb[valid]

    def _update_cloud(self, frame: CaptureFrame) -> None:
        points, colors = self._display_arrays(frame)
        if not len(points):
            return
        cloud = pv.PolyData(points)
        self._cloud_bounds = tuple(float(value) for value in cloud.bounds)
        cloud["RGB"] = colors
        self.plotter.add_points(
            cloud,
            scalars="RGB",
            rgb=True,
            name="point-cloud",
            point_size=self.point_size_spin.value(),
            render_points_as_spheres=False,
            reset_camera=not self._camera_was_reset,
        )
        if frame.source_faces is not None and len(frame.source_faces):
            source_xyz = self._display_xyz if self._display_xyz is not None else frame.xyz
            source_rgb = self._display_rgb if self._display_rgb is not None else frame.rgb
            all_points = source_xyz.reshape(-1, 3)
            face_prefix = np.full((len(frame.source_faces), 1), 3, dtype=np.int64)
            face_data = np.hstack((face_prefix, frame.source_faces)).ravel()
            source_mesh = pv.PolyData(all_points, face_data)
            source_mesh["RGB"] = source_rgb.reshape(-1, 3)
            self.plotter.add_mesh(
                source_mesh,
                scalars="RGB",
                rgb=True,
                name="source-mesh",
                show_edges=False,
                smooth_shading=False,
                reset_camera=False,
            )
        else:
            self.plotter.remove_actor("source-mesh", render=False)
        if not self._camera_was_reset:
            self.plotter.show_grid(
                bounds=self._cloud_bounds,
                color="#58606b",
                location="outer",
                all_edges=True,
            )
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self._camera_was_reset = True

    def _reset_box(self) -> None:
        if self.frame is None:
            return
        source_xyz = self._display_xyz if self._display_xyz is not None else self.frame.xyz
        points = source_xyz[self.frame.valid_mask]
        if not len(points):
            return
        bounds = list(finite_bounds(points))
        # Start around the centre, leaving room to drag the handles outward.
        for axis in range(3):
            lo_index = axis * 2
            hi_index = lo_index + 1
            centre = (bounds[lo_index] + bounds[hi_index]) * 0.5
            half = (bounds[hi_index] - bounds[lo_index]) * 0.34
            bounds[lo_index], bounds[hi_index] = centre - half, centre + half

        self.plotter.clear_box_widgets()
        if self._crop_resize_widget is not None:
            self._crop_resize_widget.remove()
        self._crop_resize_widget = None
        if self._crop_affine_widget is not None and hasattr(self._crop_affine_widget, "remove"):
            self._crop_affine_widget.remove()
        self._crop_affine_widget = None
        self.plotter.remove_actor("crop-gizmo-proxy", render=False)
        self._crop_proxy_actor = None
        self.box_origins = None
        self.box_normals = None
        self._box_widget = self.plotter.add_box_widget(
            callback=self._on_box_changed,
            bounds=tuple(bounds),
            factor=1.0,
            rotation_enabled=False,
            color="#ffd43b",
            use_planes=True,
            outline_translation=False,
            pass_widget=True,
            interaction_event="end",
        )
        self._box_widget.ScalingEnabledOn()
        self._box_widget.HandlesOff()
        self._box_widget.PickingManagedOff()
        self._box_widget.SetPriority(0.99)
        proxy = pv.Box(bounds=tuple(bounds))
        self._crop_proxy_actor = self.plotter.add_mesh(
            proxy,
            name="crop-gizmo-proxy",
            color="#ffd43b",
            style="wireframe",
            line_width=2.0,
            opacity=0.8,
            pickable=False,
            reset_camera=False,
        )
        self._crop_proxy_actor.use_bounds = False
        self._crop_resize_widget = FaceResizeWidget(
            self.plotter,
            self._box_widget,
            self._crop_proxy_actor,
            tuple(bounds),
            self._on_custom_resize_released,
        )
        self._rebuild_crop_affine_gizmo()
        self.plotter.render()

    def _on_box_changed(self, planes: object, widget: object | None = None) -> None:
        plane_points = planes.GetPoints()
        plane_normals = planes.GetNormals()
        count = planes.GetNumberOfPlanes()
        self.box_origins = np.asarray(
            [plane_points.GetPoint(index) for index in range(count)], dtype=np.float64
        )
        self.box_normals = np.asarray(
            [plane_normals.GetTuple(index) for index in range(count)], dtype=np.float64
        )
        self._update_selection_preview()
        if self.show_normals_check.isChecked():
            self.normal_status.setText(
                "The crop changed. Click Recompute and display normals to refresh the arrows."
            )
        if widget is not None and not self._crop_gizmo_internal_update:
            self._sync_crop_proxy_from_box(widget)

    def _sync_crop_proxy_from_box(self, widget: object) -> None:
        if self._crop_proxy_actor is None:
            return
        import vtk

        transform = vtk.vtkTransform()
        widget.GetTransform(transform)
        matrix = pv.array_from_vtkmatrix(transform.GetMatrix())
        self._crop_proxy_actor.user_matrix = matrix
        if self._crop_affine_widget is None:
            self._rebuild_crop_affine_gizmo()
            return
        self._crop_affine_widget.origin = tuple(self._crop_proxy_actor.center)
        self._crop_affine_widget.axes = orthonormalize_rotation(matrix[:3, :3]).T
        self._crop_affine_widget._cached_matrix = matrix.copy()

    def _rebuild_crop_affine_gizmo(self) -> None:
        if self._crop_affine_widget is not None and hasattr(self._crop_affine_widget, "remove"):
            self._crop_affine_widget.remove()
        self._crop_affine_widget = None
        if self._crop_proxy_actor is None:
            return
        self._crop_affine_widget = self.plotter.add_affine_transform_widget(
            self._crop_proxy_actor,
            origin=self._crop_proxy_actor.center,
            start=False,
            scale=0.18,
            line_radius=0.025,
            axes_colors=("#ff3b30", "#34c759", "#0a84ff"),
            release_callback=self._on_crop_gizmo_released,
        )
        self._apply_interaction_tool()

    def _on_crop_gizmo_released(self, matrix: np.ndarray) -> None:
        if self._box_widget is None:
            return
        import vtk

        transform = vtk.vtkTransform()
        transform.SetMatrix(pv.vtkmatrix_from_array(np.asarray(matrix, dtype=np.float64)))
        self._crop_gizmo_internal_update = True
        try:
            self._box_widget.SetTransform(transform)
            planes = vtk.vtkPlanes()
            self._box_widget.GetPlanes(planes)
            self._on_box_changed(planes)
        finally:
            self._crop_gizmo_internal_update = False
        self._sync_crop_proxy_from_box(self._box_widget)
        self.plotter.render()

    def _on_custom_resize_released(self, planes: object) -> None:
        self._on_box_changed(planes)
        if self._box_widget is not None:
            self._sync_crop_proxy_from_box(self._box_widget)

    @staticmethod
    def _set_affine_widget_active(widget: object | None, active: bool) -> None:
        if widget is None:
            return
        if hasattr(widget, "disable"):
            widget.disable()
        actors = list(getattr(widget, "_arrows", [])) + list(getattr(widget, "_circles", []))
        for actor in actors:
            actor.SetVisibility(active)
        if active and hasattr(widget, "enable"):
            widget.enable()

    def _set_interaction_tool(self, mode: str) -> None:
        if not hasattr(self, "interaction_tool_combo"):
            return
        index = self.interaction_tool_combo.findData(mode)
        if index < 0:
            raise ValueError(f"Unknown interaction tool: {mode}")
        if self.interaction_tool_combo.currentIndex() == index:
            self._on_interaction_tool_changed()
        else:
            self.interaction_tool_combo.setCurrentIndex(index)

    def _on_interaction_tool_changed(self) -> None:
        mode = str(self.interaction_tool_combo.currentData())
        for key, button in self.interaction_tool_buttons.items():
            button.setChecked(key == mode)
        self._apply_interaction_tool()

    def _apply_interaction_tool(self) -> None:
        if not hasattr(self, "plotter"):
            return
        mode = str(self.interaction_tool_combo.currentData())
        self._set_affine_widget_active(self._crop_affine_widget, False)
        self._set_affine_widget_active(self._origin_widget, False)
        if self._crop_resize_widget is not None:
            self._crop_resize_widget.disable()
        if self._box_widget is not None:
            self._box_widget.Off()
        if self._crop_proxy_actor is not None:
            self._crop_proxy_actor.SetVisibility(True)
        for name in ("export-axis-x", "export-axis-y", "export-axis-z"):
            actor = self.plotter.renderer.actors.get(name)
            if actor is not None:
                actor.SetVisibility(mode != "export_frame")
        if mode == "crop_resize" and self._crop_resize_widget is not None:
            self._crop_resize_widget.enable()
        elif mode == "crop_transform":
            self._set_affine_widget_active(self._crop_affine_widget, True)
        elif mode == "export_frame":
            self._set_affine_widget_active(self._origin_widget, True)
        self.plotter.render()

    def _current_origin_mode(self) -> str:
        return str(self.origin_mode_combo.currentData())

    def _on_origin_mode_changed(self) -> None:
        if self._origin_internal_update:
            return
        mode = self._current_origin_mode()
        if self.frame is None:
            return
        if mode == "manual":
            self._set_interaction_tool("export_frame")
            self._render_origin_controls(manual=True, rebuild_widget=True)
            return
        self._recalculate_automatic_origin()

    def _on_origin_coordinate_edited(self) -> None:
        if self._origin_internal_update:
            return
        origin = np.asarray([spin.value() for spin in self.origin_spins], dtype=np.float64) / 1000.0
        if self._current_origin_mode() != "manual":
            manual_index = self.origin_mode_combo.findData("manual")
            self._origin_internal_update = True
            self.origin_mode_combo.setCurrentIndex(manual_index)
            self._origin_internal_update = False
        self._set_interaction_tool("export_frame")
        self._set_export_origin(origin, move_widget=True)
        self._render_origin_controls(manual=True, rebuild_widget=True)

    def _on_origin_rotation_edited(self) -> None:
        if self._origin_internal_update:
            return
        angles = np.asarray([spin.value() for spin in self.origin_rotation_spins])
        self.export_rotation = euler_xyz_degrees_to_matrix(angles)
        self._set_interaction_tool("export_frame")
        self._render_origin_controls(
            manual=self._current_origin_mode() == "manual", rebuild_widget=True
        )

    def _reset_export_orientation(self) -> None:
        self.export_rotation = np.eye(3, dtype=np.float64)
        self._sync_origin_rotation_spins()
        self._set_interaction_tool("export_frame")
        self._render_origin_controls(
            manual=self._current_origin_mode() == "manual", rebuild_widget=True
        )

    def _sync_origin_rotation_spins(self) -> None:
        angles = matrix_to_euler_xyz_degrees(self.export_rotation)
        self._origin_internal_update = True
        for spin, angle in zip(self.origin_rotation_spins, angles, strict=True):
            spin.setValue(float(angle))
        self._origin_internal_update = False

    def _origin_points(self) -> np.ndarray:
        if self.frame is None:
            return np.empty((0, 3), dtype=np.float32)
        mask = self._last_selection_mask
        if mask is None or not mask.any():
            return self.frame.valid_points
        return self.frame.xyz[mask]

    def _recalculate_automatic_origin(self) -> None:
        mode = self._current_origin_mode()
        if mode == "manual" or self.frame is None:
            return
        try:
            origin = calculate_origin_anchor(self._origin_points(), mode)
            if (
                self._origin_widget is not None
                and np.allclose(origin, self.export_origin_m, atol=1e-10, rtol=0.0)
            ):
                return
            self._set_export_origin(origin, move_widget=False)
            self._render_origin_controls(manual=False, rebuild_widget=True)
        except ValueError:
            return

    def _set_export_origin(self, origin_m: np.ndarray, move_widget: bool) -> None:
        origin = np.asarray(origin_m, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Export origin must contain three finite coordinates")
        self.export_origin_m = origin.copy()
        self._origin_internal_update = True
        for spin, value_m in zip(self.origin_spins, origin, strict=True):
            spin.setValue(float(value_m * 1000.0))
        self._origin_internal_update = False

    def _true_to_display_point(self, point: np.ndarray) -> np.ndarray:
        return transform_points(np.asarray(point, dtype=np.float64)[None, :], self._display_transform)[0]

    def _display_to_true_point(self, point: np.ndarray) -> np.ndarray:
        return transform_points(
            np.asarray(point, dtype=np.float64)[None, :], self._inverse_display_transform
        )[0]

    def _origin_visual_scale(self) -> tuple[float, float]:
        if self.frame is None:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            source_xyz = self._display_xyz if self._display_xyz is not None else self.frame.xyz
            mask = self._last_selection_mask
            points = source_xyz[mask] if mask is not None and mask.any() else source_xyz[self.frame.valid_mask]
        if len(points):
            diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        else:
            diagonal = 0.1
        axis_length = max(diagonal * 0.16, 0.01)
        return axis_length, max(axis_length * 0.075, 0.0015)

    def _render_origin_controls(self, manual: bool, rebuild_widget: bool) -> None:
        axis_length, radius = self._origin_visual_scale()
        self._origin_axis_length_m = axis_length
        if rebuild_widget:
            if self._origin_widget is not None and hasattr(self._origin_widget, "remove"):
                self._origin_widget.remove()
            self._origin_widget = None
            self.plotter.remove_actor("export-origin-marker", render=False)
            centre = self._true_to_display_point(self.export_origin_m)
            marker = pv.Sphere(radius=radius, center=centre)
            actor = self.plotter.add_mesh(
                marker,
                name="export-origin-marker",
                color="#00d9ff",
                opacity=0.65,
                pickable=False,
                reset_camera=False,
            )
            actor.use_bounds = False
            self._origin_proxy_actor = actor
            display_axes = self._display_export_axes()
            self._origin_gizmo_base = (centre.copy(), display_axes.copy())
            self._origin_widget = self.plotter.add_affine_transform_widget(
                actor,
                origin=tuple(float(value) for value in centre),
                start=False,
                scale=2.0,
                line_radius=0.025,
                axes_colors=("#ff3b30", "#34c759", "#0a84ff"),
                axes=display_axes.T,
                release_callback=self._on_origin_gizmo_released,
            )
        self._draw_origin_axes()
        self._apply_interaction_tool()
        self.plotter.render()

    def _display_export_axes(self) -> np.ndarray:
        origin = self._true_to_display_point(self.export_origin_m)
        directions = np.empty((3, 3), dtype=np.float64)
        for index in range(3):
            endpoint = self._true_to_display_point(
                self.export_origin_m + self.export_rotation[:, index]
            )
            direction = endpoint - origin
            directions[:, index] = direction / max(np.linalg.norm(direction), 1e-12)
        return orthonormalize_rotation(directions)

    def _on_origin_gizmo_released(self, matrix: np.ndarray) -> None:
        if self._origin_internal_update or self._origin_gizmo_base is None:
            return
        centre, display_axes = self._origin_gizmo_base
        samples = np.vstack((centre, centre + display_axes.T))
        moved = transform_points(samples, np.asarray(matrix, dtype=np.float64))
        true_samples = transform_points(moved, self._inverse_display_transform)
        new_origin = true_samples[0].astype(np.float64)
        new_axes = (true_samples[1:] - new_origin).T
        self.export_rotation = orthonormalize_rotation(new_axes)
        if self._current_origin_mode() != "manual":
            manual_index = self.origin_mode_combo.findData("manual")
            self._origin_internal_update = True
            self.origin_mode_combo.setCurrentIndex(manual_index)
            self._origin_internal_update = False
        self._set_export_origin(new_origin, move_widget=False)
        self._sync_origin_rotation_spins()
        if self._origin_widget is not None:
            self._origin_widget.origin = tuple(self._true_to_display_point(new_origin))
            self._origin_widget.axes = self._display_export_axes().T
        if str(self.interaction_tool_combo.currentData()) != "export_frame":
            self._draw_origin_axes()
        self.plotter.render()

    def _draw_origin_axes(self) -> None:
        origin = self._true_to_display_point(self.export_origin_m)
        length = self._origin_axis_length_m
        for index, (name, color) in enumerate((
            ("export-axis-x", "#ff3b30"),
            ("export-axis-y", "#34c759"),
            ("export-axis-z", "#0a84ff"),
        )):
            display_end = self._true_to_display_point(
                self.export_origin_m + self.export_rotation[:, index]
            )
            display_direction = display_end - origin
            direction_length = np.linalg.norm(display_direction)
            if direction_length > 1e-12:
                display_direction /= direction_length
            arrow = pv.Arrow(start=origin, direction=display_direction, scale=length)
            actor = self.plotter.add_mesh(
                arrow,
                name=name,
                color=color,
                pickable=False,
                reset_camera=False,
            )
            actor.use_bounds = False

    def _update_normals_if_visible(self) -> None:
        if self.show_normals_check.isChecked():
            self._update_normals()

    def _update_normals(self) -> None:
        self.plotter.remove_actor("surface-normals", render=False)
        if self.frame is None:
            self.normal_status.setText("Open or capture 3D data first.")
            return
        if not self.show_normals_check.isChecked() and not self.export_normals_check.isChecked():
            self._normal_result = None
            self.normal_status.setText("Normal display and export are disabled.")
            self.plotter.render()
            return
        try:
            selected_mask = None
            if self.normals_selected_only_check.isChecked():
                selected_mask = self._selection_mask(self.frame)
            camera_position = np.asarray(self.plotter.camera.position, dtype=np.float64)
            viewpoint = self._display_to_true_point(camera_position)
            result = estimate_normals(
                self.frame,
                method=str(self.normal_method_combo.currentData()),
                orientation=str(self.normal_orientation_combo.currentData()),
                neighbors=self.normal_neighbors_spin.value(),
                max_edge_m=self.edge_spin.value() / 1000.0,
                selected_mask=selected_mask,
                viewpoint=viewpoint,
            )
            self._normal_result = result
            active = result.valid_mask.copy()
            if selected_mask is not None:
                active &= selected_mask
            count = int(active.sum())
            self.normal_status.setText(
                f"{count:,} normals computed with {result.method}; direction: {result.orientation}."
            )
            if not self.show_normals_check.isChecked() or count == 0:
                self.plotter.render()
                return
            indices = np.flatnonzero(active)
            maximum = self.normal_count_spin.value()
            if len(indices) > maximum:
                indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]
            source_points = self.frame.xyz.reshape(-1, 3)[indices]
            display_points = transform_points(source_points, self._display_transform)
            source_normals = result.normals.reshape(-1, 3)[indices].astype(np.float64)
            linear = self._display_transform[:3, :3]
            display_normals = source_normals @ np.linalg.inv(linear)
            lengths = np.linalg.norm(display_normals, axis=1)
            good = np.isfinite(display_normals).all(axis=1) & (lengths > 1e-10)
            display_points = display_points[good]
            display_normals = display_normals[good] / lengths[good, None]
            glyph_points = pv.PolyData(display_points)
            glyph_points["normal_vectors"] = (
                display_normals * (self.normal_length_spin.value() / 1000.0)
            )
            arrows = glyph_points.glyph(
                orient="normal_vectors",
                scale="normal_vectors",
                factor=1.0,
                geom=pv.Arrow(tip_resolution=8, shaft_resolution=8),
            )
            actor = self.plotter.add_mesh(
                arrows,
                name="surface-normals",
                color="#ffea00",
                lighting=False,
                pickable=False,
                reset_camera=False,
            )
            actor.use_bounds = False
            self.plotter.render()
        except Exception as exc:
            self._normal_result = None
            self.normal_status.setText(f"Normal calculation failed: {exc}")
            self.plotter.render()

    def _selection_mask(self, frame: CaptureFrame) -> np.ndarray:
        if self.box_origins is None or self.box_normals is None:
            raise ValueError("Bounding box is not ready")
        valid = frame.valid_mask
        source_xyz = self._display_xyz if self._display_xyz is not None else frame.xyz
        points = source_xyz[valid]
        if not len(points):
            return np.zeros(valid.shape, dtype=bool)
        selected_valid = points_inside_planes(
            points, self.box_origins, self.box_normals, tolerance_m=1e-7
        )
        selected = np.zeros(valid.shape, dtype=bool)
        selected[valid] = selected_valid
        return selected

    def _update_selection_preview(self) -> None:
        if self.frame is None or self.box_origins is None:
            return
        try:
            mask = self._selection_mask(self.frame)
            self._last_selection_mask = mask
            count = int(mask.sum())
            self.selection_label.setText(f"{count:,} points")

            step = self.display_step.value()
            sample_mask = mask[::step, ::step]
            source_xyz = self._display_xyz if self._display_xyz is not None else self.frame.xyz
            xyz = source_xyz[::step, ::step]
            if sample_mask.any() and self.selection_preview_check.isChecked():
                preview = pv.PolyData(xyz[sample_mask])
                self.plotter.add_points(
                    preview,
                    name="selection-preview",
                    color="#ff3b30",
                    point_size=6,
                    render_points_as_spheres=False,
                    reset_camera=False,
                )
            else:
                self.plotter.remove_actor("selection-preview", render=False)
            if self._current_origin_mode() != "manual":
                self._recalculate_automatic_origin()
            self.plotter.render()
        except Exception:
            self.selection_label.setText("Selection calculation failed")

    def _choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Output root folder", self.output_edit.text()
        )
        if folder:
            self.output_edit.setText(folder)

    def _save_selection(self) -> None:
        if self.frame is None or self.box_origins is None or self.box_normals is None:
            QMessageBox.information(
                self, "No frame", "Open a file, connect a camera, or start Demo Mode first."
            )
            return
        was_frozen = self.frozen
        self.frozen = True
        self.save_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mask = self._selection_mask(self.frame)
            if self.export_normals_check.isChecked():
                self._update_normals()
            normal_result = self._normal_result if self.export_normals_check.isChecked() else None
            result = export_selection(
                Path(self.output_edit.text()),
                self.frame,
                mask,
                {
                    "plane_origins": self.box_origins.tolist(),
                    "plane_normals": self.box_normals.tolist(),
                },
                self.edge_spin.value() / 1000.0,
                export_origin_m=self.export_origin_m,
                origin_mode=self._current_origin_mode(),
                export_rotation=self.export_rotation,
                normals=None if normal_result is None else normal_result.normals,
                normal_metadata=None if normal_result is None else {
                    "method": normal_result.method,
                    "orientation": normal_result.orientation,
                    "invalid_vectors_are_zero": True,
                    "coordinate_frame": "export coordinate frame",
                },
            )
            self.statusBar().showMessage(f"Export saved: {result.folder}", 15000)
            QMessageBox.information(
                self,
                "Export complete",
                f"Saved {result.point_count:,} points and {result.triangle_count:,} triangles.\n\n"
                f"Folder:\n{result.folder}",
            )
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            QMessageBox.critical(self, "Export error", detail)
        finally:
            QApplication.restoreOverrideCursor()
            self.save_button.setEnabled(True)
            self.frozen = was_frozen

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        try:
            self.session.stop()
        finally:
            self.plotter.close()
        event.accept()
