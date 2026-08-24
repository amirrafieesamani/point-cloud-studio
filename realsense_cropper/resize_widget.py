from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, Qt
from vtkmodules.vtkCommonDataModel import vtkPlanes
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingCore import vtkPropPicker


class FaceResizeWidget(QObject):
    """Mouse-driven oriented-box resizing with six explicitly pickable face handles."""

    def __init__(
        self,
        plotter: object,
        box_widget: object,
        proxy_actor: object,
        base_bounds: tuple[float, float, float, float, float, float],
        release_callback: Callable[[object], None],
    ) -> None:
        super().__init__()
        self.plotter = plotter
        self.box_widget = box_widget
        self.proxy_actor = proxy_actor
        self.release_callback = release_callback
        bounds = np.asarray(base_bounds, dtype=np.float64)
        self.base_center = np.array(
            [(bounds[0] + bounds[1]) * 0.5,
             (bounds[2] + bounds[3]) * 0.5,
             (bounds[4] + bounds[5]) * 0.5]
        )
        self.base_half_sizes = np.array(
            [(bounds[1] - bounds[0]) * 0.5,
             (bounds[3] - bounds[2]) * 0.5,
             (bounds[5] - bounds[4]) * 0.5]
        )
        self.handle_actors: list[object] = []
        self._enabled = False
        self._selected_index: int | None = None
        self._hovered_index: int | None = None
        self._drag_state: dict[str, np.ndarray | float | int] | None = None
        self._picker = vtkPropPicker()
        self._picker.PickFromListOn()
        self._create_handles()
        self.disable()

    def _box_polydata(self) -> pv.PolyData:
        poly = pv.PolyData()
        self.box_widget.GetPolyData(poly)
        return poly

    def _create_handles(self) -> None:
        poly = self._box_polydata()
        diagonal = max(float(np.linalg.norm(np.ptp(poly.points[:8], axis=0))), 0.01)
        radius = max(diagonal * 0.026, 0.0018)
        geometry = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20)
        for index, point in enumerate(poly.points[8:14]):
            actor = self.plotter.add_mesh(
                geometry.copy(),
                name=f"crop-resize-face-{index}",
                color="#f5f5f5",
                smooth_shading=True,
                pickable=True,
                reset_camera=False,
            )
            actor.position = tuple(float(value) for value in point)
            actor.use_bounds = False
            self.handle_actors.append(actor)
            self._picker.AddPickList(actor)

    def _set_handle_color(self, index: int, color: str) -> None:
        self.handle_actors[index].prop.color = color

    def update_from_box(self) -> None:
        poly = self._box_polydata()
        for actor, point in zip(self.handle_actors, poly.points[8:14], strict=True):
            actor.position = tuple(float(value) for value in point)

    def enable(self) -> None:
        self.disable()
        self.update_from_box()
        for actor in self.handle_actors:
            actor.SetVisibility(True)
        self.plotter.interactor.installEventFilter(self)
        self._enabled = True

    def disable(self) -> None:
        if self._enabled:
            self.plotter.interactor.removeEventFilter(self)
        self._enabled = False
        self._selected_index = None
        self._drag_state = None
        for actor in self.handle_actors:
            actor.SetVisibility(False)

    def eventFilter(self, watched: object, event: object) -> bool:
        if not self._enabled or watched is not self.plotter.interactor:
            return False
        event_type = event.type()
        if event_type not in {
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        }:
            return False
        if event_type != QEvent.Type.MouseMove and event.button() != Qt.MouseButton.LeftButton:
            return False
        widget = self.plotter.interactor
        render_width, render_height = self.plotter.render_window.GetSize()
        position = event.position()
        x = int(round(position.x() * render_width / max(widget.width(), 1)))
        y = int(round((widget.height() - position.y()) * render_height / max(widget.height(), 1)))
        interactor = self.plotter.iren.interactor
        interactor.SetEventPosition(x, y)
        if event_type == QEvent.Type.MouseButtonPress:
            self._on_left_press(interactor, None)
            return self._selected_index is not None
        if event_type == QEvent.Type.MouseMove:
            was_dragging = self._selected_index is not None
            self._on_mouse_move(interactor, None)
            return was_dragging or self._selected_index is not None
        was_dragging = self._selected_index is not None
        self._on_left_release(interactor, None)
        return was_dragging

    def remove(self) -> None:
        self.disable()
        for index in range(len(self.handle_actors)):
            self.plotter.remove_actor(f"crop-resize-face-{index}", render=False)
        self.handle_actors = []

    def _pick_handle(self, interactor: object) -> int | None:
        x, y = interactor.GetEventPosition()
        renderer = self.plotter.iren.get_poked_renderer()
        self._picker.Pick(x, y, 0.0, renderer)
        actor = self._picker.GetActor()
        for index, candidate in enumerate(self.handle_actors):
            if candidate is actor:
                return index
        return None

    def _project(self, point: np.ndarray) -> np.ndarray:
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        return np.asarray(renderer.GetDisplayPoint()[:2], dtype=np.float64)

    def _on_left_press(self, interactor: object, _event: object) -> None:
        index = self._pick_handle(interactor)
        if index is None:
            return
        poly = self._box_polydata()
        negative = poly.points[[8, 10, 12]].astype(np.float64)
        positive = poly.points[[9, 11, 13]].astype(np.float64)
        axes = (positive - negative).T
        lengths = np.linalg.norm(axes, axis=0)
        axes /= lengths[None, :]
        half_sizes = lengths * 0.5
        axis_index = index // 2
        face_point = poly.points[8 + index].astype(np.float64)
        opposite_point = poly.points[8 + (index ^ 1)].astype(np.float64)
        axis = axes[:, axis_index] * (-1.0 if index % 2 == 0 else 1.0)
        reference = max(float(np.linalg.norm(poly.points[6] - poly.points[0])) * 0.25, 0.001)
        projected = self._project(face_point + axis * reference) - self._project(face_point)
        projected_length = float(np.linalg.norm(projected))
        if projected_length < 4.0:
            screen_direction = np.array([0.0, 1.0])
            pixels_per_metre = 220.0 / max(reference * 4.0, 0.001)
        else:
            screen_direction = projected / projected_length
            pixels_per_metre = projected_length / reference
        self._selected_index = index
        self._drag_state = {
            "mouse": np.asarray(interactor.GetEventPosition(), dtype=np.float64),
            "face": face_point,
            "opposite": opposite_point,
            "axis": axis,
            "axes": axes,
            "half_sizes": half_sizes,
            "axis_index": axis_index,
            "screen_direction": screen_direction,
            "pixels_per_metre": pixels_per_metre,
        }
        self._set_handle_color(index, "#ffd43b")

    def _on_mouse_move(self, interactor: object, _event: object) -> None:
        if self._selected_index is None or self._drag_state is None:
            hovered = self._pick_handle(interactor)
            if hovered == self._hovered_index:
                return
            if self._hovered_index is not None:
                self._set_handle_color(self._hovered_index, "#f5f5f5")
            self._hovered_index = hovered
            if hovered is not None:
                self._set_handle_color(hovered, "#8be9fd")
            self.plotter.render()
            return
        state = self._drag_state
        mouse_delta = np.asarray(interactor.GetEventPosition(), dtype=np.float64) - state["mouse"]
        distance = float(np.dot(mouse_delta, state["screen_direction"])) / float(
            state["pixels_per_metre"]
        )
        axis = np.asarray(state["axis"])
        opposite = np.asarray(state["opposite"])
        face = np.asarray(state["face"]) + axis * distance
        full_length = float(np.dot(face - opposite, axis))
        minimum_length = max(float(np.min(self.base_half_sizes)) * 0.02, 0.0005)
        full_length = max(full_length, minimum_length)
        face = opposite + axis * full_length
        centre = (face + opposite) * 0.5
        half_sizes = np.asarray(state["half_sizes"]).copy()
        half_sizes[int(state["axis_index"])] = full_length * 0.5
        axes = np.asarray(state["axes"])
        linear = axes @ np.diag(half_sizes / self.base_half_sizes)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = linear
        matrix[:3, 3] = centre - linear @ self.base_center
        transform = vtkTransform()
        transform.SetMatrix(pv.vtkmatrix_from_array(matrix))
        self.box_widget.SetTransform(transform)
        self.proxy_actor.user_matrix = matrix
        self.update_from_box()
        self.plotter.render()

    def _on_left_release(self, _interactor: object, _event: object) -> None:
        if self._selected_index is None:
            return
        self._set_handle_color(self._selected_index, "#f5f5f5")
        self._selected_index = None
        self._drag_state = None
        planes = vtkPlanes()
        self.box_widget.GetPlanes(planes)
        self.release_callback(planes)
        self.plotter.render()
