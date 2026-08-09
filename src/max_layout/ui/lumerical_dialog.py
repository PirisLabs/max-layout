"""Non-fabrication port editing and Lumerical notebook export dialogs."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QBrush, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..constants import LAYER_NAME_MAP
from ..gds.build import component_geometry_arrays
from ..lumerical import (
    LUMERICAL_SWEEP_MAX_RUNS,
    LUMERICAL_SWEEP_WARNING_RUNS,
    MATERIAL_CHOICES,
    STACK_PRESETS,
    available_geometry_layers,
    default_stack,
    normalize_lumerical_sweep_spec,
    seed_simulation_ports,
    sweepable_component_parameters,
)
from ..lumerical_optimization import (
    adjoint_optimizable_component_parameters,
    normalize_lumerical_optimization_spec,
)


def _checked_item(checked: bool) -> QTableWidgetItem:
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


_MATERIAL_COLORS = {
    "Air": QColor("#fef3c7"),
    "Si (Silicon) - Palik": QColor("#475569"),
    "SiO2 (Glass) - Palik": QColor("#7dd3fc"),
    "LiNbO3": QColor("#2dd4bf"),
    "Al2O3": QColor("#f59e0b"),
    "Au (Gold) - CRC": QColor("#facc15"),
    "Al (Aluminium) - Palik": QColor("#cbd5e1"),
    "Ag (Silver) - Palik": QColor("#e2e8f0"),
}


def _material_color(name: str, alpha: int = 185) -> QColor:
    color = QColor(_MATERIAL_COLORS.get(str(name), QColor("#a78bfa")))
    color.setAlpha(alpha)
    return color


def _anchored_stack_ranges(stack: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float, float]]:
    active = [row for row in stack if float(row.get("thickness_um", 0.0)) > 0.0]
    if not active:
        return []
    anchor = next(
        (index for index, row in enumerate(active) if str(row.get("role", "background")) == "geometry"),
        len(active) // 2,
    )
    thickness = float(active[anchor]["thickness_um"])
    ranges: list[tuple[dict[str, Any], float, float] | None] = [None] * len(active)
    ranges[anchor] = (active[anchor], -0.5 * thickness, 0.5 * thickness)
    cursor = -0.5 * thickness
    for index in range(anchor - 1, -1, -1):
        thickness = float(active[index]["thickness_um"])
        ranges[index] = (active[index], cursor - thickness, cursor)
        cursor -= thickness
    cursor = 0.5 * float(active[anchor]["thickness_um"])
    for index in range(anchor + 1, len(active)):
        thickness = float(active[index]["thickness_um"])
        ranges[index] = (active[index], cursor, cursor + thickness)
        cursor += thickness
    return [value for value in ranges if value is not None]


def _conformal_fill_start(
    stack_ranges: list[tuple[dict[str, Any], float, float]],
    row_index: int,
    default_z0: float,
) -> float:
    """Mirror the notebook's conformal-fill depth in the editor previews."""
    consecutive_geometry: list[tuple[dict[str, Any], float, float]] = []
    for previous in reversed(stack_ranges[:row_index]):
        previous_row, _previous_z0, _previous_z1 = previous
        if str(previous_row.get("role", "background")) != "geometry":
            if consecutive_geometry:
                break
            continue
        consecutive_geometry.append(previous)
    if len(consecutive_geometry) > 1:
        return min(default_z0, min(previous_z0 for _row, previous_z0, _z1 in consecutive_geometry))
    if consecutive_geometry:
        previous_row, previous_z0, previous_z1 = consecutive_geometry[0]
        thickness = previous_z1 - previous_z0
        etch_depth = min(
            thickness,
            max(0.0, float(previous_row.get("etch_depth_um", thickness))),
        )
        return min(default_z0, previous_z1 - etch_depth)
    return default_z0


def _offset_preview_polygon(points: Any, distance_um: float) -> np.ndarray:
    """Return a winding-independent miter offset with one vertex per input point.

    This is intentionally a lightweight preview operation rather than a GDS
    Boolean.  Keeping the vertex count stable lets the 3D renderer join each
    top edge to the corresponding bottom edge while still showing concave and
    curved component outlines.
    """
    original = np.asarray(points, dtype=float)
    if original.ndim != 2 or original.shape[1] != 2 or len(original) < 3:
        return original.copy()
    if not np.isfinite(original).all() or abs(float(distance_um)) <= 1e-12:
        return original.copy()
    closed = bool(np.linalg.norm(original[0] - original[-1]) <= 1e-12)
    polygon = original[:-1].copy() if closed else original.copy()
    if len(polygon) < 3:
        return original.copy()
    signed_twice_area = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    if abs(signed_twice_area) <= 1e-18:
        return original.copy()
    winding = 1.0 if signed_twice_area > 0.0 else -1.0
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.zeros_like(edges)
    valid = lengths > 1e-15
    normals[valid, 0] = winding * edges[valid, 1] / lengths[valid]
    normals[valid, 1] = -winding * edges[valid, 0] / lengths[valid]
    if not np.any(valid):
        return original.copy()

    result = polygon.copy()
    count = len(polygon)
    for index in range(count):
        previous_index = (index - 1) % count
        while not valid[previous_index] and previous_index != index:
            previous_index = (previous_index - 1) % count
        next_index = index
        while not valid[next_index] and next_index != previous_index:
            next_index = (next_index + 1) % count
        previous_normal = normals[previous_index]
        next_normal = normals[next_index]
        combined = previous_normal + next_normal
        denominator = 1.0 + float(np.dot(previous_normal, next_normal))
        if denominator <= 1e-10 or float(np.linalg.norm(combined)) <= 1e-12:
            shift = float(distance_um) * next_normal
        else:
            shift = float(distance_um) * combined / denominator
        # Extremely acute decorative vertices can produce an unbounded miter.
        # Limit only that singular visual spike; ordinary device edges retain
        # the exact Layer Builder offset.
        maximum_shift = 50.0 * abs(float(distance_um))
        shift_length = float(np.linalg.norm(shift))
        if maximum_shift > 0.0 and shift_length > maximum_shift:
            shift *= maximum_shift / shift_length
        result[index] += shift
    return np.vstack([result, result[0]]) if closed else result


def _sidewall_face_points(
    points: Any, etch_depth_um: float, sidewall_angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Top/bottom XY faces matching Lumerical's middle sidewall reference."""
    depth = max(0.0, float(etch_depth_um))
    angle = min(179.999, max(0.001, float(sidewall_angle_deg)))
    tangent = math.tan(math.radians(angle))
    half_offset_um = 0.0 if abs(tangent) <= 1e-12 else 0.5 * depth / tangent
    top = _offset_preview_polygon(points, -half_offset_um)
    bottom = _offset_preview_polygon(points, half_offset_um)
    return top, bottom


def _stack_reference_z(
    params: dict[str, Any],
    stack_ranges: list[tuple[dict[str, Any], float, float]],
    device_top: float,
    stack_top: float,
) -> float:
    reference = str(params.get("z reference", "device top")).strip().lower()
    if reference in {"center of sio2 cladding", "center of silica cladding", "cladding center"}:
        silica_rows = []
        for row, row_z0, row_z1 in stack_ranges:
            label = (str(row.get("name", "")) + " " + str(row.get("material", ""))).lower()
            if ("sio2" in label or "silica" in label or "glass" in label) and row_z1 >= device_top - 1e-12:
                silica_rows.append((bool(row.get("conformal", False)), float(row_z0), float(row_z1)))
        candidates = [entry for entry in silica_rows if entry[0]] or silica_rows
        if candidates:
            _conformal, row_z0, row_z1 = max(candidates, key=lambda entry: entry[2])
            return 0.5 * (row_z0 + row_z1)
        return device_top
    if reference in {"top of sio2 cladding", "top of silica cladding", "top cladding"}:
        silica_rows = []
        for row, _z0, row_z1 in stack_ranges:
            label = (str(row.get("name", "")) + " " + str(row.get("material", ""))).lower()
            if ("sio2" in label or "silica" in label or "glass" in label) and row_z1 >= device_top - 1e-12:
                silica_rows.append((bool(row.get("conformal", False)), float(row_z1)))
        conformal_tops = [row_z1 for conformal, row_z1 in silica_rows if conformal]
        if conformal_tops:
            return max(conformal_tops)
        return max((row_z1 for _conformal, row_z1 in silica_rows), default=device_top)
    if reference == "top of stack":
        return stack_top
    return device_top


def _fiber_axis_plane_center(
    component: dict[str, Any],
    components: list[dict[str, Any]],
    stack_ranges: list[tuple[dict[str, Any], float, float]],
    device_top: float,
    stack_top: float,
) -> tuple[float, float, float]:
    """Place a Z-referenced plane at its intersection with a tilted fiber axis."""
    params = component.get("params", {})
    reference_z = _stack_reference_z(params, stack_ranges, device_top, stack_top)
    plane_z = reference_z + float(params.get("distance_um", 0.0))
    cx, cy = float(component.get("x", 0.0)), float(component.get("y", 0.0))
    if not bool(params.get("align to fiber axis", False)):
        return cx, cy, plane_z
    parent_uid = int(component.get("simulation_parent_uid", -1))
    fibers = [candidate for candidate in components if str(candidate.get("kind", "")) in {"Fiber geometry", "Fiber port"}]
    matching = [candidate for candidate in fibers if int(candidate.get("simulation_parent_uid", -2)) == parent_uid]
    candidates = matching or fibers
    if not candidates:
        return cx, cy, plane_z
    fiber = min(
        candidates,
        key=lambda candidate: math.hypot(
            float(candidate.get("x", 0.0)) - cx,
            float(candidate.get("y", 0.0)) - cy,
        ),
    )
    fiber_params = fiber.get("params", {})
    fiber_bottom_z = _stack_reference_z(
        fiber_params, stack_ranges, device_top, stack_top
    ) + float(fiber_params.get("distance_um", 0.0))
    theta = math.radians(float(fiber_params.get("angle theta", params.get("angle theta", 0.0))))
    phi = math.radians(
        float(fiber.get("orientation_deg", 0.0)) + float(fiber_params.get("angle phi", 0.0))
    )
    lateral = (plane_z - fiber_bottom_z) * math.tan(theta)
    return (
        float(fiber.get("x", 0.0)) + lateral * math.cos(phi),
        float(fiber.get("y", 0.0)) + lateral * math.sin(phi),
        plane_z,
    )


class CrossSectionDomainPreview(QWidget):
    """Live XZ stack or XY/GDS preview with draggable FDTD boundaries."""

    def __init__(self, dialog: "LumericalExportDialog", plane: str) -> None:
        super().__init__(dialog)
        self.dialog = dialog
        self.plane = str(plane).upper()
        if self.plane not in {"XZ", "XY"}:
            raise ValueError("FDTD preview plane must be XZ or XY")
        # The stack-direction section and GDS-like top view are displayed side
        # by side, so keep each canvas compact enough to leave the shared
        # numeric X/Y/Z domain controls visible below them.
        self.setMinimumHeight(350)
        self.setMinimumWidth(360)
        self.setMouseTracking(True)
        self._drag_edges: set[str] = set()
        self._drag_origin_world: tuple[float, float] | None = None
        self._drag_domain: tuple[float, float, float, float] | None = None
        self._plot_rect = QRectF()
        self._world = (-1.0, 1.0, -1.0, 1.0)
        self._view_plane: str | None = None
        self._view_locked = False

    def reset_view(self) -> None:
        """Reframe once; subsequent FDTD edits move only the red boundary."""
        self._view_locked = False
        self.update()

    def _pixel(self, horizontal: float, vertical: float) -> QPointF:
        h0, h1, z0, z1 = self._world
        x = self._plot_rect.left() + (horizontal - h0) / max(1e-12, h1 - h0) * self._plot_rect.width()
        y = self._plot_rect.bottom() - (vertical - z0) / max(1e-12, z1 - z0) * self._plot_rect.height()
        return QPointF(x, y)

    def _world_at(self, point: QPointF) -> tuple[float, float]:
        h0, h1, z0, z1 = self._world
        horizontal = h0 + (point.x() - self._plot_rect.left()) / max(1.0, self._plot_rect.width()) * (h1 - h0)
        vertical = z0 + (self._plot_rect.bottom() - point.y()) / max(1.0, self._plot_rect.height()) * (z1 - z0)
        return horizontal, vertical

    def _projection_state(
        self, state: dict[str, Any]
    ) -> tuple[str, str, tuple[float, float], tuple[float, float], tuple[float, float, float, float]]:
        """Resolve the two displayed axes and their exact FDTD boundaries."""
        horizontal_axis = "x"
        vertical_axis = "z" if self.plane == "XZ" else "y"
        horizontal_base = tuple(state[horizontal_axis + "_base"])
        vertical_base = tuple(state[vertical_axis + "_base"])
        domain = (
            float(horizontal_base[0]) - float(state["padding"][horizontal_axis + "_min"]),
            float(horizontal_base[1]) + float(state["padding"][horizontal_axis + "_max"]),
            float(vertical_base[0]) - float(state["padding"][vertical_axis + "_min"]),
            float(vertical_base[1]) + float(state["padding"][vertical_axis + "_max"]),
        )
        return horizontal_axis, vertical_axis, horizontal_base, vertical_base, domain

    @staticmethod
    def _layer_color(layer: int) -> QColor:
        """Give each visible GDS layer a stable, translucent editor color."""
        color = QColor.fromHsv((int(layer) * 53 + 205) % 360, 145, 225)
        color.setAlpha(155)
        return color

    def _draw_stack_section(
        self,
        painter: QPainter,
        state: dict[str, Any],
        horizontal_base: tuple[float, float],
    ) -> None:
        """Draw fixed process materials in the stack-direction XZ view."""
        for row_index, (row, row_z0, row_z1) in enumerate(state["stack_ranges"]):
            clipped_z0 = (
                _conformal_fill_start(state["stack_ranges"], row_index, row_z0)
                if bool(row.get("conformal", False))
                else row_z0
            )
            clipped_z1 = row_z1
            if clipped_z1 <= clipped_z0:
                continue
            role = str(row.get("role", "background"))
            if role == "geometry":
                horizontal_min, horizontal_max = horizontal_base
            else:
                # Full films are conceptually extended media. Draw them across
                # the fixed viewport so moving the FDTD box never resizes them.
                horizontal_min, horizontal_max = self._world[0], self._world[1]
            top_left = self._pixel(horizontal_min, clipped_z1)
            bottom_right = self._pixel(horizontal_max, clipped_z0)
            rect = QRectF(top_left, bottom_right).normalized()
            painter.setPen(QPen(QColor("#475569"), 1))
            painter.setBrush(
                QBrush(
                    _material_color(
                        str(row.get("material", "")),
                        165 if role == "geometry" else 105,
                    )
                )
            )
            painter.drawRect(rect)
            if rect.height() >= 16:
                painter.setPen(QPen(QColor("#0f172a"), 1))
                painter.drawText(
                    rect.adjusted(5, 1, -5, -1),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    f"{row.get('name', 'layer')} · {row.get('material', '')}",
                )

    def _draw_gds_top_view(self, painter: QPainter, state: dict[str, Any]) -> None:
        """Draw the exact selected export polygons in their global XY positions."""
        for points, layer in state["polygons"]:
            array = np.asarray(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
                continue
            polygon = QPolygonF(
                [self._pixel(float(point[0]), float(point[1])) for point in array]
            )
            painter.setPen(QPen(self._layer_color(int(layer)).darker(155), 1))
            painter.setBrush(QBrush(self._layer_color(int(layer))))
            painter.drawPolygon(polygon)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        state = self.dialog.preview_state()
        (
            horizontal_axis,
            vertical_axis,
            horizontal_base,
            vertical_base,
            domain,
        ) = self._projection_state(state)
        horizontal_min, horizontal_max, vertical_min, vertical_max = domain
        self._plot_rect = QRectF(
            72.0,
            44.0,
            max(120.0, self.width() - 112.0),
            max(120.0, self.height() - 98.0),
        )
        if not self._view_locked or self._view_plane != self.plane:
            # Reserve generous fixed space around the current domain. Keeping
            # this mapping locked is what makes later FDTD-edge movement
            # visible instead of making the fixed geometry appear to resize.
            framed_h_min = min(horizontal_min, float(horizontal_base[0]))
            framed_h_max = max(horizontal_max, float(horizontal_base[1]))
            framed_v_min = min(vertical_min, float(vertical_base[0]))
            framed_v_max = max(vertical_max, float(vertical_base[1]))
            horizontal_span = max(1e-6, framed_h_max - framed_h_min)
            vertical_span = max(1e-6, framed_v_max - framed_v_min)
            if self.plane == "XZ":
                # Preserve the established readable stack framing in the left
                # view, including room to drag the red box beyond the device.
                horizontal_margin = max(2.0, 0.35 * horizontal_span)
                vertical_margin = max(1.0, 0.35 * vertical_span)
            else:
                horizontal_margin = max(0.05, 0.22 * horizontal_span)
                vertical_margin = max(0.05, 0.22 * vertical_span)
            world_h_min = framed_h_min - horizontal_margin
            world_h_max = framed_h_max + horizontal_margin
            world_v_min = framed_v_min - vertical_margin
            world_v_max = framed_v_max + vertical_margin
            if self.plane == "XY":
                # Keep one micrometre the same size in X and Y, as in the GDS
                # editor. The XZ stack view deliberately fills its canvas so
                # thin films remain readable beside long photonic devices.
                pixel_aspect = self._plot_rect.width() / max(1.0, self._plot_rect.height())
                world_h_span = world_h_max - world_h_min
                world_v_span = world_v_max - world_v_min
                world_aspect = world_h_span / max(1e-12, world_v_span)
                if world_aspect < pixel_aspect:
                    addition = 0.5 * (world_v_span * pixel_aspect - world_h_span)
                    world_h_min -= addition
                    world_h_max += addition
                else:
                    addition = 0.5 * (world_h_span / pixel_aspect - world_v_span)
                    world_v_min -= addition
                    world_v_max += addition
            self._world = (world_h_min, world_h_max, world_v_min, world_v_max)
            self._view_plane = self.plane
            self._view_locked = True

        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRect(self._plot_rect)
        painter.save()
        painter.setClipRect(self._plot_rect)
        if self.plane == "XZ":
            self._draw_stack_section(painter, state, horizontal_base)
        else:
            self._draw_gds_top_view(painter, state)
        painter.restore()

        fdtd_top_left = self._pixel(horizontal_min, vertical_max)
        fdtd_bottom_right = self._pixel(horizontal_max, vertical_min)
        fdtd_rect = QRectF(fdtd_top_left, fdtd_bottom_right).normalized()
        pml_width = max(7.0, min(22.0, 0.055 * fdtd_rect.width()))
        pml_height = max(7.0, min(22.0, 0.07 * fdtd_rect.height()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(239, 68, 68, 35)))
        painter.drawRect(QRectF(fdtd_rect.left(), fdtd_rect.top(), pml_width, fdtd_rect.height()))
        painter.drawRect(QRectF(fdtd_rect.right() - pml_width, fdtd_rect.top(), pml_width, fdtd_rect.height()))
        painter.drawRect(QRectF(fdtd_rect.left(), fdtd_rect.top(), fdtd_rect.width(), pml_height))
        painter.drawRect(QRectF(fdtd_rect.left(), fdtd_rect.bottom() - pml_height, fdtd_rect.width(), pml_height))
        painter.setPen(QPen(QColor("#dc2626"), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(fdtd_rect)
        painter.setBrush(QBrush(QColor("#dc2626")))
        for corner in (fdtd_rect.topLeft(), fdtd_rect.topRight(), fdtd_rect.bottomLeft(), fdtd_rect.bottomRight()):
            painter.drawRect(QRectF(corner.x() - 4.0, corner.y() - 4.0, 8.0, 8.0))
        painter.setPen(QPen(QColor("#991b1b"), 1))
        painter.drawText(fdtd_rect.adjusted(7, 5, -7, -5), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, "3D FDTD · drag inside to move · edges to resize")
        painter.setPen(QPen(QColor("#334155"), 1))
        title = (
            "XZ · stack-direction cross-section   X →   Z ↑"
            if self.plane == "XZ"
            else "XY · TOP / GDS view   X →   Y ↑"
        )
        painter.drawText(10, 24, title + "   ·   red dashed box = FDTD domain")
        painter.drawText(10, self.height() - 12, "Drag the red boundary; use Fit preview only when you want to reframe the camera.")
        self._fdtd_rect = fdtd_rect

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not hasattr(self, "_fdtd_rect"):
            return
        point = event.position()
        state = self.dialog.preview_state()
        horizontal_axis, vertical_axis, _horizontal_base, _vertical_base, domain = self._projection_state(state)
        horizontal_edges = {
            horizontal_axis + "_min": abs(point.x() - self._fdtd_rect.left()),
            horizontal_axis + "_max": abs(point.x() - self._fdtd_rect.right()),
        }
        vertical_edges = {
            vertical_axis + "_max": abs(point.y() - self._fdtd_rect.top()),
            vertical_axis + "_min": abs(point.y() - self._fdtd_rect.bottom()),
        }
        horizontal_edge, horizontal_distance = min(horizontal_edges.items(), key=lambda item: item[1])
        vertical_edge, vertical_distance = min(vertical_edges.items(), key=lambda item: item[1])
        threshold = 14.0
        # Corners resize both axes. Edges resize one axis and must be grabbed
        # alongside the visible FDTD rectangle, not anywhere on the canvas.
        if horizontal_distance <= threshold and vertical_distance <= threshold:
            self._drag_edges = {horizontal_edge, vertical_edge}
        elif horizontal_distance <= threshold and self._fdtd_rect.top() - threshold <= point.y() <= self._fdtd_rect.bottom() + threshold:
            self._drag_edges = {horizontal_edge}
        elif vertical_distance <= threshold and self._fdtd_rect.left() - threshold <= point.x() <= self._fdtd_rect.right() + threshold:
            self._drag_edges = {vertical_edge}
        elif self._fdtd_rect.contains(point):
            self._drag_edges = {"move"}
        else:
            self._drag_edges.clear()
        if self._drag_edges:
            horizontal, vertical = self._world_at(point)
            self._drag_origin_world = (horizontal, vertical)
            self._drag_domain = domain
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._drag_edges:
            return
        horizontal, vertical = self._world_at(event.position())
        state = self.dialog.preview_state()
        (
            horizontal_axis,
            vertical_axis,
            horizontal_base,
            vertical_base,
            _domain,
        ) = self._projection_state(state)
        if "move" in self._drag_edges and self._drag_origin_world is not None and self._drag_domain is not None:
            delta_h = horizontal - self._drag_origin_world[0]
            delta_v = vertical - self._drag_origin_world[1]
            original_h_min, original_h_max, original_v_min, original_v_max = self._drag_domain
            self.dialog.padding_spin(horizontal_axis + "_min").setValue(
                float(horizontal_base[0]) - (original_h_min + delta_h)
            )
            self.dialog.padding_spin(horizontal_axis + "_max").setValue(
                (original_h_max + delta_h) - float(horizontal_base[1])
            )
            self.dialog.padding_spin(vertical_axis + "_min").setValue(
                float(vertical_base[0]) - (original_v_min + delta_v)
            )
            self.dialog.padding_spin(vertical_axis + "_max").setValue(
                (original_v_max + delta_v) - float(vertical_base[1])
            )
            self.update()
            return
        if horizontal_axis + "_min" in self._drag_edges:
            self.dialog.padding_spin(horizontal_axis + "_min").setValue(
                float(horizontal_base[0]) - horizontal
            )
        if horizontal_axis + "_max" in self._drag_edges:
            self.dialog.padding_spin(horizontal_axis + "_max").setValue(
                horizontal - float(horizontal_base[1])
            )
        if vertical_axis + "_min" in self._drag_edges:
            self.dialog.padding_spin(vertical_axis + "_min").setValue(
                float(vertical_base[0]) - vertical
            )
        if vertical_axis + "_max" in self._drag_edges:
            self.dialog.padding_spin(vertical_axis + "_max").setValue(
                vertical - float(vertical_base[1])
            )
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag_edges.clear()
        self._drag_origin_world = None
        self._drag_domain = None


class ThreeDModelPreview(QWidget):
    """Interactive isometric preview of the exact polygons, stack, fiber, ports, and FDTD box."""

    def __init__(self, dialog: "LumericalExportDialog") -> None:
        super().__init__()
        self.dialog = dialog
        self.azimuth_deg = 35.0
        self.elevation_deg = 20.0
        self.zoom_factor = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.last_position: QPointF | None = None
        self.pan_mode = False
        self.show_stack = True
        self.show_device = True
        self.show_ports = True
        self.show_fiber = True
        self.show_fdtd = True
        self.hidden_stack_rows: set[int] = set()
        self.setMinimumSize(1050, 700)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.last_position = event.position()
        self.pan_mode = bool(
            event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}
            or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.last_position is None:
            return
        delta = event.position() - self.last_position
        if self.pan_mode:
            self.pan += delta
        else:
            self.azimuth_deg += 0.45 * delta.x()
            self.elevation_deg = min(85.0, max(-85.0, self.elevation_deg - 0.35 * delta.y()))
        self.last_position = event.position()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.last_position = None

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        steps = event.angleDelta().y() / 120.0
        self.zoom_factor = min(12.0, max(0.12, self.zoom_factor * (1.15 ** steps)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        state = self.dialog.preview_state()
        padding = state["padding"]
        x0, x1 = state["x_base"][0] - padding["x_min"], state["x_base"][1] + padding["x_max"]
        y0, y1 = state["y_base"][0] - padding["y_min"], state["y_base"][1] + padding["y_max"]
        z0, z1 = state["z_base"][0] - padding["z_min"], state["z_base"][1] + padding["z_max"]
        angle = self.azimuth_deg * 3.141592653589793 / 180.0
        elevation = self.elevation_deg * 3.141592653589793 / 180.0

        def raw(point: tuple[float, float, float]) -> QPointF:
            x, y, z = point
            horizontal = x * math.cos(angle) - y * math.sin(angle)
            depth = x * math.sin(angle) + y * math.cos(angle)
            vertical = -z * math.cos(elevation) + depth * math.sin(elevation)
            return QPointF(horizontal, vertical)

        # Camera fitting is based on the immutable device/stack extents, not
        # the draggable FDTD box. Resizing the red domain therefore cannot
        # make the physical geometry appear to resize. Zoom out to inspect a
        # deliberately much larger domain.
        fit_x0, fit_x1 = state["x_base"]
        fit_y0, fit_y1 = state["y_base"]
        fit_z0, fit_z1 = state["z_base"]
        all_points = [raw((x, y, z)) for x in (fit_x0, fit_x1) for y in (fit_y0, fit_y1) for z in (fit_z0, fit_z1)]
        polygon_rows = []
        for points, layer in state["polygons"]:
            matching = [entry for entry in state["stack_ranges"] if str(entry[0].get("role")) == "geometry" and int(layer) in {int(v) for v in entry[0].get("gds_layers", [])}]
            if not matching:
                continue
            for row, row_z0, row_z1 in matching:
                etch_depth = min(row_z1 - row_z0, max(0.0, float(row.get("etch_depth_um", row_z1 - row_z0))))
                film_top = row_z1 - etch_depth
                if (
                    str(row.get("slab_extent", "full")).strip().lower() == "geometry"
                    and film_top > row_z0
                ):
                    projected_slab_top = [raw((float(point[0]), float(point[1]), film_top)) for point in points]
                    projected_slab_bottom = [raw((float(point[0]), float(point[1]), row_z0)) for point in points]
                    all_points.extend(projected_slab_top)
                    all_points.extend(projected_slab_bottom)
                    polygon_rows.append((projected_slab_top, projected_slab_bottom, row))
                if etch_depth <= 0.0:
                    continue
                patterned_z0 = row_z1 - etch_depth
                top_points, bottom_points = _sidewall_face_points(
                    points,
                    etch_depth,
                    float(row.get("sidewall_angle_deg", 90.0)),
                )
                projected_top = [raw((float(point[0]), float(point[1]), row_z1)) for point in top_points]
                projected_bottom = [raw((float(point[0]), float(point[1]), patterned_z0)) for point in bottom_points]
                all_points.extend(projected_top)
                all_points.extend(projected_bottom)
                polygon_rows.append((projected_top, projected_bottom, row))
        min_x = min(point.x() for point in all_points); max_x = max(point.x() for point in all_points)
        min_y = min(point.y() for point in all_points); max_y = max(point.y() for point in all_points)
        scale = self.zoom_factor * min((self.width() - 100.0) / max(1e-9, max_x - min_x), (self.height() - 120.0) / max(1e-9, max_y - min_y))
        fitted_width = (max_x - min_x) * scale
        fitted_height = (max_y - min_y) * scale

        def screen(point: QPointF) -> QPointF:
            return QPointF(
                0.5 * (self.width() - fitted_width) + (point.x() - min_x) * scale + self.pan.x(),
                0.5 * (self.height() - fitted_height) + (point.y() - min_y) * scale + self.pan.y(),
            )

        def draw_background_box(
            row: dict[str, Any],
            row_z0: float,
            row_z1: float,
            side_alpha: int = 58,
            top_alpha: int = 82,
        ) -> None:
            clipped_top = min(z1, row_z1)
            clipped_bottom = max(z0, row_z0)
            if clipped_top <= clipped_bottom:
                return
            top_face = [screen(raw((x0, y0, clipped_top))), screen(raw((x1, y0, clipped_top))),
                        screen(raw((x1, y1, clipped_top))), screen(raw((x0, y1, clipped_top)))]
            bottom_face = [screen(raw((x0, y0, clipped_bottom))), screen(raw((x1, y0, clipped_bottom))),
                           screen(raw((x1, y1, clipped_bottom))), screen(raw((x0, y1, clipped_bottom)))]
            material = str(row.get("material", ""))
            painter.setPen(QPen(QColor("#64748b"), 0.7))
            painter.setBrush(QBrush(_material_color(material, side_alpha)))
            for edge_index in range(4):
                next_index = (edge_index + 1) % 4
                painter.drawPolygon(QPolygonF([
                    bottom_face[edge_index], bottom_face[next_index],
                    top_face[next_index], top_face[edge_index],
                ]))
            painter.setBrush(QBrush(_material_color(material, top_alpha)))
            painter.drawPolygon(QPolygonF(top_face))

        if self.show_stack:
            for row, row_z0, row_z1 in state["stack_ranges"]:
                if str(row.get("role", "background")) != "background":
                    continue
                if int(row.get("_preview_id", -1)) in self.hidden_stack_rows:
                    continue
                # Draw conformal claddings after the device so the preview
                # visibly shows that they cover every tooth and output sector.
                if bool(row.get("conformal", False)):
                    continue
                draw_background_box(row, row_z0, row_z1)

        if self.show_device:
            # A partially etched geometry row also contains its fixed
            # unetched base film below the patterned volume.
            for row, row_z0, row_z1 in state["stack_ranges"]:
                if str(row.get("role", "background")) != "geometry":
                    continue
                if int(row.get("_preview_id", -1)) in self.hidden_stack_rows:
                    continue
                etch_depth = min(row_z1 - row_z0, max(0.0, float(row.get("etch_depth_um", row_z1 - row_z0))))
                film_top = row_z1 - etch_depth
                if film_top <= row_z0:
                    continue
                if str(row.get("slab_extent", "full")).strip().lower() == "geometry":
                    continue
                top_face = [screen(raw((x0, y0, film_top))), screen(raw((x1, y0, film_top))),
                            screen(raw((x1, y1, film_top))), screen(raw((x0, y1, film_top)))]
                bottom_face = [screen(raw((x0, y0, row_z0))), screen(raw((x1, y0, row_z0))),
                               screen(raw((x1, y1, row_z0))), screen(raw((x0, y1, row_z0)))]
                painter.setPen(QPen(QColor("#475569"), 0.7))
                painter.setBrush(QBrush(_material_color(str(row.get("material", "")), 70)))
                for edge_index in range(4):
                    next_index = (edge_index + 1) % 4
                    painter.drawPolygon(QPolygonF([
                        bottom_face[edge_index], bottom_face[next_index],
                        top_face[next_index], top_face[edge_index],
                    ]))
                painter.drawPolygon(QPolygonF(top_face))

            for projected_top, projected_bottom, row in polygon_rows:
                if int(row.get("_preview_id", -1)) in self.hidden_stack_rows:
                    continue
                painter.setPen(QPen(QColor("#334155"), 0.8))
                material = str(row.get("material", ""))
                top_screen = [screen(point) for point in projected_top]
                bottom_screen = [screen(point) for point in projected_bottom]
                painter.setBrush(QBrush(_material_color(material, 145)))
                for edge_index in range(len(top_screen)):
                    next_index = (edge_index + 1) % len(top_screen)
                    painter.drawPolygon(QPolygonF([
                        bottom_screen[edge_index], bottom_screen[next_index],
                        top_screen[next_index], top_screen[edge_index],
                    ]))
                painter.setBrush(QBrush(_material_color(material, 210)))
                painter.drawPolygon(QPolygonF(top_screen))

        if self.show_stack:
            for row_index, (row, row_z0, row_z1) in enumerate(state["stack_ranges"]):
                if str(row.get("role", "background")) != "background" or not bool(row.get("conformal", False)):
                    continue
                if int(row.get("_preview_id", -1)) in self.hidden_stack_rows:
                    continue
                fill_z0 = _conformal_fill_start(state["stack_ranges"], row_index, row_z0)
                draw_background_box(row, fill_z0, row_z1, side_alpha=48, top_alpha=68)

        geometry_tops = [row_z1 for row, _, row_z1 in state["stack_ranges"] if str(row.get("role", "background")) == "geometry"]
        device_top = max(geometry_tops, default=0.0)
        stack_top = state["stack_ranges"][-1][2] if state["stack_ranges"] else device_top
        for component in state["components"]:
            kind = str(component.get("kind", ""))
            params = component.get("params", {})
            cx, cy = float(component.get("x", 0.0)), float(component.get("y", 0.0))
            if kind in {"Fiber geometry", "Fiber port"} and self.show_fiber:
                reference_z = _stack_reference_z(params, state["stack_ranges"], device_top, stack_top)
                contact_z = reference_z + float(params.get("distance_um", 0.0))
                length = float(params.get("fiber length_um", 20.0))
                theta = math.radians(float(params.get("angle theta", 10.0)))
                phi = math.radians(float(params.get("angle phi", 0.0)) + float(component.get("orientation_deg", 0.0)))
                fiber_z0 = contact_z - 0.5 * length
                fiber_z1 = contact_z + 0.5 * length
                visible_z0 = max(z0, fiber_z0)
                visible_z1 = min(z1, fiber_z1)
                slope_x = math.tan(theta) * math.cos(phi)
                slope_y = math.tan(theta) * math.sin(phi)
                start = screen(raw((
                    cx + (visible_z0 - contact_z) * slope_x,
                    cy + (visible_z0 - contact_z) * slope_y,
                    visible_z0,
                )))
                stop = screen(raw((
                    cx + (visible_z1 - contact_z) * slope_x,
                    cy + (visible_z1 - contact_z) * slope_y,
                    visible_z1,
                )))
                painter.setPen(QPen(QColor(186, 230, 253, 18), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(start, stop)
                painter.setPen(QPen(QColor(14, 116, 144, 110), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(start, stop)
            elif kind in {"FDTD port", "Fiber-axis FDTD port", "Power monitor", "Mode expansion monitor", "Field profile monitor"} and self.show_ports:
                normal = str(params.get("plane normal", "X")).upper()
                span = 0.5 * float(params.get("span_um", max(params.get("x span", 0.0), params.get("y span", 0.0), 2.5)))
                if normal == "Z":
                    cx, cy, port_z = _fiber_axis_plane_center(
                        component, state["components"], state["stack_ranges"], device_top, stack_top
                    )
                    theta = math.radians(float(params.get("angle theta", 0.0)))
                    phi = math.radians(float(component.get("orientation_deg", 0.0)) + float(params.get("angle phi", 0.0)))
                    transverse = np.asarray([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
                    tilted = np.asarray(
                        [math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)],
                        dtype=float,
                    )
                    center_3d = np.asarray([cx, cy, port_z], dtype=float)
                    vertices = [
                        center_3d - span * transverse - span * tilted,
                        center_3d + span * transverse - span * tilted,
                        center_3d + span * transverse + span * tilted,
                        center_3d - span * transverse + span * tilted,
                    ]
                    plane = [screen(raw(tuple(vertex))) for vertex in vertices]
                else:
                    normal_angle = float(component.get("orientation_deg", 0.0)) + (90.0 if normal == "Y" else 0.0)
                    nearest = int(round(normal_angle / 90.0) * 90) % 360
                    normal = "X" if nearest in (0, 180) else "Y"
                    port_z0 = max(z0, -0.5 * float(params.get("z_span_um", 2.25)))
                    port_z1 = min(z1, 0.5 * float(params.get("z_span_um", 2.25)))
                    if normal == "Y":
                        plane = [screen(raw((cx - span, cy, port_z0))), screen(raw((cx + span, cy, port_z0))),
                                 screen(raw((cx + span, cy, port_z1))), screen(raw((cx - span, cy, port_z1)))]
                    else:
                        plane = [screen(raw((cx, cy - span, port_z0))), screen(raw((cx, cy + span, port_z0))),
                                 screen(raw((cx, cy + span, port_z1))), screen(raw((cx, cy - span, port_z1)))]
                is_monitor = (
                    kind in {"Power monitor", "Mode expansion monitor", "Field profile monitor"}
                    or component.get("simulation_parent_port") == "fiber_input_power"
                    or str(params.get("fiber plane role", "")).lower() == "input power measurement"
                )
                painter.setPen(QPen(QColor("#059669" if is_monitor else "#7c3aed"), 1.7))
                painter.setBrush(QBrush(QColor(5, 150, 105, 42) if is_monitor else QColor(124, 58, 237, 42)))
                painter.drawPolygon(QPolygonF(plane))

        corners = {(ix, iy, iz): screen(raw((x, y, z))) for ix, x in enumerate((x0, x1)) for iy, y in enumerate((y0, y1)) for iz, z in enumerate((z0, z1))}
        edges = []
        for ix in (0, 1):
            for iy in (0, 1): edges.append((corners[ix, iy, 0], corners[ix, iy, 1]))
        for iz in (0, 1):
            for ix in (0, 1): edges.append((corners[ix, 0, iz], corners[ix, 1, iz]))
            for iy in (0, 1): edges.append((corners[0, iy, iz], corners[1, iy, iz]))
        if self.show_fdtd:
            painter.setPen(QPen(QColor("#dc2626"), 2, Qt.PenStyle.DashLine))
            for first, second in edges: painter.drawLine(first, second)
        painter.setPen(QPen(QColor("#0f172a"), 1))
        painter.drawText(20, 28, "Pre-export 3D model · drag to orbit · Shift/right-drag to pan · wheel to zoom")
        painter.drawText(20, 49, "Red wireframe = FDTD only; resizing it never scales the fixed device polygons")

        # Keep the material identity visible while the model is rotated. Each
        # active stack row receives a swatch and its exact Lumerical name.
        legend_rows = [
            entry[0] for entry in state["stack_ranges"]
            if int(entry[0].get("_preview_id", -1)) not in self.hidden_stack_rows
        ] if self.show_stack or self.show_device else []
        legend_width = min(420.0, max(300.0, self.width() * 0.34))
        legend_x = self.width() - legend_width - 18.0
        legend_y = 68.0
        legend_height = 29.0 + 25.0 * len(legend_rows)
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(QRectF(legend_x, legend_y, legend_width, legend_height), 7.0, 7.0)
        painter.setPen(QPen(QColor("#0f172a"), 1))
        painter.drawText(QRectF(legend_x + 10.0, legend_y + 4.0, legend_width - 20.0, 22.0), Qt.AlignmentFlag.AlignVCenter, "Material stack (bottom → top)")
        for index, row in enumerate(legend_rows):
            row_y = legend_y + 29.0 + 25.0 * index
            swatch = _material_color(str(row.get("material", "")), 220)
            # Slightly vary repeated materials so adjacent layers remain distinct.
            if index % 2:
                swatch = swatch.lighter(112)
            painter.setPen(QPen(QColor("#64748b"), 0.8))
            painter.setBrush(QBrush(swatch))
            painter.drawRect(QRectF(legend_x + 10.0, row_y + 3.0, 18.0, 16.0))
            painter.setPen(QPen(QColor("#1e293b"), 1))
            label = f"{row.get('name', 'Layer')} — {row.get('material', '')}"
            if str(row.get("role", "background")) == "geometry":
                label += f" · sidewall {float(row.get('sidewall_angle_deg', 90.0)):.3g}°"
            painter.drawText(QRectF(legend_x + 36.0, row_y, legend_width - 46.0, 22.0), Qt.AlignmentFlag.AlignVCenter, label)


class SimulationPortsDialog(QDialog):
    """Edit persistent, simulation-only ports for one layout component."""

    HEADERS = [
        "Use", "name", "dir", "loc", "pos", "order", "Port geometry",
        "X (µm)", "Y (µm)", "Distance (µm)", "Outward °", "Span (µm)",
        "Z span (µm)", "Mode", "angle theta", "angle phi", "core diameter (µm)",
        "core index", "cladding index", "Domain",
    ]

    def __init__(self, component: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.component = component
        self.ports = deepcopy(component.get("simulation_ports") or seed_simulation_ports(deepcopy(component)))
        self.setWindowTitle(f"Simulation ports — {component.get('kind', 'component')}")
        self.resize(1500, 560)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Ports are standard editable FDTD line/surface objects with a physical offset distance. "
            "Fiber core/cladding is placed separately using the Fiber geometry group in the left library. "
            "The JSON names name, dir, loc, pos, and order match the Lumerical examples exactly. "
            "All port objects are editor metadata only and never appear in GDS files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(13, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        for port in self.ports:
            self._append_port(port)

        row_buttons = QHBoxLayout()
        add_button = QPushButton("Add port")
        remove_button = QPushButton("Remove selected")
        reset_button = QPushButton("Reset from component points")
        add_button.clicked.connect(self.add_port)
        remove_button.clicked.connect(self.remove_selected)
        reset_button.clicked.connect(self.reset_ports)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        row_buttons.addStretch(1)
        row_buttons.addWidget(reset_button)
        layout.addLayout(row_buttons)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_port(self, port: dict[str, Any]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        center = port.get("center", (0.0, 0.0))
        values = [
            None,
            str(port.get("name", f"opt_{row + 1}")),
            str(port.get("dir", "Bidirectional")),
            f"{float(port.get('loc', 0.5)):.9g}",
            str(port.get("pos", "Right")),
            str(int(port.get("order", row + 1))),
            str(port.get("port geometry", "surface")),
            f"{float(center[0]):.9g}",
            f"{float(center[1]):.9g}",
            f"{float(port.get('distance_um', 0.0)):.9g}",
            f"{float(port.get('outward_orientation_deg', 0.0)):.9g}",
            f"{float(port.get('span_um', 2.0)):.9g}",
            f"{float(port.get('z_span_um', 2.0)):.9g}",
            str(port.get("mode", "fundamental TE mode")),
            f"{float(port.get('angle theta', 0.0)):.9g}",
            f"{float(port.get('angle phi', port.get('outward_orientation_deg', 0.0))):.9g}",
            f"{float(port.get('core diameter_um', 2.0 * float(port.get('waist radius w0_um', 4.5)))):.9g}",
            f"{float(port.get('core index', 1.44427)):.9g}",
            f"{float(port.get('cladding index', 1.43482)):.9g}",
            str(port.get("domain", "optical")),
        ]
        self.table.setItem(row, 0, _checked_item(bool(port.get("enabled", True))))
        for column, value in enumerate(values[1:], start=1):
            self.table.setItem(row, column, QTableWidgetItem(value))
        for column, choices, current in (
            (2, ["Bidirectional", "Forward", "Backward"], values[2]),
            (4, ["Left", "Right", "Top", "Bottom"], values[4]),
            (6, ["line", "surface"], values[6]),
            (19, ["optical", "rf"], values[19]),
        ):
            combo = QComboBox()
            combo.addItems(choices)
            combo.setCurrentText(str(current))
            self.table.setCellWidget(row, column, combo)

    def add_port(self) -> None:
        row = self.table.rowCount() + 1
        self._append_port(
            {
                "name": f"opt_{row}", "center": [0.0, 0.0], "outward_orientation_deg": 0.0,
                "domain": "optical", "dir": "Bidirectional", "pos": "Right", "order": row,
                "loc": 0.5, "port geometry": "surface", "distance_um": 0.0,
                "span_um": 2.0, "z_span_um": 2.0, "angle theta": 0.0, "angle phi": 0.0,
                "core diameter_um": 9.0, "core index": 1.44427, "cladding index": 1.43482,
                "cladding diameter_um": 50.0, "fiber length_um": 20.0,
                "mode": "fundamental TE mode", "enabled": True,
            }
        )

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def reset_ports(self) -> None:
        candidate = deepcopy(self.component)
        seed_simulation_ports(candidate, replace=True)
        self.table.setRowCount(0)
        for port in candidate.get("simulation_ports", []):
            self._append_port(port)

    def result_ports(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in range(self.table.rowCount()):
            def text(column: int, fallback: str = "") -> str:
                widget = self.table.cellWidget(row, column)
                if isinstance(widget, QComboBox):
                    return widget.currentText().strip()
                item = self.table.item(row, column)
                return item.text().strip() if item else fallback

            name = text(1, f"opt_{row + 1}") or f"opt_{row + 1}"
            if name in seen:
                name = f"{name}_{row + 1}"
            seen.add(name)
            try:
                loc = min(1.0, max(0.0, float(text(3, "0.5"))))
                order = max(1, int(float(text(5, str(row + 1)))))
                x, y = float(text(7, "0")), float(text(8, "0"))
                distance = float(text(9, "0"))
                angle = float(text(10, "0")) % 360.0
                span = max(1e-6, float(text(11, "2")))
                z_span = max(0.0, float(text(12, "2")))
                theta = float(text(14, "0"))
                phi = float(text(15, str(angle))) % 360.0
                core_diameter = max(1e-6, float(text(16, "9")))
                core_index = max(1.0, float(text(17, "1.44427")))
                cladding_index = max(1.0, float(text(18, "1.43482")))
            except ValueError as exc:
                raise ValueError(f"Port row {row + 1} contains a non-numeric coordinate, angle, order, or span.") from exc
            result.append(
                {
                    "name": name,
                    "center": [x, y],
                    "outward_orientation_deg": angle,
                    "domain": text(19, "optical") or "optical",
                    "dir": text(2, "Bidirectional") or "Bidirectional",
                    "loc": loc,
                    "pos": text(4, "Right") or "Right",
                    "order": order,
                    "port geometry": (text(6, "surface") or "surface").lower(),
                    "distance_um": distance,
                    "span_um": span,
                    "z_span_um": z_span,
                    "mode": text(13, "fundamental TE mode") or "fundamental TE mode",
                    "angle theta": theta,
                    "angle phi": phi,
                    "core diameter_um": core_diameter,
                    "core index": core_index,
                    "cladding index": cladding_index,
                    "cladding diameter_um": 50.0,
                    "fiber length_um": 20.0,
                    "enabled": self.table.item(row, 0).checkState() == Qt.CheckState.Checked,
                }
            )
        return result


class LumericalSweepDialog(QDialog):
    """Choose one or more exact component-JSON parameters and Cartesian ranges."""

    def __init__(
        self,
        component: dict[str, Any],
        saved: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.component = component
        self.saved = deepcopy(saved or {})
        self.parameters = sweepable_component_parameters(component)
        self.editors: list[dict[str, Any]] = []
        self._accepted_spec: dict[str, Any] | None = None
        self.setWindowTitle(f"Lumerical parameter sweep — {component.get('kind', 'component')}")
        self.resize(1120, min(880, max(430, 250 + 52 * len(self.parameters))))
        self.setMinimumSize(920, 430)
        outer = QVBoxLayout(self)
        self.outer_layout = outer
        explanation = QLabel(
            "Select the geometry parameters to sweep. Start and stop are inclusive; Points controls "
            "how many values are simulated on that axis. Multiple checked rows form a Cartesian grid. "
            "The exact JSON parameter name is shown in brackets."
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        speed_note = QLabel(
            "Fast mode: one licence session, one static model build, one nominal FSP, then in-place "
            "geometry updates and sequential GPU solves. Numerical spectra are downloaded once and all plots are made on CPU."
        )
        speed_note.setWordWrap(True)
        speed_note.setStyleSheet("color:#0f766e; font-weight:600;")
        outer.addWidget(speed_note)
        self.speed_note = speed_note

        self.table = QTableWidget(len(self.parameters), 7)
        self.table.setHorizontalHeaderLabels(
            ["Sweep", "Parameter [JSON name]", "Current", "Start", "Stop", "Points", "Axis runs"]
        )
        self.table.verticalHeader().setDefaultSectionSize(46)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        saved_axes = {
            str(axis.get("parameter", "")): axis
            for axis in self.saved.get("axes", [])
        }
        component_kind = str(component.get("kind", ""))
        default_keys = (
            {"pitch", "duty_cycle", "fill_factor"}
            if component_kind in {"Grating coupler", "GC-SOI"}
            else {"mmi_length"}
            if component_kind == "1x2 MMI"
            else ({str(self.parameters[0]["parameter"])} if self.parameters else set())
        )
        for row, metadata in enumerate(self.parameters):
            key = str(metadata["parameter"])
            value_type = str(metadata["value_type"])
            nominal = metadata["nominal"]
            saved_axis = saved_axes.get(key)
            saved_values = list(saved_axis.get("values", [])) if saved_axis else []
            checked = bool(saved_axis) or (not saved_axes and key in default_keys)
            check = QCheckBox()
            check.setChecked(checked)
            check.setToolTip("Include this parameter in the Cartesian sweep")
            self.table.setCellWidget(row, 0, check)
            item = QTableWidgetItem(f"{metadata['label']}  [{key}]")
            item.setToolTip(f"Exact project JSON key: {key}")
            self.table.setItem(row, 1, item)
            self.table.setItem(row, 2, QTableWidgetItem(str(nominal)))

            if saved_values:
                start_value, stop_value = saved_values[0], saved_values[-1]
                point_count = len(saved_values)
            elif value_type == "int":
                start_value = max(1, int(nominal) - 1)
                stop_value = max(start_value + 1, int(nominal) + 1)
                point_count = int(stop_value - start_value + 1)
            else:
                numeric = float(nominal)
                if key in {"duty_cycle", "fill_factor"}:
                    start_value, stop_value = max(0.001, numeric - 0.05), min(0.999, numeric + 0.05)
                else:
                    delta = max(abs(numeric) * 0.05, 0.01)
                    start_value, stop_value = numeric - delta, numeric + delta
                point_count = 3

            if value_type == "int":
                start_box = QSpinBox(); start_box.setRange(-1_000_000_000, 1_000_000_000); start_box.setValue(int(start_value))
                stop_box = QSpinBox(); stop_box.setRange(-1_000_000_000, 1_000_000_000); stop_box.setValue(int(stop_value))
            else:
                start_box = QDoubleSpinBox(); start_box.setRange(-1e9, 1e9); start_box.setDecimals(9); start_box.setValue(float(start_value))
                stop_box = QDoubleSpinBox(); stop_box.setRange(-1e9, 1e9); stop_box.setDecimals(9); stop_box.setValue(float(stop_value))
            for box in (start_box, stop_box):
                box.setMinimumWidth(125)
                box.setKeyboardTracking(False)
            points_box = QSpinBox()
            points_box.setRange(2, LUMERICAL_SWEEP_MAX_RUNS)
            points_box.setValue(max(2, int(point_count)))
            points_box.setMinimumWidth(82)
            runs_item = QTableWidgetItem(str(points_box.value()))
            runs_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, runs_item)
            self.table.setCellWidget(row, 3, start_box)
            self.table.setCellWidget(row, 4, stop_box)
            self.table.setCellWidget(row, 5, points_box)
            editor = {
                **metadata,
                "check": check,
                "start": start_box,
                "stop": stop_box,
                "points": points_box,
                "runs_item": runs_item,
            }
            self.editors.append(editor)
            check.toggled.connect(self._update_run_count)
            points_box.valueChanged.connect(self._update_run_count)
        outer.addWidget(self.table, 1)

        self.run_count = QLabel()
        self.run_count.setWordWrap(True)
        outer.addWidget(self.run_count)
        output_note = QLabel(
            (
                "MMI result names include every selected value. Each case stores upper/input, "
                "lower/input, total/input, and the 50/50 splitting fractions in linear power. "
                "The sweep summary uses upper-branch power divided by measured input power."
            )
            if component_kind == "1x2 MMI"
            else (
                "Grating result names include every selected value, for example CE-P=0.75-F=0.56.png. "
                "A maximum-CE summary plot and CSV are generated after all runs."
            )
        )
        output_note.setWordWrap(True)
        outer.addWidget(output_note)
        self.output_note = output_note
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue to stack and FDTD settings…")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.buttons = buttons
        self._update_run_count()

    def _update_run_count(self, *_args) -> None:
        total = 1
        selected = 0
        for editor in self.editors:
            editor["runs_item"].setText(str(editor["points"].value()))
            if editor["check"].isChecked():
                selected += 1
                total *= int(editor["points"].value())
        if not selected:
            self.run_count.setText("Select at least one parameter.")
            self.run_count.setStyleSheet("color:#b91c1c; font-weight:600;")
        elif total > LUMERICAL_SWEEP_MAX_RUNS:
            self.run_count.setText(
                f"{total} total GPU simulations — above the {LUMERICAL_SWEEP_MAX_RUNS}-run safety limit."
            )
            self.run_count.setStyleSheet("color:#b91c1c; font-weight:600;")
        elif total > LUMERICAL_SWEEP_WARNING_RUNS:
            self.run_count.setText(
                f"{total} total GPU simulations. This is a large sweep; compact checkpoint files permit restart."
            )
            self.run_count.setStyleSheet("color:#b45309; font-weight:600;")
        else:
            self.run_count.setText(f"{total} total GPU simulation{'s' if total != 1 else ''}.")
            self.run_count.setStyleSheet("color:#0f766e; font-weight:600;")

    def _raw_axes(self) -> list[dict[str, Any]]:
        axes = []
        for editor in self.editors:
            if not editor["check"].isChecked():
                continue
            start = editor["start"].value()
            stop = editor["stop"].value()
            point_count = int(editor["points"].value())
            values = np.linspace(float(start), float(stop), point_count).tolist()
            axes.append({"parameter": editor["parameter"], "values": values})
        return axes

    def _validate_and_accept(self) -> None:
        try:
            spec = normalize_lumerical_sweep_spec(self.component, self._raw_axes())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical sweep", str(exc))
            return
        if int(spec["point_count"]) > LUMERICAL_SWEEP_WARNING_RUNS:
            answer = QMessageBox.question(
                self,
                "Confirm large GPU sweep",
                f"This will run {spec['point_count']} separate 3D GPU simulations. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._accepted_spec = spec
        self.accept()

    def sweep_spec(self) -> dict[str, Any]:
        if self._accepted_spec is None:
            return normalize_lumerical_sweep_spec(self.component, self._raw_axes())
        return deepcopy(self._accepted_spec)


class LumericalOptimizationDialog(QDialog):
    """Choose adjoint geometry/alignment parameters and the target band."""

    def __init__(
        self,
        component: dict[str, Any],
        saved: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.component = component
        self.saved = deepcopy(saved or {})
        self.parameters = adjoint_optimizable_component_parameters(component)
        self.editors: list[dict[str, Any]] = []
        self._accepted_spec: dict[str, Any] | None = None
        component_kind = str(component.get("kind", "component"))
        self.setWindowTitle(f"Lumerical adjoint optimization — {component_kind}")
        self.resize(1120, min(900, max(610, 390 + 52 * len(self.parameters))))
        self.setMinimumSize(920, 610)

        outer = QVBoxLayout(self)
        explanation = QLabel(
            "Select the continuous geometry or fiber-alignment parameters that the optimizer may change. "
            "Minimum and maximum are hard bounds; Current is the initial value. The exact project "
            "JSON parameter name is shown in brackets."
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        speed_note = QLabel(
            (
                "Device-geometry variables use true shape-adjoint gradients. angle_theta and "
                "fiber_offset use a bounded GPU forward-solve alignment stage, where the fiber, "
                "source, and passive power plane move together, and are then frozen for LumOpt."
            )
            if component_kind in {"Grating coupler", "GC-SOI"}
            else (
                "True shape-adjoint mode uses one forward and one adjoint electromagnetic solve per "
                "iteration, independent of the number of selected continuous geometry parameters."
            )
        )
        speed_note.setWordWrap(True)
        speed_note.setStyleSheet("color:#0f766e; font-weight:600;")
        outer.addWidget(speed_note)

        self.table = QTableWidget(len(self.parameters), 5)
        self.table.setHorizontalHeaderLabels(
            ["Optimize", "Parameter [JSON name]", "Current", "Minimum", "Maximum"]
        )
        self.table.verticalHeader().setDefaultSectionSize(46)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        saved_parameters = {
            str(row.get("parameter", "")): row
            for row in self.saved.get("parameters", [])
        }
        default_keys = (
            {"pitch", "duty_cycle", "fill_factor"}
            if component_kind in {"Grating coupler", "GC-SOI"}
            else {"mmi_length", "mmi_width"}
            if component_kind == "1x2 MMI"
            else set()
        )
        for row, metadata in enumerate(self.parameters):
            key = str(metadata["parameter"])
            nominal = float(metadata.get("nominal", metadata.get("initial", 0.0)))
            saved_row = saved_parameters.get(key)
            checked = bool(saved_row) or (not saved_parameters and key in default_keys)
            check = QCheckBox()
            check.setChecked(checked)
            check.setToolTip("Allow the adjoint optimizer to change this geometry parameter")
            self.table.setCellWidget(row, 0, check)

            label = QTableWidgetItem(f"{metadata.get('label', key)}  [{key}]")
            label.setToolTip(f"Exact project JSON key: {key}")
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, label)
            current = QTableWidgetItem(format(nominal, ".9g"))
            current.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            current.setFlags(current.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, current)

            if saved_row:
                minimum_value = float(
                    saved_row.get("minimum", saved_row.get("min", nominal))
                )
                maximum_value = float(
                    saved_row.get("maximum", saved_row.get("max", nominal))
                )
            elif key in {"duty_cycle", "fill_factor"}:
                minimum_value = max(0.001, nominal - 0.05)
                maximum_value = min(0.999, nominal + 0.05)
            else:
                delta = max(abs(nominal) * 0.05, 0.01)
                minimum_value = nominal - delta
                maximum_value = nominal + delta
                if component_kind == "1x2 MMI" and key == "mmi_width":
                    # Both symmetric output tapers must fit across the MMI body:
                    # mmi_width >= port_sep + taper_width.  The stock 6 um body
                    # has only 50 nm of lower-side clearance, so a generic -5%
                    # default would make the very first export fail preflight.
                    required_width = float(
                        component.get("params", {}).get("port_sep", 0.0)
                    ) + float(component.get("params", {}).get("taper_width", 0.0))
                    minimum_value = max(minimum_value, required_width)

            minimum = QDoubleSpinBox()
            maximum = QDoubleSpinBox()
            for box, value in ((minimum, minimum_value), (maximum, maximum_value)):
                box.setRange(-1e9, 1e9)
                box.setDecimals(9)
                box.setValue(float(value))
                box.setMinimumWidth(145)
                box.setKeyboardTracking(False)
            self.table.setCellWidget(row, 3, minimum)
            self.table.setCellWidget(row, 4, maximum)
            self.editors.append(
                {
                    **metadata,
                    "check": check,
                    "minimum": minimum,
                    "maximum": maximum,
                    "initial": nominal,
                }
            )
        outer.addWidget(self.table, 1)

        objective_group = QWidget()
        objective_form = QFormLayout(objective_group)
        objective = dict(self.saved.get("objective", {}))
        default_center_um = 1.55 if component_kind == "GC-SOI" else 1.30
        center_um = float(
            objective.get(
                "center_wavelength_um",
                self.saved.get("center_wavelength_um", default_center_um),
            )
        )
        bandwidth_nm = float(
            objective.get("bandwidth_nm", self.saved.get("bandwidth_nm", 100.0))
        )
        wavelength_points = int(
            objective.get(
                "wavelength_points",
                self.saved.get("wavelength_points", 7 if bandwidth_nm > 0.0 else 1),
            )
        )
        optimizer = dict(self.saved.get("optimizer", {}))

        self.center_wavelength = QDoubleSpinBox()
        self.center_wavelength.setRange(0.01, 1000.0)
        self.center_wavelength.setDecimals(9)
        self.center_wavelength.setValue(center_um)
        self.center_wavelength.setSuffix(" µm")
        self.center_wavelength.setMinimumWidth(180)
        objective_form.addRow("Center wavelength", self.center_wavelength)

        self.bandwidth = QDoubleSpinBox()
        self.bandwidth.setRange(0.0, 1e6)
        self.bandwidth.setDecimals(6)
        self.bandwidth.setValue(max(0.0, bandwidth_nm))
        self.bandwidth.setSuffix(" nm")
        self.bandwidth.setSpecialValueText("Single wavelength")
        self.bandwidth.setMinimumWidth(180)
        objective_form.addRow("Optimization bandwidth", self.bandwidth)

        self.wavelength_points = QSpinBox()
        self.wavelength_points.setRange(1, 1001)
        self.wavelength_points.setValue(max(1, wavelength_points))
        self.wavelength_points.setMinimumWidth(180)
        self.wavelength_points.setToolTip(
            "Spectral samples used by the broadband adjoint objective. Keep this modest for 3D optimization."
        )
        objective_form.addRow("Objective wavelength samples", self.wavelength_points)

        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 10000)
        self.max_iterations.setValue(
            max(1, int(optimizer.get("max_iterations", self.saved.get("max_iterations", 30))))
        )
        self.max_iterations.setMinimumWidth(180)
        objective_form.addRow("Maximum optimizer iterations", self.max_iterations)
        outer.addWidget(objective_group)

        objective_note = QLabel(
            (
                "Objective: maximize fiber-to-waveguide coupling efficiency across the selected band. "
                "The final notebook also reports the measured forward-fiber normalization."
            )
            if component_kind in {"Grating coupler", "GC-SOI"}
            else (
                "Objective: maximize top/upper output-branch fundamental-TE power divided by input power. "
                "The MMI parameterization remains mirror-symmetric, and both output branches are reported."
            )
        )
        objective_note.setWordWrap(True)
        outer.addWidget(objective_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Continue to stack and FDTD settings…"
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.bandwidth.valueChanged.connect(self._bandwidth_changed)
        self._bandwidth_changed(self.bandwidth.value())

    def _bandwidth_changed(self, bandwidth_nm: float) -> None:
        single_wavelength = float(bandwidth_nm) <= 0.0
        if single_wavelength:
            self.wavelength_points.setValue(1)
        elif self.wavelength_points.value() < 2:
            self.wavelength_points.setValue(7)
        self.wavelength_points.setEnabled(not single_wavelength)

    def _raw_parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "parameter": str(editor["parameter"]),
                "initial": float(editor["initial"]),
                "minimum": float(editor["minimum"].value()),
                "maximum": float(editor["maximum"].value()),
            }
            for editor in self.editors
            if editor["check"].isChecked()
        ]

    def _normalized_spec(self) -> dict[str, Any]:
        return normalize_lumerical_optimization_spec(
            self.component,
            self._raw_parameters(),
            center_wavelength_um=float(self.center_wavelength.value()),
            bandwidth_nm=float(self.bandwidth.value()),
            wavelength_points=int(self.wavelength_points.value()),
            max_iterations=int(self.max_iterations.value()),
        )

    def _validate_and_accept(self) -> None:
        try:
            self._accepted_spec = self._normalized_spec()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical optimization", str(exc))
            return
        self.accept()

    def optimization_spec(self) -> dict[str, Any]:
        if self._accepted_spec is None:
            return self._normalized_spec()
        return deepcopy(self._accepted_spec)


class LumericalMultigpuSweepDialog(LumericalSweepDialog):
    """Collect sweep axes plus the independent multi-A100 worker allocation."""

    def __init__(
        self,
        component: dict[str, Any],
        saved: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(component, saved=saved, parent=parent)
        self.setWindowTitle(
            f"Lumerical sweep-multithread — {component.get('kind', 'component')}"
        )
        self.resize(1120, min(940, max(650, 390 + 52 * len(self.parameters))))
        self.setMinimumSize(920, 610)
        self.speed_note.setText(
            "Multi-GPU mode: independent persistent workers share one prepared model and distribute "
            "sweep points across A100 nodes. This is a separate export and does not change the "
            "single-session sequential sweep."
        )

        stored_parallel = dict(self.saved.get("parallel", {}))
        self.parallel_panel = QWidget()
        parallel_layout = QVBoxLayout(self.parallel_panel)
        parallel_layout.setContentsMargins(0, 8, 0, 8)
        heading = QLabel("Parallel A100 allocation")
        heading.setStyleSheet("font-weight:700;")
        parallel_layout.addWidget(heading)
        parallel_form = QFormLayout()
        self.a100_nodes = QSpinBox()
        self.a100_nodes.setRange(1, 64)
        self.a100_nodes.setValue(
            max(1, min(64, int(stored_parallel.get("node_count", 8))))
        )
        self.a100_nodes.setSuffix(" A100 node(s)")
        self.a100_nodes.setToolTip(
            "Number of existing, idle, pre-authorized 1xA100 nodes made available to the exported notebook."
        )
        parallel_form.addRow("A100 nodes", self.a100_nodes)

        self.simulations_per_gpu = QSpinBox()
        # Production safety policy: each A100 owns exactly one independent
        # FDTD process.  Do not honor older saved experiments that requested
        # two processes on a GPU; safe sharing needs model-memory and licence
        # preflight that the exporter does not implement yet.
        self.simulations_per_gpu.setRange(1, 1)
        self.simulations_per_gpu.setValue(1)
        self.simulations_per_gpu.setEnabled(False)
        self.simulations_per_gpu.setSuffix(" simulation(s) / GPU")
        self.simulations_per_gpu.setToolTip(
            "Fixed at one independent Lumerical FDTD process per A100. Multiple processes on one GPU "
            "remain unavailable until model-memory and licence preflight is implemented."
        )
        parallel_form.addRow(
            "Simulations per GPU (production limit)", self.simulations_per_gpu
        )
        parallel_layout.addLayout(parallel_form)

        self.parallel_summary = QLabel()
        self.parallel_summary.setWordWrap(True)
        parallel_layout.addWidget(self.parallel_summary)
        warning = QLabel(
            "Production safeguard: each selected A100 runs exactly one independent simulation. "
            "Multiple Lumerical processes on one GPU are intentionally unavailable until "
            "model-memory and licence preflight is implemented."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#0f766e; font-weight:600;")
        parallel_layout.addWidget(warning)

        insert_at = self.outer_layout.indexOf(self.output_note)
        self.outer_layout.insertWidget(insert_at, self.parallel_panel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Continue to stack parameters…"
        )
        self.a100_nodes.valueChanged.connect(self._update_parallel_summary)
        self.simulations_per_gpu.valueChanged.connect(self._update_parallel_summary)
        self._update_parallel_summary()

    def _update_parallel_summary(self, *_args) -> None:
        slots = int(self.a100_nodes.value()) * int(self.simulations_per_gpu.value())
        self.parallel_summary.setText(
            f"Up to {slots} sweep simulations can run at the same time: one independent process "
            "on each selected A100. The actual worker count is limited by the sweep-point count "
            "and the A100 nodes available to the launcher."
        )
        self.parallel_summary.setStyleSheet("color:#0f766e; font-weight:600;")

    def parallel_configuration(self) -> dict[str, int]:
        node_count = int(self.a100_nodes.value())
        simulations_per_gpu = 1
        return {
            "node_count": node_count,
            "simulations_per_gpu": simulations_per_gpu,
            "max_parallel_simulations": node_count * simulations_per_gpu,
        }


class LumericalExportDialog(QDialog):
    """Collect geometry, stack, port, solver, and resource choices."""

    def __init__(
        self,
        all_components: list[dict[str, Any]],
        scope_options: list[tuple[str, list[int]]],
        saved: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.all_components = all_components
        self.scope_options = scope_options
        self.saved = deepcopy(saved or {})
        self.setWindowTitle("Export Lumerical simulation notebook")
        self.resize(1420, 860)
        self.setMinimumSize(1180, 720)
        outer = QVBoxLayout(self)
        summary = QLabel(
            "Choose exactly what geometry to embed, define the bottom-to-top material stack, "
            "and configure the 3D GPU solve. The notebook is generated even when some layers are disabled."
        )
        summary.setWordWrap(True)
        outer.addWidget(summary)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self._make_geometry_tab()
        self._make_stack_tab()
        self._make_solver_tab()
        self._make_preview_tab()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Choose notebook file…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _make_geometry_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.scope = QComboBox()
        for label, uids in self.scope_options:
            self.scope.addItem(label, list(uids))
        form.addRow("Geometry to export", self.scope)
        layout.addLayout(form)
        self.geometry_table = QTableWidget(0, 4)
        self.geometry_table.setHorizontalHeaderLabels(["Include", "GDS layer", "Datatype", "Editor name"])
        self.geometry_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        saved_layers = {tuple(map(int, value)) for value in self.saved.get("included_layers", [])}
        candidates = [component for component in self.all_components if component.get("kind") != "E-beam multipass"]
        layers = available_geometry_layers(candidates)
        for layer, datatype in layers:
            row = self.geometry_table.rowCount()
            self.geometry_table.insertRow(row)
            enabled = not saved_layers or (layer, datatype) in saved_layers
            self.geometry_table.setItem(row, 0, _checked_item(enabled))
            self.geometry_table.setItem(row, 1, QTableWidgetItem(str(layer)))
            self.geometry_table.setItem(row, 2, QTableWidgetItem(str(datatype)))
            self.geometry_table.setItem(row, 3, QTableWidgetItem(LAYER_NAME_MAP.get(layer, f"Layer {layer}")))
        layout.addWidget(QLabel("Only checked GDS layers are embedded as polygon arrays in the notebook."))
        layout.addWidget(self.geometry_table)
        self.include_ports = QCheckBox("Include the ports placed from the Ports & monitors library")
        self.include_ports.setChecked(bool(self.saved.get("include_ports", True)))
        layout.addWidget(self.include_ports)
        placed_note = QLabel(
            "Straight waveguides, tapers, bends, MMIs, and grating couplers receive movable starter FDTD ports when they are added. "
            "Drag any starter port independently in the top-view editor to detach it from its automatic position; you can also edit, remove, "
            "or add ports from the left library. A grating coupler also receives a separate tilted fiber "
            "core/cladding group and a standard Z-axis FDTD receiver port through it. These simulation objects never enter GDS."
        )
        placed_note.setWordWrap(True)
        layout.addWidget(placed_note)
        self.tabs.addTab(tab, "Geometry, ports and monitors")

    def _make_stack_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(STACK_PRESETS.keys())
        self.preset.setCurrentText(str(self.saved.get("stack_preset", "TFLN on SiO2")))
        load_button = QPushButton("Load preset")
        add_button = QPushButton("Add row")
        remove_button = QPushButton("Remove selected")
        controls.addWidget(QLabel("Starting stack"))
        controls.addWidget(self.preset, 1)
        controls.addWidget(load_button)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        layout.addLayout(controls)
        self.stack_table = QTableWidget(0, 11)
        self.stack_table.setHorizontalHeaderLabels(
            [
                "Layer name",
                "Lumerical material",
                "Thickness (µm)",
                "Etch depth (µm)",
                "Sidewall angle (°)",
                "Layer type",
                "GDS layers",
                "Unetched slab extent",
                "Mesh factor × λ/n",
                "Mesh order",
                "Conformal fill",
            ]
        )
        self.stack_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stack_table.setMinimumHeight(430)
        self.stack_table.verticalHeader().setDefaultSectionSize(48)
        self.stack_table.horizontalHeader().setMinimumSectionSize(120)
        for column, width in enumerate((205, 290, 165, 165, 180, 215, 135, 205, 165, 130, 145)):
            self.stack_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.stack_table.setColumnWidth(column, width)
        self.stack_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stack_table)
        note = QLabel(
            "Rows are ordered bottom-to-top. Full-film rows become simulation slabs. Exported cross-section rows use the selected "
            "layout polygons, etch depth, and waveguide sidewall angle. 90° is vertical; below 90° is wider at the bottom. "
            "Etch depth 0 keeps a full unetched film; etch depth equal to thickness is a full etch. A thickness of 0 means the layer is absent. "
            "For a partially etched cross-section, Unetched slab extent can keep the slab across the full FDTD plane or only beneath the selected GDS geometry. "
            "Mesh factor × λ/n makes the notebook calculate the layer mesh from the shortest simulated wavelength and that material's dispersive index. For example, 0.5 means 0.5 × λ₀/n; 0 uses Lumerical's automatic mesh. "
            "Mesh order controls overlapping-material priority; lower numbers win. "
            "Conformal fill extends a background/cladding layer down into the etched openings of the patterned layer below. "
            "Air is available as a material. Very large first/last background thicknesses are allowed and are cropped by the draggable FDTD domain so those media extend through the PML."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        load_button.clicked.connect(self.load_preset)
        add_button.clicked.connect(
            lambda: self._append_stack_row(
                {
                    "name": "New layer",
                    "material": MATERIAL_CHOICES[0],
                    "thickness_um": 0.0,
                    "etch_depth_um": 0.0,
                    "sidewall_angle_deg": 90.0,
                    "role": "background",
                    "gds_layer": 0,
                }
            )
        )
        remove_button.clicked.connect(self.remove_stack_rows)
        stack = self.saved.get("material_stack") or default_stack(self.preset.currentText())
        for row in stack:
            self._append_stack_row(row)
        self.tabs.addTab(tab, "Material stack")

    def _append_stack_row(self, data: dict[str, Any]) -> None:
        row = self.stack_table.rowCount()
        self.stack_table.insertRow(row)
        self.stack_table.setItem(row, 0, QTableWidgetItem(str(data.get("name", f"Layer {row + 1}"))))
        material = QComboBox()
        material.setEditable(True)
        material.addItems(MATERIAL_CHOICES)
        material.setCurrentText(str(data.get("material", MATERIAL_CHOICES[0])))
        material.setMinimumSize(270, 38)
        self.stack_table.setCellWidget(row, 1, material)
        thickness = QDoubleSpinBox()
        thickness.setRange(0.0, 1e6)
        thickness.setDecimals(9)
        thickness.setValue(max(0.0, float(data.get("thickness_um", 0.0))))
        thickness.setMinimumSize(155, 38)
        self.stack_table.setCellWidget(row, 2, thickness)
        etch = QDoubleSpinBox()
        etch.setRange(0.0, 1e6)
        etch.setDecimals(9)
        default_etch = float(data.get("thickness_um", 0.0)) if str(data.get("role", "background")) == "geometry" else 0.0
        etch.setValue(max(0.0, float(data.get("etch_depth_um", default_etch))))
        etch.setMinimumSize(155, 38)
        self.stack_table.setCellWidget(row, 3, etch)
        sidewall = QDoubleSpinBox()
        sidewall.setRange(0.001, 179.999)
        sidewall.setDecimals(3)
        sidewall.setSingleStep(1.0)
        sidewall.setSuffix(" °")
        sidewall.setValue(min(179.999, max(0.001, float(data.get("sidewall_angle_deg", 90.0)))))
        sidewall.setMinimumSize(155, 38)
        sidewall.setToolTip("90° is vertical. Below 90° makes the cross-section wider at the bottom; above 90° makes it narrower.")
        self.stack_table.setCellWidget(row, 4, sidewall)
        role = QComboBox()
        role.addItems(["Full film / background", "Exported cross-section"])
        role.setCurrentText(
            "Exported cross-section"
            if str(data.get("role", "background")).lower() == "geometry"
            else "Full film / background"
        )
        role.setMinimumSize(190, 38)
        self.stack_table.setCellWidget(row, 5, role)
        layers = data.get("gds_layers", [data.get("gds_layer", 0)])
        if isinstance(layers, (str, int, float)):
            layers = [layers]
        layer_entry = QLineEdit(", ".join(str(int(value)) for value in layers))
        layer_entry.setPlaceholderText("Example: 1, 2")
        layer_entry.setToolTip("Every listed GDS layer receives this patterned material at the same vertical position.")
        layer_entry.setMinimumSize(115, 38)
        self.stack_table.setCellWidget(row, 6, layer_entry)
        slab_extent = QComboBox()
        slab_extent.addItems(["Full FDTD plane", "Under geometry"])
        slab_extent.setCurrentText(
            "Under geometry"
            if str(data.get("slab_extent", "full")).strip().lower() == "geometry"
            else "Full FDTD plane"
        )
        slab_extent.setMinimumSize(180, 38)
        slab_extent.setToolTip(
            "Controls the lateral extent of the unetched slab below a partially etched exported cross-section."
        )
        self.stack_table.setCellWidget(row, 7, slab_extent)
        mesh_factor = QDoubleSpinBox()
        mesh_factor.setRange(0.0, 1000.0)
        mesh_factor.setDecimals(6)
        mesh_factor.setSingleStep(0.05)
        mesh_factor.setSpecialValueText("Automatic")
        mesh_factor.setValue(max(0.0, float(data.get("mesh_factor", 0.2))))
        mesh_factor.setMinimumSize(155, 38)
        mesh_factor.setToolTip(
            "Isotropic mesh step as a factor of λ₀/n at the shortest simulated wavelength. For anisotropic media, n is the largest index component."
        )
        self.stack_table.setCellWidget(row, 8, mesh_factor)
        mesh_order = QSpinBox()
        mesh_order.setRange(1, 1000)
        mesh_order.setValue(max(1, int(data.get("mesh_order", 3 if bool(data.get("conformal", False)) else 2))))
        mesh_order.setMinimumSize(115, 38)
        mesh_order.setToolTip("Lumerical material-overlap priority. Lower mesh-order numbers take priority.")
        self.stack_table.setCellWidget(row, 9, mesh_order)
        self.stack_table.setItem(row, 10, _checked_item(bool(data.get("conformal", False))))
        if hasattr(self, "cross_section_preview"):
            self._connect_stack_preview_signals()
            self._refresh_previews()

    def load_preset(self) -> None:
        self.stack_table.setRowCount(0)
        for row in default_stack(self.preset.currentText()):
            self._append_stack_row(row)
        self._refresh_previews()

    def remove_stack_rows(self) -> None:
        rows = sorted({index.row() for index in self.stack_table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.stack_table.removeRow(row)
        self._refresh_previews()

    def _make_solver_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.wavelength_start = QDoubleSpinBox(); self.wavelength_start.setRange(0.01, 1000); self.wavelength_start.setDecimals(6); self.wavelength_start.setValue(float(self.saved.get("wavelength_start_um", 1.25)))
        self.wavelength_stop = QDoubleSpinBox(); self.wavelength_stop.setRange(0.01, 1000); self.wavelength_stop.setDecimals(6); self.wavelength_stop.setValue(float(self.saved.get("wavelength_stop_um", 1.35)))
        saved_domain = dict(self.saved.get("domain_padding_um", {}))
        legacy_xy = float(self.saved.get("xy_padding_um", 2.0))
        legacy_z = float(self.saved.get("z_padding_um", 1.0))
        self.domain_padding_spins: dict[str, QDoubleSpinBox] = {}
        for key, fallback in (
            ("x_min", legacy_xy), ("x_max", legacy_xy),
            ("y_min", legacy_xy), ("y_max", legacy_xy),
            ("z_min", legacy_z), ("z_max", legacy_z),
        ):
            spin = QDoubleSpinBox(); spin.setRange(-1e6, 1e6); spin.setDecimals(6); spin.setValue(float(saved_domain.get(key, fallback)))
            spin.setSuffix(" µm")
            self.domain_padding_spins[key] = spin
        self.xy_padding = self.domain_padding_spins["x_min"]
        self.z_padding = self.domain_padding_spins["z_min"]
        self.mesh_accuracy = QSpinBox(); self.mesh_accuracy.setRange(1, 8); self.mesh_accuracy.setValue(int(self.saved.get("mesh_accuracy", 2)))
        self.frequency_points = QSpinBox(); self.frequency_points.setRange(3, 10001); self.frequency_points.setValue(int(self.saved.get("frequency_points", 31)))
        self.frequency_points.setToolTip("31 samples is the quick grating-coupler default. Increase this only for a final high-resolution spectrum.")
        self.dt_stability = QDoubleSpinBox(); self.dt_stability.setRange(0.1, 0.99); self.dt_stability.setDecimals(3); self.dt_stability.setSingleStep(0.05); self.dt_stability.setValue(float(self.saved.get("dt_stability_factor", 0.99)))
        self.pml_profile = QComboBox(); self.pml_profile.addItems(["Standard", "Stabilized"]); self.pml_profile.setCurrentText(str(self.saved.get("pml_profile", "Standard")).title())
        saved_simulation_time = float(self.saved.get("simulation_time_fs", 10000.0))
        if "auto_shutoff_min" not in self.saved and abs(saved_simulation_time - 2000.0) < 1e-9:
            # Migrate the former 2 ps grating default together with the new
            # explicit convergence threshold so the dialog matches exports.
            saved_simulation_time = 10000.0
        self.simulation_time = QDoubleSpinBox(); self.simulation_time.setRange(1, 1e9); self.simulation_time.setDecimals(3); self.simulation_time.setValue(saved_simulation_time)
        self.auto_shutoff_min = QDoubleSpinBox(); self.auto_shutoff_min.setRange(1e-12, 1.0); self.auto_shutoff_min.setDecimals(12); self.auto_shutoff_min.setSingleStep(1e-6); self.auto_shutoff_min.setValue(float(self.saved.get("auto_shutoff_min", 1e-6)))
        self.auto_shutoff_min.setToolTip("Stop early only after the remaining field energy falls below this fraction. The grating default is 1e-6.")
        self.pml_geometry_overlap = QDoubleSpinBox(); self.pml_geometry_overlap.setRange(0.0, 1e4); self.pml_geometry_overlap.setDecimals(6); self.pml_geometry_overlap.setSuffix(" µm"); self.pml_geometry_overlap.setValue(float(self.saved.get("pml_geometry_overlap_um", 1.0)))
        self.pml_geometry_overlap.setToolTip("Distance ported waveguides and outer stack media continue beyond each FDTD boundary.")
        self.dimension = QLabel("3D — required for every exported simulation")
        self.dimension.setWordWrap(True)
        self.resource_mode = QComboBox(); self.resource_mode.addItems(["GPU", "CPU"]); self.resource_mode.setCurrentText(str(self.saved.get("resource_mode", "GPU")))
        self.tfln_crystal_cut = QComboBox(); self.tfln_crystal_cut.addItems(["X", "Y", "Z"]); self.tfln_crystal_cut.setCurrentText(str(self.saved.get("tfln_crystal_cut", "X")).upper())
        self.tfln_temperature = QDoubleSpinBox(); self.tfln_temperature.setRange(1.0, 2000.0); self.tfln_temperature.setDecimals(3); self.tfln_temperature.setSuffix(" K"); self.tfln_temperature.setValue(float(self.saved.get("tfln_temperature_K", 296.3)))
        self.project_file = QLineEdit(str(self.saved.get("project_file", "exported_component.fsp")))
        self.hide_cad = QCheckBox("Start Lumerical without showing the CAD window"); self.hide_cad.setChecked(bool(self.saved.get("hide_cad", False)))
        self.run_after_build = QCheckBox("Run automatically after building and saving the .fsp project"); self.run_after_build.setChecked(bool(self.saved.get("run_after_build", True)))
        self.official_gc_domain = QCheckBox("Use compact official Ansys GC-SOI FDTD settings")
        self.official_gc_domain.setChecked(bool(self.saved.get("official_gc_domain", False)))
        self.official_gc_domain.setToolTip(
            "Keeps manually typed/draggable bounds, permits the grating and waveguide to cross the PML, and uses the official boundary setup."
        )
        self.use_y_antisymmetry = QCheckBox("Use Y=0 anti-symmetry for a symmetric grating")
        self.use_y_antisymmetry.setChecked(bool(self.saved.get("use_y_antisymmetry", False)))
        self.use_y_antisymmetry.setToolTip(
            "Cuts the simulated grating width in half. Turn this off if the geometry, fiber placement, or material stack is not symmetric about local Y=0."
        )
        form.addRow("Wavelength start (µm)", self.wavelength_start)
        form.addRow("Wavelength stop (µm)", self.wavelength_stop)
        form.addRow("X-min signed offset", self.domain_padding_spins["x_min"])
        form.addRow("X-max signed offset", self.domain_padding_spins["x_max"])
        form.addRow("Y-min signed offset", self.domain_padding_spins["y_min"])
        form.addRow("Y-max signed offset", self.domain_padding_spins["y_max"])
        form.addRow("Z-min signed offset", self.domain_padding_spins["z_min"])
        form.addRow("Z-max signed offset", self.domain_padding_spins["z_max"])
        form.addRow("Mesh accuracy", self.mesh_accuracy)
        form.addRow("Wavelength samples", self.frequency_points)
        form.addRow("Time-step stability factor", self.dt_stability)
        form.addRow("PML profile", self.pml_profile)
        form.addRow("Simulation time (fs)", self.simulation_time)
        form.addRow("Auto shutoff minimum", self.auto_shutoff_min)
        form.addRow("Geometry overlap beyond FDTD boundary", self.pml_geometry_overlap)
        form.addRow("FDTD dimension", self.dimension)
        form.addRow("Compute resource", self.resource_mode)
        form.addRow("TFLN crystal cut", self.tfln_crystal_cut)
        form.addRow("TFLN temperature", self.tfln_temperature)
        form.addRow("Lumerical project file", self.project_file)
        form.addRow(self.hide_cad)
        form.addRow(self.official_gc_domain)
        form.addRow(self.use_y_antisymmetry)
        form.addRow(self.run_after_build)
        resource_note = QLabel(
            "GPU is the default for every 3D simulation. The notebook detects the GPU, sets its SM licence estimate, "
            "keeps the CPU row active for meshing, solves with run(\"FDTD\", \"GPU\"), then switches back to "
            "the 30-thread CPU resource for result extraction and plotting."
        )
        resource_note.setWordWrap(True)
        form.addRow(resource_note)
        self.tabs.addTab(tab, "FDTD and compute")

    def padding_spin(self, key: str) -> QDoubleSpinBox:
        return self.domain_padding_spins[key]

    def _current_stack(self, strict: bool = False) -> list[dict[str, Any]]:
        stack = []
        for row in range(self.stack_table.rowCount()):
            layer_text = self.stack_table.cellWidget(row, 6).text().replace(";", ",")
            try:
                gds_layers = [int(value.strip()) for value in layer_text.split(",") if value.strip()]
            except ValueError as exc:
                if strict:
                    raise ValueError(f"Material row {row + 1} contains an invalid GDS layer list.") from exc
                gds_layers = [0]
            stack.append(
                {
                    "name": self.stack_table.item(row, 0).text().strip() or f"Layer {row + 1}",
                    "material": self.stack_table.cellWidget(row, 1).currentText().strip(),
                    "thickness_um": float(self.stack_table.cellWidget(row, 2).value()),
                    "etch_depth_um": float(self.stack_table.cellWidget(row, 3).value()),
                    "sidewall_angle_deg": float(self.stack_table.cellWidget(row, 4).value()),
                    "role": "geometry" if self.stack_table.cellWidget(row, 5).currentText() == "Exported cross-section" else "background",
                    "gds_layers": gds_layers or [0],
                    "slab_extent": "geometry"
                    if self.stack_table.cellWidget(row, 7).currentText() == "Under geometry"
                    else "full",
                    "mesh_factor": float(self.stack_table.cellWidget(row, 8).value()),
                    "mesh_order": int(self.stack_table.cellWidget(row, 9).value()),
                    "conformal": self.stack_table.item(row, 10).checkState() == Qt.CheckState.Checked,
                }
            )
        return stack

    def _preview_components(self) -> list[dict[str, Any]]:
        selected = {int(uid) for uid in (self.scope.currentData() or [])}
        return [component for component in self.all_components if int(component.get("uid", 0)) in selected]

    def preview_state(self) -> dict[str, Any]:
        included = {
            (int(self.geometry_table.item(row, 1).text()), int(self.geometry_table.item(row, 2).text()))
            for row in range(self.geometry_table.rowCount())
            if self.geometry_table.item(row, 0).checkState() == Qt.CheckState.Checked
        }
        polygons: list[tuple[Any, int]] = []
        all_points = []
        for component in self._preview_components():
            if component.get("kind") == "E-beam multipass" or component.get("kind") in {
                "FDTD port", "Fiber-axis FDTD port", "Fiber geometry", "Fiber port",
                "Power monitor", "Mode expansion monitor", "Field profile monitor",
            }:
                continue
            try:
                arrays, _ = component_geometry_arrays(component)
            except Exception:
                continue
            for points, layer, datatype in arrays:
                if (int(layer), int(datatype)) not in included:
                    continue
                polygons.append((points, int(layer)))
                all_points.extend(points)
        x_values = [float(point[0]) for point in all_points]
        y_values = [float(point[1]) for point in all_points]
        for component in self._preview_components():
            kind = str(component.get("kind", ""))
            if kind not in {"FDTD port", "Fiber-axis FDTD port", "Power monitor", "Mode expansion monitor", "Field profile monitor"}:
                continue
            params = component.get("params", {})
            span = float(params.get("span_um", max(params.get("x span", 0.0), params.get("y span", 0.0), 2.0)))
            half = 0.5 * max(0.0, span)
            cx, cy = float(component.get("x", 0.0)), float(component.get("y", 0.0))
            normal = str(params.get("plane normal", "X")).upper()
            if normal != "Z":
                normal_angle = float(component.get("orientation_deg", 0.0)) + (90.0 if normal == "Y" else 0.0)
                nearest = int(round(normal_angle / 90.0) * 90) % 360
                normal = "X" if nearest in (0, 180) else "Y"
            if normal == "X":
                x_values.append(cx); y_values.extend((cy - half, cy + half))
            elif normal == "Y":
                x_values.extend((cx - half, cx + half)); y_values.append(cy)
            else:
                x_values.extend((cx - half, cx + half)); y_values.extend((cy - half, cy + half))
        if x_values:
            x_base = (min(x_values), max(x_values))
            y_base = (min(y_values), max(y_values))
        else:
            x_base = (-1.0, 1.0); y_base = (-1.0, 1.0)
        raw_stack_ranges = _anchored_stack_ranges(self._current_stack())
        stack_ranges = []
        for preview_id, (row, row_z0, row_z1) in enumerate(raw_stack_ranges):
            preview_row = dict(row)
            preview_row["_preview_id"] = preview_id
            stack_ranges.append((preview_row, row_z0, row_z1))
        if stack_ranges:
            z_base_min = stack_ranges[0][1]
            z_base_max = stack_ranges[-1][2]
            if len(stack_ranges) > 1 and str(stack_ranges[0][0].get("role", "background")) == "background":
                z_base_min = stack_ranges[0][2]
            if len(stack_ranges) > 1 and str(stack_ranges[-1][0].get("role", "background")) == "background":
                z_base_max = stack_ranges[-1][1]
        else:
            z_base_min, z_base_max = -1.0, 1.0
        geometry_ranges = [
            (z0, z1)
            for row, z0, z1 in stack_ranges
            if str(row.get("role", "background")) == "geometry"
        ]
        device_top = max((z1 for _z0, z1 in geometry_ranges), default=0.0)
        device_bottom = min((z0 for z0, _z1 in geometry_ranges), default=device_top)
        device_z = 0.5 * (device_bottom + device_top)
        stack_top = stack_ranges[-1][2] if stack_ranges else device_top
        for component in self._preview_components():
            kind = str(component.get("kind", ""))
            params = component.get("params", {})
            if kind not in {"FDTD port", "Fiber-axis FDTD port", "Power monitor", "Mode expansion monitor", "Field profile monitor"}:
                continue
            plane_normal = str(params.get("plane normal", "X")).upper()
            if plane_normal == "Z":
                reference_z = _stack_reference_z(params, stack_ranges, device_top, stack_top)
                plane_z = reference_z + float(params.get("distance_um", 0.0))
                z_base_min = min(z_base_min, plane_z)
                z_base_max = max(z_base_max, plane_z)
                continue
            z_span = max(
                0.0,
                float(params.get("z span", params.get("z_span_um", 2.0))),
            )
            if kind in {"Power monitor", "Mode expansion monitor", "Field profile monitor"} and str(
                params.get("monitor geometry", "surface")
            ).lower() == "line":
                z_span = 0.0
            z_base_min = min(z_base_min, device_z - 0.5 * z_span)
            z_base_max = max(z_base_max, device_z + 0.5 * z_span)
        padding = {key: spin.value() for key, spin in self.domain_padding_spins.items()}
        return {
            "polygons": polygons,
            "x_base": x_base,
            "y_base": y_base,
            "z_base": (z_base_min, z_base_max),
            "stack_ranges": stack_ranges,
            "padding": padding,
            "components": self._preview_components(),
        }

    def _refresh_previews(self, *args) -> None:
        if hasattr(self, "cross_section_previews"):
            self._sync_domain_bounds()
            for preview in self.cross_section_previews:
                preview.update()

    def _sync_domain_bounds(self) -> None:
        """Show the exact FDTD coordinates alongside the clearance controls."""
        if not hasattr(self, "domain_bound_spins"):
            return
        state = self.preview_state()
        values = {
            "x_min": state["x_base"][0] - state["padding"]["x_min"],
            "x_max": state["x_base"][1] + state["padding"]["x_max"],
            "y_min": state["y_base"][0] - state["padding"]["y_min"],
            "y_max": state["y_base"][1] + state["padding"]["y_max"],
            "z_min": state["z_base"][0] - state["padding"]["z_min"],
            "z_max": state["z_base"][1] + state["padding"]["z_max"],
        }
        for key, value in values.items():
            spin = self.domain_bound_spins[key]
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        if hasattr(self, "domain_center_spins"):
            for axis in ("x", "y", "z"):
                minimum = float(values[axis + "_min"])
                maximum = float(values[axis + "_max"])
                center_spin = self.domain_center_spins[axis]
                size_spin = self.domain_size_spins[axis]
                center_spin.blockSignals(True)
                size_spin.blockSignals(True)
                center_spin.setValue(0.5 * (minimum + maximum))
                size_spin.setValue(maximum - minimum)
                center_spin.blockSignals(False)
                size_spin.blockSignals(False)

    def _set_exact_domain_bound(self, key: str, value: float) -> None:
        """Convert an exact typed FDTD boundary back into geometry clearance."""
        state = self.preview_state()
        base = state["z_base"] if key.startswith("z_") else state[key[0] + "_base"]
        padding = float(base[0]) - value if key.endswith("_min") else value - float(base[1])
        self.domain_padding_spins[key].setValue(padding)
        self._sync_domain_bounds()
        for preview in self.cross_section_previews:
            preview.update()

    def _set_domain_center_or_size(
        self,
        axis: str,
        *,
        center: float | None = None,
        size: float | None = None,
    ) -> None:
        """Update one exact FDTD center/span pair without changing geometry."""
        state = self.preview_state()
        base_min, base_max = state[axis + "_base"]
        current_min = float(base_min) - float(state["padding"][axis + "_min"])
        current_max = float(base_max) + float(state["padding"][axis + "_max"])
        current_center = 0.5 * (current_min + current_max)
        current_size = current_max - current_min
        resolved_center = current_center if center is None else float(center)
        resolved_size = current_size if size is None else max(1e-9, float(size))
        new_min = resolved_center - 0.5 * resolved_size
        new_max = resolved_center + 0.5 * resolved_size
        minimum_padding = float(base_min) - new_min
        maximum_padding = new_max - float(base_max)
        minimum_spin = self.domain_padding_spins[axis + "_min"]
        maximum_spin = self.domain_padding_spins[axis + "_max"]
        minimum_spin.blockSignals(True)
        maximum_spin.blockSignals(True)
        minimum_spin.setValue(minimum_padding)
        maximum_spin.setValue(maximum_padding)
        minimum_spin.blockSignals(False)
        maximum_spin.blockSignals(False)
        self._refresh_previews()

    def _connect_stack_preview_signals(self) -> None:
        for row in range(self.stack_table.rowCount()):
            for column in (1, 5, 7):
                widget = self.stack_table.cellWidget(row, column)
                if isinstance(widget, QComboBox) and not widget.property("preview_connected"):
                    widget.currentTextChanged.connect(self._refresh_previews)
                    widget.setProperty("preview_connected", True)
            for column in (2, 3, 4, 8):
                widget = self.stack_table.cellWidget(row, column)
                if isinstance(widget, QDoubleSpinBox) and not widget.property("preview_connected"):
                    widget.valueChanged.connect(self._refresh_previews)
                    widget.setProperty("preview_connected", True)
            entry = self.stack_table.cellWidget(row, 6)
            if isinstance(entry, QLineEdit) and not entry.property("preview_connected"):
                entry.textChanged.connect(self._refresh_previews)
                entry.setProperty("preview_connected", True)

    def _make_preview_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        reset_button = QPushButton("Reset domain clearances to λ/4")
        official_gc_button = QPushButton("Apply fast GC-SOI domain")
        fit_preview_button = QPushButton("Fit both views")
        show_3d_button = QPushButton("Show me a 3D version of the file I have built")
        show_3d_button.setMinimumHeight(44)
        top.addWidget(QLabel("Synchronized FDTD views: stack cross-section (XZ) + top/GDS layout (XY)"))
        top.addStretch(1); top.addWidget(fit_preview_button); top.addWidget(reset_button); top.addWidget(official_gc_button); top.addWidget(show_3d_button)
        layout.addLayout(top)
        preview_row = QHBoxLayout()
        self.xz_cross_section_preview = CrossSectionDomainPreview(self, "XZ")
        self.xy_top_view_preview = CrossSectionDomainPreview(self, "XY")
        self.cross_section_previews = (
            self.xz_cross_section_preview,
            self.xy_top_view_preview,
        )
        # Keep the historical attribute as an XZ alias for compatibility with
        # callers that only need one preview reference.
        self.cross_section_preview = self.xz_cross_section_preview
        preview_row.addWidget(self.xz_cross_section_preview, 1)
        preview_row.addWidget(self.xy_top_view_preview, 1)
        layout.addLayout(preview_row, 1)
        bounds_grid = QGridLayout()
        bounds_grid.addWidget(QLabel("Exact FDTD bounds (µm)"), 0, 0, 1, 2)
        self.domain_bound_spins: dict[str, QDoubleSpinBox] = {}
        labels = (("x_min", "X min"), ("x_max", "X max"), ("y_min", "Y min"), ("y_max", "Y max"), ("z_min", "Z min"), ("z_max", "Z max"))
        for index, (key, label) in enumerate(labels):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(6)
            spin.setSuffix(" µm")
            spin.setMinimumWidth(165)
            spin.valueChanged.connect(lambda value, bound_key=key: self._set_exact_domain_bound(bound_key, value))
            self.domain_bound_spins[key] = spin
            column = 2 * (index % 3)
            row = 1 + index // 3
            bounds_grid.addWidget(QLabel(label), row, column)
            bounds_grid.addWidget(spin, row, column + 1)
        layout.addLayout(bounds_grid)
        center_size_grid = QGridLayout()
        center_size_grid.addWidget(QLabel("Direct FDTD center and size (µm)"), 0, 0, 1, 6)
        self.domain_center_spins: dict[str, QDoubleSpinBox] = {}
        self.domain_size_spins: dict[str, QDoubleSpinBox] = {}
        for column_index, axis in enumerate(("x", "y", "z")):
            center_spin = QDoubleSpinBox()
            center_spin.setRange(-1e6, 1e6)
            center_spin.setDecimals(6)
            center_spin.setSuffix(" µm")
            center_spin.setMinimumWidth(165)
            size_spin = QDoubleSpinBox()
            size_spin.setRange(1e-6, 2e6)
            size_spin.setDecimals(6)
            size_spin.setSuffix(" µm")
            size_spin.setMinimumWidth(165)
            center_spin.valueChanged.connect(
                lambda value, domain_axis=axis: self._set_domain_center_or_size(
                    domain_axis, center=value
                )
            )
            size_spin.valueChanged.connect(
                lambda value, domain_axis=axis: self._set_domain_center_or_size(
                    domain_axis, size=value
                )
            )
            self.domain_center_spins[axis] = center_spin
            self.domain_size_spins[axis] = size_spin
            grid_column = 2 * column_index
            center_size_grid.addWidget(QLabel(axis.upper() + " center"), 1, grid_column)
            center_size_grid.addWidget(center_spin, 1, grid_column + 1)
            center_size_grid.addWidget(QLabel(axis.upper() + " size"), 2, grid_column)
            center_size_grid.addWidget(size_spin, 2, grid_column + 1)
        layout.addLayout(center_size_grid)
        instructions = QLabel(
            "Drag inside the red FDTD box to move its center; drag an edge/corner to resize, or type exact X/Y/Z bounds. "
            "The left XZ view looks through the material stack and edits X/Z. The right XY view is the true top/GDS view of the selected exported polygons and edits X/Y. "
            "Both views share the same X-min/X-max values, so an X drag in either view updates both immediately. "
            "Boundaries are unrestricted and may be placed inside Air or other layers. The material stack remains fixed while the red box moves independently."
        )
        instructions.setWordWrap(True); layout.addWidget(instructions)
        self.tabs.addTab(tab, "Cross-section & 3D preview")

        def reset_domain() -> None:
            quarter_wave = 0.25 * min(self.wavelength_start.value(), self.wavelength_stop.value())
            for spin in self.domain_padding_spins.values(): spin.setValue(quarter_wave)

        def fit_cross_section_previews() -> None:
            for preview in self.cross_section_previews:
                preview.reset_view()

        def apply_official_gc_domain() -> None:
            """Place the editable FDTD box like the official 3D SOI example."""
            grating = next(
                (component for component in self._preview_components() if str(component.get("kind", "")) == "GC-SOI"),
                None,
            )
            if grating is None:
                return
            theta = math.radians(float(grating.get("orientation_deg", 0.0)))
            cosine, sine = math.cos(theta), math.sin(theta)
            mirrored = bool(grating.get("mirrored", False))
            local_y_min = 0.0 if self.use_y_antisymmetry.isChecked() else -10.0
            local_y_max = 10.0
            # Keep the manually shifted waveguide port 2 um inside X-min. The
            # input waveguide crosses only the left PML; the complete output
            # grating, including its thick terminal arc, remains at least
            # lambda/4 inside the right boundary.
            local_x_max = 51.0
            try:
                grating_polygons, _ = component_geometry_arrays(grating)
                origin = np.asarray(
                    [float(grating.get("x", 0.0)), float(grating.get("y", 0.0))],
                    dtype=float,
                )
                local_x_values: list[float] = []
                local_y_values: list[float] = []
                mirror_sign = -1.0 if mirrored else 1.0
                for points, _layer, _datatype in grating_polygons:
                    points_array = np.asarray(points, dtype=float)
                    delta_x = points_array[:, 0] - origin[0]
                    delta_y = points_array[:, 1] - origin[1]
                    local_x_values.extend((delta_x * cosine + delta_y * sine).tolist())
                    local_y_values.extend((mirror_sign * (-delta_x * sine + delta_y * cosine)).tolist())
                quarter_wave = 0.25 * min(self.wavelength_start.value(), self.wavelength_stop.value())
                local_x_max = max(local_x_values) + quarter_wave
                local_y_max = max(local_y_values) + quarter_wave
                if not self.use_y_antisymmetry.isChecked():
                    local_y_min = min(local_y_values) - quarter_wave
            except Exception:
                pass
            local_corners = (
                (0.0, local_y_min), (0.0, local_y_max),
                (local_x_max, local_y_min), (local_x_max, local_y_max),
            )
            transformed = []
            for local_x, local_y in local_corners:
                if mirrored:
                    local_y = -local_y
                transformed.append(
                    (
                        float(grating.get("x", 0.0)) + local_x * cosine - local_y * sine,
                        float(grating.get("y", 0.0)) + local_x * sine + local_y * cosine,
                    )
                )
            state = self.preview_state()
            geometry_ranges = [
                (z0, z1) for row, z0, z1 in state["stack_ranges"]
                if str(row.get("role", "background")) == "geometry"
            ]
            device_base = min((z0 for z0, _z1 in geometry_ranges), default=0.0)
            exact = {
                "x_min": min(point[0] for point in transformed),
                "x_max": max(point[0] for point in transformed),
                "y_min": min(point[1] for point in transformed),
                "y_max": max(point[1] for point in transformed),
                "z_min": device_base - 1.2,
                "z_max": device_base + 1.3,
            }
            for key, value in exact.items():
                base = state["z_base"] if key.startswith("z_") else state[key[0] + "_base"]
                padding = float(base[0]) - value if key.endswith("_min") else value - float(base[1])
                self.domain_padding_spins[key].setValue(padding)
            self.official_gc_domain.setChecked(True)
            self._sync_domain_bounds()
            for preview in self.cross_section_previews:
                preview.reset_view()

        def show_3d() -> None:
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Pre-export 3D Lumerical model")
            preview_dialog.resize(1200, 820)
            preview_layout = QVBoxLayout(preview_dialog)
            visibility_row = QHBoxLayout()
            visibility_row.addWidget(QLabel("Show/hide:"))
            preview = ThreeDModelPreview(self)
            for label, attribute in (
                ("Device geometry", "show_device"),
                ("Material stack", "show_stack"),
                ("Ports", "show_ports"),
                ("Fiber", "show_fiber"),
                ("FDTD box", "show_fdtd"),
            ):
                checkbox = QCheckBox(label)
                checkbox.setChecked(True)
                checkbox.toggled.connect(lambda checked, name=attribute: (setattr(preview, name, bool(checked)), preview.update()))
                visibility_row.addWidget(checkbox)
            visibility_row.addStretch(1)
            reset_view = QPushButton("Reset 3D view")
            reset_view.clicked.connect(lambda: (
                setattr(preview, "azimuth_deg", 35.0),
                setattr(preview, "elevation_deg", 20.0),
                setattr(preview, "zoom_factor", 1.0),
                setattr(preview, "pan", QPointF(0.0, 0.0)),
                preview.update(),
            ))
            visibility_row.addWidget(reset_view)
            preview_layout.addLayout(visibility_row)

            layer_grid = QGridLayout()
            layer_grid.addWidget(QLabel("Individual stack layers:"), 0, 0)

            def set_layer_visible(row_id: int, visible: bool) -> None:
                if visible:
                    preview.hidden_stack_rows.discard(row_id)
                else:
                    preview.hidden_stack_rows.add(row_id)
                preview.update()

            for index, (row, _z0, _z1) in enumerate(self.preview_state()["stack_ranges"]):
                row_id = int(row.get("_preview_id", index))
                layer_checkbox = QCheckBox(f"{row.get('name', 'Layer')} — {row.get('material', '')}")
                layer_checkbox.setChecked(True)
                layer_checkbox.toggled.connect(lambda checked, rid=row_id: set_layer_visible(rid, checked))
                grid_index = index + 1
                layer_grid.addWidget(layer_checkbox, 1 + (grid_index - 1) // 3, (grid_index - 1) % 3)
            preview_layout.addLayout(layer_grid)
            preview_layout.addWidget(preview, 1)
            close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close_buttons.rejected.connect(preview_dialog.reject)
            preview_layout.addWidget(close_buttons)
            preview_dialog.exec()

        reset_button.clicked.connect(reset_domain)
        official_gc_button.clicked.connect(apply_official_gc_domain)
        fit_preview_button.clicked.connect(fit_cross_section_previews)
        show_3d_button.clicked.connect(show_3d)
        self.scope.currentIndexChanged.connect(self._refresh_previews)
        self.geometry_table.itemChanged.connect(self._refresh_previews)
        self.stack_table.itemChanged.connect(self._refresh_previews)
        self.wavelength_start.valueChanged.connect(self._refresh_previews)
        self.wavelength_stop.valueChanged.connect(self._refresh_previews)
        for spin in self.domain_padding_spins.values(): spin.valueChanged.connect(self._refresh_previews)
        self._connect_stack_preview_signals()
        self._sync_domain_bounds()
        if self.official_gc_domain.isChecked() and not self.saved.get("domain_padding_um"):
            apply_official_gc_domain()

    def configuration(self) -> dict[str, Any]:
        included_layers = []
        for row in range(self.geometry_table.rowCount()):
            if self.geometry_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                included_layers.append([int(self.geometry_table.item(row, 1).text()), int(self.geometry_table.item(row, 2).text())])
        stack = self._current_stack(strict=True)
        state = self.preview_state()
        for axis in ("x", "y", "z"):
            base_min, base_max = state[axis + "_base"]
            domain_min = base_min - state["padding"][axis + "_min"]
            domain_max = base_max + state["padding"][axis + "_max"]
            if domain_max <= domain_min:
                raise ValueError(f"FDTD {axis.upper()} max must be greater than {axis.upper()} min.")
        symmetry_boundary = ""
        if self.use_y_antisymmetry.isChecked():
            grating = next(
                (component for component in self._preview_components() if str(component.get("kind", "")) == "GC-SOI"),
                None,
            )
            if grating is not None:
                theta = math.radians(float(grating.get("orientation_deg", 0.0)))
                sign = -1.0 if bool(grating.get("mirrored", False)) else 1.0
                normal_x, normal_y = -sign * math.sin(theta), sign * math.cos(theta)
                if abs(normal_x) > abs(normal_y):
                    symmetry_boundary = "x min" if normal_x > 0.0 else "x max"
                else:
                    symmetry_boundary = "y min" if normal_y > 0.0 else "y max"
        return {
            "scope_uids": [int(uid) for uid in (self.scope.currentData() or [])],
            "scope_label": self.scope.currentText(),
            "included_layers": included_layers,
            "include_ports": self.include_ports.isChecked(),
            "stack_preset": self.preset.currentText(),
            "material_stack": stack,
            "wavelength_start_um": self.wavelength_start.value(),
            "wavelength_stop_um": self.wavelength_stop.value(),
            "xy_padding_um": self.xy_padding.value(),
            "z_padding_um": self.z_padding.value(),
            "domain_padding_um": {key: spin.value() for key, spin in self.domain_padding_spins.items()},
            "mesh_accuracy": self.mesh_accuracy.value(),
            "frequency_points": self.frequency_points.value(),
            "dt_stability_factor": self.dt_stability.value(),
            "pml_profile": self.pml_profile.currentText(),
            "pml_geometry_overlap_um": self.pml_geometry_overlap.value(),
            "simulation_time_fs": self.simulation_time.value(),
            "auto_shutoff_min": self.auto_shutoff_min.value(),
            "dimension": "3D",
            "resource_mode": self.resource_mode.currentText(),
            "tfln_crystal_cut": self.tfln_crystal_cut.currentText(),
            "tfln_temperature_K": self.tfln_temperature.value(),
            "project_file": self.project_file.text().strip() or "exported_component.fsp",
            "hide_cad": self.hide_cad.isChecked(),
            "official_gc_domain": self.official_gc_domain.isChecked(),
            "use_y_antisymmetry": self.use_y_antisymmetry.isChecked(),
            "antisymmetry_boundary": symmetry_boundary,
            "gc_domain_version": 3,
            "run_after_build": self.run_after_build.isChecked(),
        }
