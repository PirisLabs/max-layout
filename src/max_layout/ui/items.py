"""QGraphicsItem subclasses for components and write fields."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
import json
import math
import os

import gdstk
import numpy as np

from PySide6.QtCore import QLineF, QPoint, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup, QGraphicsLineItem, QGraphicsObject, QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QWidget

from ..constants import EBEAM_LAYER, SIMULATION_COMPONENT_KINDS, SIMULATION_LAYER
from ..gds.build import _add_component_geometry_to_cell, _canonicalize_component_layers
from ..gds.ebeam import multipass_field_layout
from ..geometry.transforms import scene_to_world_point
from ..ports import component_local_ports
from ..ui.theme import color_for_layer
from ..utils import safe_json_copy

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:
    QOpenGLWidget = None

_PREVIEW_GEOMETRY_CACHE: OrderedDict[str, list[tuple[np.ndarray, int, int]]] = OrderedDict()
_PREVIEW_PATH_CACHE: OrderedDict[str, dict[tuple[int, int], QPainterPath]] = OrderedDict()

_PREVIEW_GEOMETRY_CACHE_LIMIT = 2048
_PREVIEW_PATH_CACHE_LIMIT = 512

_PREVIEW_MAX_VERTICES = max(32, int(os.environ.get("MAX_LAYOUT_PREVIEW_MAX_VERTICES", "256")))
# Beyond this extent an item paints directly instead of through Qt's device-coordinate
# raster cache; see ComponentGraphicsItem.apply_cache_mode.
_DIRECT_PAINT_EXTENT_UM = float(os.environ.get("MAX_LAYOUT_DIRECT_PAINT_EXTENT_UM", "1200"))
_PREVIEW_MAX_POLYGONS_PER_LAYER = max(100, int(os.environ.get("MAX_LAYOUT_PREVIEW_MAX_POLYGONS_PER_LAYER", "500")))
_PREVIEW_CACHE_MAX_VERTICES = max(
    1000, int(os.environ.get("MAX_LAYOUT_PREVIEW_CACHE_MAX_VERTICES", "250000"))
)
# Exact preview is the default: the canvas and flattened GDS are derived from
# identical polygons.  Very large legacy layouts may explicitly opt into the
# old approximate display behavior through this environment switch; export is
# full resolution in either mode.
_APPROXIMATE_PREVIEW = str(os.environ.get("MAX_LAYOUT_APPROXIMATE_PREVIEW", "0")).strip().lower() in {
    "1", "true", "yes", "on",
}


def clear_preview_caches() -> None:
    """Discard all derived GUI geometry; project and export data are untouched."""
    _PREVIEW_GEOMETRY_CACHE.clear()
    _PREVIEW_PATH_CACHE.clear()


def _apply_preview_resolution(component: dict[str, Any]) -> None:
    """Lower only the interactive-canvas resolution; never mutate project data."""
    params = component.setdefault("params", {})
    for key, limit in {"rf_edge_bend_points": 96, "gc_points": 96}.items():
        value = params.get(key)
        if isinstance(value, (int, float)):
            params[key] = min(int(value), limit)


def _sample_dense_preview_polygons(
    polygons: list[tuple[np.ndarray, int, int]],
) -> list[tuple[np.ndarray, int, int]]:
    """Keep structural planes plus representative repeated preview details."""
    grouped: OrderedDict[tuple[int, int], list[tuple[np.ndarray, int, int]]] = OrderedDict()
    for polygon in polygons:
        grouped.setdefault((polygon[1], polygon[2]), []).append(polygon)
    reduced: list[tuple[np.ndarray, int, int]] = []
    for group in grouped.values():
        if len(group) <= _PREVIEW_MAX_POLYGONS_PER_LAYER:
            reduced.extend(group)
            continue
        areas = []
        for index, (points, _, _) in enumerate(group):
            x = points[:, 0]
            y = points[:, 1]
            area = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0
            areas.append((area, index))
        structural_count = min(32, max(3, _PREVIEW_MAX_POLYGONS_PER_LAYER // 10))
        structural = {index for _, index in sorted(areas, reverse=True)[:structural_count]}
        remaining = [index for index in range(len(group)) if index not in structural]
        sample_count = max(0, _PREVIEW_MAX_POLYGONS_PER_LAYER - len(structural))
        if remaining and sample_count:
            positions = np.linspace(0, len(remaining) - 1, sample_count, dtype=int)
            structural.update(remaining[int(position)] for position in np.unique(positions))
        reduced.extend(group[index] for index in sorted(structural))
    return reduced


def _needs_exact_t_electrode_preview(component:dict[str,Any])->bool:
    kind=str(component.get("kind",""));params=component.get("params",{})
    if kind=="Segmented electrode":
        return True
    return kind=="RF test block" and (
        str(params.get("rf_component_kind",""))=="Segmented electrode"
        or (bool(params.get("taper_test_structure",False)) and str(params.get("taper_test_center",""))=="T electrode")
    )


def _union_preview_polygons(polygons:list[tuple[np.ndarray,int,int]])->list[tuple[np.ndarray,int,int]]:
    """Merge overlapping metal only for the GUI; exported polygons are untouched."""
    grouped:OrderedDict[tuple[int,int],list[np.ndarray]]=OrderedDict()
    for points,layer,datatype in polygons:grouped.setdefault((layer,datatype),[]).append(points)
    result:list[tuple[np.ndarray,int,int]]=[]
    for (layer,datatype),groups in grouped.items():
        try:
            sources=[gdstk.Polygon(points,layer=layer,datatype=datatype) for points in groups]
            merged=gdstk.boolean(sources,[],"or",precision=0.001,layer=layer,datatype=datatype)
            result.extend((np.asarray(polygon.points,float),layer,datatype) for polygon in merged)
        except Exception:
            result.extend((points,layer,datatype) for points in groups)
    return result


def component_preview_cache_key(component: dict[str, Any]) -> str:
    local = safe_json_copy(component)
    local["x"] = 0.0
    local["y"] = 0.0
    local["orientation_deg"] = 0.0
    local["attachment"] = None
    _canonicalize_component_layers(local)
    if _APPROXIMATE_PREVIEW:
        _apply_preview_resolution(local)
    # Preview geometry is independent of UID and global transform.  Caching it
    # avoids rebuilding identical gdstk cells on every scene refresh.
    cache_payload = {
        "kind": local.get("kind"),
        "mirrored": bool(local.get("mirrored", False)),
        "params": local.get("params", {}),
        "approximate_preview": _APPROXIMATE_PREVIEW,
    }
    return json.dumps(cache_payload, sort_keys=True, separators=(",", ":"), default=str)


def component_local_polygons(component: dict[str, Any]) -> list[tuple[np.ndarray, int, int]]:
    if component.get("kind") == "E-beam multipass":
        return []
    if component.get("kind") in SIMULATION_COMPONENT_KINDS:
        kind = str(component.get("kind"))
        params = component.get("params", {})
        span = max(0.001, float(params.get("span_um", 2.0)))
        if kind in {"Fiber geometry", "Fiber port"}:
            radius = max(0.001, 0.5 * float(params.get("cladding diameter_um", span)))
            angles = np.linspace(0.0, 2.0 * math.pi, 97, endpoint=False)
            points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        else:
            distance = float(params.get("distance_um", 0.0))
            geometry = str(params.get("port geometry", params.get("monitor geometry", "surface"))).lower()
            thickness = max(0.08, min(1.0 if geometry == "surface" else 0.15, span / 10.0))
            plane_normal = str(
                params.get("plane normal", params.get("injection axis", "X"))
            ).upper()
            x_span = max(0.0, float(params.get("x span", span if plane_normal != "X" else 0.0)))
            y_span = max(0.0, float(params.get("y span", span if plane_normal != "Y" else 0.0)))
            transverse_span = max(x_span, y_span, span)
            if plane_normal == "Z":
                width = max(thickness, x_span or transverse_span)
                height = max(thickness, y_span or transverse_span)
                points = np.array(
                    [[-width / 2.0, -height / 2.0], [width / 2.0, -height / 2.0],
                     [width / 2.0, height / 2.0], [-width / 2.0, height / 2.0]],
                    dtype=float,
                )
            elif plane_normal == "Y":
                width = max(thickness, x_span or transverse_span)
                points = np.array(
                    [[-width / 2.0, distance - thickness / 2.0], [width / 2.0, distance - thickness / 2.0],
                     [width / 2.0, distance + thickness / 2.0], [-width / 2.0, distance + thickness / 2.0]],
                    dtype=float,
                )
            else:
                height = max(thickness, y_span or transverse_span)
                points = np.array(
                    [[distance - thickness / 2.0, -height / 2.0], [distance + thickness / 2.0, -height / 2.0],
                     [distance + thickness / 2.0, height / 2.0], [distance - thickness / 2.0, height / 2.0]],
                    dtype=float,
                )
        return [(points, SIMULATION_LAYER, 0)]
    local = safe_json_copy(component)
    local["x"] = 0.0
    local["y"] = 0.0
    local["orientation_deg"] = 0.0
    local["attachment"] = None
    _canonicalize_component_layers(local)
    if _APPROXIMATE_PREVIEW:
        _apply_preview_resolution(local)
    cache_key = component_preview_cache_key(component)
    cached = _PREVIEW_GEOMETRY_CACHE.get(cache_key)
    if cached is not None:
        _PREVIEW_GEOMETRY_CACHE.move_to_end(cache_key)
        return cached
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("PREVIEW")
    _add_component_geometry_to_cell(local, cell)
    polygons: list[Any] = []
    try:
        polygons.extend(cell.get_polygons(apply_repetitions=True, include_paths=True))
    except Exception:
        polygons.extend(cell.polygons)
        for path in getattr(cell, "paths", []):
            try:
                polygons.extend(path.to_polygons())
            except Exception:
                pass
    result: list[tuple[np.ndarray, int, int]] = []
    for polygon in polygons:
        points = np.asarray(getattr(polygon, "points", polygon), dtype=float)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        layer = int(getattr(polygon, "layer", local.get("params", {}).get("layer", 1)))
        datatype = int(getattr(polygon, "datatype", local.get("params", {}).get("datatype", 0)))
        result.append((points, layer, datatype))
    if _needs_exact_t_electrode_preview(local):
        result=_union_preview_polygons(result)
    elif _APPROXIMATE_PREVIEW:
        result = _sample_dense_preview_polygons(result)
    # A single exact test block can contain hundreds of thousands of vertices.
    # Draw it exactly, but do not retain an unbounded copy in both geometry and
    # painter-path caches.  Ordinary components still get the fast shared cache.
    if sum(len(points) for points, _layer, _datatype in result) <= _PREVIEW_CACHE_MAX_VERTICES:
        _PREVIEW_GEOMETRY_CACHE[cache_key] = result
        _PREVIEW_GEOMETRY_CACHE.move_to_end(cache_key)
        while len(_PREVIEW_GEOMETRY_CACHE) > _PREVIEW_GEOMETRY_CACHE_LIMIT:
            _PREVIEW_GEOMETRY_CACHE.popitem(last=False)
    return result


def component_layer_paths(
    component: dict[str, Any],
    polygons: list[tuple[np.ndarray, int, int]],
) -> dict[tuple[int, int], QPainterPath]:
    """Return immutable standard Qt paths shared by identical components."""
    cache_key = component_preview_cache_key(component)
    cacheable = sum(len(points) for points, _layer, _datatype in polygons) <= _PREVIEW_CACHE_MAX_VERTICES
    if cacheable:
        cached = _PREVIEW_PATH_CACHE.get(cache_key)
        if cached is not None:
            _PREVIEW_PATH_CACHE.move_to_end(cache_key)
            return cached
    layer_paths: dict[tuple[int, int], QPainterPath] = {}
    for points, layer, datatype in polygons:
        preview_points = points
        if _APPROXIMATE_PREVIEW and component.get("kind") == "Boolean geometry" and len(points) > 2000:
            stride = max(1, math.ceil(len(points) / 2000))
            preview_points = points[::stride]
        # Every polygon on a layer shares one path, and adjoining sections of a device overlap
        # at their joints (straight into S-bend into taper into grating).  Under Qt's default
        # odd-even rule those overlaps cancel and render as holes, which is why a feedline used
        # to look wrong on canvas while its GDS was correct.  Winding fill unions them instead,
        # but only when the contours share an orientation, and some builders emit a mix of both
        # -- so normalise to counter-clockwise here.  Preview only; exported geometry is untouched.
        x = preview_points[:, 0]
        y = preview_points[:, 1]
        if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0.0:
            preview_points = preview_points[::-1]
        polygon = QPolygonF([QPointF(float(px), -float(py)) for px, py in preview_points])
        key = (int(layer), int(datatype))
        path = layer_paths.setdefault(key, QPainterPath())
        path.setFillRule(Qt.FillRule.WindingFill)
        path.addPolygon(polygon)
        path.closeSubpath()
    if cacheable:
        _PREVIEW_PATH_CACHE[cache_key] = layer_paths
        _PREVIEW_PATH_CACHE.move_to_end(cache_key)
        while len(_PREVIEW_PATH_CACHE) > _PREVIEW_PATH_CACHE_LIMIT:
            _PREVIEW_PATH_CACHE.popitem(last=False)
    return layer_paths


class LayoutView(QGraphicsView):
    cursorWorldChanged = Signal(float, float)
    zoomChanged = Signal(float)
    regionFitRequested = Signal(QRectF)

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None, use_opengl: bool | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontSavePainterState
            | QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#0c1118")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setMouseTracking(True)
        self.show_grid = True
        self.show_axes = True
        self.show_rulers = True
        self.measure_mode = False
        self.measure_start: QPointF | None = None
        self.measure_end: QPointF | None = None
        self.measure_third: QPointF | None = None
        self.fit_region_mode = False
        self.fit_region_start: QPointF | None = None
        self.fit_region_end: QPointF | None = None
        self.sketch_mode=False
        self.sketch_strokes:list[list[QPointF]]=[]
        self._active_sketch:list[QPointF]|None=None
        self._middle_pan = False
        self._last_pan = QPoint()
        self.opengl_enabled = False
        if use_opengl is None:
            use_opengl = str(os.environ.get("PHOTONIC_LAYOUT_USE_OPENGL", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.set_opengl_enabled(bool(use_opengl))

    def set_opengl_enabled(self, enabled: bool) -> bool:
        """Switch between GPU-assisted OpenGL painting and CPU raster."""
        if enabled and QOpenGLWidget is not None:
            try:
                self.setViewport(QOpenGLWidget())
                # QOpenGLWidget supports partial updates.  Repainting only the
                # changed scene region avoids redrawing every dense polygon
                # whenever a component is added, selected, or moved.
                self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
                self.opengl_enabled = True
                return True
            except Exception:
                pass
        self.setViewport(QWidget())
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.opengl_enabled = False
        return False

    def current_zoom_percent(self) -> float:
        return abs(float(self.transform().m11())) * 100.0

    def emit_zoom(self) -> None:
        self.zoomChanged.emit(self.current_zoom_percent())

    def zoom_by(self, factor: float) -> None:
        current = max(abs(float(self.transform().m11())), 1e-12)
        target = max(1e-6, min(256.0, current * float(factor)))
        applied = target / current
        if abs(applied - 1.0) > 1e-12:
            self.scale(applied, applied)
        self.emit_zoom()

    def begin_fit_region(self) -> None:
        self.fit_region_mode = True
        self.fit_region_start = None
        self.fit_region_end = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()

    def reset_one_to_one(self) -> None:
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(center)
        self.emit_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.zoom_by(1.18 if event.angleDelta().y() > 0 else 1 / 1.18)
        event.accept()

    @staticmethod
    def event_view_pos(event) -> QPoint:
        position = getattr(event, "position", None)
        if callable(position):
            return position().toPoint()
        return event.pos()

    def mousePressEvent(self, event) -> None:
        if self.fit_region_mode and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(self.event_view_pos(event));self.fit_region_start=point;self.fit_region_end=point;event.accept();return
        if self.sketch_mode and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(self.event_view_pos(event));self._active_sketch=[point];event.accept();return
        if self.measure_mode and event.button() == Qt.MouseButton.LeftButton:
            point=self.mapToScene(self.event_view_pos(event))
            if self.measure_start is None or self.measure_third is not None:self.measure_start=point;self.measure_end=None;self.measure_third=None
            elif self.measure_end is None:self.measure_end=point
            else:self.measure_third=point
            self.viewport().update();event.accept();return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan = True
            self._last_pan = self.event_view_pos(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        view_pos = self.event_view_pos(event)
        scene_point = self.mapToScene(view_pos)
        self.cursorWorldChanged.emit(float(scene_point.x()), -float(scene_point.y()))
        if self.sketch_mode and self._active_sketch is not None:
            if not self._active_sketch or QLineF(self._active_sketch[-1],scene_point).length()>1/max(abs(self.transform().m11()),1e-9):self._active_sketch.append(scene_point)
            self.viewport().update();event.accept();return
        if self._middle_pan:
            delta = view_pos - self._last_pan
            self._last_pan = view_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.fit_region_mode and self.fit_region_start is not None:
            self.fit_region_end=scene_point;self.viewport().update();event.accept();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.fit_region_mode and event.button()==Qt.MouseButton.LeftButton and self.fit_region_start is not None:
            self.fit_region_end=self.mapToScene(self.event_view_pos(event));rect=QRectF(self.fit_region_start,self.fit_region_end).normalized();self.fit_region_mode=False;self.fit_region_start=None;self.fit_region_end=None;self.setCursor(Qt.CursorShape.ArrowCursor);self.viewport().update()
            if rect.width()>1e-9 and rect.height()>1e-9:self.regionFitRequested.emit(rect)
            event.accept();return
        if self.sketch_mode and event.button()==Qt.MouseButton.LeftButton and self._active_sketch is not None:
            point=self.mapToScene(self.event_view_pos(event));self._active_sketch.append(point)
            if len(self._active_sketch)>=2:self.sketch_strokes.append(self._active_sketch)
            self._active_sketch=None;self.viewport().update();event.accept();return
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan:
            self._middle_pan = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        scale = max(abs(self.transform().m11()), 1e-12)
        target = 80.0 / scale
        magnitude = 10 ** math.floor(math.log10(max(target, 1e-9)))
        spacing = magnitude
        for multiplier in (1, 2, 5, 10):
            if magnitude * multiplier >= target:
                spacing = magnitude * multiplier
                break

        if self.show_grid:
            left = math.floor(rect.left() / spacing) * spacing
            top = math.floor(rect.top() / spacing) * spacing
            minor_pen = QPen(QColor(255, 255, 255, 20))
            minor_pen.setCosmetic(True)
            painter.setPen(minor_pen)
            x = left
            while x <= rect.right():
                painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
                x += spacing
            y = top
            while y <= rect.bottom():
                painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
                y += spacing

        if self.show_axes:
            axis_pen = QPen(QColor(112, 214, 255, 115))
            axis_pen.setCosmetic(True)
            painter.setPen(axis_pen)
            painter.drawLine(QLineF(0, rect.top(), 0, rect.bottom()))
            painter.drawLine(QLineF(rect.left(), 0, rect.right(), 0))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if self.fit_region_start is not None and self.fit_region_end is not None:
            painter.save();pen=QPen(QColor('#65e6ff'),1.5,Qt.PenStyle.DashLine);pen.setCosmetic(True);painter.setPen(pen);painter.setBrush(QColor(101,230,255,35));painter.drawRect(QRectF(self.fit_region_start,self.fit_region_end).normalized());painter.restore()
        if self.sketch_strokes or self._active_sketch:
            painter.save();pen=QPen(QColor('#ffd166'),2);pen.setCosmetic(True);painter.setPen(pen);painter.setBrush(Qt.BrushStyle.NoBrush)
            for stroke in self.sketch_strokes+([self._active_sketch] if self._active_sketch else []):
                if not stroke:continue
                path=QPainterPath(stroke[0])
                for point in stroke[1:]:path.lineTo(point)
                painter.drawPath(path)
            painter.restore()
        if self.measure_start is not None:
            painter.save();painter.resetTransform();a=self.mapFromScene(self.measure_start);b=self.mapFromScene(self.measure_end or self.measure_start);painter.setPen(QPen(QColor('#ff5ca8'),2));painter.drawLine(a,b);painter.setBrush(QColor('#ff5ca8'));painter.drawEllipse(a,4,4);painter.drawEllipse(b,4,4)
            if self.measure_end is not None:
                dx=self.measure_end.x()-self.measure_start.x();dy=self.measure_end.y()-self.measure_start.y();distance=math.hypot(dx,dy);painter.drawText((a.x()+b.x())//2+8,(a.y()+b.y())//2-8,f'{distance:.6g} µm   Δx={dx:.6g}  Δy={-dy:.6g}')
            else:painter.drawText(a.x()+8,a.y()-8,'Click second point')
            if self.measure_end is not None and self.measure_third is None:painter.drawText(b.x()+8,b.y()+18,'Click third point for angle')
            if self.measure_end is not None and self.measure_third is not None:
                c=self.mapFromScene(self.measure_third);painter.setPen(QPen(QColor('#65e6ff'),2));painter.drawLine(b,c);painter.setBrush(QColor('#65e6ff'));painter.drawEllipse(c,4,4)
                v1=np.array([self.measure_start.x()-self.measure_end.x(),self.measure_start.y()-self.measure_end.y()]);v2=np.array([self.measure_third.x()-self.measure_end.x(),self.measure_third.y()-self.measure_end.y()]);den=max(float(np.linalg.norm(v1)*np.linalg.norm(v2)),1e-30);angle=math.degrees(math.acos(float(np.clip(np.dot(v1,v2)/den,-1,1))));second=math.hypot(v2[0],v2[1]);painter.drawText(b.x()+10,b.y()+18,f'∠ABC={angle:.4g}°   BC={second:.6g} µm')
            painter.restore()
        if not self.show_rulers:return
        scale = max(abs(self.transform().m11()), 1e-12)
        target = 100.0 / scale
        magnitude = 10 ** math.floor(math.log10(max(target, 1e-9)))
        spacing = next((magnitude*m for m in (1,2,5,10) if magnitude*m >= target), magnitude*10)
        left = math.floor(rect.left()/spacing)*spacing
        top = math.floor(rect.top()/spacing)*spacing
        painter.save();painter.resetTransform()
        vp=self.viewport().rect();painter.fillRect(QRectF(0,0,vp.width(),24),QColor(18,27,39,225));painter.fillRect(QRectF(0,0,54,vp.height()),QColor(18,27,39,225));painter.setPen(QPen(QColor('#9edcff'),1))
        x=left
        while x<=rect.right():
            px=self.mapFromScene(QPointF(x,0)).x();painter.drawLine(px,16,px,24);painter.drawText(px+3,13,f'{x:g}') ;x+=spacing
        y=top
        while y<=rect.bottom():
            py=self.mapFromScene(QPointF(0,y)).y();painter.drawLine(46,py,54,py);painter.drawText(3,py-3,f'{-y:g}');y+=spacing
        painter.setPen(QColor('#ffffff'));painter.drawText(4,16,'µm');painter.restore()


class ResizeHandleItem(QGraphicsRectItem):
    def __init__(
        self,
        parent_component: "ComponentGraphicsItem",
        corner: str,
        position: QPointF,
    ) -> None:
        super().__init__(QRectF(-5, -5, 10, 10), parent_component)
        self.parent_component = parent_component
        self.corner = corner
        self.setPos(position)
        self.setZValue(200)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setBrush(QBrush(QColor("#67e8f9")))
        pen = QPen(QColor("#ffffff"), 1.2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setCursor(
            Qt.CursorShape.SizeFDiagCursor
            if corner in {"nw", "se"}
            else Qt.CursorShape.SizeBDiagCursor
        )
        self._snapshot: str | None = None
        self._initial_params: dict[str, Any] | None = None
        self._initial_bounds = QRectF()
        self._current_local = QPointF(position)

    def mousePressEvent(self, event) -> None:
        event.accept()
        self._snapshot = self.parent_component.main_window.snapshot()
        self._initial_params = safe_json_copy(self.parent_component.component.get("params", {}))
        self._initial_bounds = QRectF(self.parent_component.geometry_bounds)
        self._current_local = self.parent_component.mapFromScene(event.scenePos())

    def mouseMoveEvent(self, event) -> None:
        event.accept()
        self._current_local = self.parent_component.mapFromScene(event.scenePos())
        self.setPos(self._current_local)

    def mouseReleaseEvent(self, event) -> None:
        event.accept()
        if self._initial_params is None or self._snapshot is None:
            return
        center = self._initial_bounds.center()
        width = max(1e-9, self._initial_bounds.width())
        height = max(1e-9, self._initial_bounds.height())
        scale_x = max(0.03, 2.0 * abs(self._current_local.x() - center.x()) / width)
        scale_y = max(0.03, 2.0 * abs(self._current_local.y() - center.y()) / height)
        main_window = self.parent_component.main_window
        uid = self.parent_component.uid
        initial_parameters = safe_json_copy(self._initial_params)
        snapshot_before = str(self._snapshot)
        QTimer.singleShot(
            0,
            lambda: main_window.resize_component_from_handle(
                uid,
                initial_parameters,
                scale_x,
                scale_y,
                snapshot_before,
            ),
        )


class ComponentGraphicsItem(QGraphicsItemGroup):
    def __init__(self, main_window: "NativeLayoutWindow", component: dict[str, Any]) -> None:
        super().__init__()
        self.main_window = main_window
        self.component = component
        self.uid = int(component["uid"])
        self._drag_snapshot: str | None = None
        self._drag_start_position = QPointF()
        self._press_scene_position: QPointF | None = None
        self.geometry_bounds = QRectF(-15, -10, 30, 20)
        self.port_items: list[QGraphicsItem] = []
        self.resize_handles: list[ResizeHandleItem] = []
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setZValue(10)
        self.rebuild_geometry()
        self.sync_transform()

    def apply_cache_mode(self) -> None:
        """Pick the cache mode from the item's real extent, not from its kind.

        Qt's device-coordinate cache rasterises the whole item extent into one pixmap.  Past a
        few thousand microns that raster approaches the engine's 32,767-pixel limit, and the item
        draws wrong or not at all -- a default feedline is already 3383 um long, and a test block
        far more.  Long, thin devices are exactly the case where the raster also loses a 1.2 um
        waveguide entirely.  The shared painter paths are the cache that actually matters here,
        so anything large simply paints directly.
        """
        rect = self.geometry_bounds
        extent = max(abs(rect.width()), abs(rect.height()))
        large = extent > _DIRECT_PAINT_EXTENT_UM or self.component.get("kind") in {"RF test block", "Photonic test block"}
        self.setCacheMode(
            QGraphicsItem.CacheMode.NoCache if large else QGraphicsItem.CacheMode.DeviceCoordinateCache
        )

    def rebuild_geometry(self) -> None:
        for child in list(self.childItems()):
            if isinstance(child, QGraphicsItem):
                try:
                    self.removeFromGroup(child)
                except Exception:
                    pass
            if child.scene() is not None:
                child.scene().removeItem(child)
        self.port_items.clear()
        self.resize_handles.clear()
        bounds: QRectF | None = None
        try:
            polygons = component_local_polygons(self.component)
        except Exception as exc:
            polygons = []
            path = QPainterPath()
            path.addRect(QRectF(-30, -15, 60, 30))
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(QColor(255, 80, 80, 90)))
            item.setPen(QPen(QColor("#ff7675"), 0))
            item.setToolTip(str(exc))
            self.addToGroup(item)
            bounds = item.boundingRect()

        # Batch every polygon sharing a layer/datatype into one painter path.
        # A photonic-crystal array can contain tens of thousands of polygons;
        # creating one QGraphicsItem per hole makes scene construction and
        # selection unnecessarily expensive.
        layer_paths = component_layer_paths(self.component, polygons)
        for (layer,datatype),path in layer_paths.items():
            child=QGraphicsPathItem(path);child.setBrush(QBrush(color_for_layer(layer,135)));pen=QPen(color_for_layer(layer,230),0);pen.setCosmetic(True);child.setPen(pen);child.setData(0,layer);child.setData(1,datatype);child.setData(2,"geometry");self.addToGroup(child);rect=path.boundingRect();bounds=rect if bounds is None else bounds.united(rect)

        if not polygons and bounds is None:
            child = QGraphicsRectItem(QRectF(-15, -10, 30, 20))
            child.setBrush(QBrush(QColor(140, 140, 140, 60)))
            pen = QPen(QColor("#b2bec3"), 0)
            pen.setCosmetic(True)
            child.setPen(pen)
            child.setData(2, "geometry")
            self.addToGroup(child)
            bounds = child.rect()

        self.geometry_bounds = bounds or QRectF(-15, -10, 30, 20)
        self.apply_cache_mode()
        self.add_port_markers()
        if self.component.get("kind") not in SIMULATION_COMPONENT_KINDS:
            self.add_resize_handles()
        self.set_handles_visible(self.isSelected())
        if self.component.get("kind") in SIMULATION_COMPONENT_KINDS:
            self.setToolTip(
                f"{self.component.get('kind')}\nUID {self.uid}\n"
                "Simulation-only object — exported to Lumerical notebooks, never to GDS"
            )
        else:
            self.setToolTip(
                f"{self.component.get('kind')}\nUID {self.uid}\n"
                f"Layer map: 1 WG, 2 GC, 3 Marker, 4 RF, 5 Probe, 6 Ebeam"
            )

    def add_port_markers(self) -> None:
        try:
            ports = component_local_ports(self.component)
        except Exception:
            ports = {}
        for name, port in ports.items():
            domain = str(port.get("domain", "optical"))
            if name == "center":
                color = QColor("#ffffff")
                diameter = 10
            elif domain == "alignment":
                color = QColor("#4ade80")
                diameter = 8
            elif domain == "rf":
                color = QColor("#2563eb")
                diameter = 8
            else:
                color = QColor("#38bdf8")
                diameter = 8
            marker = QGraphicsEllipseItem(
                QRectF(-diameter / 2, -diameter / 2, diameter, diameter),
                self,
            )
            center = port["center"]
            marker.setPos(float(center[0]), -float(center[1]))
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            marker.setZValue(150)
            marker.setBrush(QBrush(color))
            pen = QPen(QColor("#0f172a"), 1.0)
            pen.setCosmetic(True)
            marker.setPen(pen)
            marker.setData(2, "port")
            marker.setToolTip(
                f"{name}\nDomain: {domain}\n"
                f"({float(center[0]):.6f}, {float(center[1]):.6f}) µm"
            )
            self.port_items.append(marker)

    def add_resize_handles(self) -> None:
        rect = self.geometry_bounds
        positions = {
            "nw": rect.topLeft(),
            "ne": rect.topRight(),
            "se": rect.bottomRight(),
            "sw": rect.bottomLeft(),
        }
        for corner, position in positions.items():
            handle = ResizeHandleItem(self, corner, position)
            self.resize_handles.append(handle)

    def set_handles_visible(self, visible: bool) -> None:
        for handle in self.resize_handles:
            handle.setVisible(bool(visible))

    def sync_transform(self) -> None:
        self.setPos(float(self.component.get("x", 0.0)), -float(self.component.get("y", 0.0)))
        self.setRotation(-float(self.component.get("orientation_deg", 0.0)))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            x, y = scene_to_world_point(self.pos())
            self.component["x"] = x
            self.component["y"] = y
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.set_handles_visible(bool(value))
            QTimer.singleShot(0, self.main_window.on_scene_selection_changed)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        self._drag_snapshot = self.main_window.snapshot()
        self._drag_start_position = QPointF(self.pos())
        self._press_scene_position = QPointF(event.scenePos())
        # A manually moved component detaches first; snapping can create a
        # fresh exact attachment when the mouse is released.
        self.component["attachment"] = None
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            group_id = self.component.get("group_id")
            if group_id and self.component.get("kind") not in SIMULATION_COMPONENT_KINDS:
                self.main_window.select_group(str(group_id))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        start_payload = self._drag_snapshot
        self._drag_snapshot = None
        press_scene_position = (
            QPointF(self._press_scene_position)
            if self._press_scene_position is not None
            else None
        )
        self._press_scene_position = None
        try:
            super().mouseReleaseEvent(event)
        except Exception as exc:
            # Do not let a Qt/PySide event-wrapper exception escape: affected
            # PySide6 macOS builds can segfault while formatting that error.
            self.main_window.statusBar().showMessage(
                f"Selection event recovered safely: {exc}", 6000
            )
        moved = QLineF(self._drag_start_position, self.pos()).length() > 1e-9
        # Finish bookkeeping after Qt has returned from the graphics-scene
        # mouse event.  This avoids re-entering itemAt/property rebuilding
        # while QGraphicsScene is still dispatching the release to a child.
        QTimer.singleShot(
            0,
            lambda: self._finish_mouse_release(
                start_payload,
                moved,
                press_scene_position,
            ),
        )

    def _finish_mouse_release(
        self,
        start_payload: str | None,
        moved: bool,
        press_scene_position: QPointF | None,
    ) -> None:
        try:
            if moved and self.component.get("kind") in SIMULATION_COMPONENT_KINDS:
                # A port/monitor/fiber dragged by hand remains part of the
                # export group but no longer snaps back to its starter point.
                self.component["auto_placed"] = False
            self.main_window.move_group_with_primary(self)
            if moved:
                self.main_window.snap_component_after_move(self.uid)
            if start_payload is not None:
                self.main_window.commit_interaction_snapshot(start_payload)
            self.main_window.refresh_project_tree()
            self.main_window.on_scene_selection_changed()
            if not moved and press_scene_position is not None:
                self.main_window.show_properties_for_scene_click(
                    self,
                    press_scene_position,
                )
        except Exception as exc:
            self.main_window.statusBar().showMessage(
                f"Selection update recovered safely: {exc}", 6000
            )


class WriteFieldGroupHandle(QGraphicsRectItem):
    """Screen-sized handle that moves one complete write-field component."""
    def __init__(self, container: "EbeamContainerItem") -> None:
        super().__init__(QRectF(-9,-9,18,18),container)
        self.container=container;self.setZValue(260);self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,True);self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton);self.setBrush(QBrush(QColor("#facc15")));pen=QPen(QColor("#ffffff"),1.5);pen.setCosmetic(True);self.setPen(pen);self.setCursor(Qt.CursorShape.SizeAllCursor);self.setToolTip("Drag all write fields together")
        self._snapshot=None;self._start_scene=QPointF();self._start_container=QPointF()

    def mousePressEvent(self,event) -> None:
        self._snapshot=self.container.main_window.snapshot();self._start_scene=event.scenePos();self._start_container=QPointF(self.container.pos());self.container.setSelected(True);event.accept()

    def mouseMoveEvent(self,event) -> None:
        delta=event.scenePos()-self._start_scene;self.container.setPos(self._start_container+delta);event.accept()

    def mouseReleaseEvent(self,event) -> None:
        snapshot=self._snapshot;self._snapshot=None
        if snapshot is not None:
            self.container.main_window.finish_ebeam_group_move(self.container.uid,snapshot)
        event.accept()


class WriteFieldItem(QGraphicsRectItem):
    def __init__(
        self,
        container: "EbeamContainerItem",
        field: dict[str, Any],
        global_order: int,
    ) -> None:
        self.container = container
        self.field = field
        self.field_key = str(field["field_key"])
        rect_data = field.get("rect")
        if "width" in field:
            width = float(field["width"])
        elif isinstance(rect_data, (list, tuple)) and len(rect_data) == 4:
            width = float(rect_data[2]) - float(rect_data[0])
        else:
            width = float(container.component.get("params", {}).get("field_size", 520.0))
        if "height" in field:
            height = float(field["height"])
        elif isinstance(rect_data, (list, tuple)) and len(rect_data) == 4:
            height = float(rect_data[3]) - float(rect_data[1])
        else:
            height = float(container.component.get("params", {}).get("field_size", 520.0))
        super().__init__(QRectF(-width / 2, -height / 2, width, height), container)
        center = field["center"]
        self.setPos(float(center[0]), -float(center[1]))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(30)
        pen = QPen(color_for_layer(EBEAM_LAYER, 240), 0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(color_for_layer(EBEAM_LAYER, 24)))
        self.order_label = QGraphicsSimpleTextItem(str(global_order), self)
        self.order_label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.order_label.setBrush(QBrush(QColor("#ffffff")))
        font = QFont()
        font.setBold(True)
        self.order_label.setFont(font)
        self.order_label.setPos(-width / 2 + 5, -height / 2 + 5)
        self.order_label.setVisible(True)
        self._drag_snapshot: str | None = None
        self.playback_state = "future"
        self.setToolTip(f"Write field {global_order}\n{self.field_key}")

    def set_playback_state(self, state: str) -> None:
        self.playback_state = state
        pen = QPen(color_for_layer(EBEAM_LAYER, 245), 0)
        pen.setCosmetic(True)
        if state == "active":
            pen.setColor(QColor("#facc15"))
            pen.setWidthF(3.0)
            self.setBrush(QBrush(QColor(250, 204, 21, 95)))
        elif state == "complete":
            pen.setColor(QColor("#22c55e"))
            pen.setWidthF(2.0)
            self.setBrush(QBrush(QColor(34, 197, 94, 70)))
        else:
            pen.setWidthF(1.5)
            self.setBrush(QBrush(color_for_layer(EBEAM_LAYER, 28)))
        self.setPen(pen)

    def set_global_order(self, order: int) -> None:
        self.order_label.setText(str(order))
        self.setToolTip(f"Write field {order}\n{self.field_key}")

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            local_center = (float(self.pos().x()), -float(self.pos().y()))
            base = self.field.get("base_center", self.field["center"])
            offsets = dict(self.container.component["params"].get("manual_field_offsets", {}))
            offsets[self.field_key] = [
                local_center[0] - float(base[0]),
                local_center[1] - float(base[1]),
            ]
            self.container.component["params"]["manual_field_offsets"] = offsets
            self.field["center"] = local_center
            width = float(self.rect().width())
            height = float(self.rect().height())
            self.field["rect"] = [
                local_center[0] - width / 2,
                local_center[1] - height / 2,
                local_center[0] + width / 2,
                local_center[1] + height / 2,
            ]
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if bool(value):
                self.container.main_window.active_field = (self.container.uid, self.field_key)
            QTimer.singleShot(0, self.container.main_window.on_scene_selection_changed)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        self._drag_snapshot = self.container.main_window.snapshot()
        self.container.main_window.active_field = (self.container.uid, self.field_key)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        snapshot = self._drag_snapshot
        self._drag_snapshot = None
        super().mouseReleaseEvent(event)
        if snapshot is not None:
            self.container.main_window.finish_individual_field_move(self.container.uid,self.field_key,snapshot)


class EbeamContainerItem(QGraphicsObject):
    def __init__(self, main_window: "NativeLayoutWindow", component: dict[str, Any]) -> None:
        super().__init__()
        self.main_window = main_window
        self.component = component
        self.uid = int(component["uid"])
        self.field_items: dict[str, WriteFieldItem] = {}
        self.group_handle: WriteFieldGroupHandle | None = None
        self._bounds = QRectF()
        self._drag_snapshot: str | None = None
        self._drag_start_position = QPointF()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(20)
        self.sync_transform()
        self.rebuild_fields()

    def boundingRect(self) -> QRectF:
        return self._bounds.adjusted(-8, -8, 8, 8)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        # The complete Ebeam boundary remains visible even when the component
        # is not selected, so generated coverage can always be inspected.
        pen = QPen(QColor("#ffffff") if self.isSelected() else color_for_layer(EBEAM_LAYER, 255), 0)
        pen.setCosmetic(True)
        pen.setWidthF(2.4 if self.isSelected() else 1.8)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(color_for_layer(EBEAM_LAYER, 18)))
        painter.drawRect(self._bounds)
        center_pen = QPen(QColor("#ffffff"), 0)
        center_pen.setCosmetic(True)
        painter.setPen(center_pen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        radius = max(1.5, 4.0 / max(abs(self.sceneTransform().m11()), 1e-9))
        painter.drawEllipse(QPointF(0.0, 0.0), radius, radius)

    def sync_transform(self) -> None:
        self.setPos(float(self.component.get("x", 0.0)), -float(self.component.get("y", 0.0)))
        self.setRotation(-float(self.component.get("orientation_deg", 0.0)))

    def rebuild_fields(self) -> None:
        for item in list(self.field_items.values()):
            item.setParentItem(None)
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.field_items.clear()
        layout = multipass_field_layout(self.component["params"])
        rect_union: QRectF | None = None
        for field in layout["fields"]:
            item = WriteFieldItem(self, field, int(field["order"]))
            self.field_items[str(field["field_key"])] = item
            mapped = item.mapRectToParent(item.rect())
            rect_union = mapped if rect_union is None else rect_union.united(mapped)
        self.prepareGeometryChange()
        self._bounds = rect_union or QRectF(-10, -10, 20, 20)
        if self.group_handle is None:self.group_handle=WriteFieldGroupHandle(self)
        self.group_handle.setPos(self._bounds.left(),self._bounds.top())
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            x, y = scene_to_world_point(self.pos())
            self.component["x"] = x
            self.component["y"] = y
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            QTimer.singleShot(0, self.main_window.on_scene_selection_changed)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        self._drag_snapshot = self.main_window.snapshot()
        self._drag_start_position = QPointF(self.pos())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        snapshot = self._drag_snapshot
        self._drag_snapshot = None
        super().mouseReleaseEvent(event)
        if snapshot is not None:
            moved = QLineF(self._drag_start_position, self.pos()).length() > 1e-9
            if moved:
                self.main_window.finish_ebeam_group_move(self.uid, snapshot)
            else:
                self.main_window.commit_interaction_snapshot(snapshot)
        self.main_window.refresh_project_tree()
