"""Rotation, point transforms, and polygon placement."""

from __future__ import annotations

from typing import Any
import math

import gdstk
import numpy as np

from PySide6.QtCore import QPointF


def rot(point: tuple[float, float] | np.ndarray, angle_deg: float) -> np.ndarray:
    t = math.radians(angle_deg)
    matrix = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=float)
    return matrix @ np.asarray(point, dtype=float)


def transform_points(points: np.ndarray, center: tuple[float, float], angle_deg: float) -> np.ndarray:
    t = math.radians(angle_deg)
    values = np.asarray(points, dtype=float)
    cosine, sine = math.cos(t), math.sin(t)
    # Explicit 2-D rotation avoids spurious Accelerate/BLAS matmul warnings on
    # large (>10k vertex) photolithography taper polygons.
    return np.column_stack((
        float(center[0]) + values[:, 0] * cosine - values[:, 1] * sine,
        float(center[1]) + values[:, 0] * sine + values[:, 1] * cosine,
    ))


def add_local_polygon(top: gdstk.Cell, points: np.ndarray, center: tuple[float, float], orientation: float, layer: int, datatype: int) -> None:
    top.add(gdstk.Polygon(transform_points(points, center, orientation), layer=layer, datatype=datatype))


def _transform_polygon(polygon: gdstk.Polygon, center: tuple[float, float], orientation: float) -> gdstk.Polygon:
    """Return a transformed copy without changing the source polygon."""
    result = polygon.copy()
    if orientation:
        result.rotate(math.radians(float(orientation)), center=(0.0, 0.0))
    result.translate(float(center[0]), float(center[1]))
    return result


def world_to_scene_point(point: tuple[float, float] | list[float] | np.ndarray) -> QPointF:
    return QPointF(float(point[0]), -float(point[1]))


def scene_to_world_point(point: QPointF) -> tuple[float, float]:
    return (float(point.x()), -float(point.y()))


def transformed_local_points(
    points: list[tuple[float, float]] | np.ndarray,
    component: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if bool(component.get("mirrored", False)):
        values = values.copy()
        values[:, 1] *= -1.0
    return transform_points(
        values,
        (float(component.get("x", 0.0)), float(component.get("y", 0.0))),
        float(component.get("orientation_deg", 0.0)),
    )
