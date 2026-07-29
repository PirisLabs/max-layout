"""Analytic polygon generators for rings, arcs, and capsules."""

from __future__ import annotations

from typing import Any

import gdstk
import numpy as np


def mmi_total_length(p: dict[str, Any]) -> float:
    return float(p["input_length"]) + float(p["input_taper_length"]) + float(p["mmi_length"]) + float(p["output_taper_length"]) + float(p["output_length"])


def cpw_annular_sector_points(
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    start_angle_rad: float,
    end_angle_rad: float,
    points: int,
) -> np.ndarray:
    count = max(16, int(points))
    angles = np.linspace(start_angle_rad, end_angle_rad, count)
    outer = np.column_stack(
        (
            center[0] + outer_radius * np.cos(angles),
            center[1] + outer_radius * np.sin(angles),
        )
    )
    reverse_angles = angles[::-1]
    inner = np.column_stack(
        (
            center[0] + inner_radius * np.cos(reverse_angles),
            center[1] + inner_radius * np.sin(reverse_angles),
        )
    )
    return np.vstack((outer, inner))


def annulus_points(radius: float, width: float, points: int = 256) -> np.ndarray:
    """Legacy preview helper retained for compatibility.

    GDS export uses ``gdstk.ellipse(..., inner_radius=...)`` below so the ring
    is a physically continuous closed annulus without an open seam.
    """
    if radius <= 0 or width <= 0 or radius <= width / 2.0:
        raise ValueError("Ring radius must be larger than half the waveguide width.")
    count = max(32, int(points))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    outer = np.column_stack(((radius + width / 2.0) * np.cos(angles), (radius + width / 2.0) * np.sin(angles)))
    inner_angles = angles[::-1]
    inner = np.column_stack(((radius - width / 2.0) * np.cos(inner_angles), (radius - width / 2.0) * np.sin(inner_angles)))
    return np.vstack((outer, inner))


def annular_arc_points(center: tuple[float, float], radius: float, width: float, angle_start: float, angle_end: float, points: int = 128) -> np.ndarray:
    count = max(16, int(points))
    angles = np.linspace(angle_start, angle_end, count)
    outer_radius = radius + width / 2.0
    inner_radius = radius - width / 2.0
    if inner_radius <= 0:
        raise ValueError("Racetrack radius must be larger than half the waveguide width.")
    outer = np.column_stack((center[0] + outer_radius * np.cos(angles), center[1] + outer_radius * np.sin(angles)))
    reverse = angles[::-1]
    inner = np.column_stack((center[0] + inner_radius * np.cos(reverse), center[1] + inner_radius * np.sin(reverse)))
    return np.vstack((outer, inner))


def _capsule_polygons(cx: float, cy: float, radius: float, coupling_length: float, layer: int, datatype: int, tolerance: float) -> list[gdstk.Polygon]:
    """Create a filled horizontal capsule centered at ``(cx, cy)``."""
    half = coupling_length / 2.0
    shapes: list[gdstk.Polygon] = [
        gdstk.rectangle((cx - half, cy - radius), (cx + half, cy + radius), layer=layer, datatype=datatype),
        gdstk.ellipse((cx - half, cy), radius, tolerance=tolerance, layer=layer, datatype=datatype),
        gdstk.ellipse((cx + half, cy), radius, tolerance=tolerance, layer=layer, datatype=datatype),
    ]
    merged = gdstk.boolean(shapes, [], "or", layer=layer, datatype=datatype)
    return list(merged or shapes)
