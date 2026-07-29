"""Low-level gdstk cell and polygon primitives."""

from __future__ import annotations

from typing import Any
import re

import gdstk
import numpy as np

from ..geometry.transforms import transform_points


def _safe_component_cell_name(component: dict[str, Any]) -> str:
    uid = int(component.get("uid", 0))
    slug = re.sub(r"[^A-Za-z0-9_]", "_", str(component.get("kind", "COMPONENT"))).strip("_") or "COMPONENT"
    return f"C_{uid:06d}_{slug}"[:32]


def _merge_touching_component_polygons(cell: gdstk.Cell) -> None:
    """Boolean-union touching polygons per layer/datatype inside one component cell.

    This removes microscopic seams at interfaces while retaining separate GDS
    polygons when the shapes are intentionally disconnected (for example GC
    teeth or CPW conductors).  The component remains grouped by its subcell.
    """
    grouped: dict[tuple[int, int], list[gdstk.Polygon]] = {}
    for polygon in list(cell.polygons):
        grouped.setdefault((int(polygon.layer), int(polygon.datatype)), []).append(polygon)
    if not grouped:
        return
    for polygon in list(cell.polygons):
        cell.remove(polygon)
    for (layer, datatype), polygons in grouped.items():
        if len(polygons) == 1:
            cell.add(polygons[0])
            continue
        merged = gdstk.boolean(polygons, [], "or", layer=layer, datatype=datatype)
        cell.add(*(merged or polygons))


def copy_cell_polygons_to_top(source_cell: gdstk.Cell, top_cell: gdstk.Cell) -> None:
    for polygon in source_cell.polygons:
        top_cell.add(gdstk.Polygon(np.asarray(polygon.points).copy(), layer=int(polygon.layer), datatype=int(polygon.datatype)))


def add_rect(top: gdstk.Cell, local: np.ndarray, center: tuple[float, float], orientation: float, layer: int, datatype: int) -> None:
    top.add(gdstk.Polygon(transform_points(local, center, orientation), layer=layer, datatype=datatype))
