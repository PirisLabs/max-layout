"""E-beam write-field planning and field annotation."""

from __future__ import annotations

from typing import Any
import math

import gdstk
import numpy as np

from ..constants import DEFAULT_DATATYPE, EBEAM_LAYER, MARKER_LAYER
from ..gds.primitives import add_rect
from ..geometry.transforms import transform_points


def add_ebeam_field_outline(top: gdstk.Cell, center: tuple[float,float], orientation: float, size: float) -> None:
    """Add a lightweight 520-µm write-field boundary on the E-beam layer."""
    half=float(size)/2;line=1.0
    for points in (
        [[-half,-half], [half,-half], [half,-half+line], [-half,-half+line]],
        [[-half,half-line], [half,half-line], [half,half], [-half,half]],
        [[-half,-half], [-half+line,-half], [-half+line,half], [-half,half]],
        [[half-line,-half], [half,-half], [half,half], [half-line,half]],
    ):add_rect(top,np.asarray(points,float),center,orientation,EBEAM_LAYER,DEFAULT_DATATYPE)


def add_ebeam_parameter_text(top: gdstk.Cell, text: str, center: tuple[float,float], orientation: float, field_size: float, height: float = 10.0) -> None:
    """Add compact, fabrication-safe polygon text just above a write field."""
    height=float(height)
    if height<=0:return
    polygons=list(gdstk.text(str(text),height,(0,0),layer=MARKER_LAYER,datatype=DEFAULT_DATATYPE))
    if not polygons:return
    points=np.vstack([np.asarray(poly.points,float) for poly in polygons]);text_center=(points.min(axis=0)+points.max(axis=0))/2
    local_center=np.array([0.0,float(field_size)/2+height],float)
    for polygon in polygons:
        local=np.asarray(polygon.points,float)-text_center+local_center
        top.add(gdstk.Polygon(transform_points(local,center,orientation),layer=MARKER_LAYER,datatype=DEFAULT_DATATYPE))


def geometry_field_coverage(
    fields: list[dict[str, Any]],
    source_polygons: list[gdstk.Polygon],
    component_center: tuple[float, float] = (0.0, 0.0),
    orientation_deg: float = 0.0,
    *,
    precision: float = 1e-6,
) -> tuple[set[str], dict[str, list[str]]]:
    """Return occupied fields and their true source-geometry connectivity.

    The write-field grid alone is ambiguous for a meander: two fields can
    touch even though the device travels between them only by going around a
    different row.  This helper clips the *unioned source geometry* to each
    touching pair of fields and adds an edge only when one clipped geometry
    component actually occupies both fields.  The resulting graph lets field
    numbering follow the fabricated path instead of a spatial serpentine.

    ``source_polygons`` are world-space GDS polygons.  Field rectangles are
    local to the E-beam component, so the source polygons are transformed back
    into that local coordinate system before Boolean operations.
    """

    if not fields or not source_polygons:
        return set(), {
            str(field.get("field_key", "")): []
            for field in fields
            if str(field.get("field_key", ""))
        }

    center_x, center_y = map(float, component_center)
    theta = math.radians(float(orientation_deg))
    cosine, sine = math.cos(theta), math.sin(theta)

    local_sources: list[gdstk.Polygon] = []
    for polygon in source_polygons:
        points = np.asarray(polygon.points, dtype=float)
        dx = points[:, 0] - center_x
        dy = points[:, 1] - center_y
        local_points = np.column_stack((
            dx * cosine + dy * sine,
            -dx * sine + dy * cosine,
        ))
        local_sources.append(gdstk.Polygon(local_points))
    merged_sources = gdstk.boolean(
        local_sources, [], "or", precision=float(precision)
    )
    if not merged_sources:
        return set(), {
            str(field.get("field_key", "")): []
            for field in fields
            if str(field.get("field_key", ""))
        }

    # A source component can contain hundreds or thousands of mutually distant
    # polygons.  Passing that complete list into every field Boolean makes the
    # clipping cost grow with ``field count * complete component complexity``.
    # Keep the exact Boolean decisions, but first reject polygons whose bounding
    # boxes cannot possibly touch the requested window.  This inexpensive
    # spatial prefilter is especially important for dense photonic test blocks.
    source_bounds: list[tuple[float, float, float, float]] = []
    for polygon in merged_sources:
        bounding_box = np.asarray(polygon.bounding_box(), dtype=float)
        source_bounds.append((
            float(bounding_box[0, 0]),
            float(bounding_box[0, 1]),
            float(bounding_box[1, 0]),
            float(bounding_box[1, 1]),
        ))

    spatial_tolerance = max(float(precision) * 2.0, 1e-9)

    def sources_near(
        bounds: tuple[float, float, float, float],
    ) -> list[gdstk.Polygon]:
        x0, y0, x1, y1 = bounds
        return [
            polygon
            for polygon, (px0, py0, px1, py1) in zip(
                merged_sources, source_bounds
            )
            if px1 >= x0 - spatial_tolerance
            and px0 <= x1 + spatial_tolerance
            and py1 >= y0 - spatial_tolerance
            and py0 <= y1 + spatial_tolerance
        ]

    rectangles: dict[str, tuple[tuple[float, float, float, float], gdstk.Polygon]] = {}
    for field in fields:
        key = str(field.get("field_key", ""))
        rect = field.get("rect")
        if not key or not isinstance(rect, (list, tuple)) or len(rect) != 4:
            continue
        x0, y0, x1, y1 = map(float, rect)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        rectangles[key] = (
            (x0, y0, x1, y1),
            gdstk.rectangle((x0, y0), (x1, y1)),
        )

    occupied: set[str] = set()
    for key, (bounds, rectangle) in rectangles.items():
        nearby_sources = sources_near(bounds)
        if nearby_sources and gdstk.boolean(
            nearby_sources, [rectangle], "and", precision=float(precision)
        ):
            occupied.add(key)

    adjacency: dict[str, set[str]] = {key: set() for key in occupied}
    occupied_keys = sorted(occupied)
    tolerance = spatial_tolerance
    for index, key_a in enumerate(occupied_keys):
        bounds_a, rectangle_a = rectangles[key_a]
        ax0, ay0, ax1, ay1 = bounds_a
        for key_b in occupied_keys[index + 1:]:
            bounds_b, rectangle_b = rectangles[key_b]
            bx0, by0, bx1, by1 = bounds_b
            x_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            y_gap = max(0.0, max(ay0, by0) - min(ay1, by1))
            if x_gap > tolerance or y_gap > tolerance:
                continue

            x_overlap = min(ax1, bx1) - max(ax0, bx0)
            y_overlap = min(ay1, by1) - max(ay0, by0)
            if x_overlap <= tolerance and y_overlap <= tolerance:
                # A single shared corner does not establish write-path
                # continuity.  This explicitly prevents diagonal jumps.
                continue

            # Restrict the geometry to this pair of fields.  A clipped
            # component must have non-zero area in *both* rectangles; paths
            # that reconnect only through a third field therefore do not
            # create a false shortcut edge.
            pair_bounds = (
                min(ax0, bx0), min(ay0, by0),
                max(ax1, bx1), max(ay1, by1),
            )
            pair_window = gdstk.rectangle(
                pair_bounds[:2], pair_bounds[2:]
            )
            nearby_sources = sources_near(pair_bounds)
            if not nearby_sources:
                continue
            clipped = gdstk.boolean(
                nearby_sources,
                [pair_window],
                "and",
                precision=float(precision),
            )
            connected = False
            for geometry_piece in clipped:
                in_a = gdstk.boolean(
                    [geometry_piece], [rectangle_a], "and",
                    precision=float(precision),
                )
                if not in_a:
                    continue
                in_b = gdstk.boolean(
                    [geometry_piece], [rectangle_b], "and",
                    precision=float(precision),
                )
                if in_b:
                    connected = True
                    break
            if connected:
                adjacency[key_a].add(key_b)
                adjacency[key_b].add(key_a)

    # Include occupied endpoints with an explicit empty list.  Consumers can
    # distinguish a real geometry endpoint/island from legacy data that has no
    # connectivity graph at all.
    return occupied, {
        key: sorted(adjacency.get(key, set())) for key in sorted(occupied)
    }


def multipass_field_layout(params: dict[str, Any]) -> dict[str, Any]:
    """Calculate centered, ordered, and optionally hand-adjusted square write fields."""

    raw_manual_order = params.get("manual_field_order", {})
    manual_order = raw_manual_order if isinstance(raw_manual_order, dict) else {}

    def requested_first_key(active_keys: set[str]) -> str | None:
        """Return the user-selected first active field, if one was assigned."""
        candidates: list[str] = []
        for raw_key, raw_value in manual_order.items():
            key = str(raw_key)
            if key not in active_keys:
                continue
            try:
                requested = int(round(float(raw_value)))
            except (TypeError, ValueError):
                continue
            if requested == 1:
                candidates.append(key)
        return min(candidates) if candidates else None

    explicit_fields = params.get("explicit_fields")
    if isinstance(explicit_fields, list) and explicit_fields:
        raw_offsets = params.get("manual_field_offsets", {})
        manual_offsets = raw_offsets if isinstance(raw_offsets, dict) else {}
        removed_keys = {
            str(value)
            for value in params.get("removed_field_keys", [])
        } if isinstance(params.get("removed_field_keys", []), (list, tuple, set)) else set()
        auto_pruned_keys = {
            str(value)
            for value in params.get("auto_pruned_field_keys", [])
        } if isinstance(params.get("auto_pruned_field_keys", []), (list, tuple, set)) else set()

        fields: list[dict[str, Any]] = []
        for index, entry in enumerate(explicit_fields, start=1):
            if not isinstance(entry, dict):
                continue
            bounds = entry.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                continue
            try:
                x0, y0, x1, y1 = (float(value) for value in bounds)
            except (TypeError, ValueError):
                continue
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            field_key = str(entry.get("field_key") or f"import_{index}")
            if field_key in removed_keys or field_key in auto_pruned_keys:
                continue
            raw_offset = manual_offsets.get(field_key, (0.0, 0.0))
            try:
                dx = float(raw_offset[0])
                dy = float(raw_offset[1])
            except (TypeError, ValueError, IndexError, KeyError):
                dx = 0.0
                dy = 0.0
            base_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            center = (base_center[0] + dx, base_center[1] + dy)
            width = x1 - x0
            height = y1 - y0
            fields.append({
                "order": len(fields) + 1,
                "field_key": field_key,
                "column": index,
                "row": 1,
                "base_center": base_center,
                "manual_offset": (dx, dy),
                "center": center,
                "width": width,
                "height": height,
                "rect": (
                    center[0] - width / 2.0,
                    center[1] - height / 2.0,
                    center[0] + width / 2.0,
                    center[1] + height / 2.0,
                ),
                "region_name": str(entry.get("region_name") or f"R{index}"),
            })

        # Imported/explicit fields do not have grid row/column connectivity.
        # When the user chooses field 1, walk outward from that exact field by
        # the shortest physical jump instead of retaining the import order.
        # This makes the remaining sequence follow an arbitrarily oriented or
        # irregular device without assuming it begins at a particular corner.
        active_explicit_keys = {
            str(field["field_key"]) for field in fields
        }
        explicit_prefix: list[str] = []
        if fields:
            requested_by_order: dict[int, str] = {}
            for raw_key, raw_value in manual_order.items():
                key = str(raw_key)
                if key not in active_explicit_keys:
                    continue
                try:
                    requested = int(round(float(raw_value)))
                except (TypeError, ValueError):
                    continue
                if 1 <= requested <= len(fields):
                    requested_by_order.setdefault(requested, key)
            while len(explicit_prefix) + 1 in requested_by_order:
                explicit_prefix.append(
                    requested_by_order[len(explicit_prefix) + 1]
                )
        first_explicit_key = (
            explicit_prefix[0] if explicit_prefix else None
        )
        if fields and first_explicit_key is not None:
            by_explicit_key = {
                str(field["field_key"]): field for field in fields
            }
            route = [by_explicit_key[key] for key in explicit_prefix]
            remaining = {
                key for key in by_explicit_key if key not in explicit_prefix
            }
            while remaining:
                current = route[-1]
                cx, cy = map(float, current["center"])
                next_key = min(
                    remaining,
                    key=lambda key: (
                        (float(by_explicit_key[key]["center"][0]) - cx) ** 2
                        + (float(by_explicit_key[key]["center"][1]) - cy) ** 2,
                        str(key),
                    ),
                )
                route.append(by_explicit_key[next_key])
                remaining.remove(next_key)
            fields = route

        if fields and manual_order:
            count = len(fields)
            active_by_key = {str(field["field_key"]): field for field in fields}
            slots: list[dict[str, Any] | None] = [None] * count
            assigned_keys: set[str] = set()
            assignments: list[tuple[str, int]] = []
            for key, value in manual_order.items():
                key = str(key)
                if key not in active_by_key:
                    continue
                try:
                    requested = int(round(float(value)))
                except (TypeError, ValueError):
                    continue
                assignments.append((key, requested))
            assignments.sort(key=lambda item: item[1])
            for key, requested in assignments:
                position = max(1, min(count, requested)) - 1
                while position < count and slots[position] is not None:
                    position += 1
                if position >= count:
                    position = max(0, min(count - 1, requested - 1))
                    while position >= 0 and slots[position] is not None:
                        position -= 1
                if position >= 0 and slots[position] is None:
                    slots[position] = active_by_key[key]
                    assigned_keys.add(key)
            remaining = [
                field for field in fields
                if str(field["field_key"]) not in assigned_keys
            ]
            remaining_index = 0
            for slot_index in range(count):
                if slots[slot_index] is None:
                    slots[slot_index] = remaining[remaining_index]
                    remaining_index += 1
            fields = [field for field in slots if field is not None]

        for order, field in enumerate(fields, start=1):
            field["order"] = order

        if fields:
            x0 = min(float(field["rect"][0]) for field in fields)
            y0 = min(float(field["rect"][1]) for field in fields)
            x1 = max(float(field["rect"][2]) for field in fields)
            y1 = max(float(field["rect"][3]) for field in fields)
        else:
            x0 = y0 = x1 = y1 = 0.0
        nominal_size = max(
            (
                max(float(field["width"]), float(field["height"]))
                for field in fields
            ),
            default=float(params.get("field_size", 520.0) or 520.0),
        )
        return {
            "fields": fields,
            "field_size": nominal_size,
            "step_x": 0.0,
            "step_y": 0.0,
            "nx": len(fields),
            "ny": 1 if fields else 0,
            "total_grid_fields": len(explicit_fields),
            "auto_pruned_count": len(auto_pruned_keys),
            "required_width": x1 - x0,
            "required_height": y1 - y0,
            "coverage_width": x1 - x0,
            "coverage_height": y1 - y0,
            "margin_x": 0.0,
            "margin_y": 0.0,
            "coverage_bounds": (x0, y0, x1, y1),
        }

    field_size = float(params.get("field_size", 520.0))
    if field_size <= 0:
        raise ValueError("E-beam write-field size A must be positive.")

    target_width = max(0.0, float(params.get("target_width", 0.0)))
    target_height = max(0.0, float(params.get("target_height", 0.0)))
    clearance = max(0.0, float(params.get("edge_clearance", 10.0)))
    required_width = target_width + 2.0 * clearance
    required_height = target_height + 2.0 * clearance

    overlap_x = float(params.get("overlap_x_percent", 0.0)) if bool(params.get("overlap_x_enabled", False)) else 0.0
    overlap_y = float(params.get("overlap_y_percent", 0.0)) if bool(params.get("overlap_y_enabled", False)) else 0.0
    if not 0.0 <= overlap_x < 100.0 or not 0.0 <= overlap_y < 100.0:
        raise ValueError("E-beam overlap percentages must be at least 0 and less than 100.")

    step_x = field_size * (1.0 - overlap_x / 100.0)
    step_y = field_size * (1.0 - overlap_y / 100.0)
    if step_x <= 0 or step_y <= 0:
        raise ValueError("E-beam write-field steps must be positive.")

    def count_for(span: float, step: float) -> int:
        if span <= field_size + 1e-12:
            return 1
        return int(math.ceil((span - field_size) / step - 1e-12)) + 1

    nx = count_for(required_width, step_x)
    ny = count_for(required_height, step_y)

    # Center the complete write-field grid on the covered geometry.  Any extra
    # span caused by the discrete field pitch is split equally between both
    # sides, so the device has at least the requested clearance in every
    # direction whenever the automatic grid is used.
    grid_width = field_size + max(0, nx - 1) * step_x
    grid_height = field_size + max(0, ny - 1) * step_y
    xs = [-grid_width / 2.0 + field_size / 2.0 + i * step_x for i in range(nx)]
    ys = [grid_height / 2.0 - field_size / 2.0 - j * step_y for j in range(ny)]

    raw_offsets = params.get("manual_field_offsets", {})
    manual_offsets = raw_offsets if isinstance(raw_offsets, dict) else {}
    raw_removed = params.get("removed_field_keys", [])
    removed_keys = {str(value) for value in raw_removed} if isinstance(raw_removed, (list, tuple, set)) else set()
    raw_auto_pruned = params.get("auto_pruned_field_keys", [])
    auto_pruned_keys = {str(value) for value in raw_auto_pruned} if isinstance(raw_auto_pruned, (list, tuple, set)) else set()

    start_corner = str(params.get("start_corner", "top-left"))
    primary_axis = str(params.get("primary_axis", "x"))
    serpentine = bool(params.get("serpentine", True))
    x_indices = list(range(nx)) if start_corner.endswith("left") else list(reversed(range(nx)))
    y_indices = list(range(ny)) if start_corner.startswith("top") else list(reversed(range(ny)))

    ordered: list[dict[str, Any]] = []

    def add_field(ix: int, iy: int) -> None:
        field_key = f"c{ix + 1}_r{iy + 1}"
        if field_key in removed_keys or field_key in auto_pruned_keys:
            return
        offset_value = manual_offsets.get(field_key, (0.0, 0.0))
        try:
            dx = float(offset_value[0])
            dy = float(offset_value[1])
        except (TypeError, ValueError, IndexError, KeyError):
            dx = 0.0
            dy = 0.0
        base_center = (xs[ix], ys[iy])
        center = (base_center[0] + dx, base_center[1] + dy)
        ordered.append({
            "order": len(ordered) + 1,
            "field_key": field_key,
            "column": ix + 1,
            "row": iy + 1,
            "base_center": base_center,
            "manual_offset": (dx, dy),
            "center": center,
            "width": field_size,
            "height": field_size,
            "rect": (
                center[0] - field_size / 2.0,
                center[1] - field_size / 2.0,
                center[0] + field_size / 2.0,
                center[1] + field_size / 2.0,
            ),
        })

    if primary_axis == "y":
        for outer_index, ix in enumerate(x_indices):
            inner = y_indices if not (serpentine and outer_index % 2) else list(reversed(y_indices))
            for iy in inner:
                add_field(ix, iy)
    else:
        for outer_index, iy in enumerate(y_indices):
            inner = x_indices if not (serpentine and outer_index % 2) else list(reversed(x_indices))
            for ix in inner:
                add_field(ix, iy)

    # Renumber only active fields after pruning and manual movement.  When the
    # editor has recorded source-geometry connectivity, follow only that graph.
    # Legacy layouts without a graph retain the orthogonal-grid fallback.  A
    # shortest-distance jump is used only between genuinely disconnected
    # geometry islands or when a branch has no Hamiltonian path.
    if ordered:
        by_key = {
            (int(field["column"]), int(field["row"])): field
            for field in ordered
        }

        def field_distance_sq(a: dict[str, Any], b: dict[str, Any]) -> float:
            return (
                (float(a["center"][0]) - float(b["center"][0])) ** 2
                + (float(a["center"][1]) - float(b["center"][1])) ** 2
            )

        field_key_to_grid_key = {
            str(field["field_key"]): key for key, field in by_key.items()
        }
        raw_adjacency = params.get("geometry_field_adjacency", {})
        geometry_adjacency = (
            raw_adjacency if isinstance(raw_adjacency, dict) else {}
        )
        raw_physical_adjacency = params.get("field_adjacency", {})
        physical_adjacency = (
            raw_physical_adjacency
            if isinstance(raw_physical_adjacency, dict)
            else {}
        )

        def configured_neighbors(
            key: tuple[int, int], adjacency: dict[str, Any]
        ) -> set[tuple[int, int]]:
            field_key = str(by_key[key]["field_key"])
            raw_values = adjacency.get(field_key, [])
            if not isinstance(raw_values, (list, tuple, set)):
                return set()
            return {
                field_key_to_grid_key[str(value)]
                for value in raw_values
                if str(value) in field_key_to_grid_key
            }

        def neighbor_keys(key: tuple[int, int], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
            field_key = str(by_key[key]["field_key"])
            if field_key in geometry_adjacency:
                stored = configured_neighbors(key, geometry_adjacency)
                return [candidate for candidate in stored if candidate in allowed]
            if field_key in physical_adjacency:
                stored = configured_neighbors(key, physical_adjacency)
                return [candidate for candidate in stored if candidate in allowed]
            column, row = key
            # Backward-compatible fallback for projects saved before exact
            # geometry connectivity was recorded.  Avoid diagonal shortcuts:
            # they are the most common cause of visible numbering jumps.
            candidates = [
                (column - 1, row),
                (column + 1, row),
                (column, row - 1),
                (column, row + 1),
            ]
            return [candidate for candidate in candidates if candidate in allowed]

        requested_key_by_order: dict[int, tuple[int, int]] = {}
        for raw_key, raw_value in manual_order.items():
            grid_key = field_key_to_grid_key.get(str(raw_key))
            if grid_key is None:
                continue
            try:
                requested = int(round(float(raw_value)))
            except (TypeError, ValueError):
                continue
            requested_key_by_order.setdefault(requested, grid_key)

        consecutive_prefix: list[tuple[int, int]] = []
        while len(consecutive_prefix) + 1 in requested_key_by_order:
            consecutive_prefix.append(
                requested_key_by_order[len(consecutive_prefix) + 1]
            )

        all_keys = set(by_key)
        active_x = [float(field["center"][0]) for field in ordered]
        active_y = [float(field["center"][1]) for field in ordered]
        manual_first = requested_first_key({
            str(field["field_key"]) for field in ordered
        })
        first_key = next(
            (
                key for key, field in by_key.items()
                if str(field["field_key"]) == manual_first
            ),
            None,
        )
        if first_key is None:
            corner_x = min(active_x) if start_corner.endswith("left") else max(active_x)
            corner_y = max(active_y) if start_corner.startswith("top") else min(active_y)
            first_key = min(
                all_keys,
                key=lambda key: (
                    (float(by_key[key]["center"][0]) - corner_x) ** 2
                    + (float(by_key[key]["center"][1]) - corner_y) ** 2,
                    int(by_key[key]["row"]),
                    int(by_key[key]["column"]),
                ),
            )

        # Split the occupied write fields into orthogonally connected islands.
        components: list[set[tuple[int, int]]] = []
        unseen = set(all_keys)
        while unseen:
            seed = next(iter(unseen))
            component = {seed}
            stack = [seed]
            unseen.remove(seed)
            while stack:
                key = stack.pop()
                for neighbor in neighbor_keys(key, all_keys):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

        def component_path(
            component: set[tuple[int, int]],
            start_key: tuple[int, int],
        ) -> list[tuple[int, int]]:
            """Walk one geometry island without combinatorial backtracking.

            Real device meanders normally form a degree-one/two path, so the
            minimum-unused-degree rule follows them exactly.  Branched or solid
            occupied regions use the same deterministic rule and make a shortest
            spatial jump only if a branch cannot be covered without revisiting a
            field.  Runtime is linear for path-like geometry and near-linear for
            the small-degree write-field graphs used by the editor.
            """
            if len(component) <= 1:
                return [start_key]
            route = [start_key]
            remaining = set(component)
            remaining.remove(start_key)
            while remaining:
                current_key = route[-1]
                adjacent = neighbor_keys(current_key, remaining)
                expected_key = requested_key_by_order.get(len(route) + 1)
                pool = adjacent if adjacent else remaining
                next_key = min(
                    pool,
                    key=lambda key: (
                        0 if key == expected_key else 1,
                        len(neighbor_keys(key, remaining - {key})),
                        field_distance_sq(by_key[current_key], by_key[key]),
                        abs(int(by_key[key]["row"]) - int(by_key[current_key]["row"]))
                        if primary_axis == "x"
                        else abs(int(by_key[key]["column"]) - int(by_key[current_key]["column"])),
                        int(by_key[key]["row"]),
                        int(by_key[key]["column"]),
                    ),
                )
                route.append(next_key)
                remaining.remove(next_key)
            return route

        first_component_index = next(
            index for index, component in enumerate(components) if first_key in component
        )
        ordered_components = [components.pop(first_component_index)]
        first_component = ordered_components[0]
        valid_prefix = (
            consecutive_prefix
            if consecutive_prefix
            and consecutive_prefix[0] == first_key
            and all(key in first_component for key in consecutive_prefix)
            and all(
                second in neighbor_keys(first, first_component)
                for first, second in zip(
                    consecutive_prefix, consecutive_prefix[1:]
                )
            )
            else [first_key]
        )
        if len(valid_prefix) > 1:
            # Reserve the explicit direction prefix and solve the remaining
            # geometry from its last field.  This is how field 2 disambiguates
            # an equal left/right branch after choosing a top-center field 1.
            tail_component = first_component - set(valid_prefix[:-1])
            tail = component_path(tail_component, valid_prefix[-1])
            route_keys = valid_prefix[:-1] + tail
        else:
            route_keys = component_path(first_component, first_key)

        while components:
            current_key = route_keys[-1]
            best_component_index, best_entry = min(
                (
                    (component_index, entry_key)
                    for component_index, component in enumerate(components)
                    for entry_key in component
                ),
                key=lambda item: (
                    field_distance_sq(by_key[current_key], by_key[item[1]]),
                    int(by_key[item[1]]["row"]),
                    int(by_key[item[1]]["column"]),
                ),
            )
            component = components.pop(best_component_index)
            route_keys.extend(component_path(component, best_entry))

        ordered = [by_key[key] for key in route_keys]
        for index, field in enumerate(ordered, start=1):
            field["order"] = index

    # Apply explicit user-assigned order numbers as fixed slots, then fill
    # every remaining slot using the automatic adjacent/minimum-travel route.
    if ordered and manual_order:
        count = len(ordered)
        active_by_key = {str(field["field_key"]): field for field in ordered}
        slots: list[dict[str, Any] | None] = [None] * count
        assigned_keys: set[str] = set()
        assignments: list[tuple[str, int, int]] = []
        automatic_index = {str(field["field_key"]): index for index, field in enumerate(ordered)}
        for key, value in manual_order.items():
            key = str(key)
            if key not in active_by_key:
                continue
            try:
                requested = int(round(float(value)))
            except (TypeError, ValueError):
                continue
            assignments.append((key, requested, automatic_index[key]))
        assignments.sort(key=lambda item: (item[1], item[2]))
        for key, requested, _ in assignments:
            position = max(1, min(count, requested)) - 1
            while position < count and slots[position] is not None:
                position += 1
            if position >= count:
                position = max(0, min(count - 1, requested - 1))
                while position >= 0 and slots[position] is not None:
                    position -= 1
            if position >= 0 and slots[position] is None:
                slots[position] = active_by_key[key]
                assigned_keys.add(key)
        remaining = [field for field in ordered if str(field["field_key"]) not in assigned_keys]
        remaining_index = 0
        for slot_index in range(count):
            if slots[slot_index] is None:
                slots[slot_index] = remaining[remaining_index]
                remaining_index += 1
        ordered = [field for field in slots if field is not None]
        for index, field in enumerate(ordered, start=1):
            field["order"] = index

    return {
        "fields": ordered,
        "field_size": field_size,
        "step_x": step_x,
        "step_y": step_y,
        "nx": nx,
        "ny": ny,
        "total_grid_fields": nx * ny,
        "auto_pruned_count": len(auto_pruned_keys),
        "required_width": required_width,
        "required_height": required_height,
        "coverage_width": grid_width,
        "coverage_height": grid_height,
        "margin_x": (grid_width - target_width) / 2.0,
        "margin_y": (grid_height - target_height) / 2.0,
        "coverage_bounds": (-grid_width / 2.0, -grid_height / 2.0, grid_width / 2.0, grid_height / 2.0),
    }


def add_explicit_ebeam_fields_to_library(
    library: gdstk.Library,
    fields: list[dict[str, Any]],
) -> int:
    """Add exact browser-resolved Ebeam write fields to the single TOP cell."""
    if not library.cells:
        top = library.new_cell("TOP")
    else:
        top = library.cells[0]
    count = 0
    for index, field in enumerate(fields, start=1):
        if not isinstance(field, dict):
            raise ValueError(f"Invalid Ebeam field entry at index {index}.")
        raw_bounds = field.get("bounds")
        if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 4:
            try:
                x0, y0, x1, y1 = (float(value) for value in raw_bounds)
            except (TypeError, ValueError):
                raise ValueError(f"Ebeam field {index} has invalid bounds.")
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
        else:
            center = field.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError(f"Ebeam field {index} is missing its [x, y] center.")
            try:
                cx = float(center[0])
                cy = float(center[1])
            except (TypeError, ValueError):
                raise ValueError(f"Ebeam field {index} has invalid center coordinates.")
            raw_size = field.get("field_size", 520.0)
            if raw_size is None:
                raw_size = 550.0
            try:
                size = float(raw_size)
            except (TypeError, ValueError):
                size = 550.0
            half = size / 2.0
            x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Ebeam field {index} has invalid geometry.")
        top.add(
            gdstk.rectangle(
                (x0, y0),
                (x1, y1),
                layer=EBEAM_LAYER,
                datatype=DEFAULT_DATATYPE,
            )
        )
        count += 1
    return count
