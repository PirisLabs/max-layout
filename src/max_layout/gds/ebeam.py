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
        first_explicit_key = requested_first_key({
            str(field["field_key"]) for field in fields
        })
        if fields and first_explicit_key is not None:
            by_explicit_key = {
                str(field["field_key"]): field for field in fields
            }
            route = [by_explicit_key[first_explicit_key]]
            remaining = {
                key for key in by_explicit_key if key != first_explicit_key
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

    # Renumber only active fields after pruning and manual movement.  Fields
    # that are orthogonal neighbors in the write-field grid are kept adjacent
    # whenever an adjacent path exists.  A shortest-distance jump is used only
    # between disconnected occupied regions or when a pruned shape has no
    # Hamiltonian adjacent path.
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

        def neighbor_keys(key: tuple[int, int], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
            column, row = key
            candidates = [
                (column - 1, row),
                (column + 1, row),
                (column, row - 1),
                (column, row + 1),
            ]
            return [candidate for candidate in candidates if candidate in allowed]

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
            if len(component) <= 1:
                return [start_key]

            path = [start_key]
            unused = set(component)
            unused.remove(start_key)
            search_steps = 0
            max_search_steps = 300000 if len(component) <= 90 else 30000

            def remaining_is_reachable(current_key: tuple[int, int]) -> bool:
                if not unused:
                    return True
                entrances = neighbor_keys(current_key, unused)
                if not entrances:
                    return False
                reached = {entrances[0]}
                stack = [entrances[0]]
                while stack:
                    key = stack.pop()
                    for neighbor in neighbor_keys(key, unused):
                        if neighbor not in reached:
                            reached.add(neighbor)
                            stack.append(neighbor)
                return reached == unused

            def dfs(current_key: tuple[int, int]) -> bool:
                nonlocal search_steps
                search_steps += 1
                if search_steps > max_search_steps:
                    return False
                if not unused:
                    return True
                candidates = neighbor_keys(current_key, unused)
                candidates.sort(
                    key=lambda key: (
                        len(neighbor_keys(key, unused - {key})),
                        field_distance_sq(by_key[current_key], by_key[key]),
                        abs(int(by_key[key]["row"]) - int(by_key[current_key]["row"]))
                        if primary_axis == "x"
                        else abs(int(by_key[key]["column"]) - int(by_key[current_key]["column"])),
                        int(by_key[key]["row"]),
                        int(by_key[key]["column"]),
                    )
                )
                for candidate in candidates:
                    unused.remove(candidate)
                    path.append(candidate)
                    if remaining_is_reachable(candidate) and dfs(candidate):
                        return True
                    path.pop()
                    unused.add(candidate)
                return False

            if dfs(start_key):
                return path

            # Fallback for a connected occupied shape that has no all-adjacent
            # Hamiltonian path: stay adjacent while possible, then make only
            # the minimum-distance jump needed to continue.
            route = [start_key]
            remaining = set(component)
            remaining.remove(start_key)
            while remaining:
                current_key = route[-1]
                adjacent = neighbor_keys(current_key, remaining)
                pool = adjacent if adjacent else list(remaining)
                next_key = min(
                    pool,
                    key=lambda key: (
                        field_distance_sq(by_key[current_key], by_key[key]),
                        len(neighbor_keys(key, remaining - {key})),
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
        route_keys = component_path(ordered_components[0], first_key)

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
