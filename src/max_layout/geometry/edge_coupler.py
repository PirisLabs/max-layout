"""Shared geometry and migration for the two-layer edge couplers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import (
    DEFAULT_DATATYPE,
    EDGE_COUPLER_COMPONENT_KINDS,
    EDGE_COUPLER_FEED_KINDS,
    LINEAR_EDGE_COUPLER_FEED_KIND,
    LOCAL_MARKER_LAYER,
    MODULATOR_COMPONENT_KINDS,
    WG_TIP_LAYER,
    component_edge_coupler_slot_count,
    edge_coupler_local_marker_layer,
    edge_coupler_slab_layer,
)


EDGE_COUPLER_GEOMETRY_FRONT_PARAMETERS = (
    "top_taper_length",
    "top_start_offset",
    "narrow_slab_straight_length",
    "wide_slab_straight_length",
    "slab_start_width",
    "slab_end_width",
    "top_tip_width",
    "top_end_width",
)

EDGE_COUPLER_LOCAL_MARK_FRONT_PARAMETERS = (
    "local_alignment_mark_size",
    "local_alignment_mark_vertical_offset",
)

EDGE_COUPLER_COMMON_FRONT_PARAMETERS = (
    EDGE_COUPLER_GEOMETRY_FRONT_PARAMETERS
    + EDGE_COUPLER_LOCAL_MARK_FRONT_PARAMETERS
)

EDGE_COUPLER_FRONT_PARAMETERS = (
    EDGE_COUPLER_GEOMETRY_FRONT_PARAMETERS
    + ("slab_exponential_factor",)
    + EDGE_COUPLER_LOCAL_MARK_FRONT_PARAMETERS
)

EDGE_COUPLER_FEED_FRONT_PARAMETERS = (
    ("Feed_length",) + EDGE_COUPLER_FRONT_PARAMETERS
)

EDGE_COUPLER_LINEAR_FEED_FRONT_PARAMETERS = (
    ("Feed_length",) + EDGE_COUPLER_COMMON_FRONT_PARAMETERS
)


def is_linear_edge_coupler_kind(kind: str) -> bool:
    return str(kind) in {"Linear edge coupler", LINEAR_EDGE_COUPLER_FEED_KIND}


def edge_coupler_front_parameters(kind: str) -> tuple[str, ...]:
    """Return the intuitive controls shown first for each library variant."""
    if str(kind) == LINEAR_EDGE_COUPLER_FEED_KIND:
        return EDGE_COUPLER_LINEAR_FEED_FRONT_PARAMETERS
    if str(kind) in EDGE_COUPLER_FEED_KINDS:
        return EDGE_COUPLER_FEED_FRONT_PARAMETERS
    if is_linear_edge_coupler_kind(kind):
        return EDGE_COUPLER_COMMON_FRONT_PARAMETERS
    return EDGE_COUPLER_FRONT_PARAMETERS


def resolved_edge_coupler_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return canonical values while accepting all former parameter names."""
    return {
        "Feed_length": float(parameters.get("Feed_length", 2500.0)),
        "top_taper_length": float(
            parameters.get(
                "top_taper_length",
                parameters.get("length", parameters.get("taper_length", 10.0)),
            )
        ),
        "top_start_offset": float(
            parameters.get(
                "top_start_offset",
                parameters.get(
                    "top_straight_length",
                    parameters.get("wg_straight_length", 0.0),
                ),
            )
        ),
        "narrow_slab_straight_length": float(
            parameters.get("narrow_slab_straight_length", 100.0)
        ),
        "wide_slab_straight_length": float(
            parameters.get(
                "wide_slab_straight_length",
                parameters.get("slab_straight_length", 10.0),
            )
        ),
        "slab_start_width": float(
            parameters.get(
                "slab_start_width", parameters.get("slab_initial_width", 0.5)
            )
        ),
        "slab_end_width": float(
            parameters.get(
                "slab_end_width", parameters.get("slab_final_width", 5.0)
            )
        ),
        "top_tip_width": float(
            parameters.get(
                "top_tip_width",
                parameters.get("top_initial_width", parameters.get("tip_width", 0.2)),
            )
        ),
        "top_end_width": float(
            parameters.get(
                "top_end_width",
                parameters.get("top_final_width", parameters.get("wg_width", 1.2)),
            )
        ),
        "slab_exponential_factor": float(
            parameters.get(
                "slab_exponential_factor",
                parameters.get("exponential_factor", 1.3),
            )
        ),
        "local_alignment_mark_size": float(
            parameters.get("local_alignment_mark_size", 50.0)
        ),
        "local_alignment_mark_vertical_offset": float(
            parameters.get("local_alignment_mark_vertical_offset", 100.0)
        ),
        "points": max(2, int(parameters.get("points", 1001))),
        "layer": int(parameters.get("layer", WG_TIP_LAYER)),
        "datatype": int(parameters.get("datatype", DEFAULT_DATATYPE)),
        "slab_layer": int(parameters.get("slab_layer", edge_coupler_slab_layer(1))),
        "slab_datatype": int(parameters.get("slab_datatype", DEFAULT_DATATYPE)),
        "local_alignment_mark_name": str(
            parameters.get("local_alignment_mark_name", "L1")
        ).strip()
        or "L1",
        "local_alignment_mark_layer": int(
            parameters.get("local_alignment_mark_layer", LOCAL_MARKER_LAYER)
        ),
        "local_alignment_mark_datatype": int(
            parameters.get("local_alignment_mark_datatype", DEFAULT_DATATYPE)
        ),
}


def assign_edge_coupler_layers(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign one numbered SLAB layer per end and one shared LOCAL layer."""
    edge_couplers = sorted(
        (
            component
            for component in components
            if component_edge_coupler_slot_count(component) > 0
        ),
        key=lambda component: int(component.get("uid", 0)),
    )
    next_number = 1
    for component in edge_couplers:
        kind = str(component.get("kind", ""))
        slot_count = component_edge_coupler_slot_count(component)
        numbers = list(range(next_number, next_number + slot_count))
        component["edge_coupler_number"] = numbers[0]
        component["edge_coupler_numbers"] = numbers
        parameters = component.setdefault("params", {})
        if kind in MODULATOR_COMPONENT_KINDS:
            parameters["left_edge_coupler_slab_layer"] = edge_coupler_slab_layer(
                numbers[0]
            )
            parameters["left_edge_coupler_local_marker_name"] = f"L{numbers[0]}"
            parameters["left_edge_coupler_local_marker_layer"] = (
                edge_coupler_local_marker_layer(numbers[0])
            )
            parameters["right_edge_coupler_slab_layer"] = edge_coupler_slab_layer(
                numbers[1]
            )
            parameters["right_edge_coupler_local_marker_name"] = f"L{numbers[1]}"
            parameters["right_edge_coupler_local_marker_layer"] = (
                edge_coupler_local_marker_layer(numbers[1])
            )
        else:
            parameters["layer"] = WG_TIP_LAYER
            parameters["datatype"] = DEFAULT_DATATYPE
            parameters["slab_layer"] = edge_coupler_slab_layer(numbers[0])
            parameters["slab_datatype"] = DEFAULT_DATATYPE
            parameters["local_alignment_mark_name"] = f"L{numbers[0]}"
            parameters["local_alignment_mark_layer"] = (
                edge_coupler_local_marker_layer(numbers[0])
            )
            parameters["local_alignment_mark_datatype"] = DEFAULT_DATATYPE
        if slot_count == 2 and kind in EDGE_COUPLER_FEED_KINDS:
            parameters["right_slab_layer"] = edge_coupler_slab_layer(numbers[1])
            parameters["right_slab_datatype"] = DEFAULT_DATATYPE
            parameters["right_local_alignment_mark_name"] = f"L{numbers[1]}"
            parameters["right_local_alignment_mark_layer"] = (
                edge_coupler_local_marker_layer(numbers[1])
            )
            parameters["right_local_alignment_mark_datatype"] = DEFAULT_DATATYPE
        next_number += slot_count
    return edge_couplers


def migrate_edge_coupler_parameters(component: dict[str, Any]) -> bool:
    """Upgrade legacy edge-coupler kinds and keys to the intuitive model."""
    kind = str(component.get("kind", ""))
    if kind not in EDGE_COUPLER_COMPONENT_KINDS:
        return False
    changed = False
    if kind == "Edge coupler":
        kind = "Exponential edge coupler"
        component["kind"] = kind
        changed = True

    parameters = component.setdefault("params", {})
    number = max(1, int(component.get("edge_coupler_number", 1)))
    component["edge_coupler_number"] = number
    numbers = [number, number + 1] if kind in EDGE_COUPLER_FEED_KINDS else [number]
    component["edge_coupler_numbers"] = numbers
    values = resolved_edge_coupler_parameters(parameters)
    canonical = {
        key: values[key] for key in edge_coupler_front_parameters(kind)
    }
    canonical.update(
        {
            "points": values["points"],
            "layer": WG_TIP_LAYER,
            "datatype": values["datatype"],
            "slab_layer": edge_coupler_slab_layer(number),
            "slab_datatype": DEFAULT_DATATYPE,
            "local_alignment_mark_name": f"L{number}",
            "local_alignment_mark_layer": edge_coupler_local_marker_layer(number),
            "local_alignment_mark_datatype": DEFAULT_DATATYPE,
        }
    )
    if kind in EDGE_COUPLER_FEED_KINDS:
        canonical.update(
            {
                "right_slab_layer": edge_coupler_slab_layer(numbers[1]),
                "right_slab_datatype": DEFAULT_DATATYPE,
                "right_local_alignment_mark_name": f"L{numbers[1]}",
                "right_local_alignment_mark_layer": edge_coupler_local_marker_layer(
                    numbers[1]
                ),
                "right_local_alignment_mark_datatype": DEFAULT_DATATYPE,
            }
        )
    legacy_keys = {
        "length",
        "taper_length",
        "slab_initial_width",
        "slab_final_width",
        "top_initial_width",
        "top_final_width",
        "tip_width",
        "wg_width",
        "exponential_factor",
        "top_straight_length",
        "wg_straight_length",
        "slab_straight_length",
    }
    if is_linear_edge_coupler_kind(kind):
        legacy_keys.add("slab_exponential_factor")
    changed = changed or any(parameters.get(key) != value for key, value in canonical.items())
    changed = changed or any(key in parameters for key in legacy_keys)
    for key in legacy_keys:
        parameters.pop(key, None)
    parameters.update(canonical)
    return changed


def edge_coupler_slab_profile(
    parameters: dict[str, Any],
    kind: str = "Exponential edge coupler",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SLAB taper across L1 + L2 for the selected profile."""
    values = resolved_edge_coupler_parameters(parameters)
    top_length = values["top_taper_length"]
    offset = values["top_start_offset"]
    total_length = offset + top_length
    start_width = values["slab_start_width"]
    end_width = values["slab_end_width"]
    factor = values["slab_exponential_factor"]
    if top_length <= 0.0 or offset < 0.0:
        raise ValueError(
            "Top WG taper length must be positive and its start offset cannot be negative."
        )
    if start_width <= 0.0 or end_width <= 0.0:
        raise ValueError("SLAB start and end widths must be positive.")
    if factor <= 0.0:
        raise ValueError("SLAB exponential factor must be positive.")
    u = np.linspace(0.0, 1.0, values["points"])
    if is_linear_edge_coupler_kind(kind):
        widths = start_width + (end_width - start_width) * u
    else:
        widths = start_width * (end_width / start_width) ** (u**factor)
    widths[0] = start_width
    widths[-1] = end_width
    return total_length * u, widths


def edge_coupler_feed_landmarks(parameters: dict[str, Any]) -> dict[str, float]:
    """Return the shared connection and end coordinates for a coupler pair."""
    values = resolved_edge_coupler_parameters(parameters)
    feed_length = values["Feed_length"]
    if feed_length <= 0.0:
        raise ValueError("Feed_length must be positive.")
    slab_taper_start = values["narrow_slab_straight_length"]
    top_tip_x = slab_taper_start + values["top_start_offset"]
    connection_x = top_tip_x + values["top_taper_length"]
    total_length = 2.0 * connection_x + feed_length
    return {
        "top_tip_x": top_tip_x,
        "left_connection_x": connection_x,
        "right_connection_x": connection_x + feed_length,
        "right_origin_x": total_length,
        "right_tip_x": total_length - top_tip_x,
        "total_length": total_length,
    }
