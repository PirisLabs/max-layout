"""Shared geometry resolution for symmetric MZI electro-optic modulators."""

from __future__ import annotations

from typing import Any
import math

from ..constants import WG_TIP_LAYER, edge_coupler_local_marker_layer, edge_coupler_slab_layer
from .landmarks import cpw_bend_landmarks, segmented_electrode_landmarks
from .shapes import mmi_total_length


CONTINUOUS_MODULATOR_KIND = "MZI + CPW modulator"
SEGMENTED_MODULATOR_KIND = "MZI + segmented-electrode modulator"
MODULATOR_KINDS = {CONTINUOUS_MODULATOR_KIND, SEGMENTED_MODULATOR_KIND}


def _positive(parameters: dict[str, Any], key: str) -> float:
    value = float(parameters[key])
    if value <= 0.0:
        raise ValueError(f"Modulator {key} must be positive.")
    return value


def _point_along(
    origin: tuple[float, float], angle_deg: float, distance: float
) -> tuple[float, float]:
    angle = math.radians(float(angle_deg))
    return (
        float(origin[0]) + float(distance) * math.cos(angle),
        float(origin[1]) + float(distance) * math.sin(angle),
    )


def resolve_modulator_layout(kind: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Resolve every coupled child parameter and exact local connection point.

    The component origin is the left endpoint of the center RF electrode.  Both
    outer RF routes are exact 180-degree rotations of one another around the
    center electrode.  The optical straight-arm region is centered on that RF
    electrode even when their independently editable lengths differ.
    """
    if kind not in MODULATOR_KINDS:
        raise ValueError(f"Unsupported modulator kind: {kind}")

    initial_gap = _positive(parameters, "initial_gap")
    final_gap = _positive(parameters, "final_gap")
    signal_width = _positive(parameters, "signal_width")
    ground_width = _positive(parameters, "ground_width")
    taper_length = _positive(parameters, "taper_length")
    mzi_length = _positive(parameters, "mzi_length")
    electrode_bend_overlap = float(parameters.get("electrode_bend_overlap", 2.0))
    if electrode_bend_overlap < 0.0:
        raise ValueError("Modulator electrode_bend_overlap cannot be negative.")
    s_bend_length = _positive(parameters, "s_bend_length")
    bend_radius = _positive(parameters, "bend_radius")
    bend_angle = float(parameters.get("bend_angle_deg", 90.0))
    if bend_angle <= 0.0 or bend_angle > 180.0:
        raise ValueError("Modulator bend_angle_deg must be positive and at most 180 degrees.")
    optical_wg_width = _positive(parameters, "wg_width")
    if initial_gap + 1e-12 < optical_wg_width:
        raise ValueError(
            "Modulator initial_gap must be at least as wide as the optical waveguide."
        )
    add_edge_coupler_tapers = bool(
        parameters.get("add_edge_coupler_tapers", False)
    )
    edge_coupler_profile = str(
        parameters.get("edge_coupler_profile", "exponential")
    ).strip().lower()
    if edge_coupler_profile not in {"exponential", "linear"}:
        raise ValueError("Modulator edge_coupler_profile must be exponential or linear.")
    edge_coupler_tip_width = float(parameters.get("edge_coupler_tip_width", 0.3))
    if edge_coupler_tip_width <= 0.0:
        raise ValueError("Modulator edge_coupler_tip_width must be positive.")
    left_edge_coupler_taper_length = float(
        parameters.get("left_edge_coupler_taper_length", 365.0)
    )
    right_edge_coupler_taper_length = float(
        parameters.get("right_edge_coupler_taper_length", 365.0)
    )
    edge_coupler_top_start_offset = float(
        parameters.get("edge_coupler_top_start_offset", 10.0)
    )
    edge_coupler_narrow_slab_straight_length = float(
        parameters.get("edge_coupler_narrow_slab_straight_length", 100.0)
    )
    edge_coupler_wide_slab_straight_length = float(
        parameters.get("edge_coupler_wide_slab_straight_length", 10.0)
    )
    edge_coupler_slab_start_width = float(
        parameters.get("edge_coupler_slab_start_width", 0.5)
    )
    edge_coupler_slab_end_width = float(
        parameters.get("edge_coupler_slab_end_width", 5.0)
    )
    edge_coupler_exponential_factor = float(
        parameters.get("edge_coupler_exponential_factor", 1.3)
    )
    edge_coupler_local_marker_size = float(
        parameters.get("edge_coupler_local_marker_size", 50.0)
    )
    edge_coupler_local_marker_vertical_offset = float(
        parameters.get("edge_coupler_local_marker_vertical_offset", 100.0)
    )
    if add_edge_coupler_tapers and (
        left_edge_coupler_taper_length <= 0.0
        or right_edge_coupler_taper_length <= 0.0
    ):
        raise ValueError(
            "Enabled modulator edge-coupler taper lengths must be positive."
        )
    if (
        edge_coupler_top_start_offset < 0.0
        or edge_coupler_narrow_slab_straight_length < 0.0
        or edge_coupler_wide_slab_straight_length < 0.0
        or edge_coupler_slab_start_width <= 0.0
        or edge_coupler_slab_end_width <= 0.0
        or edge_coupler_exponential_factor <= 0.0
        or edge_coupler_local_marker_size <= 0.0
        or edge_coupler_local_marker_vertical_offset < 0.0
    ):
        raise ValueError(
            "Modulator edge-coupler widths, factor, and marker size must be positive; "
            "offsets and straight lengths cannot be negative."
        )
    if add_edge_coupler_tapers and bool(
        parameters.get("add_grating_couplers", False)
    ):
        raise ValueError(
            "Modulator edge-coupler tapers and grating couplers cannot be enabled together."
        )

    rf_layer = int(parameters.get("rf_layer", 4))
    rf_datatype = int(parameters.get("rf_datatype", 0))
    rf_bends_layer = int(parameters.get("rf_bends_layer", 9))
    rf_bends_datatype = int(parameters.get("rf_bends_datatype", 0))
    photonic_layer = int(parameters.get("layer", 1))
    photonic_datatype = int(parameters.get("datatype", 0))
    arm_separation = signal_width + initial_gap

    if kind == CONTINUOUS_MODULATOR_KIND:
        electrode_length = _positive(parameters, "electrode_length")
        electrode_kind = "CPW"
        electrode_parameters = {
            "length": electrode_length,
            "signal_width": signal_width,
            "gap": initial_gap,
            "ground_width": ground_width,
            "layer": rf_layer,
            "datatype": rf_datatype,
        }
    else:
        electrode_kind = "Segmented electrode"
        electrode_parameters = {
            "signal_width": signal_width,
            # One shared interaction gap drives both the finger-tip residual
            # gap and the plain CPW endpoint attached to each bend.
            "gap": initial_gap,
            "end_gap": initial_gap,
            "ground_width": ground_width,
            "transition_length": float(parameters.get("transition_length", 0.0)),
            "end_flat_length": float(parameters.get("end_flat_length", 100.0)),
            "inner_flat_length": float(parameters.get("inner_flat_length", 0.0)),
            "t_top_width": float(parameters.get("t_top_width", 2.0)),
            "t_top_length": float(parameters.get("t_top_length", 45.0)),
            "t_neck_width": float(parameters.get("t_neck_width", 4.0)),
            "t_neck_length": float(parameters.get("t_neck_length", 18.0)),
            "segment_spacing": float(parameters.get("segment_spacing", 3.0)),
            "segment_count": int(parameters.get("segment_count", 235)),
            "include_oxide_masks": bool(parameters.get("include_oxide_masks", False)),
            "layer": rf_layer,
            "datatype": rf_datatype,
            "oxide_layer": int(parameters.get("oxide_layer", rf_layer)),
            "oxide_datatype": int(parameters.get("oxide_datatype", rf_datatype)),
        }
        electrode_length = float(
            segmented_electrode_landmarks(electrode_parameters)["total_length"]
        )
    if 2.0 * electrode_bend_overlap >= electrode_length:
        raise ValueError(
            "Modulator electrode_bend_overlap must be less than half the center electrode length."
        )

    bend_parameters = {
        "R_eff": bend_radius,
        "bend_angle_deg": bend_angle,
        "signal_width": signal_width,
        "gap": initial_gap,
        "ground_width": ground_width,
        "points": max(2, int(parameters.get("bend_points", 261))),
        "layer": rf_bends_layer,
        "datatype": rf_bends_datatype,
    }
    bend = cpw_bend_landmarks(bend_parameters, mirrored=False)
    bend_end = tuple(float(value) for value in bend["signal_end"])
    signed_bend_angle = float(bend["angle_deg"])

    # Start each bend inside the horizontal center electrode so the two metal
    # sections have a deliberate longitudinal overlap instead of merely
    # touching at a zero-area boundary.  The starts remain point-symmetric.
    right_bend_start = (electrode_length - electrode_bend_overlap, 0.0)
    right_bend_end = (
        right_bend_start[0] + bend_end[0],
        bend_end[1],
    )
    # Exact 180-degree rotation of the right route around electrode center.
    left_bend_start = (electrode_bend_overlap, 0.0)
    left_bend_end = (
        left_bend_start[0] - bend_end[0],
        -bend_end[1],
    )
    right_taper_angle = signed_bend_angle % 360.0
    left_taper_angle = (180.0 + signed_bend_angle) % 360.0
    right_rf_external = _point_along(right_bend_end, right_taper_angle, taper_length)
    left_rf_external = _point_along(left_bend_end, left_taper_angle, taper_length)

    taper_parameters = {
        "length": taper_length,
        "signal_width": signal_width,
        "initial_gap": initial_gap,
        "final_gap": final_gap,
        "ground_width": ground_width,
        "profile": str(parameters.get("profile", "klopfenstein")),
        "target_s11_db": float(parameters.get("target_s11_db", 20.0)),
        "exponential_factor": float(parameters.get("exponential_factor", 1.0)),
        "points": max(2, int(math.ceil(taper_length / 0.5)) + 1),
        "layer": rf_bends_layer,
        "datatype": rf_bends_datatype,
    }

    mzi_parameters = {
        "mmi_width": float(parameters["mmi_width"]),
        "mmi_length": float(parameters["mmi_length"]),
        "wg_width": optical_wg_width,
        "taper_width": float(parameters["taper_width"]),
        "input_taper_length": float(parameters["input_taper_length"]),
        "output_taper_length": float(parameters["output_taper_length"]),
        "input_length": float(parameters["input_length"]),
        "output_length": float(parameters["output_length"]),
        "port_sep": float(parameters["port_sep"]),
        "arm_separation": arm_separation,
        "s_bend_length": s_bend_length,
        "arm_length": mzi_length,
        "taper_power": float(parameters.get("taper_power", 1.0)),
        "taper_points": max(2, int(parameters.get("taper_points", 200))),
        "add_grating_couplers": bool(parameters.get("add_grating_couplers", False)),
        "gc_input_route": str(parameters.get("gc_input_route", "straight")),
        "gc_output_route": str(parameters.get("gc_output_route", "straight")),
        "gc_euler_radius": float(parameters.get("gc_euler_radius", 200.0)),
        "gc_euler_fraction": float(parameters.get("gc_euler_fraction", 1.0)),
        "gc_pitch": float(parameters.get("gc_pitch", 0.75)),
        "gc_fill_factor": float(parameters.get("gc_fill_factor", 0.57)),
        "gc_N": int(parameters.get("gc_N", 30)),
        "gc_alpha_t": float(parameters.get("gc_alpha_t", 25.0)),
        "gc_taper_L": float(parameters.get("gc_taper_L", 22.0)),
        "gc_wg_length": float(parameters.get("gc_wg_length", 200.0)),
        "gc_tolerance": float(parameters.get("gc_tolerance", 0.0005)),
        "gc_euler_tolerance": float(parameters.get("gc_euler_tolerance", 0.001)),
        "layer": photonic_layer,
        "datatype": photonic_datatype,
        "gc_layer": int(parameters.get("gc_layer", 2)),
        "gc_datatype": int(parameters.get("gc_datatype", 0)),
    }
    mmi_length = mmi_total_length(mzi_parameters)
    optical_active_start = (electrode_length - mzi_length) / 2.0
    mzi_start = optical_active_start - mmi_length - s_bend_length
    optical_total_length = 2.0 * mmi_length + 2.0 * s_bend_length + mzi_length
    optical_mzi_left = (mzi_start, 0.0)
    optical_mzi_right = (mzi_start + optical_total_length, 0.0)
    edge_taper_common = {
        "width_start": optical_wg_width,
        "width_end": edge_coupler_tip_width,
        "layer": WG_TIP_LAYER,
        "datatype": photonic_datatype,
    }
    left_edge_taper_parameters = dict(
        edge_taper_common, length=left_edge_coupler_taper_length
    )
    right_edge_taper_parameters = dict(
        edge_taper_common, length=right_edge_coupler_taper_length
    )
    edge_coupler_common = {
        "top_start_offset": edge_coupler_top_start_offset,
        "narrow_slab_straight_length": edge_coupler_narrow_slab_straight_length,
        "wide_slab_straight_length": edge_coupler_wide_slab_straight_length,
        "slab_start_width": edge_coupler_slab_start_width,
        "slab_end_width": edge_coupler_slab_end_width,
        "top_tip_width": edge_coupler_tip_width,
        "top_end_width": optical_wg_width,
        "slab_exponential_factor": edge_coupler_exponential_factor,
        "local_alignment_mark_size": edge_coupler_local_marker_size,
        "local_alignment_mark_vertical_offset": edge_coupler_local_marker_vertical_offset,
        "points": max(2, int(parameters.get("edge_coupler_points", 1001))),
        "layer": WG_TIP_LAYER,
        "datatype": photonic_datatype,
    }
    left_edge_coupler_parameters = dict(
        edge_coupler_common,
        top_taper_length=left_edge_coupler_taper_length,
        slab_layer=int(
            parameters.get("left_edge_coupler_slab_layer", edge_coupler_slab_layer(1))
        ),
        slab_datatype=0,
        local_alignment_mark_name=str(
            parameters.get("left_edge_coupler_local_marker_name", "L1")
        ),
        local_alignment_mark_layer=int(
            parameters.get(
                "left_edge_coupler_local_marker_layer",
                edge_coupler_local_marker_layer(1),
            )
        ),
        local_alignment_mark_datatype=0,
    )
    right_edge_coupler_parameters = dict(
        edge_coupler_common,
        top_taper_length=right_edge_coupler_taper_length,
        slab_layer=int(
            parameters.get("right_edge_coupler_slab_layer", edge_coupler_slab_layer(2))
        ),
        slab_datatype=0,
        local_alignment_mark_name=str(
            parameters.get("right_edge_coupler_local_marker_name", "L2")
        ),
        local_alignment_mark_layer=int(
            parameters.get(
                "right_edge_coupler_local_marker_layer",
                edge_coupler_local_marker_layer(2),
            )
        ),
        local_alignment_mark_datatype=0,
    )
    left_connection_length = (
        edge_coupler_narrow_slab_straight_length
        + edge_coupler_top_start_offset
        + left_edge_coupler_taper_length
    )
    right_connection_length = (
        edge_coupler_narrow_slab_straight_length
        + edge_coupler_top_start_offset
        + right_edge_coupler_taper_length
    )
    left_edge_coupler_start = (
        optical_mzi_left[0] - left_connection_length,
        0.0,
    )
    right_edge_coupler_start = (
        optical_mzi_right[0] + right_connection_length,
        0.0,
    )
    optical_left = (
        optical_mzi_left[0] - left_edge_coupler_taper_length,
        0.0,
    ) if add_edge_coupler_tapers else optical_mzi_left
    optical_right = (
        optical_mzi_right[0] + right_edge_coupler_taper_length,
        0.0,
    ) if add_edge_coupler_tapers else optical_mzi_right

    return {
        "arm_separation": arm_separation,
        "electrode_kind": electrode_kind,
        "electrode_parameters": electrode_parameters,
        "electrode_length": electrode_length,
        "electrode_bend_overlap": electrode_bend_overlap,
        "electrode_center": (electrode_length / 2.0, 0.0),
        "bend_parameters": bend_parameters,
        "left_bend_start": left_bend_start,
        "right_bend_start": right_bend_start,
        "left_bend_end": left_bend_end,
        "right_bend_end": right_bend_end,
        "left_taper_angle": left_taper_angle,
        "right_taper_angle": right_taper_angle,
        "left_rf_external": left_rf_external,
        "right_rf_external": right_rf_external,
        "taper_parameters": taper_parameters,
        "mzi_parameters": mzi_parameters,
        "mzi_start": (mzi_start, 0.0),
        "add_edge_coupler_tapers": add_edge_coupler_tapers,
        "edge_coupler_tip_width": edge_coupler_tip_width,
        "edge_coupler_profile": edge_coupler_profile,
        "edge_coupler_kind": (
            "Linear edge coupler"
            if edge_coupler_profile == "linear"
            else "Exponential edge coupler"
        ),
        "left_edge_taper_parameters": left_edge_taper_parameters,
        "right_edge_taper_parameters": right_edge_taper_parameters,
        "left_edge_coupler_parameters": left_edge_coupler_parameters,
        "right_edge_coupler_parameters": right_edge_coupler_parameters,
        "left_edge_coupler_start": left_edge_coupler_start,
        "right_edge_coupler_start": right_edge_coupler_start,
        "edge_coupler_local_marker_vertical_offset": edge_coupler_local_marker_vertical_offset,
        "optical_mzi_left": optical_mzi_left,
        "optical_mzi_right": optical_mzi_right,
        "optical_active_start": (optical_active_start, 0.0),
        "optical_active_end": (optical_active_start + mzi_length, 0.0),
        "optical_left": optical_left,
        "optical_right": optical_right,
    }
