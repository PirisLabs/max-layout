"""Component parameter resizing rules."""

from __future__ import annotations

from typing import Any
import math

from .utils import safe_json_copy


def resize_component_parameters(
    kind: str,
    parameters: dict[str, Any],
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    """Scale the same editable dimensions used by the browser corner handles."""
    result = safe_json_copy(parameters)
    sx = max(0.03, float(scale_x))
    sy = max(0.03, float(scale_y))
    su = math.sqrt(max(0.001, sx * sy))

    def apply(names: tuple[str, ...], factor: float) -> None:
        for name in names:
            if name in result and isinstance(result[name], (int, float)):
                value = float(result[name]) * factor
                result[name] = max(0.001, value)

    x_names: dict[str, tuple[str, ...]] = {
        "Straight": ("length",),
        "Taper": ("length",),
        "S-bend": ("length",),
        "Grating coupler": ("wg_length", "taper_L"),
        "1x2 MMI": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length"),
        "Cascaded MMI": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "s_bend_length", "output_s_bend_length"),
        "MMI + Reference": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "gc_s_bend_length"),
        "MZI": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "s_bend_length", "arm_length"),
        "MZI vertical GC": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "s_bend_length", "arm_length", "gc_prebend_straight", "gc_inward_run"),
        "Vertical-GC MZI test block": ("mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "s_bend_length", "arm_length", "gc_prebend_straight", "gc_inward_run"),
        "Long MZI test block": ("mzi_total_length", "gc_straight_length", "mmi_length", "input_taper_length", "output_taper_length", "input_length", "output_length", "s_bend_length", "gc_taper_L", "gc_wg_length"),
        "Chip marker block": ("chip_width", "corner_square_size", "edge_clearance", "vernier_pitch", "vernier_pitch_delta", "vernier_finger_width"),
        "MZI + CPW module": ("rf_input_taper_length", "s_bend_length", "arm_length", "rf_output_taper_length"),
        "CPW": ("length",),
        "CPW open": ("length", "signal_recess"),
        "CPW short": ("length", "bridge_length"),
        "Tapered CPW": ("length",),
        "Symmetric CPW taper": ("end_straight_length", "taper_length", "middle_straight_length"),
        "CPW bend": ("R_eff",),
        "Segmented electrode": ("transition_length", "t_top_length", "t_neck_length", "segment_spacing"),
        "Chip outline": ("width",),
        "E-beam multipass": ("target_width",),
        "Vernier mark": ("pitch", "pitch_delta", "finger_width"),
        "Racetrack": ("coupling_length",),
        "Ring + two feedlines": ("feedline_length", "s_bend_length", "taper_L", "gc_wg_length"),
        "Double-ring test block": ("column_spacing", "grating_end_to_end_distance", "s_bend_length", "taper_L", "gc_wg_length"),
        "Grating test block": ("device_x_spacing", "grating_end_to_end_distance", "s_bend_length", "taper_L", "gc_wg_length"),
        "Grating angle-taper test block": ("device_x_spacing", "grating_end_to_end_distance", "s_bend_length", "taper_length_start", "taper_length_stop", "taper_length_step", "gc_wg_length"),
        "MMI + Reference test block": ("device_x_spacing", "mmi_length_start", "mmi_length_stop", "mmi_length_step", "input_taper_length", "output_taper_length", "input_length", "output_length", "taper_L", "gc_wg_length"),
        "MMI split-combine test block": ("device_x_spacing", "taper_length_start", "taper_length_stop", "taper_length_step", "input_length", "output_length", "interconnect_length", "output_s_bend_length", "gc_wg_length"),
        "Edge coupler": ("taper_length", "wg_straight_length"),
        "Loopback mirror": ("Lc", "s_bend_length"),
        "Feedline": ("input_straight_length", "s_bend_length", "Lc", "output_straight_length"),
        "Ring + feedline": ("input_straight_length", "s_bend_length", "Lc", "output_straight_length", "resonator_spacing"),
        "Racetrack + feedline": ("input_straight_length", "s_bend_length", "Lc", "output_straight_length", "resonator_spacing", "racetrack_coupling_length"),
    }
    y_names: dict[str, tuple[str, ...]] = {
        "Straight": ("width",),
        "Taper": ("width_start", "width_end"),
        "S-bend": ("offset",),
        "Grating coupler": ("wg_width",),
        "1x2 MMI": ("mmi_width", "wg_width", "taper_width", "port_sep"),
        "Cascaded MMI": ("mmi_width", "wg_width", "taper_width", "port_sep", "output_gc_spacing", "minimum_s_bend_radius"),
        "MMI + Reference": ("mmi_width", "wg_width", "taper_width", "port_sep", "reference_dy"),
        "MZI": ("mmi_width", "wg_width", "taper_width", "port_sep", "arm_separation"),
        "MZI vertical GC": ("mmi_width", "wg_width", "taper_width", "port_sep", "arm_separation", "gc_vertical_run", "gc_euler_radius"),
        "Vertical-GC MZI test block": ("vertical_spacing", "mmi_width", "wg_width", "taper_width", "port_sep", "arm_separation", "gc_vertical_run", "gc_euler_radius"),
        "Long MZI test block": ("mmi_width", "wg_width", "taper_width", "port_sep", "arm_separation"),
        "Chip marker block": ("chip_height", "corner_square_size", "edge_clearance", "vernier_finger_length", "vernier_row_gap", "vernier_base_thickness"),
        "MZI + CPW module": ("signal_width", "interaction_gap", "external_gap", "ground_width", "arm_separation"),
        "CPW": ("signal_width", "gap", "ground_width"),
        "CPW open": ("signal_width", "gap", "ground_width"),
        "CPW short": ("signal_width", "gap", "ground_width"),
        "Tapered CPW": ("signal_width", "initial_gap", "final_gap", "ground_width"),
        "Symmetric CPW taper": ("signal_width", "initial_gap", "middle_gap", "ground_width"),
        "CPW bend": ("signal_width", "gap", "ground_width"),
        "Segmented electrode": ("signal_width", "gap", "ground_width", "t_top_width", "t_neck_width"),
        "Chip outline": ("height",),
        "E-beam multipass": ("target_height",),
        "Vernier mark": ("finger_length", "row_gap", "base_thickness"),
        "Racetrack": ("radius",),
        "Ring + two feedlines": ("ring_radius", "coupling_gap", "grating_coupler_separation", "s_bend_offset"),
        "Double-ring test block": ("row_spacing", "ring_width", "feedline_width", "s_bend_offset"),
        "Grating test block": ("packing_pitch", "write_field_size", "endpoint_offset", "wg_width"),
        "Grating angle-taper test block": ("packing_pitch", "write_field_size", "endpoint_offset", "wg_width"),
        "MMI + Reference test block": ("device_y_spacing", "taper_width_start", "taper_width_stop", "taper_width_step", "mmi_width", "wg_width", "port_sep", "reference_dy"),
        "MMI split-combine test block": ("device_y_spacing", "taper_width_start", "taper_width_stop", "taper_width_step", "nominal_taper_width", "mmi_width", "wg_width", "port_sep", "reference_vertical_offset", "output_s_bend_offset"),
        "Text / Number": ("height",),
        "Edge coupler": ("tip_width", "wg_width"),
        "Loopback mirror": ("gap", "arc_radius"),
        "Feedline": ("offset",),
        "Ring + feedline": ("offset", "ring_radius", "coupling_gap"),
        "Racetrack + feedline": ("offset", "racetrack_radius", "coupling_gap"),
    }
    uniform_names: dict[str, tuple[str, ...]] = {
        "Euler bend": ("radius", "width"),
        "S-bend": ("width",),
        "Chip outline": ("line_width",),
        "E-beam multipass": ("field_size", "edge_clearance", "outline_width"),
        "Square mark": ("size", "line_width"),
        "Cross mark": ("size", "line_width"),
        "Pointy cross mark": ("size", "line_width", "tip_length"),
        "Cross + squares mark": ("size", "bar_width", "square_size", "square_gap"),
        "Ring": ("radius", "width"),
        "Photonic crystal": ("length", "width", "pitch_x", "pitch_y", "hole_radius_x", "hole_radius_y"),
        "Racetrack": ("width",),
        "Ring + two feedlines": ("ring_width", "feedline_width"),
        "Loopback mirror": ("width",),
        "Feedline": ("wg_width",),
        "Ring + feedline": ("wg_width", "ring_width"),
        "Racetrack + feedline": ("wg_width", "racetrack_width"),
    }
    apply(x_names.get(kind, ()), sx)
    apply(y_names.get(kind, ()), sy)
    apply(uniform_names.get(kind, ()), su)
    return result
