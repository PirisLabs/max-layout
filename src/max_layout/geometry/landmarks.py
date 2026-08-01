"""Landmark point sets for CPW, feedlines, and electrodes."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


def cpw_bend_landmarks(p: dict[str, Any], mirrored: bool = False) -> dict[str, Any]:
    """Return local geometry landmarks for a constant-width CPW bend.

    ``R_eff`` is the radius of the signal-conductor centerline.  The signal,
    upper gap, and lower gap therefore remain exactly registered at both ends.
    """
    radius = float(p["R_eff"])
    angle_deg = (-1.0 if mirrored else 1.0) * float(p["bend_angle_deg"])
    ws = float(p["signal_width"])
    gap = float(p["gap"])
    wg = float(p["ground_width"])
    if radius <= 0 or ws <= 0 or gap < 0 or wg <= 0:
        raise ValueError("CPW bend requires positive R_eff, signal_width, and ground_width; gap cannot be negative.")
    if abs(angle_deg) < 1e-12 or abs(angle_deg) > 180.0:
        raise ValueError("CPW bend_angle_deg must be nonzero and between -180 and +180 degrees.")
    inner_ground_radius = radius - ws / 2.0 - gap - wg
    if inner_ground_radius <= 0:
        raise ValueError("R_eff is too small for the selected signal width, gap, and ground width.")

    sign = 1.0 if angle_deg > 0 else -1.0
    theta = math.radians(angle_deg)
    curvature_center = np.array([0.0, sign * radius], dtype=float)
    start_angle = -sign * math.pi / 2.0
    end_angle = start_angle + theta

    def point_on_radius(radial_distance: float, polar_angle: float) -> tuple[float, float]:
        point = curvature_center + radial_distance * np.array(
            [math.cos(polar_angle), math.sin(polar_angle)], dtype=float
        )
        return (float(point[0]), float(point[1]))

    signal_start = (0.0, 0.0)
    signal_end = point_on_radius(radius, end_angle)
    tangent_end = math.radians(angle_deg)
    transverse_end = np.array([-math.sin(tangent_end), math.cos(tangent_end)], dtype=float)
    gap_center_offset = ws / 2.0 + gap / 2.0
    ground_center_offset = ws / 2.0 + gap + wg / 2.0
    signal_end_array = np.asarray(signal_end, dtype=float)
    upper_gap_end = signal_end_array + gap_center_offset * transverse_end
    lower_gap_end = signal_end_array - gap_center_offset * transverse_end
    upper_ground_end = signal_end_array + ground_center_offset * transverse_end
    lower_ground_end = signal_end_array - ground_center_offset * transverse_end
    mid_angle = start_angle + theta / 2.0

    return {
        "angle_deg": float(angle_deg),
        "curvature_center": (float(curvature_center[0]), float(curvature_center[1])),
        "start_angle_rad": float(start_angle),
        "end_angle_rad": float(end_angle),
        "signal_start": signal_start,
        "upper_gap_start": (0.0, gap_center_offset),
        "lower_gap_start": (0.0, -gap_center_offset),
        "upper_ground_start": (0.0, ground_center_offset),
        "lower_ground_start": (0.0, -ground_center_offset),
        "signal_end": signal_end,
        "upper_gap_end": (float(upper_gap_end[0]), float(upper_gap_end[1])),
        "lower_gap_end": (float(lower_gap_end[0]), float(lower_gap_end[1])),
        "upper_ground_end": (float(upper_ground_end[0]), float(upper_ground_end[1])),
        "lower_ground_end": (float(lower_ground_end[0]), float(lower_ground_end[1])),
        "center": point_on_radius(radius, mid_angle),
        "radial_ranges": {
            "inner_ground": (radius - ws / 2.0 - gap - wg, radius - ws / 2.0 - gap),
            "signal": (radius - ws / 2.0, radius + ws / 2.0),
            "outer_ground": (radius + ws / 2.0 + gap, radius + ws / 2.0 + gap + wg),
        },
    }


def segmented_electrode_landmarks(p: dict[str, Any]) -> dict[str, Any]:
    """Return landmarks for the aligned segmented T-electrode CPW.

    ``signal_width`` and ``ground_width`` are the user-defined Ws and Wg of
    the plain CPW at the beginning and end of the component.  The T-electrode
    section is calculated internally from those endpoint dimensions and the
    T-electrode transverse dimensions.

    Each endpoint has a user-defined-gap port lead, a linear transition, and then a
    50-µm plain CPW section whose gap equals the finger-tip gap before the
    patterned T-electrode begins.

    Let ``extension = t_top_width + t_neck_width``.  The internally derived
    segmented-section dimensions are::

        segmented_signal_width = end_Ws - 2 * extension
        segmented_ground_width = end_Wg - extension
        segmented_full_gap = end_gap + 2 * extension

    Finger centers are located at::

        transition_length + i * period,  i = 0 ... segment_count

    The first and last rows are clipped at the patterned boundaries, leaving
    contained half-fingers while preserving exact center-to-center periodicity.
    """
    ws_end = float(p["signal_width"])
    residual_gap = float(p["gap"])
    end_gap = float(p.get("end_gap", 3.0))
    wg_end = float(p["ground_width"])
    transition_length = float(p.get("transition_length", 1.0))
    end_flat_length = float(p.get("end_flat_length", 50.0))
    inner_flat_length = float(p.get("inner_flat_length", 50.0))
    s = float(p["t_top_width"])
    r = float(p["t_top_length"])
    h = float(p["t_neck_width"])
    t = float(p["t_neck_length"])
    c = float(p["segment_spacing"])
    n = max(1, int(p["segment_count"]))

    if min(ws_end, wg_end, s, r, h, t) <= 0 or residual_gap < 0 or end_gap < 0 or c < 0 or transition_length < 0 or end_flat_length < 0 or inner_flat_length < 0:
        raise ValueError(
            "Segmented-electrode dimensions must be positive; gap, end_gap, spacing, "
            "transition_length, end_flat_length, and inner_flat_length cannot be negative."
        )

    extension = s + h
    period = r + c
    wide_gap = residual_gap + 2.0 * extension
    ws_segmented = ws_end - 2.0 * extension
    wg_segmented = wg_end - extension
    if ws_segmented <= 0:
        raise ValueError(
            "End CPW Ws (signal_width) must be larger than "
            "2 * (t_top_width + t_neck_width)."
        )
    if wg_segmented <= 0:
        raise ValueError(
            "End CPW Wg (ground_width) must be larger than "
            "t_top_width + t_neck_width."
        )

    patterned_length = n * period
    segment_start = end_flat_length + transition_length + inner_flat_length
    segment_end = segment_start + patterned_length
    total_length = segment_end + inner_flat_length + transition_length + end_flat_length
    finger_centers = [segment_start + i * period for i in range(n + 1)]

    # User-defined plain CPW endpoints.
    plain_signal_upper = ws_end / 2.0
    plain_signal_lower = -ws_end / 2.0
    plain_upper_ground_inner = plain_signal_upper + end_gap
    plain_lower_ground_inner = plain_signal_lower - end_gap
    plain_upper_ground_outer = plain_upper_ground_inner + wg_end
    plain_lower_ground_outer = plain_lower_ground_inner - wg_end

    # Internally derived segmented T-electrode section.
    signal_upper = ws_segmented / 2.0
    signal_lower = -ws_segmented / 2.0
    upper_ground_inner = signal_upper + wide_gap
    lower_ground_inner = signal_lower - wide_gap
    upper_ground_outer = upper_ground_inner + wg_segmented
    lower_ground_outer = lower_ground_inner - wg_segmented

    return {
        "signal_width": ws_end,
        "plain_signal_width": ws_end,
        "segmented_signal_width": ws_segmented,
        "residual_gap": residual_gap,
        "end_gap": end_gap,
        "wide_gap": wide_gap,
        "ground_width": wg_end,
        "segmented_ground_width": wg_segmented,
        "transition_length": transition_length,
        "end_flat_length": end_flat_length,
        "inner_flat_length": inner_flat_length,
        "extension": extension,
        "s": s,
        "r": r,
        "h": h,
        "t": t,
        "c": c,
        "period": period,
        "segment_count": n,
        "finger_count": n + 1,
        "patterned_length": patterned_length,
        "total_length": total_length,
        "segmented_length": patterned_length,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "geometry_start": 0.0,
        "geometry_end": total_length,
        "first_finger_center": finger_centers[0],
        "last_finger_center": finger_centers[-1],
        "signal_upper": signal_upper,
        "signal_lower": signal_lower,
        "upper_ground_inner": upper_ground_inner,
        "upper_ground_outer": upper_ground_outer,
        "lower_ground_inner": lower_ground_inner,
        "lower_ground_outer": lower_ground_outer,
        "plain_signal_upper": plain_signal_upper,
        "plain_signal_lower": plain_signal_lower,
        "plain_upper_ground_inner": plain_upper_ground_inner,
        "plain_upper_ground_outer": plain_upper_ground_outer,
        "plain_lower_ground_inner": plain_lower_ground_inner,
        "plain_lower_ground_outer": plain_lower_ground_outer,
        "inner_signal_upper": ws_end / 2.0,
        "inner_signal_lower": -ws_end / 2.0,
        "inner_upper_ground_inner": ws_end / 2.0 + residual_gap,
        "inner_upper_ground_outer": ws_end / 2.0 + residual_gap + wg_end,
        "inner_lower_ground_inner": -ws_end / 2.0 - residual_gap,
        "inner_lower_ground_outer": -ws_end / 2.0 - residual_gap - wg_end,
        "upper_gap_center": plain_signal_upper + end_gap / 2.0,
        "lower_gap_center": plain_signal_lower - end_gap / 2.0,
        "wide_upper_gap_center": signal_upper + wide_gap / 2.0,
        "wide_lower_gap_center": signal_lower - wide_gap / 2.0,
        "finger_centers": finger_centers,
    }


def _signed_direction(value: Any, default: str = "up") -> float:
    text = str(value if value is not None else default).strip().lower()
    return -1.0 if text in {"down", "lower", "negative", "-", "-1"} else 1.0


def feedline_landmarks(p: dict[str, Any], mirrored: bool) -> dict[str, tuple[float, float]]:
    mirror_sign = -1.0 if mirrored else 1.0
    input_sign = mirror_sign * _signed_direction(p.get("input_s_bend_direction", "up"))
    output_sign = mirror_sign * _signed_direction(p.get("output_s_bend_direction", "down"), "down")
    x_input_straight_end = float(p["input_straight_length"])
    x_first_s_bend_end = x_input_straight_end + float(p["s_bend_length"])
    y_first = input_sign * abs(float(p["offset"]))
    x_lc_end = x_first_s_bend_end + float(p["Lc"])
    x_second_s_bend_end = x_lc_end + float(p["s_bend_length"])
    y_second = y_first + output_sign * abs(float(p["offset"]))
    x_output = x_second_s_bend_end + float(p["output_straight_length"])
    return {
        "input": (0.0, 0.0),
        "input_straight_end": (x_input_straight_end, 0.0),
        "first_s_bend_end": (x_first_s_bend_end, y_first),
        "lc_end": (x_lc_end, y_first),
        "second_s_bend_end": (x_second_s_bend_end, y_second),
        "output": (x_output, y_second),
        "input_offset": (0.0, y_first),
        "output_offset": (0.0, y_second - y_first),
    }


def ring_two_feedline_landmarks(p: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Landmarks for a ring with two symmetric buses and four grating couplers.

    The buses remain at the ring coupling separation through the center region.
    Smooth S-bends at both ends transition to the independently specified
    grating-coupler separation.
    """
    length = float(p["feedline_length"])
    s_bend_length = float(p["s_bend_length"])
    ring_radius = float(p["ring_radius"])
    ring_width = float(p["ring_width"])
    feedline_width = float(p["feedline_width"])
    coupling_gap = float(p["coupling_gap"])
    gc_separation = float(p["grating_coupler_separation"])
    if length <= 2.0 * s_bend_length:
        raise ValueError("feedline_length must be greater than 2 × s_bend_length.")
    if gc_separation < 0:
        raise ValueError("grating_coupler_separation cannot be negative.")
    bus_y = ring_radius + ring_width / 2.0 + coupling_gap + feedline_width / 2.0
    bend_offset = abs(float(p.get("s_bend_offset", max(0.0, gc_separation / 2.0 - bus_y))))
    input_sign = _signed_direction(p.get("input_s_bend_direction", "down"), "down")
    output_sign = _signed_direction(p.get("output_s_bend_direction", "up"), "up")
    x_left = -length / 2.0
    x_bus_left = x_left + s_bend_length
    x_bus_right = length / 2.0 - s_bend_length
    x_right = length / 2.0
    return {
        "upper_left_gc": (x_left, bus_y-input_sign*bend_offset),
        "upper_left_bus": (x_bus_left, bus_y),
        "upper_right_bus": (x_bus_right, bus_y),
        "upper_right_gc": (x_right, bus_y+output_sign*bend_offset),
        "lower_left_gc": (x_left, -bus_y-input_sign*bend_offset),
        "lower_left_bus": (x_bus_left, -bus_y),
        "lower_right_bus": (x_bus_right, -bus_y),
        "lower_right_gc": (x_right, -bus_y+output_sign*bend_offset),
        "upper_bus_center": (0.0, bus_y),
        "lower_bus_center": (0.0, -bus_y),
        "center": (0.0, 0.0),
    }


def loopback_landmarks(p: dict[str, Any], mirrored: bool) -> dict[str, tuple[float, float]]:
    """Landmarks for a two-arm loopback mirror.

    Two parallel straight waveguides begin at the left with an edge-to-edge
    gap ``gap``.  After the common straight length ``Lc`` each arm follows a
    smooth S-bend to the two endpoints of a right-facing semicircular arc.
    """
    ms = -1.0 if mirrored else 1.0
    width = float(p["width"])
    gap = float(p["gap"])
    lc = float(p["Lc"])
    sb = float(p["s_bend_length"])
    radius = float(p["arc_radius"])
    input_half_sep = 0.5 * (gap + width)
    arc_half_sep = radius
    offset = max(0.0, arc_half_sep - input_half_sep)
    upper_y0 = ms * input_half_sep
    lower_y0 = -ms * input_half_sep
    upper_y1 = ms * arc_half_sep
    lower_y1 = -ms * arc_half_sep
    arc_x = lc + sb
    return {
        "left_upper": (0.0, upper_y0),
        "left_lower": (0.0, lower_y0),
        "upper_straight_end": (lc, upper_y0),
        "lower_straight_end": (lc, lower_y0),
        "upper_s_bend_end": (arc_x, upper_y1),
        "lower_s_bend_end": (arc_x, lower_y1),
        "arc_center": (arc_x, 0.0),
        "offset": (offset, 0.0),
    }


def resonator_x_positions(p: dict[str, Any], x_start: float, x_end: float) -> list[float]:
    count = max(1, int(p.get("resonator_count", 1)))
    middle = 0.5 * (x_start + x_end)
    if count == 1:
        return [middle]
    spacing = float(p.get("resonator_spacing", 100.0))
    first = middle - 0.5 * spacing * (count - 1)
    return [first + i * spacing for i in range(count)]
