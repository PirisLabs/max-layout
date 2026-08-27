"""Resolved geometry for an Euler racetrack with a vertical point-coupled feedline."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


VERTICAL_RACETRACK_FEEDLINE_KIND = "Racetrack + vertical feedline"


def _full_euler_180_metrics(
    total_length: float,
    radius: float,
    samples: int,
) -> tuple[float, float, float]:
    """Return one straight length, bend height, and positive-X bend bulge."""
    total = float(total_length)
    r_eff = float(radius)
    if r_eff <= 0.0:
        raise ValueError("Vertical-feedline racetrack R_eff must be positive.")
    arc = 2.0 * math.pi * r_eff
    straight = total / 2.0 - arc
    if straight < 0.0:
        raise ValueError(
            f"Racetrack total length {total:g} um is shorter than its two Euler bends "
            f"({2.0 * arc:.3f} um at R_eff {r_eff:g} um)."
        )

    count = max(101, int(samples))
    s = np.linspace(0.0, arc, count)
    half = arc / 2.0
    theta = math.pi
    phi = np.where(
        s <= half,
        (1.0 / r_eff / arc) * s**2,
        theta - (1.0 / r_eff / arc) * (arc - s) ** 2,
    )
    step = np.diff(s)
    mid = (phi[:-1] + phi[1:]) / 2.0
    x = np.concatenate(([0.0], np.cumsum(step * np.cos(mid))))
    y = np.concatenate(([0.0], np.cumsum(step * np.sin(mid))))
    return straight, float(y[-1]), float(np.max(x))


def resolve_vertical_racetrack_feedline(
    parameters: dict[str, Any],
    mirrored: bool = False,
) -> dict[str, Any]:
    """Resolve the symmetric child feedline and its placement around the racetrack.

    The racetrack center is the composite anchor.  The feedline coupling-section
    midpoint is always at local ``y = 0``, and the two grating anchors remain
    exactly half of ``feedline_length`` below and above it.
    """
    p = dict(parameters)
    total_length = float(p["racetrack_length"])
    r_eff = float(p["R_eff"])
    race_width = float(p["racetrack_width"])
    feed_width = float(p["wg_width"])
    gap = float(p["coupling_gap"])
    feedline_length = float(p["feedline_length"])
    input_straight = float(p["input_straight_length"])
    total_s_bend_length = float(p["s_bend_length"])
    total_lateral_offset = float(p["offset"])
    coupling_length = float(p["Lc"])
    points = max(101, int(p.get("points", 4001)))

    if min(
        race_width,
        feed_width,
        feedline_length,
        total_s_bend_length,
        coupling_length,
    ) <= 0.0:
        raise ValueError(
            "Vertical-feedline racetrack widths, feedline length, S-bend length, "
            "and coupling-section length must be positive."
        )
    if min(gap, input_straight, total_lateral_offset) < 0.0:
        raise ValueError(
            "Vertical-feedline racetrack gap, input straight, and lateral offset "
            "cannot be negative."
        )

    straight, bend_height, bend_bulge = _full_euler_180_metrics(
        total_length,
        r_eff,
        points,
    )
    half_feedline = feedline_length / 2.0
    each_s_bend_length = total_s_bend_length / 2.0
    each_lateral_offset = total_lateral_offset / 2.0
    pre_coupling = (
        half_feedline
        - input_straight
        - each_s_bend_length
        - coupling_length / 2.0
    )
    post_coupling = pre_coupling
    output_straight = input_straight
    if pre_coupling < 0.0:
        minimum = 2.0 * (
            input_straight + each_s_bend_length + coupling_length / 2.0
        )
        raise ValueError(
            f"Symmetric vertical feedline is too short; use at least {minimum:g} um "
            "for the current input straight, S-bend, and coupling section."
        )

    side = str(p.get("coupling_side", "right")).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("Vertical feedline coupling_side must be left or right.")
    side_sign = 1.0 if side == "right" else -1.0
    racetrack_half_x = straight / 2.0 + bend_bulge
    coupling_x = side_sign * (
        racetrack_half_x + gap + (race_width + feed_width) / 2.0
    )
    grating_x = coupling_x + side_sign * each_lateral_offset

    mirror_sign = -1.0 if mirrored else 1.0
    feed_orientation = 90.0 * mirror_sign
    feed_start_y = -mirror_sign * half_feedline
    direction_sign = side_sign * mirror_sign
    input_direction = "up" if direction_sign > 0.0 else "down"

    feed_params = {
        "pitch": float(p.get("pitch", 0.83)),
        "fill_factor": float(p.get("fill_factor", 0.47)),
        "N": int(p.get("N", 30)),
        "alpha_t": float(p.get("alpha_t", 25.0)),
        "taper_L": float(p.get("taper_L", 22.0)),
        "wg_width": feed_width,
        "gc_wg_length": float(p.get("gc_wg_length", 20.0)),
        "input_straight_length": input_straight,
        "s_bend_length": each_s_bend_length,
        "offset": each_lateral_offset,
        "pre_coupling_straight_length": pre_coupling,
        "Lc": coupling_length,
        "post_coupling_straight_length": post_coupling,
        "output_straight_length": output_straight,
        "layer": int(p.get("layer", 1)),
        "datatype": int(p.get("datatype", 0)),
        "tolerance": float(p.get("tolerance", 0.001)),
        "gc_tolerance": float(p.get("gc_tolerance", 0.0005)),
        "add_grating_couplers": bool(p.get("add_grating_couplers", True)),
        "add_output_s_bend": True,
        "input_s_bend_direction": input_direction,
        # Same path-traversal sign at both ends places the two grating anchors
        # on opposite lateral sides of the centered coupling line.
        "output_s_bend_direction": input_direction,
        "gc_layer": int(p.get("gc_layer", 2)),
        "gc_datatype": int(p.get("gc_datatype", 0)),
    }
    return {
        "racetrack_straight_length": straight,
        "racetrack_bend_height": bend_height,
        "racetrack_half_x": racetrack_half_x,
        "each_s_bend_length": each_s_bend_length,
        "each_lateral_offset": each_lateral_offset,
        "coupling_center": (coupling_x, 0.0),
        "feed_start": (grating_x, feed_start_y),
        "feed_orientation_deg": feed_orientation,
        "feed_params": feed_params,
    }
