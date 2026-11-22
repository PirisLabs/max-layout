"""Resolved geometry for an Euler racetrack with a dual-slot lumped electrode."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


RACETRACK_LUMPED_ELECTRODE_KIND = "Euler racetrack + dual-slot lumped electrode"


def euler_bend_centerline(
    radius: float,
    bend_angle_deg: float,
    samples: int = 4001,
) -> tuple[np.ndarray, float]:
    """Centerline of a full-Euler bend and its arc length."""
    theta = math.radians(float(bend_angle_deg))
    radius = float(radius)
    if radius <= 0.0 or theta <= 0.0:
        raise ValueError("Euler bend needs a positive radius and angle.")
    arc = 2.0 * radius * theta
    s = np.linspace(0.0, arc, max(101, int(samples)))
    half = arc / 2.0
    phi = np.where(
        s <= half,
        (1.0 / radius / arc) * s**2,
        theta - (1.0 / radius / arc) * (arc - s) ** 2,
    )
    step = np.diff(s)
    mid = (phi[:-1] + phi[1:]) / 2.0
    x = np.concatenate(([0.0], np.cumsum(step * np.cos(mid))))
    y = np.concatenate(([0.0], np.cumsum(step * np.sin(mid))))
    return np.column_stack((x, y)), arc


def euler_racetrack_metrics(
    total_length: float,
    radius: float,
    samples: int = 4001,
) -> tuple[float, float, float]:
    """Return straight length per side, one bend arc, and bend offset."""
    bend, arc = euler_bend_centerline(radius, 180.0, samples)
    straight = float(total_length) / 2.0 - arc
    if straight < 0.0:
        raise ValueError(
            f"Racetrack total length {float(total_length):g} um is shorter than its two Euler bends "
            f"({2.0 * arc:.3f} um at R_eff {float(radius):g} um); increase the length or reduce R_eff."
        )
    return straight, arc, float(bend[-1][1])


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def resolve_racetrack_lumped_electrode(p: dict[str, Any]) -> dict[str, Any]:
    """Resolve optical landmarks and RF polygons from one parameter model.

    Each optical straight is centered in an electrode gap.  The upper pair has
    its ground conductor toward the racetrack center and its signal conductor
    outside; the lower pair is reversed.  This reproduces the existing
    ``modulator_RF2_bridge`` polarity while deriving pair separation from the
    racetrack rather than from a separately editable distance.
    """
    total_length = float(p["racetrack_length"])
    radius = float(p["R_eff"])
    race_width = float(p["racetrack_width"])
    points = max(101, int(p.get("points", 4001)))
    straight, bend_arc, height = euler_racetrack_metrics(total_length, radius, points)

    clearance = float(p.get("electrode_end_clearance", 100.0))
    mode = str(p.get("electrode_length_mode", "fit straight")).strip().lower()
    if mode == "fit straight":
        electrode_length = straight - 2.0 * clearance
    elif mode == "manual":
        electrode_length = float(p.get("electrode_length", straight))
    else:
        raise ValueError("Electrode length mode must be 'fit straight' or 'manual'.")

    signal_width = float(p["signal_width"])
    ground_width = float(p["ground_width"])
    gap = float(p["interaction_gap"])
    bus_width = float(p["center_bus_width"])
    connection = str(p.get("center_connection", "inner join")).strip().lower()
    if race_width <= 0.0:
        raise ValueError("Racetrack width must be positive.")
    if clearance < 0.0:
        raise ValueError("Electrode end clearance cannot be negative.")
    if electrode_length <= 0.0:
        raise ValueError("Electrode end clearances leave no positive interaction length.")
    if electrode_length > straight + 1e-9:
        raise ValueError("Electrode length cannot exceed the Euler-racetrack straight length.")
    if min(signal_width, ground_width, gap, bus_width) <= 0.0:
        raise ValueError("Electrode conductor widths, interaction gap, and center bus width must be positive.")
    if connection not in {"none", "inner join"}:
        raise ValueError("Center connection must be 'none' or 'inner join'.")

    x0 = -electrode_length / 2.0
    x1 = electrode_length / 2.0
    upper_gap_y = height / 2.0
    lower_gap_y = -height / 2.0

    top_ground_low = upper_gap_y - gap / 2.0 - ground_width
    top_ground_high = upper_gap_y - gap / 2.0
    top_signal_low = upper_gap_y + gap / 2.0
    top_signal_high = top_signal_low + signal_width
    bottom_signal_low = lower_gap_y + gap / 2.0
    bottom_signal_high = bottom_signal_low + signal_width
    bottom_ground_high = lower_gap_y - gap / 2.0
    bottom_ground_low = bottom_ground_high - ground_width

    rf_polygons = [
        _rectangle(x0, top_signal_low, x1, top_signal_high),
        _rectangle(x0, top_ground_low, x1, top_ground_high),
        _rectangle(x0, bottom_signal_low, x1, bottom_signal_high),
        _rectangle(x0, bottom_ground_low, x1, bottom_ground_high),
    ]
    if connection == "inner join":
        rf_polygons.append(
            _rectangle(
                -bus_width / 2.0,
                bottom_signal_high,
                bus_width / 2.0,
                top_ground_low,
            )
        )

    return {
        "straight_length": straight,
        "bend_arc": bend_arc,
        "racetrack_height": height,
        "electrode_length": electrode_length,
        "electrode_x0": x0,
        "electrode_x1": x1,
        "upper_gap_y": upper_gap_y,
        "lower_gap_y": lower_gap_y,
        "rf_polygons": rf_polygons,
        "center_connection": connection,
    }
