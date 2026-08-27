"""Exponential width profile for the dedicated SLAB-layer taper."""

from __future__ import annotations

from typing import Any

import numpy as np


def slab_taper_profile(
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitudinal positions and exact exponential slab widths."""
    length = float(parameters["length"])
    initial_width = float(parameters["initial_width"])
    final_width = float(parameters["final_width"])
    exponential_factor = float(parameters.get("exponential_factor", 1.3))
    points = max(2, int(parameters.get("points", 1001)))
    if length <= 0.0:
        raise ValueError("SLAB taper length must be positive.")
    if initial_width <= 0.0 or final_width <= 0.0:
        raise ValueError("SLAB taper initial and final widths must be positive.")
    if exponential_factor <= 0.0:
        raise ValueError("SLAB taper exponential_factor must be positive.")
    u = np.linspace(0.0, 1.0, points)
    widths = initial_width * (final_width / initial_width) ** (
        u**exponential_factor
    )
    widths[0] = initial_width
    widths[-1] = final_width
    return length * u, widths
