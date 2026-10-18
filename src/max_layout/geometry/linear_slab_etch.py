"""Parameterized TFLN edge coupler derived from the supplied reference GDS."""

from __future__ import annotations

from typing import Any

import numpy as np


LINEAR_SLAB_ETCH_KIND = "Linear Slab Etch"

# Normalized vertices preserve the measured reference-GDS taper profiles while
# allowing the requested endpoint widths and lengths to remain editable.
_SLAB_X = np.asarray([0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0])
_SLAB_WIDTH = np.asarray([0.2, 0.34, 0.5, 0.8, 1.2, 2.5, 8.0, 32.0])
_RIB_X = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
_RIB_WIDTH = np.asarray([0.3, 0.5, 0.8, 1.1, 1.2])


def _scaled_profile(fractions: np.ndarray, reference_widths: np.ndarray,
                    start_width: float, end_width: float, start_x: float,
                    length: float) -> tuple[np.ndarray, np.ndarray]:
    normalized = (reference_widths - reference_widths[0]) / (
        reference_widths[-1] - reference_widths[0]
    )
    return start_x + length * fractions, start_width + (end_width - start_width) * normalized


def resolve_linear_slab_etch(parameters: dict[str, Any]) -> dict[str, Any]:
    p = {
        "slab_tip_width": float(parameters.get("slab_tip_width", 0.2)),
        "slab_tip_taper_length": float(parameters.get("slab_tip_taper_length", 100.0)),
        "slab_tip_length": float(parameters.get("slab_tip_length", 6.0)),
        "slab_wg_width": float(parameters.get("slab_wg_width", 32.0)),
        "slab_wg_length": float(parameters.get("slab_wg_length", 11.0)),
        "rib_tip_width": float(parameters.get("rib_tip_width", 0.3)),
        "rib_taper_length": float(parameters.get("rib_taper_length", 60.0)),
        "rib_wg_width": float(parameters.get("rib_wg_width", 1.2)),
        "rib_wg_length": float(parameters.get("rib_wg_length", 11.0)),
        "slab_tfln_thickness": float(parameters.get("slab_tfln_thickness", 0.2)),
        "rib_tfln_thickness": float(parameters.get("rib_tfln_thickness", 0.2)),
        "input_port_offset": float(parameters.get("input_port_offset", 1.0)),
    }
    if any(value <= 0.0 for value in p.values()):
        raise ValueError("Linear Slab Etch dimensions, thicknesses, and input-port offset must be positive.")
    if p["rib_taper_length"] > p["slab_tip_taper_length"]:
        raise ValueError("Rib taper length cannot exceed slab tip taper length.")
    taper_start = p["slab_tip_length"]
    taper_end = taper_start + p["slab_tip_taper_length"]
    rib_taper_start = taper_end - p["rib_taper_length"]
    slab_x, slab_width = _scaled_profile(
        _SLAB_X, _SLAB_WIDTH, p["slab_tip_width"], p["slab_wg_width"],
        taper_start, p["slab_tip_taper_length"],
    )
    rib_x, rib_width = _scaled_profile(
        _RIB_X, _RIB_WIDTH, p["rib_tip_width"], p["rib_wg_width"],
        rib_taper_start, p["rib_taper_length"],
    )
    p.update({
        "slab_taper_start": taper_start,
        "slab_taper_end": taper_end,
        "slab_end": taper_end + p["slab_wg_length"],
        "rib_taper_start": rib_taper_start,
        "rib_taper_end": taper_end,
        "rib_end": taper_end + p["rib_wg_length"],
        "input_port_x": -p["input_port_offset"],
        "waveguide_port_x": taper_end + 0.5 * p["rib_wg_length"],
        "slab_taper_x": slab_x,
        "slab_taper_width": slab_width,
        "rib_taper_x": rib_x,
        "rib_taper_width": rib_width,
    })
    return p


def symmetric_polygon(x: np.ndarray, width: np.ndarray) -> np.ndarray:
    upper = np.column_stack((x, width / 2.0))
    lower = np.column_stack((x[::-1], -width[::-1] / 2.0))
    return np.vstack((upper, lower))
