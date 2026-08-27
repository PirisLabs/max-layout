"""Landmarks for automatic global E-beam markers inside a chip frame."""

from __future__ import annotations

from typing import Any


def chip_outline_global_marker_centers(
    parameters: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """Return centered marker groups on all four sides in frame-local coordinates."""
    width = float(parameters["width"])
    height = float(parameters["height"])
    line_width = float(parameters.get("line_width", 10.0))
    marker_size = float(parameters.get("ebeam_global_marker_size", 50.0))
    clearance = float(parameters.get("ebeam_global_marker_edge_clearance", 0.0))
    marker_count = int(parameters.get("ebeam_global_marker_count", 3))
    marker_spacing = float(parameters.get("ebeam_global_marker_spacing", 200.0))
    if width <= 0.0 or height <= 0.0 or line_width <= 0.0:
        raise ValueError("Chip frame dimensions and line width must be positive.")
    if marker_size <= 0.0 or clearance < 0.0:
        raise ValueError(
            "Ebeam global marker size must be positive and edge clearance cannot be negative."
        )
    if marker_count < 1 or (marker_count > 1 and marker_spacing <= 0.0):
        raise ValueError(
            "Ebeam global marker count must be at least one and spacing must be positive for marker groups."
        )
    inset = line_width + clearance + marker_size / 2.0
    if 2.0 * inset >= min(width, height):
        raise ValueError("Ebeam global markers do not fit inside the chip frame.")
    offsets = [
        (index - (marker_count - 1) / 2.0) * marker_spacing
        for index in range(marker_count)
    ]
    if max(abs(offset) for offset in offsets) > min(width, height) / 2.0 - inset:
        raise ValueError("Ebeam global marker groups do not fit inside the chip frame.")

    centers: dict[str, tuple[float, float]] = {}
    for side in ("top", "bottom", "left", "right"):
        for index, offset in enumerate(offsets, start=1):
            name = side if marker_count == 1 else f"{side}_{index}"
            if side == "top":
                center = (offset, height / 2.0 - inset)
            elif side == "bottom":
                center = (offset, -height / 2.0 + inset)
            elif side == "left":
                center = (-width / 2.0 + inset, offset)
            else:
                center = (width / 2.0 - inset, offset)
            centers[name] = center
    return centers
