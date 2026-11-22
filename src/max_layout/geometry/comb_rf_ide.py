"""Geometry model for the CPW-coupled comb interdigitated electrode."""

from __future__ import annotations

import math
from typing import Any


COMB_RF_IDE_KIND = "comb_RF_IDE"


def resolve_comb_rf_ide(parameters: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters and return local rectangular geometry landmarks.

    The IDE is a capacitive break in the CPW signal trace. Alternating fingers
    attach to left and right signal buses and are packed from the signal's top
    edge toward its bottom edge. ``gap`` is the clearance from each finger tip
    to the opposing bus; ``finger_gap`` is the transverse finger spacing.
    """
    length = float(parameters["length"])
    signal_width = float(parameters["signal_width"])
    gap = float(parameters["gap"])
    ground_width = float(parameters["ground_width"])
    finger_width = float(parameters["finger_width"])
    finger_length = float(parameters["finger_length"])
    finger_gap = float(parameters["finger_gap"])
    ide_offset = float(parameters["IDE_Offset"])

    if min(length, signal_width, ground_width, finger_width, finger_length) <= 0.0:
        raise ValueError("comb_RF_IDE requires positive length, conductor widths, finger width, and finger length.")
    if gap < 0.0 or finger_gap < 0.0 or ide_offset <= 0.0:
        raise ValueError("comb_RF_IDE requires nonnegative gaps and a positive IDE_Offset.")
    ide_right = ide_offset + finger_length + gap
    if ide_right >= length:
        raise ValueError("comb_RF_IDE requires IDE_Offset + finger_length + gap to be less than length.")
    finger_count = int(math.floor((signal_width + finger_gap) / (finger_width + finger_gap)))
    if finger_count < 1:
        raise ValueError("comb_RF_IDE signal_width must fit at least one finger.")

    signal_low = -signal_width / 2.0
    signal_high = signal_width / 2.0
    finger_rectangles = []
    for index in range(finger_count):
        y_high = signal_high - index * (finger_width + finger_gap)
        y_low = y_high - finger_width
        if index % 2 == 0:
            x_low, x_high = ide_offset, ide_offset + finger_length
            bus = "left"
        else:
            x_low, x_high = ide_offset + gap, ide_right
            bus = "right"
        finger_rectangles.append(
            {"index": index, "bus": bus, "bounds": (x_low, y_low, x_high, y_high)}
        )

    return {
        "length": length,
        "signal_width": signal_width,
        "gap": gap,
        "ground_width": ground_width,
        "IDE_Offset": ide_offset,
        "ide_right": ide_right,
        "finger_count": finger_count,
        "finger_rectangles": finger_rectangles,
        "left_signal_bus": (0.0, signal_low, ide_offset, signal_high),
        "right_signal_bus": (ide_right, signal_low, length, signal_high),
        "upper_ground": (0.0, signal_high + gap, length, signal_high + gap + ground_width),
        "lower_ground": (0.0, signal_low - gap - ground_width, length, signal_low - gap),
    }
