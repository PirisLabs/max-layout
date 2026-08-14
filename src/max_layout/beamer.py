"""Small, deterministic helpers for BEAMER flow and CJOB exports."""

from __future__ import annotations

from pathlib import PureWindowsPath
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote
from xml.sax.saxutils import escape


DEFAULT_FRAME_SIZE_UM = (14000.0, 12000.0)
DEFAULT_GPF_NAME = "photonic_layout.gpf"


def _positive_pair(values: Sequence[float], label: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{label} must contain a width and height")
    width_um, height_um = (float(values[0]), float(values[1]))
    if width_um <= 0.0 or height_um <= 0.0:
        raise ValueError(f"{label} dimensions must be positive")
    return width_um, height_um


def beamer_frame_size_um(
    components: Iterable[Mapping[str, Any]],
    field_records: Iterable[Mapping[str, Any]] = (),
    fallback: Sequence[float] = DEFAULT_FRAME_SIZE_UM,
) -> tuple[float, float]:
    """Return the layout frame used to size an EBPG mask.

    A ``Chip outline`` is the authoritative frame.  If an older project has no
    outline, the complete write-field extent is used when it is large enough;
    the application default chip size is the final compatibility fallback.
    """

    outlines = [
        component
        for component in components
        if str(component.get("kind", "")) == "Chip outline"
    ]
    if outlines:
        # A project should normally contain one outline.  Choosing the largest
        # keeps legacy files containing a small alignment-frame outline usable.
        dimensions = []
        for component in outlines:
            params = component.get("params", {})
            width_um, height_um = _positive_pair(
                (params.get("width", 0.0), params.get("height", 0.0)),
                "Chip outline",
            )
            dimensions.append((width_um * height_um, width_um, height_um))
        _area, width_um, height_um = max(dimensions)
        return width_um, height_um

    records = list(field_records)
    if records:
        xmin = min(float(record["xmin"]) for record in records)
        ymin = min(float(record["ymin"]) for record in records)
        xmax = max(float(record["xmax"]) for record in records)
        ymax = max(float(record["ymax"]) for record in records)
        width_um, height_um = xmax - xmin, ymax - ymin
        # The CJOB mask is 2 mm smaller than its frame.  A write-field extent
        # below that threshold cannot define a valid substrate, so retain the
        # established application frame instead.
        if width_um > 2000.0 and height_um > 2000.0:
            return width_um, height_um

    return _positive_pair(fallback, "Fallback frame")


def beamer_mask_size_mm(
    frame_size_um: Sequence[float],
    reduction_mm: float = 2.0,
) -> tuple[float, float]:
    """Make each mask dimension ``reduction_mm`` smaller than its frame."""

    width_um, height_um = _positive_pair(frame_size_um, "Layout frame")
    reduction_mm = float(reduction_mm)
    if reduction_mm < 0.0:
        raise ValueError("Mask-size reduction cannot be negative")
    width_mm = width_um / 1000.0 - reduction_mm
    height_mm = height_um / 1000.0 - reduction_mm
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError(
            "The layout frame must exceed the CJOB mask reduction in both dimensions"
        )
    return width_mm, height_mm


def beamer_gpf_name(flow: str) -> str:
    """Read the exported GPF basename from a BEAMER FTXT flow."""

    matches = re.findall(
        r"^FILE_NAME\s*=\s*(\S+?\.gpf)\s*$",
        str(flow),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return DEFAULT_GPF_NAME
    decoded = unquote(matches[-1]).replace("/", "\\")
    return PureWindowsPath(decoded).name or DEFAULT_GPF_NAME


def _number(value: float) -> str:
    return f"{float(value):.12g}"


def beamer_cjob_template(
    pattern_name: str,
    frame_size_um: Sequence[float],
    *,
    exposure_name: str = "photonic_layout",
    beam_name: str = "10na_300.beam_100",
    dose: float = 1000.0,
) -> str:
    """Create the single-pattern EBPG 5200 CJOB paired with an FTXT export."""

    pattern_name = str(pattern_name).strip() or DEFAULT_GPF_NAME
    exposure_name = str(exposure_name).strip() or "photonic_layout"
    beam_name = str(beam_name).strip() or "10na_300.beam_100"
    mask_width_mm, mask_height_mm = beamer_mask_size_mm(frame_size_um)

    def attribute(value: object) -> str:
        return escape(str(value), {'"': "&quot;"})

    mask_size = f"{_number(mask_width_mm)}mmx{_number(mask_height_mm)}mm"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<cjob type="ebpg5200" version="v02_23">
  <color rgb="0 255 0" pattern="{attribute(pattern_name)}" num="1"/>
  <substrate substrate="mask">
    <mask size="{mask_size}"/>
    <color substrate="mask" rgb="200 200 200"/>
    <position coord="0,0"/>
    <exposure height="check" workinglevel="high" ht="100kV" name="{attribute(exposure_name)}">
      <position coord="0,0"/>
      <checks enabled="false"/>
      <pattern name="{attribute(pattern_name)}">
        <position coord="0,0"/>
        <beam dose="{_number(dose)}" defocus="#0" name="{attribute(beam_name)}"/>
      </pattern>
    </exposure>
  </substrate>
</cjob>
'''
