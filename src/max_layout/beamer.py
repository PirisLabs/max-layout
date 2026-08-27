"""Small, deterministic helpers for BEAMER flow and CJOB exports."""

from __future__ import annotations

from pathlib import PureWindowsPath
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote
from xml.sax.saxutils import escape

from .constants import (
    EDGE_COUPLER_FEED_KINDS,
    GLOBAL_MARKER_LAYER,
    LOCAL_MARKER_LAYER,
    MODULATOR_COMPONENT_KINDS,
    component_edge_coupler_slot_count,
    edge_coupler_local_marker_layer,
    edge_coupler_slab_layer,
)
from .geometry.edge_coupler import resolved_edge_coupler_parameters
from .ports import component_global_ports


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


def _edge_coupler_slots(
    components: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            component
            for component in components
            if component_edge_coupler_slot_count(component) > 0
        ),
        key=lambda component: int(component.get("uid", 0)),
    )
    slots: list[dict[str, Any]] = []
    number = 1
    for component in candidates:
        kind = str(component.get("kind", ""))
        if kind in EDGE_COUPLER_FEED_KINDS or kind in MODULATOR_COMPONENT_KINDS:
            ends = ("left", "right")
        else:
            ends = ("single",)
        for end in ends:
            slots.append(
                {
                    "number": number,
                    "component": component,
                    "end": end,
                }
            )
            number += 1
    return slots


def beamer_local_alignment_marks(
    components: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect global centers of automatic edge-coupler LOCAL squares."""
    records: list[dict[str, Any]] = []
    for slot in _edge_coupler_slots(components):
        number = int(slot["number"])
        component = slot["component"]
        end = str(slot["end"])
        values = resolved_edge_coupler_parameters(dict(component.get("params", {})))
        ports = component_global_ports(dict(component))
        prefix = "" if end == "single" else f"{end}_"
        for side in ("top", "bottom"):
            port_name = f"{prefix}local_alignment_{side}"
            center = ports[port_name]["center"]
            records.append(
                {
                    "ident": f"L{number}",
                    "name": f"L-marker {number}",
                    "scope": "local",
                    "coupler_number": number,
                    "end": end,
                    "uid": int(component.get("uid", 0)),
                    "side": side,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "size_um": float(
                        component.get("params", {}).get(
                            "edge_coupler_local_marker_size",
                            values["local_alignment_mark_size"],
                        )
                    ),
                    "layer": edge_coupler_local_marker_layer(number),
                    "slab_layer": edge_coupler_slab_layer(number),
                }
            )
    return records


def beamer_edge_coupler_patterns(
    components: Iterable[Mapping[str, Any]],
    main_pattern_name: str,
) -> list[dict[str, Any]]:
    """Return one numbered GPF association for every edge-coupler SLAB."""
    basename = PureWindowsPath(str(main_pattern_name)).name or DEFAULT_GPF_NAME
    stem = re.sub(r"\.gpf$", "", basename, flags=re.IGNORECASE) or "photonic_layout"
    return [
        {
            "coupler_number": int(slot["number"]),
            "uid": int(slot["component"].get("uid", 0)),
            "end": str(slot["end"]),
            "name": f"{stem}_slab_{int(slot['number']):03d}.gpf",
            "slab_layer": edge_coupler_slab_layer(int(slot["number"])),
            "local_marker_layer": edge_coupler_local_marker_layer(int(slot["number"])),
            "local_marker_ident": f"L{int(slot['number'])}",
        }
        for slot in _edge_coupler_slots(components)
    ]


def beamer_global_alignment_marks(
    components: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect global centers of frame-owned Ebeam global marker squares."""
    records: list[dict[str, Any]] = []
    for component in components:
        if str(component.get("kind", "")) != "Chip outline":
            continue
        params = dict(component.get("params", {}))
        ports = component_global_ports(dict(component))
        for port_name, port in ports.items():
            if not port_name.startswith("ebeam_global_"):
                continue
            side = port_name.removeprefix("ebeam_global_")
            center = port["center"]
            records.append(
                {
                    "ident": "GLOBAL",
                    "name": str(
                        params.get("ebeam_global_marker_name", "Ebeam global marker")
                    ),
                    "scope": "global",
                    "uid": int(component.get("uid", 0)),
                    "side": side,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "size_um": float(params.get("ebeam_global_marker_size", 50.0)),
                    "layer": GLOBAL_MARKER_LAYER,
                }
            )
    return records


def beamer_alignment_marks(
    components: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect all LOCAL edge-coupler and GLOBAL chip-frame marks."""
    component_list = list(components)
    return beamer_local_alignment_marks(
        component_list
    ) + beamer_global_alignment_marks(component_list)


def beamer_cjob_template(
    pattern_name: str,
    frame_size_um: Sequence[float],
    *,
    exposure_name: str = "photonic_layout",
    beam_name: str = "10na_300.beam_100",
    dose: float = 1000.0,
    local_alignment_marks: Iterable[Mapping[str, Any]] = (),
    edge_coupler_patterns: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Create an EBPG 5200 CJOB with numbered edge-coupler GPF associations."""

    pattern_name = str(pattern_name).strip() or DEFAULT_GPF_NAME
    exposure_name = str(exposure_name).strip() or "photonic_layout"
    beam_name = str(beam_name).strip() or "10na_300.beam_100"
    mask_width_mm, mask_height_mm = beamer_mask_size_mm(frame_size_um)

    def attribute(value: object) -> str:
        return escape(str(value), {'"': "&quot;"})

    edge_patterns = [dict(record) for record in edge_coupler_patterns]
    patterns_by_coupler = {
        int(record.get("coupler_number", 0)): record for record in edge_patterns
    }

    mark_lines: list[str] = []
    for mark in local_alignment_marks:
        ident = attribute(str(mark.get("ident", "LOCAL")).replace("--", "-"))
        side = attribute(str(mark.get("side", "")).replace("--", "-"))
        name = attribute(str(mark.get("name", ident)).replace("--", "-"))
        scope = str(mark.get("scope", "local")).strip().lower()
        record_type = "GLOBAL_ALIGNMENT" if scope == "global" else "LOCAL_ALIGNMENT"
        coupler_number = int(mark.get("coupler_number", 0))
        pattern = patterns_by_coupler.get(coupler_number, {})
        association = ""
        if scope == "local":
            gpf_name = attribute(
                str(pattern.get("name", mark.get("gpf", pattern_name)))
            )
            association = (
                f' coupler="{coupler_number}" '
                f'end="{attribute(str(mark.get("end", "single")))}" '
                f'slab_layer="{int(mark.get("slab_layer", edge_coupler_slab_layer(max(1, coupler_number))))}" '
                f'gpf="{gpf_name}"'
            )
        mark_lines.append(
            f"      <!-- {record_type} "
            f'ident="{ident}" uid="{int(mark.get("uid", 0))}" side="{side}" '
            f'coord_um="{_number(float(mark["x"]))},{_number(float(mark["y"]))}" '
            f'size_um="{_number(float(mark.get("size_um", 50.0)))}" '
            f'layer="{int(mark.get("layer", LOCAL_MARKER_LAYER))}" name="{name}"'
            f'{association} -->'
        )
    mark_metadata = ""
    if mark_lines:
        mark_metadata = (
            "      <!-- Max Layout alignment centers; layout coordinates in micrometres. -->\n"
            + "\n".join(mark_lines)
            + "\n"
        )

    mask_size = f"{_number(mask_width_mm)}mmx{_number(mask_height_mm)}mm"
    all_patterns = [{"name": pattern_name}] + edge_patterns
    color_lines = "\n".join(
        f'  <color rgb="0 255 0" pattern="{attribute(record["name"])}" num="{index}"/>'
        for index, record in enumerate(all_patterns, start=1)
    )
    pattern_lines: list[str] = []
    for record in all_patterns:
        if int(record.get("coupler_number", 0)) > 0:
            pattern_lines.append(
                "      <!-- EDGE_COUPLER_PATTERN "
                f'coupler="{int(record["coupler_number"])}" '
                f'uid="{int(record.get("uid", 0))}" '
                f'end="{attribute(str(record.get("end", "single")))}" '
                f'slab_layer="{int(record["slab_layer"])}" '
                f'local_marker_layer="{int(record["local_marker_layer"])}" '
                f'local_marker_ident="{attribute(record["local_marker_ident"])}" -->'
            )
        pattern_lines.extend(
            [
                f'      <pattern name="{attribute(record["name"])}">',
                '        <position coord="0,0"/>',
                f'        <beam dose="{_number(dose)}" defocus="#0" name="{attribute(beam_name)}"/>',
                "      </pattern>",
            ]
        )
    patterns_xml = "\n".join(pattern_lines)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<cjob type="ebpg5200" version="v02_23">
{color_lines}
  <substrate substrate="mask">
    <mask size="{mask_size}"/>
    <color substrate="mask" rgb="200 200 200"/>
    <position coord="0,0"/>
    <exposure height="check" workinglevel="high" ht="100kV" name="{attribute(exposure_name)}">
      <position coord="0,0"/>
{mark_metadata}      <checks enabled="false"/>
{patterns_xml}
    </exposure>
  </substrate>
</cjob>
'''
