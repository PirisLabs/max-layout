"""Lumerical simulation metadata and self-contained notebook export.

Ports and monitors are explicit simulation-only objects placed from the editor
library.  The GDS builder ignores them, so they never become fabrication
geometry.
"""

from __future__ import annotations

from copy import deepcopy
import ast
import base64
import hashlib
import itertools
from pathlib import Path
from typing import Any, Iterable
import json
import math
import pprint
import re
import textwrap
import zlib

import numpy as np

from .constants import COMPONENT_SPECS, LAYER_NAME_MAP, SIMULATION_COMPONENT_KINDS
from .gds.build import component_geometry_arrays


MATERIAL_CHOICES = (
    "Air",
    "Si (Silicon) - Palik",
    "SiO2 (Glass) - Palik",
    "LiNbO3",
    "Al2O3",
    "Au (Gold) - CRC",
    "Al (Aluminium) - Palik",
    "Ag (Silver) - Palik",
)


LUMERICAL_SWEEP_MAX_RUNS = 300
LUMERICAL_SWEEP_WARNING_RUNS = 25

_SWEEP_PARAMETER_LABELS = {
    "pitch": "Pitch",
    "duty_cycle": "Filling factor",
    "fill_factor": "Filling factor",
    "N": "Number of grating periods",
    "wg_width": "Waveguide width",
    "width": "Width",
    "length": "Length",
    "radius": "Radius",
    "taper_L": "Taper length",
    "mmi_length": "MMI length",
    "mmi_width": "MMI width",
    "fiber_offset": "Fiber offset",
    "angle_theta": "Angle theta",
    "gap": "Gap",
}

_SWEEP_PARAMETER_CODES = {
    "pitch": "P",
    "duty_cycle": "F",
    "fill_factor": "F",
    "N": "N",
    "fiber_offset": "FO",
    "angle_theta": "TH",
}


def sweep_parameter_label(parameter: str) -> str:
    """Return a concise user-facing name while preserving the JSON key separately."""
    key = str(parameter)
    return _SWEEP_PARAMETER_LABELS.get(key, key.replace("_", " ").strip().title())


def sweep_parameter_code(parameter: str) -> str:
    """Short, filesystem-safe code used in per-point result names."""
    key = str(parameter)
    if key in _SWEEP_PARAMETER_CODES:
        return _SWEEP_PARAMETER_CODES[key]
    words = [word for word in re.split(r"[^A-Za-z0-9]+", key) if word]
    code = "".join(word[0].upper() for word in words) if len(words) > 1 else (words[0][:3].upper() if words else "X")
    return code or "X"


def sweepable_component_parameters(component: dict[str, Any]) -> list[dict[str, Any]]:
    """List scalar geometry parameters suitable for the fast Layer Builder sweep."""
    kind = str(component.get("kind", ""))
    params = component.get("params", {})
    apodized_fill_keys: set[str] = set()
    if str(params.get("fill_factors", "") or "").strip():
        apodized_fill_keys.update({"fill_factor", "duty_cycle"})
    if str(params.get("gc_fill_factors", "") or "").strip():
        apodized_fill_keys.add("gc_fill_factor")
    specs = COMPONENT_SPECS.get(kind, {})
    exact_exclusions = {
        "layer", "datatype", "tolerance", "h_total", "etch_depth",
        "waveguide_effective_index", "waveguide_neff_tolerance",
        "waveguide_mode_search_count", "waveguide_monitor_span_um",
        "waveguide_total_power_before_mode_um", "fdtd_port_offset_from_waveguide_end_um",
        "fdtd_port_clearance_um", "input_reference_before_taper_um",
    }
    excluded_fragments = (
        "wavelength", "mesh", "monitor", "port_", "fiber_", "datatype",
        "tolerance", "layer", "points", "resolution", "order", "label",
    )
    result: list[dict[str, Any]] = []
    for key, value in params.items():
        spec = specs.get(key, [])
        value_type = str(spec[0]) if spec else (
            "int" if isinstance(value, int) and not isinstance(value, bool)
            else "float" if isinstance(value, float) else ""
        )
        lower = str(key).lower()
        if value_type not in {"float", "int"} or isinstance(value, bool):
            continue
        if str(key) in apodized_fill_keys:
            # The tooth-by-tooth array is the geometry authority. Offering a
            # scalar fill axis here would imply that changing it changes the
            # grating, while the array would actually continue to win.
            continue
        if key in exact_exclusions or lower in exact_exclusions:
            continue
        if lower.endswith(("_layer", "_datatype", "_points", "_tolerance")):
            continue
        # ``port_sep`` is physical MMI branch spacing, not a simulation-port
        # placement control.  Keep it sweepable while excluding the actual
        # port/monitor settings covered by the broad fragment rules.
        if lower not in {"port_sep", "fiber_offset"} and any(
            fragment in lower for fragment in excluded_fragments
        ):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        result.append(
            {
                "parameter": str(key),
                "label": sweep_parameter_label(str(key)),
                "short_name": sweep_parameter_code(str(key)),
                "value_type": value_type,
                "nominal": int(value) if value_type == "int" else numeric,
            }
        )
    priority = {
        "pitch": 0, "duty_cycle": 1, "fill_factor": 1, "N": 2,
        "angle_theta": 3, "fiber_offset": 4,
        "mmi_length": 5, "mmi_width": 6, "taper_L": 7,
        "wg_width": 8, "width": 9, "length": 10, "radius": 11, "gap": 12,
    }
    return sorted(result, key=lambda item: (priority.get(item["parameter"], 100), item["label"]))


def normalize_lumerical_sweep_spec(
    component: dict[str, Any],
    axes: Iterable[dict[str, Any]],
    *,
    max_runs: int = LUMERICAL_SWEEP_MAX_RUNS,
) -> dict[str, Any]:
    """Validate and canonicalize explicit Cartesian sweep axes."""
    eligible = {item["parameter"]: item for item in sweepable_component_parameters(component)}
    normalized_axes: list[dict[str, Any]] = []
    seen: set[str] = set()
    point_count = 1
    for raw_axis in axes:
        parameter = str(raw_axis.get("parameter", "")).strip()
        if parameter not in eligible:
            raise ValueError(f"{parameter or 'Selected parameter'} is not available for a fast Lumerical geometry sweep.")
        if parameter in seen:
            raise ValueError(f"Sweep parameter {parameter!r} was selected more than once.")
        seen.add(parameter)
        metadata = eligible[parameter]
        raw_values = list(raw_axis.get("values", []))
        if len(raw_values) < 2:
            raise ValueError(f"{metadata['label']} needs at least two sweep values.")
        values: list[int | float] = []
        for raw_value in raw_values:
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{metadata['label']} contains a non-finite sweep value.")
            if parameter == "angle_theta" and not (0.0 <= value < 90.0):
                raise ValueError("Angle theta sweep values must be at least 0 and below 90 degrees.")
            if metadata["value_type"] == "int":
                if not value.is_integer():
                    raise ValueError(f"{metadata['label']} is an integer parameter; every value must be a whole number.")
                values.append(int(value))
            else:
                values.append(value)
        if len({float(value) for value in values}) != len(values):
            raise ValueError(f"{metadata['label']} contains duplicate values, which would waste GPU runs.")
        point_count *= len(values)
        if point_count > int(max_runs):
            raise ValueError(
                f"This Cartesian sweep contains {point_count} runs; the limit is {int(max_runs)}. "
                "Reduce the number of values or parameters."
            )
        normalized_axes.append({**metadata, "values": values})
    if not normalized_axes:
        raise ValueError("Select at least one parameter to sweep.")
    return {
        "version": 1,
        "component_uid": int(component.get("uid", 0)),
        "component_kind": str(component.get("kind", "")),
        "combination_mode": "cartesian",
        "axes": normalized_axes,
        "point_count": point_count,
        "save_each_fsp": False,
    }


def expand_lumerical_sweep_points(spec: dict[str, Any]) -> list[dict[str, int | float]]:
    """Return the stable Cartesian product encoded by a normalized sweep spec."""
    axes = list(spec.get("axes", []))
    names = [str(axis["parameter"]) for axis in axes]
    return [
        dict(zip(names, values))
        for values in itertools.product(*(list(axis["values"]) for axis in axes))
    ]


def apply_lumerical_sweep_values(
    component: dict[str, Any], values: dict[str, int | float]
) -> dict[str, Any]:
    """Apply one point to a temporary component used for notebook preflight.

    A tooth-by-tooth apodization array is authoritative.  A scalar filling-
    factor sweep is rejected instead of silently erasing that project data.
    """
    params = component.setdefault("params", {})
    selected = {str(parameter) for parameter in values}
    conflicts: list[str] = []
    if selected.intersection({"fill_factor", "duty_cycle"}) and str(
        params.get("fill_factors", "") or ""
    ).strip():
        conflicts.append("fill_factors")
    if "gc_fill_factor" in selected:
        if str(params.get("gc_fill_factors", "") or "").strip():
            conflicts.append("gc_fill_factors")
        elif str(params.get("fill_factors", "") or "").strip():
            conflicts.append("fill_factors")
    if conflicts:
        raise ValueError(
            "A scalar filling-factor sweep cannot be applied while the "
            + " and ".join(conflicts)
            + " apodization array is non-empty. Clear the array or sweep a parameter such as pitch."
        )
    for parameter, value in values.items():
        params[str(parameter)] = value
    return component


STACK_PRESETS: dict[str, list[dict[str, Any]]] = {
    "TFLN on SiO2": [
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "SiO2 BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 5.0, "role": "background", "gds_layer": 0},
        {"name": "Exported TFLN cross-section", "material": "LiNbO3", "thickness_um": 0.4, "etch_depth_um": 0.2, "sidewall_angle_deg": 79.0, "role": "geometry", "gds_layers": [1, 2]},
        {"name": "SiO2 cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
        {"name": "Top air", "material": "Air", "thickness_um": 1.0, "role": "background", "gds_layer": 0},
        {"name": "Al2O3", "material": "Al2O3", "thickness_um": 0.0, "role": "geometry", "gds_layer": 1},
        {"name": "Metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layers": [4, 5]},
    ],
    "TFLN MMI (3 um SiO2)": [
        {"name": "Bottom SiO2", "material": "SiO2 (Glass) - Palik", "thickness_um": 3.0, "role": "background", "gds_layer": 0},
        {"name": "Exported TFLN cross-section", "material": "LiNbO3", "thickness_um": 0.4, "etch_depth_um": 0.2, "sidewall_angle_deg": 79.0, "role": "geometry", "gds_layers": [1, 2]},
        {"name": "SiO2 cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
        {"name": "Top air", "material": "Air", "thickness_um": 1.0, "role": "background", "gds_layer": 0},
        {"name": "Al2O3", "material": "Al2O3", "thickness_um": 0.0, "role": "geometry", "gds_layer": 1},
        {"name": "Metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layers": [4, 5]},
    ],
    "SOI": [
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "SiO2 BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "Exported cross-section", "material": "Si (Silicon) - Palik", "thickness_um": 0.22, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1, 2]},
        {"name": "SiO2 top cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
        {"name": "Al2O3", "material": "Al2O3", "thickness_um": 0.0, "role": "geometry", "gds_layer": 1},
        {"name": "Metal", "material": "Al (Aluminium) - Palik", "thickness_um": 0.0, "role": "geometry", "gds_layers": [4, 5]},
    ],
    "SOI grating coupler (Ansys)": [
        # The official Ansys file uses FDTD mesh accuracy 2 without layer
        # mesh-override objects.  A factor of zero means Automatic here.
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 3.0, "role": "background", "gds_layer": 0, "mesh_factor": 0.0, "mesh_order": 2},
        {"name": "SiO2 BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "mesh_factor": 0.0, "mesh_order": 2},
        {"name": "GC-SOI residual slab", "material": "Si (Silicon) - Palik", "thickness_um": 0.12, "etch_depth_um": 0.12, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1], "slab_extent": "geometry", "mesh_factor": 0.0, "mesh_order": 2},
        {"name": "GC-SOI upper silicon", "material": "Si (Silicon) - Palik", "thickness_um": 0.10, "etch_depth_um": 0.10, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [2], "slab_extent": "geometry", "mesh_factor": 0.0, "mesh_order": 2},
        {"name": "SiO2 TOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 0.7, "role": "background", "gds_layer": 0, "conformal": True, "mesh_factor": 0.0, "mesh_order": 3},
        {"name": "Top air", "material": "Air", "thickness_um": 0.7, "role": "background", "gds_layer": 0, "mesh_factor": 0.0, "mesh_order": 1},
    ],
    "Al2O3 on SiO2": [
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "SiO2", "material": "SiO2 (Glass) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "Exported cross-section", "material": "Al2O3", "thickness_um": 0.4, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1, 2]},
        {"name": "Top cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
        {"name": "TFLN", "material": "LiNbO3", "thickness_um": 0.0, "role": "geometry", "gds_layer": 1},
        {"name": "Metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layers": [4, 5]},
    ],
    "Custom / start empty": [
        {"name": "Layer 1", "material": "SiO2 (Glass) - Palik", "thickness_um": 0.0, "role": "background", "gds_layer": 0},
        {"name": "Exported cross-section", "material": "Si (Silicon) - Palik", "thickness_um": 0.0, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1, 2]},
        {"name": "Layer 3", "material": "LiNbO3", "thickness_um": 0.0, "role": "background", "gds_layer": 0},
        {"name": "Metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layers": [4, 5]},
    ],
}


def default_stack(preset: str = "TFLN on SiO2") -> list[dict[str, Any]]:
    """Return an independent editable copy of a material-stack preset."""
    stack = deepcopy(STACK_PRESETS.get(preset, STACK_PRESETS["TFLN on SiO2"]))
    for row in stack:
        row.setdefault("mesh_factor", 0.2)
        row.setdefault("mesh_order", 3 if bool(row.get("conformal", False)) else 2)
    return stack


def _cardinal_position(angle_deg: float) -> str:
    angle = int(round(float(angle_deg) / 90.0) * 90) % 360
    return {0: "Right", 90: "Top", 180: "Left", 270: "Bottom"}[angle]


def seed_simulation_ports(component: dict[str, Any], replace: bool = False) -> list[dict[str, Any]]:
    """Compatibility helper: automatic component-port generation is disabled."""
    existing = component.get("simulation_ports")
    if replace or not isinstance(existing, list):
        component["simulation_ports"] = []
    return component["simulation_ports"]


def simulation_port_global(component: dict[str, Any], port: dict[str, Any]) -> dict[str, Any]:
    """Transform a stored component-local simulation port to layout coordinates."""
    local = port.get("center", (0.0, 0.0))
    angle = math.radians(float(component.get("orientation_deg", 0.0)))
    c, s = math.cos(angle), math.sin(angle)
    x = float(component.get("x", 0.0)) + c * float(local[0]) - s * float(local[1])
    y = float(component.get("y", 0.0)) + s * float(local[0]) + c * float(local[1])
    result = deepcopy(port)
    result["center"] = [x, y]
    result["outward_orientation_deg"] = (
        float(component.get("orientation_deg", 0.0))
        + float(port.get("outward_orientation_deg", 0.0))
    ) % 360.0
    result["angle phi"] = (
        float(component.get("orientation_deg", 0.0)) + float(port.get("angle phi", port.get("outward_orientation_deg", 0.0)))
    ) % 360.0
    result["component_uid"] = int(component.get("uid", 0))
    result["component_kind"] = str(component.get("kind", ""))
    return result


def available_geometry_layers(components: Iterable[dict[str, Any]]) -> list[tuple[int, int]]:
    layers: set[tuple[int, int]] = set()
    for component in components:
        if component.get("kind") == "E-beam multipass" or component.get("kind") in SIMULATION_COMPONENT_KINDS:
            continue
        try:
            polygons, _ = component_geometry_arrays(component)
        except Exception:
            continue
        layers.update((int(layer), int(datatype)) for _, layer, datatype in polygons)
    return sorted(layers)


def _standalone_port(component: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(component.get("params", {}))
    params["center"] = [0.0, 0.0]
    plane_normal = str(params.get("plane normal", "X")).upper()
    pos = str(params.get("pos", "Right"))
    if plane_normal == "Y":
        params["outward_orientation_deg"] = 270.0 if pos == "Bottom" else 90.0
    else:
        params["outward_orientation_deg"] = 180.0 if pos == "Left" else 0.0
    params["domain"] = "optical"
    params["enabled"] = True
    params.setdefault("port geometry", "surface")
    params.setdefault("z_span_um", 2.0)
    params.setdefault("mode", "fundamental mode")
    result = simulation_port_global(component, params)
    if component.get("simulation_parent_uid") is not None:
        result["parent_component_uid"] = int(component["simulation_parent_uid"])
    if component.get("simulation_parent_port") is not None:
        result["parent_port_name"] = str(component["simulation_parent_port"])
    if plane_normal == "Z":
        result["plane normal"] = "Z"
    else:
        nearest = int(round(float(result["outward_orientation_deg"]) / 90.0) * 90) % 360
        result["plane normal"] = "X" if nearest in (0, 180) else "Y"
    return result


def _standalone_fiber_geometry(component: dict[str, Any]) -> dict[str, Any]:
    """Transform an editor fiber structure into simulation coordinates without creating a source."""
    params = deepcopy(component.get("params", {}))
    if bool(component.get("auto_placed", False)):
        params.setdefault("z reference", "top of SiO2 cladding")
    params["center"] = [0.0, 0.0]
    params["component_uid"] = int(component.get("uid", 0))
    params["component_kind"] = str(component.get("kind", "Fiber geometry"))
    if component.get("simulation_parent_uid") is not None:
        params["parent_component_uid"] = int(component["simulation_parent_uid"])
    params["name"] = str(params.get("name", "fiber"))
    params["angle phi"] = (
        float(component.get("orientation_deg", 0.0)) + float(params.get("angle phi", 0.0))
    ) % 360.0
    angle = math.radians(float(component.get("orientation_deg", 0.0)))
    local = params.get("center", (0.0, 0.0))
    params["center"] = [
        float(component.get("x", 0.0)) + math.cos(angle) * float(local[0]) - math.sin(angle) * float(local[1]),
        float(component.get("y", 0.0)) + math.sin(angle) * float(local[0]) + math.cos(angle) * float(local[1]),
    ]
    return params


def _standalone_gaussian_source(component: dict[str, Any]) -> dict[str, Any]:
    """Transform one movable editor Gaussian plane into solver coordinates."""
    params = deepcopy(component.get("params", {}))
    component_angle = float(component.get("orientation_deg", 0.0)) % 360.0
    params.update(
        {
            "name": str(params.get("name", "gaussian_source")),
            "center": [
                float(component.get("x", 0.0)),
                float(component.get("y", 0.0)),
            ],
            "component_uid": int(component.get("uid", 0)),
            "component_kind": "Gaussian source",
            "injection axis": "Z",
            "direction": "Backward",
            "angle phi": (
                component_angle + float(params.get("angle phi", 0.0))
            )
            % 360.0,
            # S polarization is perpendicular to the incidence/grating plane,
            # so this remains local TE for every in-plane device rotation.
            "polarization": "local TE",
            "polarization angle": 90.0,
        }
    )
    if component.get("simulation_parent_uid") is not None:
        params["parent_component_uid"] = int(component["simulation_parent_uid"])
    if component.get("simulation_parent_port") is not None:
        params["parent_port_name"] = str(component["simulation_parent_port"])
    return params


def _standalone_monitor(component: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(component.get("params", {}))
    local_normal = str(params.get("plane normal", "X")).upper()
    component_angle = float(component.get("orientation_deg", 0.0)) % 360.0
    if local_normal == "Z":
        global_normal = "Z"
        normal_angle = component_angle
    else:
        normal_angle = (component_angle + (90.0 if local_normal == "Y" else 0.0)) % 360.0
        nearest = int(round(normal_angle / 90.0) * 90) % 360
        global_normal = "X" if nearest in (0, 180) else "Y"
    legacy_span = max(0.0, float(params.get("span_um", 4.0)))
    has_explicit_x_span = "x span" in params
    has_explicit_y_span = "y span" in params
    local_x_span = max(0.0, float(params.get("x span", 0.0 if local_normal == "X" else legacy_span)))
    local_y_span = max(0.0, float(params.get("y span", 0.0 if local_normal == "Y" else legacy_span)))
    z_span = max(0.0, float(params.get("z span", params.get("z_span_um", 2.0))))
    if global_normal == "X":
        explicit_transverse = max(local_x_span, local_y_span)
        x_span = 0.0
        y_span = explicit_transverse if (has_explicit_x_span or has_explicit_y_span) else legacy_span
    elif global_normal == "Y":
        explicit_transverse = max(local_x_span, local_y_span)
        x_span = explicit_transverse if (has_explicit_x_span or has_explicit_y_span) else legacy_span
        y_span = 0.0
    else:
        x_span = local_x_span or legacy_span
        y_span = local_y_span or legacy_span
        z_span = 0.0
    params.update(
        {
            "monitor_kind": str(component.get("kind", "Power monitor")),
            "center": [float(component.get("x", 0.0)), float(component.get("y", 0.0))],
            "orientation_deg": normal_angle,
            "plane normal": global_normal,
            "x span": x_span,
            "y span": y_span,
            "z span": z_span,
            "component_uid": int(component.get("uid", 0)),
        }
    )
    if component.get("simulation_parent_uid") is not None:
        params["parent_component_uid"] = int(component["simulation_parent_uid"])
    if component.get("simulation_parent_port") is not None:
        params["parent_port_name"] = str(component["simulation_parent_port"])
    if component.get("grating_monitor_role") is not None:
        params["grating_monitor_role"] = str(component["grating_monitor_role"])
    return params


def _collect_export_data(
    components: list[dict[str, Any]],
    included_layers: set[tuple[int, int]],
    origin_um: Iterable[float] | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], list[float], list[str],
]:
    geometry: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    fiber_geometries: list[dict[str, Any]] = []
    gaussian_sources: list[dict[str, Any]] = []
    monitors: list[dict[str, Any]] = []
    warnings: list[str] = []
    grating_excitation_by_uid = {
        int(component.get("uid", 0)): str(
            component.get("params", {}).get("excitation_type", "fiber_mode")
        ).strip().lower()
        for component in components
        if str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"}
    }
    for component in components:
        component_kind = str(component.get("kind", ""))
        parent_uid = int(component.get("simulation_parent_uid", -1))
        parent_excitation = grating_excitation_by_uid.get(parent_uid)
        component_plane_normal = str(
            component.get("params", {}).get("plane normal", "X")
        ).upper()
        is_parent_fiber_object = (
            component_kind in {"Fiber geometry", "Fiber port", "Fiber-axis FDTD port"}
            or (
                component_kind == "FDTD port"
                and component_plane_normal == "Z"
                and str(component.get("simulation_parent_port", "")) != "waveguide_point"
            )
        )
        if parent_excitation == "gaussian_beam" and is_parent_fiber_object:
            warnings.append(
                "UID %s: removed stale parent-owned %s because the grating uses Gaussian excitation."
                % (component.get("uid", 0), component_kind)
            )
            continue
        if parent_excitation == "fiber_mode" and component_kind == "Gaussian source":
            warnings.append(
                "UID %s: removed stale parent-owned Gaussian source because the grating uses fiber-mode excitation."
                % component.get("uid", 0)
            )
            continue
        if component_kind not in SIMULATION_COMPONENT_KINDS and component_kind != "E-beam multipass":
            try:
                polygons, _ = component_geometry_arrays(component)
                for index, (points, layer, datatype) in enumerate(polygons, start=1):
                    key = (int(layer), int(datatype))
                    if key not in included_layers:
                        continue
                    geometry.append(
                        {
                            "name": f"uid_{int(component.get('uid', 0))}_polygon_{index}",
                            "component_uid": int(component.get("uid", 0)),
                            "component_kind": str(component.get("kind", "")),
                            "layer": key[0],
                            "datatype": key[1],
                            "vertices_um": np.asarray(points, dtype=float).tolist(),
                        }
                    )
            except Exception as exc:
                warnings.append(f"UID {component.get('uid')}: geometry could not be embedded ({exc}).")
        if component_kind in {"FDTD port", "Fiber-axis FDTD port"}:
            ports.append(_standalone_port(component))
            continue
        if component_kind in {"Fiber geometry", "Fiber port"}:
            fiber_geometries.append(_standalone_fiber_geometry(component))
            if component_kind == "Fiber port":
                warnings.append(
                    f"UID {component.get('uid')}: legacy combined Fiber port was converted to fiber geometry only; "
                    "place a standard Fiber-axis FDTD port through it."
                )
            continue
        if component_kind == "Gaussian source":
            gaussian_sources.append(_standalone_gaussian_source(component))
            continue
        if component_kind in {"Power monitor", "Mode expansion monitor", "Field profile monitor"}:
            monitors.append(_standalone_monitor(component))
            continue
        if component.get("simulation_ports"):
            warnings.append(
                f"UID {component.get('uid')}: legacy embedded simulation ports were ignored; "
                "place ports explicitly from the Ports & monitors library."
            )

    # Old editor builds could leave two parent-owned copies of the automatic
    # incident-power plane after switching source families.  Their semantic
    # role is a singleton, so retain one deterministic copy and do not build a
    # second overlapping DFT monitor in the notebook.
    duplicate_monitor_ids: set[int] = set()
    for parent_uid in grating_excitation_by_uid:
        matching = [
            monitor for monitor in monitors
            if int(monitor.get("parent_component_uid", -1)) == parent_uid
            and str(monitor.get("monitor_kind", "")) == "Power monitor"
            and str(monitor.get("parent_port_name", "")) == "fiber_input_power"
        ]
        if len(matching) <= 1:
            continue
        ordered = sorted(matching, key=lambda monitor: int(monitor.get("component_uid", 0)))
        duplicate_monitor_ids.update(
            int(monitor.get("component_uid", -1)) for monitor in ordered[1:]
        )
        warnings.append(
            "Grating UID %s: removed %d duplicate automatic incident input-power monitor(s)."
            % (parent_uid, len(ordered) - 1)
        )
    if duplicate_monitor_ids:
        monitors = [
            monitor for monitor in monitors
            if int(monitor.get("component_uid", -1)) not in duplicate_monitor_ids
        ]

    if geometry:
        all_points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in geometry])
        minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
        origin = (
            np.asarray(list(origin_um), dtype=float)
            if origin_um is not None else 0.5 * (minimum + maximum)
        )
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("The fixed Lumerical simulation origin must contain two finite XY values")
        bbox = [float(minimum[0] - origin[0]), float(minimum[1] - origin[1]),
                float(maximum[0] - origin[0]), float(maximum[1] - origin[1])]
        for item in geometry:
            item["vertices_um"] = (np.asarray(item["vertices_um"], dtype=float) - origin).tolist()
        for port in ports:
            port["center"] = [float(port["center"][0] - origin[0]), float(port["center"][1] - origin[1])]
        for monitor in monitors:
            monitor["center"] = [float(monitor["center"][0] - origin[0]), float(monitor["center"][1] - origin[1])]
        for fiber in fiber_geometries:
            fiber["center"] = [float(fiber["center"][0] - origin[0]), float(fiber["center"][1] - origin[1])]
        for source in gaussian_sources:
            source["center"] = [float(source["center"][0] - origin[0]), float(source["center"][1] - origin[1])]
    else:
        centers = (
            [port.get("center", (0.0, 0.0)) for port in ports]
            + [fiber.get("center", (0.0, 0.0)) for fiber in fiber_geometries]
            + [source.get("center", (0.0, 0.0)) for source in gaussian_sources]
            + [monitor.get("center", (0.0, 0.0)) for monitor in monitors]
        )
        if centers:
            points = np.asarray(centers, dtype=float)
            origin = (
                np.asarray(list(origin_um), dtype=float)
                if origin_um is not None else 0.5 * (points.min(axis=0) + points.max(axis=0))
            )
            if origin.shape != (2,) or not np.all(np.isfinite(origin)):
                raise ValueError("The fixed Lumerical simulation origin must contain two finite XY values")
            for port in ports:
                port["center"] = [float(port["center"][0] - origin[0]), float(port["center"][1] - origin[1])]
            for monitor in monitors:
                monitor["center"] = [float(monitor["center"][0] - origin[0]), float(monitor["center"][1] - origin[1])]
            for fiber in fiber_geometries:
                fiber["center"] = [float(fiber["center"][0] - origin[0]), float(fiber["center"][1] - origin[1])]
            for source in gaussian_sources:
                source["center"] = [float(source["center"][0] - origin[0]), float(source["center"][1] - origin[1])]
            shifted = points - origin
            bbox = [float(shifted[:, 0].min() - 1.0), float(shifted[:, 1].min() - 1.0),
                    float(shifted[:, 0].max() + 1.0), float(shifted[:, 1].max() + 1.0)]
        else:
            origin = (
                np.asarray(list(origin_um), dtype=float)
                if origin_um is not None else np.array([0.0, 0.0])
            )
            bbox = [-1.0, -1.0, 1.0, 1.0]
        warnings.append(
            "No physical device polygons were selected. This notebook contains only ports/monitors and the background stack; "
            "choose a device-containing scope before solving a component response."
        )
    return (
        geometry, ports, fiber_geometries, gaussian_sources, monitors, bbox,
        warnings + [f"Layout origin moved by ({origin[0]:.6g}, {origin[1]:.6g}) µm for simulation."],
    )


def _repair_missing_grating_gaussian_sources(
    components: list[dict[str, Any]],
    gaussian_sources: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Recover the automatic Gaussian source omitted by an older project.

    Early Gaussian-excitation projects could retain the input-power monitor
    while losing their automatic source during the fiber-to-Gaussian switch.
    The monitor still records the beam axis and stack-relative source plane,
    so the missing source can be reconstructed without guessing its optical
    placement.  This keeps old projects simulatable and restores their CE
    analysis/plots on the next notebook export.
    """
    sources_by_parent = {
        int(source.get("parent_component_uid", -1))
        for source in gaussian_sources
    }
    for component in components:
        if str(component.get("kind", "")) not in {"Grating coupler", "GC-SOI"}:
            continue
        params = component.get("params", {})
        if str(params.get("excitation_type", "fiber_mode")).strip().lower() != "gaussian_beam":
            continue
        parent_uid = int(component.get("uid", 0))
        if parent_uid in sources_by_parent:
            continue
        input_monitors = [
            monitor for monitor in monitors
            if int(monitor.get("parent_component_uid", -1)) == parent_uid
            and str(monitor.get("monitor_kind", "")) == "Power monitor"
            and str(monitor.get("plane normal", "Z")).upper() == "Z"
            and (
                str(monitor.get("parent_port_name", "")) == "fiber_input_power"
                or str(monitor.get("fiber plane role", "")).strip().lower()
                == "input power measurement"
            )
        ]
        if len(input_monitors) != 1:
            continue
        input_monitor = input_monitors[0]
        theta_deg = float(params.get("angle_theta", input_monitor.get("angle theta", 0.0)))
        phi_deg = float(component.get("orientation_deg", input_monitor.get("orientation_deg", 0.0))) % 360.0
        below_source_um = max(
            0.001,
            float(params.get("fiber_power_monitor_below_source_um", 0.1)),
        )
        lateral_um = below_source_um * math.tan(math.radians(theta_deg))
        phi_rad = math.radians(phi_deg)
        monitor_center = np.asarray(input_monitor.get("center", (0.0, 0.0)), dtype=float)
        source_center = monitor_center + lateral_um * np.asarray(
            [math.cos(phi_rad), math.sin(phi_rad)]
        )
        gaussian_sources.append(
            {
                "name": f"uid_{parent_uid}_gaussian_source",
                "center": [float(source_center[0]), float(source_center[1])],
                "component_uid": parent_uid,
                "component_kind": "Gaussian source",
                "parent_component_uid": parent_uid,
                "parent_port_name": "gaussian_source",
                "injection axis": "Z",
                "direction": "Backward",
                "angle theta": theta_deg,
                "angle phi": phi_deg,
                "polarization": "local TE",
                "polarization angle": 90.0,
                "waist radius_um": max(
                    0.001, float(params.get("gaussian_waist_radius_um", 4.5))
                ),
                "distance from waist_um": float(
                    params.get("gaussian_distance_from_waist_um", 0.0)
                ),
                "span_um": max(
                    0.001, float(params.get("gaussian_source_span_um", 20.0))
                ),
                "amplitude": 1.0,
                "multifrequency beam calculation": True,
                "frequency points": max(
                    1, int(params.get("gaussian_multifrequency_points", 5))
                ),
                "input monitor span scale": max(
                    1.0,
                    float(params.get("gaussian_input_monitor_span_scale", 1.2)),
                ),
                "z reference": str(input_monitor.get("z reference", "top of stack")),
                "distance_um": float(input_monitor.get("distance_um", 0.0))
                + below_source_um,
            }
        )
        sources_by_parent.add(parent_uid)
        warnings.append(
            "Repaired the missing automatic Gaussian source for grating UID %s during export; "
            "refresh and save the project to persist it in the editor." % parent_uid
        )


def _apply_authoritative_grating_angles(
    components: list[dict[str, Any]],
    ports: list[dict[str, Any]],
    fiber_geometries: list[dict[str, Any]],
    gaussian_sources: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Make each grating parent's ``angle_theta`` the sole tilt authority.

    Automatically placed fiber geometry, source, and passive measurement
    objects are convenient editable views of the parent component.  They are
    not independent simulation inputs.  Re-applying the parent value during
    export prevents an old companion object's default angle from silently
    disagreeing with the project JSON.
    """
    canonical_by_uid: dict[int, float] = {}
    for component in components:
        if str(component.get("kind", "")) not in {"Grating coupler", "GC-SOI"}:
            continue
        params = component.setdefault("params", {})
        if "angle_theta" in params:
            theta_deg = float(params["angle_theta"])
        elif "fiber_tilt_deg" in params:
            theta_deg = float(params.pop("fiber_tilt_deg"))
            params["angle_theta"] = theta_deg
            warnings.append(
                "Migrated legacy fiber_tilt_deg to the authoritative parent angle_theta."
            )
        else:
            matching_values = [
                float(item["angle theta"])
                for item in [*fiber_geometries, *gaussian_sources, *ports, *monitors]
                if int(item.get("parent_component_uid", -1))
                == int(component.get("uid", 0))
                and "angle theta" in item
            ]
            if not matching_values:
                raise ValueError(
                    "Grating component UID %s has no authoritative angle_theta"
                    % component.get("uid", 0)
                )
            theta_deg = matching_values[0]
            params["angle_theta"] = theta_deg
            warnings.append(
                "Migrated an older grating companion tilt to the authoritative parent angle_theta."
            )
        if not math.isfinite(theta_deg) or theta_deg < 0.0 or theta_deg >= 90.0:
            raise ValueError(
                "Grating component UID %s angle_theta must be at least 0 and below 90 degrees"
                % component.get("uid", 0)
            )
        canonical_by_uid[int(component.get("uid", 0))] = theta_deg

    for item in [*fiber_geometries, *gaussian_sources, *ports, *monitors]:
        parent_uid = int(item.get("parent_component_uid", -1))
        if parent_uid not in canonical_by_uid:
            continue
        theta_deg = canonical_by_uid[parent_uid]
        previous = item.get("angle theta")
        item["angle theta"] = theta_deg
        if previous is not None and abs(float(previous) - theta_deg) > 1e-9:
            warnings.append(
                "Synchronized %s angle theta from %.6g° to parent angle_theta %.6g°."
                % (item.get("name", "simulation object"), float(previous), theta_deg)
            )


def _stack_vertical_levels(stack: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Return device top, stack top, and upper-silica top/center."""
    active = [row for row in stack if float(row.get("thickness_um", 0.0)) > 0.0]
    if not active:
        return 0.0, 0.0, 0.0, 0.0
    anchor_index = next(
        (index for index, row in enumerate(active) if str(row.get("role", "background")).lower() == "geometry"),
        len(active) // 2,
    )
    anchor_thickness = float(active[anchor_index].get("thickness_um", 0.0))
    ranges: list[tuple[dict[str, Any], float, float] | None] = [None] * len(active)
    ranges[anchor_index] = (active[anchor_index], -0.5 * anchor_thickness, 0.5 * anchor_thickness)
    cursor = -0.5 * anchor_thickness
    for index in range(anchor_index - 1, -1, -1):
        thickness = float(active[index].get("thickness_um", 0.0))
        ranges[index] = (active[index], cursor - thickness, cursor)
        cursor -= thickness
    cursor = 0.5 * anchor_thickness
    for index in range(anchor_index + 1, len(active)):
        thickness = float(active[index].get("thickness_um", 0.0))
        ranges[index] = (active[index], cursor, cursor + thickness)
        cursor += thickness
    resolved = [entry for entry in ranges if entry is not None]
    geometry_tops = [
        z1 for row, _z0, z1 in resolved
        if str(row.get("role", "background")).lower() == "geometry"
    ]
    device_top = max(geometry_tops, default=0.0)
    stack_top = resolved[-1][2]
    silica_rows = []
    for row, z0, z1 in resolved:
        label = (str(row.get("name", "")) + " " + str(row.get("material", ""))).lower()
        if ("sio2" in label or "silica" in label or "glass" in label) and z1 >= device_top - 1e-12:
            silica_rows.append((bool(row.get("conformal", False)), float(z0), float(z1)))
    candidates = [entry for entry in silica_rows if entry[0]] or silica_rows
    selected = max(candidates, key=lambda entry: entry[2]) if candidates else (False, device_top, device_top)
    silica_top = selected[2]
    silica_center = 0.5 * (selected[1] + selected[2])
    return float(device_top), float(stack_top), float(silica_top), float(silica_center)


def _item_vertical_reference(item: dict[str, Any], levels: tuple[float, float, float, float]) -> float:
    device_top, stack_top, silica_top, silica_center = levels
    reference = str(item.get("z reference", "device top")).strip().lower()
    if reference in {"center of sio2 cladding", "center of silica cladding", "cladding center"}:
        return silica_center
    if reference in {"top of sio2 cladding", "top of silica cladding", "top cladding"}:
        return silica_top
    if reference == "top of stack":
        return stack_top
    return device_top


def _synchronize_fiber_port_parameters(
    ports: list[dict[str, Any]],
    fiber_geometries: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
    material_stack: list[dict[str, Any]],
) -> None:
    """Place tilted source/measurement planes on the matching fiber axis.

    A fiber's editor center is its bottom-center contact point on the cladding.
    A source or monitor at another Z plane therefore needs a lateral shift of
    ``delta_z * tan(theta)`` to remain concentric with the tilted core.
    """
    levels = _stack_vertical_levels(material_stack)

    def matching_fiber(item: dict[str, Any]) -> dict[str, Any] | None:
        if not fiber_geometries:
            return None
        parent_uid = int(item.get("parent_component_uid", -1))
        matching = [
            fiber for fiber in fiber_geometries
            if int(fiber.get("parent_component_uid", -2)) == parent_uid
        ]
        candidates = matching or fiber_geometries
        item_center = np.asarray(item.get("center", (0.0, 0.0)), dtype=float)
        return min(
            candidates,
            key=lambda fiber: float(
                np.linalg.norm(np.asarray(fiber.get("center", (0.0, 0.0)), dtype=float) - item_center)
            ),
        )

    def align_plane(item: dict[str, Any], fiber: dict[str, Any]) -> None:
        if "angle theta" not in fiber:
            raise ValueError(
                "Fiber geometry %s has no angle theta; its grating parent angle_theta must be exported"
                % fiber.get("name", "fiber")
            )
        theta_deg = float(fiber["angle theta"])
        phi_deg = float(fiber.get("angle phi", item.get("angle phi", 0.0)))
        item["angle theta"] = theta_deg
        item["angle phi"] = phi_deg
        core_index = float(fiber.get("core index", 1.44427))
        cladding_index = float(fiber.get("cladding index", core_index))
        item["fiber target neff"] = 0.5 * (core_index + cladding_index)
        if not bool(item.get("align to fiber axis", True)):
            return
        fiber_bottom_z = _item_vertical_reference(fiber, levels) + float(fiber.get("distance_um", 0.0))
        plane_z = _item_vertical_reference(item, levels) + float(item.get("distance_um", 0.0))
        axial_height_um = plane_z - fiber_bottom_z
        lateral_um = axial_height_um * math.tan(math.radians(theta_deg))
        phi_rad = math.radians(phi_deg)
        bottom_center = np.asarray(fiber.get("center", (0.0, 0.0)), dtype=float)
        axis_center = bottom_center + lateral_um * np.asarray([math.cos(phi_rad), math.sin(phi_rad)])
        item["center"] = [float(axis_center[0]), float(axis_center[1])]
        item["fiber bottom center_um"] = [float(bottom_center[0]), float(bottom_center[1])]
        item["fiber axis height_um"] = float(axial_height_um)

    for port in ports:
        if str(port.get("plane normal", "X")).upper() != "Z":
            continue
        fiber = matching_fiber(port)
        if fiber is None:
            continue
        theta_deg = float(fiber["angle theta"])
        phi_deg = float(fiber.get("angle phi", port.get("angle phi", 0.0)))
        previous_theta = float(port.get("angle theta", theta_deg))
        previous_phi = float(port.get("angle phi", phi_deg))
        core_diameter_um = max(1e-6, float(fiber.get("core diameter_um", 9.0)))
        align_plane(port, fiber)
        port["rotation offset_um"] = 4.0 * core_diameter_um * math.tan(math.radians(theta_deg))
        if abs(previous_theta - theta_deg) > 1e-9 or abs(previous_phi - phi_deg) > 1e-9:
            warnings.append(
                "Fiber-axis port %s was synchronized to fiber %s: theta %.6g°, phi %.6g°."
                % (port.get("name", ""), fiber.get("name", ""), theta_deg, phi_deg)
            )

    for monitor in monitors:
        if str(monitor.get("plane normal", "X")).upper() != "Z" or not bool(monitor.get("align to fiber axis", False)):
            continue
        fiber = matching_fiber(monitor)
        if fiber is not None:
            align_plane(monitor, fiber)
            if str(monitor.get("fiber plane role", "")).strip().lower() == "input power measurement":
                parent_uid = int(monitor.get("parent_component_uid", -1))
                source_ports = [
                    port for port in ports
                    if int(port.get("parent_component_uid", -2)) == parent_uid
                    and str(port.get("plane normal", "X")).upper() == "Z"
                    and str(port.get("fiber plane role", "source")).strip().lower() == "source"
                ]
                source_span_um = max(
                    [float(port.get("span_um", 20.0)) for port in source_ports]
                    or [float(monitor.get("x span", monitor.get("y span", 20.0)))]
                )
                projected_span_um = source_span_um / max(
                    math.cos(math.radians(float(fiber["angle theta"]))), 1e-3
                )
                monitor["x span"] = projected_span_um
                monitor["y span"] = projected_span_um
                monitor["expected propagation sign"] = -1.0


def _synchronize_gaussian_source_parameters(
    gaussian_sources: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
    material_stack: list[dict[str, Any]],
) -> None:
    """Keep a Gaussian source and its horizontal Pin monitor on one beam axis."""
    levels = _stack_vertical_levels(material_stack)
    for source in gaussian_sources:
        parent_uid = int(source.get("parent_component_uid", -1))
        theta_deg = float(source.get("angle theta", 0.0))
        phi_deg = float(source.get("angle phi", 0.0)) % 360.0
        if not math.isfinite(theta_deg) or theta_deg < 0.0 or theta_deg >= 90.0:
            raise ValueError(
                "Gaussian source %s angle theta must be at least 0 and below 90 degrees"
                % source.get("name", "gaussian_source")
            )
        source["injection axis"] = "Z"
        source["direction"] = "Backward"
        source["polarization"] = "local TE"
        source["polarization angle"] = 90.0
        source["angle phi"] = phi_deg
        waist_um = float(source.get("waist radius_um", 4.5))
        span_um = float(source.get("span_um", 20.0))
        if not math.isfinite(waist_um) or waist_um <= 0.0:
            raise ValueError("Gaussian waist radius must be positive")
        if not math.isfinite(span_um) or span_um <= 2.0 * waist_um:
            warnings.append(
                "Gaussian source %s span %.6g um is not larger than its %.6g um "
                "1/e^2-power diameter; the injected beam may be clipped."
                % (source.get("name", "gaussian_source"), span_um, 2.0 * waist_um)
            )
        source_z_um = _item_vertical_reference(source, levels) + float(
            source.get("distance_um", 0.0)
        )
        input_monitors = [
            monitor for monitor in monitors
            if int(monitor.get("parent_component_uid", -2)) == parent_uid
            and str(monitor.get("parent_port_name", "")) == "fiber_input_power"
            and str(monitor.get("monitor_kind", "")) == "Power monitor"
        ]
        for monitor in input_monitors:
            monitor_z_um = _item_vertical_reference(monitor, levels) + float(
                monitor.get("distance_um", 0.0)
            )
            axial_delta_um = source_z_um - monitor_z_um
            lateral_um = axial_delta_um * math.tan(math.radians(theta_deg))
            phi_rad = math.radians(phi_deg)
            source_center = np.asarray(source.get("center", (0.0, 0.0)), dtype=float)
            monitor_center = source_center - lateral_um * np.asarray(
                [math.cos(phi_rad), math.sin(phi_rad)]
            )
            projected_span_um = max(
                1.0, float(source.get("input monitor span scale", 1.2))
            ) * span_um / max(
                math.cos(math.radians(theta_deg)), 1e-3
            )
            monitor.update(
                {
                    "center": [float(monitor_center[0]), float(monitor_center[1])],
                    "angle theta": theta_deg,
                    "angle phi": phi_deg,
                    "align to fiber axis": True,
                    "x span": projected_span_um,
                    "y span": projected_span_um,
                    "z span": 0.0,
                    "expected propagation sign": -1.0,
                }
            )


def _normalize_grating_measurement_objects(
    ports: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Normalize legacy grating companions to the two-port measurement model.

    A grating simulation has exactly two modal FDTD ports: the tilted fiber
    source and the access-waveguide receiver.  Incident power is deliberately
    measured with a separate Z-normal power monitor below the source.  Older
    projects briefly represented that plane as a second fiber port; demote it
    here so an old JSON file cannot silently add a third modal port.
    """

    def is_fiber_measurement(item: dict[str, Any]) -> bool:
        role = str(item.get("fiber plane role", "")).strip().lower()
        parent_name = str(item.get("parent_port_name", "")).strip().lower()
        return parent_name == "fiber_input_power" or role in {
            "input power measurement",
            "passive fiber measurement",
            "fiber power measurement",
        }

    existing_monitor_parents = {
        int(monitor.get("parent_component_uid", -1))
        for monitor in monitors if is_fiber_measurement(monitor)
    }
    for legacy_port in list(ports):
        if not is_fiber_measurement(legacy_port):
            continue
        parent_uid = int(legacy_port.get("parent_component_uid", -1))
        if parent_uid not in existing_monitor_parents:
            source_span_um = max(1e-6, float(legacy_port.get("span_um", 20.0)))
            theta_deg = float(legacy_port.get("angle theta", 0.0))
            span_um = source_span_um / max(math.cos(math.radians(theta_deg)), 1e-3)
            monitor = deepcopy(legacy_port)
            for key in (
                "component_kind", "domain", "enabled", "port geometry",
                "outward_orientation_deg", "z_span_um", "dir", "mode",
                "mode number", "polarization", "candidate mode numbers",
                "mode degeneracy tolerance", "minimum local TE fraction",
                "rotation offset_um", "order", "loc", "pos",
            ):
                monitor.pop(key, None)
            monitor.update(
                {
                    "monitor_kind": "Power monitor",
                    "monitor geometry": "surface",
                    "plane normal": "Z",
                    "orientation_deg": float(
                        legacy_port.get("outward_orientation_deg", 0.0)
                    ),
                    "x span": span_um,
                    "y span": span_um,
                    "z span": 0.0,
                    "fiber plane role": "input power measurement",
                    "parent_port_name": "fiber_input_power",
                    "align to fiber axis": True,
                    # A Backward Z-axis fiber source propagates toward -Z, so
                    # a Z-normal monitor reports negative signed flux.
                    "expected propagation sign": -1.0,
                }
            )
            monitors.append(monitor)
            existing_monitor_parents.add(parent_uid)
            warnings.append(
                "Converted legacy passive fiber port %s to the non-modal input-power monitor."
                % monitor.get("name", "fiber_input_power")
            )
        ports.remove(legacy_port)

    # Restore the access-waveguide receiver when loading projects exported
    # during the short-lived mode-expansion-only implementation.
    receiver_parents = {
        int(port.get("parent_component_uid", -1))
        for port in ports
        if str(port.get("plane normal", "X")).upper() in {"X", "Y"}
        and str(port.get("parent_port_name", "")) == "waveguide_point"
    }
    for legacy_monitor in list(monitors):
        if (
            str(legacy_monitor.get("monitor_kind", "")) != "Mode expansion monitor"
            or str(legacy_monitor.get("grating_monitor_role", ""))
            != "waveguide_mode_expansion"
        ):
            continue
        parent_uid = int(legacy_monitor.get("parent_component_uid", -1))
        if parent_uid not in receiver_parents:
            normal = str(legacy_monitor.get("plane normal", "X")).upper()
            transverse_span = (
                float(legacy_monitor.get("y span", legacy_monitor.get("span_um", 3.0)))
                if normal == "X"
                else float(legacy_monitor.get("x span", legacy_monitor.get("span_um", 3.0)))
            )
            receiver_name = f"uid_{parent_uid}_waveguide_point"
            receiver = {
                "name": receiver_name,
                "component_uid": int(legacy_monitor.get("component_uid", 0)),
                "component_kind": "FDTD port",
                "parent_component_uid": parent_uid,
                "parent_port_name": "waveguide_point",
                "domain": "optical",
                "enabled": True,
                "center": list(legacy_monitor.get("center", (0.0, 0.0))),
                "port geometry": "surface",
                "plane normal": normal,
                "outward_orientation_deg": float(
                    legacy_monitor.get("orientation_deg", 180.0)
                ),
                "distance_um": float(legacy_monitor.get("distance_um", 0.0)),
                "span_um": max(1e-6, transverse_span),
                "z_span_um": max(
                    1e-6, float(legacy_monitor.get("z span", 2.25))
                ),
                "dir": "Bidirectional",
                "mode": "fundamental TE mode",
                "mode number": 0,
                "polarization": "local TE",
                "target neff": float(legacy_monitor.get("target neff", 0.0)),
                "target neff strategy": str(
                    legacy_monitor.get(
                        "target neff strategy",
                        "automatic material-index midpoint",
                    )
                ),
                "neff tolerance": float(
                    legacy_monitor.get("neff tolerance", 0.3)
                ),
                "mode search count": max(
                    1, int(legacy_monitor.get("mode search count", 20))
                ),
                "order": 2,
                "loc": 0.5,
                "pos": "Left",
            }
            ports.append(receiver)
            receiver_parents.add(parent_uid)
            warnings.append(
                "Converted legacy grating mode-expansion monitor %s to passive waveguide receiver %s."
                % (legacy_monitor.get("name", "waveguide_mode"), receiver_name)
            )
        monitors.remove(legacy_monitor)


def _notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _quick_run_options_cell(configuration: dict[str, Any], *, workflow: str) -> str:
    """Put the remaining execution choices at the top of a notebook."""
    return (
        "# ==============================================================================\n"
        "# QUICK RUN OPTIONS — edit these before running any other cell.\n"
        "# ==============================================================================\n"
        f"RUN_SIMULATION = {bool(configuration.get('run_after_build', True))!r}\n"
        f"SHOW_GEOMETRY_PREVIEW = {bool(configuration.get('show_geometry_preview', True))!r}\n"
        f"SHOW_PORT_MODE_PREVIEW = {bool(configuration.get('show_port_mode_preview', True))!r}\n"
        f"RUN_GPU_SYSTEM_CHECK = {bool(configuration.get('run_gpu_system_check', False))!r}\n"
        f"HPC_PACK_DURATION_MINUTES = {int(configuration.get('hpc_pack_duration_minutes', 30))!r}\n"
        f"HPC_PACK_COUNT = {int(configuration.get('hpc_pack_count', 3))!r}\n"
        "# HPC_PACK_DURATION_MINUTES controls the roaming checkout below; edit it before section 2.\n"
        "# HPC_PACK_COUNT is the requested total. The H100 launcher overrides its default to 4.\n"
        "# One pre-solve inspection FSP and one solved/best FSP are always stored.\n"
        "# ==============================================================================\n"
        "print('Project-file saving is always enabled: inspection plus solved/best FSP.')\n\n"
        + _PIRIS_PATHS_CELL
    )


def _runtime_setup_source(builder_source: str) -> str:
    """Apply first-cell switches and enforce project-file persistence."""
    return (
        "\n# Apply first-cell switches. Project saving is mandatory.\n"
        "SETTINGS['run_after_build'] = bool(RUN_SIMULATION)\n"
        "SETTINGS['save_inspection_fsp'] = True\n"
        "SETTINGS['save_final_fsp'] = True\n"
        "SETTINGS['run_gpu_system_check'] = bool(RUN_GPU_SYSTEM_CHECK)\n"
        "print('Execution mode:', 'run' if SETTINGS['run_after_build'] else 'build only')\n"
        "print('Inspection and solved/best FSP saving: always enabled')\n"
    )


_PIRIS_PATHS_CELL = r'''# Piris Labs project paths (managed by the 3D Simulations launcher)
import os as _piris_os
import sys as _piris_sys
from pathlib import Path as _PirisPath

PIRIS_PROJECT_ROOT = _PirisPath(
    _piris_os.environ.get("PIRIS_PROJECT_ROOT", _PirisPath.cwd())
).expanduser().resolve()
PIRIS_SESSION_DIR = _PirisPath(
    _piris_os.environ.get("PIRIS_SESSION_DIR", _PirisPath.cwd())
).expanduser().resolve()
PIRIS_NOTEBOOK_DIR = PIRIS_PROJECT_ROOT / "Notebook"
PIRIS_FSP_DIR = PIRIS_PROJECT_ROOT / "fsp"
PIRIS_RESULTS_DIR = _PirisPath(
    _piris_os.environ.get("PIRIS_RESULTS_DIR", PIRIS_SESSION_DIR)
).expanduser().resolve()
PIRIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PIRIS_FSP_DIR.mkdir(parents=True, exist_ok=True)
_piris_os.environ["PIRIS_RESULTS_DIR"] = str(PIRIS_RESULTS_DIR)
_piris_lumerical = _PirisPath.home() / "lumerical"
if _piris_lumerical.is_dir() and str(_piris_lumerical) not in _piris_sys.path:
    _piris_sys.path.insert(0, str(_piris_lumerical))
_piris_os.chdir(PIRIS_RESULTS_DIR)
print("Project:", PIRIS_PROJECT_ROOT.name)
print("Session:", PIRIS_SESSION_DIR.name)
print("Results:", PIRIS_RESULTS_DIR)
print("FSP projects:", PIRIS_FSP_DIR)
'''


_LAMBDA_CONNECT_CELL = r'''import base64
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
sys.path.insert(0, os.path.expanduser("~/Desktop/lumerical"))
from lambda_remote import Lambda, _SSH, HOST

_remote_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", PIRIS_SESSION_DIR.name).strip("._") or "max_layout"
_remote_root = os.environ.get("PIRIS_LUMERICAL_REMOTE_WORK_ROOT", "").strip()
if not _remote_root:
    _root_probe = subprocess.run(
        _SSH + [HOST,
            "if test -d /lambda/nfs/piris-lumerical -a -w /lambda/nfs/piris-lumerical; "
            "then printf %s /lambda/nfs/piris-lumerical/projects/max_layout; "
            "else printf %s /home/ubuntu/.piris-launch/work/max_layout; fi"],
        capture_output=True, text=True, timeout=30,
    )
    if _root_probe.returncode != 0 or not _root_probe.stdout.strip():
        raise RuntimeError(
            "Could not determine a writable Lumerical work directory: "
            + (_root_probe.stdout + _root_probe.stderr)[-700:]
        )
    _remote_root = _root_probe.stdout.strip().splitlines()[-1]
REMOTE_WORK = _remote_root.rstrip("/") + "/" + _remote_slug
lam = Lambda(work=REMOTE_WORK)
print("Remote work:", REMOTE_WORK)


def _guard_remote_code(code, label):
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    ok_marker = "__MAX_LAYOUT_REMOTE_OK__"
    error_marker = "__MAX_LAYOUT_REMOTE_ERROR__"
    guarded = (
        "import base64 as _ml_b64, traceback as _ml_traceback, sys as _ml_sys\n"
        "try:\n"
        "    exec(compile(_ml_b64.b64decode(%r).decode('utf-8'), '<%s>', 'exec'))\n"
        "except Exception:\n"
        "    _ml_traceback.print_exc(file=_ml_sys.stdout)\n"
        "    _ml_sys.stdout.flush()\n"
        "    print(%r, flush=True)\n"
        "else:\n"
        "    print(%r, flush=True)\n"
    ) % (encoded, label.replace("'", "_"), error_marker, ok_marker)
    return guarded, ok_marker, error_marker


def _check_remote_output(output, label, ok_marker, error_marker, already_printed=False):
    visible = "\n".join(
        line for line in output.splitlines()
        if line.strip() not in {ok_marker, error_marker}
    ).strip()
    if visible and not already_printed:
        print(visible)
    if error_marker in output or ok_marker not in output:
        raise RuntimeError(label + " failed on Lambda. Run the final licence-release cell before retrying.\n" + visible[-3000:])
    return output


def run_remote_checked(code, label, timeout=1800):
    """Run one remote stage and surface errors hidden by Lambda.run's REPL wrapper."""
    guarded, ok_marker, error_marker = _guard_remote_code(code, label)
    output = lam.run(guarded, quiet=True, timeout=timeout)
    return _check_remote_output(output, label, ok_marker, error_marker)


def _read_live_optimization_rows(progress_file):
    """Read complete JSONL iteration records over Lambda's second SSH link."""
    ssh = getattr(lam, "_ssh", _SSH)
    host = getattr(lam, "host", HOST)
    command = "cat -- %s 2>/dev/null || true" % shlex.quote(str(progress_file))
    try:
        result = subprocess.run(
            ssh + [host, command], capture_output=True, text=True, timeout=20
        )
    except Exception:
        return []
    rows = []
    seen = set()
    for raw_line in result.stdout.splitlines():
        try:
            row = json.loads(raw_line)
            sequence = int(row["sequence"])
            if sequence in seen:
                continue
            seen.add(sequence)
            rows.append(row)
        except Exception:
            # A writer may be between bytes during the SSH read. The next poll
            # will see the complete record; reporting must never stop a solve.
            continue
    return sorted(rows, key=lambda row: int(row["sequence"]))


def _format_live_optimization_rows(rows):
    lines = []
    for row in rows:
        names = list(map(str, row.get("parameter_names", [])))
        parameters = dict(row.get("parameters", {}))
        parameter_text = ", ".join(
            "%s=%.9g" % (name, float(parameters[name]))
            for name in names if name in parameters
        )
        try:
            objective_text = "%.9g" % float(row["objective"])
        except Exception:
            objective_text = "unavailable"
        lines.append(
            "[%s iteration %d] objective=%s | %s"
            % (
                str(row.get("stage", "optimization")),
                int(row.get("iteration", 0)),
                objective_text,
                parameter_text,
            )
        )
    return lines


def _format_live_sweep_rows(rows, elapsed_seconds=None):
    """Format the latest sweep state without dumping one line per poll."""
    if not rows:
        return []
    latest = rows[-1]
    total = max(0, int(latest.get("total_count", 0)))
    completed = max(0, int(latest.get("completed_count", 0)))
    failed = max(0, int(latest.get("failed_count", 0)))
    processed = min(total, completed + failed) if total else completed + failed
    percent = 100.0 * processed / total if total else 0.0
    elapsed = (
        max(0.0, float(elapsed_seconds))
        if elapsed_seconds is not None
        else max(0.0, float(latest.get("elapsed_seconds", 0.0)))
    )
    eta_text = "estimating"
    if processed > 0 and total > processed:
        durations = []
        for row in rows:
            if str(row.get("status", "")) not in {"completed", "failed"}:
                continue
            try:
                duration = float(row["case_seconds"])
            except Exception:
                continue
            if duration > 0.0:
                durations.append(duration)
        mean_case_seconds = (
            sum(durations) / len(durations)
            if durations else elapsed / processed
        )
        eta_seconds = mean_case_seconds * (total - processed)
        eta_text = "%.1f min" % (eta_seconds / 60.0)
    elif total and processed >= total:
        eta_text = "0.0 min"

    values = dict(latest.get("values", {}))
    parameter_text = ", ".join(
        "%s=%.9g" % (str(name), float(value))
        for name, value in values.items()
    )
    status = str(latest.get("status", "running"))
    current = str(latest.get("display_label", "")).strip() or parameter_text or "preparing"
    lines = [
        "Sweep progress: %d/%d finished (%6.2f%%) | failed %d | elapsed %.1f min | ETA %s"
        % (processed, total, percent, failed, elapsed / 60.0, eta_text),
        "Current: %s | %s" % (status, current),
    ]

    terminal_rows = [
        row for row in rows
        if str(row.get("status", "")) in {"completed", "reused", "failed"}
    ]
    if terminal_rows:
        lines.append("Latest completed points:")
        for row in terminal_rows[-5:]:
            label_text = str(row.get("display_label", "")).strip()
            result_text = ""
            try:
                peak = float(row["peak_response"])
                wavelength_nm = float(row["peak_wavelength_nm"])
                result_text = " | peak %.6g at %.3f nm" % (peak, wavelength_nm)
            except Exception:
                pass
            lines.append(
                "- %d/%d %s: %s%s"
                % (
                    int(row.get("case_index", -1)) + 1,
                    total,
                    str(row.get("status", "completed")),
                    label_text,
                    result_text,
                )
            )
    return lines


def _solve_with_live_optimization_progress(
    code, label, poll, timeout, progress_file, progress_mode="optimization"
):
    """Run remotely while redrawing solver percentage plus streamed progress."""
    try:
        from IPython.display import clear_output
    except Exception:
        def clear_output(wait=False):
            pass

    ssh = getattr(lam, "_ssh", _SSH)
    host = getattr(lam, "host", HOST)
    work = str(getattr(lam, "work", REMOTE_WORK))
    cleanup = "rm -f %s/*_p0.log %s" % (
        shlex.quote(work), shlex.quote(str(progress_file))
    )
    subprocess.run(ssh + [host, cleanup], capture_output=True, text=True)

    done, result, error = threading.Event(), {}, {}

    def worker():
        try:
            result["out"] = lam.run(code, quiet=True, timeout=timeout)
        except Exception as exc:
            error["exception"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    last_screen = ""
    latest_rows = []
    while not done.is_set():
        percent = 0.0
        try:
            command = (
                "grep -ohE '[0-9.]+%% complete' %s/*_p0.log "
                "2>/dev/null | tail -1" % shlex.quote(work)
            )
            response = subprocess.run(
                ssh + [host, command], capture_output=True, text=True, timeout=20
            )
            token = response.stdout.strip().split("%")[0]
            if token:
                percent = float(token)
        except Exception:
            pass
        latest_rows = _read_live_optimization_rows(progress_file)
        if progress_mode == "sweep" and latest_rows:
            latest = latest_rows[-1]
            total = max(0, int(latest.get("total_count", 0)))
            processed = max(0, int(latest.get("completed_count", 0))) + max(
                0, int(latest.get("failed_count", 0))
            )
            if total:
                # Blend the current FDTD solver percentage into the completed
                # case count so the overall bar moves during a long point too.
                current_fraction = (
                    max(0.0, min(100.0, percent)) / 100.0
                    if str(latest.get("status", "")) == "running" else 0.0
                )
                percent = 100.0 * min(total, processed + current_fraction) / total
        count = int(max(0.0, min(100.0, percent)) / 2.0)
        screen_lines = [
            "%s [%s%s] %6.2f%%   %6.1f s"
            % (
                label,
                "#" * count,
                "-" * (50 - count),
                percent,
                time.monotonic() - started,
            )
        ]
        if progress_mode == "sweep":
            formatted_rows = _format_live_sweep_rows(
                latest_rows, elapsed_seconds=time.monotonic() - started
            )
            if formatted_rows:
                screen_lines.extend(["", *formatted_rows])
        else:
            formatted_rows = _format_live_optimization_rows(latest_rows)
            if formatted_rows:
                screen_lines.extend(["", "Completed optimization iterations:"])
                screen_lines.extend(formatted_rows)
        screen = "\n".join(screen_lines)
        if screen != last_screen:
            clear_output(wait=True)
            print(screen, flush=True)
            last_screen = screen
        time.sleep(float(poll))

    thread.join()
    latest_rows = _read_live_optimization_rows(progress_file)
    clear_output(wait=True)
    final_lines = [
        "%s [%s] 100.00%%   %.1f s  DONE"
        % (label, "#" * 50, time.monotonic() - started)
    ]
    if progress_mode == "sweep":
        formatted_rows = _format_live_sweep_rows(
            latest_rows, elapsed_seconds=time.monotonic() - started
        )
        if formatted_rows:
            final_lines.extend(["", *formatted_rows])
    else:
        formatted_rows = _format_live_optimization_rows(latest_rows)
        if formatted_rows:
            final_lines.extend(["", "Completed optimization iterations:"])
            final_lines.extend(formatted_rows)
    print("\n".join(final_lines), flush=True)
    if "exception" in error:
        raise error["exception"]
    output = result.get("out", "")
    if output.strip():
        print(output.rstrip())
    return output


def solve_remote_checked(
    code, label, timeout=21600, progress_file=None, progress_mode="optimization"
):
    """Keep the live progress display while still detecting a solver traceback."""
    guarded, ok_marker, error_marker = _guard_remote_code(code, label)
    if progress_file:
        output = _solve_with_live_optimization_progress(
            guarded, label=label, poll=5.0, timeout=timeout,
            progress_file=progress_file, progress_mode=progress_mode,
        )
    else:
        output = lam.solve(guarded, label=label, poll=5.0, timeout=timeout)
    try:
        return _check_remote_output(output, label, ok_marker, error_marker, already_printed=True)
    except RuntimeError as exc:
        log_reader = (
            "import glob, os\n"
            "_ml_logs = sorted(glob.glob(os.path.join(%r, '*_p0.log')))\n"
            "print('No Lumerical *_p0.log file was created.' if not _ml_logs else '')\n"
            "for _ml_log in _ml_logs:\n"
            "    print('--- ' + os.path.basename(_ml_log) + ' (last 160 lines) ---')\n"
            "    with open(_ml_log, 'r', encoding='utf-8', errors='replace') as _ml_stream:\n"
            "        print(''.join(_ml_stream.readlines()[-160:]))\n"
        ) % REMOTE_WORK
        try:
            diagnostics = lam.run(log_reader, quiet=True, timeout=120).strip()
        except Exception as diagnostic_exc:
            diagnostics = "Could not read the solver log: " + str(diagnostic_exc)
        if diagnostics:
            raise RuntimeError(str(exc) + "\n\nLumerical solver log:\n" + diagnostics[-12000:]) from None
        raise
'''


_LICENSE_CHECKOUT_CELL = r'''import json
import os
import subprocess
from lambda_remote import _SSH, HOST

LIC = "/opt/lumerical/v261/licensingclient/linx64"
HPC_PACK_NAME = "Ansys HPC Pack - Shared Web"
try:
    HPC_PACK_COUNT = int(HPC_PACK_COUNT)
except (NameError, TypeError, ValueError):
    raise ValueError("Set HPC_PACK_COUNT in cell 1 to a positive whole number") from None
_launcher_pack_count = int(os.environ.get("PIRIS_HPC_PACK_COUNT", "0") or 0)
if _launcher_pack_count > 0 and HPC_PACK_COUNT == 3:
    HPC_PACK_COUNT = _launcher_pack_count
if HPC_PACK_COUNT <= 0:
    raise ValueError("HPC_PACK_COUNT must be greater than zero")
try:
    HPC_PACK_DURATION_MINUTES = int(HPC_PACK_DURATION_MINUTES)
except (NameError, TypeError, ValueError):
    raise ValueError("Set HPC_PACK_DURATION_MINUTES in cell 1 to a positive whole number of minutes") from None
if HPC_PACK_DURATION_MINUTES <= 0:
    raise ValueError("HPC_PACK_DURATION_MINUTES must be greater than zero")
HPC_PACK_EXPIRY = "PT%dM" % HPC_PACK_DURATION_MINUTES


def _ansys_json_object(raw_output, label):
    """Extract and validate the LicensingSettings JSON object, ignoring warnings."""
    text = str(raw_output or "")
    decoder = json.JSONDecoder()
    objects = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "status" in value:
            objects.append(value)
    if not objects:
        raise RuntimeError(label + " returned no readable LicensingSettings JSON: " + text[-700:])
    value = objects[-1]
    if str(value.get("status", "")).upper() != "SUCCESS":
        raise RuntimeError(label + " failed: " + repr(value))
    return value


def _ansys_in_use(raw_output, label):
    value = _ansys_json_object(raw_output, label)
    usage = value.get("usage")
    if usage is None and "no products to display" in str(value.get("message", "")).casefold():
        usage = []
    if not isinstance(usage, list) or any(not isinstance(item, dict) for item in usage):
        raise RuntimeError(label + " returned an invalid usage list: " + repr(value))
    return usage


def _hpc_pack_count(usage, label):
    total = 0
    for item in usage:
        if str(item.get("name", "")) != HPC_PACK_NAME:
            continue
        if item.get("roaming") is not True:
            raise RuntimeError(label + " reported the HPC Pack as non-roaming: " + repr(item))
        count = item.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or int(count) != count or count < 0:
            raise RuntimeError(label + " reported an invalid HPC Pack count: " + repr(item))
        total += int(count)
    return total

# 1. seed the Ansys web sign-in from the shared token (no-op if already seeded)
r = subprocess.run(_SSH + [HOST,
    'if test -s ~/.ansys/ansysid/token.json; then echo "sign-in already seeded"; '
    'else test -s ~/remote-token.json || { echo "ERROR: ~/remote-token.json missing on the node"; exit 1; }; '
    f'ANSYS_LICENSING_WEB=1 {LIC}/ansyscl -WebLoginInput ~/remote-token.json || exit 1; '
    'echo "sign-in seeded from ~/remote-token.json"; fi; '
    f'{LIC}/LicensingSettings web shared enable --mode user >/dev/null 2>&1'],
    capture_output=True, text=True, timeout=180)
print((r.stdout + r.stderr).strip())
if r.returncode != 0:
    raise RuntimeError("Ansys web sign-in could not be seeded")

# 2. Query this host first so rerunning the cell never blindly reserves another 3.
_in_use_command = (
    f'{LIC}/LicensingSettings web shared products in-use '
    '--type roaming --mode user'
)
_before = subprocess.run(
    _SSH + [HOST, _in_use_command], capture_output=True, text=True, timeout=180
)
_before_out = (_before.stdout + _before.stderr).strip()
if _before.returncode != 0:
    raise RuntimeError("Pre-check of roaming HPC Packs failed: " + _before_out[-700:])
_existing_count = _hpc_pack_count(_ansys_in_use(_before_out, "HPC Pack pre-check"), "HPC Pack pre-check")

# 3. Bring the local roaming total to the requested count, using cell-1 settings.
_needed_count = max(0, HPC_PACK_COUNT - _existing_count)
if _needed_count:
    r = subprocess.run(_SSH + [HOST,
        f'{LIC}/LicensingSettings web shared products checkout '
        f'--name "{HPC_PACK_NAME}" --count {_needed_count} --expires "{HPC_PACK_EXPIRY}" '
        '--licenseModel "Shared Web" --mode user'],
        capture_output=True, text=True, timeout=180)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise RuntimeError("HPC Pack checkout command failed: " + out[-700:])
    _ansys_json_object(out, "HPC Pack checkout")
else:
    print("HPC Packs: existing local roaming reservation already satisfies %d packs" % HPC_PACK_COUNT)

_after = subprocess.run(
    _SSH + [HOST, _in_use_command], capture_output=True, text=True, timeout=180
)
_after_out = (_after.stdout + _after.stderr).strip()
if _after.returncode != 0:
    raise RuntimeError("Post-check of roaming HPC Packs failed: " + _after_out[-700:])
_verified_count = _hpc_pack_count(_ansys_in_use(_after_out, "HPC Pack post-check"), "HPC Pack post-check")
if _verified_count < HPC_PACK_COUNT:
    raise RuntimeError("HPC Pack checkout was not verified: %d of %d packs are visible" % (_verified_count, HPC_PACK_COUNT))
print("HPC Packs: %d roaming packs verified on this host (requested expiry %s)" % (_verified_count, HPC_PACK_EXPIRY))
'''


_BUILD_CELL = r'''import os
import json
import re
import shutil
import time
import uuid
import numpy as np
import lumapi

UM = 1e-6
GAUSSIAN_SOURCES = list(globals().get("GAUSSIAN_SOURCES", []))

# A notebook cell may be rerun after changing its first-cell options.  Close
# the previous CAD owner before creating/loading another one; otherwise the
# persistent Lambda Python process can accumulate hidden FDTD processes and
# make every subsequent build progressively slower.
_previous_fdtd = globals().get("fdtd")
if _previous_fdtd is not None:
    try:
        _previous_fdtd.close()
        print("Closed the previous live FDTD model before rebuilding.")
    except Exception as _previous_close_exc:
        print("Previous FDTD close warning:", str(_previous_close_exc)[:240])
    finally:
        globals().pop("fdtd", None)

os.makedirs(REMOTE_WORK, exist_ok=True)
REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")
os.makedirs(REMOTE_FSP_DIR, exist_ok=True)
os.chdir(REMOTE_WORK)
for _old_name in {
    os.path.basename(str(SETTINGS.get("project_file", "exported_component.fsp"))),
    "_max_layout_mode_seed.fsp",
    "_max_layout_runtime.fsp",
    "geometry_xyz_projections.png",
    "port_mode_Ex_Ey.png",
    "max_layout_results.npz",
    "max_layout_results.json",
    "summary.txt",
    "grating_response.png",
    "grating_analysis.npz",
    "mmi_splitting_ratio.png",
    "mmi_field_distribution.png",
    "mmi_analysis.npz",
}:
    for _old_path in (
        os.path.join(REMOTE_WORK, _old_name),
        os.path.join(REMOTE_FSP_DIR, _old_name),
    ):
        if os.path.isfile(_old_path):
            os.remove(_old_path)


def _active_stack(stack):
    """Thickness 0 means the layer is absent."""
    return [row for row in stack if float(row.get("thickness_um", 0.0)) > 0.0]


def _stack_z_ranges(stack):
    active = _active_stack(stack)
    if not active:
        return []
    anchor_index = next(
        (index for index, row in enumerate(active) if str(row.get("role", "background")) == "geometry"),
        len(active) // 2,
    )
    anchor_thickness = float(active[anchor_index]["thickness_um"])
    result = [None] * len(active)
    result[anchor_index] = (active[anchor_index], -0.5 * anchor_thickness, 0.5 * anchor_thickness)
    cursor = -0.5 * anchor_thickness
    for index in range(anchor_index - 1, -1, -1):
        thickness = float(active[index]["thickness_um"])
        result[index] = (active[index], cursor - thickness, cursor)
        cursor -= thickness
    cursor = 0.5 * anchor_thickness
    for index in range(anchor_index + 1, len(active)):
        thickness = float(active[index]["thickness_um"])
        result[index] = (active[index], cursor, cursor + thickness)
        cursor += thickness
    return result


def _add_required_materials(fdtd):
    """Create a dispersive anisotropic LiNbO3 material when the stack requires it."""
    active_materials = {str(row.get("material", "")) for row in _active_stack(MATERIAL_STACK)}
    if "Air" in active_materials and not fdtd.materialexists("Air"):
        air_id = fdtd.addmaterial("Dielectric")
        fdtd.setmaterial(air_id, "name", "Air")
        fdtd.setmaterial("Air", "Refractive Index", 1.0)
    if "LiNbO3" not in active_materials:
        return
    if fdtd.materialexists("LiNbO3"):
        print("Using LiNbO3 already present in the current material database")
        return

    # Ansys LNO example: Zelmon three-oscillator Sellmeier model, wavelength in um.
    wavelength_start_um = float(SETTINGS.get("wavelength_start_um", 1.25))
    wavelength_stop_um = float(SETTINGS.get("wavelength_stop_um", 1.35))
    wavelength_min_um = min(wavelength_start_um, wavelength_stop_um)
    wavelength_max_um = max(wavelength_start_um, wavelength_stop_um)
    if wavelength_min_um < 0.4 or wavelength_max_um > 5.0:
        raise ValueError("LiNbO3 Sellmeier sampling supports wavelengths from 0.4 to 5.0 um")

    sample_min_um = max(0.4, 0.9 * wavelength_min_um)
    sample_max_um = min(5.0, 1.1 * wavelength_max_um)
    wavelength_um = np.linspace(sample_min_um, sample_max_um, 401)
    wavelength_sq = wavelength_um ** 2

    B_o = np.asarray([2.6734, 1.2290, 12.614], dtype=float)
    C_o = np.asarray([0.01764, 0.05914, 474.6], dtype=float)
    B_e = np.asarray([2.9804, 0.5981, 8.9543], dtype=float)
    C_e = np.asarray([0.02047, 0.0666, 416.08], dtype=float)
    n_o = np.sqrt(1.0 + sum(B * wavelength_sq / (wavelength_sq - C) for B, C in zip(B_o, C_o)))
    n_e = np.sqrt(1.0 + sum(B * wavelength_sq / (wavelength_sq - C) for B, C in zip(B_e, C_e)))

    # Moretti thermo-optic correction used by the same Ansys example; 296.3 K is its reference.
    temperature_K = float(SETTINGS.get("tfln_temperature_K", 296.3))
    reference_temperature_K = 296.3
    delta_temperature = temperature_K - reference_temperature_K
    delta_temperature_sq = temperature_K ** 2 - reference_temperature_K ** 2
    a_o = (0.897867565 * wavelength_um - 2.2674523) * 1e-5
    b_o = (-4.377104377e-3 * wavelength_um + 9.666329966e-3) * 1e-5
    a_e = np.full_like(wavelength_um, -2.6e-5)
    b_e = (-2.918069585e-3 * wavelength_um + 24.24421998e-3) * 1e-5
    n_o = n_o + a_o * delta_temperature + 0.5 * b_o * delta_temperature_sq
    n_e = n_e + a_e * delta_temperature + 0.5 * b_e * delta_temperature_sq

    # Match the cut-to-axis mapping in Ansys's linbo3_index.lsf example.
    crystal_cut = str(SETTINGS.get("tfln_crystal_cut", "X")).strip().upper()
    index_by_cut = {
        "X": (n_o, n_e, n_o),
        "Y": (n_e, n_o, n_o),
        "Z": (n_o, n_o, n_e),
    }
    if crystal_cut not in index_by_cut:
        raise ValueError("tfln_crystal_cut must be X, Y, or Z")
    n_x, n_y, n_z = index_by_cut[crystal_cut]

    frequency_hz = 299792458.0 / (wavelength_um * 1e-6)
    sampled_data = np.column_stack(
        (frequency_hz, n_x.astype(complex) ** 2, n_y.astype(complex) ** 2, n_z.astype(complex) ** 2)
    )
    sampled_data = sampled_data[np.argsort(sampled_data[:, 0].real)]

    material_id = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material_id, "name", "LiNbO3")
    fdtd.setmaterial("LiNbO3", "anisotropy", 1)
    fdtd.setmaterial("LiNbO3", "tolerance", 0.0)
    fdtd.setmaterial("LiNbO3", "max coefficients", 6)
    fdtd.setmaterial("LiNbO3", "make fit passive", True)
    fdtd.setmaterial("LiNbO3", "improve numerical stability", True)
    fdtd.setmaterial("LiNbO3", "specify fit range", True)
    fdtd.setmaterial("LiNbO3", "wavelength min", wavelength_min_um * UM)
    fdtd.setmaterial("LiNbO3", "wavelength max", wavelength_max_um * UM)
    fdtd.setmaterial("LiNbO3", "sampled 3d data", sampled_data)

    center_um = 0.5 * (wavelength_min_um + wavelength_max_um)
    center_index = int(np.argmin(np.abs(wavelength_um - center_um)))
    print(
        "Created dispersive anisotropic LiNbO3: cut={}, T={:.1f} K, "
        "n_o({:.4f} um)={:.6f}, n_e({:.4f} um)={:.6f}".format(
            crystal_cut,
            temperature_K,
            wavelength_um[center_index],
            n_o[center_index],
            wavelength_um[center_index],
            n_e[center_index],
        )
    )


def _maximum_material_index(fdtd, material, frequency_hz):
    """Return the largest finite index component at one frequency."""
    if str(material).strip().lower() in {"air", "vacuum", "<vacuum>"}:
        return 1.0
    values = np.asarray(fdtd.getindex(str(material), float(frequency_hz)))
    finite = np.abs(values[np.isfinite(values)])
    if finite.size < 1 or float(np.max(finite)) <= 0.0:
        raise RuntimeError(
            "Could not resolve a finite refractive index for material %s"
            % material
        )
    return float(np.max(finite))


def _derive_waveguide_neff_from_stack(fdtd, z_ranges):
    """Estimate the guided-mode index from the actual dispersive stack.

    The geometry material is the core.  The closest active background films
    immediately below and above the contiguous geometry block are its
    surroundings.  Their largest index is deliberately conservative for an
    asymmetric stack.  The midpoint is a stable eigensolver target without
    encoding platform-specific constants such as 2.0 or 2.5.
    """
    analysis_uid = int(
        (GRATING_ANALYSIS or MMI_ANALYSIS or {}).get("component_uid", -1)
    )
    device_gds_layers = {
        int(polygon.get("layer", -1))
        for polygon in GEOMETRY
        if int(polygon.get("component_uid", -2)) == analysis_uid
    }
    metal_tokens = ("gold", "silver", "aluminium", "aluminum", "copper", "au (", "ag (", "al (")

    def row_layers(row):
        values = row.get("gds_layers", [row.get("gds_layer", -1)])
        if isinstance(values, (str, int, float)):
            values = [values]
        return {int(value) for value in values}

    def is_device_dielectric(row):
        material = str(row.get("material", "")).strip().lower()
        return (
            str(row.get("role", "background")).lower() == "geometry"
            and not any(token in material for token in metal_tokens)
            and (not device_gds_layers or bool(row_layers(row) & device_gds_layers))
        )

    def is_background_dielectric(row):
        material = str(row.get("material", "")).strip().lower()
        return (
            str(row.get("role", "background")).lower() != "geometry"
            and not any(token in material for token in metal_tokens)
        )

    geometry_indices = [
        index
        for index, (row, _z0, _z1) in enumerate(z_ranges)
        if is_device_dielectric(row)
    ]
    if not geometry_indices:
        raise RuntimeError(
            "Automatic waveguide mode selection needs one active geometry material"
        )
    wavelength_center_um = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    )
    frequency_hz = 299792458.0 / (wavelength_center_um * UM)
    core_rows = [z_ranges[index][0] for index in geometry_indices]
    core_index = max(
        _maximum_material_index(fdtd, row.get("material", ""), frequency_hz)
        for row in core_rows
    )

    first_geometry = min(geometry_indices)
    last_geometry = max(geometry_indices)
    surrounding_rows = []
    for index in range(first_geometry - 1, -1, -1):
        row = z_ranges[index][0]
        if is_background_dielectric(row):
            surrounding_rows.append(row)
            break
    for index in range(last_geometry + 1, len(z_ranges)):
        row = z_ranges[index][0]
        if is_background_dielectric(row):
            surrounding_rows.append(row)
            break
    if not surrounding_rows:
        raise RuntimeError(
            "Automatic waveguide mode selection needs an active surrounding dielectric"
        )
    surrounding_indices = [
        _maximum_material_index(fdtd, row.get("material", ""), frequency_hz)
        for row in surrounding_rows
    ]
    cladding_index = max(surrounding_indices)
    if core_index <= cladding_index:
        raise RuntimeError(
            "The geometry material index %.6g is not above the surrounding index %.6g"
            % (core_index, cladding_index)
        )
    target_neff = 0.5 * (core_index + cladding_index)
    result = {
        "strategy": "midpoint of actual core and adjacent dielectric indices",
        "wavelength_um": float(wavelength_center_um),
        "core_materials": [str(row.get("material", "")) for row in core_rows],
        "surrounding_materials": [
            str(row.get("material", "")) for row in surrounding_rows
        ],
        "core_index": float(core_index),
        "surrounding_index": float(cladding_index),
        "target_neff": float(target_neff),
    }
    print(
        "Derived waveguide mode target at %.6g um: core n=%.6g, "
        "surrounding n=%.6g, midpoint neff=%.6g."
        % (wavelength_center_um, core_index, cladding_index, target_neff)
    )
    return result


def _layer_builder_geometry(origin_x_um, origin_y_um, geometry=None):
    """Convert global exported polygons into the Layer Builder's local XY frame."""
    result = {}
    local_origin_um = np.asarray([origin_x_um, origin_y_um], dtype=float)
    source_geometry = GEOMETRY if geometry is None else geometry
    for polygon in source_geometry:
        key = f'{int(polygon["layer"])}:{int(polygon.get("datatype", 0))}'
        global_vertices_um = np.asarray(polygon["vertices_um"], dtype=float)
        result.setdefault(key, []).append((global_vertices_um - local_origin_um) * UM)
    return result


def _conformal_fill_start(z_ranges, row_index, default_z_min_um):
    """Bottom of a planarized overclad that fills every contiguous etched device layer."""
    consecutive_geometry = []
    for previous_row, previous_z0, previous_z1 in reversed(z_ranges[:row_index]):
        if str(previous_row.get("role", "background")) != "geometry":
            if consecutive_geometry:
                break
            continue
        consecutive_geometry.append((previous_row, previous_z0, previous_z1))
    if len(consecutive_geometry) > 1:
        # A multi-mask vertical device (for example the official SOI
        # residual-slab + upper-silicon pair) is one physical film.  The
        # overclad fills all voids beside every mask down to the device base.
        return min(default_z_min_um, min(item[1] for item in consecutive_geometry))
    if consecutive_geometry:
        previous_row, previous_z0, previous_z1 = consecutive_geometry[0]
        previous_thickness = previous_z1 - previous_z0
        previous_etch = min(
            previous_thickness,
            max(0.0, float(previous_row.get("etch_depth_um", previous_thickness))),
        )
        return min(default_z_min_um, previous_z1 - previous_etch)
    return default_z_min_um


def _add_material_stack(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um, pml_geometry_overlap_um):
    """Build films and tapered cross-sections with Lumerical's Layer Builder."""
    fdtd.addlayerbuilder()
    fdtd.set("name", "Max Layout material stack")
    geometry_mesh_orders = [
        max(1, int(row.get("mesh_order", 2)))
        for row, _z0, _z1 in z_ranges
        if str(row.get("role", "background")) == "geometry"
    ]
    layer_builder_mesh_order = min(geometry_mesh_orders, default=2)
    if any(order != layer_builder_mesh_order for order in geometry_mesh_orders):
        print(
            "Warning: Layer Builder grow layers share one mesh order; using the smallest requested value %d."
            % layer_builder_mesh_order
        )
    fdtd.set("base mesh order", layer_builder_mesh_order)
    fdtd.set("process name", "Max Layout export")
    fdtd.set("GDS sidewall angle position reference", "middle")

    # A manually resized FDTD domain is allowed to cut through a waveguide or
    # another device polygon so the geometry reaches the PML.  Material films
    # must not be clipped to that solver box: doing so could leave the part of
    # an exported tooth, flare, terminal arc, or waveguide outside the oxide
    # cladding and previously caused the conformal-coverage check to fail.
    # Keep the FDTD bounds unchanged and independently enlarge the material
    # extent to the union of the solver box and all exported polygons.
    material_x_min_um = float(bounds[0]) - pml_geometry_overlap_um
    material_x_max_um = float(bounds[2]) + pml_geometry_overlap_um
    material_y_min_um = float(bounds[1]) - pml_geometry_overlap_um
    material_y_max_um = float(bounds[3]) + pml_geometry_overlap_um
    geometry_bounds_um = None
    if GEOMETRY:
        geometry_points_um = np.vstack([
            np.asarray(polygon["vertices_um"], dtype=float) for polygon in GEOMETRY
        ])
        geometry_bounds_um = (
            float(np.min(geometry_points_um[:, 0])),
            float(np.min(geometry_points_um[:, 1])),
            float(np.max(geometry_points_um[:, 0])),
            float(np.max(geometry_points_um[:, 1])),
        )
        material_x_min_um = min(
            material_x_min_um,
            geometry_bounds_um[0] - pml_geometry_overlap_um,
        )
        material_x_max_um = max(
            material_x_max_um,
            geometry_bounds_um[2] + pml_geometry_overlap_um,
        )
        material_y_min_um = min(
            material_y_min_um,
            geometry_bounds_um[1] - pml_geometry_overlap_um,
        )
        material_y_max_um = max(
            material_y_max_um,
            geometry_bounds_um[3] + pml_geometry_overlap_um,
        )

    layer_builder_x_um = 0.5 * (material_x_min_um + material_x_max_um)
    layer_builder_y_um = 0.5 * (material_y_min_um + material_y_max_um)
    fdtd.set("x", layer_builder_x_um * UM)
    fdtd.set("y", layer_builder_y_um * UM)
    fdtd.set("z", 0.0)
    fdtd.set("x span", (material_x_max_um - material_x_min_um) * UM)
    fdtd.set("y span", (material_y_max_um - material_y_min_um) * UM)

    # GEOMETRY vertices are absolute layout coordinates, while Layer Builder
    # interprets its geometry struct relative to the object's own origin.
    # Subtracting that origin prevents asymmetric FDTD padding from shifting
    # the physical device away from ports, fibers, and monitors.
    geometry_by_layer = _layer_builder_geometry(layer_builder_x_um, layer_builder_y_um)
    if geometry_by_layer:
        # Python dict -> Lumerical struct; Python list -> Lumerical cell array.
        fdtd.set("geometry", geometry_by_layer)

    background_volumes = []

    def add_background(name, material, z_min_um, z_max_um, mesh_order, conformal=False):
        if z_max_um <= z_min_um:
            return
        background_volumes.append(
            (
                name, material, float(z_min_um), float(z_max_um),
                max(1, int(mesh_order)), bool(conformal),
            )
        )

    def matching_geometry_keys(row):
        target_layers = {int(value) for value in row.get("gds_layers", [row.get("gds_layer", 1)])}
        return [
            key for key in geometry_by_layer
            if int(key.split(":", 1)[0]) in target_layers
        ]

    def add_patterned_layers(name_prefix, row, z_min_um, z_max_um, sidewall_angle_deg):
        if z_max_um <= z_min_um:
            return 0
        matching_keys = matching_geometry_keys(row)
        for key_index, layer_key in enumerate(matching_keys, start=1):
            process_name = f"{name_prefix} {key_index} ({layer_key})"
            fdtd.addlayer(process_name)
            fdtd.setlayer(process_name, "layer number", layer_key)
            fdtd.setlayer(process_name, "start position", z_min_um * UM)
            fdtd.setlayer(process_name, "thickness", (z_max_um - z_min_um) * UM)
            fdtd.setlayer(process_name, "process", "grow")
            fdtd.setlayer(process_name, "pattern material", row["material"])
            fdtd.setlayer(process_name, "sidewall angle", sidewall_angle_deg)
        return len(matching_keys)

    for row_index, (row, z0, z1) in enumerate(z_ranges, start=1):
        base_name = f'{row_index:02d} {row["name"]}'
        if row.get("role") != "geometry":
            # Air is the FDTD background (n=1), not an overlapping geometry.
            # Omitting an explicit air process layer prevents it from winning
            # a same-order Layer Builder overlap with the conformal cladding.
            if str(row.get("material", "")).strip().lower() == "air":
                print("Using FDTD background index 1 for %s; no overlapping air layer was created." % row["name"])
                continue
            is_conformal = bool(row.get("conformal", False))
            background_z0 = (
                _conformal_fill_start(z_ranges, row_index - 1, z0)
                if is_conformal else z0
            )
            background_z1 = z1
            if row_index == 1:
                background_z0 = min(background_z0, simulation_z_min_um - pml_geometry_overlap_um)
            if row_index == len(z_ranges):
                background_z1 = max(background_z1, simulation_z_max_um + pml_geometry_overlap_um)
            add_background(
                base_name,
                row["material"],
                background_z0,
                background_z1,
                row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2),
                conformal=is_conformal,
            )
            if background_z0 < z0:
                print(
                    "Conformal cladding {} fills etched openings from z={:.6g} to {:.6g} um".format(
                        row["name"], background_z0, z1
                    )
                )
            continue

        thickness_um = z1 - z0
        etch_depth_um = min(thickness_um, max(0.0, float(row.get("etch_depth_um", thickness_um))))
        if etch_depth_um < thickness_um:
            slab_extent = str(row.get("slab_extent", "full")).strip().lower()
            if slab_extent == "geometry":
                slab_count = add_patterned_layers(
                    f"{base_name} footprint slab",
                    row,
                    z0,
                    z1 - etch_depth_um,
                    90.0,
                )
                if slab_count:
                    print("Limited unetched slab %s to the exported geometry footprint." % row["name"])
                else:
                    print(
                        "Warning: %s requested a geometry-limited slab but no matching GDS polygons were found."
                        % row["name"]
                    )
            else:
                add_background(
                    f"{base_name} unetched film",
                    row["material"],
                    z0,
                    z1 - etch_depth_um,
                    row.get("mesh_order", layer_builder_mesh_order),
                )
        if etch_depth_um <= 0.0:
            continue

        target_layers = {int(value) for value in row.get("gds_layers", [row.get("gds_layer", 1)])}
        matching_keys = matching_geometry_keys(row)
        if not matching_keys:
            print(f'Warning: {row["name"]} has no polygons on GDS layers {sorted(target_layers)}.')
            continue
        sidewall_angle_deg = min(179.999, max(0.001, float(row.get("sidewall_angle_deg", 90.0))))
        add_patterned_layers(
            f"{base_name} pattern",
            row,
            z1 - etch_depth_um,
            z1,
            sidewall_angle_deg,
        )

    # Layer Builder forces every background process layer to base order + 2.
    # Create full-film volumes explicitly so the requested substrate, BOX, and
    # conformal-cladding orders are preserved independently.
    cladding_x_min_um = material_x_min_um
    cladding_x_max_um = material_x_max_um
    cladding_y_min_um = material_y_min_um
    cladding_y_max_um = material_y_max_um
    for name, material, z_min_um, z_max_um, mesh_order, is_conformal in background_volumes:
        fdtd.addrect()
        fdtd.set("name", "Max Layout " + name)
        fdtd.set("material", str(material))
        fdtd.set("x", 0.5 * (material_x_min_um + material_x_max_um) * UM)
        fdtd.set("y", 0.5 * (material_y_min_um + material_y_max_um) * UM)
        fdtd.set("x span", (material_x_max_um - material_x_min_um) * UM)
        fdtd.set("y span", (material_y_max_um - material_y_min_um) * UM)
        fdtd.set("z min", z_min_um * UM)
        fdtd.set("z max", z_max_um * UM)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", mesh_order)
        print("Added material volume %s with mesh order %d." % (name, mesh_order))
        if is_conformal:
            if geometry_bounds_um is not None:
                geometry_x_min_um = geometry_bounds_um[0]
                geometry_y_min_um = geometry_bounds_um[1]
                geometry_x_max_um = geometry_bounds_um[2]
                geometry_y_max_um = geometry_bounds_um[3]
                if (
                    geometry_x_min_um < cladding_x_min_um - 1e-9
                    or geometry_x_max_um > cladding_x_max_um + 1e-9
                    or geometry_y_min_um < cladding_y_min_um - 1e-9
                    or geometry_y_max_um > cladding_y_max_um + 1e-9
                ):
                    raise RuntimeError(
                        "Conformal cladding %s does not cover every exported geometry polygon in XY"
                        % name
                    )
            print(
                "Verified full-domain conformal cladding %s: X [%.6g, %.6g] um, "
                "Y [%.6g, %.6g] um, Z [%.6g, %.6g] um; fills all etched holes and "
                "covers every waveguide, grating tooth, flare, terminal arc, and extension."
                % (
                    name,
                    cladding_x_min_um, cladding_x_max_um,
                    cladding_y_min_um, cladding_y_max_um,
                    z_min_um, z_max_um,
                )
            )


def _add_layer_mesh_overrides(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um):
    """Resolve dispersive material indices and apply wavelength-scaled layer meshes."""
    wavelength_min_um = min(
        float(SETTINGS.get("wavelength_start_um", 1.25)),
        float(SETTINGS.get("wavelength_stop_um", 1.35)),
    )
    frequency_hz = 299792458.0 / (wavelength_min_um * UM)
    for row_index, (row, z0, z1) in enumerate(z_ranges, start=1):
        mesh_factor = max(0.0, float(row.get("mesh_factor", 0.2)))
        mesh_z0 = max(float(z0), float(simulation_z_min_um))
        mesh_z1 = min(float(z1), float(simulation_z_max_um))
        if mesh_factor <= 0.0 or mesh_z1 <= mesh_z0:
            continue
        material = str(row.get("material", ""))
        material_index = np.asarray(fdtd.getindex(material, frequency_hz))
        finite_indices = np.abs(material_index[np.isfinite(material_index)])
        if finite_indices.size == 0 or float(np.max(finite_indices)) <= 0.0:
            raise RuntimeError("Could not determine a finite refractive index for mesh layer " + material)
        # The maximum component is deliberately used for anisotropic media so
        # no crystal axis receives a coarser-than-requested optical mesh.
        maximum_index = float(np.max(finite_indices))
        mesh_step_um = mesh_factor * wavelength_min_um / maximum_index
        fdtd.addmesh()
        fdtd.set("name", "mesh %02d %s" % (row_index, str(row.get("name", "layer"))))
        fdtd.set("x min", float(bounds[0]) * UM)
        fdtd.set("x max", float(bounds[2]) * UM)
        fdtd.set("y min", float(bounds[1]) * UM)
        fdtd.set("y max", float(bounds[3]) * UM)
        fdtd.set("z min", mesh_z0 * UM)
        fdtd.set("z max", mesh_z1 * UM)
        fdtd.set("override x mesh", True)
        fdtd.set("override y mesh", True)
        fdtd.set("override z mesh", True)
        fdtd.set("dx", mesh_step_um * UM)
        fdtd.set("dy", mesh_step_um * UM)
        fdtd.set("dz", mesh_step_um * UM)
        print(
            "Layer mesh override %s: factor %.6g x lambda0/n, |n|max %.6g, %.6g um isotropic step."
            % (row.get("name", "layer"), mesh_factor, maximum_index, mesh_step_um)
        )


def _nearest_port_axis(outward_angle_deg):
    nearest = int(round(float(outward_angle_deg) / 90.0) * 90) % 360
    axis = "x-axis" if nearest in (0, 180) else "y-axis"
    # Direction points from the exterior port plane into the selected geometry.
    direction = "Forward" if nearest in (180, 270) else "Backward"
    return nearest, axis, direction


def _silica_cladding_top_um(z_ranges, device_top_um):
    """Top of the upper conformal silica cladding, never the air above it."""
    silica_rows = []
    for row, _z0, z1 in z_ranges:
        label = (str(row.get("name", "")) + " " + str(row.get("material", ""))).lower()
        is_silica = "sio2" in label or "silica" in label or "glass" in label
        if is_silica and float(z1) >= float(device_top_um) - 1e-12:
            silica_rows.append((bool(row.get("conformal", False)), float(z1)))
    conformal_tops = [z1 for conformal, z1 in silica_rows if conformal]
    if conformal_tops:
        return max(conformal_tops)
    return max((z1 for _conformal, z1 in silica_rows), default=float(device_top_um))


def _silica_cladding_center_um(z_ranges, device_top_um):
    """Center of the same upper silica film selected by _silica_cladding_top_um."""
    silica_rows = []
    for row, z0, z1 in z_ranges:
        label = (str(row.get("name", "")) + " " + str(row.get("material", ""))).lower()
        is_silica = "sio2" in label or "silica" in label or "glass" in label
        if is_silica and float(z1) >= float(device_top_um) - 1e-12:
            silica_rows.append((bool(row.get("conformal", False)), float(z0), float(z1)))
    candidates = [entry for entry in silica_rows if entry[0]] or silica_rows
    if not candidates:
        return float(device_top_um)
    _conformal, z0, z1 = max(candidates, key=lambda entry: entry[2])
    return 0.5 * (z0 + z1)


def _vertical_reference_um(
    item, device_top_um, stack_top_um,
    silica_cladding_top_um=None, silica_cladding_center_um=None,
):
    reference = str(item.get("z reference", "device top")).strip().lower()
    if reference in {"center of sio2 cladding", "center of silica cladding", "cladding center"}:
        return device_top_um if silica_cladding_center_um is None else silica_cladding_center_um
    if reference in {"top of sio2 cladding", "top of silica cladding", "top cladding"}:
        return device_top_um if silica_cladding_top_um is None else silica_cladding_top_um
    if reference == "top of stack":
        return stack_top_um
    return device_top_um


def _add_fiber_geometries(
    fdtd, device_top_um, stack_top_um, silica_cladding_top_um, silica_cladding_center_um
):
    """Add only the official tilted core/cladding structure groups; never add a source here."""
    used_names = set()
    for index, fiber in enumerate(FIBER_GEOMETRIES, start=1):
        name = str(fiber.get("name") or f"fiber_{index}")
        if name in used_names:
            name = f"uid_{fiber.get('component_uid', 0)}_{name}"
        used_names.add(name)
        bottom_x_um, bottom_y_um = map(float, fiber.get("center", (0.0, 0.0)))
        if "angle theta" not in fiber:
            raise ValueError(
                "Fiber geometry %s has no angle theta"
                % fiber.get("name", name)
            )
        theta_deg = float(fiber["angle theta"])
        phi_deg = float(fiber.get("angle phi", 0.0))
        core_diameter_um = max(1e-6, float(fiber.get("core diameter_um", 9.0)))
        cladding_diameter_um = max(core_diameter_um, float(fiber.get("cladding diameter_um", 50.0)))
        fiber_length_um = max(1e-6, float(fiber.get("fiber length_um", 20.0)))
        bottom_gap_um = float(fiber.get("distance_um", 0.0))
        reference_z_um = _vertical_reference_um(
            fiber, device_top_um, stack_top_um,
            silica_cladding_top_um, silica_cladding_center_um,
        )
        # Match the official Ansys construction: core and cladding are
        # through-going tilted cylinders centered on the nominal polished
        # contact point. They visibly cross the complete model in both axial
        # directions; mesh precedence clips their effective material at the
        # grating, top oxide, BOX, and substrate.
        center_z_um = reference_z_um + bottom_gap_um
        x_um = bottom_x_um
        y_um = bottom_y_um

        fdtd.addstructuregroup()
        fdtd.set("name", name)
        fdtd.set("x", x_um * UM)
        fdtd.set("y", y_um * UM)
        fdtd.set("z", center_z_um * UM)
        fdtd.set("use relative coordinates", True)
        if abs(phi_deg) > 1e-12:
            fdtd.set("first axis", "z")
            fdtd.set("rotation 1", phi_deg)

        # Use the same user-property names and setup-script construction as
        # the official Ansys grating-coupler fiber. Objects created by a group
        # script are born in its relative coordinate system, avoiding the
        # addtogroup translation that could apply the group center twice.
        fdtd.adduserprop("core diameter", 2, core_diameter_um * UM)
        fdtd.adduserprop("cladding diameter", 2, cladding_diameter_um * UM)
        fdtd.adduserprop("z span", 2, fiber_length_um * UM)
        fdtd.adduserprop("theta", 0, theta_deg)
        fdtd.adduserprop("core index", 0, float(fiber.get("core index", 1.44427)))
        fdtd.adduserprop("cladding index", 0, float(fiber.get("cladding index", 1.43482)))
        # Keep both overlap priorities visible on the fiber group instead of
        # burying a legacy mesh-order value in its setup script.
        fdtd.adduserprop("core mesh order", 0, 4)
        fdtd.adduserprop("cladding mesh order", 0, 5)
        fiber_setup_script = r"""
deleteall;
core_index = %core index%;
cladding_index = %cladding index%;
core_mesh_order = %core mesh order%;
cladding_mesh_order = %cladding mesh order%;
core_radius = %core diameter%/2.0;
cladding_radius = %cladding diameter%/2.0;
theta_rad = theta*pi/180.0;
L = %z span%/cos(theta_rad);
addcircle;
set("name","cladding");
set("radius",cladding_radius);
set("material","<Object defined dielectric>");
set("index",cladding_index);
set("override color opacity from material database",1);
set("alpha",0.03);
set("override mesh order from material database",1);
set("mesh order",cladding_mesh_order);
set("x",0.0);
set("y",0.0);
set("z",0.0);
set("z span",L);
set("first axis","y");
set("rotation 1",theta);
addcircle;
set("name","core");
set("radius",core_radius);
set("material","<Object defined dielectric>");
set("index",core_index);
set("override color opacity from material database",1);
set("alpha",0.35);
set("override mesh order from material database",1);
set("mesh order",core_mesh_order);
set("x",0.0);
set("y",0.0);
set("z",0.0);
set("z span",L);
set("first axis","y");
set("rotation 1",theta);
"""
        fdtd.set("script", fiber_setup_script)
        print(
            "Added scripted Ansys fiber property group %s with core/cladding internal offsets (0, 0, 0) um "
            "as through-going cylinders centered on the nominal contact at (%.6g, %.6g) um "
            "(mesh precedence clips the effective fiber material; no source or port was created)."
            % (name, bottom_x_um, bottom_y_um)
        )


def _add_gaussian_sources(
    fdtd, device_top_um, stack_top_um,
    silica_cladding_top_um, silica_cladding_center_um,
):
    """Create analytic Gaussian excitation planes; these are not FDTD ports."""
    used_names = set()
    for index, source in enumerate(GAUSSIAN_SOURCES, start=1):
        name = str(source.get("name") or "gaussian_%d" % index)
        if name in used_names:
            name = "uid_%s_%s" % (source.get("component_uid", 0), name)
        used_names.add(name)
        x_um, y_um = map(float, source.get("center", (0.0, 0.0)))
        z_um = _vertical_reference_um(
            source, device_top_um, stack_top_um,
            silica_cladding_top_um, silica_cladding_center_um,
        ) + float(source.get("distance_um", 0.0))
        theta_deg = float(source.get("angle theta", 0.0))
        phi_deg = float(source.get("angle phi", 0.0))
        span_um = max(1e-6, float(source.get("span_um", 20.0)))
        waist_um = max(1e-9, float(source.get("waist radius_um", 4.5)))
        distance_from_waist_um = float(
            source.get("distance from waist_um", 0.0)
        )
        fdtd.addgaussian()
        fdtd.set("name", name)
        fdtd.set("injection axis", "z")
        fdtd.set("direction", "backward")
        fdtd.set("x", x_um * UM)
        fdtd.set("x span", span_um * UM)
        fdtd.set("y", y_um * UM)
        fdtd.set("y span", span_um * UM)
        fdtd.set("z", z_um * UM)
        fdtd.set("angle theta", theta_deg)
        fdtd.set("angle phi", phi_deg)
        fdtd.set("polarization angle", 90.0)
        fdtd.set("amplitude", float(source.get("amplitude", 1.0)))
        fdtd.set("use scalar approximation", True)
        fdtd.set("waist radius w0", waist_um * UM)
        fdtd.set("distance from waist", distance_from_waist_um * UM)
        if bool(source.get("multifrequency beam calculation", True)):
            requested_profile_samples = max(
                2, int(source.get("frequency points", 5))
            )
            # In v261 the Gaussian object's sample-count property is the
            # authoritative multifrequency control.  Set it before enabling
            # multifrequency mode when possible.  Some older builds expose
            # that property only after activation, so retry in that order and
            # then verify both values rather than silently falling back to a
            # single-frequency beam.
            try:
                fdtd.set(
                    "number of field profile samples",
                    requested_profile_samples,
                )
            except Exception as samples_first_exc:
                print(
                    "Gaussian multifrequency activation-order fallback for %s: %s"
                    % (name, str(samples_first_exc)[:180])
                )
                try:
                    fdtd.set("multifrequency beam calculation", True)
                    fdtd.set(
                        "number of field profile samples",
                        requested_profile_samples,
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "Gaussian source %s could not enable its requested %d-point "
                        "multifrequency field profile: %s"
                        % (
                            name,
                            requested_profile_samples,
                            str(fallback_exc)[:240],
                        )
                    ) from fallback_exc
            else:
                fdtd.set("multifrequency beam calculation", True)
            try:
                actual_profile_samples = int(round(float(np.asarray(
                    fdtd.get("number of field profile samples")
                ).squeeze())))
                multifrequency_enabled = bool(np.asarray(
                    fdtd.get("multifrequency beam calculation")
                ).squeeze())
            except Exception as readback_exc:
                raise RuntimeError(
                    "Gaussian source %s multifrequency settings were written but "
                    "could not be verified: %s"
                    % (name, str(readback_exc)[:240])
                ) from readback_exc
            if (
                actual_profile_samples != requested_profile_samples
                or not multifrequency_enabled
            ):
                raise RuntimeError(
                    "Gaussian source %s rejected its multifrequency settings: "
                    "requested %d samples/enabled, read back %d/%r"
                    % (
                        name,
                        requested_profile_samples,
                        actual_profile_samples,
                        multifrequency_enabled,
                    )
                )
        source["name"] = name
        print(
            "Added Gaussian grating source %s: backward Z, theta %.6g deg, "
            "phi %.6g deg, S/local-TE polarization, waist radius %.6g um."
            % (name, theta_deg, phi_deg, waist_um)
        )


def _mode_profile_vector(mode_profile, mode_number):
    """Return one selected port mode as a complex (..., 3) Cartesian field."""
    available = list(mode_profile.keys())
    candidates = ["E%d" % int(mode_number), "E"]
    rejected = []
    for key in candidates:
        if key not in available:
            continue
        electric = np.asarray(mode_profile[key])
        if electric.ndim == 0:
            rejected.append((key, electric.shape))
            continue
        if electric.shape[-1] == 3:
            component_axis = electric.ndim - 1
        else:
            component_axes = [axis for axis, size in enumerate(electric.shape) if size == 3]
            component_axis = component_axes[-1] if component_axes else None
        if component_axis is None:
            rejected.append((key, electric.shape))
            continue
        electric = np.moveaxis(electric, component_axis, -1)
        if key == "E":
            try:
                mode_coordinates = np.asarray(mode_profile["n"]).ravel()
            except Exception:
                mode_coordinates = np.asarray([])
            if mode_coordinates.size > 1:
                mode_axes = [
                    axis for axis, size in enumerate(electric.shape[:-1])
                    if size == mode_coordinates.size
                ]
                if mode_axes:
                    matches = np.flatnonzero(
                        np.isclose(mode_coordinates.astype(float), float(mode_number))
                    )
                    mode_index = (
                        int(matches[0]) if matches.size
                        else int(mode_number) - 1
                    )
                    if not 0 <= mode_index < mode_coordinates.size:
                        rejected.append((key, electric.shape))
                        continue
                    electric = np.take(electric, mode_index, axis=mode_axes[-1])
        electric = np.squeeze(electric)
        while electric.ndim > 3:
            electric = np.take(electric, 0, axis=-2)
        if electric.ndim < 2 or electric.shape[-1] != 3:
            rejected.append((key, electric.shape))
            continue
        return np.asarray(electric, dtype=complex)
    raise RuntimeError(
        "Mode %d has no readable vector electric field. Available fields: %r; rejected: %r"
        % (int(mode_number), available, rejected)
    )


def _fiber_local_te_score(mode_profile, mode_number, grating_axis_deg):
    """Power fraction normal to the grating propagation axis in global XY."""
    electric = _mode_profile_vector(mode_profile, mode_number)
    phi = np.deg2rad(float(grating_axis_deg))
    target_x = -np.sin(phi)
    target_y = np.cos(phi)
    desired = target_x * electric[..., 0] + target_y * electric[..., 1]
    desired_power = float(np.sum(np.abs(desired) ** 2))
    total_power = float(np.sum(np.abs(electric) ** 2))
    return desired_power / max(total_power, 1e-300)


def _fiber_gaussian_circular_scores(mode_profile, mode_number):
    """Return Gaussian-overlap and circular-second-moment scores in [0, 1]."""
    electric = _mode_profile_vector(mode_profile, mode_number)
    intensity = np.asarray(np.sum(np.abs(electric) ** 2, axis=-1), dtype=float)
    intensity = np.squeeze(intensity)
    while intensity.ndim > 2:
        intensity = np.sum(intensity, axis=0)
    if intensity.ndim != 2 or min(intensity.shape) < 2:
        return 0.0, 0.0
    peak = float(np.max(intensity))
    total = float(np.sum(intensity))
    if not np.isfinite(peak) or peak <= 0.0 or total <= 0.0:
        return 0.0, 0.0
    weights = intensity / total
    coordinate_0 = np.linspace(-1.0, 1.0, intensity.shape[0])
    coordinate_1 = np.linspace(-1.0, 1.0, intensity.shape[1])
    grid_0, grid_1 = np.meshgrid(coordinate_0, coordinate_1, indexing="ij")
    center_0 = float(np.sum(weights * grid_0))
    center_1 = float(np.sum(weights * grid_1))
    delta_0 = grid_0 - center_0
    delta_1 = grid_1 - center_1
    covariance = np.asarray(
        [
            [np.sum(weights * delta_0 * delta_0), np.sum(weights * delta_0 * delta_1)],
            [np.sum(weights * delta_0 * delta_1), np.sum(weights * delta_1 * delta_1)],
        ],
        dtype=float,
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    circularity = float(np.clip(eigenvalues[0] / eigenvalues[-1], 0.0, 1.0))
    inverse_covariance = np.linalg.pinv(covariance + np.eye(2) * 1e-12)
    radius_squared = (
        inverse_covariance[0, 0] * delta_0 ** 2
        + 2.0 * inverse_covariance[0, 1] * delta_0 * delta_1
        + inverse_covariance[1, 1] * delta_1 ** 2
    )
    gaussian = np.exp(-0.5 * radius_squared)
    gaussian_similarity = float(
        np.sum(intensity * gaussian)
        / max(
            np.sqrt(np.sum(intensity ** 2) * np.sum(gaussian ** 2)),
            1e-300,
        )
    )
    boundary_peak = float(
        max(
            np.max(intensity[0, :]), np.max(intensity[-1, :]),
            np.max(intensity[:, 0]), np.max(intensity[:, -1]),
        )
    )
    gaussian_quality = gaussian_similarity * max(0.0, 1.0 - boundary_peak / peak)
    return float(np.clip(gaussian_quality, 0.0, 1.0)), circularity


def _fiber_candidate_neff(fdtd, port_path, candidate_modes):
    """Central neff per mode, including wavelength-by-mode result tensors."""
    try:
        dataset = fdtd.getresult(port_path, "neff")
        available = list(dataset.keys())
    except Exception:
        return {}
    result = {}
    normalized = {
        "".join(character for character in str(key).lower() if character.isalnum()): key
        for key in available
    }
    for mode_number in candidate_modes:
        key = normalized.get("neff%d" % int(mode_number))
        if key is None:
            continue
        values = np.real(np.asarray(dataset[key])).ravel()
        finite = values[np.isfinite(values)]
        if finite.size:
            result[int(mode_number)] = float(np.median(finite))
    if len(result) == len(candidate_modes):
        return result
    try:
        plain_key = normalized.get("neff")
        values = np.real(np.asarray(dataset[plain_key])).squeeze()
        if values.ndim == 0:
            values = values.reshape(1)
        mode_key = normalized.get("n")
        if mode_key is not None:
            mode_coordinate = np.asarray(dataset[mode_key]).squeeze().ravel()
            mode_numbers = [int(round(float(value))) for value in mode_coordinate]
        else:
            mode_numbers = list(map(int, candidate_modes))
        mode_count = len(mode_numbers)
        if values.ndim == 1 and values.size == mode_count:
            mode_matrix = values.reshape(1, mode_count)
        else:
            candidate_axes = [
                axis for axis, size in enumerate(values.shape)
                if size == mode_count
            ]
            spectral_key = normalized.get("lambda") or normalized.get("f")
            spectral_size = None
            if spectral_key is not None:
                spectral_size = np.asarray(dataset[spectral_key]).squeeze().size
            if len(candidate_axes) > 1 and spectral_size is not None:
                non_spectral_axes = [
                    axis for axis in candidate_axes
                    if values.shape[axis] != spectral_size
                ]
                if non_spectral_axes:
                    candidate_axes = non_spectral_axes
            if not candidate_axes:
                raise ValueError("neff tensor has no mode-coordinate axis")
            # Lumerical datasets conventionally place the mode coordinate last;
            # choosing the last match also resolves equal wavelength/mode counts.
            mode_axis = candidate_axes[-1]
            mode_matrix = np.moveaxis(values, mode_axis, -1).reshape(-1, mode_count)
        for column, mode_number in enumerate(mode_numbers):
            if int(mode_number) not in set(map(int, candidate_modes)):
                continue
            finite = np.asarray(mode_matrix[:, column]).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size:
                result[int(mode_number)] = float(np.median(finite))
    except Exception:
        pass
    return result


def _select_fiber_local_te_mode(fdtd, port_path, port):
    """Solve three candidates and choose the Gaussian local-TE HE11 partner."""
    raw_candidates = port.get("candidate mode numbers", (1, 2, 3))
    candidate_modes = []
    for value in raw_candidates:
        mode_number = int(value)
        if mode_number > 0 and mode_number not in candidate_modes:
            candidate_modes.append(mode_number)
    if len(candidate_modes) < 2:
        raise RuntimeError(
            "Fiber port %s requires at least two candidate mode numbers"
            % port.get("name", port_path)
        )
    candidate_modes = candidate_modes[:3]
    fdtd.select(port_path)
    update_status = fdtd.updateportmodes(np.asarray(candidate_modes, dtype=int))
    if update_status is not None and float(np.asarray(update_status).squeeze()) < 0.0:
        raise RuntimeError(
            "Lumerical could not calculate the fiber candidates %r for %s"
            % (candidate_modes, port.get("name", port_path))
        )
    mode_profile = fdtd.getresult(port_path, "mode profiles")
    grating_axis_deg = float(port.get("angle phi", 0.0)) % 360.0
    scores = {
        mode_number: _fiber_local_te_score(
            mode_profile, mode_number, grating_axis_deg
        )
        for mode_number in candidate_modes
    }
    gaussian_scores = {}
    circularity_scores = {}
    for mode_number in candidate_modes:
        gaussian_scores[mode_number], circularity_scores[mode_number] = (
            _fiber_gaussian_circular_scores(mode_profile, mode_number)
        )
    neff_by_mode = _fiber_candidate_neff(fdtd, port_path, candidate_modes)
    degeneracy_tolerance = max(
        0.0, float(port.get("mode degeneracy tolerance", 0.01))
    )
    target_neff = float(port.get("fiber target neff", 1.44))
    if len(neff_by_mode) == len(candidate_modes):
        candidate_pairs = [
            (first, second)
            for pair_index, first in enumerate(candidate_modes)
            for second in candidate_modes[pair_index + 1:]
            if abs(neff_by_mode[first] - neff_by_mode[second])
            <= degeneracy_tolerance
        ]
        if not candidate_pairs:
            raise RuntimeError(
                "None of the first three fiber modes at %s form the expected near-degenerate "
                "HE11 pair: neff=%r, tolerance=%.6g"
                % (port.get("name", port_path), neff_by_mode, degeneracy_tolerance)
            )
        degenerate_pair = min(
            candidate_pairs,
            key=lambda pair: (
                abs(0.5 * (neff_by_mode[pair[0]] + neff_by_mode[pair[1]]) - target_neff),
                abs(neff_by_mode[pair[0]] - neff_by_mode[pair[1]]),
            ),
        )
    else:
        # Some Lumerical builds expose vector fields before the per-mode neff
        # keys.  Keep all three solved fields, but fall back to the two modes
        # nearest the eigensolver's fundamental ordering.
        degenerate_pair = tuple(candidate_modes[:2])
    composite_scores = {
        mode_number: (
            0.75 * scores[mode_number]
            + 0.15 * gaussian_scores[mode_number]
            + 0.10 * circularity_scores[mode_number]
        )
        for mode_number in degenerate_pair
    }
    selected_mode = max(
        degenerate_pair, key=lambda mode_number: composite_scores[mode_number]
    )
    minimum_fraction = float(port.get("minimum local TE fraction", 0.8))
    if scores[selected_mode] < minimum_fraction:
        raise RuntimeError(
            "Neither near-degenerate fiber mode is sufficiently polarized normal to the "
            "grating axis at %.6g deg. Scores=%r; required >= %.6g."
            % (grating_axis_deg, scores, minimum_fraction)
        )
    partner_mode = next(
        mode_number for mode_number in degenerate_pair if mode_number != selected_mode
    )
    retained_modes = [selected_mode, partner_mode] + [
        mode_number
        for mode_number in candidate_modes
        if mode_number not in {selected_mode, partner_mode}
    ]
    degeneracy_delta = None
    if all(mode_number in neff_by_mode for mode_number in degenerate_pair):
        degeneracy_delta = abs(
            neff_by_mode[degenerate_pair[0]] - neff_by_mode[degenerate_pair[1]]
        )
        if degeneracy_delta > degeneracy_tolerance:
            raise RuntimeError(
                "Fiber modes %r at %s are not the expected near-degenerate pair: "
                "neff=%r, delta=%.6g > %.6g"
                % (
                    degenerate_pair, port.get("name", port_path), neff_by_mode,
                    degeneracy_delta, degeneracy_tolerance,
                )
            )
    target_x = -np.sin(np.deg2rad(grating_axis_deg))
    target_y = np.cos(np.deg2rad(grating_axis_deg))
    selection = {
        "mode number": int(selected_mode),
        "selected mode order": list(map(int, retained_modes)),
        # Port-group source labels and expansion-result ``n`` coordinates use
        # the eigensolver mode number.  Keep that identity explicit so a
        # multi-mode fiber result is never reduced by position alone.
        "selected mode result number": int(selected_mode),
        "candidate mode numbers": list(map(int, candidate_modes)),
        "degenerate mode pair": list(map(int, degenerate_pair)),
        "local TE scores": {str(key): float(value) for key, value in scores.items()},
        "gaussian scores": {
            str(key): float(value) for key, value in gaussian_scores.items()
        },
        "circularity scores": {
            str(key): float(value) for key, value in circularity_scores.items()
        },
        "composite scores": {
            str(key): float(value) for key, value in composite_scores.items()
        },
        "grating axis deg": float(grating_axis_deg),
        "target polarization xy": [float(target_x), float(target_y)],
        "polarization": "local TE",
        "candidate neff": {str(key): float(value) for key, value in neff_by_mode.items()},
        "fiber target neff": float(target_neff),
        "neff degeneracy delta": degeneracy_delta,
        "minimum local TE fraction": float(minimum_fraction),
    }
    print(
        "Fiber port %s calculated modes %r, identified HE11 pair %r, and selected mode %d: "
        "local-TE %.6f, Gaussian %.6f, circularity %.6f for target "
        "(Ex,Ey)=(%.6g,%.6g), grating axis %.6g deg%s."
        % (
            port.get("name", port_path), candidate_modes, degenerate_pair,
            selected_mode, scores[selected_mode], gaussian_scores[selected_mode],
            circularity_scores[selected_mode], target_x, target_y, grating_axis_deg,
            "" if degeneracy_delta is None else ", neff delta %.6g" % degeneracy_delta,
        )
    )
    return selection


def _reuse_verified_fiber_local_te_mode(fdtd, port_path, port, source_selection):
    """Reuse the source's verified polarization at a concentric passive plane.

    The automatic passive plane is only 0.1 um below the source and has the
    same fiber, tilt, span, and axis.  Calculating both degenerate partners a
    second time is redundant.  Calculate only the source winner here, verify
    its local-TE fraction, and fall back to a full three-candidate search if the local
    mode numbering unexpectedly changes.
    """
    selected_mode = max(1, int(source_selection.get("mode number", 1)))
    fdtd.select(port_path)
    update_status = fdtd.updateportmodes(selected_mode)
    if update_status is not None and float(np.asarray(update_status).squeeze()) < 0.0:
        raise RuntimeError(
            "Lumerical could not calculate inherited fiber mode %d for %s"
            % (selected_mode, port.get("name", port_path))
        )
    mode_profile = fdtd.getresult(port_path, "mode profiles")
    grating_axis_deg = float(port.get("angle phi", 0.0)) % 360.0
    local_te_score = _fiber_local_te_score(
        mode_profile, selected_mode, grating_axis_deg
    )
    minimum_fraction = float(port.get("minimum local TE fraction", 0.8))
    if local_te_score < minimum_fraction:
        print(
            "Passive fiber mode %d at %s is not the source's local-TE partner "
            "(score %.6f); running the full three-candidate fallback."
            % (selected_mode, port.get("name", port_path), local_te_score)
        )
        return _select_fiber_local_te_mode(fdtd, port_path, port)
    selection = dict(source_selection)
    selection.update({
        "mode number": int(selected_mode),
        "selected mode order": [int(selected_mode)],
        "selected mode result number": int(selected_mode),
        "local TE scores": {str(selected_mode): float(local_te_score)},
        "grating axis deg": float(grating_axis_deg),
        "polarization": "local TE",
        "inherited from source mode": True,
    })
    print(
        "Passive fiber port %s reused verified source eigensolver mode %d: "
        "local-TE score %.6f; skipped redundant partner calculation."
        % (port.get("name", port_path), selected_mode, local_te_score)
    )
    return selection


PORT_MODE_SELECTIONS = {}


def _add_ports(
    fdtd, z_center_um, device_top_um, stack_top_um,
    silica_cladding_top_um, silica_cladding_center_um,
):
    used_names = set()
    for index, port in enumerate(PORTS, start=1):
        name = str(port.get("name") or f"opt_{index}")
        if name in used_names:
            name = f"uid_{port.get('component_uid', 0)}_{name}"
        used_names.add(name)
        actual = float(port.get("outward_orientation_deg", 0.0)) % 360.0
        distance_um = float(port.get("distance_um", 0.0))
        x_um, y_um = map(float, port.get("center", (0.0, 0.0)))
        nearest, axis, direction = _nearest_port_axis(actual)
        plane_normal = str(port.get("plane normal", "")).upper()
        if plane_normal == "Z":
            axis = "z-axis"
        elif plane_normal == "Y":
            axis = "y-axis"
        elif plane_normal == "X":
            axis = "x-axis"
        if axis == "z-axis":
            z_um = _vertical_reference_um(
                port, device_top_um, stack_top_um,
                silica_cladding_top_um, silica_cladding_center_um,
            ) + distance_um
            direction = "Backward"
            parent_uid = int(port.get("parent_component_uid", -1))
            matching_fibers = [
                fiber for fiber in FIBER_GEOMETRIES
                if int(fiber.get("parent_component_uid", -2)) == parent_uid
            ]
            if not matching_fibers and FIBER_GEOMETRIES:
                matching_fibers = list(FIBER_GEOMETRIES)
            if matching_fibers:
                fiber = min(
                    matching_fibers,
                    key=lambda candidate: float(np.linalg.norm(
                        np.asarray(candidate.get("center", (0.0, 0.0)), dtype=float)
                        - np.asarray((x_um, y_um), dtype=float)
                    )),
                )
                fiber_z_um = _vertical_reference_um(
                    fiber, device_top_um, stack_top_um,
                    silica_cladding_top_um, silica_cladding_center_um,
                ) + float(fiber.get("distance_um", 0.0))
                fiber_theta_deg = float(fiber["angle theta"])
                fiber_phi_deg = float(fiber.get("angle phi", port.get("angle phi", 0.0)))
                axial_height_um = z_um - fiber_z_um
                lateral_um = axial_height_um * np.tan(np.deg2rad(fiber_theta_deg))
                expected_x_um = float(fiber.get("center", (0.0, 0.0))[0]) + lateral_um * np.cos(np.deg2rad(fiber_phi_deg))
                expected_y_um = float(fiber.get("center", (0.0, 0.0))[1]) + lateral_um * np.sin(np.deg2rad(fiber_phi_deg))
                concentric_error_um = float(np.hypot(x_um - expected_x_um, y_um - expected_y_um))
                if concentric_error_um > 1e-6:
                    raise RuntimeError(
                        "Fiber port %s is not concentric with %s at its tilted cross-section: %.9g um error"
                        % (name, fiber.get("name", "fiber"), concentric_error_um)
                    )
                print(
                    "Verified fiber/port concentricity %s: %.3g um center error."
                    % (name, concentric_error_um)
                )
        else:
            x_um += distance_um * np.cos(np.deg2rad(actual))
            y_um += distance_um * np.sin(np.deg2rad(actual))
            z_um = z_center_um
        if abs(((actual - nearest + 180.0) % 360.0) - 180.0) > 1e-6:
            print(f"Warning: {name} at {actual:g}° was mapped to the nearest FDTD port axis ({nearest}°).")
        requested_direction = str(port.get("dir", direction)).strip().capitalize()
        if requested_direction in {"Forward", "Backward"}:
            direction = requested_direction
        fdtd.addport()
        fdtd.set("name", name)
        fdtd.set("direction", direction)
        fdtd.set("injection axis", axis)
        fdtd.set("x", x_um * UM)
        fdtd.set("y", y_um * UM)
        fdtd.set("z", z_um * UM)
        if axis == "x-axis":
            fdtd.set("y span", float(port.get("span_um", 2.0)) * UM)
            fdtd.set("z span", float(port.get("z_span_um", 2.0)) * UM)
        elif axis == "y-axis":
            fdtd.set("x span", float(port.get("span_um", 2.0)) * UM)
            fdtd.set("z span", float(port.get("z_span_um", 2.0)) * UM)
        else:
            fdtd.set("x span", float(port.get("span_um", 2.0)) * UM)
            fdtd.set("y span", float(port.get("span_um", 2.0)) * UM)
            # The editor/JSON uses descriptive angle keys, but the standard
            # Ansys FDTD port object's actual lumapi properties are theta/phi.
            # This is the same mapping used by the official 3D grating setup.
            theta_deg = float(port.get("angle theta", 0.0))
            phi_deg = float(port.get("angle phi", actual))
            fdtd.set("theta", theta_deg)
            if abs(phi_deg) > 1e-12:
                fdtd.set("phi", phi_deg)
            fdtd.set("rotation offset", float(port.get("rotation offset_um", 0.0)) * UM)
        fiber_plane_role = str(port.get("fiber plane role", "")).strip().lower()
        grating_fiber_names = set()
        if GRATING_ANALYSIS:
            grating_fiber_names = {
                str(GRATING_ANALYSIS.get("fiber_port_name", "")),
            }
        is_fiber_mode_port = bool(
            axis == "z-axis"
            and (
                name in grating_fiber_names
                or bool(fiber_plane_role)
            )
        )
        requested_mode_number = max(0, int(port.get("mode number", 0)))
        mode_selection = (
            "user select" if is_fiber_mode_port or requested_mode_number
            else str(port.get("mode", "fundamental TE mode"))
        )
        fdtd.set("mode selection", mode_selection)
        if GRATING_ANALYSIS:
            # Match the official Ansys 3D grating-port profile treatment.
            # A single central-frequency profile is reused across the sweep;
            # otherwise getresult(..., "T") can trigger expensive modal
            # post-processing at every sampled wavelength.
            fdtd.set("frequency dependent profile", False)
        # Force the embedded eigensolver to store modal profiles now. Without
        # this step a programmatically created tilted port can finish an FDTD
        # solve yet expose neither S nor expansion results.
        port_path = "FDTD::ports::" + name
        fdtd.select(port_path)
        if is_fiber_mode_port:
            fiber_selection = _select_fiber_local_te_mode(fdtd, port_path, port)
            PORT_MODE_SELECTIONS[name] = fiber_selection
            requested_mode_number = int(fiber_selection["mode number"])
            port["mode number"] = requested_mode_number
            port["polarization"] = "local TE"
            port["selected mode order"] = list(
                fiber_selection["selected mode order"]
            )
            if GRATING_ANALYSIS and name == str(GRATING_ANALYSIS.get("fiber_port_name", "")):
                GRATING_ANALYSIS["fiber_source_mode"] = "mode %d" % requested_mode_number
                GRATING_ANALYSIS["fiber_source_mode_number"] = requested_mode_number
                GRATING_ANALYSIS["fiber_polarization"] = "local TE"
                GRATING_ANALYSIS["fiber_selected_mode_order"] = list(
                    fiber_selection["selected mode order"]
                )
        else:
            mode_update_status = (
                fdtd.updateportmodes(requested_mode_number)
                if requested_mode_number else fdtd.updateportmodes()
            )
            if mode_update_status is not None and float(np.asarray(mode_update_status).squeeze()) < 0.0:
                raise RuntimeError("Lumerical could not calculate the selected mode for FDTD port " + name)
        if not bool(fdtd.haveresult(port_path, "mode profiles")):
            raise RuntimeError(
                "FDTD port %s has no mode profile after updateportmodes; enlarge/reposition the port so it crosses its waveguide or fiber core"
                % name
            )
        target_neff = float(port.get("target neff", 0.0))
        automatic_waveguide_port = bool(
            (
                MMI_ANALYSIS
                and name in {
                    str(MMI_ANALYSIS.get("input_port_name", "")),
                    *map(str, MMI_ANALYSIS.get("output_port_names", [])),
                }
            )
            or (
                GRATING_ANALYSIS
                and axis != "z-axis"
                and int(port.get("parent_component_uid", -1))
                == int(GRATING_ANALYSIS.get("component_uid", -2))
            )
        )
        if automatic_waveguide_port:
            estimate = dict(globals().get("WAVEGUIDE_INDEX_ESTIMATE", {}))
            target_neff = float(estimate.get("target_neff", target_neff))
            port["target neff"] = target_neff
            port["target neff strategy"] = str(
                estimate.get("strategy", "material-index midpoint")
            )
        if MMI_ANALYSIS and name in {
            str(MMI_ANALYSIS.get("input_port_name", "")),
            *map(str, MMI_ANALYSIS.get("output_port_names", [])),
        }:
            MMI_ANALYSIS["port_target_neff"] = target_neff
        if target_neff > 0.0:
            neff_tolerance = max(0.0, float(port.get("neff tolerance", 0.3)))
            try:
                neff_data = fdtd.getresult(port_path, "neff")
                neff_key = "neff" if "neff" in neff_data else next(
                    key for key in neff_data.keys()
                    if str(key).lower().replace("_", " ").strip() == "neff"
                )
                selected_neff_values = np.real(
                    np.squeeze(np.asarray(neff_data[neff_key]))
                ).ravel()
                finite_neff = selected_neff_values[np.isfinite(selected_neff_values)]
            except Exception as exc:
                raise RuntimeError(
                    "FDTD port %s did not expose its selected effective index after mode calculation: %s"
                    % (name, exc)
                ) from None
            if finite_neff.size < 1:
                raise RuntimeError("FDTD port %s returned no finite effective index" % name)
            selected_neff = float(np.median(finite_neff))
            if abs(selected_neff - target_neff) > neff_tolerance:
                raise RuntimeError(
                    "FDTD port %s selected neff %.6g, outside the automatic access-waveguide "
                    "material-derived target %.6g +/- %.6g. Verify that the port crosses the intended access "
                    "waveguide and that the material stack describes that waveguide."
                    % (name, selected_neff, target_neff, neff_tolerance)
                )
            PORT_MODE_SELECTIONS[name] = {
                "neff": selected_neff,
                "target neff": target_neff,
                "neff tolerance": neff_tolerance,
                "mode number": requested_mode_number or 1,
                "polarization": str(port.get("polarization", "")),
            }
            print(
                "Validated FDTD port %s mode: neff %.6g around shared material-derived target %.6g."
                % (name, selected_neff, target_neff)
            )
        if is_fiber_mode_port:
            retained_modes = list(
                PORT_MODE_SELECTIONS.get(name, {}).get(
                    "selected mode order", [requested_mode_number]
                )
            )
            print(
                "Selected rotation-aware local TE on FDTD port %s using eigensolver mode %d; retained mode data %r."
                % (name, requested_mode_number, retained_modes)
            )
        elif requested_mode_number:
            print(
                "Selected %s polarization on FDTD port %s using eigensolver mode %d."
                % (port.get("polarization", "requested"), name, requested_mode_number)
            )
        else:
            print("Updated selected modal data for FDTD port " + name)

    if MMI_ANALYSIS:
        mmi_port_names = [
            str(MMI_ANALYSIS["input_port_name"]),
            *list(map(str, MMI_ANALYSIS["output_port_names"])),
        ]
        missing_selections = [
            name for name in mmi_port_names if name not in PORT_MODE_SELECTIONS
        ]
        if missing_selections:
            raise RuntimeError(
                "MMI port effective-index validation was not completed for: "
                + ", ".join(missing_selections)
            )
        selected_indices = [
            float(PORT_MODE_SELECTIONS[name]["neff"]) for name in mmi_port_names
        ]
        shared_tolerance = max(
            0.0, float(MMI_ANALYSIS.get("port_neff_tolerance", 0.3))
        )
        index_spread = max(selected_indices) - min(selected_indices)
        if index_spread > shared_tolerance:
            raise RuntimeError(
                "The three MMI access-port modes are not in the same effective-index family: "
                "%s (spread %.6g > %.6g)."
                % (
                    ", ".join(
                        "%s=%.6g" % (name, value)
                        for name, value in zip(mmi_port_names, selected_indices)
                    ),
                    index_spread,
                    shared_tolerance,
                )
            )
        print(
            "Verified common MMI access-port mode family: %s."
            % ", ".join(
                "%s neff %.6g" % (name, value)
                for name, value in zip(mmi_port_names, selected_indices)
            )
        )


WAVEGUIDE_MODE_SELECTIONS = {}


def _add_monitors(
    fdtd, z_center_um, device_top_um, stack_top_um,
    silica_cladding_top_um, silica_cladding_center_um,
):
    used_names = set()
    pending_expansions = []
    for index, monitor in enumerate(MONITORS, start=1):
        name = str(monitor.get("name") or f"monitor_{index}")
        if name in used_names:
            name = f"uid_{monitor.get('component_uid', 0)}_{name}"
        used_names.add(name)
        actual = float(monitor.get("orientation_deg", 0.0)) % 360.0
        distance_um = float(monitor.get("distance_um", 0.0))
        x_um, y_um = map(float, monitor.get("center", (0.0, 0.0)))
        nearest, axis, _ = _nearest_port_axis(actual)
        geometry_type = str(monitor.get("monitor geometry", "surface")).lower()
        plane_normal = str(monitor.get("plane normal", "")).upper()
        if plane_normal not in {"X", "Y", "Z"}:
            x_span = float(monitor.get("x span", monitor.get("span_um", 4.0)))
            y_span = float(monitor.get("y span", monitor.get("span_um", 4.0)))
            z_span = float(monitor.get("z span", monitor.get("z_span_um", 2.0)))
            plane_normal = min(((abs(x_span), "X"), (abs(y_span), "Y"), (abs(z_span), "Z")))[1]
        axis = {"X": "x-axis", "Y": "y-axis", "Z": "z-axis"}[plane_normal]
        if axis == "z-axis":
            z_reference = str(monitor.get("z reference", "device top")).strip().lower()
            z_um = (
                z_center_um
                if z_reference == "device center"
                else _vertical_reference_um(
                    monitor, device_top_um, stack_top_um,
                    silica_cladding_top_um, silica_cladding_center_um,
                )
            ) + distance_um
        else:
            x_um += distance_um * np.cos(np.deg2rad(actual))
            y_um += distance_um * np.sin(np.deg2rad(actual))
            z_um = z_center_um
        if geometry_type == "line":
            monitor_type = "Linear Y" if axis == "x-axis" else "Linear X"
        else:
            monitor_type = f"2D {plane_normal}-normal"
        monitor_kind = str(monitor.get("monitor_kind", "Power monitor"))
        if monitor_kind == "Mode expansion monitor":
            fdtd.addmodeexpansion()
        elif monitor_kind == "Field profile monitor":
            fdtd.addprofile()
        else:
            fdtd.addpower()
        fdtd.set("name", name)
        fdtd.set("monitor type", monitor_type)
        fdtd.set("x", x_um * UM)
        fdtd.set("y", y_um * UM)
        fdtd.set("z", z_um * UM)
        if (
            monitor_kind == "Power monitor"
            and str(monitor.get("fiber plane role", "")).strip().lower()
            == "input power measurement"
        ):
            # This monitor inherits its frequency range from the global
            # monitor settings configured after all monitors are created.
            # Setting "use source limits" locally while inheritance is active
            # raises "requested property is inactive" in Lumerical 2026 R1.
            fdtd.set("output power", True)
        if monitor_kind == "Field profile monitor":
            # The MMI single-run diagnostic plots |E|^2.  Make the recording
            # contract explicit because saved projects and Lumerical releases
            # can otherwise retain a subset of the Cartesian components.
            for component_name in ("Ex", "Ey", "Ez"):
                fdtd.set("output " + component_name, True)
        x_span = max(0.0, float(monitor.get("x span", 0.0 if axis == "x-axis" else monitor.get("span_um", 4.0))))
        y_span = max(0.0, float(monitor.get("y span", 0.0 if axis == "y-axis" else monitor.get("span_um", 4.0))))
        z_span = max(0.0, float(monitor.get("z span", 0.0 if axis == "z-axis" else monitor.get("z_span_um", 2.0))))
        # A surface monitor is a plane: the span along its normal is zero and is not set in Lumerical.
        if axis != "x-axis" and x_span > 0.0:
            fdtd.set("x span", x_span * UM)
        if axis != "y-axis" and y_span > 0.0:
            fdtd.set("y span", y_span * UM)
        if geometry_type != "line" and axis != "z-axis" and z_span > 0.0:
            fdtd.set("z span", z_span * UM)
        if monitor_kind == "Mode expansion monitor":
            target_neff = float(monitor.get("target neff", 0.0))
            if str(monitor.get("grating_monitor_role", "")) == "waveguide_mode_expansion":
                estimate = dict(globals().get("WAVEGUIDE_INDEX_ESTIMATE", {}))
                target_neff = float(estimate.get("target_neff", target_neff))
                monitor["target neff"] = target_neff
                monitor["target neff strategy"] = str(
                    estimate.get("strategy", "material-index midpoint")
                )
            if target_neff > 0.0:
                neff_tolerance = max(0.0, float(monitor.get("neff tolerance", 0.3)))
                # Use the same robust rule for every grating platform: select
                # the automatic fundamental (highest-index) mode.  Its target
                # is derived from the actual dispersive core and adjacent
                # dielectric indices, never a platform-specific constant. In FDTD
                # 2026 R1, "use max index" and "number of trial modes" are
                # read-only while this automatic selection is active; writing
                # either raises "requested property is inactive".  A fresh
                # monitor already uses the official max-index default, and
                # updatemodes() selects that highest-index mode when no stored
                # profile exists.  The material-derived target validates it.
                # Searching a small user-selected mode list near an entered n
                # can instead return the SiO2 slab/cladding mode (~1.42).
                fdtd.set("mode selection", "fundamental mode")
                fdtd.select(name)
                status = fdtd.updatemodes()
                if status is not None and float(np.asarray(status).squeeze()) < 0.0:
                    raise RuntimeError("Lumerical could not calculate the fundamental mode for " + name)
                neff_data = fdtd.getresult(name, "neff")
                selected_neff_values = np.real(
                    np.squeeze(np.asarray(neff_data["neff"]))
                ).ravel()
                finite_neff = selected_neff_values[np.isfinite(selected_neff_values)]
                if finite_neff.size < 1:
                    raise RuntimeError("Mode expansion monitor %s returned no effective indices" % name)
                # A mode-expansion monitor can return one neff sample or a
                # wavelength vector.  The median is the central-band value
                # for the monotonic, narrow grating sweep and is independent
                # of whether lumapi exposes wavelength as the first or last
                # array dimension.
                selected_neff = float(np.median(finite_neff))
                selected_mode_number = 1
                neff_error = abs(selected_neff - target_neff)
                if neff_error > neff_tolerance:
                    raise RuntimeError(
                        "Waveguide mode monitor %s found fundamental neff %.6g, outside the material-derived "
                        "target %.6g +/- %.6g. Verify that the intended waveguide cross-section crosses both "
                        "the power and mode-expansion planes and that the material stack is correct."
                        % (name, selected_neff, target_neff, neff_tolerance)
                    )
                WAVEGUIDE_MODE_SELECTIONS[name] = {
                    "mode number": selected_mode_number,
                    "neff": selected_neff,
                    "target neff": target_neff,
                    "neff tolerance": neff_tolerance,
                    "index estimate": dict(
                        globals().get("WAVEGUIDE_INDEX_ESTIMATE", {})
                    ),
                    "selection": "automatic fundamental highest-index mode",
                }
                print(
                    "Selected fundamental highest-index waveguide mode %d on %s: "
                    "neff %.6g (material-derived target %.6g)."
                    % (selected_mode_number, name, selected_neff, target_neff)
                )
            else:
                fdtd.set("mode selection", str(monitor.get("mode", "fundamental TE mode")))
                fdtd.select(name)
                fdtd.updatemodes()
            expansion_for = str(monitor.get("expansion for", "")).strip()
            if expansion_for:
                pending_expansions.append(
                    (
                        name,
                        str(monitor.get("expansion result name", "input")),
                        expansion_for,
                    )
                )

    for expansion_name, result_name, input_monitor_name in pending_expansions:
        fdtd.select(expansion_name)
        fdtd.setexpansion(result_name, input_monitor_name)
        print(
            "Mode expansion %s analyzes power monitor %s as result %s."
            % (expansion_name, input_monitor_name, result_name)
        )


def build_simulation():
    build_started = time.perf_counter()
    build_timings = {}
    available_cpu_cores = os.cpu_count() or 1
    build_cpu_threads = max(
        1,
        min(int(SETTINGS.get("build_cpu_threads", 30)), available_cpu_cores),
    )
    stage_started = time.perf_counter()
    fdtd = lumapi.FDTD(
        hide=bool(SETTINGS.get("hide_cad", True)),
        serverArgs={"threads": str(build_cpu_threads)},
    )
    # Geometry, Layer Builder setup, meshing, and all embedded eigensolvers
    # are CPU work.  Configure that explicitly before any model object is
    # created; the GPU row is activated only later, immediately before run().
    fdtd.setresource("FDTD", 1, "device type", "CPU")
    fdtd.setresource("FDTD", 1, "active", True)
    fdtd.setresource("FDTD", 1, "processes", 1)
    fdtd.setresource("FDTD", 1, "threads", build_cpu_threads)
    build_timings["FDTD startup"] = time.perf_counter() - stage_started
    print(
        "Model construction CPU allocation: %d thread%s"
        % (build_cpu_threads, "" if build_cpu_threads == 1 else "s")
    )
    stage_started = time.perf_counter()
    _add_required_materials(fdtd)
    build_timings["materials"] = time.perf_counter() - stage_started
    bounds = list(BOUNDING_BOX_UM)
    if SETTINGS.get("include_ports", True):
        for port in PORTS:
            actual = float(port.get("outward_orientation_deg", 0.0)) % 360.0
            distance_um = float(port.get("distance_um", 0.0))
            x_um, y_um = map(float, port.get("center", (0.0, 0.0)))
            if str(port.get("plane normal", "X")).upper() != "Z":
                x_um += distance_um * np.cos(np.deg2rad(actual))
                y_um += distance_um * np.sin(np.deg2rad(actual))
            half_span = float(port.get("span_um", 2.0)) / 2.0
            plane_normal = str(port.get("plane normal", "X")).upper()
            x_half = 0.0 if plane_normal == "X" else half_span
            y_half = 0.0 if plane_normal == "Y" else half_span
            bounds[0] = min(bounds[0], x_um - x_half)
            bounds[1] = min(bounds[1], y_um - y_half)
            bounds[2] = max(bounds[2], x_um + x_half)
            bounds[3] = max(bounds[3], y_um + y_half)
    for source in GAUSSIAN_SOURCES:
        x_um, y_um = map(float, source.get("center", (0.0, 0.0)))
        half_span_um = 0.5 * max(1e-6, float(source.get("span_um", 20.0)))
        bounds[0] = min(bounds[0], x_um - half_span_um)
        bounds[1] = min(bounds[1], y_um - half_span_um)
        bounds[2] = max(bounds[2], x_um + half_span_um)
        bounds[3] = max(bounds[3], y_um + half_span_um)
    for monitor in MONITORS:
        actual = float(monitor.get("orientation_deg", 0.0)) % 360.0
        distance_um = float(monitor.get("distance_um", 0.0))
        x_um, y_um = map(float, monitor.get("center", (0.0, 0.0)))
        if str(monitor.get("plane normal", "X")).upper() != "Z":
            x_um += distance_um * np.cos(np.deg2rad(actual))
            y_um += distance_um * np.sin(np.deg2rad(actual))
        plane_normal = str(monitor.get("plane normal", "X")).upper()
        fallback_span = float(monitor.get("span_um", 4.0))
        x_half = 0.0 if plane_normal == "X" else 0.5 * max(fallback_span, float(monitor.get("x span", 0.0)))
        y_half = 0.0 if plane_normal == "Y" else 0.5 * max(fallback_span, float(monitor.get("y span", 0.0)))
        bounds[0] = min(bounds[0], x_um - x_half)
        bounds[1] = min(bounds[1], y_um - y_half)
        bounds[2] = max(bounds[2], x_um + x_half)
        bounds[3] = max(bounds[3], y_um + y_half)
    pml_geometry_overlap_um = max(0.0, float(SETTINGS.get("pml_geometry_overlap_um", 1.0)))
    domain_padding = dict(SETTINGS.get("domain_padding_um", {}))
    legacy_xy_padding = float(SETTINGS.get("xy_padding_um", 2.0))
    x_min_padding = float(domain_padding.get("x_min", legacy_xy_padding))
    x_max_padding = float(domain_padding.get("x_max", legacy_xy_padding))
    y_min_padding = float(domain_padding.get("y_min", legacy_xy_padding))
    y_max_padding = float(domain_padding.get("y_max", legacy_xy_padding))
    bounds = [
        bounds[0] - x_min_padding,
        bounds[1] - y_min_padding,
        bounds[2] + x_max_padding,
        bounds[3] + y_max_padding,
    ]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("The freely positioned FDTD X/Y bounds must have positive spans")
    z_ranges = _stack_z_ranges(MATERIAL_STACK)
    if GRATING_ANALYSIS or MMI_ANALYSIS:
        waveguide_index_estimate = _derive_waveguide_neff_from_stack(fdtd, z_ranges)
        globals()["WAVEGUIDE_INDEX_ESTIMATE"] = waveguide_index_estimate
        if GRATING_ANALYSIS:
            GRATING_ANALYSIS["waveguide_target_neff"] = float(
                waveguide_index_estimate["target_neff"]
            )
            GRATING_ANALYSIS["waveguide_index_estimate"] = dict(
                waveguide_index_estimate
            )
        if MMI_ANALYSIS:
            MMI_ANALYSIS["port_target_neff"] = float(
                waveguide_index_estimate["target_neff"]
            )
            MMI_ANALYSIS["waveguide_index_estimate"] = dict(
                waveguide_index_estimate
            )
    if not z_ranges:
        print("Warning: every stack thickness is zero; no material objects will be added.")
        z_min_um, z_max_um, device_z_um, device_top_um = -1.0, 1.0, 0.0, 0.0
    else:
        z_min_um = z_ranges[0][1]
        z_max_um = z_ranges[-1][2]
        geometry_ranges = [(z0, z1) for row, z0, z1 in z_ranges if row.get("role") == "geometry"]
        device_z_um = (
            0.5 * (min(z0 for z0, _z1 in geometry_ranges) + max(z1 for _z0, z1 in geometry_ranges))
            if geometry_ranges else (z_min_um + z_max_um) / 2.0
        )
        device_top_um = max((z1 for z0, z1 in geometry_ranges), default=device_z_um)
    stack_top_um = z_max_um
    silica_cladding_top_um = _silica_cladding_top_um(z_ranges, device_top_um)
    silica_cladding_center_um = _silica_cladding_center_um(z_ranges, device_top_um)

    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("background index", 1.0)
    fdtd.set("x", 0.5 * (bounds[0] + bounds[2]) * UM)
    fdtd.set("y", 0.5 * (bounds[1] + bounds[3]) * UM)
    fdtd.set("x span", (bounds[2] - bounds[0]) * UM)
    fdtd.set("y span", (bounds[3] - bounds[1]) * UM)
    # The solver rejects sources, ports, and monitors that touch or cross an
    # FDTD boundary.  Include their real Z extents before applying padding.
    z_extent_min_um = z_min_um
    z_extent_max_um = z_max_um
    if len(z_ranges) > 1 and str(z_ranges[0][0].get("role", "background")) == "background":
        z_extent_min_um = z_ranges[0][2]
    if len(z_ranges) > 1 and str(z_ranges[-1][0].get("role", "background")) == "background":
        z_extent_max_um = z_ranges[-1][1]
    sampling_z_extents = []
    if SETTINGS.get("include_ports", True):
        for port in PORTS:
            sampling_name = "FDTD port %s" % str(port.get("name", "unnamed"))
            if str(port.get("plane normal", "X")).upper() == "Z":
                source_z_um = _vertical_reference_um(
                    port, device_top_um, stack_top_um,
                    silica_cladding_top_um, silica_cladding_center_um,
                ) + float(port.get("distance_um", 0.0))
                sampling_z_min_um = source_z_um
                sampling_z_max_um = source_z_um
            else:
                plane_normal = str(port.get("plane normal", "")).upper()
                if plane_normal not in {"X", "Y", "Z"}:
                    nearest, _, _ = _nearest_port_axis(float(port.get("outward_orientation_deg", 0.0)))
                    plane_normal = "X" if nearest in (0, 180) else "Y"
                z_span_um = max(0.0, float(port.get("z_span_um", 2.0)))
                sampling_z_min_um = device_z_um - 0.5 * z_span_um
                sampling_z_max_um = device_z_um + 0.5 * z_span_um
            sampling_z_extents.append(
                (sampling_name, sampling_z_min_um, sampling_z_max_um)
            )
            z_extent_min_um = min(z_extent_min_um, sampling_z_min_um)
            z_extent_max_um = max(z_extent_max_um, sampling_z_max_um)
    for source in GAUSSIAN_SOURCES:
        source_z_um = _vertical_reference_um(
            source, device_top_um, stack_top_um,
            silica_cladding_top_um, silica_cladding_center_um,
        ) + float(source.get("distance_um", 0.0))
        sampling_z_extents.append(
            ("Gaussian source %s" % str(source.get("name", "unnamed")), source_z_um, source_z_um)
        )
        z_extent_min_um = min(z_extent_min_um, source_z_um)
        z_extent_max_um = max(z_extent_max_um, source_z_um)
    for monitor in MONITORS:
        sampling_name = "%s %s" % (
            str(monitor.get("monitor_kind", "monitor")),
            str(monitor.get("name", "unnamed")),
        )
        geometry_type = str(monitor.get("monitor geometry", "surface")).lower()
        plane_normal = str(monitor.get("plane normal", "")).upper()
        if plane_normal not in {"X", "Y", "Z"}:
            x_span = float(monitor.get("x span", monitor.get("span_um", 4.0)))
            y_span = float(monitor.get("y span", monitor.get("span_um", 4.0)))
            z_span = float(monitor.get("z span", monitor.get("z_span_um", 2.0)))
            plane_normal = min(((abs(x_span), "X"), (abs(y_span), "Y"), (abs(z_span), "Z")))[1]
        if plane_normal == "Z":
            monitor_z_um = _vertical_reference_um(
                monitor, device_top_um, stack_top_um,
                silica_cladding_top_um, silica_cladding_center_um,
            ) + float(monitor.get("distance_um", 0.0))
            sampling_z_min_um = monitor_z_um
            sampling_z_max_um = monitor_z_um
        else:
            z_span_um = 0.0
            if geometry_type != "line":
                z_span_um = max(0.0, float(monitor.get("z span", monitor.get("z_span_um", 2.0))))
            sampling_z_min_um = device_z_um - 0.5 * z_span_um
            sampling_z_max_um = device_z_um + 0.5 * z_span_um
        sampling_z_extents.append(
            (sampling_name, sampling_z_min_um, sampling_z_max_um)
        )
        z_extent_min_um = min(z_extent_min_um, sampling_z_min_um)
        z_extent_max_um = max(z_extent_max_um, sampling_z_max_um)
    fixed_sampling_z_bounds = SETTINGS.get("fixed_sampling_z_bounds_um")
    fixed_sampling_label = "optimization"
    if fixed_sampling_z_bounds is None:
        fixed_sampling_z_bounds = SETTINGS.get("sweep_sampling_z_bounds_um")
        fixed_sampling_label = "sweep"
    if isinstance(fixed_sampling_z_bounds, (list, tuple)) and len(fixed_sampling_z_bounds) == 2:
        fixed_z_min_um, fixed_z_max_um = map(float, fixed_sampling_z_bounds)
        if not np.all(np.isfinite([fixed_z_min_um, fixed_z_max_um])) or fixed_z_max_um < fixed_z_min_um:
            raise ValueError("fixed sampling Z bounds must contain finite ordered values")
        z_extent_min_um = min(z_extent_min_um, fixed_z_min_um)
        z_extent_max_um = max(z_extent_max_um, fixed_z_max_um)
        print(
            "Reserved fixed %s Z envelope [%.6g, %.6g] um for movable source/monitor planes."
            % (fixed_sampling_label, fixed_z_min_um, fixed_z_max_um)
        )
    legacy_z_padding = float(SETTINGS.get("z_padding_um", 1.0))
    requested_z_min_padding = float(domain_padding.get("z_min", legacy_z_padding))
    requested_z_max_padding = float(domain_padding.get("z_max", legacy_z_padding))
    # Honor the exact Z boundaries selected in the editor.  The lambda/4 value
    # is a convenient UI reset only; the model builder must never silently
    # enlarge a manually typed or dragged domain.
    simulation_z_min_um = z_extent_min_um - requested_z_min_padding
    simulation_z_max_um = z_extent_max_um + requested_z_max_padding
    if simulation_z_max_um <= simulation_z_min_um:
        raise ValueError("The freely positioned FDTD Z bounds must have a positive span")
    sampling_outside = [
        "%s [%.6g, %.6g] um" % (name, sample_min, sample_max)
        for name, sample_min, sample_max in sampling_z_extents
        if sample_min <= simulation_z_min_um or sample_max >= simulation_z_max_um
    ]
    if sampling_outside:
        raise ValueError(
            "FDTD Z bounds [%.6g, %.6g] um place these ports/monitors on or "
            "outside the boundary: %s. Expand the manually selected Z-min/Z-max "
            "bounds so every sampling object is strictly inside."
            % (
                simulation_z_min_um,
                simulation_z_max_um,
                "; ".join(sampling_outside),
            )
        )
    fdtd.set("z min", simulation_z_min_um * UM)
    fdtd.set("z max", simulation_z_max_um * UM)
    fdtd.set("mesh accuracy", int(SETTINGS.get("mesh_accuracy", 2)))
    dt_stability_factor = min(0.99, max(0.1, float(SETTINGS.get("dt_stability_factor", 0.99))))
    fdtd.set("dt stability factor", dt_stability_factor)
    for boundary_property in ("x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"):
        fdtd.set(boundary_property, "PML")
    antisymmetry_boundary = str(SETTINGS.get("antisymmetry_boundary", "")).strip().lower()
    if bool(SETTINGS.get("use_y_antisymmetry", False)) and antisymmetry_boundary in {
        "x min", "x max", "y min", "y max"
    }:
        fdtd.set(antisymmetry_boundary + " bc", "Anti-Symmetric")
        print("Enabled official grating symmetry at the %s boundary." % antisymmetry_boundary.upper())
    pml_profile_name = str(SETTINGS.get("pml_profile", "standard")).strip().lower()
    if pml_profile_name not in {"standard", "stabilized"}:
        raise ValueError("pml_profile must be Standard or Stabilized")
    fdtd.set("pml profile", 2 if pml_profile_name == "stabilized" else 1)
    fdtd.set("auto scale pml parameters", False if GRATING_ANALYSIS else True)
    simulation_time_fs = max(1.0, float(SETTINGS.get("simulation_time_fs", 10000.0)))
    auto_shutoff_min = min(1.0, max(1e-12, float(SETTINGS.get("auto_shutoff_min", 1e-6))))
    fdtd.set("simulation time", simulation_time_fs * 1e-15)
    fdtd.set("auto shutoff min", auto_shutoff_min)
    print(
        "FDTD stability: dt factor %.3g, %s PML, %.6g ps maximum, auto shutoff %.3g"
        % (dt_stability_factor, pml_profile_name, simulation_time_fs * 1e-3, auto_shutoff_min)
    )
    stack_mesh_summary = [
        "%s %d" % (
            str(row.get("name", row.get("material", "layer"))),
            max(
                1,
                int(row.get(
                    "mesh_order", 3 if bool(row.get("conformal", False)) else 2
                )),
            ),
        )
        for row in _active_stack(MATERIAL_STACK)
    ]
    print(
        "Material mesh orders: %s; fiber core 4; fiber cladding 5."
        % "; ".join(stack_mesh_summary)
    )

    stage_started = time.perf_counter()
    if z_ranges:
        _add_material_stack(
            fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um, pml_geometry_overlap_um
        )
        _add_layer_mesh_overrides(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um)
    build_timings["stack and mesh"] = time.perf_counter() - stage_started
    # Port eigensolvers must see the requested wavelength range before their
    # mode profiles are explicitly updated.
    fdtd.setglobalsource("wavelength start", float(SETTINGS.get("wavelength_start_um", 1.25)) * UM)
    fdtd.setglobalsource("wavelength stop", float(SETTINGS.get("wavelength_stop_um", 1.35)) * UM)
    stage_started = time.perf_counter()
    _add_fiber_geometries(
        fdtd, device_top_um, stack_top_um,
        silica_cladding_top_um, silica_cladding_center_um,
    )
    if FIBER_GEOMETRIES or PORTS or any(
        str(monitor.get("monitor_kind", "")) == "Mode expansion monitor"
        for monitor in MONITORS
    ):
        # Force every structure/property-group setup script to finish in the
        # live CAD session before asking an embedded port/monitor eigensolver
        # for modes.  This is the in-memory equivalent of the former temporary
        # mode-seed FSP save and avoids a disk write in pure Fast Run mode.
        fdtd.runsetup()
        print("Committed geometry in memory before embedded mode calculations.")
    build_timings["fiber geometry and one setup pass"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    _add_gaussian_sources(
        fdtd, device_top_um, stack_top_um,
        silica_cladding_top_um, silica_cladding_center_um,
    )
    build_timings["Gaussian sources"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if SETTINGS.get("include_ports", True):
        _add_ports(
            fdtd, device_z_um, device_top_um, stack_top_um,
            silica_cladding_top_um, silica_cladding_center_um,
        )
        if PORTS:
            fdtd.select("FDTD::ports")
            fdtd.set("monitor frequency points", int(SETTINGS.get("frequency_points", 31)))
    build_timings["FDTD port modes"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    _add_monitors(
        fdtd, device_z_um, device_top_um, stack_top_um,
        silica_cladding_top_um, silica_cladding_center_um,
    )
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("frequency points", int(SETTINGS.get("frequency_points", 31)))
    build_timings["monitors and expansion modes"] = time.perf_counter() - stage_started
    model_bounds_um = [
        float(bounds[0]), float(bounds[1]), float(simulation_z_min_um),
        float(bounds[2]), float(bounds[3]), float(simulation_z_max_um),
    ]
    build_timings["total"] = time.perf_counter() - build_started
    print(
        "Pre-solve CPU timing: "
        + "; ".join(
            "%s %.2f s" % (name, seconds)
            for name, seconds in build_timings.items()
        )
    )
    return fdtd, model_bounds_um


REMOTE_RUNTIME_PROJECT_FILE = ""
fdtd, MODEL_BOUNDS_UM = build_simulation()
print("Built directly in memory; exact-model caching is disabled.")

ACTUAL_FDTD_DIMENSION = str(fdtd.getnamed("FDTD", "dimension"))
if ACTUAL_FDTD_DIMENSION.upper() != "3D":
    raise RuntimeError("Max Layout requires a 3D FDTD region, but Lumerical reported " + ACTUAL_FDTD_DIMENSION)
print(f"Built a verified 3D model with {len(GEOMETRY)} polygons, {len(PORTS)} standard FDTD ports, {len(FIBER_GEOMETRIES)} fiber geometry groups, {len(GAUSSIAN_SOURCES)} Gaussian sources, {len(MONITORS)} monitors, and {len(_active_stack(MATERIAL_STACK))} active stack layers.")
'''


_GEOMETRY_PROJECTIONS_REMOTE = r'''# Render the exact embedded XY polygons and their Layer Builder XZ/YZ process projections.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle

# Read the bounds back from the model that was actually built.  Do not rely on
# MODEL_BOUNDS_UM surviving as a Python variable between separate Lambda.run
# stages; the persistent FDTD session is the source of truth for the preview.
try:
    MODEL_BOUNDS_UM = [
        float(np.asarray(fdtd.getnamed("FDTD", property_name)).squeeze()) / 1e-6
        for property_name in ("x min", "y min", "z min", "x max", "y max", "z max")
    ]
except NameError as exc:
    raise RuntimeError(
        "The built FDTD model is unavailable. Run the model-build cell before rendering its geometry."
    ) from exc
except Exception as exc:
    raise RuntimeError("Could not read the actual FDTD bounds for geometry verification: %s" % exc) from exc
if (
    len(MODEL_BOUNDS_UM) != 6
    or not np.all(np.isfinite(MODEL_BOUNDS_UM))
    or MODEL_BOUNDS_UM[3] <= MODEL_BOUNDS_UM[0]
    or MODEL_BOUNDS_UM[4] <= MODEL_BOUNDS_UM[1]
    or MODEL_BOUNDS_UM[5] <= MODEL_BOUNDS_UM[2]
):
    raise RuntimeError("The built FDTD region returned invalid geometry-verification bounds: %r" % MODEL_BOUNDS_UM)


def _projection_intervals(coordinate_index, stack_row):
    target_layers = {int(value) for value in stack_row.get("gds_layers", [])}
    intervals = []
    for polygon in GEOMETRY:
        if int(polygon.get("layer", -1)) not in target_layers:
            continue
        values = np.asarray(polygon["vertices_um"], dtype=float)[:, coordinate_index]
        intervals.append((float(np.min(values)), float(np.max(values))))
    return intervals


_material_palette = {
    "Air": "#e0f2fe",
    "Si (Silicon) - Palik": "#64748b",
    "SiO2 (Glass) - Palik": "#bae6fd",
    "LiNbO3": "#14b8a6",
    "Al2O3": "#f59e0b",
    "Au (Gold) - CRC": "#facc15",
    "Al (Aluminium) - Palik": "#cbd5e1",
    "Ag (Silver) - Palik": "#e2e8f0",
}
_fallback_palette = ["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#16a34a", "#0891b2"]


def _material_color(material, index=0):
    return _material_palette.get(str(material), _fallback_palette[index % len(_fallback_palette)])


def _draw_process_projection(axis, coordinate_index, coordinate_label):
    coordinate_min = float(MODEL_BOUNDS_UM[coordinate_index])
    coordinate_max = float(MODEL_BOUNDS_UM[coordinate_index + 3])
    z_ranges = _stack_z_ranges(MATERIAL_STACK)
    legend_handles = []
    legend_materials = set()
    for row_index, (row, z0, z1) in enumerate(z_ranges):
        material = str(row.get("material", "material"))
        color = _material_color(material, row_index)
        if material not in legend_materials:
            legend_handles.append(Patch(facecolor=color, edgecolor="#334155", alpha=0.72, label=material))
            legend_materials.add(material)
        if str(row.get("role", "background")) != "geometry":
            draw_z0 = (
                _conformal_fill_start(z_ranges, row_index, z0)
                if bool(row.get("conformal", False)) else z0
            )
            axis.add_patch(Rectangle(
                (coordinate_min, draw_z0), coordinate_max - coordinate_min, z1 - draw_z0,
                facecolor=color, edgecolor="#475569", linewidth=0.5, alpha=0.30,
            ))
            continue

        thickness = float(z1 - z0)
        etch_depth = min(thickness, max(0.0, float(row.get("etch_depth_um", thickness))))
        patterned_z0 = float(z1 - etch_depth)
        if patterned_z0 > z0:
            if str(row.get("slab_extent", "full")).strip().lower() == "geometry":
                for interval_min, interval_max in _projection_intervals(coordinate_index, row):
                    axis.add_patch(Rectangle(
                        (interval_min, z0), interval_max - interval_min, patterned_z0 - z0,
                        facecolor=color, edgecolor="#475569", linewidth=0.5, alpha=0.48,
                    ))
            else:
                axis.add_patch(Rectangle(
                    (coordinate_min, z0), coordinate_max - coordinate_min, patterned_z0 - z0,
                    facecolor=color, edgecolor="#475569", linewidth=0.5, alpha=0.48,
                ))
        if etch_depth <= 0.0:
            continue

        angle_deg = min(179.999, max(0.001, float(row.get("sidewall_angle_deg", 90.0))))
        tangent = np.tan(np.deg2rad(angle_deg))
        half_offset = 0.0 if abs(tangent) < 1e-12 else 0.5 * etch_depth / tangent
        for interval_min, interval_max in _projection_intervals(coordinate_index, row):
            vertices = np.asarray([
                [interval_min - half_offset, patterned_z0],
                [interval_max + half_offset, patterned_z0],
                [interval_max - half_offset, z1],
                [interval_min + half_offset, z1],
            ])
            axis.add_patch(MplPolygon(
                vertices, closed=True, facecolor=color, edgecolor="#0f172a", linewidth=0.55, alpha=0.82,
            ))

    geometry_tops = [z1 for row, _, z1 in z_ranges if str(row.get("role", "background")) == "geometry"]
    device_top = max(geometry_tops, default=0.0)
    stack_top = z_ranges[-1][2] if z_ranges else device_top
    silica_cladding_top = _silica_cladding_top_um(z_ranges, device_top)
    silica_cladding_center = _silica_cladding_center_um(z_ranges, device_top)
    for fiber in FIBER_GEOMETRIES:
        x_um, y_um = map(float, fiber.get("center", (0.0, 0.0)))
        start_horizontal = x_um if coordinate_index == 0 else y_um
        start_z = _vertical_reference_um(
            fiber, device_top, stack_top, silica_cladding_top, silica_cladding_center
        ) + float(fiber.get("distance_um", 0.0))
        length = float(fiber.get("fiber length_um", 20.0))
        theta = np.deg2rad(float(fiber["angle theta"]))
        phi = np.deg2rad(float(fiber.get("angle phi", 0.0)))
        horizontal_delta = 0.5 * length * np.tan(theta) * (
            np.cos(phi) if coordinate_index == 0 else np.sin(phi)
        )
        axis.plot(
            [start_horizontal - horizontal_delta, start_horizontal + horizontal_delta],
            [start_z - 0.5 * length, start_z + 0.5 * length],
            color="#bae6fd", linewidth=10.0, alpha=0.06, solid_capstyle="round",
        )
        axis.plot(
            [start_horizontal - horizontal_delta, start_horizontal + horizontal_delta],
            [start_z - 0.5 * length, start_z + 0.5 * length],
            color="#0e7490", linewidth=4.0, alpha=0.35, solid_capstyle="round",
        )
    for source in GAUSSIAN_SOURCES:
        source_x_um, source_y_um = map(float, source.get("center", (0.0, 0.0)))
        source_horizontal_um = source_x_um if coordinate_index == 0 else source_y_um
        source_z_um = _vertical_reference_um(
            source, device_top, stack_top,
            silica_cladding_top, silica_cladding_center,
        ) + float(source.get("distance_um", 0.0))
        theta = np.deg2rad(float(source.get("angle theta", 0.0)))
        phi = np.deg2rad(float(source.get("angle phi", 0.0)))
        ray_length_um = min(8.0, max(2.0, float(source.get("span_um", 20.0)) / 3.0))
        horizontal_delta = -ray_length_um * np.sin(theta) * (
            np.cos(phi) if coordinate_index == 0 else np.sin(phi)
        )
        vertical_delta = -ray_length_um * np.cos(theta)
        axis.annotate(
            "",
            xy=(source_horizontal_um + horizontal_delta, source_z_um + vertical_delta),
            xytext=(source_horizontal_um, source_z_um),
            arrowprops=dict(arrowstyle="->", color="#c026d3", linewidth=2.0),
        )
        axis.scatter(
            [source_horizontal_um], [source_z_um], marker="o", s=36,
            facecolors="none", edgecolors="#c026d3", linewidths=1.3,
        )

    axis.add_patch(Rectangle(
        (coordinate_min, float(MODEL_BOUNDS_UM[2])),
        coordinate_max - coordinate_min,
        float(MODEL_BOUNDS_UM[5] - MODEL_BOUNDS_UM[2]),
        fill=False, edgecolor="#7c3aed", linewidth=1.4, linestyle="--",
    ))
    axis.set_xlim(coordinate_min, coordinate_max)
    axis.set_ylim(float(MODEL_BOUNDS_UM[2]), float(MODEL_BOUNDS_UM[5]))
    axis.set_xlabel(coordinate_label + " [µm]")
    axis.set_ylabel("Z [µm]")
    axis.grid(alpha=0.18, linewidth=0.4)
    axis.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.92)


def _placed_xy(item, orientation_key):
    x_um, y_um = map(float, item.get("center", (0.0, 0.0)))
    angle = float(item.get(orientation_key, 0.0)) % 360.0
    distance_um = float(item.get("distance_um", 0.0))
    x_um += distance_um * np.cos(np.deg2rad(angle))
    y_um += distance_um * np.sin(np.deg2rad(angle))
    return x_um, y_um


figure, (xy_axis, xz_axis, yz_axis) = plt.subplots(1, 3, figsize=(19, 6.4))

_layer_groups = {}
for polygon in GEOMETRY:
    key = (int(polygon.get("layer", 0)), int(polygon.get("datatype", 0)))
    _layer_groups.setdefault(key, []).append(np.asarray(polygon["vertices_um"], dtype=float))
for layer_index, (layer_key, polygons) in enumerate(sorted(_layer_groups.items())):
    color = _fallback_palette[layer_index % len(_fallback_palette)]
    for polygon_index, polygon in enumerate(polygons):
        xy_axis.add_patch(MplPolygon(
            polygon, closed=True, facecolor=color, edgecolor="#0f172a", linewidth=0.45, alpha=0.72,
            label=("GDS %d:%d" % layer_key) if polygon_index == 0 else None,
        ))

xy_axis.add_patch(Rectangle(
    (float(MODEL_BOUNDS_UM[0]), float(MODEL_BOUNDS_UM[1])),
    float(MODEL_BOUNDS_UM[3] - MODEL_BOUNDS_UM[0]),
    float(MODEL_BOUNDS_UM[4] - MODEL_BOUNDS_UM[1]),
    fill=False, edgecolor="#7c3aed", linewidth=1.4, linestyle="--", label="3D FDTD boundary",
))

for fiber in FIBER_GEOMETRIES:
    x_um, y_um = map(float, fiber.get("center", (0.0, 0.0)))
    cladding_radius = 0.5 * float(fiber.get("cladding diameter_um", 50.0))
    core_radius = 0.5 * float(fiber.get("core diameter_um", 9.0))
    xy_axis.add_patch(plt.Circle((x_um, y_um), cladding_radius, facecolor="#bae6fd", edgecolor="#7dd3fc", linewidth=0.8, linestyle=":", alpha=0.06))
    xy_axis.add_patch(plt.Circle((x_um, y_um), core_radius, facecolor="#0e7490", edgecolor="#155e75", linewidth=1.4, alpha=0.35))
    xy_axis.annotate(str(fiber.get("name", "fiber geometry")), (x_um, y_um), xytext=(4, 5), textcoords="offset points", fontsize=7, color="#0e7490")

for source in GAUSSIAN_SOURCES:
    x_um, y_um = map(float, source.get("center", (0.0, 0.0)))
    waist_um = float(source.get("waist radius_um", 4.5))
    phi = np.deg2rad(float(source.get("angle phi", 0.0)))
    xy_axis.add_patch(plt.Circle(
        (x_um, y_um), waist_um, facecolor="#e879f9", edgecolor="#a21caf",
        linewidth=1.4, alpha=0.20,
    ))
    xy_axis.arrow(
        x_um, y_um, -waist_um * np.cos(phi), -waist_um * np.sin(phi),
        color="#c026d3", width=0.03, head_width=max(0.2, 0.15 * waist_um),
        length_includes_head=True,
    )
    xy_axis.annotate(str(source.get("name", "Gaussian source")), (x_um, y_um), xytext=(4, 5), textcoords="offset points", fontsize=7, color="#86198f")

for port in PORTS:
    normal = str(port.get("plane normal", "X")).upper()
    x_um, y_um = (
        tuple(map(float, port.get("center", (0.0, 0.0))))
        if normal == "Z"
        else _placed_xy(port, "outward_orientation_deg")
    )
    span = float(port.get("span_um", 2.0))
    if normal == "Y":
        xy_axis.plot([x_um - span / 2.0, x_um + span / 2.0], [y_um, y_um], color="#dc2626", linewidth=2.0)
    elif normal == "Z":
        xy_axis.scatter([x_um], [y_um], marker="s", s=70, facecolors="none", edgecolors="#dc2626", linewidths=1.8)
    else:
        xy_axis.plot([x_um, x_um], [y_um - span / 2.0, y_um + span / 2.0], color="#dc2626", linewidth=2.0)
    xy_axis.annotate(str(port.get("name", "port")), (x_um, y_um), xytext=(4, 5), textcoords="offset points", fontsize=7, color="#991b1b")

for monitor in MONITORS:
    normal = str(monitor.get("plane normal", "X")).upper()
    x_um, y_um = (
        tuple(map(float, monitor.get("center", (0.0, 0.0))))
        if normal == "Z"
        else _placed_xy(monitor, "orientation_deg")
    )
    x_span = float(monitor.get("x span", 0.0))
    y_span = float(monitor.get("y span", 0.0))
    if normal == "Y":
        xy_axis.plot([x_um - x_span / 2.0, x_um + x_span / 2.0], [y_um, y_um], color="#059669", linewidth=1.6, linestyle="-.")
    elif normal == "Z":
        xy_axis.add_patch(Rectangle(
            (x_um - x_span / 2.0, y_um - y_span / 2.0), x_span, y_span,
            fill=False, edgecolor="#059669", linewidth=1.3, linestyle="-.",
        ))
    else:
        xy_axis.plot([x_um, x_um], [y_um - y_span / 2.0, y_um + y_span / 2.0], color="#059669", linewidth=1.6, linestyle="-.")

xy_axis.set_xlim(float(MODEL_BOUNDS_UM[0]), float(MODEL_BOUNDS_UM[3]))
xy_axis.set_ylim(float(MODEL_BOUNDS_UM[1]), float(MODEL_BOUNDS_UM[4]))
xy_axis.set_aspect("equal", adjustable="box")
xy_axis.set_xlabel("X [µm]")
xy_axis.set_ylabel("Y [µm]")
xy_axis.set_title("XY — top view")
xy_axis.grid(alpha=0.18, linewidth=0.4)
_xy_handles, _xy_labels = xy_axis.get_legend_handles_labels()
_xy_handles.extend([
    Line2D([0], [0], color="#dc2626", linewidth=2.0, label="ports"),
    Line2D([0], [0], color="#059669", linewidth=1.6, linestyle="-.", label="monitors"),
    Line2D([0], [0], color="#c026d3", linewidth=2.0, label="Gaussian source"),
])
xy_axis.legend(handles=_xy_handles, loc="upper right", fontsize=7, framealpha=0.92)

_draw_process_projection(xz_axis, 0, "X")
xz_axis.set_title("XZ — side view")
_draw_process_projection(yz_axis, 1, "Y")
yz_axis.set_title("YZ — end view")

actual_dimension = str(fdtd.getnamed("FDTD", "dimension"))
figure.suptitle("As-built Lumerical geometry verification — %s simulation" % actual_dimension, fontsize=14, fontweight="bold")
figure.text(
    0.5, 0.01,
    "XY uses the exact embedded polygons. XZ/YZ use the same bottom-to-top films, etch depths, and Layer Builder sidewall angles used to construct the model.",
    ha="center", fontsize=9, color="#334155",
)
figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.94))
GEOMETRY_PROJECTIONS_FILE = os.path.join(REMOTE_WORK, "geometry_xyz_projections.png")
figure.savefig(GEOMETRY_PROJECTIONS_FILE, dpi=180, bbox_inches="tight")
plt.close(figure)
if not os.path.isfile(GEOMETRY_PROJECTIONS_FILE):
    raise RuntimeError("Geometry projection image was not created: " + GEOMETRY_PROJECTIONS_FILE)
print("Saved 3-axis geometry verification:", GEOMETRY_PROJECTIONS_FILE)
'''


_PORT_MODE_PROFILES_REMOTE = r'''# Visualize the selected port modes before solving.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _field_plane(mode_profile, component_index, preferred_mode_number=0, magnitude=True):
    component_name = ("Ex", "Ey")[component_index]
    try:
        available = list(mode_profile.keys())
    except Exception:
        available = []
    vector_candidates = []
    if int(preferred_mode_number) > 0:
        vector_candidates.append("E%d" % int(preferred_mode_number))
    vector_candidates.append("E")
    vector_candidates.extend(
        sorted(
            (
                key for key in available
                if isinstance(key, str) and len(key) > 1 and key[0] == "E" and key[1:].isdigit()
            ),
            key=lambda key: int(key[1:]),
        )
    )
    # Prefer the explicitly selected E# vector. With two retained fiber modes,
    # a shared E dataset can carry an n dimension; select the requested n
    # rather than silently taking column zero.
    field = None
    rejected_vector_shapes = []
    for candidate in dict.fromkeys(vector_candidates):
        try:
            electric = np.asarray(mode_profile[candidate])
        except (KeyError, TypeError, IndexError):
            continue
        if electric.ndim == 0:
            rejected_vector_shapes.append((candidate, electric.shape))
            continue
        component_axis = electric.ndim - 1 if electric.shape[-1] == 3 else [
            axis for axis, size in enumerate(electric.shape) if size == 3
        ][-1] if 3 in electric.shape else None
        if component_axis is None:
            rejected_vector_shapes.append((candidate, electric.shape))
            continue
        electric = np.moveaxis(electric, component_axis, -1)
        if candidate == "E" and int(preferred_mode_number) > 0:
            try:
                mode_coordinates = np.asarray(mode_profile["n"]).ravel()
            except Exception:
                mode_coordinates = np.asarray([])
            if mode_coordinates.size > 1:
                mode_axes = [
                    axis for axis, size in enumerate(electric.shape[:-1])
                    if size == mode_coordinates.size
                ]
                if mode_axes:
                    matches = np.flatnonzero(np.isclose(
                        mode_coordinates.astype(float), float(preferred_mode_number)
                    ))
                    mode_index = (
                        int(matches[0]) if matches.size
                        else int(preferred_mode_number) - 1
                    )
                    if not 0 <= mode_index < mode_coordinates.size:
                        rejected_vector_shapes.append((candidate, electric.shape))
                        continue
                    electric = np.take(electric, mode_index, axis=mode_axes[-1])
        vector_field = electric[..., component_index]
        field = vector_field
        break
    if field is None:
        try:
            field = np.asarray(mode_profile[component_name])
        except (KeyError, TypeError, IndexError):
            field = None
    if field is None:
        raise RuntimeError(
            "The port mode-profile result contains neither %s nor a readable E/E# vector. "
            "Available fields: %s; rejected vector shapes: %s"
            % (component_name, available, rejected_vector_shapes)
        )
    field = np.squeeze(field)
    while field.ndim > 2:
        field = np.take(field, 0, axis=-1)
    if field.ndim == 1:
        field = field[:, np.newaxis]
    return np.abs(field) if magnitude else np.asarray(field, dtype=complex)


def _plot_coordinates(mode_profile, plane_normal, shape):
    coordinate_keys = {
        "X": ("y", "z"),
        "Y": ("x", "z"),
        "Z": ("x", "y"),
    }[plane_normal]
    coordinates = []
    for key, size in zip(coordinate_keys, shape):
        values = np.squeeze(np.asarray(mode_profile.get(key, np.arange(size)), dtype=float)).ravel()
        if values.size != size:
            values = np.linspace(0.0, float(max(0, size - 1)), size)
        maximum_coordinate = float(np.max(np.abs(values))) if values.size else 0.0
        coordinates.append(values / UM if maximum_coordinate < 1.0 else values)
    return coordinate_keys, coordinates


if GRATING_ANALYSIS and str(GRATING_ANALYSIS.get("excitation_type", "fiber_mode")) == "gaussian_beam":
    profile_specs = [
        (
            "Gaussian source field",
            str(GRATING_ANALYSIS["source_name"]),
            "local TE",
            "gaussian",
        ),
        (
            "Waveguide receiver fundamental mode",
            str(GRATING_ANALYSIS["waveguide_port_name"]),
            "local TE",
            "port",
        ),
    ]
elif GRATING_ANALYSIS:
    profile_specs = [
        ("Fiber source", str(GRATING_ANALYSIS["fiber_port_name"]), "local TE", "port"),
        (
            "Waveguide receiver fundamental mode",
            str(GRATING_ANALYSIS["waveguide_port_name"]),
            "local TE",
            "port",
        ),
    ]
elif MMI_ANALYSIS:
    mmi_port_names = [
        str(MMI_ANALYSIS["input_port_name"]),
        *list(map(str, MMI_ANALYSIS["output_port_names"])),
    ]
    mmi_port_labels = list(MMI_ANALYSIS.get(
        "port_profile_labels",
        ["MMI input", "MMI upper output", "MMI lower output"],
    ))
    profile_specs = [
        (
            mmi_port_labels[index] if index < len(mmi_port_labels) else "MMI port %d" % (index + 1),
            port_name,
            str(MMI_ANALYSIS.get("port_required_polarization", "Ey")),
            "port",
        )
        for index, port_name in enumerate(mmi_port_names)
    ]
else:
    profile_specs = [
        ("Port %d" % (index + 1), str(port.get("name", "port_%d" % (index + 1))), None, "port")
        for index, port in enumerate(PORTS[:4])
    ]

PORT_POLARIZATION_VALID = True
PORT_MODE_CONFINEMENT_VALID = True
PORT_POLARIZATION_REPORT = []
if not profile_specs:
    print("No FDTD ports were exported; pre-solve Ex/Ey visualization is not required.")
else:
    figure, axes = plt.subplots(
        len(profile_specs), 2,
        figsize=(10.5, max(4.0, 3.8 * len(profile_specs))),
        squeeze=False,
    )
    object_by_name = {
        str(item.get("name", "")): item
        for item in list(PORTS) + list(MONITORS) + list(GAUSSIAN_SOURCES)
    }
    for row_index, (label, object_name, required_polarization, object_kind) in enumerate(profile_specs):
        profile_object = object_by_name.get(object_name, {})
        preferred_mode_number = max(0, int(profile_object.get("mode number", 0)))
        resolved_required_polarization = str(required_polarization or "")
        local_te_required = resolved_required_polarization.strip().lower() in {
            "te", "local te", "fundamental te"
        }
        if object_kind == "gaussian":
            # A Gaussian source is an analytic beam object rather than an
            # eigenmode result provider.  Render its actual scalar transverse
            # envelope and global Ex/Ey decomposition directly from the same
            # waist, distance, phi, and local-TE/S-polarization parameters
            # used by addgaussian().
            span_um = max(1e-6, float(profile_object.get("span_um", 20.0)))
            waist_um = max(1e-6, float(profile_object.get("waist radius_um", 4.5)))
            waist_distance_um = float(
                profile_object.get("distance from waist_um", 0.0)
            )
            wavelength_um = 0.5 * (
                float(SETTINGS.get("wavelength_start_um", 1.25))
                + float(SETTINGS.get("wavelength_stop_um", 1.35))
            )
            rayleigh_um = np.pi * waist_um ** 2 / max(wavelength_um, 1e-9)
            plane_waist_um = waist_um * np.sqrt(
                1.0 + (waist_distance_um / max(rayleigh_um, 1e-12)) ** 2
            )
            coordinate_um = np.linspace(-0.5 * span_um, 0.5 * span_um, 241)
            grid_x_um, grid_y_um = np.meshgrid(
                coordinate_um, coordinate_um, indexing="ij"
            )
            scalar_field = np.exp(
                -(grid_x_um ** 2 + grid_y_um ** 2) / plane_waist_um ** 2
            ).astype(complex)
            phi_rad = np.deg2rad(float(profile_object.get("angle phi", 0.0)))
            # S polarization is local TE: E is normal to the grating/incidence
            # plane, so eS = (-sin(phi), cos(phi), 0).
            ex_complex = -np.sin(phi_rad) * scalar_field
            ey_complex = np.cos(phi_rad) * scalar_field
            mode_profile = {
                "x": coordinate_um,
                "y": coordinate_um,
            }
            profile_object = dict(profile_object)
            profile_object["plane normal"] = "Z"
            print(
                "Gaussian source profile %s: w0 %.6g um, source-plane radius %.6g um, "
                "phi %.6g deg, local TE/S polarization."
                % (
                    object_name, waist_um, plane_waist_um,
                    float(profile_object.get("angle phi", 0.0)),
                )
            )
        else:
            result_path = "FDTD::ports::" + object_name if object_kind == "port" else object_name
            mode_profile = fdtd.getresult(result_path, "mode profiles")
            ex_complex = _field_plane(
                mode_profile, 0, preferred_mode_number, magnitude=False
            )
            ey_complex = _field_plane(
                mode_profile, 1, preferred_mode_number, magnitude=False
            )
        ex = np.abs(ex_complex)
        ey = np.abs(ey_complex)
        if ex.shape != ey.shape:
            raise RuntimeError("Mode object %s returned incompatible Ex/Ey profile shapes" % object_name)
        ex_power = float(np.sum(ex ** 2))
        ey_power = float(np.sum(ey ** 2))
        transverse_power = max(ex_power + ey_power, 1e-300)
        ex_fraction = ex_power / transverse_power
        ey_fraction = ey_power / transverse_power
        local_te_fraction = None
        if local_te_required:
            plane_normal = str(profile_object.get("plane normal", "X")).upper()
            propagation_angle_deg = float(
                profile_object.get("angle phi", 0.0)
                if plane_normal == "Z"
                else profile_object.get(
                    "outward_orientation_deg",
                    profile_object.get("orientation_deg", 0.0),
                )
            )
            propagation_angle = np.deg2rad(propagation_angle_deg)
            target_x = -np.sin(propagation_angle)
            target_y = np.cos(propagation_angle)
            local_te_field = target_x * ex_complex + target_y * ey_complex
            local_parallel_field = (
                np.cos(propagation_angle) * ex_complex
                + np.sin(propagation_angle) * ey_complex
            )
            local_te_power = float(np.sum(np.abs(local_te_field) ** 2))
            local_parallel_power = float(np.sum(np.abs(local_parallel_field) ** 2))
            local_te_fraction = local_te_power / max(
                local_te_power + local_parallel_power, 1e-300
            )
        transverse_field = np.sqrt(ex ** 2 + ey ** 2)
        peak_field = max(float(np.max(transverse_field)) if transverse_field.size else 0.0, 1e-300)
        boundary_field = np.concatenate((
            transverse_field[0, :].ravel(), transverse_field[-1, :].ravel(),
            transverse_field[:, 0].ravel(), transverse_field[:, -1].ravel(),
        ))
        edge_fraction = (float(np.max(boundary_field)) if boundary_field.size else 0.0) / peak_field
        report = "%s: Ex %.4f%%, Ey %.4f%%, boundary field %.4f%% of peak" % (
            object_name, 100.0 * ex_fraction, 100.0 * ey_fraction, 100.0 * edge_fraction,
        )
        if local_te_fraction is not None:
            report += ", local TE %.4f%% normal to %.6g deg axis" % (
                100.0 * local_te_fraction, propagation_angle_deg,
            )
        selection = {}
        if object_kind == "monitor":
            selection = dict(WAVEGUIDE_MODE_SELECTIONS.get(object_name, {}))
            report += ", selected neff %.6g (mode %d)" % (
                float(selection.get("neff", 0.0)),
                int(selection.get("mode number", 0)),
            )
        elif object_name in PORT_MODE_SELECTIONS:
            selection = dict(PORT_MODE_SELECTIONS.get(object_name, {}))
            if "target neff" in selection:
                report += ", selected neff %.6g (shared target %.6g)" % (
                    float(selection.get("neff", 0.0)),
                    float(selection.get("target neff", 0.0)),
                )
            else:
                report += ", selected mode %d after candidates %r" % (
                    int(selection.get("mode number", preferred_mode_number)),
                    selection.get("candidate mode numbers", []),
                )
        PORT_POLARIZATION_REPORT.append(report)
        print(report)
        if local_te_required and local_te_fraction is not None and local_te_fraction <= 0.5:
            PORT_POLARIZATION_VALID = False
            print(
                "ERROR — %s is not polarized normal to its propagation/grating axis; do not run the FDTD solve."
                % label
            )
        if not local_te_required and resolved_required_polarization == "Ey" and ey_power <= ex_power:
            PORT_POLARIZATION_VALID = False
            print("ERROR — %s is not Ey-dominant; do not run the FDTD solve." % label)
        if not local_te_required and resolved_required_polarization == "Ex" and ex_power <= ey_power:
            PORT_POLARIZATION_VALID = False
            print("ERROR — %s is not Ex-dominant; do not run the FDTD solve." % label)
        if (label.startswith("Waveguide") or label.startswith("MMI")) and edge_fraction > 0.05:
            PORT_MODE_CONFINEMENT_VALID = False
            print(
                "ERROR — the %s field is %.4f%% of peak at a port/monitor boundary; "
                "increase its span or adjust the target effective index before solving."
                % (label, 100.0 * edge_fraction)
            )

        maximum_ex = float(np.max(ex)) if ex.size else 0.0
        maximum_ey = float(np.max(ey)) if ey.size else 0.0
        common_scale = max(maximum_ex, maximum_ey, 1e-300)
        plane_normal = str(profile_object.get("plane normal", "Z")).upper()
        coordinate_keys, coordinates = _plot_coordinates(mode_profile, plane_normal, ex.shape)
        extent = [
            float(coordinates[0][0]), float(coordinates[0][-1]),
            float(coordinates[1][0]), float(coordinates[1][-1]),
        ]
        for column_index, (field, component_name, fraction) in enumerate((
            (ex, "|Ex|", ex_fraction),
            (ey, "|Ey|", ey_fraction),
        )):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                (field / common_scale).T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xlabel(coordinate_keys[0] + " [um]")
            axis.set_ylabel(coordinate_keys[1] + " [um]")
            neff_suffix = (
                ", neff %.5g" % float(selection["neff"])
                if "neff" in selection else ""
            )
            axis.set_title(
                "%s — %s (%.3f%%%s)"
                % (label, component_name, 100.0 * fraction, neff_suffix)
            )
            figure.colorbar(image, ax=axis, label="normalized field magnitude")
    figure.suptitle("Excitation and selected receiver fields before simulation", fontsize=14)
    figure.tight_layout()
    PORT_MODE_PROFILES_FILE = os.path.join(REMOTE_WORK, "port_mode_Ex_Ey.png")
    figure.savefig(PORT_MODE_PROFILES_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if not os.path.isfile(PORT_MODE_PROFILES_FILE) or os.path.getsize(PORT_MODE_PROFILES_FILE) <= 0:
        raise RuntimeError("Port Ex/Ey verification image was not created")
    print("Saved pre-solve port Ex/Ey verification:", PORT_MODE_PROFILES_FILE)
PORT_MODE_VALID = bool(PORT_POLARIZATION_VALID and PORT_MODE_CONFINEMENT_VALID)
'''


_REMOTE_RESOURCE_AND_SAVE = r'''import re
import time

resource_mode = str(SETTINGS.get("resource_mode", "GPU")).strip().upper()
TOTAL_CORES = max(1, min(int(SETTINGS.get("build_cpu_threads", 30)), os.cpu_count() or 1))

if resource_mode == "GPU":
    if str(SETTINGS.get("dimension", "3D")).strip().upper() != "3D":
        raise RuntimeError("GPU execution is configured only for a 3D FDTD simulation")
    gpu_specs = fdtd.gpuspecs()
    if not gpu_specs:
        raise RuntimeError("No Lumerical-compatible GPU was detected by gpuspecs()")
    gpu = gpu_specs[0]
    SM = int(gpu["deviceSMCount"])

    # GPU is selected per run. Keep the CPU row active for meshing and script operations.
    fdtd.setresource("FDTD", 1, "device type", "GPU")
    fdtd.setresource("FDTD", 1, "active", True)
    fdtd.setresource("FDTD", 1, "sm estimate", SM)
    try:
        fdtd.setresource("FDTD", 1, "threads", "auto")
    except Exception:
        fdtd.setresource("FDTD", 1, "threads", TOTAL_CORES)
    fdtd.setresource("FDTD", 2, "device type", "CPU")
    fdtd.setresource("FDTD", 2, "active", True)
    fdtd.setresource("FDTD", 2, "processes", 1)
    fdtd.setresource("FDTD", 2, "threads", TOTAL_CORES)

    print("GPU selected for the 3D FDTD solve")
    print("  device:", gpu.get("userReadableDeviceName", "GPU 0"))
    print("  SM count:", SM)
    print("  GPU resource active:", fdtd.getresource("FDTD", 1, "active"))
    print("  CPU support resource active:", fdtd.getresource("FDTD", 2, "active"))
    print("  SM licence estimate:", fdtd.getresource("FDTD", 1, "sm estimate"))
    if bool(SETTINGS.get("run_gpu_system_check", False)):
        try:
            estimate = fdtd.getlicenseestimate("FDTD", "1")
            print("  GPU licence feature:", estimate.get("feature"))
            print("  GPU single-run licences:", estimate.get("single"))
        except Exception as exc:
            print("  GPU licence-estimate warning:", str(exc)[:180])
        try:
            print("  GPU system check:", fdtd.runsystemcheck("FDTD", "GPU"))
        except Exception as exc:
            print("  GPU system-check warning:", str(exc)[:180])
    else:
        print("  GPU diagnostic skipped by cell 1; the solve still explicitly requests GPU.")
elif resource_mode == "CPU":
    fdtd.setresource("FDTD", 1, "active", False)
    fdtd.setresource("FDTD", 2, "device type", "CPU")
    fdtd.setresource("FDTD", 2, "active", True)
    fdtd.setresource("FDTD", 2, "processes", 1)
    fdtd.setresource("FDTD", 2, "threads", TOTAL_CORES)
    print("CPU resource active: 1 process x %d threads" % TOTAL_CORES)
else:
    raise ValueError("resource_mode must be GPU or CPU")

if GRATING_ANALYSIS and str(GRATING_ANALYSIS.get("excitation_type", "fiber_mode")) == "gaussian_beam":
    fdtd.switchtolayout()
    if PORTS:
        fdtd.select("FDTD::ports")
        # Keep the waveguide port passive.  The independent analytic Gaussian
        # object is the only source in this excitation mode.
        fdtd.set("source port", "")
    gaussian_source_name = str(GRATING_ANALYSIS["source_name"])
    for gaussian_source in GAUSSIAN_SOURCES:
        candidate_name = str(gaussian_source.get("name", ""))
        if candidate_name:
            fdtd.setnamed(
                candidate_name, "enabled",
                candidate_name == gaussian_source_name,
            )
    print("Grating excitation source: Gaussian beam " + gaussian_source_name)
    print("Grating excitation direction: Backward along tilted Z injection")
    print(
        "Grating excitation polarization: S/local TE at 90 degrees, normal to grating axis %.6g deg"
        % float(GRATING_ANALYSIS.get("gaussian_axis_orientation_deg", 0.0))
    )
    print(
        "Gaussian waist radius %.6g um; distance from waist %.6g um."
        % (
            float(GRATING_ANALYSIS.get("gaussian_waist_radius_um", 4.5)),
            float(GRATING_ANALYSIS.get("gaussian_distance_from_waist_um", 0.0)),
        )
    )
    print(
        "Incident-power monitor: %s (ordinary Z-normal power monitor, expected signed T factor %.0f)."
        % (
            GRATING_ANALYSIS["fiber_input_power_monitor_name"],
            float(GRATING_ANALYSIS.get("fiber_input_power_sign", -1.0)),
        )
    )
    print("Waveguide total-power monitor: " + str(GRATING_ANALYSIS["waveguide_power_monitor_name"]))
    print("Passive waveguide receiver: FDTD::ports::" + str(GRATING_ANALYSIS["waveguide_port_name"]))
elif GRATING_ANALYSIS:
    fdtd.switchtolayout()
    if str(GRATING_ANALYSIS["waveguide_port_name"]) == str(GRATING_ANALYSIS["fiber_port_name"]):
        raise RuntimeError("The passive waveguide receiver must be distinct from the fiber source port")
    fdtd.select("FDTD::ports")
    fdtd.set("source port", str(GRATING_ANALYSIS["fiber_port_name"]))
    fiber_source_mode = str(GRATING_ANALYSIS.get("fiber_source_mode", "auto local TE"))
    if not re.match(r"^mode\s+[1-9][0-9]*$", fiber_source_mode.strip(), flags=re.IGNORECASE):
        raise RuntimeError(
            "The fiber source mode was not resolved from its near-degenerate pair before resource setup: "
            + fiber_source_mode
        )
    fdtd.set("source mode", fiber_source_mode)
    configured_fiber_source_mode = str(fdtd.get("source mode")).strip()
    requested_source_match = re.fullmatch(
        r"mode\s+([1-9][0-9]*)", fiber_source_mode.strip(), flags=re.IGNORECASE
    )
    configured_source_match = re.fullmatch(
        r"mode\s+([1-9][0-9]*)", configured_fiber_source_mode, flags=re.IGNORECASE
    )
    if (
        requested_source_match is None
        or configured_source_match is None
        or int(requested_source_match.group(1)) != int(configured_source_match.group(1))
    ):
        raise RuntimeError(
            "The FDTD port group did not retain the verified local-TE source mode: "
            "requested %r, configured %r"
            % (fiber_source_mode, configured_fiber_source_mode)
        )
    source_selection = dict(PORT_MODE_SELECTIONS.get(
        str(GRATING_ANALYSIS["fiber_port_name"]), {}
    ))
    source_mode_number = int(requested_source_match.group(1))
    source_te_score = float(
        dict(source_selection.get("local TE scores", {})).get(
            str(source_mode_number), 0.0
        )
    )
    if source_te_score < float(source_selection.get("minimum local TE fraction", 0.8)):
        raise RuntimeError(
            "Configured fiber source mode %d is not the verified local-TE/Ey winner; "
            "score %.6g, selection metadata %r"
            % (source_mode_number, source_te_score, source_selection)
        )
    print("Grating excitation source: FDTD::ports::" + str(GRATING_ANALYSIS["fiber_port_name"]))
    print("Grating excitation direction: Backward along the tilted Z-axis fiber port")
    print(
        "Grating excitation mode: %s (%s, normal to grating axis %.6g deg)"
        % (
            fiber_source_mode,
            GRATING_ANALYSIS.get("fiber_polarization", "local TE"),
            float(GRATING_ANALYSIS.get("fiber_axis_orientation_deg", 0.0)),
        )
    )
    print(
        "Verified active fiber source readback: %s, local-TE score %.6f."
        % (configured_fiber_source_mode, source_te_score)
    )
    print(
        "Fiber incident-power monitor: %s (non-modal Z-normal power monitor, expected signed T factor %.0f)."
        % (
            GRATING_ANALYSIS["fiber_input_power_monitor_name"],
            float(GRATING_ANALYSIS.get("fiber_input_power_sign", -1.0)),
        )
    )
    print("Waveguide total-power monitor: " + str(GRATING_ANALYSIS["waveguide_power_monitor_name"]))
    print(
        "Passive waveguide receiver: FDTD::ports::%s, target neff %.6g; modal result %s/%s"
        % (
            GRATING_ANALYSIS["waveguide_port_name"],
            float(GRATING_ANALYSIS["waveguide_target_neff"]),
            GRATING_ANALYSIS.get("waveguide_port_expansion_result_name", "expansion for port monitor"),
            GRATING_ANALYSIS.get("waveguide_port_modal_direction", "T_out"),
        )
    )
elif MMI_ANALYSIS:
    fdtd.switchtolayout()
    fdtd.select("FDTD::ports")
    fdtd.set("source port", str(MMI_ANALYSIS["input_port_name"]))
    fdtd.set("source mode", "mode 1")
    print("MMI excitation source: FDTD::ports::" + str(MMI_ANALYSIS["input_port_name"]))
    print("MMI excitation mode: mode 1")
    print("MMI output ports:", ", ".join(map(str, MMI_ANALYSIS["output_port_names"])))
elif PORTS:
    # Generic components use the lowest-order enabled FDTD port as their
    # source; all remaining ports and power monitors are passive receivers.
    source_port = min(
        (port for port in PORTS if bool(port.get("enabled", True))),
        key=lambda port: float(port.get("order", 1)),
    )
    fdtd.switchtolayout()
    fdtd.select("FDTD::ports")
    fdtd.set("source port", str(source_port["name"]))
    source_mode_number = max(1, int(source_port.get("mode number", 1)))
    fdtd.set("source mode", "mode %d" % source_mode_number)
    print(
        "Generic component excitation: FDTD::ports::%s, mode %d"
        % (source_port["name"], source_mode_number)
    )

_project_name = os.path.basename(str(SETTINGS.get("project_file", "exported_component.fsp")))
if not _project_name.lower().endswith(".fsp"):
    _project_name += ".fsp"
REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")
os.makedirs(REMOTE_FSP_DIR, exist_ok=True)
REMOTE_PROJECT_FILE = os.path.join(REMOTE_FSP_DIR, _project_name)
REMOTE_INSPECTION_PROJECT_FILE = os.path.join(REMOTE_FSP_DIR, "inspection_" + _project_name)
REMOTE_INSPECTION_FSP_SAVED = False
REMOTE_FINAL_FSP_SAVED = False


def save_verified_project(project_path=REMOTE_PROJECT_FILE):
    """Save to shared storage and do not report success until a non-empty .fsp exists."""
    fdtd.save(project_path)
    for _attempt in range(40):
        if os.path.isfile(project_path) and os.path.getsize(project_path) > 0:
            print("Verified remote project: %s (%d bytes)" % (
                project_path, os.path.getsize(project_path)
            ))
            return project_path
        time.sleep(0.25)
    nearby = sorted(name for name in os.listdir(REMOTE_FSP_DIR) if name.lower().endswith(".fsp"))
    raise RuntimeError(
        "Lumerical save returned without creating %s. Nearby .fsp files: %s"
        % (project_path, nearby)
    )


save_verified_project(REMOTE_INSPECTION_PROJECT_FILE)
REMOTE_INSPECTION_FSP_SAVED = True
print("Saved required pre-solve inspection FSP.")
'''


_SWITCH_TO_CPU_ANALYSIS_REMOTE = r'''# GPU work is complete; keep result data but return post-processing to CPU.
analysis_threads = max(1, min(int(SETTINGS.get("build_cpu_threads", 30)), os.cpu_count() or 1))
try:
    fdtd.setresource("FDTD", 1, "active", False)
    fdtd.setresource("FDTD", 2, "device type", "CPU")
    fdtd.setresource("FDTD", 2, "active", True)
    fdtd.setresource("FDTD", 2, "processes", 1)
    fdtd.setresource("FDTD", 2, "threads", analysis_threads)
    print("GPU solve complete; post-processing resource is CPU: 1 process x %d threads." % analysis_threads)
except Exception as exc:
    # Plotting still occurs in the notebook's local CPU Python kernel.  Keep
    # solved monitor data intact even if this Lumerical build locks resources
    # while it remains in analysis mode.
    print("CPU post-processing resource switch warning:", str(exc)[:240])
'''


_SAVE_REMOTE_RESULTS = r'''# Save numerical results before any licence is released.
RESULT_ARRAYS = {}
RESULT_ERRORS = []


def _collect_numeric(prefix, value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "Lumerical_dataset":
                continue
            _collect_numeric(prefix + "__" + str(key), child)
        return
    try:
        array = np.asarray(value)
    except Exception:
        return
    if array.dtype.kind in "buifc" and array.size:
        RESULT_ARRAYS[prefix] = array


if SETTINGS.get("run_after_build", False):
    if GRATING_ANALYSIS and isinstance(globals().get("GRATING_RESULT_ARRAYS"), dict):
        _collect_numeric("grating", GRATING_RESULT_ARRAYS)
        print("Reused the already extracted grating spectrum for the result bundle.")
    for port in ([] if GRATING_ANALYSIS and isinstance(globals().get("GRATING_RESULT_ARRAYS"), dict) else PORTS):
        name = str(port.get("name", ""))
        if not name:
            continue
        for result_name in ("S", "T"):
            try:
                _collect_numeric(
                    "port__" + name + "__" + result_name,
                    fdtd.getresult("FDTD::ports::" + name, result_name),
                )
            except Exception as exc:
                RESULT_ERRORS.append("port %s %s: %s" % (name, result_name, str(exc)[:160]))

    for monitor in MONITORS:
        name = str(monitor.get("name", ""))
        kind = str(monitor.get("monitor_kind", "Power monitor"))
        candidates = {
            "Power monitor": ("T",),
            "Field profile monitor": ("E", "H", "P"),
            "Mode expansion monitor": (
                "expansion for " + str(monitor.get("expansion result name", "input")),
                "mode profiles",
                "neff",
            ),
        }.get(kind, ("T", "E"))
        saved = False
        for result_name in candidates:
            try:
                _collect_numeric("monitor__" + name + "__" + result_name, fdtd.getresult(name, result_name))
                saved = True
            except Exception as exc:
                RESULT_ERRORS.append("monitor %s %s: %s" % (name, result_name, str(exc)[:160]))
        if not saved:
            RESULT_ERRORS.append("monitor %s had no readable numeric result" % name)

REMOTE_RESULTS_NPZ = os.path.join(REMOTE_WORK, "max_layout_results.npz")
np.savez_compressed(REMOTE_RESULTS_NPZ, **RESULT_ARRAYS)
REMOTE_RESULTS_JSON = os.path.join(REMOTE_WORK, "max_layout_results.json")
REMOTE_TEXT_SUMMARY = os.path.join(REMOTE_WORK, "summary.txt")


def _summary_number(value, digits=8):
    try:
        number = float(np.asarray(value).ravel()[0])
    except Exception:
        return str(value)
    return ("%.*g" % (int(digits), number)) if np.isfinite(number) else str(number)


def _summary_json(value):
    def _reported(item):
        if isinstance(item, dict):
            return {
                str(key): _reported(child)
                for key, child in item.items()
                if str(key) != "waveguide_effective_index"
            }
        if isinstance(item, (list, tuple)):
            return [_reported(child) for child in item]
        return item
    return json.dumps(_reported(value), sort_keys=True, separators=(",", ":"), default=str)


def _summary_section(lines, title):
    if lines:
        lines.append("")
    lines.append(str(title))
    lines.append("-" * len(str(title)))


_SUMMARY_MAJOR_PARAMETERS = (
    ("pitch", "Pitch", "um"),
    ("fill_factor", "Fill factor", ""),
    ("duty_cycle", "Duty cycle", ""),
    ("fill_factors", "Apodized fill factors", ""),
    ("tooth_shape", "Tooth geometry", ""),
    ("N", "Number of grating periods", ""),
    ("target_length", "Target grating length", "um"),
    ("h_total", "Device-layer thickness", "um"),
    ("etch_depth", "Etch depth", "um"),
    ("alpha_t", "Aperture angle", "deg"),
    ("taper_L", "Taper length", "um"),
    ("radius", "Focusing radius", "um"),
    ("y_span", "Grating Y span", "um"),
    ("L_extra", "Thick-end extension", "um"),
    ("wg_width", "Waveguide width", "um"),
    ("wg_length", "Waveguide length", "um"),
    ("taper_exponent", "Grating taper exponent", ""),
    ("width", "Width", "um"),
    ("length", "Length", "um"),
    ("width_start", "Starting width", "um"),
    ("width_end", "Ending width", "um"),
    ("mmi_width", "MMI width", "um"),
    ("mmi_length", "MMI length", "um"),
    ("taper_width", "MMI taper width", "um"),
    ("input_taper_length", "Input taper length", "um"),
    ("output_taper_length", "Output taper length", "um"),
    ("input_length", "Input access length", "um"),
    ("output_length", "Output access length", "um"),
    ("port_sep", "Output branch separation", "um"),
    ("taper_power", "MMI taper profile exponent", ""),
    ("taper_points", "MMI taper discretization points", ""),
    ("input_reference_before_taper_um", "Input power-reference distance before taper", "um"),
    ("fdtd_port_clearance_um", "MMI access-port clearance from waveguide end", "um"),
    ("fiber_offset", "Fiber offset", "um"),
    ("angle_theta", "Fiber angle theta", "deg"),
    ("excitation_type", "Grating excitation", ""),
    ("gaussian_waist_radius_um", "Gaussian waist radius (1/e field)", "um"),
    ("gaussian_distance_from_waist_um", "Gaussian distance from waist", "um"),
    ("gaussian_source_span_um", "Gaussian source span", "um"),
    ("gaussian_multifrequency_points", "Gaussian multifrequency profile points", ""),
    ("gaussian_source_depth_in_cladding_um", "Gaussian source depth inside SiO2 cladding", "um"),
    ("gaussian_input_monitor_span_scale", "Gaussian input-power monitor span scale", "x"),
    ("fiber_tox_offset_um", "Fiber bottom offset above SiO2 cladding", "um"),
    ("fiber_core_diameter_um", "Fiber core diameter", "um"),
    ("fiber_core_index", "Fiber core refractive index", ""),
    ("fiber_cladding_diameter_um", "Fiber cladding diameter", "um"),
    ("fiber_cladding_index", "Fiber cladding refractive index", ""),
    ("fiber_length_um", "Fiber length", "um"),
    ("fiber_power_monitor_below_source_um", "Horizontal fiber-input monitor distance below source", "um"),
    ("fdtd_port_offset_from_waveguide_end_um", "Waveguide FDTD-port offset from waveguide end", "um"),
    ("waveguide_monitor_span_um", "Waveguide receiver-port transverse span", "um"),
    ("waveguide_total_power_before_mode_um", "Total-power plane distance before receiver port", "um"),
    ("waveguide_neff_tolerance", "Waveguide effective-index tolerance", ""),
    ("waveguide_mode_search_count", "Waveguide eigensolver modes searched", ""),
    ("tolerance", "Geometry build tolerance", "um"),
)


def _summary_parameter_value(value):
    if isinstance(value, (dict, list, tuple)):
        return _summary_json(value)
    if isinstance(value, str):
        return value
    return _summary_number(value)


def _append_major_component_parameters(lines, component):
    params = dict(component.get("params", {}))
    found = 0
    shown = set()
    for key, label, unit in _SUMMARY_MAJOR_PARAMETERS:
        if key not in params or params[key] in (None, ""):
            continue
        suffix = (" " + unit) if unit else ""
        lines.append("  - %s: %s%s" % (label, _summary_parameter_value(params[key]), suffix))
        found += 1
        shown.add(key)
    # Preserve future geometry parameters even before they receive a curated
    # label above.  Layer/datatype selectors are already captured by the exact
    # JSON and material-stack sections, so omit only those layout identifiers.
    ignored = {"name", "layer", "datatype", "waveguide_effective_index"}
    for key, value in params.items():
        if (
            key in shown or key in ignored or key.endswith("_layer")
            or key.endswith("_datatype") or value in (None, "")
        ):
            continue
        if isinstance(value, (bool, int, float, str, list, tuple, dict)):
            lines.append(
                "  - %s: %s"
                % (str(key).replace("_", " ").title(), _summary_parameter_value(value))
            )
            found += 1


def _fdtd_summary_value(property_name):
    try:
        return float(np.asarray(fdtd.getnamed("FDTD", property_name)).ravel()[0])
    except Exception:
        return None


summary_lines = ["MAX LAYOUT — LUMERICAL SIMULATION SUMMARY"]
_summary_section(summary_lines, "PROJECT")
summary_lines.extend([
    "Solved/final FSP: %s" % REMOTE_PROJECT_FILE,
    "Pre-solve inspection FSP: %s" % REMOTE_INSPECTION_PROJECT_FILE,
    "Export scope: %s" % EXPORT_SCOPE_LABEL,
    "Run status: %s" % ("simulation completed" if SETTINGS.get("run_after_build", False) else "model built only; solve disabled"),
])

_summary_section(summary_lines, "PARAMETERS")
summary_lines.append("Major physical parameters are listed in editor units (lengths in um; angles in deg).")
source_components = list(globals().get("SOURCE_COMPONENTS_JSON", []))
if source_components:
    for component_index, component in enumerate(source_components, start=1):
        kind = str(component.get("kind", "component"))
        uid = int(component.get("uid", 0))
        component_name = str(
            component.get("name")
            or component.get("params", {}).get("name")
            or kind
        )
        summary_lines.append("Component %d: %s | kind=%s | UID=%d" % (component_index, component_name, kind, uid))
        summary_lines.append(
            "  - Position: x=%s um | y=%s um | orientation=%s deg"
            % (
                _summary_number(component.get("x", 0.0)),
                _summary_number(component.get("y", 0.0)),
                _summary_number(component.get("orientation_deg", 0.0)),
            )
        )
        _append_major_component_parameters(summary_lines, component)
        summary_lines.append("  - Exact source parameters (JSON): %s" % _summary_json(component.get("params", {})))
else:
    summary_lines.append("Components: %s" % _summary_json(EXPORTED_COMPONENTS))

_summary_section(summary_lines, "MATERIAL STACK AND MESH")
summary_lines.append("Bottom-to-top layer order; mesh factor means factor x lambda0 / maximum material index.")
for index, row in enumerate(MATERIAL_STACK, start=1):
    summary_lines.append(
        "- %02d %s | material=%s | thickness=%s um | etch=%s um | sidewall=%s deg | mesh_factor=%s | mesh_order=%s | role=%s | conformal=%s | slab_extent=%s | GDS_layers=%s"
        % (
            index,
            str(row.get("name", "layer")),
            str(row.get("material", "")),
            _summary_number(row.get("thickness_um", 0.0)),
            _summary_number(row.get("etch_depth_um", 0.0)),
            _summary_number(row.get("sidewall_angle_deg", 90.0)),
            _summary_number(row.get("mesh_factor", 0.2)),
            str(row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2)),
            str(row.get("role", "background")),
            str(bool(row.get("conformal", False))),
            str(row.get("slab_extent", "full FDTD plane")),
            _summary_json(row.get("gds_layers", [])),
        )
    )

domain_values = [_fdtd_summary_value(name) for name in (
    "x min", "x max", "y min", "y max", "z min", "z max"
)]
domain_um = [None if value is None else value / UM for value in domain_values]
_summary_section(summary_lines, "SIMULATION SETTINGS")
summary_lines.extend([
    "- Solver: 3D FDTD",
    "- Wavelength sweep: %s to %s um | %d points"
    % (
        _summary_number(SETTINGS.get("wavelength_start_um")),
        _summary_number(SETTINGS.get("wavelength_stop_um")),
        int(SETTINGS.get("frequency_points", 0)),
    ),
    "- Domain [xmin,xmax,ymin,ymax,zmin,zmax]: %s um" % _summary_json(domain_um),
    "- Resources: solve=%s | model-build CPU threads=%s | CPU post-processing=True"
    % (
        str(SETTINGS.get("resource_mode", "GPU")),
        str(SETTINGS.get("build_cpu_threads", 30)),
    ),
    "- Numerical controls: mesh accuracy=%s | dt factor=%s | PML=%s | geometry/PML overlap=%s um"
    % (
        str(SETTINGS.get("mesh_accuracy", 2)),
        _summary_number(SETTINGS.get("dt_stability_factor", 0.99)),
        str(SETTINGS.get("pml_profile", "Standard")),
        _summary_number(SETTINGS.get("pml_geometry_overlap_um", 1.0)),
    ),
    "- Time controls: maximum=%s ps (%s fs) | auto shutoff=%s"
    % (
        _summary_number(float(SETTINGS.get("simulation_time_fs", 10000.0)) / 1000.0),
        _summary_number(SETTINGS.get("simulation_time_fs", 10000.0)),
        _summary_number(SETTINGS.get("auto_shutoff_min", 1e-6)),
    ),
    "- TFLN material model: crystal cut=%s | temperature=%s K"
    % (
        str(SETTINGS.get("tfln_crystal_cut", "X")),
        _summary_number(SETTINGS.get("tfln_temperature_K", 296.3)),
    ),
    "- Symmetry boundary: enabled=%s | boundary=%s"
    % (
        str(bool(SETTINGS.get("use_y_antisymmetry", False))),
        str(SETTINGS.get("antisymmetry_boundary", "") or "none"),
    ),
])
waveguide_index_estimate = dict(globals().get("WAVEGUIDE_INDEX_ESTIMATE", {}))
if waveguide_index_estimate:
    summary_lines.append(
        "- Automatic waveguide mode target: core n=%s | adjacent dielectric n=%s | "
        "midpoint neff=%s at %s um | core=%s | surroundings=%s"
        % (
            _summary_number(waveguide_index_estimate.get("core_index")),
            _summary_number(waveguide_index_estimate.get("surrounding_index")),
            _summary_number(waveguide_index_estimate.get("target_neff")),
            _summary_number(waveguide_index_estimate.get("wavelength_um")),
            _summary_json(waveguide_index_estimate.get("core_materials", [])),
            _summary_json(waveguide_index_estimate.get("surrounding_materials", [])),
        )
    )

_summary_section(summary_lines, "SOURCES / PORTS / MONITORS")
if GRATING_ANALYSIS:
    source_name = str(
        GRATING_ANALYSIS.get(
            "source_name", GRATING_ANALYSIS.get("fiber_port_name", "")
        )
    )
elif MMI_ANALYSIS:
    source_name = str(MMI_ANALYSIS.get("input_port_name", ""))
else:
    source_name = "not automatically assigned"
summary_lines.append("Active source: %s" % source_name)
summary_lines.append("Fiber geometries:")
if FIBER_GEOMETRIES:
    for fiber in FIBER_GEOMETRIES:
        summary_lines.append("- %s" % _summary_json({
            "name": fiber.get("name"), "center_um": fiber.get("center"),
            "z_reference": fiber.get("z reference"), "z_offset_um": fiber.get("z offset_um", 0.0),
            "theta_deg": fiber.get("angle theta", 0.0), "phi_deg": fiber.get("angle phi", 0.0),
            "core_diameter_um": fiber.get("core diameter_um"), "core_index": fiber.get("core index"),
            "cladding_diameter_um": fiber.get("cladding diameter_um"), "cladding_index": fiber.get("cladding index"),
            "length_um": fiber.get("length_um"),
            "internal_offsets_um": [fiber.get("core x offset_um", 0.0), fiber.get("core y offset_um", 0.0), fiber.get("core z offset_um", 0.0)],
        }))
else:
    summary_lines.append("- none")
summary_lines.append("FDTD ports:")
if PORTS:
    for port in PORTS:
        summary_lines.append("- %s" % _summary_json({
            "name": port.get("name"), "normal": port.get("plane normal"),
            "center_um": port.get("center"), "direction": port.get("dir"),
            "spans_um": [port.get("x span"), port.get("y span"), port.get("z span", port.get("z_span_um"))],
            "theta_deg": port.get("angle theta", 0.0), "phi_deg": port.get("angle phi", 0.0),
            "mode": port.get("mode"), "mode_number": port.get("mode number"),
            "polarization": port.get("polarization"), "target_neff": port.get("target neff"),
            "role": port.get("parent_port_name"),
        }))
else:
    summary_lines.append("- none")
summary_lines.append("Monitors:")
if MONITORS:
    for monitor in MONITORS:
        summary_lines.append("- %s" % _summary_json({
            "name": monitor.get("name"), "kind": monitor.get("monitor_kind"),
            "normal": monitor.get("plane normal"), "center_um": monitor.get("center"),
            "spans_um": [monitor.get("x span"), monitor.get("y span"), monitor.get("z span")],
            "theta_deg": monitor.get("angle theta", 0.0), "phi_deg": monitor.get("angle phi", 0.0),
            "role": monitor.get("grating_monitor_role", monitor.get("parent_port_name")),
            "target_neff": monitor.get("target neff"),
        }))
else:
    summary_lines.append("- none")

_summary_section(summary_lines, "RESULTS SUMMARY")

if not SETTINGS.get("run_after_build", False):
    summary_lines.append("- Simulation was not run; this summary records the built model only.")
elif GRATING_ANALYSIS and isinstance(globals().get("GRATING_RESULT_ARRAYS"), dict):
    grating_arrays = GRATING_RESULT_ARRAYS
    wavelengths = np.asarray(grating_arrays["wavelength_m"], dtype=float).ravel()
    coupling = np.asarray(grating_arrays["fiber_coupling"], dtype=float).ravel()
    peak_index = int(np.nanargmax(coupling))
    target_wavelength = float(np.asarray(grating_arrays["target_wavelength_m"]).ravel()[0])
    target_index = int(np.argmin(np.abs(wavelengths - target_wavelength)))
    def _grating_value_at(key, index):
        try:
            values = np.asarray(grating_arrays[key], dtype=float).ravel()
            return values[min(index, max(0, values.size - 1))]
        except Exception:
            return np.nan
    summary_lines.append(
        "- Grating coupling efficiency (linear): peak=%s (%s%%) at %s nm | target %s nm=%s (%s%%)"
        % (
            _summary_number(coupling[peak_index]),
            _summary_number(100.0 * coupling[peak_index]),
            _summary_number(wavelengths[peak_index] * 1e9),
            _summary_number(wavelengths[target_index] * 1e9),
            _summary_number(coupling[target_index]),
            _summary_number(100.0 * coupling[target_index]),
        )
    )
    summary_lines.append(
        "- Grating coupling efficiency (dB): peak=%s dB | target=%s dB"
        % (
            _summary_number(10.0 * np.log10(max(coupling[peak_index], 1e-15))),
            _summary_number(10.0 * np.log10(max(coupling[target_index], 1e-15))),
        )
    )
    summary_lines.append(
        "- Target power accounting: measured incident input=%s | waveguide selected-TE=%s | "
        "waveguide total=%s | TE/input=%s | total/input=%s"
        % (
            _summary_number(_grating_value_at("fiber_input_power", target_index)),
            _summary_number(_grating_value_at("waveguide_mode_power", target_index)),
            _summary_number(_grating_value_at("waveguide_total_power", target_index)),
            _summary_number(_grating_value_at("coupling_efficiency", target_index)),
            _summary_number(_grating_value_at("waveguide_total_transmission", target_index)),
        )
    )
    summary_lines.append(
        "- Selected waveguide mode: neff=%s | mode=%s"
        % (
            _summary_number(grating_arrays.get("waveguide_selected_neff", np.nan)),
            _summary_number(grating_arrays.get("waveguide_selected_mode_number", np.nan)),
        )
    )
elif MMI_ANALYSIS:
    mmi_path = os.path.join(REMOTE_WORK, "mmi_analysis.npz")
    if os.path.isfile(mmi_path):
        with np.load(mmi_path) as mmi_data:
            wavelengths = np.asarray(mmi_data["wavelength_m"], dtype=float).ravel()
            target_wavelength = float(np.asarray(mmi_data["target_wavelength_m"]).ravel()[0])
            target_index = int(np.argmin(np.abs(wavelengths - target_wavelength)))
            summary_lines.append(
                "- MMI linear power at %s nm: Pin=%s | upper/Pin=%s (%s%%) | lower/Pin=%s (%s%%) | total/Pin=%s"
                % (
                    _summary_number(wavelengths[target_index] * 1e9),
                    _summary_number(mmi_data["input_power"][target_index]),
                    _summary_number(mmi_data["output_1_over_input"][target_index]),
                    _summary_number(100.0 * mmi_data["output_1_over_input"][target_index]),
                    _summary_number(mmi_data["output_2_over_input"][target_index]),
                    _summary_number(100.0 * mmi_data["output_2_over_input"][target_index]),
                    _summary_number(mmi_data["total_output_over_input"][target_index]),
                )
            )
            summary_lines.append(
                "- MMI split of transmitted power: upper=%s | lower=%s | symmetry error=%s percentage points | selected neff=%s"
                % (
                    _summary_number(mmi_data["output_1_ratio"][target_index]),
                    _summary_number(mmi_data["output_2_ratio"][target_index]),
                    _summary_number(mmi_data["symmetry_error_percent"][target_index]),
                    _summary_json(np.asarray(mmi_data["port_selected_neff"]).tolist()),
                )
            )
    else:
        summary_lines.append("- MMI solve completed, but mmi_analysis.npz was unavailable.")
else:
    summary_lines.append(
        "- Generic numeric providers saved (%d arrays): %s"
        % (len(RESULT_ARRAYS), ", ".join(sorted(RESULT_ARRAYS)) or "none")
    )

_summary_section(summary_lines, "WARNINGS / NOTES")
summary_notes = [str(item) for item in globals().get("EXPORT_WARNINGS", [])]
summary_notes.extend(str(item) for item in RESULT_ERRORS)
if summary_notes:
    summary_lines.extend("- %s" % note for note in summary_notes)
else:
    summary_lines.append("- No export or result-extraction warnings were recorded.")

with open(REMOTE_TEXT_SUMMARY, "w", encoding="utf-8") as stream:
    stream.write("\n".join(summary_lines).rstrip() + "\n")
with open(REMOTE_RESULTS_JSON, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "project_file": REMOTE_PROJECT_FILE,
            "inspection_project_file": (
                REMOTE_INSPECTION_PROJECT_FILE if REMOTE_INSPECTION_FSP_SAVED else None
            ),
            "simulation_ran": bool(SETTINGS.get("run_after_build", False)),
            "resource_mode": SETTINGS.get("resource_mode"),
            "saved_numeric_keys": sorted(RESULT_ARRAYS),
            "result_notes": RESULT_ERRORS,
            "ports_json": PORTS_JSON,
            "fiber_geometries": FIBER_GEOMETRIES,
            "gaussian_sources": GAUSSIAN_SOURCES,
            "export_scope": EXPORT_SCOPE_LABEL,
            "exported_components": EXPORTED_COMPONENTS,
            "source_components_json": source_components,
            "material_stack": MATERIAL_STACK,
            "grating_analysis": GRATING_ANALYSIS,
            "mmi_analysis": MMI_ANALYSIS,
            "text_summary": REMOTE_TEXT_SUMMARY,
        },
        stream,
        indent=2,
    )
for required_path in (REMOTE_RESULTS_NPZ, REMOTE_RESULTS_JSON, REMOTE_TEXT_SUMMARY):
    if not os.path.isfile(required_path) or os.path.getsize(required_path) <= 0:
        raise RuntimeError("Required result artifact was not created: " + required_path)
if not bool(globals().get("REMOTE_FINAL_FSP_SAVED", False)):
    raise RuntimeError("The dedicated pre-analysis final-FSP stage did not complete")
if not os.path.isfile(REMOTE_PROJECT_FILE) or os.path.getsize(REMOTE_PROJECT_FILE) <= 0:
    raise RuntimeError("The verified solved/final FSP is missing or empty: " + REMOTE_PROJECT_FILE)
print("Verified previously saved solved/final FSP:", REMOTE_PROJECT_FILE)
if (
    os.path.isfile(globals().get("REMOTE_RUNTIME_PROJECT_FILE", ""))
    and os.path.abspath(REMOTE_RUNTIME_PROJECT_FILE)
    not in {
        os.path.abspath(REMOTE_PROJECT_FILE),
        os.path.abspath(REMOTE_INSPECTION_PROJECT_FILE),
    }
):
    try:
        os.remove(REMOTE_RUNTIME_PROJECT_FILE)
        print("Removed transient runtime project after result extraction.")
    except Exception as exc:
        print("Transient runtime cleanup warning:", str(exc)[:180])
print("Saved result bundle:", REMOTE_RESULTS_NPZ)
print("Saved result summary:", REMOTE_RESULTS_JSON)
print("Saved human-readable summary:", REMOTE_TEXT_SUMMARY)
'''


_GRATING_ANALYSIS_REMOTE = r'''# Extract measured-input-normalized grating power on Lambda.

if not GRATING_ANALYSIS:
    print("No grating coupler was exported; grating analysis is not required.")
elif not SETTINGS.get("run_after_build", False):
    print("Grating analysis is ready but unsolved. Set SETTINGS['run_after_build'] = True and rerun from section 6.")
else:
    excitation_type = str(
        GRATING_ANALYSIS.get("excitation_type", "fiber_mode")
    ).strip().lower()
    input_source_label = (
        "Gaussian-beam input" if excitation_type == "gaussian_beam"
        else "fiber-mode input"
    )
    fiber_input_monitor_name = str(GRATING_ANALYSIS["fiber_input_power_monitor_name"])
    fiber_input_sign = float(GRATING_ANALYSIS.get("fiber_input_power_sign", -1.0))
    waveguide_port_name = str(GRATING_ANALYSIS["waveguide_port_name"])
    waveguide_port_result_name = str(
        GRATING_ANALYSIS.get(
            "waveguide_port_expansion_result_name", "expansion for port monitor"
        )
    )
    waveguide_modal_key = str(
        GRATING_ANALYSIS.get("waveguide_port_modal_direction", "T_out")
    )
    waveguide_power_monitor_name = str(GRATING_ANALYSIS["waveguide_power_monitor_name"])
    waveguide_total_sign = float(
        GRATING_ANALYSIS.get("waveguide_total_power_sign", -1.0)
    )
    waveguide_modal_sign = float(
        GRATING_ANALYSIS.get("waveguide_port_modal_sign", waveguide_total_sign)
    )

    def _normalized_result_key(value):
        return "".join(character.lower() for character in str(value) if character.isalnum())

    def _find_result_key(dataset, *candidates):
        try:
            available = list(dataset.keys())
        except Exception:
            available = []
        for candidate in candidates:
            if candidate in available:
                return candidate
        normalized = {_normalized_result_key(key): key for key in available}
        for candidate in candidates:
            match = normalized.get(_normalized_result_key(candidate))
            if match is not None:
                return match
        raise KeyError("none of %r is present; available fields: %r" % (candidates, available))

    def _one_spectrum(
        dataset, value_key, magnitude=True,
        selected_mode_number=0, selected_mode_order=None,
    ):
        wavelength_key = _find_result_key(dataset, "lambda", "wavelength") if any(
            _normalized_result_key(key) in {"lambda", "wavelength"} for key in dataset.keys()
        ) else None
        if wavelength_key is not None:
            wavelength = np.squeeze(np.asarray(dataset[wavelength_key], dtype=float)).ravel()
        else:
            frequency_key = _find_result_key(dataset, "f", "frequency")
            wavelength = 299792458.0 / np.squeeze(np.asarray(dataset[frequency_key], dtype=float)).ravel()
        resolved_value_key = _find_result_key(dataset, value_key)
        values = np.squeeze(np.asarray(dataset[resolved_value_key]))
        raw_value_shape = tuple(values.shape)
        if values.ndim == 0:
            values = np.full(wavelength.size, values)
        elif values.ndim == 1:
            values = values.ravel()
        else:
            wavelength_axes = [axis for axis, size in enumerate(values.shape) if size == wavelength.size]
            if not wavelength_axes:
                raise RuntimeError(
                    "Could not align %s data shape %s with %d wavelengths"
                    % (resolved_value_key, values.shape, wavelength.size)
                )
            values = np.moveaxis(values, wavelength_axes[0], 0)
            if int(selected_mode_number) > 0:
                selected_mode_number = int(selected_mode_number)
                selected_mode_order = [
                    int(value) for value in (selected_mode_order or [])
                    if int(value) > 0
                ]
                mode_coordinates = np.asarray(dataset.get("n", [])).squeeze().ravel()
                selected_from_coordinate = False
                if mode_coordinates.size:
                    mode_axes = [
                        axis for axis, size in enumerate(values.shape[1:], start=1)
                        if size == mode_coordinates.size
                    ]
                    matches = np.flatnonzero(np.isclose(
                        mode_coordinates.astype(float), float(selected_mode_number)
                    ))
                    if mode_axes and matches.size:
                        values = np.take(values, int(matches[0]), axis=mode_axes[0])
                        selected_from_coordinate = True
                if not selected_from_coordinate:
                    flattened = values.reshape(wavelength.size, -1)
                    if selected_mode_number in selected_mode_order:
                        selected_column = selected_mode_order.index(selected_mode_number)
                    elif flattened.shape[1] == 1:
                        selected_column = 0
                    else:
                        raise RuntimeError(
                            "Cannot identify selected fiber mode %d in %s shape %s; "
                            "dataset n=%r, retained order=%r"
                            % (
                                selected_mode_number, resolved_value_key,
                                raw_value_shape, mode_coordinates.tolist(),
                                selected_mode_order,
                            )
                        )
                    if selected_column >= flattened.shape[1]:
                        raise RuntimeError(
                            "Selected fiber mode %d maps to column %d but %s shape %s has only %d modal columns"
                            % (
                                selected_mode_number, selected_column,
                                resolved_value_key, raw_value_shape,
                                flattened.shape[1],
                            )
                        )
                    values = flattened[:, selected_column]
                else:
                    values = np.squeeze(values)
                    if values.ndim > 1:
                        flattened = values.reshape(wavelength.size, -1)
                        if flattened.shape[1] != 1:
                            raise RuntimeError(
                                "Selected fiber mode %d left ambiguous %s shape %s after using dataset n=%r"
                                % (
                                    selected_mode_number, resolved_value_key,
                                    values.shape, mode_coordinates.tolist(),
                                )
                            )
                        values = flattened[:, 0]
            else:
                values = values.reshape(wavelength.size, -1)[:, 0]
        if values.size != wavelength.size:
            raise RuntimeError(
                "%s returned %d values for %d wavelengths"
                % (resolved_value_key, values.size, wavelength.size)
            )
        order = np.argsort(wavelength)
        ordered_values = values[order]
        if magnitude:
            ordered_values = np.abs(ordered_values)
        else:
            ordered_values = np.real(ordered_values)
        return wavelength[order], np.asarray(ordered_values, dtype=float)

    def _port_expansion(port_name, requested_result_name):
        attempts = []
        paths = (
            "FDTD::ports::" + port_name,
            "::model::FDTD::ports::" + port_name,
        )
        result_names = tuple(dict.fromkeys((requested_result_name, "expansion for port monitor")))
        for path in paths:
            for result_name in result_names:
                try:
                    return fdtd.getresult(path, result_name), path, result_name
                except Exception as exc:
                    attempts.append("%s / %s: %s" % (path, result_name, str(exc)[:180]))
        available = ""
        for path in paths:
            try:
                available = str(fdtd.getresult(path))
                if available:
                    break
            except Exception:
                continue
        raise RuntimeError(
            "The passive waveguide receiver %r has no readable 'expansion for port monitor' result. "
            "Available results: %s. Attempts: %s"
            % (port_name, available, " | ".join(attempts))
        )

    try:
        fiber_input_data = fdtd.getresult(fiber_input_monitor_name, "T")
        fiber_wavelength_m, fiber_signed_flux = _one_spectrum(
            fiber_input_data, "T", magnitude=False
        )
    except Exception as exc:
        raise RuntimeError(
            "The %s power monitor %r has no readable signed T result: %s"
            % (input_source_label, fiber_input_monitor_name, exc)
        ) from None

    waveguide_expansion, waveguide_result_path, resolved_waveguide_result = _port_expansion(
        waveguide_port_name, waveguide_port_result_name
    )
    mode_selection = dict(PORT_MODE_SELECTIONS.get(waveguide_port_name, {}))
    selected_mode_number = max(1, int(mode_selection.get("mode number", 1)))
    selected_mode_order = list(
        mode_selection.get("selected mode order", [selected_mode_number])
    )
    try:
        wavelengths_m, waveguide_mode_power = _one_spectrum(
            waveguide_expansion,
            waveguide_modal_key,
            magnitude=False,
            selected_mode_number=selected_mode_number,
            selected_mode_order=selected_mode_order,
        )
    except Exception as exc:
        raise RuntimeError(
            "The waveguide receiver %r result %r has no readable %r spectrum: %s"
            % (waveguide_port_name, resolved_waveguide_result, waveguide_modal_key, exc)
        ) from None

    try:
        waveguide_power_data = fdtd.getresult(waveguide_power_monitor_name, "T")
        power_wavelength_m, waveguide_total_signed_flux = _one_spectrum(
            waveguide_power_data, "T", magnitude=False
        )
    except Exception as exc:
        raise RuntimeError(
            "The waveguide total-power monitor %r has no readable signed T result: %s"
            % (waveguide_power_monitor_name, exc)
        ) from None

    fiber_input_signed_raw = np.interp(
        wavelengths_m, fiber_wavelength_m, fiber_signed_flux
    )
    fiber_input_power = fiber_input_sign * fiber_input_signed_raw
    waveguide_total_signed_raw = np.interp(
        wavelengths_m, power_wavelength_m, waveguide_total_signed_flux
    )
    waveguide_total_power = waveguide_total_sign * waveguide_total_signed_raw

    # Port-expansion T_out is a signed Poynting-flux quantity.  Its sign
    # follows the receiver plane normal, so a left/down-facing output is
    # negative even when all power is physically outgoing.  Apply the same
    # geometry-derived propagation sign used by the adjacent total-power
    # monitor; do not reject a valid outward wave solely because its normal is
    # negative X/Y.
    waveguide_mode_signed_raw = np.real(
        np.asarray(waveguide_mode_power, dtype=float)
    )
    waveguide_mode_power = waveguide_modal_sign * waveguide_mode_signed_raw
    normalization_floor = 1e-15
    if not np.all(np.isfinite(fiber_input_power)) or float(np.min(fiber_input_power)) <= normalization_floor:
        raise RuntimeError(
            "The %s monitor has wrong/near-zero signed power after applying sign %.0f: range [%.6g, %.6g]. "
            "The source must propagate downward (-Z) through this Z-normal monitor."
            % (
                input_source_label,
                fiber_input_sign,
                float(np.nanmin(fiber_input_power)),
                float(np.nanmax(fiber_input_power)),
            )
        )
    if float(np.nanmin(waveguide_mode_power)) < -1e-9:
        raise RuntimeError(
            "Waveguide receiver %s/%s has the wrong propagation sign after applying %.0f "
            "(minimum %.6g). Check the receiver orientation."
            % (
                waveguide_result_path, waveguide_modal_key,
                waveguide_modal_sign, float(np.nanmin(waveguide_mode_power)),
            )
        )
    if float(np.nanmin(waveguide_total_power)) < -1e-9:
        raise RuntimeError(
            "Waveguide total-power monitor has the wrong propagation sign after applying %.0f (minimum %.6g)."
            % (waveguide_total_sign, float(np.nanmin(waveguide_total_power)))
        )
    fiber_coupling = waveguide_mode_power / np.maximum(fiber_input_power, normalization_floor)
    waveguide_total_transmission = waveguide_total_power / np.maximum(
        fiber_input_power, normalization_floor
    )
    fiber_coupling_db = 10.0 * np.log10(np.maximum(fiber_coupling, normalization_floor))
    waveguide_total_transmission_db = 10.0 * np.log10(
        np.maximum(waveguide_total_transmission, normalization_floor)
    )
    if not np.all(np.isfinite(fiber_coupling)) or float(np.max(fiber_coupling)) > 1.05:
        raise RuntimeError(
            "Unphysical grating modal coupling after measured-input normalization: maximum %.6g."
            % float(np.nanmax(fiber_coupling))
        )
    selected_neff = float(mode_selection.get("neff", GRATING_ANALYSIS["waveguide_target_neff"]))
    print(
        "Measured input Pin = %.0f * real(%s.T); no cos(theta) correction is applied because the full projected beam is captured."
        % (
            fiber_input_sign,
            fiber_input_monitor_name,
        )
    )
    print(
        "Selected-TE output uses %.0f * real(%s / %s field %s), waveguide mode %d (neff %.6g)."
        % (
            waveguide_modal_sign,
            waveguide_result_path,
            resolved_waveguide_result,
            waveguide_modal_key,
            selected_mode_number,
            selected_neff,
        )
    )
    print(
        "Reported traces: selected-TE receiver power / measured Pin and nearby total waveguide power / measured Pin."
    )

    wavelength_target_m = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    ) * 1e-6
    analysis_arrays = {
        "wavelength_m": wavelengths_m,
        "input_power_signed_raw": fiber_input_signed_raw,
        "input_power": fiber_input_power,
        "fiber_input_power_signed_raw": fiber_input_signed_raw,
        "fiber_input_power": fiber_input_power,
        "waveguide_mode_power_source_normalized": waveguide_mode_power,
        "waveguide_mode_power_signed_raw": waveguide_mode_signed_raw,
        "waveguide_mode_power": waveguide_mode_power,
        "waveguide_total_power_signed_raw": waveguide_total_signed_raw,
        "waveguide_total_power": waveguide_total_power,
        "waveguide_transmission": fiber_coupling,
        "waveguide_total_transmission": waveguide_total_transmission,
        "waveguide_total_transmission_db": waveguide_total_transmission_db,
        "fiber_coupling": fiber_coupling,
        "coupling_efficiency": fiber_coupling,
        "fiber_coupling_db": fiber_coupling_db,
        "fiber_source_mode_number": np.asarray([
            int(GRATING_ANALYSIS.get("fiber_source_mode_number", 0))
        ]),
        "waveguide_selected_neff": np.asarray([selected_neff]),
        "waveguide_selected_mode_number": np.asarray([selected_mode_number]),
        "target_wavelength_m": np.asarray([wavelength_target_m]),
    }
    GRATING_RESULT_ARRAYS = analysis_arrays

    grating_npz = os.path.join(REMOTE_WORK, "grating_analysis.npz")
    np.savez_compressed(grating_npz, **analysis_arrays)
    if not os.path.isfile(grating_npz) or os.path.getsize(grating_npz) <= 0:
        raise RuntimeError("Required grating artifact was not created: " + grating_npz)
    print("Saved grating analysis:", grating_npz)
'''


_MMI_ANALYSIS_REMOTE = r'''# Mode-1 input to the two MMI output waveguides.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not MMI_ANALYSIS:
    print("No supported 1x2 MMI was exported; splitting-ratio analysis is not required.")
elif not SETTINGS.get("run_after_build", False):
    print("MMI analysis is ready but unsolved. Enable run_after_build and rerun from the solve section.")
else:
    input_port_name = str(MMI_ANALYSIS["input_port_name"])
    input_reference_monitor_name = str(MMI_ANALYSIS["input_reference_monitor_name"])
    output_port_names = list(map(str, MMI_ANALYSIS["output_port_names"]))
    output_labels = list(map(str, MMI_ANALYSIS["output_labels"]))
    mmi_port_names = [input_port_name, *output_port_names]
    mmi_port_neff = np.asarray([
        float(PORT_MODE_SELECTIONS.get(name, {}).get("neff", np.nan))
        for name in mmi_port_names
    ])
    print(
        "MMI access-port selected effective indices:",
        ", ".join(
            "%s=%.6g" % (name, value)
            for name, value in zip(mmi_port_names, mmi_port_neff)
        ),
    )

    def _port_transmission(port_name):
        port_path = "::model::FDTD::ports::" + port_name
        try:
            T_data = fdtd.getresult(port_path, "T")
        except Exception as exc:
            raise RuntimeError(
                "MMI output port %r has no T result. Confirm that it is inside the FDTD region and rerun the solve. %s"
                % (port_name, exc)
            ) from None
        wavelength = np.squeeze(np.asarray(T_data["lambda"], dtype=float)).ravel()
        power = np.abs(np.squeeze(np.asarray(T_data["T"]))).ravel()
        if power.size != wavelength.size:
            wavelength_axes = [axis for axis, size in enumerate(np.asarray(T_data["T"]).shape) if size == wavelength.size]
            if not wavelength_axes:
                raise RuntimeError("Could not align MMI port %s T data with wavelength" % port_name)
            values = np.moveaxis(np.asarray(T_data["T"]), wavelength_axes[0], 0)
            power = np.abs(values.reshape(wavelength.size, -1)[:, 0])
        order = np.argsort(wavelength)
        return wavelength[order], power[order]

    def _reference_transmission(monitor_name):
        try:
            T_data = fdtd.getresult("::model::" + monitor_name, "T")
        except Exception as exc:
            raise RuntimeError(
                "MMI input reference monitor %r has no T result. %s" % (monitor_name, exc)
            ) from None
        wavelength = np.squeeze(np.asarray(T_data["lambda"], dtype=float)).ravel()
        power = np.abs(np.squeeze(np.asarray(T_data["T"]))).ravel()
        order = np.argsort(wavelength)
        return wavelength[order], power[order]

    wavelength_m, output_1_power = _port_transmission(output_port_names[0])
    wavelength_2_m, output_2_power = _port_transmission(output_port_names[1])
    input_wavelength_m, input_power = _reference_transmission(input_reference_monitor_name)
    if wavelength_2_m.size != wavelength_m.size or not np.allclose(wavelength_2_m, wavelength_m, rtol=1e-9, atol=1e-15):
        output_2_power = np.interp(wavelength_m, wavelength_2_m, output_2_power)
    if input_wavelength_m.size != wavelength_m.size or not np.allclose(input_wavelength_m, wavelength_m, rtol=1e-9, atol=1e-15):
        input_power = np.interp(wavelength_m, input_wavelength_m, input_power)

    total_output_power = output_1_power + output_2_power
    safe_total = np.maximum(total_output_power, 1e-15)
    safe_input = np.maximum(input_power, 1e-15)
    output_1_ratio = output_1_power / safe_total
    output_2_ratio = output_2_power / safe_total
    output_1_over_input = output_1_power / safe_input
    output_2_over_input = output_2_power / safe_input
    total_output_over_input = total_output_power / safe_input
    target_wavelength_m = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    ) * 1e-6
    target_index = int(np.argmin(np.abs(wavelength_m - target_wavelength_m)))
    symmetry_error_percent = abs(output_1_ratio[target_index] - 0.5) * 100.0
    print(
        "MMI transfer at %.3f nm: Pin %.6g, %s/Pin %.3f%%, %s/Pin %.3f%%, total/Pin %.3f%%; "
        "transmitted split %.3f%% / %.3f%%; symmetry error %.4f percentage points"
        % (
            wavelength_m[target_index] * 1e9,
            input_power[target_index],
            output_labels[0], output_1_over_input[target_index] * 100.0,
            output_labels[1], output_2_over_input[target_index] * 100.0,
            total_output_over_input[target_index] * 100.0,
            output_1_ratio[target_index] * 100.0,
            output_2_ratio[target_index] * 100.0,
            symmetry_error_percent,
        )
    )
    symmetry_tolerance_percent = float(MMI_ANALYSIS.get("symmetry_tolerance_percent", 1.0))
    if symmetry_error_percent <= symmetry_tolerance_percent:
        print("Verified symmetric 50/50 MMI within %.3f percentage points." % symmetry_error_percent)
    else:
        print(
            "WARNING: symmetric MMI differs from 50/50 by %.3f percentage points; check mesh and port placement."
            % symmetry_error_percent
        )

    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
    axes[0].plot(wavelength_m * 1e9, output_1_over_input, lw=2.2, label=output_labels[0] + " / input")
    axes[0].plot(wavelength_m * 1e9, output_2_over_input, lw=2.2, label=output_labels[1] + " / input")
    axes[0].plot(wavelength_m * 1e9, total_output_over_input, lw=1.7, ls="--", label="total output / input")
    axes[0].set_ylabel("output / measured input (linear)")
    axes[0].set_title("MMI branch transmission — mode 1 input")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(wavelength_m * 1e9, output_1_ratio, lw=2.0, label=output_labels[0] + " / total output")
    axes[1].plot(wavelength_m * 1e9, output_2_ratio, lw=2.0, label=output_labels[1] + " / total output")
    axes[1].axhline(0.5, color="#64748b", ls="--", lw=1.0, label="ideal 50/50")
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("split fraction (linear)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Secondary symmetry diagnostic — output / total transmitted output")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()

    mmi_png = os.path.join(REMOTE_WORK, "mmi_splitting_ratio.png")
    figure.savefig(mmi_png, dpi=170, bbox_inches="tight")
    plt.close(figure)

    # Longitudinal mode-1 field at the center wavelength.  The source is the
    # input port's selected fundamental mode, so this is the actual field as it
    # expands and interferes through the complete MMI.
    field_monitor_name = str(MMI_ANALYSIS["field_monitor_name"])
    field_monitor_path = "::model::" + field_monitor_name
    try:
        field_result = fdtd.getresult(field_monitor_path, "E")
    except Exception as exc:
        raise RuntimeError(
            "MMI field monitor %r has no E result. Confirm that the Z-normal monitor lies inside the FDTD region. %s"
            % (field_monitor_name, exc)
        ) from None
    field_x_m = np.squeeze(np.asarray(field_result["x"], dtype=float)).ravel()
    field_y_m = np.squeeze(np.asarray(field_result["y"], dtype=float)).ravel()
    field_frequency_hz = np.squeeze(np.asarray(field_result.get("f", []), dtype=float)).ravel()
    if field_frequency_hz.size:
        target_frequency_hz = 299792458.0 / target_wavelength_m
        field_frequency_index = int(np.argmin(np.abs(field_frequency_hz - target_frequency_hz)))
    else:
        field_frequency_index = 0

    try:
        field_available = list(field_result.keys())
    except Exception:
        field_available = []
    field_shapes = {}
    for field_name in field_available:
        try:
            field_shapes[str(field_name)] = tuple(np.asarray(field_result[field_name]).shape)
        except Exception:
            pass

    def _field_value(field_name):
        try:
            values = np.asarray(field_result[field_name])
        except (KeyError, TypeError, IndexError):
            return None
        return values if values.size else None

    def _field_plane(values, field_label):
        values = np.asarray(values)
        if not values.size:
            raise RuntimeError("MMI field %s is empty" % field_label)
        # Ansys raw monitor arrays are normally Nx x Ny x 1 x Nf.  Select
        # the center-wavelength frequency before removing singleton axes.
        if field_frequency_hz.size and values.ndim > 2:
            frequency_axes = [axis for axis, size in enumerate(values.shape) if size == field_frequency_hz.size]
            if frequency_axes:
                values = np.take(values, field_frequency_index, axis=frequency_axes[-1])
        values = np.squeeze(values)
        if values.shape == (field_y_m.size, field_x_m.size):
            values = values.T
        expected_shape = (field_x_m.size, field_y_m.size)
        if values.shape != expected_shape and values.size == field_x_m.size * field_y_m.size:
            values = values.reshape(field_x_m.size, field_y_m.size)
        if values.shape != expected_shape:
            raise RuntimeError(
                "MMI field %s has shape %s; expected %s. Available E-result fields/shapes: %s"
                % (field_label, values.shape, expected_shape, field_shapes)
            )
        return values

    # Collect the requested Cartesian fields independently of lumapi's
    # dataset-expansion style.  Some releases expose Ex/Ey/Ez as dataset
    # attributes, while others require the official raw getdata interface.
    field_component_planes = {}
    field_component_sources = {}
    field_attempts = []
    for component_name in ("Ex", "Ey", "Ez"):
        component_values = _field_value(component_name)
        component_source = "E-result attribute"
        if component_values is None:
            try:
                component_values = np.asarray(
                    fdtd.getdata(field_monitor_path, component_name, 1)
                )
                component_source = "Ansys getdata"
            except Exception as exc:
                field_attempts.append(component_name + " getdata: " + str(exc))
                component_values = None
        if component_values is not None:
            try:
                field_component_planes[component_name] = _field_plane(
                    component_values, component_name
                )
                field_component_sources[component_name] = component_source
            except Exception as exc:
                field_attempts.append(component_name + ": " + str(exc))

    electric_vector = _field_value("E")
    if electric_vector is not None and len(field_component_planes) < 3:
        component_axes = [
            axis for axis, size in enumerate(electric_vector.shape) if size == 3
        ]
        if component_axes:
            component_axis = (
                electric_vector.ndim - 1
                if electric_vector.shape[-1] == 3
                else component_axes[-1]
            )
            electric_vector = np.moveaxis(electric_vector, component_axis, -1)
            for component_index, component_name in enumerate(("Ex", "Ey", "Ez")):
                if component_name in field_component_planes:
                    continue
                try:
                    field_component_planes[component_name] = _field_plane(
                        electric_vector[..., component_index], component_name
                    )
                    field_component_sources[component_name] = "vector E attribute"
                except Exception as exc:
                    field_attempts.append(component_name + " from vector E: " + str(exc))
        else:
            field_attempts.append(
                "vector E: no Cartesian component axis in shape %s"
                % (electric_vector.shape,)
            )

    # `getelectric` is Ansys's direct |Ex|^2+|Ey|^2+|Ez|^2 monitor command.
    # Prefer it over assumptions about how lumapi expands vector attributes.
    field_intensity = None
    field_intensity_source = ""
    try:
        field_intensity = _field_plane(
            fdtd.getelectric(field_monitor_path, 1), "getelectric(E2)"
        )
        field_intensity_source = "Ansys getelectric"
    except Exception as exc:
        field_attempts.append("getelectric: " + str(exc))

    if field_intensity is None:
        electric_intensity = _field_value("E2")
        if electric_intensity is not None:
            try:
                field_intensity = _field_plane(electric_intensity, "E2")
                field_intensity_source = "E2 attribute"
            except Exception as exc:
                field_attempts.append("E2: " + str(exc))

    if field_intensity is None:
        if field_component_planes:
            field_intensity = sum(
                np.abs(values) ** 2 for values in field_component_planes.values()
            )
            field_intensity_source = "+".join(field_component_planes)

    if field_intensity is None:
        raise RuntimeError(
            "MMI monitor %r has no readable electric-field intensity. Available E-result fields/shapes: %s. Attempts: %s"
            % (field_monitor_name, field_shapes, " | ".join(field_attempts))
        )
    field_intensity = np.maximum(np.real(np.asarray(field_intensity, dtype=complex)), 0.0)
    if not np.any(np.isfinite(field_intensity)):
        raise RuntimeError("MMI field intensity contains no finite samples")
    print("MMI longitudinal |E|^2 uses:", field_intensity_source)
    missing_display_components = [
        name for name in ("Ex", "Ey") if name not in field_component_planes
    ]
    if missing_display_components:
        raise RuntimeError(
            "MMI monitor %r did not record the requested %s field map(s). Available E-result fields/shapes: %s. Attempts: %s. "
            "Rebuild and solve with the updated notebook, which explicitly enables output Ex, Ey, and Ez."
            % (
                field_monitor_name,
                ", ".join(missing_display_components),
                field_shapes,
                " | ".join(field_attempts),
            )
        )
    print(
        "MMI component maps:",
        ", ".join(
            "%s via %s" % (name, field_component_sources[name])
            for name in ("Ex", "Ey")
        ),
    )
    field_peak = max(float(np.nanmax(field_intensity)), 1e-30)
    field_intensity_normalized = field_intensity / field_peak
    field_wavelength_m = (
        299792458.0 / field_frequency_hz[field_frequency_index]
        if field_frequency_hz.size else target_wavelength_m
    )
    transverse_peak = max(
        float(np.nanmax(np.abs(field_component_planes["Ex"]))),
        float(np.nanmax(np.abs(field_component_planes["Ey"]))),
        1e-30,
    )
    field_ex_abs_normalized = np.abs(field_component_planes["Ex"]) / transverse_peak
    field_ey_abs_normalized = np.abs(field_component_planes["Ey"]) / transverse_peak
    field_figure, field_axes = plt.subplots(
        2, 1, figsize=(11.0, 8.0), sharex=True, sharey=True
    )
    for field_axis, component_name, component_values in (
        (field_axes[0], "Ex", field_ex_abs_normalized),
        (field_axes[1], "Ey", field_ey_abs_normalized),
    ):
        field_image = field_axis.pcolormesh(
            field_x_m * 1e6,
            field_y_m * 1e6,
            component_values.T,
            shading="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        field_axis.set_aspect("equal", adjustable="box")
        field_axis.set_ylabel("y [um]")
        field_axis.set_title("MMI |%s|" % component_name)
        field_figure.colorbar(
            field_image,
            ax=field_axis,
            label="magnitude / common Ex-Ey peak",
        )
    field_axes[-1].set_xlabel("x [um]")
    field_figure.suptitle(
        "MMI solved Ex and Ey fields at %.3f nm" % (field_wavelength_m * 1e9)
    )
    field_figure.tight_layout()
    mmi_field_png = os.path.join(REMOTE_WORK, "mmi_field_distribution.png")
    field_figure.savefig(mmi_field_png, dpi=180, bbox_inches="tight")
    plt.close(field_figure)

    mmi_npz = os.path.join(REMOTE_WORK, "mmi_analysis.npz")
    np.savez_compressed(
        mmi_npz,
        wavelength_m=wavelength_m,
        output_1_power=output_1_power,
        output_2_power=output_2_power,
        input_power=input_power,
        output_1_ratio=output_1_ratio,
        output_2_ratio=output_2_ratio,
        output_1_split_fraction=output_1_ratio,
        output_2_split_fraction=output_2_ratio,
        output_1_over_input=output_1_over_input,
        output_2_over_input=output_2_over_input,
        total_output_over_input=total_output_over_input,
        symmetry_error_percent=np.abs(output_1_ratio - 0.5) * 100.0,
        total_output_power=total_output_power,
        target_wavelength_m=np.asarray([target_wavelength_m]),
        port_names=np.asarray(mmi_port_names),
        port_selected_neff=mmi_port_neff,
        port_target_neff=np.asarray([
            float(MMI_ANALYSIS["port_target_neff"])
        ]),
        field_x_m=field_x_m,
        field_y_m=field_y_m,
        field_Ex=field_component_planes["Ex"],
        field_Ey=field_component_planes["Ey"],
        field_Ex_abs_normalized=field_ex_abs_normalized,
        field_Ey_abs_normalized=field_ey_abs_normalized,
        field_intensity_normalized=field_intensity_normalized,
        field_wavelength_m=np.asarray([field_wavelength_m]),
    )
    for required_path in (mmi_png, mmi_field_png, mmi_npz):
        if not os.path.isfile(required_path) or os.path.getsize(required_path) <= 0:
            raise RuntimeError("Required MMI artifact was not created: " + required_path)
    print("Saved MMI splitting plot:", mmi_png)
    print("Saved MMI longitudinal field plot:", mmi_field_png)
    print("Saved MMI analysis:", mmi_npz)
'''


_SWEEP_RUNTIME_REMOTE = r'''# Fast in-session Layer Builder sweep support.
import json
import os
import re
import socket
import time
import uuid
import numpy as np

SWEEP_CHECKPOINT_SCHEMA = 6
SWEEP_RUNTIME_VERSION = "max-layout-sweep-runtime-v6"
SWEEP_CODE_FINGERPRINT = str(globals().get(
    "SWEEP_CODE_FINGERPRINT", SWEEP_RUNTIME_VERSION
))


class SweepResultSchemaError(RuntimeError):
    """A solved sweep point does not expose the result provider/schema we require."""

SWEEP_CHECKPOINT_DIR = globals().get(
    "SWEEP_SHARED_CHECKPOINT_DIR",
    os.path.join(REMOTE_WORK, "sweep-checkpoint-" + str(SWEEP_HASH)[:12]),
)
os.makedirs(SWEEP_CHECKPOINT_DIR, exist_ok=True)
SWEEP_PROGRESS_FILE = str(globals().get(
    "SWEEP_PROGRESS_FILE", os.path.join(REMOTE_WORK, "sweep_live_progress.jsonl")
))
_SWEEP_PROGRESS_SEQUENCE = 0
_SWEEP_PROGRESS_STARTED = time.monotonic()
SWEEP_BASE_PORTS_BY_NAME = {
    str(port.get("name", "")): dict(port) for port in PORTS
}
SWEEP_BASE_PORT_Z_M = {}
SWEEP_BASE_MONITORS_BY_NAME = {
    str(monitor.get("name", "")): dict(monitor) for monitor in MONITORS
}
SWEEP_BASE_MONITOR_Z_M = {}
SWEEP_BASE_GAUSSIAN_SOURCES_BY_NAME = {
    str(source.get("name", "")): dict(source)
    for source in globals().get("GAUSSIAN_SOURCES", [])
}
SWEEP_BASE_GAUSSIAN_SOURCE_Z_M = {}
_SWEEP_SEED_MODE_SELECTIONS = dict(globals().get(
    "SWEEP_PORT_MODE_SELECTIONS",
    globals().get(
        "SWEEP_FIBER_MODE_SELECTIONS",
        globals().get("PORT_MODE_SELECTIONS", {}),
    ),
))
SWEEP_PORT_MODE_SELECTIONS = {
    str(name): dict(selection)
    for name, selection in _SWEEP_SEED_MODE_SELECTIONS.items()
}
SWEEP_FIBER_MODE_SELECTIONS = {
    str(name): dict(selection)
    for name, selection in SWEEP_PORT_MODE_SELECTIONS.items()
    if "candidate mode numbers" in dict(selection)
}


def _restore_sweep_fiber_mode_contract():
    """Restore resolved local-TE identities and the active source mode."""
    if not GRATING_ANALYSIS:
        return
    if str(GRATING_ANALYSIS.get("excitation_type", "fiber_mode")) == "gaussian_beam":
        if PORTS:
            fdtd.select("FDTD::ports")
            fdtd.set("source port", "")
        source_name = str(GRATING_ANALYSIS.get("source_name", ""))
        for source in GAUSSIAN_SOURCES:
            candidate_name = str(source.get("name", ""))
            if candidate_name:
                try:
                    fdtd.setnamed(
                        "::model::" + candidate_name, "enabled",
                        candidate_name == source_name,
                    )
                except Exception:
                    fdtd.select("::model::" + candidate_name)
                    fdtd.set("enabled", candidate_name == source_name)
        return
    ports_by_name = {str(port.get("name", "")): port for port in PORTS}
    source_name = str(GRATING_ANALYSIS.get("fiber_port_name", ""))
    if not source_name:
        return
    source_selection = dict(SWEEP_FIBER_MODE_SELECTIONS.get(source_name, {}))
    if not source_selection:
        raise RuntimeError(
            "The sweep has no verified local-TE/Ey source mode for %s" % source_name
        )
    source_mode_number = max(1, int(source_selection.get("mode number", 1)))
    selected_order = [
        max(1, int(value))
        for value in source_selection.get(
            "selected mode order",
            source_selection.get("candidate mode numbers", [source_mode_number]),
        )
    ]
    selected_order = list(dict.fromkeys(selected_order)) or [source_mode_number]
    source_port = ports_by_name.get(source_name)
    if source_port is not None:
        source_port["mode number"] = source_mode_number
        source_port["polarization"] = "local TE"
        source_port["selected mode order"] = list(selected_order)
    GRATING_ANALYSIS["fiber_source_mode"] = "mode %d" % source_mode_number
    GRATING_ANALYSIS["fiber_source_mode_number"] = source_mode_number
    GRATING_ANALYSIS["fiber_polarization"] = "local TE"
    GRATING_ANALYSIS["fiber_selected_mode_order"] = list(selected_order)
    fdtd.select("FDTD::ports")
    fdtd.set("source port", source_name)
    fdtd.set("source mode", "mode %d" % source_mode_number)
    configured_source_mode = str(fdtd.get("source mode")).strip()
    if configured_source_mode.lower() != ("mode %d" % source_mode_number).lower():
        raise RuntimeError(
            "The sweep could not activate its verified local-TE/Ey source mode: "
            "requested mode %d, configured %r"
            % (source_mode_number, configured_source_mode)
        )


def _sweep_mode_profile_vector(mode_profile, mode_number):
    """Read one selected fiber mode as a complex Cartesian (..., 3) field."""
    available = list(mode_profile.keys())
    for key in ("E%d" % int(mode_number), "E"):
        if key not in available:
            continue
        electric = np.asarray(mode_profile[key])
        if electric.ndim == 0:
            continue
        component_axis = (
            electric.ndim - 1
            if electric.shape[-1] == 3
            else next(
                (axis for axis in reversed(range(electric.ndim)) if electric.shape[axis] == 3),
                None,
            )
        )
        if component_axis is None:
            continue
        electric = np.moveaxis(electric, component_axis, -1)
        if key == "E":
            mode_coordinates = np.asarray(mode_profile.get("n", [])).ravel()
            if mode_coordinates.size > 1:
                mode_axes = [
                    axis for axis, size in enumerate(electric.shape[:-1])
                    if size == mode_coordinates.size
                ]
                if mode_axes:
                    matches = np.flatnonzero(
                        np.isclose(mode_coordinates.astype(float), float(mode_number))
                    )
                    mode_index = int(matches[0]) if matches.size else int(mode_number) - 1
                    if not 0 <= mode_index < mode_coordinates.size:
                        continue
                    electric = np.take(electric, mode_index, axis=mode_axes[-1])
        electric = np.squeeze(electric)
        while electric.ndim > 3:
            electric = np.take(electric, 0, axis=-2)
        if electric.ndim >= 2 and electric.shape[-1] == 3:
            return np.asarray(electric, dtype=complex)
    raise RuntimeError(
        "Fiber mode %d has no readable vector E field; available fields: %r"
        % (int(mode_number), available)
    )


def _sweep_candidate_neff(port_path, candidate_modes):
    """Read per-mode neff from keyed or wavelength-by-mode sweep datasets."""
    resolver = globals().get("_fiber_candidate_neff")
    if callable(resolver):
        return dict(resolver(fdtd, port_path, candidate_modes))
    dataset = fdtd.getresult(port_path, "neff")
    normalized = {
        "".join(character for character in str(key).lower() if character.isalnum()): key
        for key in dataset.keys()
    }
    result = {}
    for mode_number in candidate_modes:
        key = normalized.get("neff%d" % int(mode_number))
        if key is not None:
            finite = np.real(np.asarray(dataset[key])).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size:
                result[int(mode_number)] = float(np.median(finite))
    if len(result) == len(candidate_modes):
        return result
    plain_key = normalized.get("neff")
    if plain_key is None:
        return result
    values = np.real(np.asarray(dataset[plain_key])).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    mode_key = normalized.get("n")
    if mode_key is not None:
        mode_numbers = [
            int(round(float(value)))
            for value in np.asarray(dataset[mode_key]).squeeze().ravel()
        ]
    else:
        mode_numbers = list(map(int, candidate_modes))
    mode_count = len(mode_numbers)
    if values.ndim == 1 and values.size == mode_count:
        matrix = values.reshape(1, mode_count)
    else:
        axes = [axis for axis, size in enumerate(values.shape) if size == mode_count]
        spectral_key = normalized.get("lambda") or normalized.get("f")
        if len(axes) > 1 and spectral_key is not None:
            spectral_size = np.asarray(dataset[spectral_key]).squeeze().size
            axes = [axis for axis in axes if values.shape[axis] != spectral_size] or axes
        if not axes:
            return result
        matrix = np.moveaxis(values, axes[-1], -1).reshape(-1, mode_count)
    requested = set(map(int, candidate_modes))
    for column, mode_number in enumerate(mode_numbers):
        if mode_number not in requested:
            continue
        finite = np.asarray(matrix[:, column]).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            result[int(mode_number)] = float(np.median(finite))
    return result


def _sweep_gaussian_circular_scores(mode_profile, mode_number):
    """Match the seed selector's Gaussian/circular HE11 quality scoring."""
    seed_scorer = globals().get("_fiber_gaussian_circular_scores")
    if callable(seed_scorer):
        return tuple(map(float, seed_scorer(mode_profile, mode_number)))
    electric = _sweep_mode_profile_vector(mode_profile, mode_number)
    intensity = np.asarray(np.sum(np.abs(electric) ** 2, axis=-1), dtype=float)
    intensity = np.squeeze(intensity)
    while intensity.ndim > 2:
        intensity = np.sum(intensity, axis=0)
    if intensity.ndim != 2 or min(intensity.shape) < 2:
        return 0.0, 0.0
    peak = float(np.max(intensity))
    total = float(np.sum(intensity))
    if not np.isfinite(peak) or peak <= 0.0 or total <= 0.0:
        return 0.0, 0.0
    weights = intensity / total
    axis_0 = np.linspace(-1.0, 1.0, intensity.shape[0])
    axis_1 = np.linspace(-1.0, 1.0, intensity.shape[1])
    grid_0, grid_1 = np.meshgrid(axis_0, axis_1, indexing="ij")
    center_0 = float(np.sum(weights * grid_0))
    center_1 = float(np.sum(weights * grid_1))
    delta_0, delta_1 = grid_0 - center_0, grid_1 - center_1
    covariance = np.asarray([
        [np.sum(weights * delta_0 * delta_0), np.sum(weights * delta_0 * delta_1)],
        [np.sum(weights * delta_0 * delta_1), np.sum(weights * delta_1 * delta_1)],
    ], dtype=float)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    circularity = float(np.clip(eigenvalues[0] / eigenvalues[-1], 0.0, 1.0))
    inverse = np.linalg.pinv(covariance + np.eye(2) * 1e-12)
    radius_squared = (
        inverse[0, 0] * delta_0 ** 2
        + 2.0 * inverse[0, 1] * delta_0 * delta_1
        + inverse[1, 1] * delta_1 ** 2
    )
    gaussian = np.exp(-0.5 * radius_squared)
    similarity = float(
        np.sum(intensity * gaussian)
        / max(np.sqrt(np.sum(intensity ** 2) * np.sum(gaussian ** 2)), 1e-300)
    )
    boundary_peak = float(max(
        np.max(intensity[0, :]), np.max(intensity[-1, :]),
        np.max(intensity[:, 0]), np.max(intensity[:, -1]),
    ))
    quality = similarity * max(0.0, 1.0 - boundary_peak / peak)
    return float(np.clip(quality, 0.0, 1.0)), circularity


def _sweep_reselect_fiber_local_te(port_path, port, previous_selection):
    """Re-score the degenerate pair after a tilted fiber geometry change."""
    selector = globals().get("_select_fiber_local_te_mode")
    if callable(selector):
        return dict(selector(fdtd, port_path, port))
    candidates = [
        int(value)
        for value in port.get(
            "candidate mode numbers",
            previous_selection.get("candidate mode numbers", [1, 2, 3]),
        )
        if int(value) > 0
    ]
    candidates = list(dict.fromkeys(candidates))[:3]
    if len(candidates) < 2:
        raise RuntimeError("A fiber sweep requires at least two candidate modes")
    fdtd.select(port_path)
    fdtd.updateportmodes(np.asarray(candidates, dtype=int))
    profile = fdtd.getresult(port_path, "mode profiles")
    phi_deg = float(port.get("angle phi", 0.0)) % 360.0
    phi = np.deg2rad(phi_deg)
    target_x, target_y = -np.sin(phi), np.cos(phi)
    scores = {}
    gaussian_scores = {}
    circularity_scores = {}
    for mode_number in candidates:
        electric = _sweep_mode_profile_vector(profile, mode_number)
        desired = target_x * electric[..., 0] + target_y * electric[..., 1]
        scores[mode_number] = float(np.sum(np.abs(desired) ** 2)) / max(
            float(np.sum(np.abs(electric) ** 2)), 1e-300
        )
        gaussian_scores[mode_number], circularity_scores[mode_number] = (
            _sweep_gaussian_circular_scores(profile, mode_number)
        )
    neff_delta = None
    selected_pair = tuple(candidates[:2])
    try:
        neff_by_mode = _sweep_candidate_neff(port_path, candidates)
        if len(neff_by_mode) == len(candidates):
            tolerance = max(
                0.0, float(port.get("mode degeneracy tolerance", 0.01))
            )
            pairs = [
                (first, second)
                for pair_index, first in enumerate(candidates)
                for second in candidates[pair_index + 1:]
                if abs(neff_by_mode[first] - neff_by_mode[second]) <= tolerance
            ]
            if not pairs:
                raise RuntimeError(
                    "The first three fiber modes %r contain no near-degenerate pair "
                    "after tilt update: neff=%r, tolerance=%.6g"
                    % (candidates, neff_by_mode, tolerance)
                )
            fiber_target = float(port.get("fiber target neff", 1.44))
            selected_pair = min(
                pairs,
                key=lambda pair: abs(
                    0.5 * (neff_by_mode[pair[0]] + neff_by_mode[pair[1]])
                    - fiber_target
                ),
            )
            neff_delta = abs(
                neff_by_mode[selected_pair[0]] - neff_by_mode[selected_pair[1]]
            )
    except RuntimeError:
        raise
    except Exception:
        # Some Lumerical builds expose only profile fields here.  The nominal
        # seed-build degeneracy check has already validated this same pair.
        neff_delta = None
    composite_scores = {
        mode_number: (
            0.75 * scores[mode_number]
            + 0.15 * gaussian_scores[mode_number]
            + 0.10 * circularity_scores[mode_number]
        )
        for mode_number in selected_pair
    }
    selected_mode = max(selected_pair, key=lambda number: composite_scores[number])
    minimum_fraction = float(port.get("minimum local TE fraction", 0.8))
    if scores[selected_mode] < minimum_fraction:
        raise RuntimeError(
            "No fiber mode is sufficiently normal to the %.6g degree grating axis: %r"
            % (phi_deg, scores)
        )
    partner_mode = next(
        mode_number for mode_number in selected_pair if mode_number != selected_mode
    )
    selected_order = [selected_mode, partner_mode] + [
        mode_number
        for mode_number in candidates
        if mode_number not in {selected_mode, partner_mode}
    ]
    selection = dict(previous_selection)
    selection.update({
        "mode number": int(selected_mode),
        "selected mode order": list(selected_order),
        "candidate mode numbers": list(candidates),
        "degenerate mode pair": list(selected_pair),
        "local TE scores": {str(key): float(value) for key, value in scores.items()},
        "gaussian scores": {
            str(key): float(value) for key, value in gaussian_scores.items()
        },
        "circularity scores": {
            str(key): float(value) for key, value in circularity_scores.items()
        },
        "composite scores": {
            str(key): float(value) for key, value in composite_scores.items()
        },
        "grating axis deg": float(phi_deg),
        "target polarization xy": [float(target_x), float(target_y)],
        "polarization": "local TE",
        "neff degeneracy delta": neff_delta,
    })
    print(
        "Re-selected fiber port %s mode %d from pair %r after solving candidates %r; local-TE scores %r."
        % (port.get("name", port_path), selected_mode, selected_pair, candidates, scores)
    )
    return selection


def _sweep_reuse_verified_fiber_local_te(port_path, port, source_selection):
    """Calculate only the source's verified winner at the passive plane."""
    reusable = globals().get("_reuse_verified_fiber_local_te_mode")
    if callable(reusable):
        return dict(reusable(fdtd, port_path, port, source_selection))
    selected_mode = max(1, int(source_selection.get("mode number", 1)))
    fdtd.select(port_path)
    fdtd.updateportmodes(selected_mode)
    profile = fdtd.getresult(port_path, "mode profiles")
    phi_deg = float(port.get("angle phi", 0.0)) % 360.0
    phi = np.deg2rad(phi_deg)
    target_x, target_y = -np.sin(phi), np.cos(phi)
    electric = _sweep_mode_profile_vector(profile, selected_mode)
    desired = target_x * electric[..., 0] + target_y * electric[..., 1]
    score = float(np.sum(np.abs(desired) ** 2)) / max(
        float(np.sum(np.abs(electric) ** 2)), 1e-300
    )
    if score < float(port.get("minimum local TE fraction", 0.8)):
        return _sweep_reselect_fiber_local_te(
            port_path, port, source_selection
        )
    selection = dict(source_selection)
    selection.update({
        "mode number": selected_mode,
        "selected mode order": [selected_mode],
        "local TE scores": {str(selected_mode): score},
        "grating axis deg": phi_deg,
        "target polarization xy": [float(target_x), float(target_y)],
        "polarization": "local TE",
        "inherited from source mode": True,
    })
    return selection


_restore_sweep_fiber_mode_contract()


def _sweep_json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _sweep_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sweep_json_value(item) for item in value]
    return value


def _emit_sweep_progress(
    status, case_index=-1, completed_count=0, failed_count=0,
    case_seconds=None, primary_name=None, wavelength_m=None, response=None,
    error=None,
):
    """Append one durable status row; reporting must never interrupt a solve."""
    global _SWEEP_PROGRESS_SEQUENCE
    try:
        case_index = int(case_index)
        case = SWEEP_CASES[case_index] if 0 <= case_index < len(SWEEP_CASES) else {}
        row = {
            "progress_type": "sweep",
            "sequence": int(_SWEEP_PROGRESS_SEQUENCE),
            "status": str(status),
            "case_index": case_index,
            "completed_count": int(completed_count),
            "failed_count": int(failed_count),
            "total_count": int(len(SWEEP_CASES)),
            "values": _sweep_json_value(case.get("values", {})),
            "display_label": str(case.get("display_label", "")),
            "elapsed_seconds": float(time.monotonic() - _SWEEP_PROGRESS_STARTED),
        }
        if case_seconds is not None:
            row["case_seconds"] = float(case_seconds)
        if primary_name is not None:
            row["primary_name"] = str(primary_name)
        if wavelength_m is not None and response is not None:
            wavelength = np.asarray(wavelength_m, dtype=float).ravel()
            values = np.asarray(response, dtype=float).ravel()
            finite = np.isfinite(values)
            if wavelength.size == values.size and np.any(finite):
                finite_indices = np.flatnonzero(finite)
                peak_index = int(finite_indices[np.argmax(values[finite])])
                row["peak_response"] = float(values[peak_index])
                row["peak_wavelength_nm"] = float(wavelength[peak_index] / 1e-9)
        if error is not None:
            row["error"] = str(error)[:1000]
        os.makedirs(os.path.dirname(SWEEP_PROGRESS_FILE) or ".", exist_ok=True)
        with open(SWEEP_PROGRESS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _SWEEP_PROGRESS_SEQUENCE += 1
    except Exception as progress_exc:
        print("Sweep progress reporting warning:", str(progress_exc)[:240])


def _sweep_checkpoint_spectrum(case_index):
    with np.load(_sweep_case_npz(case_index), allow_pickle=False) as data:
        return (
            str(np.asarray(data["primary_name"]).ravel()[0]),
            np.asarray(data["wavelength_m"], dtype=float).ravel(),
            np.asarray(data["primary_response"], dtype=float).ravel(),
        )


def _sweep_safe_key(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "result"


def _sweep_set_named(path, property_name, value):
    try:
        fdtd.setnamed(path, property_name, value)
    except Exception:
        fdtd.select(path)
        fdtd.set(property_name, value)


def _sweep_xy(item):
    x_um, y_um = map(float, item.get("center", (0.0, 0.0)))
    if str(item.get("plane normal", "X")).upper() != "Z":
        distance_um = float(item.get("distance_um", 0.0))
        angle_deg = float(item.get("outward_orientation_deg", item.get("orientation_deg", 0.0)))
        x_um += distance_um * np.cos(np.deg2rad(angle_deg))
        y_um += distance_um * np.sin(np.deg2rad(angle_deg))
    return x_um, y_um


def _apply_sweep_case(case_index):
    """Hot-swap polygons and move linked companions without rebuilding materials or FDTD."""
    global GEOMETRY, PORTS, FIBER_GEOMETRIES, GAUSSIAN_SOURCES, MONITORS, GRATING_ANALYSIS, MMI_ANALYSIS
    case = SWEEP_CASES[int(case_index)]
    fdtd.switchtolayout()
    GEOMETRY = list(SWEEP_STATIC_GEOMETRY) + list(case["target_geometry"])
    PORTS = list(case["ports"])
    FIBER_GEOMETRIES = list(case["fiber_geometries"])
    GAUSSIAN_SOURCES = list(case.get("gaussian_sources", []))
    MONITORS = list(case["monitors"])
    GRATING_ANALYSIS = case.get("grating_analysis")
    MMI_ANALYSIS = case.get("mmi_analysis")
    waveguide_index_estimate = dict(
        globals().get("WAVEGUIDE_INDEX_ESTIMATE", {})
    )
    material_target_neff = float(
        waveguide_index_estimate.get("target_neff", 0.0)
    )
    if GRATING_ANALYSIS and material_target_neff > 0.0:
        GRATING_ANALYSIS["waveguide_target_neff"] = material_target_neff
        GRATING_ANALYSIS["waveguide_index_estimate"] = dict(
            waveguide_index_estimate
        )
    if MMI_ANALYSIS and material_target_neff > 0.0:
        MMI_ANALYSIS["port_target_neff"] = material_target_neff
        MMI_ANALYSIS["waveguide_index_estimate"] = dict(
            waveguide_index_estimate
        )
    for item in [*PORTS, *MONITORS]:
        if (
            str(item.get("grating_monitor_role", ""))
            == "waveguide_mode_expansion"
            or "target neff strategy" in item
        ) and material_target_neff > 0.0:
            item["target neff"] = material_target_neff
    _restore_sweep_fiber_mode_contract()

    layer_builder_name = "Max Layout material stack"
    layer_x_um = float(np.asarray(fdtd.getnamed(layer_builder_name, "x")).squeeze()) / UM
    layer_y_um = float(np.asarray(fdtd.getnamed(layer_builder_name, "y")).squeeze()) / UM
    fdtd.setnamed(
        layer_builder_name,
        "geometry",
        _layer_builder_geometry(layer_x_um, layer_y_um, GEOMETRY),
    )

    angle_theta_is_swept = any(
        str(axis.get("parameter", "")) == "angle_theta"
        for axis in SWEEP_SPEC.get("axes", [])
    )
    nonfiber_mode_sensitive_tokens = (
        "width", "gap", "height", "thickness", "index", "diameter",
        "cross_section",
    )
    fiber_only_mode_refresh = bool(
        angle_theta_is_swept
        and not any(
            any(
                token in str(axis.get("parameter", "")).lower()
                for token in nonfiber_mode_sensitive_tokens
            )
            for axis in SWEEP_SPEC.get("axes", [])
        )
    )
    for fiber in FIBER_GEOMETRIES:
        path = "::model::" + str(fiber["name"])
        x_um, y_um = map(float, fiber.get("center", (0.0, 0.0)))
        _sweep_set_named(path, "x", x_um * UM)
        _sweep_set_named(path, "y", y_um * UM)
        if angle_theta_is_swept:
            _sweep_set_named(path, "theta", float(fiber.get("angle theta", 0.0)))
    if angle_theta_is_swept and FIBER_GEOMETRIES:
        # Rebuild the scripted core/cladding cylinders before recalculating
        # the single tilted source-port mode.
        fdtd.runsetup()

    for source in GAUSSIAN_SOURCES:
        source_name = str(source["name"])
        path = "::model::" + source_name
        x_um, y_um = map(float, source.get("center", (0.0, 0.0)))
        _sweep_set_named(path, "x", x_um * UM)
        _sweep_set_named(path, "y", y_um * UM)
        if source_name not in SWEEP_BASE_GAUSSIAN_SOURCE_Z_M:
            SWEEP_BASE_GAUSSIAN_SOURCE_Z_M[source_name] = float(
                np.asarray(fdtd.getnamed(path, "z")).squeeze()
            )
        base_source = SWEEP_BASE_GAUSSIAN_SOURCES_BY_NAME.get(
            source_name, source
        )
        source_distance_delta_um = (
            float(source.get("distance_um", 0.0))
            - float(base_source.get("distance_um", 0.0))
        )
        _sweep_set_named(
            path, "z",
            SWEEP_BASE_GAUSSIAN_SOURCE_Z_M[source_name]
            + source_distance_delta_um * UM,
        )
        span_um = max(1e-6, float(source.get("span_um", 20.0)))
        _sweep_set_named(path, "x span", span_um * UM)
        _sweep_set_named(path, "y span", span_um * UM)
        _sweep_set_named(
            path, "angle theta", float(source.get("angle theta", 0.0))
        )
        _sweep_set_named(
            path, "angle phi", float(source.get("angle phi", 0.0))
        )
        _sweep_set_named(path, "polarization angle", 90.0)
        _sweep_set_named(
            path, "waist radius w0",
            max(1e-9, float(source.get("waist radius_um", 4.5))) * UM,
        )
        _sweep_set_named(
            path, "distance from waist",
            float(source.get("distance from waist_um", 0.0)) * UM,
        )

    for port in PORTS:
        name = str(port["name"])
        path = "::model::FDTD::ports::" + name
        x_um, y_um = _sweep_xy(port)
        _sweep_set_named(path, "x", x_um * UM)
        _sweep_set_named(path, "y", y_um * UM)
        plane_normal = str(port.get("plane normal", "X")).upper()
        span_um = max(0.0, float(port.get("span_um", 2.0)))
        if plane_normal == "X":
            _sweep_set_named(path, "y span", span_um * UM)
        elif plane_normal == "Y":
            _sweep_set_named(path, "x span", span_um * UM)
        else:
            _sweep_set_named(path, "x span", span_um * UM)
            _sweep_set_named(path, "y span", span_um * UM)
            if name not in SWEEP_BASE_PORT_Z_M:
                SWEEP_BASE_PORT_Z_M[name] = float(
                    np.asarray(fdtd.getnamed(path, "z")).squeeze()
                )
            base_port = SWEEP_BASE_PORTS_BY_NAME.get(name, port)
            distance_delta_um = (
                float(port.get("distance_um", 0.0))
                - float(base_port.get("distance_um", 0.0))
            )
            _sweep_set_named(
                path,
                "z",
                float(SWEEP_BASE_PORT_Z_M[name]) + distance_delta_um * UM,
            )
            if angle_theta_is_swept:
                _sweep_set_named(path, "theta", float(port.get("angle theta", 0.0)))
                phi_deg = float(port.get("angle phi", 0.0))
                if abs(phi_deg) > 1e-12:
                    _sweep_set_named(path, "phi", phi_deg)
                _sweep_set_named(
                    path,
                    "rotation offset",
                    float(port.get("rotation offset_um", 0.0)) * UM,
                )

    for monitor in MONITORS:
        monitor_name = str(monitor["name"])
        path = "::model::" + monitor_name
        x_um, y_um = _sweep_xy(monitor)
        _sweep_set_named(path, "x", x_um * UM)
        _sweep_set_named(path, "y", y_um * UM)
        plane_normal = str(monitor.get("plane normal", "X")).upper()
        if plane_normal == "Z":
            if monitor_name not in SWEEP_BASE_MONITOR_Z_M:
                SWEEP_BASE_MONITOR_Z_M[monitor_name] = float(
                    np.asarray(fdtd.getnamed(path, "z")).squeeze()
                )
            base_monitor = SWEEP_BASE_MONITORS_BY_NAME.get(
                monitor_name, monitor
            )
            distance_delta_um = (
                float(monitor.get("distance_um", 0.0))
                - float(base_monitor.get("distance_um", 0.0))
            )
            _sweep_set_named(
                path,
                "z",
                float(SWEEP_BASE_MONITOR_Z_M[monitor_name])
                + distance_delta_um * UM,
            )
        for property_name, key, normal in (
            ("x span", "x span", "X"),
            ("y span", "y span", "Y"),
            ("z span", "z span", "Z"),
        ):
            span_um = max(0.0, float(monitor.get(key, 0.0)))
            if plane_normal != normal and span_um > 0.0:
                _sweep_set_named(path, property_name, span_um * UM)

    if bool(SWEEP_RECOMPUTE_MODES):
        # Geometry width/gap sweeps change modal cross-sections.  Refresh only
        # for those axes; pitch/filling-factor sweeps reuse the nominal modes.
        # An angle sweep already rebuilt the scripted fiber once above.  Do
        # not repeat the full structure-group setup before mode selection.
        if not (angle_theta_is_swept and FIBER_GEOMETRIES):
            fdtd.runsetup()
        source_fiber_name = str(
            GRATING_ANALYSIS.get("fiber_port_name", "")
            if GRATING_ANALYSIS else ""
        )
        # Source first so its rotation-aware local-TE identity is restored
        # before any ordinary access-waveguide port is refreshed.
        ports_for_mode_refresh = sorted(
            PORTS,
            key=lambda candidate: (
                0 if str(candidate.get("name", "")) == source_fiber_name
                else 1
            ),
        )
        for port in ports_for_mode_refresh:
            name = str(port["name"])
            path = "FDTD::ports::" + name
            fdtd.select(path)
            fiber_selection = SWEEP_FIBER_MODE_SELECTIONS.get(name)
            port_selection = SWEEP_PORT_MODE_SELECTIONS.get(name)
            if fiber_only_mode_refresh and not fiber_selection:
                # Fiber tilt does not change any waveguide access plane.
                continue
            if fiber_selection:
                if name == source_fiber_name and angle_theta_is_swept:
                    fiber_selection = _sweep_reselect_fiber_local_te(
                        path, port, fiber_selection
                    )
                else:
                    # The fiber angle/polarization did not change. Recalculate
                    # only the already verified winner after geometry setup.
                    selected_mode = max(
                        1, int(fiber_selection.get("mode number", 1))
                    )
                    fdtd.updateportmodes(selected_mode)
                    fiber_selection = dict(fiber_selection)
                    fiber_selection["selected mode order"] = [selected_mode]
                SWEEP_FIBER_MODE_SELECTIONS[name] = dict(fiber_selection)
                SWEEP_PORT_MODE_SELECTIONS[name] = dict(fiber_selection)
                selected_mode = max(1, int(fiber_selection.get("mode number", 1)))
                selected_order = list(
                    fiber_selection.get("selected mode order", [selected_mode])
                )
                port["mode number"] = selected_mode
                port["selected mode order"] = list(selected_order)
                port["polarization"] = "local TE"
            else:
                mode_number = max(0, int(
                    dict(port_selection or {}).get(
                        "mode number", port.get("mode number", 0)
                    )
                ))
                fdtd.updateportmodes(mode_number) if mode_number else fdtd.updateportmodes()
                if port_selection is not None:
                    refreshed_selection = dict(port_selection)
                    refreshed_selection["mode number"] = mode_number or 1
                    refreshed_selection["selected mode order"] = [mode_number or 1]
                    SWEEP_PORT_MODE_SELECTIONS[name] = refreshed_selection
        _restore_sweep_fiber_mode_contract()
        for monitor in MONITORS:
            if (
                not fiber_only_mode_refresh
                and str(monitor.get("monitor_kind", "")) == "Mode expansion monitor"
            ):
                fdtd.select(str(monitor["name"]))
                fdtd.updatemodes()
        if GRATING_ANALYSIS and not fiber_only_mode_refresh:
            receiver_name = str(GRATING_ANALYSIS.get("waveguide_port_name", ""))
            receiver_port = next(
                (
                    port for port in PORTS
                    if str(port.get("name", "")) == receiver_name
                ),
                {},
            )
            target_neff = float(receiver_port.get(
                "target neff", GRATING_ANALYSIS.get("waveguide_target_neff", 0.0)
            ))
            tolerance = max(0.0, float(receiver_port.get(
                "neff tolerance",
                GRATING_ANALYSIS.get("waveguide_neff_tolerance", 0.3),
            )))
            try:
                neff_data = fdtd.getresult(
                    "FDTD::ports::" + receiver_name, "neff"
                )
                neff_key = _sweep_result_key(neff_data, "neff")
                neff_values = np.real(
                    np.squeeze(np.asarray(neff_data[neff_key]))
                ).ravel()
                finite_neff = neff_values[np.isfinite(neff_values)]
            except Exception as exc:
                raise RuntimeError(
                    "Could not validate refreshed grating receiver %s effective index: %s"
                    % (receiver_name, exc)
                ) from None
            if finite_neff.size < 1:
                raise RuntimeError(
                    "Refreshed grating receiver %s returned no finite effective index"
                    % receiver_name
                )
            selected_neff = float(np.median(finite_neff))
            if target_neff > 0.0 and abs(selected_neff - target_neff) > tolerance:
                raise RuntimeError(
                    "Refreshed grating receiver %s selected neff %.6g outside "
                    "material-derived target %.6g +/- %.6g"
                    % (receiver_name, selected_neff, target_neff, tolerance)
                )
            receiver_selection = dict(
                SWEEP_PORT_MODE_SELECTIONS.get(receiver_name, {})
            )
            receiver_selection.update({
                "neff": selected_neff,
                "target neff": target_neff,
                "neff tolerance": tolerance,
                "mode number": max(
                    1, int(receiver_selection.get("mode number", 1))
                ),
            })
            SWEEP_PORT_MODE_SELECTIONS[receiver_name] = receiver_selection
            print(
                "Validated refreshed grating receiver %s neff %.6g around target %.6g."
                % (receiver_name, selected_neff, target_neff)
            )
        if MMI_ANALYSIS:
            mmi_port_names = [
                str(MMI_ANALYSIS["input_port_name"]),
                *list(map(str, MMI_ANALYSIS["output_port_names"])),
            ]
            ports_by_name = {str(port.get("name", "")): port for port in PORTS}
            selected_indices = []
            for port_name in mmi_port_names:
                port = ports_by_name.get(port_name, {})
                target_neff = float(port.get(
                    "target neff", MMI_ANALYSIS["port_target_neff"]
                ))
                tolerance = max(0.0, float(port.get(
                    "neff tolerance", MMI_ANALYSIS.get("port_neff_tolerance", 0.3)
                )))
                try:
                    neff_data = fdtd.getresult("FDTD::ports::" + port_name, "neff")
                    neff_key = _sweep_result_key(neff_data, "neff")
                    neff_values = np.real(
                        np.squeeze(np.asarray(neff_data[neff_key]))
                    ).ravel()
                    finite_neff = neff_values[np.isfinite(neff_values)]
                except Exception as exc:
                    raise RuntimeError(
                        "Could not validate refreshed MMI port %s effective index: %s"
                        % (port_name, exc)
                    ) from None
                if finite_neff.size < 1:
                    raise RuntimeError(
                        "Refreshed MMI port %s returned no finite effective index" % port_name
                    )
                selected_neff = float(np.median(finite_neff))
                if abs(selected_neff - target_neff) > tolerance:
                    raise RuntimeError(
                        "Refreshed MMI port %s selected neff %.6g outside target %.6g +/- %.6g"
                        % (port_name, selected_neff, target_neff, tolerance)
                    )
                selected_indices.append(selected_neff)
            shared_tolerance = max(
                0.0, float(MMI_ANALYSIS.get("port_neff_tolerance", 0.3))
            )
            if max(selected_indices) - min(selected_indices) > shared_tolerance:
                raise RuntimeError(
                    "Refreshed MMI ports no longer share one effective-index family: %r"
                    % selected_indices
                )
            print(
                "Validated refreshed MMI port neff values:",
                ", ".join("%.6g" % value for value in selected_indices),
            )
    print(
        "Prepared sweep point %d/%d: %s"
        % (int(case_index) + 1, len(SWEEP_CASES), case["display_label"])
    )


def _sweep_normalized_key(value):
    return "".join(character for character in str(value).lower() if character.isalnum())


def _sweep_result_key(dataset, *candidates):
    available = list(dataset.keys())
    for candidate in candidates:
        if candidate in available:
            return candidate
    normalized = {_sweep_normalized_key(key): key for key in available}
    for candidate in candidates:
        match = normalized.get(_sweep_normalized_key(candidate))
        if match is not None:
            return match
    raise KeyError("none of %r is present; available fields: %r" % (candidates, available))


def _sweep_available_result_names(path):
    """Best-effort parsing of LumAPI's version-dependent result-name listing."""
    try:
        raw_names = fdtd.getresult(path)
    except Exception as exc:
        return [], str(exc)

    names = []

    def collect(value):
        if value is None:
            return
        if isinstance(value, dict):
            for key in value.keys():
                collect(key)
            return
        if isinstance(value, str):
            # Current LumAPI returns a newline-delimited string.  Some older
            # builds return one string per array/list entry instead.
            for line in value.replace("\r", "\n").split("\n"):
                cleaned = line.strip()
                if cleaned:
                    names.append(cleaned)
            return
        if isinstance(value, np.ndarray):
            for item in value.ravel().tolist():
                collect(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
            return
        rendered = str(value).strip()
        if rendered:
            names.append(rendered)

    collect(raw_names)
    return list(dict.fromkeys(names)), None


def _sweep_mode_expansion_result(monitor_name, configured_result_name):
    """Resolve a logical setexpansion alias to its actual result dataset."""
    monitor_name = str(monitor_name).strip()
    configured = str(configured_result_name).strip()
    if not monitor_name:
        raise SweepResultSchemaError("The waveguide mode-expansion monitor name is empty")
    if not configured:
        raise SweepResultSchemaError("The waveguide mode-expansion result name is empty")

    prefix_match = re.match(r"^\s*expansion\s+for\s+", configured, flags=re.IGNORECASE)
    logical_name = configured[prefix_match.end():].strip() if prefix_match else configured
    canonical_name = configured if prefix_match else "expansion for " + logical_name
    candidate_names = list(dict.fromkeys((canonical_name, logical_name, configured)))

    if monitor_name.startswith("::model::"):
        full_path = monitor_name
        short_path = monitor_name.rsplit("::", 1)[-1]
    else:
        short_path = monitor_name
        full_path = "::model::" + monitor_name.lstrip(":")
    candidate_paths = list(dict.fromkeys((full_path, short_path)))
    attempts = []

    # The documented result is "expansion for <setexpansion alias>".  Probe
    # that canonical name first, then retain the bare alias as compatibility
    # fallback for projects produced by other Lumerical releases.
    for result_name in candidate_names:
        for path in candidate_paths:
            try:
                return fdtd.getresult(path, result_name), path, result_name
            except Exception as exc:
                attempts.append("%s / %s: %s" % (path, result_name, str(exc)[:180]))

    available_by_path = {}
    discovered_candidates = []
    target_keys = {_sweep_normalized_key(name) for name in candidate_names}
    for path in candidate_paths:
        available, listing_error = _sweep_available_result_names(path)
        available_by_path[path] = available if listing_error is None else "listing failed: " + listing_error[:180]
        exact_matches = [
            name for name in available
            if _sweep_normalized_key(name) in target_keys
        ]
        expansion_matches = [
            name for name in available
            if _sweep_normalized_key(name).startswith("expansionfor")
        ]
        # If the configured alias was renamed while loading an older FSP, a
        # sole expansion dataset is unambiguous.  Never guess when several
        # expansion inputs are exposed by the same monitor.
        names_to_try = exact_matches or (expansion_matches if len(expansion_matches) == 1 else [])
        for result_name in names_to_try:
            pair = (path, result_name)
            if pair not in discovered_candidates:
                discovered_candidates.append(pair)

    for path, result_name in discovered_candidates:
        try:
            return fdtd.getresult(path, result_name), path, result_name
        except Exception as exc:
            attempts.append("%s / %s: %s" % (path, result_name, str(exc)[:180]))

    raise SweepResultSchemaError(
        "The mode-expansion provider %r has no readable result for logical alias %r "
        "(expected %r). Available results: %r. Attempts: %s"
        % (monitor_name, logical_name, canonical_name, available_by_path, " | ".join(attempts))
    )


def _sweep_spectrum(
    dataset, value_key, magnitude=True,
    selected_mode_number=0, selected_mode_order=None,
):
    if any(_sweep_normalized_key(key) in {"lambda", "wavelength"} for key in dataset.keys()):
        wavelength_key = _sweep_result_key(dataset, "lambda", "wavelength")
        wavelength_m = np.squeeze(np.asarray(dataset[wavelength_key], dtype=float)).ravel()
    else:
        frequency_key = _sweep_result_key(dataset, "f", "frequency")
        wavelength_m = 299792458.0 / np.squeeze(np.asarray(dataset[frequency_key], dtype=float)).ravel()
    resolved_key = _sweep_result_key(dataset, value_key)
    values = np.squeeze(np.asarray(dataset[resolved_key]))
    raw_value_shape = tuple(values.shape)
    if values.ndim == 0:
        values = np.full(wavelength_m.size, values)
    elif values.ndim == 1:
        values = values.ravel()
    else:
        wavelength_axes = [axis for axis, size in enumerate(values.shape) if size == wavelength_m.size]
        if not wavelength_axes:
            raise RuntimeError(
                "Could not align %s shape %s with %d wavelengths"
                % (resolved_key, values.shape, wavelength_m.size)
            )
        values = np.moveaxis(values, wavelength_axes[0], 0)
        if int(selected_mode_number) > 0:
            selected_mode_number = int(selected_mode_number)
            selected_mode_order = [
                int(value) for value in (selected_mode_order or [])
                if int(value) > 0
            ]
            mode_coordinates = np.asarray(dataset.get("n", [])).squeeze().ravel()
            selected_from_coordinate = False
            if mode_coordinates.size:
                mode_axes = [
                    axis for axis, size in enumerate(values.shape[1:], start=1)
                    if size == mode_coordinates.size
                ]
                matches = np.flatnonzero(np.isclose(
                    mode_coordinates.astype(float), float(selected_mode_number)
                ))
                if mode_axes and matches.size:
                    values = np.take(values, int(matches[0]), axis=mode_axes[0])
                    selected_from_coordinate = True
            if not selected_from_coordinate:
                flattened = values.reshape(wavelength_m.size, -1)
                if selected_mode_number in selected_mode_order:
                    selected_column = selected_mode_order.index(selected_mode_number)
                elif flattened.shape[1] == 1:
                    selected_column = 0
                else:
                    raise RuntimeError(
                        "Cannot identify selected fiber mode %d in %s shape %s; "
                        "dataset n=%r, retained order=%r"
                        % (
                            selected_mode_number, resolved_key, raw_value_shape,
                            mode_coordinates.tolist(), selected_mode_order,
                        )
                    )
                if selected_column >= flattened.shape[1]:
                    raise RuntimeError(
                        "Selected fiber mode %d maps to column %d but %s shape %s has only %d modal columns"
                        % (
                            selected_mode_number, selected_column, resolved_key,
                            raw_value_shape, flattened.shape[1],
                        )
                    )
                values = flattened[:, selected_column]
            else:
                values = np.squeeze(values)
                if values.ndim > 1:
                    flattened = values.reshape(wavelength_m.size, -1)
                    if flattened.shape[1] != 1:
                        raise RuntimeError(
                            "Selected fiber mode %d left ambiguous %s shape %s after using dataset n=%r"
                            % (
                                selected_mode_number, resolved_key,
                                values.shape, mode_coordinates.tolist(),
                            )
                        )
                    values = flattened[:, 0]
        else:
            values = values.reshape(wavelength_m.size, -1)[:, 0]
    if values.size != wavelength_m.size:
        raise RuntimeError("%s returned %d values for %d wavelengths" % (resolved_key, values.size, wavelength_m.size))
    order = np.argsort(wavelength_m)
    values = values[order]
    values = np.abs(values) if magnitude else np.real(values)
    return wavelength_m[order], np.asarray(values, dtype=float)


def _sweep_port_expansion(port_name, result_name="expansion for port monitor"):
    attempts = []
    for path in ("FDTD::ports::" + port_name, "::model::FDTD::ports::" + port_name):
        for candidate_name in tuple(dict.fromkeys((result_name, "expansion for port monitor"))):
            try:
                return fdtd.getresult(path, candidate_name)
            except Exception as exc:
                attempts.append("%s/%s: %s" % (path, candidate_name, str(exc)[:120]))
    raise SweepResultSchemaError("No port expansion for %s. %s" % (port_name, " | ".join(attempts)))


def _extract_sweep_result():
    arrays = {}
    if GRATING_ANALYSIS:
        fiber_input_name = str(GRATING_ANALYSIS["fiber_input_power_monitor_name"])
        fiber_input_sign = float(GRATING_ANALYSIS.get("fiber_input_power_sign", -1.0))
        waveguide_port_name = str(GRATING_ANALYSIS["waveguide_port_name"])
        waveguide_port_result_name = str(GRATING_ANALYSIS.get(
            "waveguide_port_expansion_result_name", "expansion for port monitor"
        ))
        waveguide_modal_key = str(GRATING_ANALYSIS.get(
            "waveguide_port_modal_direction", "T_out"
        ))
        waveguide_power_name = str(GRATING_ANALYSIS["waveguide_power_monitor_name"])
        waveguide_total_sign = float(
            GRATING_ANALYSIS.get("waveguide_total_power_sign", -1.0)
        )
        waveguide_modal_sign = float(
            GRATING_ANALYSIS.get("waveguide_port_modal_sign", waveguide_total_sign)
        )

        def monitor_transmission(name):
            attempts = []
            for path in tuple(dict.fromkeys((name, "::model::" + name.lstrip(":")))):
                try:
                    return fdtd.getresult(path, "T")
                except Exception as exc:
                    attempts.append("%s: %s" % (path, str(exc)[:140]))
            raise SweepResultSchemaError(
                "Power monitor %r has no readable T result. %s"
                % (name, " | ".join(attempts))
            )

        try:
            fiber_input_data = monitor_transmission(fiber_input_name)
            fiber_wavelength_m, fiber_input_signed = _sweep_spectrum(
                fiber_input_data, "T", magnitude=False
            )
        except Exception as exc:
            raise SweepResultSchemaError(
                "The fiber input-power monitor %r has no readable signed T spectrum: %s"
                % (fiber_input_name, exc)
            ) from exc

        expansion_data = _sweep_port_expansion(
            waveguide_port_name, waveguide_port_result_name
        )
        receiver_selection = dict(
            SWEEP_PORT_MODE_SELECTIONS.get(waveguide_port_name, {})
        )
        receiver_mode_number = max(
            0, int(receiver_selection.get("mode number", 0))
        )
        receiver_mode_order = list(receiver_selection.get(
            "selected mode order",
            [receiver_mode_number] if receiver_mode_number else [],
        ))
        try:
            wavelength_m, mode_power = _sweep_spectrum(
                expansion_data,
                waveguide_modal_key,
                magnitude=False,
                selected_mode_number=receiver_mode_number,
                selected_mode_order=receiver_mode_order,
            )
        except Exception as exc:
            raise SweepResultSchemaError(
                "The passive waveguide receiver %r has no readable %r spectrum in %r: %s"
                % (
                    waveguide_port_name,
                    waveguide_modal_key,
                    waveguide_port_result_name,
                    exc,
                )
            ) from exc

        try:
            waveguide_power_data = monitor_transmission(waveguide_power_name)
            power_wavelength_m, waveguide_total_signed = _sweep_spectrum(
                waveguide_power_data, "T", magnitude=False
            )
        except Exception as exc:
            raise SweepResultSchemaError(
                "The waveguide total-power monitor %r has no readable signed T spectrum: %s"
                % (waveguide_power_name, exc)
            ) from exc

        fiber_input_signed = np.interp(
            wavelength_m, fiber_wavelength_m, fiber_input_signed
        )
        fiber_input_power = fiber_input_sign * fiber_input_signed
        waveguide_total_signed = np.interp(
            wavelength_m, power_wavelength_m, waveguide_total_signed
        )
        waveguide_total_power = waveguide_total_sign * waveguide_total_signed
        waveguide_mode_signed_raw = np.real(np.asarray(mode_power, dtype=float))
        waveguide_mode_power = waveguide_modal_sign * waveguide_mode_signed_raw
        normalization_floor = 1e-15
        if (
            not np.all(np.isfinite(fiber_input_power))
            or float(np.min(fiber_input_power)) <= normalization_floor
        ):
            raise SweepResultSchemaError(
                "The fiber input monitor %r has wrong/near-zero signed power after "
                "applying sign %.0f: range [%.6g, %.6g]"
                % (
                    fiber_input_name,
                    fiber_input_sign,
                    float(np.nanmin(fiber_input_power)),
                    float(np.nanmax(fiber_input_power)),
                )
            )
        if float(np.nanmin(waveguide_mode_power)) < -1e-9:
            raise SweepResultSchemaError(
                "The passive waveguide receiver %r has the wrong %s propagation sign "
                "after applying %.0f: %.6g"
                % (
                    waveguide_port_name,
                    waveguide_modal_key,
                    waveguide_modal_sign,
                    float(np.nanmin(waveguide_mode_power)),
                )
            )
        if float(np.nanmin(waveguide_total_power)) < -1e-9:
            raise SweepResultSchemaError(
                "The waveguide total-power monitor %r has the wrong propagation sign: %.6g"
                % (waveguide_power_name, float(np.nanmin(waveguide_total_power)))
            )
        coupling_efficiency = waveguide_mode_power / np.maximum(
            fiber_input_power, normalization_floor
        )
        waveguide_total_transmission = waveguide_total_power / np.maximum(
            fiber_input_power, normalization_floor
        )
        if not np.all(np.isfinite(coupling_efficiency)) or float(np.max(coupling_efficiency)) > 1.05:
            raise SweepResultSchemaError(
                "Unphysical grating selected-mode/input ratio: maximum %.6g"
                % float(np.nanmax(coupling_efficiency))
            )
        arrays.update(
            {
                "coupling_efficiency": coupling_efficiency,
                "fiber_input_power_signed_raw": fiber_input_signed,
                "fiber_input_power": fiber_input_power,
                "waveguide_mode_power": waveguide_mode_power,
                "waveguide_mode_power_signed_raw": waveguide_mode_signed_raw,
                "waveguide_total_power_signed_raw": waveguide_total_signed,
                "waveguide_total_power": waveguide_total_power,
                "waveguide_total_transmission": waveguide_total_transmission,
            }
        )
        return "coupling_efficiency", wavelength_m, coupling_efficiency, arrays

    if MMI_ANALYSIS:
        output_names = list(map(str, MMI_ANALYSIS["output_port_names"]))
        reference_name = str(MMI_ANALYSIS["input_reference_monitor_name"])
        if len(output_names) != 2:
            raise SweepResultSchemaError(
                "A 1x2 MMI sweep requires exactly two output FDTD ports; received %r"
                % output_names
            )
        try:
            reference_data = fdtd.getresult("::model::" + reference_name, "T")
            wavelength_m, input_power = _sweep_spectrum(reference_data, "T")
        except Exception as exc:
            raise SweepResultSchemaError(
                "The MMI input-reference power monitor %r has no readable T spectrum: %s"
                % (reference_name, exc)
            ) from exc
        outputs = []
        for output_name in output_names:
            try:
                output_data = fdtd.getresult("::model::FDTD::ports::" + output_name, "T")
                output_wavelength_m, output_power = _sweep_spectrum(output_data, "T")
            except Exception as exc:
                raise SweepResultSchemaError(
                    "MMI output port %r has no readable modal T spectrum: %s"
                    % (output_name, exc)
                ) from exc
            outputs.append(np.interp(wavelength_m, output_wavelength_m, output_power))
        total_output = outputs[0] + outputs[1]
        output_1_ratio = outputs[0] / np.maximum(total_output, 1e-15)
        output_2_ratio = outputs[1] / np.maximum(total_output, 1e-15)
        safe_input = np.maximum(input_power, 1e-15)
        output_1_over_input = outputs[0] / safe_input
        output_2_over_input = outputs[1] / safe_input
        total_over_input = total_output / np.maximum(input_power, 1e-15)
        arrays.update(
            {
                "output_1_power": outputs[0],
                "output_2_power": outputs[1],
                "input_power": input_power,
                "output_1_ratio": output_1_ratio,
                "output_2_ratio": output_2_ratio,
                "output_1_over_input": output_1_over_input,
                "output_2_over_input": output_2_over_input,
                "total_output_over_input": total_over_input,
            }
        )
        # The requested MMI sweep objective is one branch's transmitted modal
        # power divided by the measured input power.  Keep the lower branch,
        # total throughput, and 50/50 fractions alongside it for diagnosis.
        return "output_1_over_input", wavelength_m, output_1_over_input, arrays

    port_candidates = []
    monitor_candidates = []
    for port in PORTS:
        name = str(port.get("name", ""))
        try:
            data = fdtd.getresult("::model::FDTD::ports::" + name, "T")
            wavelength_m, response = _sweep_spectrum(data, "T")
            port_candidates.append(("port_" + _sweep_safe_key(name), wavelength_m, response))
        except Exception:
            pass
    for monitor in MONITORS:
        if str(monitor.get("monitor_kind", "Power monitor")) != "Power monitor":
            continue
        name = str(monitor.get("name", ""))
        try:
            data = fdtd.getresult("::model::" + name, "T")
            wavelength_m, response = _sweep_spectrum(data, "T")
            monitor_candidates.append(("monitor_" + _sweep_safe_key(name), wavelength_m, response))
        except Exception:
            pass
    candidates = port_candidates + monitor_candidates
    if not candidates:
        raise RuntimeError("No readable port or power-monitor transmission spectrum was found for this sweep point")
    # Prefer an explicitly placed power monitor. Otherwise the last FDTD port
    # is normally the receiver while the first/lowest-order port is the source.
    primary_name, wavelength_m, primary_response = (
        monitor_candidates[0] if monitor_candidates else port_candidates[-1]
    )
    for name, candidate_wavelength_m, response in candidates:
        arrays[name] = np.interp(wavelength_m, candidate_wavelength_m, response)
    return primary_name, wavelength_m, primary_response, arrays


def _sweep_case_npz(case_index):
    return os.path.join(SWEEP_CHECKPOINT_DIR, "case_%04d.npz" % int(case_index))


def _sweep_expected_primary_name():
    if GRATING_ANALYSIS:
        return "coupling_efficiency"
    if MMI_ANALYSIS:
        return "output_1_over_input"
    return None


def _sweep_case_is_complete(case_index):
    """Validate a checkpoint before reusing it, including parallel NFS runs."""
    case_path = _sweep_case_npz(case_index)
    if not os.path.isfile(case_path) or os.path.getsize(case_path) <= 0:
        return False
    try:
        with np.load(case_path, allow_pickle=False) as data:
            required = {
                "checkpoint_schema", "runtime_version", "code_fingerprint",
                "sweep_hash", "case_index",
                "wavelength_m", "primary_response", "primary_name",
            }
            expected_primary = _sweep_expected_primary_name()
            if expected_primary == "coupling_efficiency":
                required.update({
                    "fiber_input_power",
                    "waveguide_mode_power",
                    "waveguide_total_power",
                    "waveguide_total_transmission",
                })
            if expected_primary == "output_1_over_input":
                required.update({
                    "output_1_power", "output_2_power", "input_power",
                    "output_1_ratio", "output_2_ratio",
                    "output_1_over_input", "output_2_over_input",
                    "total_output_over_input",
                })
            if not required.issubset(data.files):
                return False
            stored_schema = int(np.asarray(data["checkpoint_schema"]).ravel()[0])
            stored_runtime = str(np.asarray(data["runtime_version"]).ravel()[0])
            stored_code_fingerprint = str(
                np.asarray(data["code_fingerprint"]).ravel()[0]
            )
            stored_hash = str(np.asarray(data["sweep_hash"]).ravel()[0])
            stored_index = int(np.asarray(data["case_index"]).ravel()[0])
            wavelength_m = np.asarray(data["wavelength_m"], dtype=float).ravel()
            response = np.asarray(data["primary_response"], dtype=float).ravel()
            primary_name = str(np.asarray(data["primary_name"]).ravel()[0]).strip()
        return bool(
            stored_schema == int(SWEEP_CHECKPOINT_SCHEMA)
            and stored_runtime == str(SWEEP_RUNTIME_VERSION)
            and stored_code_fingerprint == str(SWEEP_CODE_FINGERPRINT)
            and stored_hash == str(SWEEP_HASH)
            and stored_index == int(case_index)
            and wavelength_m.size > 0
            and response.size == wavelength_m.size
            and np.all(np.isfinite(wavelength_m))
            and np.all(np.isfinite(response))
            and (wavelength_m.size == 1 or np.all(np.diff(wavelength_m) > 0.0))
            and bool(primary_name)
            and (expected_primary is None or primary_name == expected_primary)
        )
    except Exception:
        return False


def _save_sweep_case(case_index, primary_name, wavelength_m, primary_response, arrays):
    payload = {
        "checkpoint_schema": np.asarray([int(SWEEP_CHECKPOINT_SCHEMA)], dtype=int),
        "runtime_version": np.asarray([str(SWEEP_RUNTIME_VERSION)]),
        "code_fingerprint": np.asarray([str(SWEEP_CODE_FINGERPRINT)]),
        "sweep_hash": np.asarray([str(SWEEP_HASH)]),
        "case_index": np.asarray([int(case_index)], dtype=int),
        "wavelength_m": np.asarray(wavelength_m, dtype=float),
        "primary_response": np.asarray(primary_response, dtype=float),
        "primary_name": np.asarray([str(primary_name)]),
    }
    for key, values in arrays.items():
        payload[_sweep_safe_key(key)] = np.asarray(values)
    final_path = _sweep_case_npz(case_index)
    worker_token = "%s-%d-%s" % (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", socket.gethostname()),
        os.getpid(),
        uuid.uuid4().hex,
    )
    temporary_path = final_path + ".worker-%s-%d.tmp" % (worker_token, int(case_index))
    try:
        with open(temporary_path, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
    finally:
        if os.path.isfile(temporary_path):
            os.remove(temporary_path)
    if not _sweep_case_is_complete(case_index):
        raise RuntimeError("Sweep checkpoint failed validation: " + final_path)
    print("Checkpointed compact sweep result:", final_path)


def _finalize_sweep_results(failures):
    reference_wavelength_m = None
    response_rows = []
    success = []
    primary_name = "response"
    grating_detail_names = (
        "fiber_input_power_signed_raw",
        "fiber_input_power",
        "waveguide_mode_power",
        "waveguide_total_power_signed_raw",
        "waveguide_total_power",
        "waveguide_total_transmission",
    ) if any(case.get("grating_analysis") for case in SWEEP_CASES) else ()
    mmi_detail_names = (
        "output_1_power",
        "output_2_power",
        "input_power",
        "output_1_ratio",
        "output_2_ratio",
        "output_1_over_input",
        "output_2_over_input",
        "total_output_over_input",
    ) if any(case.get("mmi_analysis") for case in SWEEP_CASES) else ()
    detail_names = (*grating_detail_names, *mmi_detail_names)
    detail_rows = {name: [] for name in detail_names}
    for case_index in range(len(SWEEP_CASES)):
        case_path = _sweep_case_npz(case_index)
        if not _sweep_case_is_complete(case_index):
            response_rows.append(None)
            for name in detail_names:
                detail_rows[name].append(None)
            success.append(False)
            continue
        with np.load(case_path) as data:
            wavelength_m = np.asarray(data["wavelength_m"], dtype=float)
            response = np.asarray(data["primary_response"], dtype=float)
            primary_name = str(np.asarray(data["primary_name"]).ravel()[0])
            detail_values = {
                name: np.asarray(data[name], dtype=float)
                if name in data.files else None
                for name in detail_names
            }
        if reference_wavelength_m is None:
            reference_wavelength_m = wavelength_m
        response_rows.append(np.interp(reference_wavelength_m, wavelength_m, response))
        for name in detail_names:
            values = detail_values[name]
            detail_rows[name].append(
                None if values is None
                else np.interp(reference_wavelength_m, wavelength_m, values)
            )
        success.append(True)
    if reference_wavelength_m is None:
        raise RuntimeError("Every Lumerical sweep point failed; no result spectrum is available")
    response_matrix = np.full((len(SWEEP_CASES), reference_wavelength_m.size), np.nan, dtype=float)
    for index, row in enumerate(response_rows):
        if row is not None:
            response_matrix[index, :] = row
    detail_matrices = {}
    for name in detail_names:
        matrix = np.full(
            (len(SWEEP_CASES), reference_wavelength_m.size), np.nan, dtype=float
        )
        for index, row in enumerate(detail_rows[name]):
            if row is not None:
                matrix[index, :] = row
        detail_matrices[name] = matrix
    parameter_names = [str(axis["parameter"]) for axis in SWEEP_SPEC["axes"]]
    parameter_codes = [str(axis["short_name"]) for axis in SWEEP_SPEC["axes"]]
    parameter_matrix = np.asarray(
        [[float(case["values"][name]) for name in parameter_names] for case in SWEEP_CASES],
        dtype=float,
    )
    maximum_response = np.asarray([
        float(np.nanmax(row)) if np.any(np.isfinite(row)) else np.nan
        for row in response_matrix
    ])
    target_wavelength_m = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    ) * 1e-6
    target_index = int(np.argmin(np.abs(reference_wavelength_m - target_wavelength_m)))
    target_response = response_matrix[:, target_index]
    best_index = int(np.nanargmax(maximum_response))
    target_best_index = int(np.nanargmax(target_response))

    global SWEEP_RESULTS_NPZ, SWEEP_RESULTS_JSON, SWEEP_TEXT_SUMMARY
    results_root = str(globals().get("SWEEP_RESULTS_ROOT", REMOTE_WORK))
    results_basename = str(globals().get("SWEEP_RESULTS_BASENAME", "lumerical_sweep_results"))
    os.makedirs(results_root, exist_ok=True)
    SWEEP_RESULTS_NPZ = os.path.join(results_root, results_basename + ".npz")
    SWEEP_RESULTS_JSON = os.path.join(results_root, results_basename + ".json")
    SWEEP_TEXT_SUMMARY = os.path.join(results_root, "summary.txt")
    np.savez_compressed(
        SWEEP_RESULTS_NPZ,
        wavelength_m=reference_wavelength_m,
        response=response_matrix,
        maximum_response=maximum_response,
        target_response=target_response,
        target_wavelength_m=np.asarray([target_wavelength_m]),
        parameter_values=parameter_matrix,
        parameter_names=np.asarray(parameter_names),
        parameter_codes=np.asarray(parameter_codes),
        case_labels=np.asarray([str(case["display_label"]) for case in SWEEP_CASES]),
        result_stems=np.asarray([str(case["result_stem"]) for case in SWEEP_CASES]),
        success=np.asarray(success, dtype=bool),
        best_index=np.asarray([best_index], dtype=int),
        target_best_index=np.asarray([target_best_index], dtype=int),
        primary_name=np.asarray([primary_name]),
        **detail_matrices,
    )
    manifest = {
        "sweep_hash": SWEEP_HASH,
        "sweep_code_fingerprint": str(SWEEP_CODE_FINGERPRINT),
        "checkpoint_schema": int(SWEEP_CHECKPOINT_SCHEMA),
        "runtime_version": str(SWEEP_RUNTIME_VERSION),
        "component_uid": int(SWEEP_SPEC["component_uid"]),
        "component_kind": str(SWEEP_SPEC["component_kind"]),
        "axes": SWEEP_SPEC["axes"],
        "point_count": len(SWEEP_CASES),
        "completed_count": int(sum(success)),
        "failed_count": int(len(SWEEP_CASES) - sum(success)),
        "best_index": best_index,
        "best_values": SWEEP_CASES[best_index]["values"],
        "best_maximum_response": float(maximum_response[best_index]),
        "target_best_index": target_best_index,
        "target_best_values": SWEEP_CASES[target_best_index]["values"],
        "target_best_response": float(target_response[target_best_index]),
        "target_wavelength_m": float(reference_wavelength_m[target_index]),
        "primary_name": primary_name,
        "failures": failures,
        "per_case_fsp_saved": False,
        "best_fsp": None,
        "execution": str(globals().get("SWEEP_EXECUTION_MODE", "single-session-sequential")),
        "text_summary": SWEEP_TEXT_SUMMARY,
    }

    def _sweep_summary_number(value, digits=8):
        try:
            number = float(np.asarray(value).ravel()[0])
        except Exception:
            return str(value)
        return ("%.*g" % (int(digits), number)) if np.isfinite(number) else str(number)

    def _sweep_summary_json(value):
        def _reported(item):
            if isinstance(item, dict):
                return {
                    str(key): _reported(child)
                    for key, child in item.items()
                    if str(key) != "waveguide_effective_index"
                }
            if isinstance(item, (list, tuple)):
                return [_reported(child) for child in item]
            return item
        return json.dumps(_reported(value), sort_keys=True, separators=(",", ":"), default=str)

    def _sweep_summary_section(lines, title):
        if lines:
            lines.append("")
        lines.append(str(title))
        lines.append("-" * len(str(title)))

    _sweep_parameter_details = {
        "pitch": ("Pitch", "um"),
        "fill_factor": ("Fill factor", ""),
        "duty_cycle": ("Duty cycle", ""),
        "fill_factors": ("Apodized fill factors", ""),
        "tooth_shape": ("Tooth geometry", ""),
        "N": ("Number of grating periods", ""),
        "target_length": ("Target grating length", "um"),
        "h_total": ("Device-layer thickness", "um"),
        "etch_depth": ("Etch depth", "um"),
        "alpha_t": ("Aperture angle", "deg"),
        "taper_L": ("Taper length", "um"),
        "radius": ("Focusing radius", "um"),
        "y_span": ("Grating Y span", "um"),
        "L_extra": ("Thick-end extension", "um"),
        "wg_width": ("Waveguide width", "um"),
        "wg_length": ("Waveguide length", "um"),
        "taper_exponent": ("Grating taper exponent", ""),
        "mmi_width": ("MMI width", "um"),
        "mmi_length": ("MMI length", "um"),
        "taper_width": ("MMI taper width", "um"),
        "input_taper_length": ("Input taper length", "um"),
        "output_taper_length": ("Output taper length", "um"),
        "input_length": ("Input access length", "um"),
        "output_length": ("Output access length", "um"),
        "port_sep": ("Output branch separation", "um"),
        "taper_power": ("MMI taper profile exponent", ""),
        "taper_points": ("MMI taper discretization points", ""),
        "input_reference_before_taper_um": ("Input power-reference distance before taper", "um"),
        "fdtd_port_clearance_um": ("MMI access-port clearance from waveguide end", "um"),
        "fiber_offset": ("Fiber offset", "um"),
        "angle_theta": ("Fiber angle theta", "deg"),
        "fiber_tox_offset_um": ("Fiber bottom offset above SiO2 cladding", "um"),
        "fiber_core_diameter_um": ("Fiber core diameter", "um"),
        "fiber_core_index": ("Fiber core refractive index", ""),
        "fiber_cladding_diameter_um": ("Fiber cladding diameter", "um"),
        "fiber_cladding_index": ("Fiber cladding refractive index", ""),
        "fiber_length_um": ("Fiber length", "um"),
        "fiber_power_monitor_below_source_um": ("Horizontal fiber-input monitor distance below source", "um"),
        "gaussian_input_monitor_span_scale": ("Gaussian input-power monitor span scale", "x"),
        "fdtd_port_offset_from_waveguide_end_um": ("Waveguide FDTD-port offset from waveguide end", "um"),
        "waveguide_monitor_span_um": ("Waveguide receiver-port transverse span", "um"),
        "waveguide_total_power_before_mode_um": ("Total-power plane distance before receiver port", "um"),
        "waveguide_neff_tolerance": ("Waveguide effective-index tolerance", ""),
        "waveguide_mode_search_count": ("Waveguide eigensolver modes searched", ""),
        "tolerance": ("Geometry build tolerance", "um"),
    }

    def _sweep_parameter_value(value):
        if isinstance(value, (dict, list, tuple)):
            return _sweep_summary_json(value)
        if isinstance(value, str):
            return value
        return _sweep_summary_number(value)

    def _append_sweep_major_parameters(lines, parameters, prefix="- "):
        parameters = dict(parameters or {})
        found = 0
        shown = set()
        for key, (label, unit) in _sweep_parameter_details.items():
            if key not in parameters or parameters[key] in (None, ""):
                continue
            suffix = (" " + unit) if unit else ""
            lines.append("%s%s: %s%s" % (
                prefix, label, _sweep_parameter_value(parameters[key]), suffix
            ))
            found += 1
            shown.add(key)
        ignored = {"name", "layer", "datatype", "waveguide_effective_index"}
        for key, value in parameters.items():
            if (
                key in shown or key in ignored or key.endswith("_layer")
                or key.endswith("_datatype") or value in (None, "")
            ):
                continue
            if isinstance(value, (bool, int, float, str, list, tuple, dict)):
                lines.append(
                    "%s%s: %s"
                    % (prefix, str(key).replace("_", " ").title(), _sweep_parameter_value(value))
                )
                found += 1

    best_peak_index = int(np.nanargmax(response_matrix[best_index]))
    best_peak_wavelength_m = float(reference_wavelength_m[best_peak_index])
    target_source_component = next(
        (
            item for item in globals().get("SOURCE_COMPONENTS_JSON", [])
            if int(item.get("uid", -1)) == int(SWEEP_SPEC["component_uid"])
        ),
        {},
    )
    target_component_name = str(
        target_source_component.get("name")
        or target_source_component.get("params", {}).get("name")
        or SWEEP_SPEC["component_kind"]
    )
    summary_lines = ["MAX LAYOUT — LUMERICAL PARAMETER SWEEP SUMMARY"]
    _sweep_summary_section(summary_lines, "PROJECT")
    summary_lines.extend([
        "Component: name=%s | kind=%s | UID=%d" % (
            target_component_name, str(SWEEP_SPEC["component_kind"]),
            int(SWEEP_SPEC["component_uid"])
        ),
        "Execution: %s | Cartesian points=%d | completed=%d | failed=%d"
        % (
            manifest["execution"], len(SWEEP_CASES), int(sum(success)),
            int(len(SWEEP_CASES) - sum(success)),
        ),
        "Run status: sweep completed and combined result artifact written",
    ])

    nominal_parameters = dict(globals().get("SWEEP_NOMINAL_PARAMETERS", {}))
    _sweep_summary_section(summary_lines, "PARAMETERS")
    summary_lines.append("Nominal major device parameters (lengths in um; angles in deg):")
    _append_sweep_major_parameters(summary_lines, nominal_parameters)
    summary_lines.append("Exact nominal source parameters (JSON): %s" % _sweep_summary_json(nominal_parameters))

    _sweep_summary_section(summary_lines, "SWEEP DEFINITION")
    summary_lines.append("Cartesian axes and values:")
    for axis in SWEEP_SPEC["axes"]:
        values = [float(value) for value in axis.get("values", [])]
        parameter = str(axis.get("parameter", "parameter"))
        label, unit = _sweep_parameter_details.get(
            parameter, (parameter.replace("_", " ").title(), "")
        )
        step_text = "n/a"
        if len(values) >= 2:
            differences = np.diff(np.asarray(values, dtype=float))
            step_text = (
                _sweep_summary_number(differences[0])
                if np.allclose(differences, differences[0], rtol=1e-10, atol=1e-12)
                else "custom/nonuniform"
            )
        suffix = (" " + unit) if unit else ""
        summary_lines.append(
            "- %s [%s]: start=%s%s | stop=%s%s | step=%s%s | points=%d | values=%s"
            % (
                label,
                str(axis.get("short_name", "")),
                _sweep_summary_number(values[0]) if values else "n/a",
                suffix if values else "",
                _sweep_summary_number(values[-1]) if values else "n/a",
                suffix if values else "",
                step_text,
                suffix if step_text not in {"n/a", "custom/nonuniform"} else "",
                len(values),
                _sweep_summary_json(values),
            )
        )

    _sweep_summary_section(summary_lines, "MATERIAL STACK AND MESH")
    summary_lines.append("Bottom-to-top layer order; mesh factor means factor x lambda0 / maximum material index.")
    for index, row in enumerate(globals().get("MATERIAL_STACK", []), start=1):
        summary_lines.append(
            "- %02d %s | material=%s | thickness=%s um | etch=%s um | sidewall=%s deg | mesh_factor=%s | mesh_order=%s | role=%s | conformal=%s | slab_extent=%s | GDS_layers=%s"
            % (
                index,
                str(row.get("name", "layer")),
                str(row.get("material", "")),
                _sweep_summary_number(row.get("thickness_um", 0.0)),
                _sweep_summary_number(row.get("etch_depth_um", 0.0)),
                _sweep_summary_number(row.get("sidewall_angle_deg", 90.0)),
                _sweep_summary_number(row.get("mesh_factor", 0.2)),
                str(row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2)),
                str(row.get("role", "background")),
                str(bool(row.get("conformal", False))),
                str(row.get("slab_extent", "full FDTD plane")),
                _sweep_summary_json(row.get("gds_layers", [])),
            )
        )

    domain_um = []
    for property_name in ("x min", "x max", "y min", "y max", "z min", "z max"):
        try:
            domain_um.append(float(np.asarray(fdtd.getnamed("FDTD", property_name)).ravel()[0]) / UM)
        except Exception:
            domain_um.append(None)
    _sweep_summary_section(summary_lines, "SIMULATION SETTINGS")
    summary_lines.extend([
        "- Solver: 3D FDTD",
        "- Wavelength sweep: %s to %s um | %d points"
        % (
            _sweep_summary_number(SETTINGS.get("wavelength_start_um")),
            _sweep_summary_number(SETTINGS.get("wavelength_stop_um")),
            int(SETTINGS.get("frequency_points", 0)),
        ),
        "- Actual domain [xmin,xmax,ymin,ymax,zmin,zmax]: %s um" % _sweep_summary_json(domain_um),
        "- Union layout bounds: %s um | configured padding=%s um"
        % (
            _sweep_summary_json(globals().get("BOUNDING_BOX_UM", [])),
            _sweep_summary_json(SETTINGS.get("domain_padding_um", {})),
        ),
        "- Resources: solve=%s | model-build CPU threads=%s | CPU post-processing=True"
        % (
            str(SETTINGS.get("resource_mode", "GPU")),
            str(SETTINGS.get("build_cpu_threads", 30)),
        ),
        "- Reproducibility: checkpoint schema=%d | runtime=%s | code fingerprint=%s"
        % (
            int(SWEEP_CHECKPOINT_SCHEMA), str(SWEEP_RUNTIME_VERSION),
            str(SWEEP_CODE_FINGERPRINT),
        ),
        "- Numerical controls: mesh accuracy=%s | dt factor=%s | PML=%s | geometry/PML overlap=%s um"
        % (
            str(SETTINGS.get("mesh_accuracy", 2)),
            _sweep_summary_number(SETTINGS.get("dt_stability_factor", 0.99)),
            str(SETTINGS.get("pml_profile", "Standard")),
            _sweep_summary_number(SETTINGS.get("pml_geometry_overlap_um", 1.0)),
        ),
        "- Time controls: maximum=%s ps (%s fs) | auto shutoff=%s"
        % (
            _sweep_summary_number(float(SETTINGS.get("simulation_time_fs", 10000.0)) / 1000.0),
            _sweep_summary_number(SETTINGS.get("simulation_time_fs", 10000.0)),
            _sweep_summary_number(SETTINGS.get("auto_shutoff_min", 1e-6)),
        ),
        "- TFLN material model: crystal cut=%s | temperature=%s K"
        % (
            str(SETTINGS.get("tfln_crystal_cut", "X")),
            _sweep_summary_number(SETTINGS.get("tfln_temperature_K", 296.3)),
        ),
    ])
    waveguide_index_estimate = dict(
        globals().get("WAVEGUIDE_INDEX_ESTIMATE", {})
    )
    if waveguide_index_estimate:
        summary_lines.append(
            "- Automatic waveguide mode target: core n=%s | adjacent dielectric n=%s | "
            "midpoint neff=%s at %s um | core=%s | surroundings=%s"
            % (
                _sweep_summary_number(waveguide_index_estimate.get("core_index")),
                _sweep_summary_number(waveguide_index_estimate.get("surrounding_index")),
                _sweep_summary_number(waveguide_index_estimate.get("target_neff")),
                _sweep_summary_number(waveguide_index_estimate.get("wavelength_um")),
                _sweep_summary_json(waveguide_index_estimate.get("core_materials", [])),
                _sweep_summary_json(waveguide_index_estimate.get("surrounding_materials", [])),
            )
        )

    _sweep_summary_section(summary_lines, "SOURCES / PORTS / MONITORS")
    summary_lines.append("Fiber geometries: %s" % _sweep_summary_json([
        {"name": item.get("name"), "center_um": item.get("center"),
         "theta_deg": item.get("angle theta", 0.0), "phi_deg": item.get("angle phi", 0.0),
         "core_diameter_um": item.get("core diameter_um"), "core_index": item.get("core index"),
         "cladding_diameter_um": item.get("cladding diameter_um"), "cladding_index": item.get("cladding index"),
         "length_um": item.get("length_um")}
        for item in globals().get("FIBER_GEOMETRIES", [])
    ]))
    summary_lines.append("FDTD ports: %s" % _sweep_summary_json([
        {"name": item.get("name"), "normal": item.get("plane normal"),
         "center_um": item.get("center"),
         "spans_um": [item.get("x span"), item.get("y span"), item.get("z span", item.get("z_span_um"))],
         "theta_deg": item.get("angle theta", 0.0), "phi_deg": item.get("angle phi", 0.0),
         "mode": item.get("mode"), "mode_number": item.get("mode number"),
         "polarization": item.get("polarization"), "target_neff": item.get("target neff")}
        for item in globals().get("PORTS", [])
    ]))
    summary_lines.append("Monitors: %s" % _sweep_summary_json([
        {"name": item.get("name"), "kind": item.get("monitor_kind"),
         "normal": item.get("plane normal"), "center_um": item.get("center"),
         "spans_um": [item.get("x span"), item.get("y span"), item.get("z span")],
         "role": item.get("grating_monitor_role", item.get("parent_port_name")),
         "target_neff": item.get("target neff")}
        for item in globals().get("MONITORS", [])
    ]))

    _sweep_summary_section(summary_lines, "RESULTS SUMMARY")
    summary_lines.extend([
        "Peak-best selection criterion: maximum %s anywhere in the wavelength sweep." % primary_name,
        "Peak-best case: index=%d | label=%s | swept parameters=%s"
        % (best_index + 1, str(SWEEP_CASES[best_index].get("display_label", "")), _sweep_summary_json(manifest["best_values"])),
        "Peak-best major parameters:",
    ])
    _append_sweep_major_parameters(summary_lines, SWEEP_CASES[best_index].get("source_parameters", {}))
    summary_lines.extend([
        "Peak-best exact source parameters (JSON): %s" % _sweep_summary_json(
            SWEEP_CASES[best_index].get("source_parameters", {})
        ),
        "Peak-best result: peak=%s (%s%%) at %s nm | value at target %s nm=%s (%s%%)"
        % (
            _sweep_summary_number(maximum_response[best_index]),
            _sweep_summary_number(100.0 * maximum_response[best_index]),
            _sweep_summary_number(best_peak_wavelength_m * 1e9),
            _sweep_summary_number(reference_wavelength_m[target_index] * 1e9),
            _sweep_summary_number(target_response[best_index]),
            _sweep_summary_number(100.0 * target_response[best_index]),
        ),
        "Target-best case at %s nm: index=%d | label=%s | swept parameters=%s | value=%s (%s%%)"
        % (
            _sweep_summary_number(reference_wavelength_m[target_index] * 1e9),
            target_best_index + 1,
            str(SWEEP_CASES[target_best_index].get("display_label", "")),
            _sweep_summary_json(manifest["target_best_values"]),
            _sweep_summary_number(target_response[target_best_index]),
            _sweep_summary_number(100.0 * target_response[target_best_index]),
        ),
        "Target-best major parameters:",
    ])
    _append_sweep_major_parameters(
        summary_lines, SWEEP_CASES[target_best_index].get("source_parameters", {})
    )
    summary_lines.append(
        "Target-best exact source parameters (JSON): %s"
        % _sweep_summary_json(
            SWEEP_CASES[target_best_index].get("source_parameters", {})
        )
    )

    if primary_name == "coupling_efficiency" and detail_matrices.get(
        "waveguide_total_transmission"
    ) is not None:
        _grating_pin_peak = detail_matrices["fiber_input_power"][
            best_index, best_peak_index
        ]
        _grating_mode_peak = detail_matrices["waveguide_mode_power"][
            best_index, best_peak_index
        ]
        _grating_total_peak = detail_matrices["waveguide_total_power"][
            best_index, best_peak_index
        ]
        _grating_total_ratio_peak = detail_matrices[
            "waveguide_total_transmission"
        ][best_index, best_peak_index]
        summary_lines.append(
            "Grating peak-best measurement: measured Pin=%s | selected TE output=%s | "
            "total waveguide output=%s | selected TE/Pin=%s | total/Pin=%s"
            % (
                _sweep_summary_number(_grating_pin_peak),
                _sweep_summary_number(_grating_mode_peak),
                _sweep_summary_number(_grating_total_peak),
                _sweep_summary_number(maximum_response[best_index]),
                _sweep_summary_number(_grating_total_ratio_peak),
            )
        )

    if primary_name == "output_1_over_input" and detail_matrices.get("output_2_over_input") is not None:
        _mmi_upper_over_input = detail_matrices["output_1_over_input"][best_index, target_index]
        _mmi_lower_over_input = detail_matrices["output_2_over_input"][best_index, target_index]
        _mmi_total_over_input = detail_matrices["total_output_over_input"][best_index, target_index]
        _mmi_upper_split = detail_matrices["output_1_ratio"][best_index, target_index]
        _mmi_lower_split = detail_matrices["output_2_ratio"][best_index, target_index]
        _mmi_symmetry_error = abs(_mmi_upper_split - 0.5) * 100.0
        summary_lines.append(
            "MMI peak-best case at target: upper/Pin=%s | lower/Pin=%s | total/Pin=%s | upper split=%s | lower split=%s | symmetry error=%s percentage points"
            % (
                _sweep_summary_number(_mmi_upper_over_input),
                _sweep_summary_number(_mmi_lower_over_input),
                _sweep_summary_number(_mmi_total_over_input),
                _sweep_summary_number(_mmi_upper_split),
                _sweep_summary_number(_mmi_lower_split),
                _sweep_summary_number(_mmi_symmetry_error),
            )
        )

    _sweep_summary_section(summary_lines, "WARNINGS / NOTES")
    if failures:
        summary_lines.append("Failed sweep points:")
        for failure in failures:
            summary_lines.append("- index=%s | values=%s | error=%s" % (
                str(int(failure.get("index", -1)) + 1),
                _sweep_summary_json(failure.get("values", {})),
                str(failure.get("error", "unknown error")),
            ))
    else:
        summary_lines.append("- No sweep points failed.")

    _sweep_summary_section(summary_lines, "FSP PROVENANCE")
    summary_lines.append(
        "per_case_fsp_saved=false. After the compact sweep completes, only the peak-best geometry is solved once more and saved as the required winning FSP."
    )
    summary_lines.append(
        "Parity note: every sweep point uses the shared union FDTD domain. Compare a sweep spectrum with a standalone run only when geometry, material stack, mesh factors, wavelength sampling, ports, and FDTD bounds are all identical."
    )
    with open(SWEEP_TEXT_SUMMARY, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines).rstrip() + "\n")
    with open(SWEEP_RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Saved combined sweep results:", SWEEP_RESULTS_NPZ)
    print("Saved human-readable sweep summary:", SWEEP_TEXT_SUMMARY)
    print("Best sweep point:", manifest["best_values"], "maximum", manifest["best_maximum_response"])
    return best_index
'''


_SWEEP_RUNNER_REMOTE = r'''# Run every geometry point in one persistent GPU session.
resource_mode = str(SETTINGS.get("resource_mode", "GPU")).strip().upper()
if resource_mode not in {"GPU", "CPU"}:
    raise ValueError("resource_mode must be GPU or CPU")
failures = []
schema_failure = None
completed_count = 0
failed_count = 0
_SWEEP_PROGRESS_SEQUENCE = 0
_SWEEP_PROGRESS_STARTED = time.monotonic()
try:
    if os.path.isfile(SWEEP_PROGRESS_FILE):
        os.remove(SWEEP_PROGRESS_FILE)
except Exception as progress_reset_exc:
    print("Sweep progress reset warning:", str(progress_reset_exc)[:240])
_emit_sweep_progress(
    "started", completed_count=completed_count, failed_count=failed_count
)
for case_index, case in enumerate(SWEEP_CASES):
    case_path = _sweep_case_npz(case_index)
    if _sweep_case_is_complete(case_index):
        completed_count += 1
        reused_name, reused_wavelength_m, reused_response = _sweep_checkpoint_spectrum(
            case_index
        )
        _emit_sweep_progress(
            "reused", case_index=case_index, completed_count=completed_count,
            failed_count=failed_count, case_seconds=0.0,
            primary_name=reused_name, wavelength_m=reused_wavelength_m,
            response=reused_response,
        )
        print("Reusing completed sweep checkpoint %d/%d: %s" % (
            case_index + 1, len(SWEEP_CASES), case["display_label"]
        ))
        continue
    case_started = time.monotonic()
    _emit_sweep_progress(
        "running", case_index=case_index, completed_count=completed_count,
        failed_count=failed_count,
    )
    try:
        _apply_sweep_case(case_index)
        fdtd.run("FDTD", resource_mode)
        primary_name, wavelength_m, primary_response, arrays = _extract_sweep_result()
        _save_sweep_case(case_index, primary_name, wavelength_m, primary_response, arrays)
        completed_count += 1
        _emit_sweep_progress(
            "completed", case_index=case_index, completed_count=completed_count,
            failed_count=failed_count,
            case_seconds=time.monotonic() - case_started,
            primary_name=primary_name, wavelength_m=wavelength_m,
            response=primary_response,
        )
        print("Completed sweep point %d/%d." % (case_index + 1, len(SWEEP_CASES)))
    except SweepResultSchemaError as exc:
        message = str(exc)
        failed_count += 1
        failures.append({"index": int(case_index), "values": case["values"], "error": message})
        _emit_sweep_progress(
            "failed", case_index=case_index, completed_count=completed_count,
            failed_count=failed_count,
            case_seconds=time.monotonic() - case_started, error=message,
        )
        print("SWEEP RESULT SCHEMA FAILED %d/%d; remaining solves were cancelled: %s" % (
            case_index + 1, len(SWEEP_CASES), message
        ))
        schema_failure = exc
        break
    except Exception as exc:
        message = str(exc)
        failed_count += 1
        failures.append({"index": int(case_index), "values": case["values"], "error": message})
        _emit_sweep_progress(
            "failed", case_index=case_index, completed_count=completed_count,
            failed_count=failed_count,
            case_seconds=time.monotonic() - case_started, error=message,
        )
        print("SWEEP POINT FAILED %d/%d: %s" % (case_index + 1, len(SWEEP_CASES), message))
best_sweep_index = None
if schema_failure is None:
    best_sweep_index = _finalize_sweep_results(failures)
    print("Preparing the required winning solved FSP from sweep point %d." % (best_sweep_index + 1))
    _apply_sweep_case(best_sweep_index)
    fdtd.run("FDTD", resource_mode)
    _best_project_name = os.path.basename(
        str(SETTINGS.get("project_file", "lumerical_sweep.fsp"))
    )
    if not _best_project_name.lower().endswith(".fsp"):
        _best_project_name += ".fsp"
    REMOTE_BEST_SWEEP_FSP = os.path.join(
        REMOTE_FSP_DIR, "best_" + _best_project_name
    )
    fdtd.save(REMOTE_BEST_SWEEP_FSP)
    if not os.path.isfile(REMOTE_BEST_SWEEP_FSP) or os.path.getsize(REMOTE_BEST_SWEEP_FSP) <= 0:
        raise RuntimeError("The required winning sweep FSP was not created")
    with open(SWEEP_RESULTS_JSON, "r", encoding="utf-8") as _manifest_stream:
        _saved_manifest = json.load(_manifest_stream)
    _saved_manifest["best_fsp"] = REMOTE_BEST_SWEEP_FSP
    with open(SWEEP_RESULTS_JSON, "w", encoding="utf-8") as _manifest_stream:
        json.dump(_saved_manifest, _manifest_stream, indent=2)
    with open(SWEEP_TEXT_SUMMARY, "a", encoding="utf-8") as _summary_stream:
        _summary_stream.write("\nWinning solved FSP\n------------------\n%s\n" % REMOTE_BEST_SWEEP_FSP)
    print("Saved the solved winning sweep design:", REMOTE_BEST_SWEEP_FSP)
_emit_sweep_progress(
    "finished" if schema_failure is None else "cancelled",
    completed_count=completed_count, failed_count=failed_count,
)

# All electromagnetic solves are complete. Keep solved numeric artifacts on
# disk and return the Lumerical resource table to the 30-thread CPU row before
# local plotting and summary generation.
analysis_threads = max(1, min(int(SETTINGS.get("build_cpu_threads", 30)), os.cpu_count() or 1))
try:
    fdtd.setresource("FDTD", 1, "active", False)
    fdtd.setresource("FDTD", 2, "device type", "CPU")
    fdtd.setresource("FDTD", 2, "active", True)
    fdtd.setresource("FDTD", 2, "processes", 1)
    fdtd.setresource("FDTD", 2, "threads", analysis_threads)
    print("All sweep solves complete; post-processing resource is CPU: 1 process x %d threads." % analysis_threads)
except Exception as exc:
    print("CPU post-processing resource switch warning:", str(exc)[:240])
_runtime_project = str(globals().get("REMOTE_RUNTIME_PROJECT_FILE", ""))
if _runtime_project and os.path.isfile(_runtime_project):
    try:
        os.remove(_runtime_project)
        print("Removed transient sweep runtime FSP after result extraction.")
    except Exception as exc:
        print("Transient sweep runtime cleanup warning:", str(exc)[:180])
if schema_failure is not None:
    raise RuntimeError(
        "A solved sweep point exposed an incompatible Lumerical result schema; "
        "the remaining sweep points were not run. " + str(schema_failure)
    ) from schema_failure
'''


_MULTIGPU_INVENTORY_AND_LICENSE_CELL = r'''# Validate launcher-provisioned A100 inventory without acquiring solver resources.
import datetime as _multigpu_datetime
import inspect
import json
from pathlib import Path as _MultiGpuPath

_lambda_parameters = inspect.signature(Lambda).parameters
if "host" not in _lambda_parameters or "key" not in _lambda_parameters:
    raise RuntimeError(
        "This sweep-multithread notebook needs the updated Piris Requirements/lambda_remote.py "
        "with Lambda(..., host=..., key=...). Update the Requirements folder and Piris 3D "
        "Launcher before running this notebook. The ordinary single-GPU sweep is unchanged."
    )
_required_helper_methods = ("run_once", "stop_work_processes")
_missing_helper_methods = [
    name for name in _required_helper_methods
    if not callable(getattr(Lambda, name, None)) or not callable(getattr(lam, name, None))
]
if _missing_helper_methods:
    raise RuntimeError(
        "This sweep-multithread notebook needs the updated Piris Requirements/lambda_remote.py "
        "with Lambda.run_once(...) and Lambda.stop_work_processes(...). Missing: %s. Update the "
        "Requirements folder and Piris 3D Launcher before running this notebook. No model was "
        "built and no GPU licences were checked out."
        % ", ".join(_missing_helper_methods)
    )
_run_once_parameters = inspect.signature(Lambda.run_once).parameters
_stop_work_parameters = inspect.signature(Lambda.stop_work_processes).parameters
if not {"code", "quiet", "timeout"}.issubset(_run_once_parameters) or "timeout" not in _stop_work_parameters:
    raise RuntimeError(
        "The installed lambda_remote.py has an incompatible recovery API. Update the Piris "
        "Requirements folder and 3D Launcher before building this multi-GPU sweep."
    )

_inventory_value = os.environ.get("PIRIS_LUMERICAL_INVENTORY", "").strip()
if not _inventory_value:
    raise RuntimeError(
        "PIRIS_LUMERICAL_INVENTORY is not set. Relaunch this project with the updated Piris 3D "
        "Launcher after preparing the exact A100 node count selected below. No GPU licences were checked out."
    )
MULTIGPU_INVENTORY_FILE = _MultiGpuPath(_inventory_value).expanduser().resolve()
if not MULTIGPU_INVENTORY_FILE.is_file():
    raise RuntimeError("The multi-GPU node inventory does not exist: " + str(MULTIGPU_INVENTORY_FILE))
with MULTIGPU_INVENTORY_FILE.open(encoding="utf-8") as _inventory_handle:
    _inventory = json.load(_inventory_handle)
if int(_inventory.get("version", 0)) < 1:
    raise RuntimeError("Unsupported multi-GPU inventory version")
_inventory_expiry = str(_inventory.get("expires_at", "")).strip()
if _inventory_expiry:
    _expiry = _multigpu_datetime.datetime.fromisoformat(_inventory_expiry.replace("Z", "+00:00"))
    _now = _multigpu_datetime.datetime.now(_multigpu_datetime.timezone.utc)
    if _expiry.tzinfo is None:
        _expiry = _expiry.replace(tzinfo=_multigpu_datetime.timezone.utc)
    if _expiry <= _now:
        raise RuntimeError("The private multi-GPU inventory has expired; relaunch the Piris session")
_inventory_nodes = list(_inventory.get("nodes", _inventory.get("workers", [])))
_controller_nodes = [node for node in _inventory_nodes if bool(node.get("controller", False))]
if len(_controller_nodes) != 1:
    raise RuntimeError("The multi-GPU inventory must identify exactly one controller node")
_controller_node = _controller_nodes[0]
_other_nodes = [node for node in _inventory_nodes if node is not _controller_node]
MULTIGPU_NODES = [_controller_node, *_other_nodes]
_requested_nodes = int(MULTIGPU_SETTINGS["node_count"])
if len(MULTIGPU_NODES) < _requested_nodes:
    raise RuntimeError(
        "Requested %d A100 nodes, but the launcher inventory contains only %d. Make the exact pre-provisioned count available and relaunch."
        % (_requested_nodes, len(MULTIGPU_NODES))
    )
MULTIGPU_NODES = MULTIGPU_NODES[:_requested_nodes]
for _node_index, _node in enumerate(MULTIGPU_NODES):
    _node["host"] = str(_node.get("host", "")).strip()
    _node["key"] = str(_node.get("key", _inventory.get("key", ""))).strip()
    _node["node_name"] = str(_node.get("node_name", _node.get("name", "node-%02d" % (_node_index + 1))))
    if not _node["host"] or not _node["key"]:
        raise RuntimeError("Every multi-GPU inventory node needs host and key fields")
    if not os.path.isfile(os.path.expanduser(_node["key"])):
        raise RuntimeError("Worker SSH key is unavailable on the controller: " + _node["key"])
if len({node["host"] for node in MULTIGPU_NODES}) != len(MULTIGPU_NODES):
    raise RuntimeError("The launcher inventory contains duplicate A100 hosts")

MULTIGPU_ROOT = REMOTE_WORK.rstrip("/") + "/sweep-multithread-" + str(SWEEP_HASH)[:12]
MULTIGPU_WORKER_ROOT = MULTIGPU_ROOT + "/workers"
SWEEP_SHARED_CHECKPOINT_DIR = MULTIGPU_ROOT + "/checkpoints"
SWEEP_RESULTS_BASENAME = "lumerical_sweep_multithread_results"
SHARED_NOMINAL_FSP = None
run_remote_checked(
    "import os\nos.makedirs(%r, exist_ok=True)\nos.makedirs(%r, exist_ok=True)"
    % (MULTIGPU_WORKER_ROOT, SWEEP_SHARED_CHECKPOINT_DIR),
    "Create shared multi-GPU sweep folders",
    timeout=120,
)

MULTIGPU_LICENSE_CHECKOUT_REMOTE = r"""import json
import os
import subprocess
_lic = "/opt/lumerical/v261/licensingclient/linx64"
_hpc_name = "Ansys HPC Pack - Shared Web"

def _license_json(raw, label, require_usage=False):
    text = str(raw or "")
    decoder = json.JSONDecoder()
    objects = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "status" in value:
            objects.append(value)
    if not objects:
        raise RuntimeError(label + " returned no readable LicensingSettings JSON: " + text[-700:])
    value = objects[-1]
    if str(value.get("status", "")).upper() != "SUCCESS":
        raise RuntimeError(label + " failed: " + repr(value))
    if require_usage and value.get("usage") is None and "no products to display" in str(value.get("message", "")).casefold():
        value["usage"] = []
    if require_usage and (
        not isinstance(value.get("usage"), list)
        or any(not isinstance(item, dict) for item in value["usage"])
    ):
        raise RuntimeError(label + " returned an invalid usage list: " + repr(value))
    return value

def _pack_count(value, label):
    total = 0
    for item in value["usage"]:
        if str(item.get("name", "")) != _hpc_name:
            continue
        if item.get("roaming") is not True:
            raise RuntimeError(label + " reported a non-roaming HPC Pack: " + repr(item))
        count = item.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or int(count) != count or count < 0:
            raise RuntimeError(label + " reported an invalid HPC Pack count: " + repr(item))
        total += int(count)
    return total

def _in_use(label):
    result = subprocess.run(
        [os.path.join(_lic, "LicensingSettings"), "web", "shared", "products", "in-use",
         "--type", "roaming", "--mode", "user"],
        capture_output=True, text=True, timeout=180,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(label + " command failed: " + output[-700:])
    return _license_json(output, label, require_usage=True)

_token = os.path.expanduser("~/remote-token.json")
if not os.path.isfile(_token) or os.path.getsize(_token) <= 0:
    raise RuntimeError("~/remote-token.json is missing on this worker; update/relaunch Piris Requirements")
_seeded_token = os.path.expanduser("~/.ansys/ansysid/token.json")
if not os.path.isfile(_seeded_token) or os.path.getsize(_seeded_token) <= 0:
    _seed = subprocess.run(
        [os.path.join(_lic, "ansyscl"), "-WebLoginInput", _token],
        capture_output=True, text=True, timeout=180,
    )
    if _seed.returncode != 0:
        raise RuntimeError("Ansys web sign-in failed: " + (_seed.stdout + _seed.stderr)[-500:])
_enable = subprocess.run(
    [os.path.join(_lic, "LicensingSettings"), "web", "shared", "enable", "--mode", "user"],
    capture_output=True, text=True, timeout=60,
)
if _enable.returncode != 0:
    raise RuntimeError("Could not enable Ansys Shared Web licensing: " + (_enable.stdout + _enable.stderr)[-700:])
_existing = _pack_count(_in_use("HPC Pack pre-check"), "HPC Pack pre-check")
_needed = max(0, 3 - _existing)
if _needed:
    _checkout = subprocess.run(
        [os.path.join(_lic, "LicensingSettings"), "web", "shared", "products", "checkout",
         "--name", _hpc_name, "--count", str(_needed), "--expires", "__PIRIS_HPC_EXPIRY__",
         "--licenseModel", "Shared Web", "--mode", "user"],
        capture_output=True, text=True, timeout=180,
    )
    _checkout_text = (_checkout.stdout + _checkout.stderr).strip()
    if _checkout.returncode != 0:
        raise RuntimeError("Could not reserve HPC Packs for this A100 worker: " + _checkout_text[-700:])
    _license_json(_checkout_text, "HPC Pack checkout")
_verified = _pack_count(_in_use("HPC Pack post-check"), "HPC Pack post-check")
if _verified < 3:
    raise RuntimeError("Could not verify 3 HPC Packs for this A100 worker; found %d" % _verified)
print("__MULTIGPU_LICENSE_ACQUIRED__")
"""

try:
    _hpc_duration_minutes = int(HPC_PACK_DURATION_MINUTES)
except (NameError, TypeError, ValueError):
    raise ValueError("Set HPC_PACK_DURATION_MINUTES in cell 1 to a positive whole number of minutes") from None
if _hpc_duration_minutes <= 0:
    raise ValueError("HPC_PACK_DURATION_MINUTES must be greater than zero")
MULTIGPU_LICENSE_CHECKOUT_REMOTE = MULTIGPU_LICENSE_CHECKOUT_REMOTE.replace(
    "__PIRIS_HPC_EXPIRY__", "PT%dM" % _hpc_duration_minutes
)

MULTIGPU_LICENSE_RELEASE_REMOTE = r"""import json
import os
import subprocess
_lic = "/opt/lumerical/v261/licensingclient/linx64"
_hpc_name = "Ansys HPC Pack - Shared Web"

def _license_json(raw, label, require_usage=False):
    text = str(raw or "")
    decoder = json.JSONDecoder()
    objects = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "status" in value:
            objects.append(value)
    if not objects:
        raise RuntimeError(label + " returned no readable LicensingSettings JSON: " + text[-700:])
    value = objects[-1]
    if str(value.get("status", "")).upper() != "SUCCESS":
        raise RuntimeError(label + " failed: " + repr(value))
    if require_usage and value.get("usage") is None and "no products to display" in str(value.get("message", "")).casefold():
        value["usage"] = []
    if require_usage and (
        not isinstance(value.get("usage"), list)
        or any(not isinstance(item, dict) for item in value["usage"])
    ):
        raise RuntimeError(label + " returned an invalid usage list: " + repr(value))
    return value

def _in_use(label):
    result = subprocess.run(
        [os.path.join(_lic, "LicensingSettings"), "web", "shared", "products", "in-use",
         "--type", "roaming", "--mode", "user"],
        capture_output=True, text=True, timeout=180,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(label + " command failed: " + output[-700:])
    return _license_json(output, label, require_usage=True)

_before = _in_use("HPC Pack pre-release check")
_present = [item for item in _before["usage"] if str(item.get("name", "")) == _hpc_name]
if any(item.get("roaming") is not True for item in _present):
    raise RuntimeError("Pre-release query reported a non-roaming HPC Pack: " + repr(_present))
if _present:
    _release = subprocess.run(
        [os.path.join(_lic, "LicensingSettings"), "web", "shared", "products", "checkin",
         "--name", _hpc_name, "--type", "roaming", "--licenseModel", "Shared Web",
         "--mode", "user"],
        capture_output=True, text=True, timeout=180,
    )
    _release_text = (_release.stdout + _release.stderr).strip()
    if _release.returncode != 0:
        raise RuntimeError("Could not return this worker's HPC Packs: " + _release_text[-700:])
    _license_json(_release_text, "HPC Pack checkin")
_after = _in_use("HPC Pack post-release check")
if any(str(item.get("name", "")) == _hpc_name for item in _after["usage"]):
    raise RuntimeError("HPC Pack still appears in structured post-check usage: " + repr(_after["usage"]))
print("__MULTIGPU_LICENSE_RELEASED__")
"""


def _multigpu_run_checked(client, code, label, timeout=1800):
    guarded, ok_marker, error_marker = _guard_remote_code(code, label)
    output = client.run(guarded, quiet=True, timeout=timeout)
    return _check_remote_output(output, label, ok_marker, error_marker)


def _multigpu_run_once_checked(client, code, label, timeout=1800):
    """Use a fresh SSH process, including after the persistent worker was poisoned."""
    guarded, ok_marker, error_marker = _guard_remote_code(code, label)
    output = client.run_once(guarded, quiet=True, timeout=timeout)
    return _check_remote_output(output, label, ok_marker, error_marker)


CONTROLLER_PACKS_CHECKED_OUT = False
_requested_slots = int(MULTIGPU_SETTINGS["node_count"]) * int(MULTIGPU_SETTINGS["simulations_per_gpu"])
_effective_slots = min(len(SWEEP_CASES), _requested_slots)
print("Prepared A100 nodes:", len(MULTIGPU_NODES))
print("Independent GPU worker slots:", _effective_slots)
print("Required at full concurrency: %d FDTD solve seats and %d HPC Packs." % (_effective_slots, 3 * _effective_slots))
print("Inventory preflight complete; no multi-GPU HPC Packs have been checked out yet.")
'''


_MULTIGPU_LAYER_BUILDER_HELPER_REMOTE = r'''# Helper needed after loading the one shared nominal FSP.
def _layer_builder_geometry(origin_x_um, origin_y_um, geometry=None):
    result = {}
    local_origin_um = np.asarray([origin_x_um, origin_y_um], dtype=float)
    source_geometry = GEOMETRY if geometry is None else geometry
    for polygon in source_geometry:
        key = "%d:%d" % (int(polygon["layer"]), int(polygon.get("datatype", 0)))
        global_vertices_um = np.asarray(polygon["vertices_um"], dtype=float)
        result.setdefault(key, []).append((global_vertices_um - local_origin_um) * UM)
    return result
'''


_MULTIGPU_SINGLE_CASE_REMOTE = r'''# One dynamically assigned case in this worker's persistent FDTD process.
case_index = int(MULTIGPU_CASE_INDEX)
if _sweep_case_is_complete(case_index):
    case_status = "reused"
    print("Reusing completed sweep checkpoint %d/%d (validated on shared storage)" % (case_index + 1, len(SWEEP_CASES)))
else:
    _apply_sweep_case(case_index)
    fdtd.run("FDTD", "GPU")
    primary_name, wavelength_m, primary_response, arrays = _extract_sweep_result()
    _save_sweep_case(case_index, primary_name, wavelength_m, primary_response, arrays)
    case_status = "completed"
    print("Completed shared multi-GPU point %d/%d" % (case_index + 1, len(SWEEP_CASES)))
primary_name, wavelength_m, primary_response = _sweep_checkpoint_spectrum(case_index)
finite = np.isfinite(primary_response)
result_record = {
    "status": case_status,
    "primary_name": str(primary_name),
}
if np.any(finite):
    finite_indices = np.flatnonzero(finite)
    peak_index = int(finite_indices[np.argmax(primary_response[finite])])
    result_record["peak_response"] = float(primary_response[peak_index])
    result_record["peak_wavelength_nm"] = float(wavelength_m[peak_index] / 1e-9)
print("__MAX_LAYOUT_SWEEP_RESULT__" + json.dumps(result_record, sort_keys=True))
'''


_MULTIGPU_RECOVERY_CELL = r'''# Idempotent emergency cleanup: safe to run after success, failure, or interruption.
_records = list(globals().get("MULTIGPU_WORKER_RECORDS", []))
if not _records and "lam" in globals():
    _records = [{"client": lam, "label": "controller", "is_controller": True, "packs_checked_out": bool(globals().get("CONTROLLER_PACKS_CHECKED_OUT", False))}]
_uncertain_cleanup = []
for _record in _records:
    _client = _record.get("client")
    if _client is None:
        continue
    if "_cleanup_worker" in globals():
        try:
            _cleanup_worker(_record)
        except Exception as _cleanup_error:
            print(_record.get("label", "worker"), "cleanup retry error:", str(_cleanup_error)[-500:])
    else:
        _stop_code = (
            '_had_fdtd = "fdtd" in globals()\n'
            'if _had_fdtd:\n    fdtd.close()\n'
            'globals().pop("fdtd", None)\n'
            'print("__MULTIGPU_FDTD_STOPPED__")\n'
        )
        _record["fdtd_stopped"] = False
        if not bool(getattr(_client, "_poisoned", False)):
            try:
                _stop_output = _client.run(_stop_code, quiet=True, timeout=180)
                _record["fdtd_stopped"] = "__MULTIGPU_FDTD_STOPPED__" in _stop_output
            except Exception as _close_error:
                print(_record.get("label", "worker"), "persistent FDTD stop failed:", str(_close_error)[-500:])
        if not _record.get("fdtd_stopped"):
            try:
                _stop_report = _client.stop_work_processes(timeout=25)
                _record["fdtd_stopped"] = bool(
                    isinstance(_stop_report, dict)
                    and _stop_report.get("confirmed") is True
                    and not _stop_report.get("remaining_pids", [])
                )
                if not _record["fdtd_stopped"]:
                    raise RuntimeError("stop_work_processes returned an unconfirmed report: " + repr(_stop_report))
            except Exception as _close_error:
                _record["fdtd_stopped"] = False
                print(_record.get("label", "worker"), "FDTD stop still unconfirmed:", str(_close_error)[-500:])
        if _record.get("fdtd_stopped") and bool(_record.get("packs_checked_out", False)):
            try:
                _release_output = _multigpu_run_once_checked(
                    _client, MULTIGPU_LICENSE_RELEASE_REMOTE,
                    "Return " + str(_record.get("label", "worker")) + " HPC Packs", timeout=300,
                )
                if "__MULTIGPU_LICENSE_RELEASED__" not in _release_output:
                    raise RuntimeError("Licence release success marker missing")
                _record["packs_checked_out"] = False
            except Exception as _release_error:
                print(_record.get("label", "worker"), "LICENCE RELEASE ERROR:", str(_release_error)[-500:])
    if _record.get("packs_checked_out") or not _record.get("fdtd_stopped", True):
        _uncertain_cleanup.append(str(_record.get("label", "worker")))
        print(_record.get("label", "worker"), "session preserved; do not check in packs until FDTD stop is confirmed")
        continue
    try:
        _client.close()
    except Exception:
        pass
CONTROLLER_PACKS_CHECKED_OUT = any(
    bool(record.get("is_controller")) and bool(record.get("packs_checked_out"))
    for record in _records
)
if _uncertain_cleanup:
    raise RuntimeError("Cleanup is still unconfirmed for: " + ", ".join(_uncertain_cleanup))
print("Multi-GPU cleanup finished with confirmed FDTD stop and HPC Pack return.")
'''


_SWEEP_LOCAL_RESULTS_CELL = r'''# Fetch once, then create every per-case curve and summary locally on CPU.
import csv
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

_remote_sweep_npz = REMOTE_WORK + "/lumerical_sweep_results.npz"
_remote_sweep_json = REMOTE_WORK + "/lumerical_sweep_results.json"
_remote_text_summary = REMOTE_WORK + "/summary.txt"
_local_sweep_npz = PIRIS_RESULTS_DIR / "lumerical_sweep_results.npz"
_local_sweep_json = PIRIS_RESULTS_DIR / "lumerical_sweep_results.json"
_local_text_summary = PIRIS_RESULTS_DIR / "summary.txt"
lam.fetch(_remote_sweep_npz, str(_local_sweep_npz))
lam.fetch(_remote_sweep_json, str(_local_sweep_json))
lam.fetch(_remote_text_summary, str(_local_text_summary))
with open(_local_sweep_json, "r", encoding="utf-8") as _sweep_manifest_stream:
    _local_sweep_manifest = json.load(_sweep_manifest_stream)
_remote_best_fsp = str(_local_sweep_manifest.get("best_fsp") or "")
if not _remote_best_fsp:
    raise RuntimeError("The sweep manifest does not contain the required winning FSP path")
_local_best_fsp = PIRIS_FSP_DIR / os.path.basename(_remote_best_fsp)
lam.fetch(_remote_best_fsp, str(_local_best_fsp))
if not _local_best_fsp.is_file() or _local_best_fsp.stat().st_size <= 0:
    raise RuntimeError("The required winning FSP was not fetched: " + str(_local_best_fsp))
print("saved winning solved FSP ->", _local_best_fsp)

with np.load(_local_sweep_npz) as _data:
    _wavelength_nm = np.asarray(_data["wavelength_m"], dtype=float) * 1e9
    _responses = np.asarray(_data["response"], dtype=float)
    _maximum = np.asarray(_data["maximum_response"], dtype=float)
    _target = np.asarray(_data["target_response"], dtype=float)
    _parameter_values = np.asarray(_data["parameter_values"], dtype=float)
    _parameter_names = [str(value) for value in np.asarray(_data["parameter_names"]).tolist()]
    _parameter_codes = [str(value) for value in np.asarray(_data["parameter_codes"]).tolist()]
    _case_labels = [str(value) for value in np.asarray(_data["case_labels"]).tolist()]
    _result_stems = [str(value) for value in np.asarray(_data["result_stems"]).tolist()]
    _success = np.asarray(_data["success"], dtype=bool)
    _best_index = int(np.asarray(_data["best_index"]).ravel()[0])
    _primary_name = str(np.asarray(_data["primary_name"]).ravel()[0])
    _gc_total_over_input = (
        np.asarray(_data["waveguide_total_transmission"], dtype=float)
        if "waveguide_total_transmission" in _data.files else None
    )
    _mmi_output_1_over_input = (
        np.asarray(_data["output_1_over_input"], dtype=float)
        if "output_1_over_input" in _data.files else None
    )
    _mmi_output_2_over_input = (
        np.asarray(_data["output_2_over_input"], dtype=float)
        if "output_2_over_input" in _data.files else None
    )
    _mmi_total_output_over_input = (
        np.asarray(_data["total_output_over_input"], dtype=float)
        if "total_output_over_input" in _data.files else None
    )
    _mmi_output_1_ratio = (
        np.asarray(_data["output_1_ratio"], dtype=float)
        if "output_1_ratio" in _data.files else None
    )
    _mmi_output_2_ratio = (
        np.asarray(_data["output_2_ratio"], dtype=float)
        if "output_2_ratio" in _data.files else None
    )

_is_ce = _primary_name == "coupling_efficiency"
if _is_ce and _gc_total_over_input is None:
    raise RuntimeError(
        "The grating sweep bundle is missing total waveguide power / measured input"
    )
_is_mmi = (
    _primary_name == "output_1_over_input"
    and _mmi_output_1_over_input is not None
    and _mmi_output_2_over_input is not None
)
_db_floor = 1e-15
_responses_db = (
    10.0 * np.log10(np.maximum(_responses, _db_floor)) if _is_ce else None
)
_gc_total_over_input_db = (
    10.0 * np.log10(np.maximum(_gc_total_over_input, _db_floor))
    if _is_ce else None
)
_maximum_db = (
    10.0 * np.log10(np.maximum(_maximum, _db_floor)) if _is_ce else None
)
_target_db = (
    10.0 * np.log10(np.maximum(_target, _db_floor)) if _is_ce else None
)
_ylabel = (
    "coupling efficiency (linear)" if _is_ce
    else "branch power / measured input power (linear)" if _is_mmi
    else _primary_name.replace("_", " ") + " (linear)"
)
_curve_paths = []
for _index, (_label, _stem) in enumerate(zip(_case_labels, _result_stems)):
    if not _success[_index]:
        continue
    _path = PIRIS_RESULTS_DIR / (_stem + ".png")
    if _is_mmi:
        _figure, _axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
        _axis, _split_axis = _axes
        _axis.plot(
            _wavelength_nm, _mmi_output_1_over_input[_index],
            lw=2.2, color="#7c3aed", label="upper output / input",
        )
        _axis.plot(
            _wavelength_nm, _mmi_output_2_over_input[_index],
            lw=2.2, color="#0891b2", label="lower output / input",
        )
        if _mmi_total_output_over_input is not None:
            _axis.plot(
                _wavelength_nm, _mmi_total_output_over_input[_index],
                lw=1.7, ls="--", color="#475569", label="total output / input",
            )
        _axis.legend()
        if _mmi_output_1_ratio is not None and _mmi_output_2_ratio is not None:
            _split_axis.plot(
                _wavelength_nm, _mmi_output_1_ratio[_index],
                lw=2.0, color="#7c3aed", label="upper split fraction",
            )
            _split_axis.plot(
                _wavelength_nm, _mmi_output_2_ratio[_index],
                lw=2.0, color="#0891b2", label="lower split fraction",
            )
            _split_axis.axhline(
                0.5, color="#64748b", ls="--", lw=1.0, label="ideal 50/50",
            )
            _split_axis.set(
                xlabel="wavelength [nm]",
                ylabel="fraction of total output",
                ylim=(0.0, 1.0),
            )
            _split_axis.grid(alpha=0.3)
            _split_axis.legend()
    elif _is_ce:
        _figure, (_axis, _db_axis) = plt.subplots(
            1, 2, figsize=(13.0, 4.8), sharex=True,
        )
        _axis.plot(
            _wavelength_nm, _responses[_index], lw=2.4, color="#7c3aed",
            label="selected TE / measured input",
        )
        _axis.plot(
            _wavelength_nm, _gc_total_over_input[_index], lw=2.0,
            color="#f59e0b", label="total waveguide / measured input",
        )
        _db_axis.plot(
            _wavelength_nm, _responses_db[_index], lw=2.4,
            color="#7c3aed", label="selected TE / input",
        )
        _db_axis.plot(
            _wavelength_nm, _gc_total_over_input_db[_index], lw=2.0,
            color="#f59e0b", label="total / input",
        )
        _db_axis.set(
            xlabel="wavelength [nm]",
            ylabel="normalized power [dB]",
            title="Waveguide transmission — dB — " + _label,
        )
        _db_axis.grid(alpha=0.3)
        _axis.legend(loc="best")
        _db_axis.legend(loc="best")
    else:
        _figure, _axis = plt.subplots(figsize=(8.5, 4.8))
        _axis.plot(_wavelength_nm, _responses[_index], lw=2.2, color="#2563eb")
    _axis.set(
        xlabel="wavelength [nm]",
        ylabel=_ylabel,
        title=(
            "Waveguide transmission — " if _is_ce
            else "MMI branch transmission — " if _is_mmi
            else "Lumerical sweep response — "
        ) + _label,
    )
    _axis.grid(alpha=0.3)
    _figure.tight_layout()
    _figure.savefig(_path, dpi=160, bbox_inches="tight")
    plt.close(_figure)
    _curve_paths.append(_path)
    print("saved ->", _path)

if _is_ce and len(_parameter_codes) == 1:
    _summary_stem = "CE-maximum-vs-" + _parameter_codes[0]
elif _is_ce:
    _summary_stem = "CE-maximum-" + _parameter_codes[0] + "-vs-" + _parameter_codes[1]
elif _is_mmi and len(_parameter_codes) == 1:
    _summary_stem = "MMI-upper-over-input-maximum-vs-" + _parameter_codes[0]
elif _is_mmi:
    _summary_stem = "MMI-upper-over-input-maximum-" + _parameter_codes[0] + "-vs-" + _parameter_codes[1]
else:
    _summary_stem = "sweep-maximum-summary"
_summary_csv = PIRIS_RESULTS_DIR / (_summary_stem + ".csv")
with _summary_csv.open("w", newline="", encoding="utf-8") as _handle:
    _writer = csv.writer(_handle)
    _summary_header = [*_parameter_names, "maximum_response", "target_response"]
    if _is_ce:
        _summary_header.extend(["maximum_response_db", "target_response_db"])
    _summary_header.extend(["success", "curve_file"])
    _writer.writerow(_summary_header)
    for _index in range(len(_case_labels)):
        _summary_row = [
            *_parameter_values[_index].tolist(),
            _maximum[_index],
            _target[_index],
        ]
        if _is_ce:
            _summary_row.extend([_maximum_db[_index], _target_db[_index]])
        _summary_row.extend([
            bool(_success[_index]), _result_stems[_index] + ".png",
        ])
        _writer.writerow(_summary_row)

_summary_path = PIRIS_RESULTS_DIR / (_summary_stem + ".png")
if len(_parameter_names) == 1:
    _order = np.argsort(_parameter_values[:, 0])
    if _is_ce:
        _figure, (_axis, _db_axis) = plt.subplots(
            1, 2, figsize=(13.0, 5.0), sharex=True,
        )
    else:
        _figure, _axis = plt.subplots(figsize=(8.3, 5.0))
    _axis.plot(_parameter_values[_order, 0], _maximum[_order], marker="o", lw=2.2)
    _axis.set(
        xlabel=_parameter_names[0].replace("_", " "),
        ylabel=(
            "maximum coupling efficiency" if _is_ce
            else "maximum upper output / input" if _is_mmi
            else "maximum response"
        ),
        title=(
            "Maximum CE across wavelength" if _is_ce
            else "Maximum MMI upper-branch transmission across wavelength" if _is_mmi
            else "Maximum response across wavelength"
        ),
    )
    _axis.grid(alpha=0.3)
    if _is_ce:
        _db_axis.plot(
            _parameter_values[_order, 0], _maximum_db[_order],
            marker="o", lw=2.2, color="#7c3aed",
        )
        _db_axis.set(
            xlabel=_parameter_names[0].replace("_", " "),
            ylabel="maximum coupling efficiency [dB]",
            title="Maximum CE across wavelength — dB",
        )
        _db_axis.grid(alpha=0.3)
else:
    _x_values = np.unique(_parameter_values[:, 0])
    _y_values = np.unique(_parameter_values[:, 1])
    _grid = np.full((_y_values.size, _x_values.size), np.nan)
    for _row, _value in zip(_parameter_values, _maximum):
        _x_index = int(np.where(np.isclose(_x_values, _row[0]))[0][0])
        _y_index = int(np.where(np.isclose(_y_values, _row[1]))[0][0])
        _existing = _grid[_y_index, _x_index]
        _grid[_y_index, _x_index] = _value if not np.isfinite(_existing) else max(_existing, _value)
    if _is_ce:
        _figure, (_axis, _db_axis) = plt.subplots(1, 2, figsize=(13.0, 5.8))
    else:
        _figure, _axis = plt.subplots(figsize=(8.3, 5.8))
    _image = _axis.imshow(
        _grid,
        origin="lower",
        aspect="auto",
        extent=[_x_values.min(), _x_values.max(), _y_values.min(), _y_values.max()],
        cmap="viridis",
    )
    _axis.set(
        xlabel=_parameter_names[0].replace("_", " "),
        ylabel=_parameter_names[1].replace("_", " "),
        title=(
            "Maximum CE for each pitch/filling combination" if _is_ce
            else "Maximum MMI upper-branch transmission" if _is_mmi
            else "Maximum response for the first two sweep axes"
        ),
    )
    _figure.colorbar(
        _image,
        ax=_axis,
        label=(
            "maximum coupling efficiency" if _is_ce
            else "maximum upper output / input" if _is_mmi
            else "maximum response"
        ),
    )
    if _is_ce:
        _db_grid = 10.0 * np.log10(np.maximum(_grid, _db_floor))
        _db_image = _db_axis.imshow(
            _db_grid,
            origin="lower",
            aspect="auto",
            extent=[_x_values.min(), _x_values.max(), _y_values.min(), _y_values.max()],
            cmap="viridis",
        )
        _db_axis.set(
            xlabel=_parameter_names[0].replace("_", " "),
            ylabel=_parameter_names[1].replace("_", " "),
            title="Maximum CE for each parameter combination — dB",
        )
        _figure.colorbar(
            _db_image, ax=_db_axis, label="maximum coupling efficiency [dB]",
        )
_figure.tight_layout()
_figure.savefig(_summary_path, dpi=170, bbox_inches="tight")
plt.close(_figure)
display(Image(filename=str(_summary_path), width=1000))
if _is_ce:
    print(
        "best case ->", _case_labels[_best_index],
        "maximum", _maximum[_best_index], "linear /", _maximum_db[_best_index], "dB",
    )
else:
    print("best case ->", _case_labels[_best_index], "maximum", _maximum[_best_index])
print("saved ->", _summary_path)
print("saved ->", _summary_csv)
print("saved ->", _local_sweep_npz)
print("saved ->", _local_sweep_json)
print("saved ->", _local_text_summary)
'''


_FETCH_RESULTS_CELL = r'''# Fetch every verified artifact before closing Lumerical or returning the HPC Packs.
REMOTE_ARTIFACTS = [
    REMOTE_WORK + "/max_layout_results.npz",
    REMOTE_WORK + "/max_layout_results.json",
    REMOTE_WORK + "/summary.txt",
]
if SHOW_GEOMETRY_PREVIEW:
    REMOTE_ARTIFACTS.append(REMOTE_WORK + "/geometry_xyz_projections.png")
# The solved/final FSP was already verified and fetched immediately after the
# solve, before optional analysis.  Do not transfer that large file twice.
if PORTS and SHOW_PORT_MODE_PREVIEW:
    REMOTE_ARTIFACTS.append(REMOTE_WORK + "/port_mode_Ex_Ey.png")
if GRATING_ANALYSIS and SETTINGS.get("run_after_build", False):
    print("Grating analysis and its locally rendered plot were already fetched by section 10.")
elif GRATING_ANALYSIS:
    print("Grating plots were not requested because automatic solving is disabled.")
if MMI_ANALYSIS and SETTINGS.get("run_after_build", False):
    REMOTE_ARTIFACTS.extend([
        REMOTE_WORK + "/mmi_splitting_ratio.png",
        REMOTE_WORK + "/mmi_field_distribution.png",
        REMOTE_WORK + "/mmi_analysis.npz",
    ])
elif MMI_ANALYSIS:
    print("MMI splitting-ratio plots were not requested because automatic solving is disabled.")

_artifact_expression = "{path: bool(os.path.isfile(path) and os.path.getsize(path) > 0) for path in %r}" % REMOTE_ARTIFACTS
REMOTE_ARTIFACT_STATUS = lam.get(_artifact_expression)
FETCHED_RESULTS = []
MISSING_REMOTE_ARTIFACTS = []
FAILED_TRANSFERS = []
for remote_path in REMOTE_ARTIFACTS:
    if not REMOTE_ARTIFACT_STATUS.get(remote_path, False):
        MISSING_REMOTE_ARTIFACTS.append(remote_path)
        print("ERROR — remote stage did not create:", remote_path)
        continue
    local_directory = PIRIS_FSP_DIR if remote_path.lower().endswith(".fsp") else PIRIS_RESULTS_DIR
    local_path = local_directory / os.path.basename(remote_path)
    try:
        fetched = lam.fetch(remote_path, str(local_path))
        if not local_path.is_file() or local_path.stat().st_size <= 0:
            raise RuntimeError("downloaded file is missing or empty: " + str(local_path))
        FETCHED_RESULTS.append(fetched)
        print("saved ->", fetched)
    except Exception as exc:
        FAILED_TRANSFERS.append((remote_path, str(exc)))
        print("TRANSFER ERROR —", os.path.basename(remote_path), str(exc)[:200])
if MISSING_REMOTE_ARTIFACTS:
    raise RuntimeError("Required remote artifacts are missing: " + repr(MISSING_REMOTE_ARTIFACTS))
if FAILED_TRANSFERS:
    raise RuntimeError("Required artifact transfers failed: " + repr(FAILED_TRANSFERS))
'''


_RELEASE_LICENSES_CELL = r'''# Stop this notebook's FDTD work, then return and verify its roaming HPC Packs.
# This cell never terminates a Lambda node.  It uses the documented named
# roaming-product check-in; check-in does not accept the checkout --count flag.
import json

_hpc_pack_name = "Ansys HPC Pack - Shared Web"


def _release_license_json(raw_output, label, require_usage=False):
    text = str(raw_output or "")
    decoder = json.JSONDecoder()
    objects = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "status" in value:
            objects.append(value)
    if not objects:
        raise RuntimeError(label + " returned no readable LicensingSettings JSON: " + text[-700:])
    value = objects[-1]
    if str(value.get("status", "")).upper() != "SUCCESS":
        raise RuntimeError(label + " failed: " + repr(value))
    if require_usage and value.get("usage") is None and "no products to display" in str(value.get("message", "")).casefold():
        value["usage"] = []
    if require_usage and (
        not isinstance(value.get("usage"), list)
        or any(not isinstance(item, dict) for item in value["usage"])
    ):
        raise RuntimeError(label + " returned an invalid usage list: " + repr(value))
    return value


def _release_in_use(label):
    command = (
        f'{LIC}/LicensingSettings web shared products in-use '
        '--type roaming --mode user'
    )
    result = subprocess.run(
        _SSH + [HOST, command], capture_output=True, text=True, timeout=180
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(label + " command failed: " + output[-700:])
    return _release_license_json(output, label, require_usage=True)

_stop_error = None
_release_error = None
try:
    _stop_output = lam.run(
        "import os\n"
        "_fdtd_owner = globals().pop('fdtd', None)\n"
        "if _fdtd_owner is not None:\n    _fdtd_owner.close()\n"
        "_runtime_fsp = globals().get('REMOTE_RUNTIME_PROJECT_FILE', '')\n"
        "if _runtime_fsp and os.path.isfile(_runtime_fsp):\n"
        "    try:\n        os.remove(_runtime_fsp)\n        print('Removed transient runtime FSP during release.')\n"
        "    except Exception as _cleanup_exc:\n        print('Runtime-FSP cleanup warning:', str(_cleanup_exc)[:180])\n"
        "print('__MAX_LAYOUT_FDTD_CLOSED__')",
        quiet=True,
        timeout=120,
    )
    if "__MAX_LAYOUT_FDTD_CLOSED__" not in str(_stop_output):
        raise RuntimeError("the remote FDTD close marker was not returned")
except Exception as _normal_close_exc:
    _stop_error = "Normal FDTD close failed: " + str(_normal_close_exc)[-500:]

# Verify/stop only processes owned by this exact Lambda work session when the
# updated launcher helper is available.  Never check a roaming licence in
# while its solver may still be alive and able to reacquire it.
if callable(getattr(lam, "stop_work_processes", None)):
    try:
        _stop_report = lam.stop_work_processes(timeout=25)
        _stop_confirmed = bool(
            isinstance(_stop_report, dict)
            and _stop_report.get("confirmed") is True
            and not _stop_report.get("remaining_pids", [])
        )
        if not _stop_confirmed:
            raise RuntimeError("unconfirmed process-stop report: " + repr(_stop_report))
        _stop_error = None
    except Exception as _forced_stop_exc:
        _stop_error = "FDTD/process stop remains unconfirmed: " + str(_forced_stop_exc)[-500:]

try:
    if _stop_error is not None:
        raise RuntimeError(_stop_error + ". HPC Packs were NOT checked in.")
    _before = _release_in_use("HPC Pack pre-release check")
    _present = [
        item for item in _before["usage"]
        if str(item.get("name", "")) == _hpc_pack_name
    ]
    if any(item.get("roaming") is not True for item in _present):
        raise RuntimeError("Pre-release query reported a non-roaming HPC Pack: " + repr(_present))
    if _present:
        _release = subprocess.run(
            _SSH + [HOST,
                f'{LIC}/LicensingSettings web shared products checkin '
                f'--name "{_hpc_pack_name}" --type roaming '
                '--licenseModel "Shared Web" --mode user'],
            capture_output=True, text=True, timeout=180,
        )
        _release_out = (_release.stdout + _release.stderr).strip()
        if _release.returncode != 0:
            raise RuntimeError("HPC Pack checkin command failed: " + _release_out[-700:])
        _release_license_json(_release_out, "HPC Pack checkin")
    _after = _release_in_use("HPC Pack post-release check")
    if any(str(item.get("name", "")) == _hpc_pack_name for item in _after["usage"]):
        raise RuntimeError(
            "Ansys did not confirm that the roaming HPC Pack reservation disappeared: "
            + repr(_after["usage"])
        )
    print("HPC Packs: named roaming reservation returned and absent from post-check in-use output")
except Exception as _release_exc:
    _release_error = str(_release_exc)
finally:
    lam.close()
if _release_error is not None:
    raise RuntimeError(_release_error)
'''


def generate_lumerical_notebook(
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Build an nbformat-4 notebook and return it with non-fatal warnings."""
    included_raw = (
        configuration.get("included_layers", [])
        if "included_layers" in configuration
        else available_geometry_layers(components)
    )
    included = {(int(value[0]), int(value[1])) for value in included_raw}
    (
        geometry, ports, fiber_geometries, gaussian_sources, monitors, bbox,
        warnings,
    ) = _collect_export_data(
        components,
        included,
        origin_um=configuration.get("_fixed_origin_um"),
    )
    for component in components:
        if str(component.get("kind", "")) not in {
            "Grating coupler", "GC-SOI", "1x2 MMI",
        }:
            continue
        params = component.setdefault("params", {})
        if params.pop("waveguide_effective_index", None) is not None:
            warnings.append(
                "Ignored legacy waveguide_effective_index on %s UID %s; the notebook derives "
                "the target from the actual dispersive core and adjacent stack materials."
                % (component.get("kind", "component"), component.get("uid", 0))
            )
    _normalize_grating_measurement_objects(ports, monitors, warnings)
    _repair_missing_grating_gaussian_sources(
        components, gaussian_sources, monitors, warnings
    )
    if not bool(configuration.get("include_ports", True)):
        ports = []
    alignment_stack = deepcopy(configuration.get("material_stack") or default_stack())
    _apply_authoritative_grating_angles(
        components, ports, fiber_geometries, gaussian_sources, monitors, warnings
    )
    _synchronize_fiber_port_parameters(
        ports, fiber_geometries, monitors, warnings, alignment_stack
    )
    _synchronize_gaussian_source_parameters(
        gaussian_sources, monitors, warnings, alignment_stack
    )

    grating_analysis: dict[str, Any] | None = None
    grating_components = [component for component in components if str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"}]
    if grating_components:
        grating = grating_components[0]
        grating_uid = int(grating.get("uid", 0))
        excitation_type = str(
            grating.get("params", {}).get("excitation_type", "fiber_mode")
        ).strip().lower()
        if excitation_type not in {"fiber_mode", "gaussian_beam"}:
            raise ValueError(
                "Grating excitation_type must be fiber_mode or gaussian_beam"
            )
        if len(grating_components) > 1:
            warnings.append("Multiple grating couplers were exported; natural-radiation analysis uses the first one only.")
        grating_polygons = [item for item in geometry if int(item.get("component_uid", -1)) == grating_uid]
        waveguide_power_monitors = [
            monitor for monitor in monitors
            if int(monitor.get("parent_component_uid", -1)) == grating_uid
            and str(monitor.get("monitor_kind", "")) == "Power monitor"
            and str(monitor.get("grating_monitor_role", "")) == "waveguide_total_power"
        ]
        waveguide_ports = [
            port for port in ports
            if bool(port.get("enabled", True))
            and int(port.get("parent_component_uid", -1)) == grating_uid
            and str(port.get("domain", "optical")).lower() == "optical"
            and str(port.get("plane normal", "X")).upper() in {"X", "Y"}
            and str(port.get("parent_port_name", "")) == "waveguide_point"
        ]
        fiber_ports = [
            port for port in ports
            if bool(port.get("enabled", True))
            and str(port.get("domain", "optical")).lower() == "optical"
            and str(port.get("plane normal", "X")).upper() == "Z"
            and str(port.get("parent_port_name", "")) != "fiber_input_power"
            and str(port.get("fiber plane role", "source")).strip().lower() not in {
                "input power measurement", "passive fiber measurement", "fiber power measurement"
            }
        ]
        fiber_input_monitors = [
            monitor for monitor in monitors
            if int(monitor.get("parent_component_uid", -1)) == grating_uid
            and str(monitor.get("monitor_kind", "")) == "Power monitor"
            and str(monitor.get("plane normal", "Z")).upper() == "Z"
            and (
                str(monitor.get("parent_port_name", "")) == "fiber_input_power"
                or str(monitor.get("fiber plane role", "")).strip().lower()
                == "input power measurement"
            )
        ]
        matching_fiber_ports = [
            port for port in fiber_ports
            if int(port.get("parent_component_uid", -1)) == grating_uid
        ]
        if matching_fiber_ports:
            fiber_ports = matching_fiber_ports
        matching_gaussian_sources = [
            source for source in gaussian_sources
            if int(source.get("parent_component_uid", -1)) == grating_uid
        ]
        duplicate_roles = []
        for role_name, role_items in (
            ("waveguide receiver ports", waveguide_ports),
            ("waveguide total-power monitors", waveguide_power_monitors),
            ("incident input-power monitors", fiber_input_monitors),
            ("Gaussian sources", matching_gaussian_sources),
        ):
            if len(role_items) > 1:
                duplicate_roles.append("%d %s" % (len(role_items), role_name))
        if duplicate_roles:
            raise ValueError(
                "Grating component UID %s has an ambiguous simulation topology: %s. "
                "Refresh its automatic simulation setup before exporting."
                % (grating_uid, ", ".join(duplicate_roles))
            )
        # Fiber HE11 is a near-degenerate polarization pair, but the pair is
        # not guaranteed to be returned as eigensolver modes 1 and 2.  Solve
        # the first three modes, identify the pair nearest the fiber index,
        # then select its Gaussian/circular member polarized perpendicular to
        # the grating axis.  The exported global phi includes rotation.
        if excitation_type == "fiber_mode":
            for fiber_mode_port in fiber_ports:
                fiber_mode_port["mode"] = "user select"
                fiber_mode_port["mode number"] = 0
                fiber_mode_port["polarization"] = "local TE"
                fiber_mode_port["candidate mode numbers"] = [1, 2, 3]
                fiber_mode_port.setdefault("mode degeneracy tolerance", 0.01)
                fiber_mode_port.setdefault("minimum local TE fraction", 0.8)
        if not grating_polygons:
            warnings.append("Grating analysis was not added because no polygons from the grating coupler were selected.")
        elif not waveguide_power_monitors:
            warnings.append(
                "Grating analysis was not added because the waveguide total-power monitor is missing. "
                "Refresh the grating simulation setup to place it beside the passive waveguide receiver port."
            )
        elif not waveguide_ports:
            warnings.append(
                "Grating analysis was not added because the passive waveguide receiver FDTD port is missing. "
                "Refresh the grating simulation setup to select its confined mode by effective index."
            )
        elif excitation_type == "fiber_mode" and not fiber_ports:
            warnings.append(
                "Grating analysis was not added because no manually placed Ansys-style fiber port was exported. "
                "Add one from Ports & monitors above the grating exit."
            )
        elif not fiber_input_monitors:
            warnings.append(
                "Grating analysis was not added because the power monitor below the fiber source is missing. "
                "Refresh the grating simulation setup so incident power can be measured independently of the source port."
            )
        elif excitation_type == "fiber_mode" and not fiber_geometries:
            warnings.append(
                "Grating analysis was not added because no separate fiber geometry group was exported. "
                "Place the Ansys fiber geometry group and put the Fiber-axis FDTD port through its core/cladding."
            )
        elif excitation_type == "gaussian_beam" and not matching_gaussian_sources:
            warnings.append(
                "Grating analysis was not added because the Gaussian excitation source is missing. "
                "Refresh the grating simulation setup after selecting gaussian_beam."
            )
        else:
            points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in grating_polygons])
            minimum, maximum = points.min(axis=0), points.max(axis=0)
            waveguide_power_monitor = deepcopy(waveguide_power_monitors[0])
            waveguide_ports[0]["target neff"] = 0.0
            waveguide_ports[0]["target neff strategy"] = (
                "automatic midpoint of actual core and adjacent dielectric indices"
            )
            waveguide_port = deepcopy(waveguide_ports[0])
            grating_center = 0.5 * (minimum + maximum)
            source_object = min(
                matching_gaussian_sources if excitation_type == "gaussian_beam" else fiber_ports,
                key=lambda source: float(
                    np.linalg.norm(
                        np.asarray(source.get("center", (0.0, 0.0)), dtype=float)
                        - grating_center
                    )
                ),
            )
            fiber_input_monitor = min(
                fiber_input_monitors,
                key=lambda monitor: float(
                    np.linalg.norm(
                        np.asarray(monitor.get("center", (0.0, 0.0)), dtype=float)
                        - np.asarray(source_object.get("center", (0.0, 0.0)), dtype=float)
                    )
                ),
            )
            source_is_valid = True
            fiber_geometry = None
            if excitation_type == "fiber_mode":
                fiber_geometry = min(
                    fiber_geometries,
                    key=lambda fiber: float(
                        np.linalg.norm(
                            np.asarray(fiber.get("center", (0.0, 0.0)), dtype=float)
                            - np.asarray(source_object.get("center", (0.0, 0.0)), dtype=float)
                        )
                    ),
                )
                fiber_alignment_error_um = float(
                    np.linalg.norm(
                        np.asarray(fiber_geometry.get("center", (0.0, 0.0)), dtype=float)
                        - np.asarray(
                            source_object.get(
                                "fiber bottom center_um",
                                source_object.get("center", (0.0, 0.0)),
                            ),
                            dtype=float,
                        )
                    )
                )
                if fiber_alignment_error_um > 0.5 * float(
                    fiber_geometry.get("core diameter_um", 9.0)
                ):
                    source_is_valid = False
                    warnings.append(
                        "Grating analysis was not added because the Fiber-axis FDTD port does not pass through "
                        f"the selected fiber core (top-view separation {fiber_alignment_error_um:.6g} µm)."
                    )
            if source_is_valid:
                grating_analysis = {
                    "component_uid": grating_uid,
                    "excitation_type": excitation_type,
                    "source_kind": (
                        "gaussian" if excitation_type == "gaussian_beam" else "fdtd_port"
                    ),
                    "source_name": str(
                        source_object.get(
                            "name",
                            f"uid_{grating_uid}_gaussian_source"
                            if excitation_type == "gaussian_beam"
                            else f"uid_{grating_uid}_fiber_axis",
                        )
                    ),
                    "waveguide_power_monitor_name": str(
                        waveguide_power_monitor.get("name", f"uid_{grating_uid}_waveguide_total_power")
                    ),
                    "waveguide_port_name": str(
                        waveguide_port.get("name", f"uid_{grating_uid}_waveguide_point")
                    ),
                    "waveguide_port_expansion_result_name": "expansion for port monitor",
                    # Receiver power leaves the simulation through the
                    # waveguide port and is exposed as T_out by the standard
                    # FDTD-port expansion result.
                    "waveguide_port_modal_direction": "T_out",
                    "waveguide_port_modal_sign": (
                        1.0
                        if (
                            str(waveguide_port.get("plane normal", "X")).upper() == "X"
                            and math.cos(math.radians(float(waveguide_port.get("outward_orientation_deg", 180.0)))) >= 0.0
                        ) or (
                            str(waveguide_port.get("plane normal", "X")).upper() == "Y"
                            and math.sin(math.radians(float(waveguide_port.get("outward_orientation_deg", 180.0)))) >= 0.0
                        )
                        else -1.0
                    ),
                    "waveguide_total_power_sign": (
                        1.0
                        if (
                            str(waveguide_power_monitor.get("plane normal", "X")).upper() == "X"
                            and math.cos(math.radians(float(waveguide_power_monitor.get("orientation_deg", 180.0)))) >= 0.0
                        ) or (
                            str(waveguide_power_monitor.get("plane normal", "X")).upper() == "Y"
                            and math.sin(math.radians(float(waveguide_power_monitor.get("orientation_deg", 180.0)))) >= 0.0
                        )
                        else -1.0
                    ),
                    "waveguide_target_neff": 0.0,
                    "waveguide_target_strategy": (
                        "automatic midpoint of actual core and adjacent dielectric indices"
                    ),
                    "waveguide_neff_tolerance": float(waveguide_port.get("neff tolerance", 0.3)),
                    "fiber_input_power_monitor_name": str(fiber_input_monitor.get("name", "")),
                    "fiber_input_power_sign": float(
                        fiber_input_monitor.get("expected propagation sign", -1.0)
                    ),
                    "frequency_points": int(configuration.get("frequency_points", 31)),
                }
                if excitation_type == "fiber_mode":
                    grating_analysis.update(
                        {
                            "fiber_port_name": str(source_object.get("name", "fiber")),
                            "fiber_source_mode": "auto local TE",
                            "fiber_polarization": "local TE",
                            "fiber_axis_orientation_deg": float(
                                source_object.get("angle phi", 0.0)
                            ),
                            "fiber_mode_candidates": [1, 2, 3],
                            "fiber_geometry_name": str(
                                (fiber_geometry or {}).get("name", "fiber")
                            ),
                        }
                    )
                else:
                    grating_analysis.update(
                        {
                            "gaussian_polarization": "local TE / S",
                            "gaussian_polarization_angle_deg": 90.0,
                            "gaussian_axis_orientation_deg": float(
                                source_object.get("angle phi", 0.0)
                            ),
                            "gaussian_angle_theta_deg": float(
                                source_object.get("angle theta", 0.0)
                            ),
                            "gaussian_waist_radius_um": float(
                                source_object.get("waist radius_um", 4.5)
                            ),
                            "gaussian_distance_from_waist_um": float(
                                source_object.get("distance from waist_um", 0.0)
                            ),
                        }
                    )

    mmi_analysis: dict[str, Any] | None = None
    mmi_components = [
        component for component in components
        if str(component.get("kind", "")) == "1x2 MMI"
    ]
    if mmi_components and grating_analysis:
        warnings.append(
            "MMI splitting analysis was skipped because this export also contains a grating analysis; "
            "each analysis requires a different source port. Export the MMI component separately."
        )
    elif mmi_components:
        mmi = mmi_components[0]
        mmi_uid = int(mmi.get("uid", 0))
        if len(mmi_components) > 1:
            warnings.append("Multiple 1x2 MMIs were exported; splitting analysis uses the first one only.")
        matching_ports = [
            port for port in ports
            if bool(port.get("enabled", True))
            and int(port.get("parent_component_uid", -1)) == mmi_uid
        ]
        by_parent_name = {
            str(port.get("parent_port_name", "")): port
            for port in matching_ports
            if str(port.get("parent_port_name", ""))
        }
        required_names = ("left_external", "upper_right", "lower_right")
        missing_names = [name for name in required_names if name not in by_parent_name]
        if missing_names:
            warnings.append(
                "MMI splitting analysis was not added because these MMI FDTD ports are missing: "
                + ", ".join(missing_names)
                + ". Add or refresh the component simulation setup before exporting."
            )
        else:
            input_port = by_parent_name["left_external"]
            upper_port = by_parent_name["upper_right"]
            lower_port = by_parent_name["lower_right"]
            mmi_params = mmi.get("params", {})
            mmi_neff_tolerance = max(
                0.0, float(mmi_params.get("waveguide_neff_tolerance", 0.3))
            )
            mmi_mode_search_count = max(
                1, int(mmi_params.get("waveguide_mode_search_count", 20))
            )
            for port, role_label in (
                (input_port, "input"),
                (upper_port, "upper output"),
                (lower_port, "lower output"),
            ):
                # A standard 1x2 MMI has the same access-waveguide
                # cross-section at all three planes.  Apply one modal target
                # and polarization contract even when the ports came from an
                # older layout file or were placed manually.
                port["mode"] = str(port.get("mode", "fundamental TE mode"))
                port["polarization"] = "local TE"
                port["target neff"] = 0.0
                port["target neff strategy"] = (
                    "automatic midpoint of actual core and adjacent dielectric indices"
                )
                port["neff tolerance"] = mmi_neff_tolerance
                port["mode search count"] = mmi_mode_search_count
                port["mmi port role"] = role_label
            reference_monitors = [
                monitor for monitor in monitors
                if int(monitor.get("parent_component_uid", -1)) == mmi_uid
                and str(monitor.get("parent_port_name", "")) == "left_external"
                and str(monitor.get("monitor_kind", "")) == "Power monitor"
            ]
            if not reference_monitors:
                warnings.append(
                    "MMI splitting analysis was not added because the input reference power monitor is missing. "
                    "Refresh the MMI simulation setup to place it 2 um before the input taper."
                )
            else:
                reference_monitor = reference_monitors[0]
                include_field_distribution = not bool(
                    configuration.get("_sweep_export", False)
                )
                field_monitor = None
                if include_field_distribution:
                    field_monitors = [
                        monitor for monitor in monitors
                        if int(monitor.get("parent_component_uid", -1)) == mmi_uid
                        and str(monitor.get("parent_port_name", "")) == "mmi_longitudinal_field"
                        and str(monitor.get("monitor_kind", "")) == "Field profile monitor"
                    ]
                    if field_monitors:
                        field_monitor = field_monitors[0]
                    else:
                        mmi_geometry = [
                            item for item in geometry
                            if int(item.get("component_uid", -1)) == mmi_uid
                        ]
                        points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in mmi_geometry])
                        low, high = points.min(axis=0), points.max(axis=0)
                        field_monitor = {
                            "name": f"uid_{mmi_uid}_mmi_field",
                            "monitor_kind": "Field profile monitor",
                            "monitor geometry": "surface",
                            "plane normal": "Z",
                            "z reference": "device center",
                            "distance_um": 0.0,
                            "center": [float(0.5 * (low[0] + high[0])), float(0.5 * (low[1] + high[1]))],
                            "orientation_deg": float(mmi.get("orientation_deg", 0.0)),
                            "x span": float(high[0] - low[0]),
                            "y span": float(high[1] - low[1]),
                            "z span": 0.0,
                            "parent_component_uid": mmi_uid,
                            "parent_port_name": "mmi_longitudinal_field",
                        }
                        monitors.append(field_monitor)
                        warnings.append("Added the MMI longitudinal field-profile monitor to this notebook export.")
                mmi_analysis = {
                    "component_uid": mmi_uid,
                    "input_port_name": str(input_port.get("name", "mmi_input")),
                    "input_mode": "mode 1",
                    "input_reference_monitor_name": str(reference_monitor.get("name", "mmi_input_reference")),
                    "include_field_distribution": include_field_distribution,
                    "input_reference_before_taper_um": float(
                        mmi.get("params", {}).get("input_reference_before_taper_um", 2.0)
                    ),
                    "output_port_names": [
                        str(upper_port.get("name", "mmi_upper")),
                        str(lower_port.get("name", "mmi_lower")),
                    ],
                    "output_labels": ["upper output", "lower output"],
                    "port_profile_labels": [
                        "MMI input",
                        "MMI upper output",
                        "MMI lower output",
                    ],
                    "port_target_neff": 0.0,
                    "port_target_strategy": (
                        "automatic midpoint of actual core and adjacent dielectric indices"
                    ),
                    "port_neff_tolerance": mmi_neff_tolerance,
                    "port_required_polarization": "local TE",
                    "ideal_split_percent": [50.0, 50.0],
                    "symmetry_tolerance_percent": 1.0,
                    "frequency_points": int(configuration.get("frequency_points", 50)),
                }
                if field_monitor is not None:
                    mmi_analysis["field_monitor_name"] = str(
                        field_monitor.get("name", f"uid_{mmi_uid}_mmi_field")
                    )
    stack = deepcopy(configuration.get("material_stack") or default_stack())
    for row in stack:
        row["thickness_um"] = max(0.0, float(row.get("thickness_um", 0.0)))
        default_etch = row["thickness_um"] if str(row.get("role", "background")).lower() == "geometry" else 0.0
        row["etch_depth_um"] = min(row["thickness_um"], max(0.0, float(row.get("etch_depth_um", default_etch))))
        row["sidewall_angle_deg"] = min(179.999, max(0.001, float(row.get("sidewall_angle_deg", 90.0))))
        raw_layers = row.get("gds_layers", [row.get("gds_layer", 0)])
        if isinstance(raw_layers, (int, float, str)):
            raw_layers = [raw_layers]
        row["gds_layers"] = [int(value) for value in raw_layers]
        row.pop("gds_layer", None)
        row["role"] = "geometry" if str(row.get("role", "background")).lower() == "geometry" else "background"
        row["conformal"] = bool(row.get("conformal", False)) and row["role"] == "background"
        row["slab_extent"] = (
            "geometry"
            if str(row.get("slab_extent", "full")).strip().lower() == "geometry"
            else "full"
        )
        row["mesh_factor"] = max(0.0, float(row.get("mesh_factor", 0.2)))
        row["mesh_order"] = max(
            1,
            int(row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2)),
        )
        if row["thickness_um"] > 0 and not str(row.get("material", "")).strip():
            warnings.append(f"Active stack layer {row.get('name', '')!r} has no material name.")

    project_file = Path(str(configuration.get("project_file", "exported_component.fsp"))).name.strip()
    if not project_file:
        project_file = "exported_component.fsp"
    if not project_file.lower().endswith(".fsp"):
        project_file += ".fsp"
    settings = {
        "dimension": "3D",
        "wavelength_start_um": float(configuration.get("wavelength_start_um", 1.25)),
        "wavelength_stop_um": float(configuration.get("wavelength_stop_um", 1.35)),
        "xy_padding_um": float(configuration.get("xy_padding_um", 2.0)),
        "z_padding_um": float(configuration.get("z_padding_um", 1.0)),
        "domain_padding_um": {
            "x_min": float(configuration.get("domain_padding_um", {}).get("x_min", configuration.get("xy_padding_um", 2.0))),
            "x_max": float(configuration.get("domain_padding_um", {}).get("x_max", configuration.get("xy_padding_um", 2.0))),
            "y_min": float(configuration.get("domain_padding_um", {}).get("y_min", configuration.get("xy_padding_um", 2.0))),
            "y_max": float(configuration.get("domain_padding_um", {}).get("y_max", configuration.get("xy_padding_um", 2.0))),
            "z_min": float(configuration.get("domain_padding_um", {}).get("z_min", configuration.get("z_padding_um", 1.0))),
            "z_max": float(configuration.get("domain_padding_um", {}).get("z_max", configuration.get("z_padding_um", 1.0))),
        },
        "mesh_accuracy": int(configuration.get("mesh_accuracy", 2)),
        "dt_stability_factor": min(0.99, max(0.1, float(configuration.get("dt_stability_factor", 0.99)))),
        "pml_profile": str(configuration.get("pml_profile", "Standard")).strip().title(),
        "pml_geometry_overlap_um": max(0.0, float(configuration.get("pml_geometry_overlap_um", 1.0))),
        "simulation_time_fs": max(1.0, float(configuration.get("simulation_time_fs", 10000.0))),
        "auto_shutoff_min": min(1.0, max(1e-12, float(configuration.get("auto_shutoff_min", 1e-6)))),
        "frequency_points": int(configuration.get("frequency_points", 31)),
        "build_cpu_threads": max(1, int(configuration.get("build_cpu_threads", 30))),
        "resource_mode": configuration.get("resource_mode", "GPU"),
        "tfln_crystal_cut": str(configuration.get("tfln_crystal_cut", "X")).strip().upper(),
        "tfln_temperature_K": float(configuration.get("tfln_temperature_K", 296.3)),
        "include_ports": bool(configuration.get("include_ports", True)),
        "hide_cad": bool(configuration.get("hide_cad", True)),
        "official_gc_domain": bool(configuration.get("official_gc_domain", False)),
        "use_y_antisymmetry": bool(configuration.get("use_y_antisymmetry", False)),
        "antisymmetry_boundary": str(configuration.get("antisymmetry_boundary", "")).strip().lower(),
        "run_after_build": bool(configuration.get("run_after_build", False)),
        "run_gpu_system_check": bool(configuration.get("run_gpu_system_check", False)),
        "show_geometry_preview": bool(configuration.get("show_geometry_preview", True)),
        "show_port_mode_preview": bool(configuration.get("show_port_mode_preview", True)),
        "hpc_pack_duration_minutes": max(
            1, int(configuration.get("hpc_pack_duration_minutes", 30))
        ),
        "hpc_pack_count": max(1, int(configuration.get("hpc_pack_count", 3))),
        "save_inspection_fsp": True,
        "save_final_fsp": True,
        "project_file": project_file,
    }
    if grating_analysis:
        # Match the official Ansys 3D grating-coupler example. Geometry, stack,
        # wavelength limits and GPU selection remain controlled by this export.
        settings.update(
            {
                "mesh_accuracy": 2,
                "dt_stability_factor": 0.99,
                "pml_profile": "Standard",
                "frequency_points": int(configuration.get("frequency_points", 31)),
            }
        )
    if settings["wavelength_stop_um"] <= settings["wavelength_start_um"]:
        warnings.append("Wavelength stop is not above start; edit SETTINGS before running.")

    ports_json: dict[str, dict[str, Any]] = {}
    for index, port in enumerate(ports, start=1):
        key = str(port.get("name") or f"opt_{index}")
        if key in ports_json:
            key = f"uid_{port.get('component_uid', 0)}_{key}"
        ports_json[key] = {
            "dir": str(port.get("dir", "Bidirectional")),
            "loc": float(port.get("loc", 0.5)),
            "name": str(port.get("name") or key),
            "order": float(port.get("order", index)),
            "pos": str(port.get("pos", "Right")),
        }

    export_scope_label = str(configuration.get("scope_label", "Selected export geometry"))
    exported_components = [
        {"uid": int(component.get("uid", 0)), "kind": str(component.get("kind", ""))}
        for component in components
        if component.get("kind") != "E-beam multipass"
    ]
    # Preserve the exact project-side component records that produced this
    # notebook.  This is deliberately separate from EXPORTED_COMPONENTS,
    # which remains the concise human-readable UID/kind summary.
    source_components_json = deepcopy(components)
    payload_cell = (
        "# Embedded export data (layout units are micrometres).\n"
        f"EXPORT_SCOPE_LABEL = {export_scope_label!r}\n"
        f"EXPORTED_COMPONENTS = {pprint.pformat(exported_components, width=120, sort_dicts=False)}\n"
        f"SOURCE_COMPONENTS_JSON = {pprint.pformat(source_components_json, width=160, compact=True, sort_dicts=False)}\n"
        f"SETTINGS = {pprint.pformat(settings, width=120, sort_dicts=False)}\n"
        f"MATERIAL_STACK = {pprint.pformat(stack, width=120, sort_dicts=False)}\n"
        f"BOUNDING_BOX_UM = {pprint.pformat(bbox)}\n"
        f"GEOMETRY = {pprint.pformat(geometry, width=160, compact=True, sort_dicts=False)}\n"
        f"PORTS = {pprint.pformat(ports, width=120, sort_dicts=False)}\n"
        f"FIBER_GEOMETRIES = {pprint.pformat(fiber_geometries, width=120, sort_dicts=False)}\n"
        f"GAUSSIAN_SOURCES = {pprint.pformat(gaussian_sources, width=120, sort_dicts=False)}\n"
        f"PORTS_JSON = {pprint.pformat(ports_json, width=120, sort_dicts=False)}\n"
        f"MONITORS = {pprint.pformat(monitors, width=120, sort_dicts=False)}\n"
        f"GRATING_ANALYSIS = {pprint.pformat(grating_analysis, width=120, sort_dicts=False)}\n"
        f"MMI_ANALYSIS = {pprint.pformat(mmi_analysis, width=120, sort_dicts=False)}\n"
        f"EXPORT_WARNINGS = {pprint.pformat(warnings, width=120)}\n"
        "for warning in EXPORT_WARNINGS:\n"
        "    print('Export note:', warning)\n"
        + _runtime_setup_source(_BUILD_CELL)
    )
    remote_builder_source = _BUILD_CELL
    remote_build_cell = (
        "# Send the complete self-contained model to the already licensed persistent Lambda session.\n"
        f"REMOTE_MODEL_BUILDER = {repr(remote_builder_source)}\n"
        "_remote_payload = (\n"
        "    'REMOTE_WORK = ' + repr(REMOTE_WORK) + '\\n'\n"
        "    + 'EXPORT_SCOPE_LABEL = ' + repr(EXPORT_SCOPE_LABEL) + '\\n'\n"
        "    + 'EXPORTED_COMPONENTS = ' + repr(EXPORTED_COMPONENTS) + '\\n'\n"
        "    + 'SOURCE_COMPONENTS_JSON = ' + repr(SOURCE_COMPONENTS_JSON) + '\\n'\n"
        "    + 'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "    + 'MATERIAL_STACK = ' + repr(MATERIAL_STACK) + '\\n'\n"
        "    + 'BOUNDING_BOX_UM = ' + repr(BOUNDING_BOX_UM) + '\\n'\n"
        "    + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "    + 'PORTS = ' + repr(PORTS) + '\\n'\n"
        "    + 'FIBER_GEOMETRIES = ' + repr(FIBER_GEOMETRIES) + '\\n'\n"
        "    + 'GAUSSIAN_SOURCES = ' + repr(GAUSSIAN_SOURCES) + '\\n'\n"
        "    + 'PORTS_JSON = ' + repr(PORTS_JSON) + '\\n'\n"
        "    + 'MONITORS = ' + repr(MONITORS) + '\\n'\n"
        "    + 'GRATING_ANALYSIS = ' + repr(GRATING_ANALYSIS) + '\\n'\n"
        "    + 'MMI_ANALYSIS = ' + repr(MMI_ANALYSIS) + '\\n'\n"
        "    + 'EXPORT_WARNINGS = ' + repr(EXPORT_WARNINGS) + '\\n'\n"
        ")\n"
        "run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER, 'Build verified 3D model directly', timeout=1800)\n"
        "print('Built one live model directly in memory.')\n"
    )
    geometry_projection_cell = (
        "# Optionally build and display XY, XZ, and YZ projections.\n"
        f"REMOTE_GEOMETRY_PROJECTIONS = {repr(_GEOMETRY_PROJECTIONS_REMOTE)}\n"
        "if SHOW_GEOMETRY_PREVIEW:\n"
        "    run_remote_checked(REMOTE_GEOMETRY_PROJECTIONS, 'Render 3-axis geometry verification', timeout=1800)\n"
        "    GEOMETRY_PROJECTIONS_FILE = REMOTE_WORK.rstrip('/') + '/geometry_xyz_projections.png'\n"
        "    lam.show(GEOMETRY_PROJECTIONS_FILE, width=1400)\n"
        "else:\n"
        "    print('3-axis preview skipped by cell 1; the embedded geometry is unchanged.')\n"
    )
    port_mode_profiles_cell = (
        "# Optionally display modes already selected during the required model build.\n"
        f"REMOTE_PORT_MODE_PROFILES = {repr(_PORT_MODE_PROFILES_REMOTE)}\n"
        "if SHOW_PORT_MODE_PREVIEW:\n"
        "    run_remote_checked(REMOTE_PORT_MODE_PROFILES, 'Render selected port Ex and Ey fields', timeout=1800)\n"
        "    if PORTS:\n"
        "        PORT_MODE_PROFILES_FILE = REMOTE_WORK.rstrip('/') + '/port_mode_Ex_Ey.png'\n"
        "        lam.show(PORT_MODE_PROFILES_FILE, width=1400)\n"
        "        _port_polarization_valid = bool(lam.get('PORT_POLARIZATION_VALID'))\n"
        "        _port_mode_confinement_valid = bool(lam.get('PORT_MODE_CONFINEMENT_VALID'))\n"
        "        _port_mode_valid = bool(lam.get('PORT_MODE_VALID'))\n"
        "        _port_polarization_report = lam.get('PORT_POLARIZATION_REPORT')\n"
        "        print('Port transverse-polarization report:', _port_polarization_report)\n"
        "        if not _port_mode_valid:\n"
        "            raise RuntimeError('Mode validation failed: the fiber source must be local-TE and the neff-selected waveguide mode must decay below 5% at every expansion-monitor boundary. Correct the target neff or monitor span before running the GPU solve.')\n"
        "    else:\n"
        "        print('No FDTD ports were exported, so there are no port modes to display.')\n"
        "else:\n"
        "    print('Port Ex/Ey images skipped by cell 1. Required local-TE selection and neff checks already ran during model construction.')\n"
    )
    resource_save_cell = (
        "# Configure the licensed resource and save the required inspection FSP.\n"
        f"REMOTE_RESOURCE_AND_SAVE = {repr(_REMOTE_RESOURCE_AND_SAVE)}\n"
        "run_remote_checked(REMOTE_RESOURCE_AND_SAVE, 'Configure resources and save inspection project', timeout=1800)\n"
        "_RESOURCE_STATE = lam.get(\"{'project_file': REMOTE_PROJECT_FILE, 'inspection_file': REMOTE_INSPECTION_PROJECT_FILE, 'inspection_saved': bool(REMOTE_INSPECTION_FSP_SAVED)}\")\n"
        "REMOTE_PROJECT_FILE = str(_RESOURCE_STATE['project_file'])\n"
        "REMOTE_INSPECTION_PROJECT_FILE = str(_RESOURCE_STATE['inspection_file'])\n"
        "REMOTE_INSPECTION_FSP_SAVED = bool(_RESOURCE_STATE['inspection_saved'])\n"
        "PIRIS_FSP_DIR.mkdir(parents=True, exist_ok=True)\n"
        "if not REMOTE_INSPECTION_FSP_SAVED:\n"
        "    raise RuntimeError('The required pre-solve inspection FSP was not saved')\n"
        "LOCAL_INSPECTION_PROJECT_FILE = PIRIS_FSP_DIR / os.path.basename(REMOTE_INSPECTION_PROJECT_FILE)\n"
        "FETCHED_INSPECTION_PROJECT_FILE = lam.fetch(REMOTE_INSPECTION_PROJECT_FILE, str(LOCAL_INSPECTION_PROJECT_FILE))\n"
        "if not LOCAL_INSPECTION_PROJECT_FILE.is_file() or LOCAL_INSPECTION_PROJECT_FILE.stat().st_size <= 0:\n"
        "    raise RuntimeError('The required inspection .fsp could not be downloaded: ' + str(LOCAL_INSPECTION_PROJECT_FILE))\n"
        "print('saved pre-solve inspection project ->', FETCHED_INSPECTION_PROJECT_FILE)\n"
    )
    review_project_cell = (
        "# Inspect the always-saved exact pre-solve FSP.\n"
        "from IPython.display import FileLink, display\n"
        "display(FileLink(str(LOCAL_INSPECTION_PROJECT_FILE)))\n"
        "print('Open the linked .fsp with Lumerical FDTD on a computer that has the GUI installed.')\n"
        "OPEN_REMOTE_LUMERICAL_GUI = False\n"
        "if OPEN_REMOTE_LUMERICAL_GUI:\n"
        "    _remote_display = str(lam.get(\"os.environ.get('DISPLAY', '')\") or '')\n"
        "    if _remote_display:\n"
        "        run_remote_checked('fdtd.switchtolayout(); fdtd.show(); print(\"Remote GUI requested.\")', 'Open remote Lumerical GUI', timeout=120)\n"
        "    else:\n"
        "        print('Lambda is headless: no DISPLAY is available. Use the downloaded FSP link above.')\n"
    )
    solve_cell = (
        "# Run the live in-memory model. GPU remains the default for every 3D solve.\n"
        f"REMOTE_SWITCH_TO_CPU_ANALYSIS = {repr(_SWITCH_TO_CPU_ANALYSIS_REMOTE)}\n"
        "if SETTINGS.get('run_after_build', False):\n"
        "    _resource_mode = str(SETTINGS.get('resource_mode', 'GPU')).strip().upper()\n"
        "    if _resource_mode == 'GPU':\n"
        "        _solve_code = 'fdtd.run(\"FDTD\", \"GPU\")'\n"
        "    elif _resource_mode == 'CPU':\n"
        "        _solve_code = 'fdtd.run(\"FDTD\", \"CPU\")'\n"
        "    else:\n"
        "        raise ValueError('resource_mode must be GPU or CPU')\n"
        "    solve_remote_checked(_solve_code, label='Max Layout 3D FDTD [' + _resource_mode + ']', timeout=21600)\n"
        "    run_remote_checked(REMOTE_SWITCH_TO_CPU_ANALYSIS, 'Switch post-processing to CPU', timeout=300)\n"
        "    print(\"Simulation finished. The solved project will be saved in the next dedicated FSP section.\")\n"
        "else:\n"
        "    print(\"Run is disabled. The verified live model was built but not solved.\")\n"
    )
    final_fsp_cell = (
        "# Save and fetch the solved model before any optional analysis can fail.\n"
        "REMOTE_FINAL_FSP_SAVE = '''\n"
        "save_verified_project(REMOTE_PROJECT_FILE)\n"
        "REMOTE_FINAL_FSP_SAVED = True\n"
        "print('Saved solved/final FSP:', REMOTE_PROJECT_FILE)\n"
        "'''\n"
        "run_remote_checked(REMOTE_FINAL_FSP_SAVE, 'Save solved/final project', timeout=1800)\n"
        "PIRIS_FSP_DIR.mkdir(parents=True, exist_ok=True)\n"
        "LOCAL_FINAL_PROJECT_FILE = PIRIS_FSP_DIR / os.path.basename(REMOTE_PROJECT_FILE)\n"
        "FETCHED_FINAL_PROJECT_FILE = lam.fetch(REMOTE_PROJECT_FILE, str(LOCAL_FINAL_PROJECT_FILE))\n"
        "if not LOCAL_FINAL_PROJECT_FILE.is_file() or LOCAL_FINAL_PROJECT_FILE.stat().st_size <= 0:\n"
        "    raise RuntimeError('The solved/final .fsp could not be downloaded into the project fsp folder: ' + str(LOCAL_FINAL_PROJECT_FILE))\n"
        "print('saved solved/final project ->', FETCHED_FINAL_PROJECT_FILE)\n"
    )
    save_results_cell = (
        "# Serialize model results while the FDTD licence and remote session are still active.\n"
        f"REMOTE_RESULTS_SAVER = {repr(_SAVE_REMOTE_RESULTS)}\n"
        "run_remote_checked(REMOTE_RESULTS_SAVER, 'Save numerical result bundle', timeout=1800)\n"
    )
    grating_analysis_cell = (
        "# Extract the small spectrum remotely, then plot it locally without remote Matplotlib startup or PNG transfer.\n"
        f"REMOTE_GRATING_ANALYSIS = {repr(_GRATING_ANALYSIS_REMOTE)}\n"
        "run_remote_checked(REMOTE_GRATING_ANALYSIS, 'Grating coupling-efficiency analysis', timeout=1800)\n"
        "if SETTINGS.get('run_after_build', False):\n"
        "    import numpy as np\n"
        "    import matplotlib.pyplot as plt\n"
        "    from IPython.display import Image, display\n"
        "    _remote_grating_npz = REMOTE_WORK + '/grating_analysis.npz'\n"
        "    _local_grating_npz = PIRIS_RESULTS_DIR / 'grating_analysis.npz'\n"
        "    lam.fetch(_remote_grating_npz, str(_local_grating_npz))\n"
        "    with np.load(_local_grating_npz) as _grating_data:\n"
        "        _wavelength_nm = np.asarray(_grating_data['wavelength_m']) * 1e9\n"
        "        _coupling_linear = np.asarray(_grating_data['fiber_coupling'])\n"
        "        _coupling_db = np.asarray(_grating_data['fiber_coupling_db'])\n"
        "        _waveguide_mode_power = np.asarray(_grating_data['waveguide_mode_power_source_normalized'])\n"
        "        _waveguide_total_power = np.asarray(_grating_data['waveguide_total_power'])\n"
        "        _fiber_input_power = np.asarray(_grating_data['fiber_input_power'])\n"
        "        _waveguide_total_transmission = np.asarray(_grating_data['waveguide_total_transmission'])\n"
        "        _waveguide_total_transmission_db = np.asarray(_grating_data['waveguide_total_transmission_db'])\n"
        "        _selected_neff = float(np.asarray(_grating_data['waveguide_selected_neff']).ravel()[0])\n"
        "        _selected_mode_number = int(np.asarray(_grating_data['waveguide_selected_mode_number']).ravel()[0])\n"
        "        _target_nm = float(np.asarray(_grating_data['target_wavelength_m']).ravel()[0]) * 1e9\n"
        "    _local_response_png = PIRIS_RESULTS_DIR / 'grating_response.png'\n"
        "    _target_index = int(np.argmin(np.abs(_wavelength_nm - _target_nm)))\n"
        "    _input_label = ('Gaussian-beam input' if str(GRATING_ANALYSIS.get('excitation_type', 'fiber_mode')) == 'gaussian_beam' else 'fiber-mode input')\n"
        "    print('Target wavelength: %.3f nm' % _wavelength_nm[_target_index])\n"
        "    print('Measured %s power: %.8g' % (_input_label, _fiber_input_power[_target_index]))\n"
        "    print('Waveguide total power: %.8g' % _waveguide_total_power[_target_index])\n"
        "    print('Selected waveguide mode: %d, neff %.8g' % (_selected_mode_number, _selected_neff))\n"
        "    print('Source-normalized selected waveguide-mode power: %.8g' % _waveguide_mode_power[_target_index])\n"
        "    print('Selected-TE / measured input (linear): %.8g' % _coupling_linear[_target_index])\n"
        "    print('Total waveguide / measured input (linear): %.8g' % _waveguide_total_transmission[_target_index])\n"
        "    _figure, (_waveguide_axis, _waveguide_db_axis) = plt.subplots(1, 2, figsize=(13.0, 4.9), sharex=True)\n"
        "    _waveguide_axis.plot(_wavelength_nm, _coupling_linear, lw=2.6, color='#7c3aed', label='selected TE / measured input')\n"
        "    _waveguide_axis.plot(_wavelength_nm, _waveguide_total_transmission, lw=2.2, color='#f59e0b', label='total waveguide power / measured input')\n"
        "    _waveguide_axis.set(xlabel='wavelength [nm]', ylabel='normalized linear power', title='Waveguide transmission — linear')\n"
        "    _waveguide_db_axis.plot(_wavelength_nm, _coupling_db, lw=2.6, color='#7c3aed', label='selected TE / input')\n"
        "    _waveguide_db_axis.plot(_wavelength_nm, _waveguide_total_transmission_db, lw=2.2, color='#f59e0b', label='total / input')\n"
        "    _waveguide_db_axis.set(xlabel='wavelength [nm]', ylabel='normalized power [dB]', title='Waveguide transmission — dB')\n"
        "    for _axis in (_waveguide_axis, _waveguide_db_axis):\n"
        "        _axis.axvline(_target_nm, color='#64748b', ls='--', lw=1.0)\n"
        "        _axis.grid(alpha=0.3)\n"
        "        _axis.legend(loc='best')\n"
        "    _figure.tight_layout()\n"
        "    _figure.savefig(_local_response_png, dpi=160, bbox_inches='tight')\n"
        "    plt.close(_figure)\n"
        "    display(Image(filename=str(_local_response_png), width=1050))\n"
        "    print('saved ->', _local_grating_npz)\n"
        "    print('saved ->', _local_response_png)\n"
    )
    mmi_analysis_cell = (
        "# Plot MMI output/input transmission, secondary split symmetry, and solved Ex/Ey fields.\n"
        f"REMOTE_MMI_ANALYSIS = {repr(_MMI_ANALYSIS_REMOTE)}\n"
        "run_remote_checked(REMOTE_MMI_ANALYSIS, 'MMI splitting-ratio analysis', timeout=1800)\n"
        "if SETTINGS.get('run_after_build', False):\n"
        "    lam.show(REMOTE_WORK + '/mmi_splitting_ratio.png', width=1000)\n"
        "    lam.show(REMOTE_WORK + '/mmi_field_distribution.png', width=1100)\n"
    )
    active_count = sum(float(row.get("thickness_um", 0.0)) > 0 for row in stack)
    exported_component_text = ", ".join(
        f"{item['kind']} (UID {item['uid']})" for item in exported_components
    ) or "none"
    empty_geometry_warning = (
        "\n> **Stop before solving:** this scope contains no physical device polygons. "
        "It can visualize standalone ports/monitors, but it cannot produce a device response. "
        "Re-export with a device-containing scope.\n"
        if not geometry else ""
    )
    if grating_analysis and str(
        grating_analysis.get("excitation_type", "fiber_mode")
    ) == "gaussian_beam":
        grating_excitation_note = (
            "- This grating uses one analytic Gaussian beam source (not an FDTD port) with "
            "S/local-TE polarization. The access-waveguide FDTD port remains passive. A separate "
            "Z-normal power monitor measures incident input power, and selected-TE plus total "
            "waveguide spectra are normalized to that measured input."
        )
        grating_section_description = (
            "The analytic Gaussian beam is the only source. A non-modal Z-normal power monitor "
            "below it measures actual incident power. The passive waveguide FDTD port reports "
            "fundamental-TE power, and a nearby power monitor reports total waveguide power. "
            "Both outputs are divided by measured Gaussian-beam input power."
        )
    else:
        grating_excitation_note = (
            "- A fiber-mode grating export contains exactly two modal ports: the active tilted "
            "fiber source and the passive access-waveguide receiver. The first three fiber modes "
            "are solved and the rotation-aware local-TE member of the near-degenerate HE11 pair is excited."
        )
        grating_section_description = (
            "The tilted fiber FDTD port is the only source. A non-modal Z-normal power monitor "
            "below it measures actual incident power. The passive waveguide FDTD port reports "
            "fundamental-TE power, and a nearby power monitor reports total waveguide power. "
            "Both outputs are divided by measured fiber-mode input power."
        )
    intro = f"""# Max Layout → Lumerical FDTD notebook

This notebook contains **{len(geometry)} embedded polygons**, **{len(ports)} standard FDTD ports**, **{len(fiber_geometries)} fiber geometry groups**, **{len(gaussian_sources)} Gaussian sources**, **{len(monitors)} monitors**, and **{active_count} active material layers**. It is self-contained: no GDS sidecar is required.

**Export scope:** {export_scope_label}  
**Included objects:** {exported_component_text}
{empty_geometry_warning}

The notebook follows the same licence lifecycle as `TFLN_GC_1310.ipynb`: connect to Lambda, seed Ansys Shared Web, roam the configurable HPC Pack count, build/run in that licensed session, save and fetch results, close FDTD, return the roamed packs, and finally close SSH. Lumerical is not required on the local Mac.

- Every exported simulation is a **3D FDTD simulation**. A saved 2D preference is ignored; GPU is the default compute resource and is selected explicitly at solve time.
- First-cell switches render **XY, XZ, and YZ geometry projections** and selected-port **|Ex|/|Ey| fields**. They default on and remain editable; neither changes the simulated model. Required rotation-aware local-TE selection and effective-index checks still run during construction.
- Stack rows are ordered bottom-to-top.
- A material thickness of **0 µm means that material is absent**.
- Etch depth **0 µm** keeps an unetched film; etch depth equal to film thickness creates a fully etched patterned layer.
- Exported cross-section rows use Lumerical Layer Builder, including the selected waveguide sidewall angle (90° is vertical).
- A partially etched cross-section can keep its unetched slab across the full FDTD plane or restrict it to the selected GDS geometry footprint.
- Each stack row has a dimensionless mesh factor: **0 / Automatic** leaves meshing to Lumerical's FDTD mesh-accuracy setting, while a positive value produces an isotropic step of `factor × λ₀/nmax` at the shortest simulated wavelength (anisotropic media use their largest index component). New grating-coupler exports default to Automatic.
- Surface monitors carry explicit x/y/z spans; their normal-axis span is zero (Z is the into-page axis in the layout view).
- Ports are manual simulation-only objects from the left **Ports & monitors** library. No component, including a grating coupler, creates a default port automatically.
- `PORTS_JSON` uses the exact compact-model structure and field names from the reviewed Lumerical JSON examples: `name`, `dir`, `loc`, `pos`, and `order`.
- Every port becomes a standard Ansys FDTD `addport` object. Fiber is a separate geometry group containing the official example's tilted 9 µm core and 50 µm cladding; a manually placed Z-axis FDTD port passes through that geometry.
- Placed power, mode-expansion, and field-profile monitors become `addpower`, `addmodeexpansion`, and `addprofile` objects.
- The fiber geometry and its Z-axis FDTD port have independent positions and heights above the exported device. Grating-coupler exit analysis is centered on the placed fiber-axis FDTD port.
- A conformal cladding row is a continuous full-domain gap-fill volume: it fills every etched opening and covers every waveguide/grating polygon, flare, terminal arc, and extension. Lower mesh-order device material wins wherever the volumes overlap.
- The FDTD boundary keeps at least λ/4 clearance from ordinary device features. Background films may extend through their PMLs, but ports never create additional waveguide-to-PML geometry; only the polygons exported from the layout are simulated.
- `LiNbO3` is created as a frequency- and temperature-dependent anisotropic sampled material using the Zelmon/Moretti model and the selected X/Y/Z crystal cut.
{grating_excitation_note} In either excitation mode, the waveguide receiver selects the fundamental TE mode using the stack-derived effective-index check, while a nearby power monitor measures total waveguide flux. Both selected-TE and total-waveguide spectra are normalized to measured input and plotted in linear and dB units.
- A 1×2 MMI export launches mode 1 from its input port, measures input power 2 µm before the input taper, plots both output powers relative to that measured input, and plots the normalized longitudinal |E|² distribution through the complete MMI.
- GPU and CPU modes are selectable through `SETTINGS['resource_mode']`; GPU is the default for every 3D export.
- The first cell exposes run, diagnostic-preview, and GPU-system-check switches. Project saving is mandatory: one inspection FSP is written before the solve and one solved/best FSP is written afterward.
- Run the final release cell even after an interrupted simulation so the FDTD licence and roamed HPC Packs are returned.
"""
    notebook = {
        "cells": [
            _notebook_cell("code", _quick_run_options_cell(settings, workflow="single run")),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Connect to Lambda\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 2 · Acquire Ansys Shared Web licences\n\nSeed the headless sign-in and roam `HPC_PACK_COUNT` Shared Web HPC Packs for the number of minutes selected by `HPC_PACK_DURATION_MINUTES` in cell 1. The H100 launcher defaults this to four.\n"),
            _notebook_cell("code", _LICENSE_CHECKOUT_CELL),
            _notebook_cell("markdown", "## 3 · Embedded layout, stack, ports, and monitors\n"),
            _notebook_cell("code", payload_cell),
            _notebook_cell("markdown", "## 4 · Build the model inside the licensed Lambda session\n"),
            _notebook_cell("code", remote_build_cell),
            _notebook_cell("markdown", "## 5 · XY, XZ, and YZ geometry preview\n\n`SHOW_GEOMETRY_PREVIEW` defaults to `True` in cell 1 and may be disabled when the diagnostic image is not needed.\n"),
            _notebook_cell("code", geometry_projection_cell),
            _notebook_cell("markdown", "## 6 · Selected-port Ex and Ey images\n\n`SHOW_PORT_MODE_PREVIEW` defaults to `True` in cell 1 and may be disabled to skip only these diagnostic maps. Required polarization and effective-index checks always run.\n"),
            _notebook_cell("code", port_mode_profiles_cell),
            _notebook_cell("markdown", "## 7 · Configure resources and save the inspection FSP\n\nThe GPU/CPU resources are configured, then the exact pre-solve FSP is always created and downloaded.\n"),
            _notebook_cell("code", resource_save_cell),
            _notebook_cell("markdown", "## 8 · Inspect the saved pre-solve FSP\n\nThis section links the exact verified project that was saved before the simulation.\n"),
            _notebook_cell("code", review_project_cell),
            _notebook_cell("markdown", "## 9 · Run the live 3D model\n"),
            _notebook_cell("code", solve_cell),
            _notebook_cell("markdown", "## 10 · Save and download the solved FSP\n\nThis verified project is placed in the project `fsp` folder before coupling/splitting analysis begins.\n"),
            _notebook_cell("code", final_fsp_cell),
            *(
                [
                    _notebook_cell("markdown", "## 10 · Grating coupling efficiency\n\n" + grating_section_description + " The spectra are displayed in linear and dB units.\n"),
                    _notebook_cell("code", grating_analysis_cell),
                ]
                if grating_analysis
                else []
            ),
            *(
                [
                    _notebook_cell("markdown", "## 10 · MMI output/input transmission, split symmetry, and Ex/Ey fields\n\nMode 1 is launched from the input FDTD port. A power monitor 2 µm before the input taper measures the actual incident power. The primary graph is each output divided by that measured input, with total output/input alongside it. A separate secondary graph shows only the 50/50 symmetry fraction, and a Z-normal monitor plots solved |Ex| and |Ey| along the complete MMI length.\n"),
                    _notebook_cell("code", mmi_analysis_cell),
                ]
                if mmi_analysis
                else []
            ),
            _notebook_cell("markdown", "## 11 · Save numerical results before releasing licences\n"),
            _notebook_cell("code", save_results_cell),
            _notebook_cell("markdown", "## 12 · Fetch requested FSP files and compact results\n"),
            _notebook_cell("code", _FETCH_RESULTS_CELL),
            _notebook_cell("markdown", "## 13 · Release FDTD and return all roamed HPC Packs\n\nAlways run this cell, including after an interrupted solve.\n"),
            _notebook_cell("code", _RELEASE_LICENSES_CELL),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "max_layout": {
                "export": "lumerical-fdtd",
                "units": "um",
                "dimension": "3D",
                "execution": "lambda-a100-persistent-ssh",
                "license_lifecycle": "shared-web-3-hpc-packs-save-fetch-release",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, warnings


def _notebook_literal_assignments(notebook: dict[str, Any]) -> dict[str, Any]:
    """Read the generated embedded-data cell without executing notebook code."""
    wanted = {
        "EXPORT_SCOPE_LABEL", "EXPORTED_COMPONENTS", "SOURCE_COMPONENTS_JSON",
        "SETTINGS", "MATERIAL_STACK",
        "BOUNDING_BOX_UM", "GEOMETRY", "PORTS", "FIBER_GEOMETRIES",
        "GAUSSIAN_SOURCES", "PORTS_JSON",
        "MONITORS", "GRATING_ANALYSIS", "MMI_ANALYSIS", "EXPORT_WARNINGS",
    }
    result: dict[str, Any] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "MATERIAL_STACK =" not in source or "GEOMETRY =" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                if name in wanted:
                    result[name] = ast.literal_eval(node.value)
        break
    missing = sorted(wanted - result.keys())
    if missing:
        raise RuntimeError("Generated Lumerical payload is missing assignments: " + ", ".join(missing))
    return result


def _sweep_fixed_origin_um(
    sweep_cases: list[dict[str, Any]], included_layers: set[tuple[int, int]]
) -> list[float]:
    points = []
    for case in sweep_cases:
        for component in case["components"]:
            if component.get("kind") in SIMULATION_COMPONENT_KINDS or component.get("kind") == "E-beam multipass":
                continue
            polygons, _ports = component_geometry_arrays(component)
            points.extend(
                np.asarray(vertices, dtype=float)
                for vertices, layer, datatype in polygons
                if (int(layer), int(datatype)) in included_layers
            )
    if not points:
        return [0.0, 0.0]
    all_points = np.vstack(points)
    origin = 0.5 * (np.min(all_points, axis=0) + np.max(all_points, axis=0))
    return [float(origin[0]), float(origin[1])]


def _payload_xy_extent(payload: dict[str, Any]) -> list[float]:
    """Union geometry and movable simulation-plane extents for a fixed sweep domain."""
    bbox = list(map(float, payload["BOUNDING_BOX_UM"]))
    x_min, y_min, x_max, y_max = bbox
    for item in [
        *payload["PORTS"], *payload["MONITORS"],
        *payload.get("GAUSSIAN_SOURCES", []),
    ]:
        x_um, y_um = map(float, item.get("center", (0.0, 0.0)))
        normal = str(item.get("plane normal", "X")).upper()
        if normal != "Z":
            distance_um = float(item.get("distance_um", 0.0))
            angle_deg = float(item.get("outward_orientation_deg", item.get("orientation_deg", 0.0)))
            x_um += distance_um * math.cos(math.radians(angle_deg))
            y_um += distance_um * math.sin(math.radians(angle_deg))
        fallback_span = max(0.0, float(item.get("span_um", 2.0)))
        x_span = max(0.0, float(item.get("x span", 0.0 if normal == "X" else fallback_span)))
        y_span = max(0.0, float(item.get("y span", 0.0 if normal == "Y" else fallback_span)))
        x_min = min(x_min, x_um - 0.5 * x_span)
        x_max = max(x_max, x_um + 0.5 * x_span)
        y_min = min(y_min, y_um - 0.5 * y_span)
        y_max = max(y_max, y_um + 0.5 * y_span)
    return [x_min, y_min, x_max, y_max]


def _payload_z_plane_extent(payload: dict[str, Any]) -> list[float] | None:
    """Return the exported Z-plane envelope for one hot-swap sweep case."""
    levels = _stack_vertical_levels(list(payload.get("MATERIAL_STACK", [])))
    positions: list[float] = []
    for item in [
        *payload.get("PORTS", []),
        *payload.get("GAUSSIAN_SOURCES", []),
        *payload.get("MONITORS", []),
    ]:
        if str(item.get("plane normal", item.get("injection axis", "X"))).upper() != "Z":
            continue
        z_um = _item_vertical_reference(item, levels) + float(
            item.get("distance_um", 0.0)
        )
        if math.isfinite(z_um):
            positions.append(float(z_um))
    if not positions:
        return None
    return [min(positions), max(positions)]


def _format_sweep_value(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else format(float(value), ".9g")


def _sweep_case_label(spec: dict[str, Any], values: dict[str, int | float]) -> tuple[str, str]:
    parts = []
    for axis in spec["axes"]:
        key = str(axis["parameter"])
        parts.append(f"{axis['short_name']}={_format_sweep_value(values[key])}")
    label = ", ".join(parts)
    component_kind = str(spec.get("component_kind", ""))
    prefix = (
        "CE" if component_kind in {"Grating coupler", "GC-SOI"}
        else "MMI" if component_kind == "1x2 MMI"
        else "Sweep"
    )
    return label, prefix + "-" + "-".join(part.replace(" ", "") for part in parts)


def _sweep_requires_mode_refresh(
    spec: dict[str, Any], grating_analysis: dict[str, Any] | None = None
) -> bool:
    if str(spec.get("component_kind", "")) == "1x2 MMI":
        # Body width, taper width/length, MMI length and output separation do
        # not change the cross-section at the three access-port planes.  Only
        # the access-waveguide width (or an explicit material cross-section
        # axis) needs the embedded port modes recomputed for every point.
        return any(
            str(axis["parameter"]).lower() == "wg_width"
            or any(
                token in str(axis["parameter"]).lower()
                for token in ("height", "thickness", "index", "cross_section")
            )
            for axis in spec["axes"]
        )
    excitation_type = str(
        dict(grating_analysis or {}).get("excitation_type", "fiber_mode")
    ).strip().lower()
    mode_sensitive = (
        "width", "gap", "height", "thickness", "index", "diameter",
        "cross_section",
    )
    if excitation_type != "gaussian_beam":
        # A tilted fiber-mode source changes its embedded eigenmode.  An
        # analytic Gaussian source and ordinary Pin monitor only move/rotate;
        # the passive access-waveguide receiver cross-section is unchanged.
        mode_sensitive += ("angle_theta",)
    return any(
        any(token in str(axis["parameter"]).lower() for token in mode_sensitive)
        for axis in spec["axes"]
    )


def generate_lumerical_sweep_notebook(
    sweep_cases: list[dict[str, Any]],
    configuration: dict[str, Any],
    sweep_spec: dict[str, Any],
    nominal_components: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Generate one-session, geometry-hot-swap Lumerical sweep notebook."""
    if not sweep_cases:
        raise ValueError("A Lumerical sweep needs at least one simulation case")
    if len(sweep_cases) != int(sweep_spec.get("point_count", -1)):
        raise ValueError("Sweep case count does not match the normalized sweep specification")
    target_uid = int(sweep_spec["component_uid"])
    expected_points = expand_lumerical_sweep_points(sweep_spec)
    actual_points = [dict(case.get("values", {})) for case in sweep_cases]
    if actual_points != expected_points:
        raise ValueError("Sweep cases are not in the normalized Cartesian order")

    baseline_components = (
        nominal_components
        if nominal_components is not None
        else sweep_cases[0]["components"]
    )
    if not any(
        int(component.get("uid", -1)) == target_uid
        for component in baseline_components
    ):
        raise ValueError("The nominal export components do not contain the swept component")
    included_raw = configuration.get("included_layers") or available_geometry_layers(
        baseline_components
    )
    included = {(int(value[0]), int(value[1])) for value in included_raw}
    fixed_origin_um = _sweep_fixed_origin_um(
        (
            [{"components": baseline_components}]
            if nominal_components is not None
            else sweep_cases
        ),
        included,
    )
    payloads: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    def generated_payload(
        components: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        case_configuration = deepcopy(configuration)
        case_configuration["_fixed_origin_um"] = fixed_origin_um
        case_configuration["_sweep_export"] = True
        case_configuration["run_after_build"] = True
        notebook, warnings = generate_lumerical_notebook(components, case_configuration)
        payload = _notebook_literal_assignments(notebook)
        # Sweeps collect compact power spectra only. Longitudinal/full-field
        # profile monitors can dominate memory and transfer time and are not
        # needed for CE, splitting, or port/power-monitor objectives.
        payload["MONITORS"] = [
            monitor for monitor in payload["MONITORS"]
            if str(monitor.get("monitor_kind", "")) != "Field profile monitor"
        ]
        return payload, warnings

    nominal_payload: dict[str, Any] | None = None
    if nominal_components is not None:
        nominal_payload, warnings = generated_payload(nominal_components)
        all_warnings.extend(warnings)
    for case in sweep_cases:
        payload, warnings = generated_payload(case["components"])
        payloads.append(payload)
        all_warnings.extend(warnings)

    base = nominal_payload or payloads[0]
    invariant_names = {
        "ports": [str(item.get("name", "")) for item in base["PORTS"]],
        "fibers": [str(item.get("name", "")) for item in base["FIBER_GEOMETRIES"]],
        "gaussian_sources": [
            str(item.get("name", "")) for item in base.get("GAUSSIAN_SOURCES", [])
        ],
        "monitors": [str(item.get("name", "")) for item in base["MONITORS"]],
    }
    layer_keys = {(int(item["layer"]), int(item.get("datatype", 0))) for item in base["GEOMETRY"]}
    payloads_to_validate = (
        payloads if nominal_payload is not None else payloads[1:]
    )
    validation_start = 1 if nominal_payload is not None else 2
    for index, payload in enumerate(payloads_to_validate, start=validation_start):
        names = {
            "ports": [str(item.get("name", "")) for item in payload["PORTS"]],
            "fibers": [str(item.get("name", "")) for item in payload["FIBER_GEOMETRIES"]],
            "gaussian_sources": [
                str(item.get("name", ""))
                for item in payload.get("GAUSSIAN_SOURCES", [])
            ],
            "monitors": [str(item.get("name", "")) for item in payload["MONITORS"]],
        }
        if names != invariant_names:
            raise ValueError(
                f"Sweep point {index} changes the number or names of ports/monitors. "
                "Use a range that preserves the component topology."
            )
        point_layer_keys = {(int(item["layer"]), int(item.get("datatype", 0))) for item in payload["GEOMETRY"]}
        if point_layer_keys != layer_keys:
            raise ValueError(
                f"Sweep point {index} changes the exported GDS layer set; fast in-place sweeps require invariant layers."
            )
        if payload["MATERIAL_STACK"] != base["MATERIAL_STACK"]:
            raise ValueError("Fast sweeps cannot change the material-stack topology between points")

    extent_payloads = (
        [base, *payloads] if nominal_payload is not None else payloads
    )
    extents = [_payload_xy_extent(payload) for payload in extent_payloads]
    union_bbox = [
        min(extent[0] for extent in extents),
        min(extent[1] for extent in extents),
        max(extent[2] for extent in extents),
        max(extent[3] for extent in extents),
    ]
    static_geometry = [
        item for item in base["GEOMETRY"]
        if int(item.get("component_uid", -1)) != target_uid
    ]

    def source_parameters(payload: dict[str, Any]) -> dict[str, Any]:
        target = next(
            (
                component
                for component in payload["SOURCE_COMPONENTS_JSON"]
                if int(component.get("uid", -1)) == target_uid
            ),
            None,
        )
        if target is None:
            raise ValueError(
                "The embedded source JSON does not contain the swept component"
            )
        return deepcopy(dict(target.get("params", {})))

    nominal_parameters = source_parameters(base)
    remote_cases = []
    complete_signatures = []
    for case, payload in zip(sweep_cases, payloads):
        target_geometry = [
            item for item in payload["GEOMETRY"]
            if int(item.get("component_uid", -1)) == target_uid
        ]
        if not target_geometry:
            raise ValueError("The swept component has no exported geometry on the selected layers")
        display_label, result_stem = _sweep_case_label(sweep_spec, case["values"])
        remote_case = {
            "values": dict(case["values"]),
            "source_parameters": source_parameters(payload),
            "display_label": display_label,
            "result_stem": result_stem,
            "target_geometry": target_geometry,
            "ports": payload["PORTS"],
            "fiber_geometries": payload["FIBER_GEOMETRIES"],
            "gaussian_sources": payload.get("GAUSSIAN_SOURCES", []),
            "monitors": payload["MONITORS"],
            "grating_analysis": payload["GRATING_ANALYSIS"],
            "mmi_analysis": payload["MMI_ANALYSIS"],
        }
        remote_cases.append(remote_case)
        complete_signatures.append(json.dumps(
            {
                "target_geometry": target_geometry,
                "ports": payload["PORTS"],
                "fiber_geometries": payload["FIBER_GEOMETRIES"],
                "gaussian_sources": payload.get("GAUSSIAN_SOURCES", []),
                "monitors": payload["MONITORS"],
                "grating_analysis": payload["GRATING_ANALYSIS"],
                "mmi_analysis": payload["MMI_ANALYSIS"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ))

    for axis in sweep_spec["axes"]:
        parameter = str(axis["parameter"])
        other_parameters = [
            str(other_axis["parameter"])
            for other_axis in sweep_spec["axes"]
            if str(other_axis["parameter"]) != parameter
        ]
        signatures_by_other_values: dict[tuple[float, ...], list[str]] = {}
        for case, signature in zip(remote_cases, complete_signatures):
            other_values = tuple(float(case["values"][name]) for name in other_parameters)
            signatures_by_other_values.setdefault(other_values, []).append(signature)
        if not any(
            len(set(signatures)) > 1
            for signatures in signatures_by_other_values.values()
        ):
            raise ValueError(
                f"{axis['label']} does not change the exported geometry or linked simulation setup over this range."
            )

    settings = deepcopy(base["SETTINGS"])
    settings["run_after_build"] = True
    settings["sweep_mode"] = True
    settings["save_each_fsp"] = False
    z_plane_extents = [
        extent for extent in (
            _payload_z_plane_extent(payload) for payload in extent_payloads
        )
        if extent is not None
    ]
    if z_plane_extents:
        # The nominal model is built only once.  Reserve its fixed Z domain
        # for every source/monitor height reached by an angle sweep before any
        # in-memory hot swap moves those planes.
        settings["sweep_sampling_z_bounds_um"] = [
            min(extent[0] for extent in z_plane_extents),
            max(extent[1] for extent in z_plane_extents),
        ]
    project_name = Path(str(settings.get("project_file", "lumerical_sweep.fsp"))).stem
    if not project_name.endswith("_sweep"):
        project_name += "_sweep"
    settings["project_file"] = project_name + ".fsp"
    # The downloadable inspection FSP is the untouched current-project model,
    # never an arbitrary endpoint from the sweep range.
    base_geometry = deepcopy(base["GEOMETRY"])
    sweep_code_fingerprint = hashlib.sha256(
        "\0".join((
            _BUILD_CELL,
            _REMOTE_RESOURCE_AND_SAVE,
            _SWEEP_RUNTIME_REMOTE,
            _SWEEP_RUNNER_REMOTE,
            _MULTIGPU_SINGLE_CASE_REMOTE,
        )).encode("utf-8")
    ).hexdigest()
    sweep_hash_payload = {
        "code_fingerprint": sweep_code_fingerprint,
        "spec": sweep_spec,
        "nominal_parameters": nominal_parameters,
        "nominal_geometry": base_geometry,
        "source_components_json": base["SOURCE_COMPONENTS_JSON"],
        "static_geometry": static_geometry,
        "cases": remote_cases,
        "stack": base["MATERIAL_STACK"],
        "settings": settings,
        "bbox": union_bbox,
    }
    sweep_hash = hashlib.sha256(
        json.dumps(sweep_hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    compressed_cases = base64.b64encode(
        zlib.compress(json.dumps(remote_cases, separators=(",", ":")).encode("utf-8"), level=9)
    ).decode("ascii")
    unique_warnings = list(dict.fromkeys(all_warnings))
    payload_cell = (
        "# Embedded Cartesian sweep data. Geometry snapshots are compressed to keep this notebook small.\n"
        "import base64 as _sweep_b64, json as _sweep_json, zlib as _sweep_zlib\n"
        f"SWEEP_SPEC = {pprint.pformat(sweep_spec, width=120, sort_dicts=False)}\n"
        f"SWEEP_HASH = {sweep_hash!r}\n"
        f"SWEEP_CODE_FINGERPRINT = {sweep_code_fingerprint!r}\n"
        f"_SWEEP_CASES_B64 = {compressed_cases!r}\n"
        "SWEEP_CASES = _sweep_json.loads(_sweep_zlib.decompress(_sweep_b64.b64decode(_SWEEP_CASES_B64)).decode('utf-8'))\n"
        f"SWEEP_STATIC_GEOMETRY = {pprint.pformat(static_geometry, width=160, compact=True, sort_dicts=False)}\n"
        f"SWEEP_RECOMPUTE_MODES = {_sweep_requires_mode_refresh(sweep_spec, base.get('GRATING_ANALYSIS'))!r}\n"
        f"SWEEP_NOMINAL_PARAMETERS = {pprint.pformat(nominal_parameters, width=120, sort_dicts=False)}\n"
        f"EXPORT_SCOPE_LABEL = {base['EXPORT_SCOPE_LABEL']!r}\n"
        f"EXPORTED_COMPONENTS = {pprint.pformat(base['EXPORTED_COMPONENTS'], width=120, sort_dicts=False)}\n"
        f"SOURCE_COMPONENTS_JSON = {pprint.pformat(base['SOURCE_COMPONENTS_JSON'], width=160, compact=True, sort_dicts=False)}\n"
        f"SETTINGS = {pprint.pformat(settings, width=120, sort_dicts=False)}\n"
        f"MATERIAL_STACK = {pprint.pformat(base['MATERIAL_STACK'], width=120, sort_dicts=False)}\n"
        f"BOUNDING_BOX_UM = {pprint.pformat(union_bbox)}\n"
        f"GEOMETRY = {pprint.pformat(base_geometry, width=160, compact=True, sort_dicts=False)}\n"
        f"PORTS = {pprint.pformat(base['PORTS'], width=120, sort_dicts=False)}\n"
        f"FIBER_GEOMETRIES = {pprint.pformat(base['FIBER_GEOMETRIES'], width=120, sort_dicts=False)}\n"
        f"GAUSSIAN_SOURCES = {pprint.pformat(base.get('GAUSSIAN_SOURCES', []), width=120, sort_dicts=False)}\n"
        f"PORTS_JSON = {pprint.pformat(base['PORTS_JSON'], width=120, sort_dicts=False)}\n"
        f"MONITORS = {pprint.pformat(base['MONITORS'], width=120, sort_dicts=False)}\n"
        f"GRATING_ANALYSIS = {pprint.pformat(base['GRATING_ANALYSIS'], width=120, sort_dicts=False)}\n"
        f"MMI_ANALYSIS = {pprint.pformat(base['MMI_ANALYSIS'], width=120, sort_dicts=False)}\n"
        f"EXPORT_WARNINGS = {pprint.pformat(unique_warnings, width=120)}\n"
        "print('Sweep points:', len(SWEEP_CASES), '| axes:', ', '.join(axis['parameter'] for axis in SWEEP_SPEC['axes']))\n"
        "for warning in EXPORT_WARNINGS:\n"
        "    print('Export note:', warning)\n"
        + _runtime_setup_source(_BUILD_CELL)
    )
    remote_build_cell = (
        "# Build the nominal model once, then install the in-session geometry hot-swap runtime.\n"
        f"REMOTE_MODEL_BUILDER = {repr(_BUILD_CELL)}\n"
        f"REMOTE_SWEEP_RUNTIME = {repr(_SWEEP_RUNTIME_REMOTE)}\n"
        "_remote_payload = (\n"
        "    'REMOTE_WORK = ' + repr(REMOTE_WORK) + '\\n'\n"
        "    + 'EXPORT_SCOPE_LABEL = ' + repr(EXPORT_SCOPE_LABEL) + '\\n'\n"
        "    + 'EXPORTED_COMPONENTS = ' + repr(EXPORTED_COMPONENTS) + '\\n'\n"
        "    + 'SOURCE_COMPONENTS_JSON = ' + repr(SOURCE_COMPONENTS_JSON) + '\\n'\n"
        "    + 'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "    + 'MATERIAL_STACK = ' + repr(MATERIAL_STACK) + '\\n'\n"
        "    + 'BOUNDING_BOX_UM = ' + repr(BOUNDING_BOX_UM) + '\\n'\n"
        "    + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "    + 'PORTS = ' + repr(PORTS) + '\\n'\n"
        "    + 'FIBER_GEOMETRIES = ' + repr(FIBER_GEOMETRIES) + '\\n'\n"
        "    + 'GAUSSIAN_SOURCES = ' + repr(GAUSSIAN_SOURCES) + '\\n'\n"
        "    + 'PORTS_JSON = ' + repr(PORTS_JSON) + '\\n'\n"
        "    + 'MONITORS = ' + repr(MONITORS) + '\\n'\n"
        "    + 'GRATING_ANALYSIS = ' + repr(GRATING_ANALYSIS) + '\\n'\n"
        "    + 'MMI_ANALYSIS = ' + repr(MMI_ANALYSIS) + '\\n'\n"
        "    + 'EXPORT_WARNINGS = ' + repr(EXPORT_WARNINGS) + '\\n'\n"
        "    + 'SWEEP_SPEC = ' + repr(SWEEP_SPEC) + '\\n'\n"
        "    + 'SWEEP_HASH = ' + repr(SWEEP_HASH) + '\\n'\n"
        "    + 'SWEEP_CODE_FINGERPRINT = ' + repr(SWEEP_CODE_FINGERPRINT) + '\\n'\n"
        "    + 'SWEEP_CASES = ' + repr(SWEEP_CASES) + '\\n'\n"
        "    + 'SWEEP_STATIC_GEOMETRY = ' + repr(SWEEP_STATIC_GEOMETRY) + '\\n'\n"
        "    + 'SWEEP_RECOMPUTE_MODES = ' + repr(SWEEP_RECOMPUTE_MODES) + '\\n'\n"
        "    + 'SWEEP_NOMINAL_PARAMETERS = ' + repr(SWEEP_NOMINAL_PARAMETERS) + '\\n'\n"
        ")\n"
        "run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER + '\\n' + REMOTE_SWEEP_RUNTIME, 'Build one live 3D sweep model directly', timeout=1800)\n"
        "_SWEEP_BUILD_STATE = lam.get(\"{'port_modes': PORT_MODE_SELECTIONS}\")\n"
        "print('Built one reusable live sweep model directly in memory.')\n"
        "SWEEP_FIBER_MODE_SELECTIONS = dict(_SWEEP_BUILD_STATE.get('port_modes') or {})\n"
        "if SWEEP_FIBER_MODE_SELECTIONS:\n"
        "    print('Resolved rotation-aware fiber mode pairs:', SWEEP_FIBER_MODE_SELECTIONS)\n"
    )
    geometry_projection_cell = (
        "# Optionally verify the nominal geometry and union FDTD domain once.\n"
        f"REMOTE_GEOMETRY_PROJECTIONS = {repr(_GEOMETRY_PROJECTIONS_REMOTE)}\n"
        "if SHOW_GEOMETRY_PREVIEW:\n"
        "    run_remote_checked(REMOTE_GEOMETRY_PROJECTIONS, 'Render nominal sweep geometry', timeout=1800)\n"
        "    GEOMETRY_PROJECTIONS_FILE = REMOTE_WORK.rstrip('/') + '/geometry_xyz_projections.png'\n"
        "    lam.show(GEOMETRY_PROJECTIONS_FILE, width=1400)\n"
        "else:\n"
        "    print('Nominal sweep geometry preview skipped by cell 1.')\n"
    )
    port_mode_profiles_cell = (
        "# Optionally display nominal port modes; required selection already ran during the build.\n"
        f"REMOTE_PORT_MODE_PROFILES = {repr(_PORT_MODE_PROFILES_REMOTE)}\n"
        "if SHOW_PORT_MODE_PREVIEW:\n"
        "    run_remote_checked(REMOTE_PORT_MODE_PROFILES, 'Validate nominal sweep port modes', timeout=1800)\n"
        "    if PORTS:\n"
        "        PORT_MODE_PROFILES_FILE = REMOTE_WORK.rstrip('/') + '/port_mode_Ex_Ey.png'\n"
        "        lam.show(PORT_MODE_PROFILES_FILE, width=1400)\n"
        "        if not bool(lam.get('PORT_MODE_VALID')):\n"
        "            raise RuntimeError('Nominal port-mode validation failed; correct the port geometry before sweeping.')\n"
        "else:\n"
        "    print('Nominal port Ex/Ey images skipped by cell 1; mode selection and neff checks remain active.')\n"
    )
    resource_save_cell = (
        "# Configure GPU once and save/download the required nominal inspection FSP.\n"
        f"REMOTE_RESOURCE_AND_SAVE = {repr(_REMOTE_RESOURCE_AND_SAVE)}\n"
        "run_remote_checked(REMOTE_RESOURCE_AND_SAVE, 'Configure one sweep resource and save inspection FSP', timeout=1800)\n"
        "_SWEEP_RESOURCE_STATE = lam.get(\"{'project_file': REMOTE_PROJECT_FILE, 'inspection_file': REMOTE_INSPECTION_PROJECT_FILE, 'inspection_saved': bool(REMOTE_INSPECTION_FSP_SAVED)}\")\n"
        "REMOTE_PROJECT_FILE = str(_SWEEP_RESOURCE_STATE['project_file'])\n"
        "REMOTE_INSPECTION_PROJECT_FILE = str(_SWEEP_RESOURCE_STATE['inspection_file'])\n"
        "REMOTE_INSPECTION_FSP_SAVED = bool(_SWEEP_RESOURCE_STATE['inspection_saved'])\n"
        "if not REMOTE_INSPECTION_FSP_SAVED:\n"
        "    raise RuntimeError('The required nominal inspection FSP was not saved')\n"
        "LOCAL_INSPECTION_PROJECT_FILE = PIRIS_FSP_DIR / os.path.basename(REMOTE_INSPECTION_PROJECT_FILE)\n"
        "lam.fetch(REMOTE_INSPECTION_PROJECT_FILE, str(LOCAL_INSPECTION_PROJECT_FILE))\n"
        "print('saved nominal inspection project ->', LOCAL_INSPECTION_PROJECT_FILE)\n"
    )
    review_project_cell = (
        "# Inspect the required nominal FSP; no FSP is saved for each sweep point.\n"
        "from IPython.display import FileLink, display\n"
        "display(FileLink(str(LOCAL_INSPECTION_PROJECT_FILE)))\n"
        "print('The union FDTD domain covers every embedded sweep geometry.')\n"
    )
    sweep_run_cell = (
        "# Run the entire Cartesian sweep inside the one persistent Lumerical/GPU session.\n"
        f"REMOTE_SWEEP_RUNNER = {repr(_SWEEP_RUNNER_REMOTE)}\n"
        "SWEEP_PROGRESS_FILE = REMOTE_WORK.rstrip('/') + '/sweep_live_progress.jsonl'\n"
        "if SETTINGS.get('run_after_build', True):\n"
        "    _sweep_timeout = max(21600, 21600 * len(SWEEP_CASES))\n"
        "    solve_remote_checked(\n"
        "        REMOTE_SWEEP_RUNNER, label='Lumerical GPU parameter sweep',\n"
        "        timeout=_sweep_timeout, progress_file=SWEEP_PROGRESS_FILE,\n"
        "        progress_mode='sweep',\n"
        "    )\n"
        "else:\n"
        "    print('Sweep run is disabled; only the nominal model was built.')\n"
    )
    local_results_cell = (
        "if not SETTINGS.get('run_after_build', True):\n"
        "    print('Sweep was not run, so there are no sweep results to fetch.')\n"
        "else:\n"
        + textwrap.indent(_SWEEP_LOCAL_RESULTS_CELL, "    ")
    )
    intro = f"""# Max Layout → Lumerical parameter sweep

This notebook runs **{len(remote_cases)} Cartesian sweep points** for **{sweep_spec['component_kind']}** using the exact JSON parameter names: {', '.join(axis['parameter'] for axis in sweep_spec['axes'])}.

- One Shared Web licence checkout and one persistent Lumerical session are used.
- The live nominal model comes from the untouched current project JSON, even when the first sweep value differs from the nominal parameter. Its inspection FSP is always saved.
- Materials, the union FDTD region, ports, and monitors are built once.
- Each point hot-swaps only the embedded Layer Builder geometry and linked automatic companion positions.
- The GPU is configured once and no per-point FSP is saved or downloaded. The final saved FSP is the winning design only.
- A small numerical checkpoint is written after each completed point for safe restart.
- After all electromagnetic solves, Lumerical is switched back to the 30-thread CPU resource and every plot is generated locally on CPU.
- Grating sweeps save one CE spectrum per point using names such as `CE-P=0.75-F=0.56.png`, plus a maximum-CE summary heatmap and CSV.
- MMI sweeps save upper/input, lower/input, total/input, and 50/50 split curves for every point; the primary sweep objective is upper-branch power divided by measured input power.
- `SOURCE_COMPONENTS_JSON`, `SWEEP_NOMINAL_PARAMETERS`, and every case's `source_parameters` preserve the exact inputs used to build the nominal and swept geometries.
"""
    notebook = {
        "cells": [
            _notebook_cell("code", _quick_run_options_cell(settings, workflow="sequential sweep")),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Connect to Lambda\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 2 · Acquire one licensed Lumerical session\n"),
            _notebook_cell("code", _LICENSE_CHECKOUT_CELL),
            _notebook_cell("markdown", "## 3 · Embedded sweep specification and geometry snapshots\n"),
            _notebook_cell("code", payload_cell),
            _notebook_cell("markdown", "## 4 · Build one reusable nominal model\n"),
            _notebook_cell("code", remote_build_cell),
            _notebook_cell("markdown", "## 5 · Verify nominal geometry and union sweep domain\n"),
            _notebook_cell("code", geometry_projection_cell),
            _notebook_cell("markdown", "## 6 · Verify nominal port modes\n\nFor a 1×2 MMI this shows Ex and Ey for the input, upper-output, and lower-output ports and verifies that all three modes lie around the same effective-index target.\n"),
            _notebook_cell("code", port_mode_profiles_cell),
            _notebook_cell("markdown", "## 7 · Configure GPU and save one nominal inspection FSP\n"),
            _notebook_cell("code", resource_save_cell),
            _notebook_cell("markdown", "## 8 · Inspect the saved nominal FSP\n"),
            _notebook_cell("code", review_project_cell),
            _notebook_cell("markdown", "## 9 · Run every sweep point on GPU\n"),
            _notebook_cell("code", sweep_run_cell),
            _notebook_cell("markdown", "## 10 · Fetch once and create all plots on CPU\n"),
            _notebook_cell("code", local_results_cell),
            _notebook_cell("markdown", "## 11 · Release FDTD and return all roamed HPC Packs\n\nAlways run this cell, including after an interrupted sweep.\n"),
            _notebook_cell("code", _RELEASE_LICENSES_CELL),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "max_layout": {
                "export": "lumerical-fdtd-sweep",
                "units": "um",
                "dimension": "3D",
                "point_count": len(remote_cases),
                "execution": "one-session-layer-builder-hot-swap",
                "per_point_fsp": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, unique_warnings


def _normalized_lumerical_multigpu_settings(configuration: dict[str, Any]) -> dict[str, int]:
    """Return the independent multi-node execution settings used by its notebook."""
    raw = dict(configuration.get("lumerical_multigpu") or {})
    node_count = int(raw.get("node_count", 8))
    simulations_per_gpu = int(raw.get("simulations_per_gpu", 1))
    if not 1 <= node_count <= 64:
        raise ValueError("Multi-GPU A100 node count must be between 1 and 64")
    if simulations_per_gpu != 1:
        raise ValueError(
            "Production multi-GPU sweeps require exactly one simulation per GPU; "
            "use additional A100 nodes for more concurrency"
        )
    return {
        "node_count": node_count,
        "simulations_per_gpu": simulations_per_gpu,
        "max_parallel_simulations": node_count * simulations_per_gpu,
    }


def generate_lumerical_multigpu_sweep_notebook(
    sweep_cases: list[dict[str, Any]],
    configuration: dict[str, Any],
    sweep_spec: dict[str, Any],
    nominal_components: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Generate the separate multi-node/multi-process A100 sweep notebook."""
    multigpu_settings = _normalized_lumerical_multigpu_settings(configuration)
    sequential, warnings = generate_lumerical_sweep_notebook(
        sweep_cases,
        configuration,
        sweep_spec,
        nominal_components=nominal_components,
    )

    def source_containing(needle: str) -> str:
        for cell in sequential["cells"]:
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and needle in source:
                return source
        raise RuntimeError("The base sweep notebook is missing cell content: " + needle)

    payload_cell = source_containing("_SWEEP_CASES_B64")
    geometry_projection_cell = source_containing("REMOTE_GEOMETRY_PROJECTIONS")
    port_mode_profiles_cell = source_containing("REMOTE_PORT_MODE_PROFILES")
    base_resource_save_cell = source_containing("REMOTE_RESOURCE_AND_SAVE")
    review_project_cell = source_containing("The union FDTD domain covers every embedded sweep geometry")

    multigpu_settings_cell = (
        "# Independent multi-node execution settings; this does not alter the ordinary sweep exporter.\n"
        f"MULTIGPU_SETTINGS = {pprint.pformat(multigpu_settings, width=100, sort_dicts=False)}\n"
        "SETTINGS['sweep_multithread'] = True\n"
        "print('Multi-GPU request:', MULTIGPU_SETTINGS['node_count'], 'A100 nodes x', "
        "MULTIGPU_SETTINGS['simulations_per_gpu'], 'simulation(s) per GPU')\n"
    )
    remote_build_cell = (
        "# Build the nominal model exactly once on the controller and install the shared-checkpoint runtime.\n"
        f"REMOTE_MODEL_BUILDER = {repr(_BUILD_CELL)}\n"
        f"REMOTE_SWEEP_RUNTIME = {repr(_SWEEP_RUNTIME_REMOTE)}\n"
        "_remote_payload = (\n"
        "    'REMOTE_WORK = ' + repr(REMOTE_WORK) + '\\n'\n"
        "    + 'SWEEP_RESULTS_ROOT = ' + repr(REMOTE_WORK) + '\\n'\n"
        "    + 'SWEEP_RESULTS_BASENAME = ' + repr(SWEEP_RESULTS_BASENAME) + '\\n'\n"
        "    + 'SWEEP_EXECUTION_MODE = ' + repr('multi-node-parallel-workers') + '\\n'\n"
        "    + 'SWEEP_SHARED_CHECKPOINT_DIR = ' + repr(SWEEP_SHARED_CHECKPOINT_DIR) + '\\n'\n"
        "    + 'EXPORT_SCOPE_LABEL = ' + repr(EXPORT_SCOPE_LABEL) + '\\n'\n"
        "    + 'EXPORTED_COMPONENTS = ' + repr(EXPORTED_COMPONENTS) + '\\n'\n"
        "    + 'SOURCE_COMPONENTS_JSON = ' + repr(SOURCE_COMPONENTS_JSON) + '\\n'\n"
        "    + 'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "    + 'MATERIAL_STACK = ' + repr(MATERIAL_STACK) + '\\n'\n"
        "    + 'BOUNDING_BOX_UM = ' + repr(BOUNDING_BOX_UM) + '\\n'\n"
        "    + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "    + 'PORTS = ' + repr(PORTS) + '\\n'\n"
        "    + 'FIBER_GEOMETRIES = ' + repr(FIBER_GEOMETRIES) + '\\n'\n"
        "    + 'GAUSSIAN_SOURCES = ' + repr(GAUSSIAN_SOURCES) + '\\n'\n"
        "    + 'PORTS_JSON = ' + repr(PORTS_JSON) + '\\n'\n"
        "    + 'MONITORS = ' + repr(MONITORS) + '\\n'\n"
        "    + 'GRATING_ANALYSIS = ' + repr(GRATING_ANALYSIS) + '\\n'\n"
        "    + 'MMI_ANALYSIS = ' + repr(MMI_ANALYSIS) + '\\n'\n"
        "    + 'EXPORT_WARNINGS = ' + repr(EXPORT_WARNINGS) + '\\n'\n"
        "    + 'SWEEP_SPEC = ' + repr(SWEEP_SPEC) + '\\n'\n"
        "    + 'SWEEP_HASH = ' + repr(SWEEP_HASH) + '\\n'\n"
        "    + 'SWEEP_CODE_FINGERPRINT = ' + repr(SWEEP_CODE_FINGERPRINT) + '\\n'\n"
        "    + 'SWEEP_CASES = ' + repr(SWEEP_CASES) + '\\n'\n"
        "    + 'SWEEP_STATIC_GEOMETRY = ' + repr(SWEEP_STATIC_GEOMETRY) + '\\n'\n"
        "    + 'SWEEP_RECOMPUTE_MODES = ' + repr(SWEEP_RECOMPUTE_MODES) + '\\n'\n"
        "    + 'SWEEP_NOMINAL_PARAMETERS = ' + repr(SWEEP_NOMINAL_PARAMETERS) + '\\n'\n"
        ")\n"
        "run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER + '\\n' + REMOTE_SWEEP_RUNTIME, "
        "'Build one shared nominal 3D model directly', timeout=1800)\n"
        "_MULTIGPU_BUILD_STATE = lam.get(\"{'ports': PORTS, 'monitors': MONITORS, 'grating': GRATING_ANALYSIS, 'mmi': MMI_ANALYSIS, 'port_modes': PORT_MODE_SELECTIONS}\")\n"
        "PORTS = list(_MULTIGPU_BUILD_STATE.get('ports') or PORTS)\n"
        "MONITORS = list(_MULTIGPU_BUILD_STATE.get('monitors') or MONITORS)\n"
        "GRATING_ANALYSIS = dict(_MULTIGPU_BUILD_STATE.get('grating') or GRATING_ANALYSIS)\n"
        "MMI_ANALYSIS = dict(_MULTIGPU_BUILD_STATE.get('mmi') or MMI_ANALYSIS)\n"
        "SWEEP_FIBER_MODE_SELECTIONS = dict(_MULTIGPU_BUILD_STATE.get('port_modes') or {})\n"
        "if SWEEP_FIBER_MODE_SELECTIONS:\n"
        "    print('Resolved rotation-aware fiber mode pairs:', SWEEP_FIBER_MODE_SELECTIONS)\n"
    )
    shared_seed_source = (
        "# Every worker needs one shared seed because it owns a separate Lumerical process.\n"
        "if not REMOTE_INSPECTION_FSP_SAVED:\n"
        "    raise RuntimeError('The required shared inspection FSP was not saved')\n"
        "SHARED_NOMINAL_FSP = REMOTE_INSPECTION_PROJECT_FILE\n"
        "print('Multi-GPU workers use the always-saved inspection FSP.')\n"
        "if not str(SHARED_NOMINAL_FSP).startswith('/lambda/nfs/'):\n"
        "    raise RuntimeError('The nominal FSP must be on the shared /lambda/nfs filesystem')\n"
        "print('Shared seed FSP for every A100 worker:', SHARED_NOMINAL_FSP)\n"
    )
    resource_save_cell = (
        base_resource_save_cell
        + "\nif SETTINGS.get('run_after_build', True):\n"
        + textwrap.indent(shared_seed_source, "    ")
        + "else:\n"
        + "    SHARED_NOMINAL_FSP = None\n"
        + "    print('Multi-GPU run disabled; no shared worker seed was created.')\n"
    )

    worker_resource_remote = _REMOTE_RESOURCE_AND_SAVE.split("\n_project_name =", 1)[0].rstrip() + "\n"
    orchestration_cell = (
        "# Acquire licences only now, then run one persistent process on each prepared A100 node.\n"
        "from concurrent.futures import ThreadPoolExecutor, as_completed\n"
        "import json\n"
        "import os\n"
        "import queue\n"
        "import re\n"
        "import threading\n"
        "import time\n"
        f"REMOTE_LAYER_BUILDER_HELPER = {repr(_MULTIGPU_LAYER_BUILDER_HELPER_REMOTE)}\n"
        f"REMOTE_WORKER_RESOURCE_SETUP = {repr(worker_resource_remote)}\n"
        f"REMOTE_SINGLE_SWEEP_CASE = {repr(_MULTIGPU_SINGLE_CASE_REMOTE)}\n"
        f"REMOTE_CPU_RESET = {repr(_SWITCH_TO_CPU_ANALYSIS_REMOTE)}\n"
        "\n"
        "_base_results_root = REMOTE_WORK\n"
        "_slot_descriptors = [(_node_index, 0, _node) for _node_index, _node in enumerate(MULTIGPU_NODES)]\n"
        "_slot_descriptors = _slot_descriptors[:min(len(SWEEP_CASES), int(MULTIGPU_SETTINGS['max_parallel_simulations']))]\n"
        "MULTIGPU_WORKER_RECORDS = []\n"
        "MULTIGPU_FAILURES = []\n"
        "MULTIGPU_FATAL_ERRORS = []\n"
        "_failure_lock = threading.Lock()\n"
        "_progress_lock = threading.Lock()\n"
        "_progress_state = {'processed': 0, 'completed': 0, 'failed': 0, 'duration_sum': 0.0, 'duration_count': 0}\n"
        "_progress_started = time.monotonic()\n"
        "_case_queue = queue.Queue()\n"
        "\n"
        "def _multigpu_result_record(output):\n"
        "    markers = re.findall(r'__MAX_LAYOUT_SWEEP_RESULT__(\\{[^\\n]+\\})', str(output))\n"
        "    if not markers:\n"
        "        return {}\n"
        "    try:\n"
        "        return dict(json.loads(markers[-1]))\n"
        "    except Exception:\n"
        "        return {}\n"
        "\n"
        "def _report_multigpu_progress(status, case_index, worker_id, duration=0.0, result=None):\n"
        "    result = dict(result or {})\n"
        "    terminal = status in {'completed', 'reused', 'failed'}\n"
        "    with _progress_lock:\n"
        "        if terminal:\n"
        "            _progress_state['processed'] += 1\n"
        "            _progress_state['failed'] += int(status == 'failed')\n"
        "            _progress_state['completed'] += int(status != 'failed')\n"
        "            if status != 'reused':\n"
        "                _progress_state['duration_sum'] += max(0.0, float(duration))\n"
        "                _progress_state['duration_count'] += 1\n"
        "        processed = int(_progress_state['processed'])\n"
        "        total = len(SWEEP_CASES)\n"
        "        percent = 100.0 * processed / total if total else 0.0\n"
        "        elapsed = time.monotonic() - _progress_started\n"
        "        average = _progress_state['duration_sum'] / max(1, _progress_state['duration_count'])\n"
        "        workers = max(1, len(MULTIGPU_WORKER_RECORDS))\n"
        "        eta = average * max(0, total - processed) / workers if processed else float('nan')\n"
        "        filled = int(max(0.0, min(100.0, percent)) / 2.0)\n"
        "        label = SWEEP_CASES[case_index]['display_label'] if 0 <= int(case_index) < total else 'preparing'\n"
        "        result_text = ''\n"
        "        if 'peak_response' in result:\n"
        "            result_text = ' | peak %.6g at %.3f nm' % (float(result['peak_response']), float(result.get('peak_wavelength_nm', float('nan'))))\n"
        "        eta_text = ('%.1f min' % (eta / 60.0)) if eta == eta else 'estimating'\n"
        "        print('Multi-GPU sweep [%s%s] %6.2f%% | %d/%d finished | failed %d | elapsed %.1f min | ETA %s' % ('#' * filled, '-' * (50 - filled), percent, processed, total, _progress_state['failed'], elapsed / 60.0, eta_text), flush=True)\n"
        "        print('Current: %s on %s — %s%s' % (status, worker_id, label, result_text), flush=True)\n"
        "\n"
        "def _worker_payload(worker_work):\n"
        "    return (\n"
        "        'import os, shutil, numpy as np, lumapi\\nUM = 1e-6\\n'\n"
        "        + 'REMOTE_WORK = ' + repr(worker_work) + '\\n'\n"
        "        + 'SWEEP_RESULTS_ROOT = ' + repr(_base_results_root) + '\\n'\n"
        "        + 'SWEEP_RESULTS_BASENAME = ' + repr(SWEEP_RESULTS_BASENAME) + '\\n'\n"
        "        + 'SWEEP_EXECUTION_MODE = ' + repr('multi-node-parallel-workers') + '\\n'\n"
        "        + 'SWEEP_SHARED_CHECKPOINT_DIR = ' + repr(SWEEP_SHARED_CHECKPOINT_DIR) + '\\n'\n"
        "        + 'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "        + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "        + 'PORTS = ' + repr(PORTS) + '\\n'\n"
        "        + 'FIBER_GEOMETRIES = ' + repr(FIBER_GEOMETRIES) + '\\n'\n"
        "        + 'GAUSSIAN_SOURCES = ' + repr(GAUSSIAN_SOURCES) + '\\n'\n"
        "        + 'MONITORS = ' + repr(MONITORS) + '\\n'\n"
        "        + 'GRATING_ANALYSIS = ' + repr(GRATING_ANALYSIS) + '\\n'\n"
        "        + 'MMI_ANALYSIS = ' + repr(MMI_ANALYSIS) + '\\n'\n"
        "        + 'SWEEP_SPEC = ' + repr(SWEEP_SPEC) + '\\n'\n"
        "        + 'SWEEP_HASH = ' + repr(SWEEP_HASH) + '\\n'\n"
        "        + 'SWEEP_CODE_FINGERPRINT = ' + repr(SWEEP_CODE_FINGERPRINT) + '\\n'\n"
        "        + 'SWEEP_CASES = ' + repr(SWEEP_CASES) + '\\n'\n"
        "        + 'SWEEP_STATIC_GEOMETRY = ' + repr(SWEEP_STATIC_GEOMETRY) + '\\n'\n"
        "        + 'SWEEP_RECOMPUTE_MODES = ' + repr(SWEEP_RECOMPUTE_MODES) + '\\n'\n"
        "        + 'SWEEP_FIBER_MODE_SELECTIONS = ' + repr(SWEEP_FIBER_MODE_SELECTIONS) + '\\n'\n"
        "        + 'os.makedirs(REMOTE_WORK, exist_ok=True)\\nos.chdir(REMOTE_WORK)\\n'\n"
        "        + 'WORKER_RUNTIME_FSP = os.path.join(REMOTE_WORK, \"_worker_runtime_seed.fsp\")\\n'\n"
        "        + 'shutil.copy2(' + repr(SHARED_NOMINAL_FSP) + ', WORKER_RUNTIME_FSP)\\n'\n"
        "        + 'fdtd = lumapi.FDTD(hide=' + repr(bool(SETTINGS.get('hide_cad', True)))\n"
        "        + ', serverArgs={\\\"threads\\\": ' + repr(str(int(SETTINGS.get('build_cpu_threads', 30)))) + '})\\n'\n"
        "        + 'fdtd.load(WORKER_RUNTIME_FSP)\\n'\n"
        "        + REMOTE_LAYER_BUILDER_HELPER + '\\n' + REMOTE_SWEEP_RUNTIME + '\\n' + REMOTE_WORKER_RESOURCE_SETUP\n"
        "    )\n"
        "\n"
        "def _cleanup_worker(record):\n"
        "    if record.get('cleanup_complete', False):\n"
        "        return True\n"
        "    client = record.get('client')\n"
        "    if client is None:\n"
        "        return False\n"
        "    transport_poisoned = bool(record.get('poisoned', False) or getattr(client, '_poisoned', False))\n"
        "    if record.get('initialized', False) and not transport_poisoned:\n"
        "        try:\n"
        "            _multigpu_run_checked(client, REMOTE_CPU_RESET, 'Return ' + record['label'] + ' to CPU', timeout=180)\n"
        "        except Exception as exc:\n"
        "            print(record['label'], 'CPU reset note:', str(exc)[:240])\n"
        "    stop_code = (\n"
        "        '_had_fdtd = \\\"fdtd\\\" in globals()\\n'\n"
        "        + 'if _had_fdtd:\\n    fdtd.close()\\n'\n"
        "        + 'globals().pop(\\\"fdtd\\\", None)\\n'\n"
        "        + '_worker_runtime = globals().get(\\\"WORKER_RUNTIME_FSP\\\", \\\"\\\")\\n'\n"
        "        + 'if _worker_runtime and os.path.isfile(_worker_runtime): os.remove(_worker_runtime)\\n'\n"
        "        + 'print(\\\"__MULTIGPU_FDTD_STOPPED__\\\")\\n'\n"
        "    )\n"
        "    stop_error = None\n"
        "    record['fdtd_stopped'] = False\n"
        "    if not transport_poisoned:\n"
        "        try:\n"
        "            stop_output = _multigpu_run_checked(client, stop_code, 'Stop ' + record['label'] + ' FDTD', timeout=180)\n"
        "            if '__MULTIGPU_FDTD_STOPPED__' not in stop_output:\n"
        "                raise RuntimeError('FDTD stop confirmation marker missing')\n"
        "            record['fdtd_stopped'] = True\n"
        "        except Exception as stop_exc:\n"
        "            stop_error = stop_exc\n"
        "    if not record.get('fdtd_stopped', False):\n"
        "        try:\n"
        "            stop_report = client.stop_work_processes(timeout=25)\n"
        "            if not (isinstance(stop_report, dict) and stop_report.get('confirmed') is True and not stop_report.get('remaining_pids', [])):\n"
        "                raise RuntimeError('stop_work_processes returned an unconfirmed report: ' + repr(stop_report))\n"
        "            record['fdtd_stopped'] = True\n"
        "            record['poisoned'] = bool(getattr(client, '_poisoned', False))\n"
        "            print(record['label'], 'confirmed exact-work FDTD processes stopped:', stop_report)\n"
        "        except Exception as forced_stop_exc:\n"
        "            record['cleanup_uncertain'] = True\n"
        "            details = str(forced_stop_exc) if stop_error is None else str(stop_error) + ' | forced stop: ' + str(forced_stop_exc)\n"
        "            print(record['label'], 'FDTD STOP UNCONFIRMED; HPC Packs were NOT returned:', details[-500:])\n"
        "            print('Keep this notebook and run the emergency cleanup cell after the remote solve has stopped.')\n"
        "            return False\n"
        "    if record.get('packs_checked_out', False):\n"
        "        try:\n"
        "            release_output = _multigpu_run_once_checked(client, MULTIGPU_LICENSE_RELEASE_REMOTE, 'Return ' + record['label'] + ' HPC Packs', timeout=300)\n"
        "            if '__MULTIGPU_LICENSE_RELEASED__' not in release_output:\n"
        "                raise RuntimeError('Licence release success marker missing')\n"
        "            record['packs_checked_out'] = False\n"
        "            record['packs_state_uncertain'] = False\n"
        "        except Exception as release_exc:\n"
        "            record['cleanup_uncertain'] = True\n"
        "            print(record['label'], 'LICENCE RELEASE ERROR:', str(release_exc)[-500:])\n"
        "            return False\n"
        "    if not record.get('is_controller', False):\n"
        "        try:\n"
        "            client.close()\n"
        "        except Exception:\n"
        "            pass\n"
        "    record['cleanup_complete'] = True\n"
        "    record['cleanup_uncertain'] = False\n"
        "    return True\n"
        "\n"
        "def _initialize_worker(record):\n"
        "    client = record['client']\n"
        "    record['packs_state_uncertain'] = True\n"
        "    try:\n"
        "        checkout_output = _multigpu_run_checked(client, MULTIGPU_LICENSE_CHECKOUT_REMOTE, 'Reserve ' + record['label'] + ' licences', timeout=300)\n"
        "        record['packs_checked_out'] = True\n"
        "        record['packs_state_uncertain'] = False\n"
        "        if '__MULTIGPU_LICENSE_ACQUIRED__' not in checkout_output:\n"
        "            raise RuntimeError('Licence checkout success marker missing')\n"
        "        if record['is_controller']:\n"
        "            controller_prep = (\n"
        "                'import os\\nSWEEP_RESULTS_ROOT = ' + repr(_base_results_root) + '\\n'\n"
        "                + 'SWEEP_RESULTS_BASENAME = ' + repr(SWEEP_RESULTS_BASENAME) + '\\n'\n"
        "                + 'SWEEP_EXECUTION_MODE = ' + repr('multi-node-parallel-workers') + '\\n'\n"
        "                + 'REMOTE_WORK = ' + repr(record['worker_work']) + '\\n'\n"
        "                + 'os.makedirs(REMOTE_WORK, exist_ok=True)\\nos.chdir(REMOTE_WORK)\\n'\n"
        "            )\n"
        "            _multigpu_run_checked(client, controller_prep, 'Prepare controller worker workspace', timeout=120)\n"
        "            client.work = record['worker_work']\n"
        "        else:\n"
        "            _multigpu_run_checked(client, _worker_payload(record['worker_work']), 'Load shared nominal FSP on ' + record['label'], timeout=1800)\n"
        "        record['initialized'] = True\n"
        "        return record['worker_id']\n"
        "    except Exception as initialization_exc:\n"
        "        # A transport timeout after checkout is ambiguous. Treat packs as held until a confirmed check-in.\n"
        "        if record.get('packs_state_uncertain', False):\n"
        "            initialization_text = str(initialization_exc).lower()\n"
        "            transport_uncertain = 'timed out after' in initialization_text or 'remote session died' in initialization_text\n"
        "            record['packs_checked_out'] = bool(transport_uncertain)\n"
        "            record['packs_state_uncertain'] = bool(transport_uncertain)\n"
        "        raise\n"
        "\n"
        "def _worker_loop(record):\n"
        "    client = record['client']\n"
        "    completed = 0\n"
        "    try:\n"
        "        while True:\n"
        "            try:\n"
        "                case_index = _case_queue.get_nowait()\n"
        "            except queue.Empty:\n"
        "                break\n"
        "            case = SWEEP_CASES[case_index]\n"
        "            case_started = time.monotonic()\n"
        "            _report_multigpu_progress('running', case_index, record['worker_id'])\n"
        "            try:\n"
        "                case_code = 'MULTIGPU_CASE_INDEX = ' + repr(case_index) + '\\n' + REMOTE_SINGLE_SWEEP_CASE\n"
        "                case_output = _multigpu_run_checked(client, case_code, record['label'] + ' — ' + case['display_label'], timeout=21600)\n"
        "                result_record = _multigpu_result_record(case_output)\n"
        "                case_status = str(result_record.get('status', 'completed'))\n"
        "                completed += 1\n"
        "                _report_multigpu_progress(case_status, case_index, record['worker_id'], time.monotonic() - case_started, result_record)\n"
        "            except Exception as exc:\n"
        "                error_text = str(exc)\n"
        "                with _failure_lock:\n"
        "                    MULTIGPU_FAILURES.append({'index': case_index, 'values': case['values'], 'worker': record['worker_id'], 'error': error_text})\n"
        "                _report_multigpu_progress('failed', case_index, record['worker_id'], time.monotonic() - case_started)\n"
        "                print('[%s] FAILED %s: %s' % (record['worker_id'], case['display_label'], error_text[-500:]))\n"
        "                if getattr(client, '_poisoned', False) or 'timed out after' in error_text.lower() or 'remote session died' in error_text.lower():\n"
        "                    record['poisoned'] = True\n"
        "                    print(record['label'], 'was quarantined after a transport timeout; no further cases will be sent to it.')\n"
        "                    break\n"
        "            finally:\n"
        "                _case_queue.task_done()\n"
        "        return record['worker_id'], completed\n"
        "    finally:\n"
        "        _cleanup_worker(record)\n"
        "\n"
        "_orchestration_exception = None\n"
        "try:\n"
        "    for worker_index, (_node_index, _gpu_slot, _node) in enumerate(_slot_descriptors):\n"
        "        worker_id = 'worker_%02d_node_%02d_gpu_slot_%02d' % (worker_index, _node_index, _gpu_slot)\n"
        "        worker_work = os.path.join(MULTIGPU_WORKER_ROOT, worker_id)\n"
        "        is_controller = bool(_node.get('controller', False))\n"
        "        client = lam if is_controller else Lambda(work=worker_work, verbose=False, host=_node['host'], key=os.path.expanduser(_node['key']))\n"
        "        MULTIGPU_WORKER_RECORDS.append({\n"
        "            'worker_index': worker_index, 'worker_id': worker_id, 'label': worker_id + '@' + _node['node_name'],\n"
        "            'client': client, 'worker_work': worker_work, 'is_controller': is_controller,\n"
        "            'packs_checked_out': False, 'packs_state_uncertain': False, 'initialized': False,\n"
        "            'fdtd_stopped': False, 'poisoned': False, 'cleanup_complete': False,\n"
        "        })\n"
        "    with ThreadPoolExecutor(max_workers=len(MULTIGPU_WORKER_RECORDS)) as pool:\n"
        "        init_futures = {pool.submit(_initialize_worker, record): record for record in MULTIGPU_WORKER_RECORDS}\n"
        "        initialization_errors = []\n"
        "        for future in as_completed(init_futures):\n"
        "            try:\n"
        "                print('Initialized', future.result())\n"
        "            except BaseException as exc:\n"
        "                initialization_errors.append((init_futures[future]['label'], str(exc)))\n"
        "    if initialization_errors:\n"
        "        raise RuntimeError('Multi-GPU worker preflight failed before any sweep solve: ' + repr(initialization_errors))\n"
        "    # Solve exactly one point before releasing the parallel queue.  This validates the\n"
        "    # Lumerical result-provider schema once and leaves a normal reusable checkpoint.\n"
        "    # A bad monitor/result binding therefore consumes at most one GPU solve, not the sweep.\n"
        "    if not SWEEP_CASES:\n"
        "        raise RuntimeError('The multi-GPU sweep contains no cases')\n"
        "    schema_preflight_index = 0\n"
        "    schema_preflight_record = MULTIGPU_WORKER_RECORDS[0]\n"
        "    schema_preflight_case = SWEEP_CASES[schema_preflight_index]\n"
        "    schema_preflight_code = (\n"
        "        'MULTIGPU_CASE_INDEX = ' + repr(schema_preflight_index) + '\\n' + REMOTE_SINGLE_SWEEP_CASE\n"
        "    )\n"
        "    _preflight_started = time.monotonic()\n"
        "    _report_multigpu_progress('running', schema_preflight_index, schema_preflight_record['worker_id'])\n"
        "    try:\n"
        "        _preflight_output = _multigpu_run_checked(\n"
        "            schema_preflight_record['client'], schema_preflight_code,\n"
        "            'Result-schema preflight — ' + schema_preflight_case['display_label'], timeout=21600,\n"
        "        )\n"
        "    except BaseException:\n"
        "        _report_multigpu_progress('failed', schema_preflight_index, schema_preflight_record['worker_id'], time.monotonic() - _preflight_started)\n"
        "        raise\n"
        "    _preflight_result = _multigpu_result_record(_preflight_output)\n"
        "    _report_multigpu_progress(str(_preflight_result.get('status', 'completed')), schema_preflight_index, schema_preflight_record['worker_id'], time.monotonic() - _preflight_started, _preflight_result)\n"
        "    print('Result-schema preflight passed; its checkpoint will be reused during aggregation.')\n"
        "    for case_index in range(len(SWEEP_CASES)):\n"
        "        if case_index != schema_preflight_index:\n"
        "            _case_queue.put(case_index)\n"
        "    with ThreadPoolExecutor(max_workers=len(MULTIGPU_WORKER_RECORDS)) as pool:\n"
        "        solve_futures = {pool.submit(_worker_loop, record): record for record in MULTIGPU_WORKER_RECORDS}\n"
        "        for future in as_completed(solve_futures):\n"
        "            try:\n"
        "                print('Worker finished:', future.result())\n"
        "            except BaseException as exc:\n"
        "                MULTIGPU_FATAL_ERRORS.append({'worker': solve_futures[future]['label'], 'error': str(exc)})\n"
        "except BaseException as exc:\n"
        "    _orchestration_exception = exc\n"
        "    MULTIGPU_FATAL_ERRORS.append({'worker': 'orchestrator', 'error': str(exc)})\n"
        "finally:\n"
        "    for record in MULTIGPU_WORKER_RECORDS:\n"
        "        try:\n"
        "            _cleanup_worker(record)\n"
        "        except BaseException as cleanup_exc:\n"
        "            MULTIGPU_FATAL_ERRORS.append({'worker': record.get('label', 'worker'), 'error': 'cleanup: ' + str(cleanup_exc)})\n"
        "    CONTROLLER_PACKS_CHECKED_OUT = any(record.get('is_controller') and record.get('packs_checked_out') for record in MULTIGPU_WORKER_RECORDS)\n"
        "\n"
        "# Aggregate on the controller CPU directly from shared NFS, even if one persistent SSH session failed.\n"
        "_finalize_namespace = {\n"
        "    'REMOTE_WORK': _base_results_root, 'SWEEP_SHARED_CHECKPOINT_DIR': SWEEP_SHARED_CHECKPOINT_DIR,\n"
        "    'SWEEP_RESULTS_ROOT': _base_results_root, 'SWEEP_RESULTS_BASENAME': SWEEP_RESULTS_BASENAME,\n"
        "    'SWEEP_EXECUTION_MODE': 'multi-node-parallel-workers', 'SWEEP_SPEC': SWEEP_SPEC,\n"
        "    'SWEEP_HASH': SWEEP_HASH, 'SWEEP_CODE_FINGERPRINT': SWEEP_CODE_FINGERPRINT,\n"
        "    'SWEEP_CASES': SWEEP_CASES, 'SWEEP_STATIC_GEOMETRY': SWEEP_STATIC_GEOMETRY,\n"
        "    'SWEEP_RECOMPUTE_MODES': SWEEP_RECOMPUTE_MODES, 'SWEEP_NOMINAL_PARAMETERS': SWEEP_NOMINAL_PARAMETERS,\n"
        "    'SETTINGS': SETTINGS, 'MATERIAL_STACK': MATERIAL_STACK, 'BOUNDING_BOX_UM': BOUNDING_BOX_UM,\n"
        "    'SOURCE_COMPONENTS_JSON': SOURCE_COMPONENTS_JSON, 'GEOMETRY': GEOMETRY, 'PORTS': PORTS, 'FIBER_GEOMETRIES': FIBER_GEOMETRIES, 'GAUSSIAN_SOURCES': GAUSSIAN_SOURCES,\n"
        "    'MONITORS': MONITORS, 'GRATING_ANALYSIS': GRATING_ANALYSIS, 'MMI_ANALYSIS': MMI_ANALYSIS,\n"
        "}\n"
        "try:\n"
        "    exec(REMOTE_SWEEP_RUNTIME, _finalize_namespace)\n"
        "    effective_failures = [failure for failure in MULTIGPU_FAILURES if not _finalize_namespace['_sweep_case_is_complete'](failure['index'])]\n"
        "    _best_multigpu_index = int(_finalize_namespace['_finalize_sweep_results'](effective_failures))\n"
        "    _best_fsp_name = os.path.basename(str(SETTINGS.get('project_file', 'lumerical_sweep.fsp')))\n"
        "    if not _best_fsp_name.lower().endswith('.fsp'):\n"
        "        _best_fsp_name += '.fsp'\n"
        "    REMOTE_BEST_SWEEP_FSP = os.path.join(os.path.dirname(REMOTE_PROJECT_FILE), 'best_' + _best_fsp_name)\n"
        "    _best_work = os.path.join(MULTIGPU_WORKER_ROOT, 'winning_fsp_replay')\n"
        "    _best_packs_acquired = False\n"
        "    _best_fdtd_open = False\n"
        "    try:\n"
        "        _best_checkout = _multigpu_run_checked(lam, MULTIGPU_LICENSE_CHECKOUT_REMOTE, 'Reserve controller licences for winning-FSP replay', timeout=300)\n"
        "        if '__MULTIGPU_LICENSE_ACQUIRED__' not in _best_checkout:\n"
        "            raise RuntimeError('Winning-FSP replay licence marker missing')\n"
        "        _best_packs_acquired = True\n"
        "        CONTROLLER_PACKS_CHECKED_OUT = True\n"
        "        _multigpu_run_checked(lam, _worker_payload(_best_work), 'Load shared seed for winning-FSP replay', timeout=1800)\n"
        "        _best_fdtd_open = True\n"
        "        _best_save_code = (\n"
        "            '_apply_sweep_case(' + repr(_best_multigpu_index) + ')\\n'\n"
        "            + 'fdtd.run(\"FDTD\", \"GPU\")\\n'\n"
        "            + 'fdtd.save(' + repr(REMOTE_BEST_SWEEP_FSP) + ')\\n'\n"
        "            + 'import os\\n'\n"
        "            + 'assert os.path.isfile(' + repr(REMOTE_BEST_SWEEP_FSP) + ') and os.path.getsize(' + repr(REMOTE_BEST_SWEEP_FSP) + ') > 0\\n'\n"
        "            + 'print(\"Saved solved winning multi-GPU FSP.\")'\n"
        "        )\n"
        "        _multigpu_run_checked(lam, _best_save_code, 'Solve and save winning multi-GPU design', timeout=21600)\n"
        "        _results_json_path = _finalize_namespace['SWEEP_RESULTS_JSON']\n"
        "        with open(_results_json_path, 'r', encoding='utf-8') as _best_manifest_stream:\n"
        "            _best_manifest = json.load(_best_manifest_stream)\n"
        "        _best_manifest['best_fsp'] = REMOTE_BEST_SWEEP_FSP\n"
        "        with open(_results_json_path, 'w', encoding='utf-8') as _best_manifest_stream:\n"
        "            json.dump(_best_manifest, _best_manifest_stream, indent=2)\n"
        "        with open(_finalize_namespace['SWEEP_TEXT_SUMMARY'], 'a', encoding='utf-8') as _best_summary_stream:\n"
        "            _best_summary_stream.write('\\nWinning solved FSP\\n------------------\\n' + REMOTE_BEST_SWEEP_FSP + '\\n')\n"
        "        print('Saved the solved winning multi-GPU design:', REMOTE_BEST_SWEEP_FSP)\n"
        "    finally:\n"
        "        if _best_fdtd_open:\n"
        "            try:\n"
        "                _multigpu_run_checked(lam, 'import os\\nfdtd.close()\\nglobals().pop(\"fdtd\", None)\\n_p = globals().get(\"WORKER_RUNTIME_FSP\", \"\")\\nif _p and os.path.isfile(_p): os.remove(_p)', 'Close winning-FSP replay FDTD', timeout=180)\n"
        "            except BaseException as _best_close_exc:\n"
        "                MULTIGPU_FATAL_ERRORS.append({'worker': 'winning-FSP replay', 'error': 'close: ' + str(_best_close_exc)})\n"
        "        if _best_packs_acquired:\n"
        "            try:\n"
        "                _best_release = _multigpu_run_checked(lam, MULTIGPU_LICENSE_RELEASE_REMOTE, 'Return winning-FSP replay licences', timeout=300)\n"
        "                if '__MULTIGPU_LICENSE_RELEASED__' not in _best_release:\n"
        "                    raise RuntimeError('Winning-FSP replay release marker missing')\n"
        "                CONTROLLER_PACKS_CHECKED_OUT = False\n"
        "            except BaseException as _best_release_exc:\n"
        "                CONTROLLER_PACKS_CHECKED_OUT = True\n"
        "                MULTIGPU_FATAL_ERRORS.append({'worker': 'winning-FSP replay', 'error': 'release: ' + str(_best_release_exc)})\n"
        "except BaseException as finalize_exc:\n"
        "    MULTIGPU_FATAL_ERRORS.append({'worker': 'checkpoint aggregation', 'error': str(finalize_exc)})\n"
        "\n"
        "try:\n"
        "    _multigpu_run_checked(\n"
        "        lam,\n"
        "        'import os\\n_p = globals().get(\"REMOTE_RUNTIME_PROJECT_FILE\", \"\")\\nif _p and os.path.isfile(_p): os.remove(_p)\\nprint(\"Controller runtime FSP cleanup complete.\")',\n"
        "        'Remove controller transient runtime FSP', timeout=120,\n"
        "    )\n"
        "except BaseException as _controller_runtime_cleanup_exc:\n"
        "    print('Controller runtime-FSP cleanup warning:', str(_controller_runtime_cleanup_exc)[:240])\n"
        "uncertain = [record['label'] for record in MULTIGPU_WORKER_RECORDS if record.get('packs_checked_out') or record.get('cleanup_uncertain')]\n"
        "if uncertain:\n"
        "    MULTIGPU_FATAL_ERRORS.append({'worker': 'licence cleanup', 'error': 'Unconfirmed cleanup: ' + ', '.join(uncertain)})\n"
        "if MULTIGPU_FATAL_ERRORS:\n"
        "    raise RuntimeError('Multi-GPU sweep completed with fatal worker/cleanup errors after aggregating every valid checkpoint: ' + repr(MULTIGPU_FATAL_ERRORS)) from _orchestration_exception\n"
        "print('All valid checkpoints were aggregated; every worker confirmed FDTD stopped and returned its HPC Packs.')\n"
    )
    orchestration_cell = (
        "if not SETTINGS.get('run_after_build', True):\n"
        "    print('Multi-GPU sweep disabled by RUN_SIMULATION in cell 1.')\n"
        "else:\n"
        + textwrap.indent(orchestration_cell, "    ")
    )

    local_results_cell = _SWEEP_LOCAL_RESULTS_CELL.replace(
        "lumerical_sweep_results", "lumerical_sweep_multithread_results"
    )
    local_results_cell = (
        "if not SETTINGS.get('run_after_build', True):\n"
        "    print('Multi-GPU sweep was not run, so there are no sweep results to fetch.')\n"
        "else:\n"
        + textwrap.indent(local_results_cell, "    ")
    )
    intro = f"""# Max Layout → Lumerical sweep-multithread

This notebook runs **{len(sweep_cases)} Cartesian sweep points** for **{sweep_spec['component_kind']}** on up to **{multigpu_settings['node_count']} independent A100 nodes**.

- It is a separate export; the ordinary one-session sweep notebook is unchanged.
- Production mode runs exactly one simulation per GPU. Add A100 nodes for more concurrency; same-GPU oversubscription is intentionally disabled.
- The controller builds one nominal model and always saves it as the shared inspection FSP used to start the independent GPU workers. No FSP is saved per sweep case.
- Workers use unique workspaces and persistent Lumerical processes, while a dynamic queue balances sweep points across nodes.
- Atomic checkpoints are shared on `/lambda/nfs`, so rerunning resumes completed cases without rebuilding or resolving them.
- Every worker switches back to CPU, closes FDTD, and returns its own three roamed HPC Packs in `finally` cleanup before local CPU plotting.
- {multigpu_settings['max_parallel_simulations']} simultaneous workers require that many FDTD solve seats and {3 * multigpu_settings['max_parallel_simulations']} available HPC Packs.

This notebook requires the updated Piris **Requirements** folder and **3D Launcher** to validate the selected pre-provisioned nodes and publish `PIRIS_LUMERICAL_INVENTORY` securely. Every node must already be an idle 1xA100 with shared NFS access and its own private `~/remote-token.json`; the launcher never copies authentication tokens between nodes.
Keep the Piris 3D Launcher terminal open until the sweep and final cleanup cell have finished; closing it revokes the per-session worker key.
"""
    notebook = {
        "cells": [
            _notebook_cell("code", _quick_run_options_cell(configuration, workflow="multi-gpu sweep")),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Embedded sweep specification\n"),
            _notebook_cell("code", payload_cell),
            _notebook_cell("markdown", "## 2 · Multi-GPU execution settings\n"),
            _notebook_cell("code", multigpu_settings_cell),
            _notebook_cell("markdown", "## 3 · Connect to the controller\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 4 · Validate A100 inventory (no licence checkout yet)\n"),
            _notebook_cell("code", _MULTIGPU_INVENTORY_AND_LICENSE_CELL),
            _notebook_cell("markdown", "## 5 · Build one shared nominal model\n"),
            _notebook_cell("code", remote_build_cell),
            _notebook_cell("markdown", "## 6 · Verify nominal geometry and union FDTD domain\n"),
            _notebook_cell("code", geometry_projection_cell),
            _notebook_cell("markdown", "## 7 · Verify nominal port modes\n"),
            _notebook_cell("code", port_mode_profiles_cell),
            _notebook_cell("markdown", "## 8 · Configure the controller GPU and prepare one shared worker seed\n"),
            _notebook_cell("code", resource_save_cell),
            _notebook_cell("markdown", "## 9 · Inspect the saved nominal FSP\n"),
            _notebook_cell("code", review_project_cell),
            _notebook_cell("markdown", "## 10 · Run sweep points concurrently on independent A100 workers\n"),
            _notebook_cell("code", orchestration_cell),
            _notebook_cell("markdown", "## 11 · Fetch once and plot locally on CPU\n"),
            _notebook_cell("code", local_results_cell),
            _notebook_cell("markdown", "## 12 · Emergency licence recovery\n\nAlways run this after an interruption. It is safe after a successful sweep.\n"),
            _notebook_cell("code", _MULTIGPU_RECOVERY_CELL),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "max_layout": {
                "export": "lumerical-fdtd-sweep-multigpu",
                "units": "um",
                "dimension": "3D",
                "point_count": len(sweep_cases),
                "execution": "multi-node-parallel-workers",
                "nominal_fsp_count": 1,
                "per_point_fsp": False,
                "lumerical_multigpu": multigpu_settings,
                "requirements_update_required": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, warnings


def write_lumerical_sweep_notebook(
    path: str | Path,
    sweep_cases: list[dict[str, Any]],
    configuration: dict[str, Any],
    sweep_spec: dict[str, Any],
    nominal_components: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Write an optimized self-contained Lumerical parameter-sweep notebook."""
    notebook, warnings = generate_lumerical_sweep_notebook(
        sweep_cases,
        configuration,
        sweep_spec,
        nominal_components=nominal_components,
    )
    Path(path).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return warnings


def write_lumerical_multigpu_sweep_notebook(
    path: str | Path,
    sweep_cases: list[dict[str, Any]],
    configuration: dict[str, Any],
    sweep_spec: dict[str, Any],
    nominal_components: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Write the separate multi-node A100 parameter-sweep notebook."""
    notebook, warnings = generate_lumerical_multigpu_sweep_notebook(
        sweep_cases,
        configuration,
        sweep_spec,
        nominal_components=nominal_components,
    )
    Path(path).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return warnings


def write_lumerical_notebook(
    path: str | Path,
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> list[str]:
    """Always write a notebook; questionable settings are recorded as warnings."""
    notebook, warnings = generate_lumerical_notebook(components, configuration)
    Path(path).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return warnings
