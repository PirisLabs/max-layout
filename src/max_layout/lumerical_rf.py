"""Dedicated Lumerical RF notebook export for coplanar-waveguide devices.

The optical exporter intentionally remains separate.  A uniform ``CPW`` is a
2D Z-normal MODE/FDE cross-section problem.  Longitudinal discontinuities
(tapers, bends, opens and shorts) are 3D FDTD problems with explicit RF modal
reference planes.  The generated notebooks keep all dimensions in SI units at
the lumapi boundary and preserve the launcher's Shared-Web licence sandwich.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import pprint
from typing import Any, Iterable

import numpy as np

from .gds.build import component_geometry_arrays
from .lumerical import (
    _LAMBDA_CONNECT_CELL,
    _LICENSE_CHECKOUT_CELL,
    _PIRIS_PATHS_CELL,
    _notebook_cell,
)
from .rf_defaults import (
    RF_FDE_COMPONENT_KINDS,
    RF_FDTD_COMPONENT_KINDS,
    RF_SIMULATABLE_COMPONENT_KINDS,
    normalize_rf_configuration,
)


RF_MODE_PORT_KIND = "RF mode port"
RF_POWER_MONITOR_KIND = "RF power monitor"
RF_SIMULATION_ONLY_KINDS = {RF_MODE_PORT_KIND, RF_POWER_MONITOR_KIND}
RF_ONE_PORT_KINDS = {"CPW open", "CPW short"}


def _source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def _selected_components(
    components: Iterable[dict[str, Any]], configuration: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_components = [deepcopy(component) for component in components]
    scope = {int(value) for value in configuration.get("scope_uids", [])}
    physical = [
        component for component in all_components
        if str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
        and (not scope or int(component.get("uid", -1)) in scope)
    ]
    if not physical:
        raise ValueError("Select a CPW, CPW taper, bend, open, short, or segmented electrode for RF export.")
    parent_uids = {int(component.get("uid", -1)) for component in physical}
    simulation = [
        component for component in all_components
        if str(component.get("kind", "")) in RF_SIMULATION_ONLY_KINDS
        and (
            not scope
            or int(component.get("uid", -1)) in scope
            or int(component.get("simulation_parent_uid", -2)) in parent_uids
        )
    ]
    return physical, simulation


def _rotation_to_local(component: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return layout origin and global-to-component-local XY rotation."""
    angle = math.radians(float(component.get("orientation_deg", 0.0)))
    origin = np.asarray(
        [float(component.get("x", 0.0)), float(component.get("y", 0.0))], dtype=float
    )
    # Row vectors multiply this matrix: global delta @ rotation -> local.
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    return origin, rotation


def _apply_local_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a 2x2 row-vector transform without Accelerate's noisy tiny GEMM path."""
    values = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            values[..., 0] * rotation[0, 0] + values[..., 1] * rotation[1, 0],
            values[..., 0] * rotation[0, 1] + values[..., 1] * rotation[1, 1],
        )
    )


def _collect_geometry(
    physical: list[dict[str, Any]],
    primary: dict[str, Any],
    metal_gds_layers: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    origin, rotation = _rotation_to_local(primary)
    geometry: list[dict[str, Any]] = []
    for component in physical:
        polygons, _labels = component_geometry_arrays(component)
        for index, (vertices, layer, datatype) in enumerate(polygons, start=1):
            if metal_gds_layers is not None and int(layer) not in metal_gds_layers:
                continue
            local = _apply_local_rotation(np.asarray(vertices, dtype=float) - origin, rotation)
            geometry.append(
                {
                    "name": "uid_%s_rf_%d" % (component.get("uid", 0), index),
                    "component_uid": int(component.get("uid", 0)),
                    "component_kind": str(component.get("kind", "")),
                    "layer": int(layer),
                    "datatype": int(datatype),
                    "vertices_um": local.tolist(),
                }
            )
    if not geometry:
        suffix = (
            " on configured RF metal GDS layer(s) "
            + ", ".join(map(str, sorted(metal_gds_layers)))
            if metal_gds_layers is not None
            else ""
        )
        raise ValueError("The selected RF component produced no metal polygons" + suffix + ".")
    points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in geometry])
    lower, upper = points.min(axis=0), points.max(axis=0)
    return geometry, [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]


def _rf_metal_gds_layers(stack: Iterable[dict[str, Any]]) -> set[int]:
    """Return the exact GDS layers assigned to the one RF metal row."""
    metal_rows = [
        row for row in stack if str(row.get("role", "")).strip().lower() == "metal"
    ]
    if len(metal_rows) != 1:
        raise ValueError("The RF stack must contain exactly one metal row.")
    raw_layers = metal_rows[0].get("gds_layers", [])
    if isinstance(raw_layers, (str, int, float)):
        raw_layers = [raw_layers]
    layers: set[int] = set()
    for raw_layer in raw_layers:
        numeric = float(raw_layer)
        if not math.isfinite(numeric) or numeric < 0.0 or abs(numeric - round(numeric)) > 1e-12:
            raise ValueError("RF metal GDS layers must be nonnegative whole numbers.")
        layers.add(int(round(numeric)))
    if not layers:
        raise ValueError("The RF metal row needs at least one GDS layer.")
    return layers


def _local_simulation_object(
    component: dict[str, Any], primary: dict[str, Any]
) -> dict[str, Any]:
    origin, rotation = _rotation_to_local(primary)
    params = deepcopy(component.get("params", {}))
    center_global = np.asarray(
        [float(component.get("x", 0.0)), float(component.get("y", 0.0))], dtype=float
    )
    center = _apply_local_rotation((center_global - origin).reshape(1, 2), rotation)[0]
    relative_angle = (
        float(component.get("orientation_deg", 0.0))
        - float(primary.get("orientation_deg", 0.0))
    ) % 360.0
    local_normal = str(params.get("plane normal", "X")).strip().upper()
    normal_angle = relative_angle + (90.0 if local_normal == "Y" else 0.0)
    nearest = int(round(normal_angle / 90.0) * 90) % 360
    mismatch = abs(((normal_angle - nearest + 180.0) % 360.0) - 180.0)
    if mismatch > 1e-6:
        raise ValueError(
            "%s %s has a non-cardinal local plane (%.6g degrees); RF FDTD reference planes "
            "must be X- or Y-normal after rotating into the device frame."
            % (component.get("kind", "RF object"), params.get("name", "unnamed"), normal_angle)
        )
    distance = float(params.get("distance_um", 0.0))
    center += distance * np.asarray(
        [math.cos(math.radians(nearest)), math.sin(math.radians(nearest))]
    )
    return {
        "name": str(params.get("name") or "uid_%s_rf" % component.get("uid", 0)),
        "kind": str(component.get("kind", "")),
        "component_uid": int(component.get("uid", 0)),
        "parent_component_uid": int(component.get("simulation_parent_uid", -1)),
        "rf_role": str(params.get("rf role", params.get("rf_role", ""))).strip(),
        "center_um": [float(center[0]), float(center[1])],
        "plane_normal": "X" if nearest in (0, 180) else "Y",
        "normal_angle_deg": float(nearest),
        "direction": "Forward" if nearest in (0, 90) else "Backward",
        "span_um": float(params.get("span_um", 450.0)),
        "z_span_um": float(params.get("z_span_um", 650.0)),
        "mode": str(params.get("mode", "fundamental mode")),
        "order": int(params.get("order", 1)),
        "reference_impedance_ohm": float(params.get("reference impedance_ohm", 50.0)),
        "deembed_um": float(params.get("deembed_um", 0.0)),
        "multifrequency_mode_injection": bool(
            params.get("multifrequency mode injection", True)
        ),
        "expansion_port": str(params.get("expansion port", "")).strip(),
    }


def _endpoint_planes(
    primary: dict[str, Any], bounds: list[float], configuration: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Optional deterministic fallback, never used unless explicitly requested."""
    kind = str(primary.get("kind", ""))
    y_center = 0.5 * (bounds[1] + bounds[3])
    inset_in = max(0.0, float(configuration.get("input_port_inset_um", 0.0)))
    inset_out = max(0.0, float(configuration.get("output_port_inset_um", 0.0)))
    common = {
        "kind": RF_MODE_PORT_KIND,
        "component_uid": -1,
        "parent_component_uid": int(primary.get("uid", -1)),
        "span_um": float(configuration.get("port_transverse_span_um", 450.0)),
        "z_span_um": float(configuration.get("port_vertical_span_um", 650.0)),
        "mode": "fundamental mode",
        "order": 1,
        "reference_impedance_ohm": 50.0,
        "deembed_um": 0.0,
        "multifrequency_mode_injection": bool(
            configuration.get("multifrequency_mode_injection", True)
        ),
        "expansion_port": "",
    }
    source = {
        **common,
        "name": "rf_source",
        "rf_role": "Source",
        "center_um": [bounds[0] + inset_in, y_center],
        "plane_normal": "X",
        "normal_angle_deg": 0.0,
        "direction": "Forward",
    }
    ports = [source]
    monitors = [
        {
            **source,
            "kind": RF_POWER_MONITOR_KIND,
            "name": "rf_input_reference",
            "rf_role": "Input reference",
            "expansion_port": "rf_source",
        }
    ]
    if kind not in RF_ONE_PORT_KINDS:
        if kind == "CPW bend":
            bend_angle = float(primary.get("params", {}).get("bend_angle_deg", 90.0))
            nearest = int(round(bend_angle / 90.0) * 90) % 360
            if abs(bend_angle - nearest) > 1e-6:
                raise ValueError(
                    "Automatic CPW-bend endpoint ports support only cardinal bend angles; place manual RF planes."
                )
            if nearest in (90, 270):
                output_center = [
                    0.5 * (bounds[0] + bounds[2]),
                    bounds[3] - inset_out if nearest == 90 else bounds[1] + inset_out,
                ]
                plane = "Y"
                direction = "Forward" if nearest == 90 else "Backward"
            else:
                output_center = [bounds[2] - inset_out, y_center]
                plane = "X"
                direction = "Forward"
        else:
            output_center = [bounds[2] - inset_out, y_center]
            plane = "X"
            direction = "Forward"
        output = {
            **common,
            "name": "rf_output",
            "rf_role": "Output",
            "center_um": output_center,
            "plane_normal": plane,
            "normal_angle_deg": 90.0 if plane == "Y" else 0.0,
            "direction": direction,
        }
        ports.append(output)
        monitors.append(
            {
                **output,
                "kind": RF_POWER_MONITOR_KIND,
                "name": "rf_output_reference",
                "rf_role": "Output",
                "expansion_port": "rf_output",
            }
        )
    return ports, monitors


def _collect_rf_planes(
    simulation: list[dict[str, Any]],
    primary: dict[str, Any],
    bounds: list[float],
    configuration: dict[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ports = [
        _local_simulation_object(component, primary)
        for component in simulation if component.get("kind") == RF_MODE_PORT_KIND
    ]
    monitors = [
        _local_simulation_object(component, primary)
        for component in simulation if component.get("kind") == RF_POWER_MONITOR_KIND
    ]
    strategy = str(configuration.get("rf_port_strategy", "manual")).strip().lower()
    if not ports and not monitors:
        if strategy in {"component_endpoints", "manual_or_component_endpoints"}:
            ports, monitors = _endpoint_planes(primary, bounds, configuration)
            warnings.append(
                "No manual RF planes were present; explicit component-endpoint fallback was used."
            )
        else:
            warnings.append(
                "No RF mode ports or RF power monitors were exported. Add manual RF planes, or explicitly "
                "choose rf_port_strategy='component_endpoints'."
            )
    elif strategy == "manual_or_component_endpoints":
        fallback_ports, fallback_monitors = _endpoint_planes(primary, bounds, configuration)
        existing_port_roles = {_role(port.get("rf_role")) for port in ports}
        existing_plane_roles = {
            _role(item.get("rf_role")) for item in [*ports, *monitors]
        }
        for fallback in fallback_ports:
            fallback_role = _role(fallback.get("rf_role"))
            if fallback_role == "source" and "source" not in existing_port_roles:
                ports.append(fallback)
                existing_port_roles.add("source")
            elif fallback_role == "output" and "output" not in existing_plane_roles:
                ports.append(fallback)
                existing_plane_roles.add("output")
        for fallback in fallback_monitors:
            fallback_role = _role(fallback.get("rf_role"))
            if fallback_role not in existing_plane_roles:
                monitors.append(fallback)
                existing_plane_roles.add(fallback_role)
        warnings.append(
            "Missing manual RF roles were completed from the explicitly enabled component-endpoint fallback."
        )
    return ports, monitors


def _role(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _validate_planes(
    kind: str,
    ports: list[dict[str, Any]],
    monitors: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if kind in RF_FDE_COMPONENT_KINDS:
        return
    source_ports = [port for port in ports if _role(port.get("rf_role")) == "source"]
    if len(source_ports) != 1:
        warnings.append("3D RF FDTD requires exactly one RF mode port with role Source.")
    output_planes = [
        item for item in [*ports, *monitors]
        if _role(item.get("rf_role")) == "output"
    ]
    if kind not in RF_ONE_PORT_KINDS and not output_planes:
        warnings.append("This two-port RF device has no Output reference plane, so S21 cannot be calculated.")


def rf_stack_intervals(
    stack: Iterable[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], float, float]], dict[str, Any]]:
    """Return the RF stack intervals using the notebook's metal-at-z=0 rule."""
    active = [
        deepcopy(row)
        for row in stack
        if float(row.get("thickness_um", 0.0)) > 0.0
        or str(row.get("role", "")).strip().lower() == "metal"
    ]
    metal_indices = [
        index
        for index, row in enumerate(active)
        if str(row.get("role", "")).strip().lower() == "metal"
    ]
    if len(metal_indices) != 1:
        raise ValueError("The RF stack preview requires exactly one metal row.")
    metal_index = metal_indices[0]
    below = active[:metal_index]
    above = active[metal_index + 1 :]
    cursor = -sum(float(row.get("thickness_um", 0.0)) for row in below)
    intervals: list[tuple[dict[str, Any], float, float]] = []
    for row in below:
        thickness_um = float(row.get("thickness_um", 0.0))
        intervals.append((row, cursor, cursor + thickness_um))
        cursor += thickness_um
    cursor = 0.0
    for row in above:
        thickness_um = float(row.get("thickness_um", 0.0))
        intervals.append((row, cursor, cursor + thickness_um))
        cursor += thickness_um
    return intervals, active[metal_index]


def _rf_preview_material(row: dict[str, Any]) -> str:
    """Map RF electrical rows to stable visual material colors."""
    if str(row.get("role", "")).strip().lower() == "metal":
        return "RF metal"
    name = str(row.get("name", "RF dielectric"))
    lowered = name.lower()
    epsilon = float(
        row.get("relative_permittivity", row.get("relative_permittivity_x", 1.0))
    )
    if epsilon <= 1.000001 or "air" in lowered:
        return "Air"
    if "sio2" in lowered or "silica" in lowered or "oxide" in lowered:
        return "RF SiO2"
    if "tfln" in lowered or "linbo" in lowered or "lithium niobate" in lowered:
        return "RF LiNbO3"
    if "silicon" in lowered or lowered.startswith("si ") or "si handle" in lowered:
        return "RF silicon"
    if "fr4" in lowered:
        return "RF FR4"
    return "RF dielectric"


def _rf_preview_label(row: dict[str, Any]) -> str:
    name = str(row.get("name", "RF layer"))
    if str(row.get("role", "")).strip().lower() == "metal":
        model = str(row.get("metal_model", "Conductive 3D"))
        conductivity = float(row.get("conductivity_s_per_m", 0.0))
        return "%s — %s, sigma=%.6g S/m" % (name, model, conductivity)
    loss_tangent = float(row.get("loss_tangent", 0.0))
    if bool(row.get("anisotropic", False)):
        epsilon_text = "epsilon=(%.6g, %.6g, %.6g)" % (
            float(row["relative_permittivity_x"]),
            float(row["relative_permittivity_y"]),
            float(row["relative_permittivity_z"]),
        )
    else:
        epsilon_text = "epsilon_r=%.6g" % float(row.get("relative_permittivity", 1.0))
    return "%s — %s, tan(delta)=%.6g" % (name, epsilon_text, loss_tangent)


def build_lumerical_rf_preview_state(
    components: Iterable[dict[str, Any]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact local-frame stack, metal, planes, and solver box for UI preview.

    FDTD previews use the same frequency-dependent clearance equations as the
    generated notebook.  A straight CPW is honestly represented as its 2D
    MODE/FDE cross-section extruded by one micrometre for visualization only.
    """
    raw_configuration = deepcopy(configuration)
    physical, simulation = _selected_components(components, raw_configuration)
    requested_primary_uid = raw_configuration.get("primary_component_uid")
    primary = next(
        (
            component
            for component in physical
            if requested_primary_uid is not None
            and int(component.get("uid", -1)) == int(requested_primary_uid)
        ),
        physical[0],
    )
    primary_kind = str(primary.get("kind", ""))
    workflows = {
        "fde"
        if str(component.get("kind", "")) in RF_FDE_COMPONENT_KINDS
        else "fdtd"
        for component in physical
    }
    if len(workflows) != 1:
        raise ValueError(
            "Preview uniform CPW FDE cross-sections separately from 3D CPW discontinuities."
        )
    if "target_frequency_ghz" not in raw_configuration:
        start = float(raw_configuration.get("frequency_start_ghz", 1.0))
        stop = float(raw_configuration.get("frequency_stop_ghz", 100.0))
        raw_configuration["target_frequency_ghz"] = 0.5 * (start + stop)
    settings = normalize_rf_configuration(primary_kind, raw_configuration)
    workflow = str(settings["rf_workflow"])
    metal_gds_layers = _rf_metal_gds_layers(settings["material_stack"])
    geometry, device_bounds = _collect_geometry(
        physical, primary, metal_gds_layers
    )
    warnings: list[str] = []
    rf_ports: list[dict[str, Any]] = []
    rf_monitors: list[dict[str, Any]] = []
    if workflow == "fdtd":
        rf_ports, rf_monitors = _collect_rf_planes(
            simulation, primary, device_bounds, settings, warnings
        )
        _validate_planes(primary_kind, rf_ports, rf_monitors, warnings)

    intervals, metal_row = rf_stack_intervals(settings["material_stack"])
    effective_metal_row = deepcopy(metal_row)
    effective_metal_row["metal_model"] = str(
        settings.get("metal_model", metal_row.get("metal_model", "Conductive 3D"))
    )
    effective_metal_row["conductivity_s_per_m"] = float(
        settings.get(
            "metal_conductivity_s_per_m",
            metal_row.get("conductivity_s_per_m", 0.0),
        )
    )
    metal_thickness_um = max(
        float(settings.get("mesh_vertical_um", 0.1)),
        float(
            settings.get(
                "metal_thickness_um",
                effective_metal_row.get("thickness_um", 0.0),
            )
        ),
    )
    geometry_layers = sorted({int(item["layer"]) for item in geometry})
    stack_ranges: list[tuple[dict[str, Any], float, float]] = []
    for row, z0_um, z1_um in intervals:
        preview_row = deepcopy(row)
        preview_row.update(
            {
                "role": "background",
                "material": _rf_preview_material(row),
                "gds_layers": [],
                "_preview_label": _rf_preview_label(row),
            }
        )
        stack_ranges.append((preview_row, float(z0_um), float(z1_um)))
    preview_metal = deepcopy(effective_metal_row)
    preview_metal.update(
        {
            "role": "geometry",
            "material": _rf_preview_material(metal_row),
            "thickness_um": metal_thickness_um,
            "etch_depth_um": metal_thickness_um,
            "sidewall_angle_deg": 90.0,
            "gds_layers": geometry_layers,
            "slab_extent": "geometry",
            "_preview_label": _rf_preview_label(effective_metal_row),
        }
    )
    stack_ranges.append((preview_metal, 0.0, metal_thickness_um))
    preview_planes: list[dict[str, Any]] = []
    if workflow == "fde":
        params = primary.get("params", {})
        signal_width = float(params.get("signal_width", 130.0))
        gap = float(params.get("gap", params.get("initial_gap", 3.0)))
        ground_width = float(params.get("ground_width", 130.0))
        total_width = signal_width + 2.0 * (gap + ground_width)
        transverse_padding = max(
            2.0 * float(settings["mesh_bulk_um"]), 0.25 * total_width
        )
        domain_x0 = -0.5 * total_width - transverse_padding
        domain_x1 = 0.5 * total_width + transverse_padding
        domain_y0, domain_y1 = -0.5, 0.5
        domain_z0 = min(
            [z0_um for _row, z0_um, _z1_um in intervals]
            + [-float(settings["substrate_thickness_um"])]
        )
        domain_z1 = max(
            [z1_um for _row, _z0_um, z1_um in intervals]
            + [0.25 * total_width]
        )
        metal_layer = geometry_layers[0] if geometry_layers else 4
        metal_cross_sections = (
            (-0.5 * signal_width, 0.5 * signal_width),
            (
                -0.5 * signal_width - gap - ground_width,
                -0.5 * signal_width - gap,
            ),
            (
                0.5 * signal_width + gap,
                0.5 * signal_width + gap + ground_width,
            ),
        )
        preview_polygons = [
            (
                np.asarray(
                    [
                        [x0_um, domain_y0],
                        [x1_um, domain_y0],
                        [x1_um, domain_y1],
                        [x0_um, domain_y1],
                    ],
                    dtype=float,
                ),
                metal_layer,
            )
            for x0_um, x1_um in metal_cross_sections
        ]
        base_x = (domain_x0, domain_x1)
        base_y = (domain_y0, domain_y1)
        base_z = (domain_z0, domain_z1)
        backing_z0_um = domain_z0
        solver_label = "MODE/FDE 2D cross-section (1 um preview extrusion; CPU solve)"
    else:
        preview_polygons = [
            (np.asarray(item["vertices_um"], dtype=float), int(item["layer"]))
            for item in geometry
        ]
        frequency_stop_hz = float(settings["frequency_stop_ghz"]) * 1e9
        lambda_min_um = 299792458.0 / frequency_stop_hz / 1e-6
        clearance_um = max(
            5.0 * float(settings["mesh_bulk_um"]),
            float(settings.get("port_clearance_wavelengths", 0.25))
            * lambda_min_um,
        )
        domain_x0 = float(device_bounds[0]) - clearance_um
        domain_x1 = float(device_bounds[2]) + clearance_um
        domain_y0 = float(device_bounds[1]) - clearance_um
        domain_y1 = float(device_bounds[3]) + clearance_um
        z_min_stack = min(
            [z0_um for _row, z0_um, _z1_um in intervals]
            + [-float(settings["substrate_thickness_um"])]
        )
        z_max_stack = max(
            [z1_um for _row, _z0_um, z1_um in intervals]
            + [metal_thickness_um]
        )
        z_padding_um = max(
            5.0 * float(settings["mesh_bulk_um"]), 0.05 * lambda_min_um
        )
        domain_z0 = z_min_stack - z_padding_um
        domain_z1 = z_max_stack + z_padding_um
        plane_z_um = 0.5 * (z_min_stack + z_max_stack)
        preview_records = [*rf_ports, *rf_monitors]
        plane_roles = {_role(record.get("rf_role")) for record in preview_records}
        source_records = [
            record for record in rf_ports if _role(record.get("rf_role")) == "source"
        ]
        if source_records and "input reference" not in plane_roles:
            input_reference = deepcopy(source_records[0])
            input_reference.update(
                {
                    "kind": RF_POWER_MONITOR_KIND,
                    "name": str(source_records[0].get("name", "rf_source"))
                    + "_reference",
                    "rf_role": "Input reference",
                }
            )
            preview_records.append(input_reference)
        for record in preview_records:
            center_x_um, center_y_um = map(float, record["center_um"])
            preview_planes.append(
                {
                    "uid": int(record.get("component_uid", -1)),
                    "kind": str(record.get("kind", RF_POWER_MONITOR_KIND)),
                    "x": center_x_um,
                    "y": center_y_um,
                    "orientation_deg": 0.0,
                    "params": {
                        "name": str(record.get("name", "RF plane")),
                        "plane normal": str(record.get("plane_normal", "X")),
                        "span_um": float(record.get("span_um", 450.0)),
                        "z_span_um": float(record.get("z_span_um", 650.0)),
                        "z center_um": plane_z_um,
                        "rf role": str(record.get("rf_role", "")),
                    },
                }
            )
        x_values = [float(device_bounds[0]), float(device_bounds[2])]
        y_values = [float(device_bounds[1]), float(device_bounds[3])]
        z_values = [z_min_stack, z_max_stack]
        for plane in preview_planes:
            params = plane["params"]
            half_span = 0.5 * float(params["span_um"])
            half_z_span = 0.5 * float(params["z_span_um"])
            if str(params["plane normal"]).upper() == "X":
                x_values.append(float(plane["x"]))
                y_values.extend((float(plane["y"]) - half_span, float(plane["y"]) + half_span))
            else:
                x_values.extend((float(plane["x"]) - half_span, float(plane["x"]) + half_span))
                y_values.append(float(plane["y"]))
            z_values.extend((plane_z_um - half_z_span, plane_z_um + half_z_span))
        base_x = (min(x_values), max(x_values))
        base_y = (min(y_values), max(y_values))
        base_z = (min(z_values), max(z_values))
        backing_z0_um = z_min_stack
        solver_label = "3D RF FDTD domain (GPU field solve)"

    if bool(settings.get("backing_ground", False)):
        # The backing conductor is solver geometry rather than a GDS layer.
        # Give it a private preview layer so it can be hidden independently
        # while preserving the exact vertical placement used by the notebook.
        backing_layer = -2_147_483_648
        preview_polygons.append(
            (
                np.asarray(
                    [
                        [domain_x0, domain_y0],
                        [domain_x1, domain_y0],
                        [domain_x1, domain_y1],
                        [domain_x0, domain_y1],
                    ],
                    dtype=float,
                ),
                backing_layer,
            )
        )
        backing_row = deepcopy(effective_metal_row)
        backing_row.update(
            {
                "name": "Backing ground",
                "role": "geometry",
                "material": "RF metal",
                "thickness_um": metal_thickness_um,
                "etch_depth_um": metal_thickness_um,
                "sidewall_angle_deg": 90.0,
                "gds_layers": [backing_layer],
                "slab_extent": "geometry",
                "_preview_label": "Backing ground — "
                + _rf_preview_label(effective_metal_row),
            }
        )
        stack_ranges.append(
            (
                backing_row,
                float(backing_z0_um),
                float(backing_z0_um + metal_thickness_um),
            )
        )

    stack_ranges.sort(key=lambda item: (float(item[1]), float(item[2])))
    stack_ranges = [
        ({**row, "_preview_id": index}, z0_um, z1_um)
        for index, (row, z0_um, z1_um) in enumerate(stack_ranges)
    ]

    padding = {
        "x_min": float(base_x[0] - domain_x0),
        "x_max": float(domain_x1 - base_x[1]),
        "y_min": float(base_y[0] - domain_y0),
        "y_max": float(domain_y1 - base_y[1]),
        "z_min": float(base_z[0] - domain_z0),
        "z_max": float(domain_z1 - base_z[1]),
    }
    return {
        "workflow": workflow,
        "primary_kind": primary_kind,
        "polygons": preview_polygons,
        "x_base": base_x,
        "y_base": base_y,
        "z_base": base_z,
        "padding": padding,
        "stack_ranges": stack_ranges,
        "components": preview_planes,
        "solver_bounds_um": (
            domain_x0,
            domain_x1,
            domain_y0,
            domain_y1,
            domain_z0,
            domain_z1,
        ),
        "solver_label": solver_label,
        "warnings": warnings,
    }


def _literal(value: Any) -> str:
    return pprint.pformat(value, width=110, sort_dicts=True)


_RF_REMOTE_COMMON = r'''import json
import os
import time
import numpy as np
import lumapi

UM = 1e-6
GHZ = 1e9
NS = 1e-9
C0 = 299792458.0
EPS0 = 8.8541878128e-12

os.makedirs(REMOTE_WORK, exist_ok=True)
REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")
os.makedirs(REMOTE_FSP_DIR, exist_ok=True)
os.chdir(REMOTE_WORK)

_previous_owner = globals().get("fdtd")
if _previous_owner is not None:
    try:
        _previous_owner.close()
    except Exception:
        pass
    globals().pop("fdtd", None)

RF_FREQUENCIES_HZ = np.linspace(
    float(SETTINGS["frequency_start_ghz"]) * GHZ,
    float(SETTINGS["frequency_stop_ghz"]) * GHZ,
    int(SETTINGS["frequency_points"]),
)


def _safe_set(owner, key, value):
    try:
        owner.set(key, value)
        return True
    except Exception as exc:
        print("Optional property skipped:", key, str(exc)[:140])
        return False


def _safe_setanalysis(owner, key, value):
    try:
        owner.setanalysis(key, value)
        return True
    except Exception as exc:
        print("Optional analysis property skipped:", key, str(exc)[:140])
        return False


def _rf_stack_intervals():
    """Return dielectric vertical intervals with metal bottom fixed at z=0."""
    active = [row for row in MATERIAL_STACK if float(row.get("thickness_um", 0.0)) > 0.0 or str(row.get("role", "")).lower() == "metal"]
    metal_index = next(i for i, row in enumerate(active) if str(row.get("role", "")).lower() == "metal")
    below = active[:metal_index]
    above = active[metal_index + 1:]
    cursor = -sum(float(row.get("thickness_um", 0.0)) for row in below)
    intervals = []
    for row in below:
        thickness = float(row.get("thickness_um", 0.0))
        intervals.append((row, cursor, cursor + thickness))
        cursor += thickness
    cursor = 0.0
    for row in above:
        thickness = float(row.get("thickness_um", 0.0))
        intervals.append((row, cursor, cursor + thickness))
        cursor += thickness
    metal = active[metal_index]
    return intervals, metal


RF_MATERIAL_NAMES = {}


def _add_rf_dielectric(owner, row):
    key = str(row.get("name", "RF dielectric"))
    if key in RF_MATERIAL_NAMES:
        return RF_MATERIAL_NAMES[key]
    name = "Max Layout RF dielectric " + key
    material_id = owner.addmaterial("Dielectric")
    owner.setmaterial(material_id, "name", name)
    loss_tangent = max(0.0, float(row.get("loss_tangent", 0.0)))
    if bool(row.get("anisotropic", False)):
        epsilon = np.asarray([
            float(row["relative_permittivity_x"]),
            float(row["relative_permittivity_y"]),
            float(row["relative_permittivity_z"]),
        ], dtype=complex) * (1.0 - 1j * loss_tangent)
        owner.setmaterial(name, "Anisotropy", 1)
        owner.setmaterial(name, "Refractive Index", np.sqrt(epsilon))
    else:
        epsilon = complex(float(row.get("relative_permittivity", 1.0)), 0.0) * (1.0 - 1j * loss_tangent)
        owner.setmaterial(name, "Refractive Index", np.sqrt(epsilon))
    RF_MATERIAL_NAMES[key] = name
    return name


def _add_rf_metal(owner, metal_row):
    model = str(SETTINGS.get("metal_model", metal_row.get("metal_model", "PEC"))).strip().lower()
    if model.startswith("pec"):
        return "PEC (Perfect Electrical Conductor)"
    name = "Max Layout RF conductive metal"
    material_id = owner.addmaterial("Conductive 3D")
    owner.setmaterial(material_id, "name", name)
    owner.setmaterial(
        name, "conductivity",
        float(SETTINGS.get("metal_conductivity_s_per_m", metal_row.get("conductivity_s_per_m", 4.1e7))),
    )
    return name


def _add_rect(owner, name, material, x0, x1, y0, y1, z0, z1, mesh_order=2):
    owner.addrect()
    owner.set("name", str(name))
    owner.set("material", str(material))
    owner.set("x min", float(x0) * UM)
    owner.set("x max", float(x1) * UM)
    owner.set("y min", float(y0) * UM)
    owner.set("y max", float(y1) * UM)
    owner.set("z min", float(z0) * UM)
    owner.set("z max", float(z1) * UM)
    _safe_set(owner, "override mesh order from material database", True)
    _safe_set(owner, "mesh order", int(mesh_order))
'''


_RF_FDE_BUILD_REMOTE = _RF_REMOTE_COMMON + r'''
# Official CPW workflow: CPU MODE/FDE, 2D Z-normal cross-section.
fdtd = lumapi.MODE(
    hide=bool(SETTINGS.get("hide_cad", True)),
    serverArgs={"threads": str(int(SETTINGS.get("build_cpu_threads", 30)))},
)
RF_PROJECT_EXTENSION = ".lms"
REMOTE_INSPECTION_PROJECT = os.path.join(REMOTE_FSP_DIR, "rf_cpw_inspection.lms")
REMOTE_FINAL_PROJECT = os.path.join(REMOTE_FSP_DIR, "rf_cpw.lms")

intervals, metal_row = _rf_stack_intervals()
signal_width = float(PRIMARY_PARAMS.get("signal_width", 130.0))
gap = float(PRIMARY_PARAMS.get("gap", PRIMARY_PARAMS.get("initial_gap", 3.0)))
ground_width = float(PRIMARY_PARAMS.get("ground_width", 130.0))
total_width = signal_width + 2.0 * (gap + ground_width)
transverse_padding = max(2.0 * float(SETTINGS["mesh_bulk_um"]), 0.25 * total_width)
x_min = -0.5 * total_width - transverse_padding
x_max = 0.5 * total_width + transverse_padding
y_min = min([z0 for _row, z0, _z1 in intervals] + [-float(SETTINGS["substrate_thickness_um"])])
y_max = max([z1 for _row, _z0, z1 in intervals] + [0.25 * total_width])

for index, (row, y0, y1) in enumerate(intervals, start=1):
    if y1 <= y0 or float(row.get("relative_permittivity", row.get("relative_permittivity_x", 1.0))) == 1.0:
        continue
    material = _add_rf_dielectric(fdtd, row)
    _add_rect(fdtd, "%02d %s" % (index, row.get("name", "dielectric")), material,
              x_min, x_max, y0, y1, -0.5, 0.5, mesh_order=3)

metal_material = _add_rf_metal(fdtd, metal_row)
metal_thickness = max(
    float(SETTINGS.get("mesh_vertical_um", 0.1)),
    float(SETTINGS.get("metal_thickness_um", metal_row.get("thickness_um", 0.0))),
)
_add_rect(fdtd, "signal", metal_material, -0.5 * signal_width, 0.5 * signal_width,
          0.0, metal_thickness, -0.5, 0.5, mesh_order=1)
_add_rect(fdtd, "left ground", metal_material,
          -0.5 * signal_width - gap - ground_width, -0.5 * signal_width - gap,
          0.0, metal_thickness, -0.5, 0.5, mesh_order=1)
_add_rect(fdtd, "right ground", metal_material,
          0.5 * signal_width + gap, 0.5 * signal_width + gap + ground_width,
          0.0, metal_thickness, -0.5, 0.5, mesh_order=1)
if bool(SETTINGS.get("backing_ground", False)):
    _add_rect(fdtd, "backing ground", metal_material, x_min, x_max,
              y_min, y_min + metal_thickness, -0.5, 0.5, mesh_order=1)

fdtd.addfde()
fdtd.set("name", "FDE")
fdtd.set("solver type", "2D Z normal")
fdtd.set("x", 0.5 * (x_min + x_max) * UM)
fdtd.set("x span", (x_max - x_min) * UM)
fdtd.set("y", 0.5 * (y_min + y_max) * UM)
fdtd.set("y span", (y_max - y_min) * UM)
fdtd.set("mesh cells x", max(20, int(np.ceil((x_max - x_min) / float(SETTINGS["mesh_bulk_um"])))))
fdtd.set("mesh cells y", max(20, int(np.ceil((y_max - y_min) / float(SETTINGS["mesh_bulk_um"])))))
for boundary_name in ("x min bc", "x max bc", "y min bc", "y max bc"):
    _safe_set(fdtd, boundary_name, "metal")

fdtd.addmesh()
fdtd.set("name", "CPW metal-edge mesh")
fdtd.set("x", 0.0)
fdtd.set("x span", (total_width + 2.0 * float(SETTINGS["mesh_edge_um"])) * UM)
fdtd.set("y", 0.5 * metal_thickness * UM)
fdtd.set("y span", (metal_thickness + 4.0 * float(SETTINGS["mesh_vertical_um"])) * UM)
fdtd.set("override x mesh", True)
fdtd.set("override y mesh", True)
fdtd.set("dx", float(SETTINGS["mesh_edge_um"]) * UM)
fdtd.set("dy", float(SETTINGS["mesh_vertical_um"]) * UM)

fdtd.setanalysis("number of trial modes", 10)
fdtd.setanalysis("search", "near n")
fdtd.setanalysis("use max index", False)
fdtd.setanalysis("N", 4)
fdtd.setanalysis("calculate group index", True)
_safe_setanalysis(fdtd, "calculate impedance", True)
fdtd.setanalysis("frequency", float(RF_FREQUENCIES_HZ[len(RF_FREQUENCIES_HZ) // 2]))
print("Built official-style 2D Z-normal CPW FDE model on CPU.")
print("The solved quasi-TEM mode profile remains available as MODE result mode1.")
print("RF sweep: %.6g to %.6g GHz, %d points" %
      (RF_FREQUENCIES_HZ[0] / GHZ, RF_FREQUENCIES_HZ[-1] / GHZ, RF_FREQUENCIES_HZ.size))
'''


_RF_FDE_RUN_REMOTE = r'''# Official traveling-wave MODE loop: find mode and read complex RF quantities.
z0_values = np.empty(RF_FREQUENCIES_HZ.size, dtype=complex)
neff_values = np.empty(RF_FREQUENCIES_HZ.size, dtype=complex)
ng_values = np.empty(RF_FREQUENCIES_HZ.size, dtype=complex)
loss_values = np.empty(RF_FREQUENCIES_HZ.size, dtype=float)


def _mode_scalar(key, complex_value=True):
    value = np.asarray(fdtd.getdata("mode1", key)).squeeze().ravel()[0]
    return complex(value) if complex_value else float(np.real(value))


for index, frequency_hz in enumerate(RF_FREQUENCIES_HZ):
    fdtd.switchtolayout()
    fdtd.setanalysis("frequency", float(frequency_hz))
    fdtd.findmodes()
    z0_values[index] = _mode_scalar("Z0")
    neff_values[index] = _mode_scalar("neff")
    ng_values[index] = _mode_scalar("ng")
    loss_values[index] = _mode_scalar("loss", complex_value=False)
    print("MODE point %d/%d: %.6g GHz, Z0=%s" %
          (index + 1, RF_FREQUENCIES_HZ.size, frequency_hz / GHZ, z0_values[index]))

RF_RESULT_ARRAYS = {
    "frequency_hz": RF_FREQUENCIES_HZ,
    "Z0_ohm": z0_values,
    "neff": neff_values,
    "ng": ng_values,
    "loss_db_per_m": loss_values,
    "loss_db_per_cm": loss_values / 100.0,
}
'''


_RF_FDTD_BUILD_REMOTE = _RF_REMOTE_COMMON + r'''
# Official discontinuity workflow: 3D FDTD with modal source and expansion planes.
fdtd = lumapi.FDTD(
    hide=bool(SETTINGS.get("hide_cad", True)),
    serverArgs={"threads": str(int(SETTINGS.get("build_cpu_threads", 30)))},
)
RF_PROJECT_EXTENSION = ".fsp"
REMOTE_INSPECTION_PROJECT = os.path.join(REMOTE_FSP_DIR, "rf_discontinuity_inspection.fsp")
REMOTE_FINAL_PROJECT = os.path.join(REMOTE_FSP_DIR, "rf_discontinuity.fsp")

intervals, metal_row = _rf_stack_intervals()
metal_material = _add_rf_metal(fdtd, metal_row)
metal_thickness = max(
    float(SETTINGS.get("mesh_vertical_um", 0.1)),
    float(SETTINGS.get("metal_thickness_um", metal_row.get("thickness_um", 0.0))),
)
z_min_stack = min([z0 for _row, z0, _z1 in intervals] + [-float(SETTINGS["substrate_thickness_um"])])
z_max_stack = max([z1 for _row, _z0, z1 in intervals] + [metal_thickness])

lambda_min_um = C0 / float(RF_FREQUENCIES_HZ[-1]) / UM
requested_clearance_um = float(SETTINGS.get("port_clearance_wavelengths", 0.25)) * lambda_min_um
clearance_um = max(5.0 * float(SETTINGS["mesh_bulk_um"]), requested_clearance_um)
device_bounds = list(BOUNDING_BOX_UM)
x_min = device_bounds[0] - clearance_um
x_max = device_bounds[2] + clearance_um
y_min = device_bounds[1] - clearance_um
y_max = device_bounds[3] + clearance_um
z_padding = max(5.0 * float(SETTINGS["mesh_bulk_um"]), 0.05 * lambda_min_um)
z_min = z_min_stack - z_padding
z_max = z_max_stack + z_padding

fdtd.addfdtd()
fdtd.set("name", "FDTD")
fdtd.set("dimension", "3D")
fdtd.set("background index", 1.0)
fdtd.set("x min", x_min * UM)
fdtd.set("x max", x_max * UM)
fdtd.set("y min", y_min * UM)
fdtd.set("y max", y_max * UM)
fdtd.set("z min", z_min * UM)
fdtd.set("z max", z_max * UM)
fdtd.set("simulation time", float(SETTINGS["simulation_time_ns"]) * NS)
fdtd.set("auto shutoff min", float(SETTINGS["auto_shutoff"]))
fdtd.set("mesh accuracy", 2)
for boundary_name in ("x min bc", "x max bc", "y min bc", "y max bc"):
    fdtd.set(boundary_name, "PML")
for boundary_name in ("z min bc", "z max bc"):
    fdtd.set(boundary_name, "Metal" if str(SETTINGS.get("boundary_type", "")).lower().startswith("metal") else "PML")
_safe_set(fdtd, "pml layers", int(SETTINGS.get("pml_layers", 28)))

for index, (row, layer_z0, layer_z1) in enumerate(intervals, start=1):
    if layer_z1 <= layer_z0 or float(row.get("relative_permittivity", row.get("relative_permittivity_x", 1.0))) == 1.0:
        continue
    material = _add_rf_dielectric(fdtd, row)
    _add_rect(fdtd, "%02d %s" % (index, row.get("name", "dielectric")), material,
              x_min - float(SETTINGS["mesh_bulk_um"]), x_max + float(SETTINGS["mesh_bulk_um"]),
              y_min - float(SETTINGS["mesh_bulk_um"]), y_max + float(SETTINGS["mesh_bulk_um"]),
              layer_z0, layer_z1, mesh_order=3)

for polygon in GEOMETRY:
    vertices = np.asarray(polygon["vertices_um"], dtype=float)
    fdtd.addpoly()
    fdtd.set("name", str(polygon["name"]))
    fdtd.set("vertices", vertices * UM)
    fdtd.set("z min", 0.0)
    fdtd.set("z max", metal_thickness * UM)
    fdtd.set("material", metal_material)
    _safe_set(fdtd, "override mesh order from material database", True)
    _safe_set(fdtd, "mesh order", 1)

if bool(SETTINGS.get("backing_ground", False)):
    _add_rect(
        fdtd,
        "backing ground",
        metal_material,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min_stack,
        z_min_stack + metal_thickness,
        mesh_order=1,
    )

fdtd.addmesh()
fdtd.set("name", "RF metal edge mesh")
fdtd.set("x min", (device_bounds[0] - float(SETTINGS["mesh_edge_um"])) * UM)
fdtd.set("x max", (device_bounds[2] + float(SETTINGS["mesh_edge_um"])) * UM)
fdtd.set("y min", (device_bounds[1] - float(SETTINGS["mesh_edge_um"])) * UM)
fdtd.set("y max", (device_bounds[3] + float(SETTINGS["mesh_edge_um"])) * UM)
fdtd.set("z min", -2.0 * float(SETTINGS["mesh_vertical_um"]) * UM)
fdtd.set("z max", (metal_thickness + 2.0 * float(SETTINGS["mesh_vertical_um"])) * UM)
fdtd.set("override x mesh", True)
fdtd.set("override y mesh", True)
fdtd.set("override z mesh", True)
fdtd.set("dx", float(SETTINGS["mesh_edge_um"]) * UM)
fdtd.set("dy", float(SETTINGS["mesh_edge_um"]) * UM)
fdtd.set("dz", float(SETTINGS["mesh_vertical_um"]) * UM)

fdtd.setglobalsource("wavelength start", C0 / float(RF_FREQUENCIES_HZ[-1]))
fdtd.setglobalsource("wavelength stop", C0 / float(RF_FREQUENCIES_HZ[0]))
fdtd.setglobalmonitor("frequency points", int(SETTINGS["frequency_points"]))


def _plane_axis(record):
    return "x-axis" if str(record.get("plane_normal", "X")).upper() == "X" else "y-axis"


def _mode_selection(record):
    requested = str(record.get("mode", "fundamental mode")).strip()
    # The editor's descriptive quasi-TEM label maps to MODE/FDTD's actual
    # automatic fundamental-mode selection token.
    return "fundamental mode" if "quasi-tem" in requested.lower() else requested


def _set_transverse_plane(record):
    axis = _plane_axis(record)
    center_x, center_y = map(float, record["center_um"])
    fdtd.set("x", center_x * UM)
    fdtd.set("y", center_y * UM)
    fdtd.set("z", 0.5 * (z_min_stack + z_max_stack) * UM)
    if axis == "x-axis":
        fdtd.set("y span", float(record.get("span_um", SETTINGS["port_transverse_span_um"])) * UM)
    else:
        fdtd.set("x span", float(record.get("span_um", SETTINGS["port_transverse_span_um"])) * UM)
    fdtd.set("z span", float(record.get("z_span_um", SETTINGS["port_vertical_span_um"])) * UM)
    return axis


source_ports = [port for port in RF_MODE_PORTS if str(port.get("rf_role", "")).strip().lower() == "source"]
if len(source_ports) != 1:
    raise RuntimeError("3D RF FDTD needs exactly one manually placed RF mode port with role Source")
RF_SOURCE_PORT = source_ports[0]
fdtd.addmode()
fdtd.set("name", str(RF_SOURCE_PORT["name"]))
fdtd.set("injection axis", _set_transverse_plane(RF_SOURCE_PORT))
fdtd.set("direction", str(RF_SOURCE_PORT.get("direction", "Forward")))
fdtd.set("mode selection", _mode_selection(RF_SOURCE_PORT))
_safe_set(fdtd, "multifrequency mode calculation", bool(RF_SOURCE_PORT.get("multifrequency_mode_injection", True)))
fdtd.select(str(RF_SOURCE_PORT["name"]))
fdtd.updatemodes()


def _measurement_candidates():
    records = list(RF_POWER_MONITORS)
    roles = {str(item.get("rf_role", "")).strip().lower() for item in records}
    # A manually placed RF mode port can itself define a passive input/output
    # expansion plane.  Prefer an explicit RF power monitor at the same role,
    # but do not discard the mode-port-only workflow offered by the editor.
    for mode_port in RF_MODE_PORTS:
        mode_role = str(mode_port.get("rf_role", "")).strip().lower()
        if mode_role in {"input reference", "output"} and mode_role not in roles:
            records.append(dict(mode_port))
            roles.add(mode_role)
    if "input reference" not in roles:
        clone = dict(RF_SOURCE_PORT)
        clone.update({"name": str(RF_SOURCE_PORT["name"]) + "_reference", "rf_role": "Input reference"})
        records.append(clone)
        roles.add("input reference")
    if str(PRIMARY_KIND) not in ONE_PORT_KINDS and "output" not in roles:
        output_ports = [port for port in RF_MODE_PORTS if str(port.get("rf_role", "")).strip().lower() == "output"]
        if output_ports:
            records.append(dict(output_ports[0]))
    return records


RF_INPUT_EXPANSION_NAME = ""
RF_INPUT_RESULT_NAME = "rf_input"
RF_OUTPUT_EXPANSION_NAME = ""
RF_OUTPUT_RESULT_NAME = "rf_output"
for plane_index, plane in enumerate(_measurement_candidates(), start=1):
    role = str(plane.get("rf_role", "")).strip().lower()
    if role not in {"input reference", "output"}:
        continue
    base_name = str(plane.get("name") or "rf_plane_%d" % plane_index)
    power_name = base_name + "_power"
    expansion_name = base_name + "_expansion"
    result_name = RF_INPUT_RESULT_NAME if role == "input reference" else RF_OUTPUT_RESULT_NAME
    monitor_type = "2D X-normal" if str(plane.get("plane_normal", "X")).upper() == "X" else "2D Y-normal"
    fdtd.addpower()
    fdtd.set("name", power_name)
    fdtd.set("monitor type", monitor_type)
    _set_transverse_plane(plane)
    fdtd.addmodeexpansion()
    fdtd.set("name", expansion_name)
    fdtd.set("monitor type", monitor_type)
    _set_transverse_plane(plane)
    fdtd.set("mode selection", _mode_selection(plane))
    fdtd.select(expansion_name)
    fdtd.updatemodes()
    fdtd.setexpansion(result_name, power_name)
    if role == "input reference":
        RF_INPUT_EXPANSION_NAME = expansion_name
    else:
        RF_OUTPUT_EXPANSION_NAME = expansion_name

if not RF_INPUT_EXPANSION_NAME:
    raise RuntimeError("No RF Input reference plane was configured")
if str(PRIMARY_KIND) not in ONE_PORT_KINDS and not RF_OUTPUT_EXPANSION_NAME:
    raise RuntimeError("This two-port RF device has no Output reference plane")
print("Built official-style 3D RF FDTD model for", PRIMARY_KIND)
print("Source:", RF_SOURCE_PORT["name"], "input expansion:", RF_INPUT_EXPANSION_NAME,
      "output expansion:", RF_OUTPUT_EXPANSION_NAME or "one-port reflection only")
'''


_RF_FDTD_RUN_REMOTE = r'''# GPU solve; only compact modal spectra are retained.
resource_mode = str(SETTINGS.get("resource_mode", "GPU")).strip().upper()
if resource_mode != "GPU":
    raise RuntimeError("3D CPW discontinuity simulations require resource_mode='GPU'")
fdtd.run("FDTD", "GPU")


def _expansion_dataset(object_name, result_name):
    return fdtd.getresult(object_name, "expansion for " + result_name)


def _coefficient(dataset, key):
    value = np.squeeze(np.asarray(dataset[key]))
    frequency = np.squeeze(np.asarray(dataset.get("f", RF_FREQUENCIES_HZ))).ravel()
    if value.ndim == 0:
        value = np.full(frequency.size, value, dtype=complex)
    elif value.ndim > 1:
        axes = [index for index, size in enumerate(value.shape) if size == frequency.size]
        if axes:
            value = np.moveaxis(value, axes[0], 0).reshape(frequency.size, -1)[:, 0]
        else:
            value = value.ravel()[:frequency.size]
    return frequency, np.asarray(value).ravel()


input_data = _expansion_dataset(RF_INPUT_EXPANSION_NAME, RF_INPUT_RESULT_NAME)
frequency_hz, a_input = _coefficient(input_data, "a")
_frequency_b, b_input = _coefficient(input_data, "b")
s11 = b_input / np.where(np.abs(a_input) > 1e-30, a_input, np.nan + 0j)
s21 = np.full(s11.shape, np.nan + 0j)
if RF_OUTPUT_EXPANSION_NAME:
    output_data = _expansion_dataset(RF_OUTPUT_EXPANSION_NAME, RF_OUTPUT_RESULT_NAME)
    output_frequency_hz, a_output = _coefficient(output_data, "a")
    if output_frequency_hz.size != frequency_hz.size or not np.allclose(output_frequency_hz, frequency_hz):
        a_output = np.interp(frequency_hz, output_frequency_hz, np.real(a_output)) + 1j * np.interp(
            frequency_hz, output_frequency_hz, np.imag(a_output)
        )
    # Matches the official CPW FDTD example: S11=T1.b/T1.a; S21=T2.a/T1.a.
    s21 = a_output / np.where(np.abs(a_input) > 1e-30, a_input, np.nan + 0j)

RF_RESULT_ARRAYS = {
    "frequency_hz": frequency_hz,
    "S11": s11,
    "S21": s21,
    "S11_power": np.abs(s11) ** 2,
    "S21_power": np.abs(s21) ** 2,
    "S11_phase_deg": np.unwrap(np.angle(s11)) * 180.0 / np.pi,
    "S21_phase_deg": np.unwrap(np.angle(s21)) * 180.0 / np.pi,
}
# Plotting/final serialization is CPU work. The expensive solve above remains explicit GPU.
try:
    fdtd.setresource("FDTD", 1, "resource", "CPU")
    print("Post-processing resource switched back to CPU.")
except Exception as exc:
    print("CPU post-processing switch warning:", str(exc)[:180])
'''


_RF_SAVE_RESULTS_REMOTE = r'''# Save compact numerical results before closing the live CAD owner.
result_arrays = dict(globals().get("RF_RESULT_ARRAYS", {}))
results_npz = os.path.join(REMOTE_WORK, "rf_results.npz")
results_json = os.path.join(REMOTE_WORK, "rf_results.json")
summary_path = os.path.join(REMOTE_WORK, "summary.txt")
np.savez_compressed(results_npz, **result_arrays)


def _json_array(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {"real": np.real(array).tolist(), "imag": np.imag(array).tolist()}
    return array.tolist()


json_payload = {
    "workflow": str(SETTINGS["rf_workflow"]),
    "component_kind": str(PRIMARY_KIND),
    "component_parameters": PRIMARY_PARAMS,
    "simulation_settings": SETTINGS,
    "material_stack": MATERIAL_STACK,
    "rf_mode_ports": RF_MODE_PORTS,
    "rf_power_monitors": RF_POWER_MONITORS,
    "results": {key: _json_array(value) for key, value in result_arrays.items()},
}
with open(results_json, "w", encoding="utf-8") as stream:
    json.dump(json_payload, stream, indent=2, sort_keys=True, allow_nan=True)

lines = [
    "MAX LAYOUT RF SIMULATION SUMMARY",
    "================================",
    "",
    "GEOMETRY PARAMETERS",
    "-------------------",
    "Component: " + str(PRIMARY_KIND),
]
for key, value in sorted(PRIMARY_PARAMS.items()):
    lines.append("%s: %s" % (key, value))
lines.extend(["", "SIMULATION SETTINGS", "-------------------"])
for key in (
    "rf_workflow", "frequency_start_ghz", "frequency_stop_ghz", "frequency_points",
    "resource_mode", "simulation_time_ns", "auto_shutoff", "boundary_type",
    "mesh_edge_um", "mesh_vertical_um", "mesh_bulk_um", "metal_model",
    "metal_conductivity_s_per_m", "metal_thickness_um",
):
    if key in SETTINGS:
        lines.append("%s: %s" % (key, SETTINGS[key]))
lines.extend(["", "RF MATERIAL STACK", "-----------------"])
for index, row in enumerate(MATERIAL_STACK, start=1):
    lines.append("%02d %s | role=%s | thickness_um=%s | eps=%s | loss_tangent=%s | conductivity=%s" % (
        index, row.get("name", "unnamed"), row.get("role", "dielectric"),
        row.get("thickness_um", 0.0),
        row.get("relative_permittivity", [
            row.get("relative_permittivity_x"), row.get("relative_permittivity_y"),
            row.get("relative_permittivity_z"),
        ]), row.get("loss_tangent", 0.0), row.get("conductivity_s_per_m", 0.0),
    ))
lines.extend(["", "PORTS AND REFERENCE PLANES", "--------------------------"])
for record in [*RF_MODE_PORTS, *RF_POWER_MONITORS]:
    lines.append("%s | kind=%s | role=%s | center_um=%s | normal=%s | span_um=%s | z_span_um=%s" % (
        record.get("name"), record.get("kind"), record.get("rf_role"), record.get("center_um"),
        record.get("plane_normal"), record.get("span_um"), record.get("z_span_um"),
    ))
lines.extend(["", "RESULTS SUMMARY", "---------------"])
if not result_arrays:
    lines.append("The model was built but RUN_SIMULATION was False; no solved spectra are present.")
elif str(SETTINGS["rf_workflow"]) == "fde":
    target = float(SETTINGS.get("target_frequency_ghz", SETTINGS["frequency_start_ghz"])) * GHZ
    index = int(np.argmin(np.abs(np.asarray(result_arrays["frequency_hz"]) - target)))
    lines.append("Frequency_GHz: %.9g" % (np.asarray(result_arrays["frequency_hz"])[index] / GHZ))
    lines.append("Z0_ohm: %s" % np.asarray(result_arrays["Z0_ohm"])[index])
    lines.append("neff: %s" % np.asarray(result_arrays["neff"])[index])
    lines.append("ng: %s" % np.asarray(result_arrays["ng"])[index])
    lines.append("loss_db_per_m: %.9g" % np.asarray(result_arrays["loss_db_per_m"])[index])
else:
    target = float(SETTINGS.get("target_frequency_ghz", SETTINGS["frequency_start_ghz"])) * GHZ
    index = int(np.argmin(np.abs(np.asarray(result_arrays["frequency_hz"]) - target)))
    lines.append("Frequency_GHz: %.9g" % (np.asarray(result_arrays["frequency_hz"])[index] / GHZ))
    lines.append("S11: %s" % np.asarray(result_arrays["S11"])[index])
    lines.append("S11_power: %.9g" % np.asarray(result_arrays["S11_power"])[index])
    if np.isfinite(np.asarray(result_arrays["S21_power"])[index]):
        lines.append("S21: %s" % np.asarray(result_arrays["S21"])[index])
        lines.append("S21_power: %.9g" % np.asarray(result_arrays["S21_power"])[index])
        lines.append("S21_phase_deg: %.9g" % np.asarray(result_arrays["S21_phase_deg"])[index])
with open(summary_path, "w", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")

fdtd.save(REMOTE_FINAL_PROJECT)
if not os.path.isfile(REMOTE_FINAL_PROJECT) or os.path.getsize(REMOTE_FINAL_PROJECT) <= 0:
    raise RuntimeError("The final RF project was not created: " + REMOTE_FINAL_PROJECT)
print("Saved final RF project:", REMOTE_FINAL_PROJECT)
print("Saved RF numerical results:", results_npz)
print("Saved RF JSON results:", results_json)
print("Saved RF summary:", summary_path)
'''


_RF_RELEASE_CELL = r'''# Release in the reference order: close CAD, return three HPC Packs, close SSH.
try:
    lam.run(
        "try:\n    fdtd.close()\nexcept Exception:\n    pass\n",
        timeout=90,
    )
finally:
    _release = subprocess.run(_SSH + [HOST,
        f'{LIC}/LicensingSettings web shared products checkin '
        '--name "Ansys HPC Pack - Shared Web" --count 3 --mode user'],
        capture_output=True, text=True, timeout=180)
    _release_out = (_release.stdout + _release.stderr).strip()
    print("HPC Packs:", "3 returned to Shared Web" if "SUCCESS" in _release_out else _release_out[:400])
    lam.close()
'''


def _quick_options(configuration: dict[str, Any], workflow: str) -> str:
    extension = ".lms" if workflow == "fde" else ".fsp"
    return (
        "# QUICK RF RUN OPTIONS — edit these before any other cell.\n"
        f"RUN_SIMULATION = {bool(configuration.get('run_after_build', True))!r}\n"
        f"print('RF workflow: {workflow.upper()} | project extension: {extension}')\n"
        "print('Inspection and final project saving are always enabled.')\n\n"
        + _PIRIS_PATHS_CELL
    )


def _inspection_cell(workflow: str) -> str:
    return (
        "# Save the required pre-solve GUI inspection project while retaining the live model.\n"
        "run_remote_checked(\n"
        "    'fdtd.save(REMOTE_INSPECTION_PROJECT); import os; assert os.path.isfile(REMOTE_INSPECTION_PROJECT) and os.path.getsize(REMOTE_INSPECTION_PROJECT) > 0',\n"
        "    'Save RF inspection project', timeout=600,\n"
        ")\n"
        "print('Saved the pre-solve project remotely; it will be fetched with the result bundle after section 7.')\n"
    )


def _fetch_and_plot_cell(workflow: str) -> str:
    if workflow == "fde":
        plot = r'''with np.load(_local_npz) as data:
    frequency_ghz = np.asarray(data["frequency_hz"]) / 1e9
    z0 = np.asarray(data["Z0_ohm"])
    neff = np.asarray(data["neff"])
    ng = np.asarray(data["ng"])
    loss = np.asarray(data["loss_db_per_cm"])
figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
axes[0].plot(frequency_ghz, np.real(z0), label="Re(Z0)")
axes[0].plot(frequency_ghz, np.imag(z0), label="Im(Z0)")
axes[0].set_ylabel("impedance [ohm]")
axes[1].plot(frequency_ghz, np.real(neff), label="Re(neff)")
axes[1].plot(frequency_ghz, np.real(ng), label="Re(ng)")
axes[1].set_ylabel("index")
axes[2].plot(frequency_ghz, loss, label="loss")
axes[2].set_ylabel("loss [dB/cm]")
axes[2].set_xlabel("frequency [GHz]")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
'''
    else:
        plot = r'''with np.load(_local_npz) as data:
    frequency_ghz = np.asarray(data["frequency_hz"]) / 1e9
    s11_power = np.asarray(data["S11_power"])
    s21_power = np.asarray(data["S21_power"])
    s11_phase = np.asarray(data["S11_phase_deg"])
    s21_phase = np.asarray(data["S21_phase_deg"])
figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
axes[0].plot(frequency_ghz, s11_power, label="|S11|^2")
if np.any(np.isfinite(s21_power)):
    axes[0].plot(frequency_ghz, s21_power, label="|S21|^2")
axes[0].set_ylabel("linear power")
axes[1].plot(frequency_ghz, s11_phase, label="phase(S11)")
if np.any(np.isfinite(s21_phase)):
    axes[1].plot(frequency_ghz, s21_phase, label="phase(S21)")
axes[1].set_ylabel("phase [degrees]")
axes[1].set_xlabel("frequency [GHz]")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
'''
    extension = ".lms" if workflow == "fde" else ".fsp"
    return (
        "# Fetch verified compact artifacts before licence release; plot locally on CPU.\n"
        "import os\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n"
        "_remote_artifacts = [REMOTE_WORK + '/rf_results.npz', REMOTE_WORK + '/rf_results.json', REMOTE_WORK + '/summary.txt']\n"
        "_local_artifacts = []\n"
        "for _remote_path in _remote_artifacts:\n"
        "    _local_path = PIRIS_RESULTS_DIR / os.path.basename(_remote_path)\n"
        "    lam.fetch(_remote_path, str(_local_path))\n"
        "    if not _local_path.is_file() or _local_path.stat().st_size <= 0:\n"
        "        raise RuntimeError('Required RF artifact was not fetched: ' + str(_local_path))\n"
        "    _local_artifacts.append(_local_path)\n"
        "    print('saved ->', _local_path)\n"
        f"_local_inspection = PIRIS_FSP_DIR / ('rf_inspection{extension}')\n"
        "lam.fetch(REMOTE_INSPECTION_PROJECT, str(_local_inspection))\n"
        "if not _local_inspection.is_file() or _local_inspection.stat().st_size <= 0:\n"
        "    raise RuntimeError('Required RF inspection project was not fetched: ' + str(_local_inspection))\n"
        "print('saved ->', _local_inspection)\n"
        f"_local_project = PIRIS_FSP_DIR / ('rf_final{extension}')\n"
        "lam.fetch(REMOTE_FINAL_PROJECT, str(_local_project))\n"
        "if not _local_project.is_file() or _local_project.stat().st_size <= 0:\n"
        "    raise RuntimeError('Required RF final project was not fetched: ' + str(_local_project))\n"
        "print('saved ->', _local_project)\n"
        "_local_npz = PIRIS_RESULTS_DIR / 'rf_results.npz'\n"
        "if RUN_SIMULATION:\n"
        + "\n".join("    " + line for line in plot.splitlines()) + "\n"
        "    figure.tight_layout()\n"
        "    _plot_path = PIRIS_RESULTS_DIR / 'rf_response.png'\n"
        "    figure.savefig(_plot_path, dpi=170, bbox_inches='tight')\n"
        "    plt.close(figure)\n"
        "    display(Image(filename=str(_plot_path), width=1000))\n"
        "    print('saved ->', _plot_path)\n"
    )


def generate_lumerical_rf_notebook(
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Build a self-contained RF notebook and return non-fatal preflight warnings.

    ``CPW`` is always MODE/FDE on CPU.  Every supported discontinuity is
    always 3D FDTD on GPU.  A mixed FDE/FDTD selection is rejected because it
    would otherwise hide two physically different solvers behind one run.
    """
    raw_configuration = deepcopy(configuration or {})
    physical, simulation = _selected_components(components, raw_configuration)
    requested_primary_uid = raw_configuration.get("primary_component_uid")
    primary = next(
        (
            component for component in physical
            if requested_primary_uid is not None
            and int(component.get("uid", -1)) == int(requested_primary_uid)
        ),
        physical[0],
    )
    primary_kind = str(primary.get("kind", ""))
    workflows = {
        "fde" if str(component.get("kind", "")) in RF_FDE_COMPONENT_KINDS else "fdtd"
        for component in physical
    }
    if len(workflows) != 1:
        raise ValueError("Export uniform CPW FDE cross-sections separately from 3D CPW discontinuities.")
    if "target_frequency_ghz" not in raw_configuration:
        start = float(raw_configuration.get("frequency_start_ghz", 1.0))
        stop = float(raw_configuration.get("frequency_stop_ghz", 100.0))
        raw_configuration["target_frequency_ghz"] = 0.5 * (start + stop)
    settings = normalize_rf_configuration(primary_kind, raw_configuration)
    workflow = str(settings["rf_workflow"])
    for component in physical[1:]:
        expected = "fde" if str(component.get("kind", "")) in RF_FDE_COMPONENT_KINDS else "fdtd"
        if expected != workflow:
            raise ValueError("Every RF component in one notebook must use the same solver workflow.")

    warnings: list[str] = []
    metal_gds_layers = _rf_metal_gds_layers(settings["material_stack"])
    geometry, bounds = _collect_geometry(physical, primary, metal_gds_layers)
    rf_ports: list[dict[str, Any]] = []
    rf_monitors: list[dict[str, Any]] = []
    if workflow == "fdtd":
        rf_ports, rf_monitors = _collect_rf_planes(
            simulation, primary, bounds, settings, warnings
        )
        _validate_planes(primary_kind, rf_ports, rf_monitors, warnings)

    settings.setdefault("hide_cad", True)
    settings["dimension"] = "2D Z-normal" if workflow == "fde" else "3D"
    settings["run_after_build"] = bool(settings.get("run_after_build", True))
    settings.pop("save_inspection_fsp", None)
    settings.pop("save_final_fsp", None)
    material_stack = deepcopy(settings["material_stack"])
    primary_params = deepcopy(primary.get("params", {}))
    exported_components = [
        {
            "uid": int(component.get("uid", 0)),
            "kind": str(component.get("kind", "")),
            "x_um": float(component.get("x", 0.0)),
            "y_um": float(component.get("y", 0.0)),
            "orientation_deg": float(component.get("orientation_deg", 0.0)),
            "params": deepcopy(component.get("params", {})),
        }
        for component in physical
    ]

    payload_cell = (
        "# Embedded RF geometry and settings; no GDS or optical sidecar is required.\n"
        f"PRIMARY_KIND = {_literal(primary_kind)}\n"
        f"PRIMARY_PARAMS = {_literal(primary_params)}\n"
        f"EXPORTED_COMPONENTS = {_literal(exported_components)}\n"
        f"SETTINGS = {_literal(settings)}\n"
        f"MATERIAL_STACK = {_literal(material_stack)}\n"
        f"BOUNDING_BOX_UM = {_literal(bounds)}\n"
        f"GEOMETRY = {_literal(geometry)}\n"
        f"RF_MODE_PORTS = {_literal(rf_ports)}\n"
        f"RF_POWER_MONITORS = {_literal(rf_monitors)}\n"
        f"ONE_PORT_KINDS = {_literal(sorted(RF_ONE_PORT_KINDS))}\n"
        f"EXPORT_WARNINGS = {_literal(warnings)}\n"
        "SETTINGS['run_after_build'] = bool(RUN_SIMULATION)\n"
        "print('Frequency sweep [GHz]:', SETTINGS['frequency_start_ghz'], 'to', "
        "SETTINGS['frequency_stop_ghz'], 'points', SETTINGS['frequency_points'])\n"
        "print('Resource:', SETTINGS['resource_mode'], '| geometry:', SETTINGS['dimension'])\n"
        "for warning in EXPORT_WARNINGS:\n    print('WARNING:', warning)\n"
    )
    builder = _RF_FDE_BUILD_REMOTE if workflow == "fde" else _RF_FDTD_BUILD_REMOTE
    runner = _RF_FDE_RUN_REMOTE if workflow == "fde" else _RF_FDTD_RUN_REMOTE
    remote_build_cell = (
        "# Send the complete RF model to the already licensed persistent Lambda process.\n"
        f"REMOTE_RF_BUILDER = {builder!r}\n"
        "_rf_payload = (\n"
        "    'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "    + 'MATERIAL_STACK = ' + repr(MATERIAL_STACK) + '\\n'\n"
        "    + 'BOUNDING_BOX_UM = ' + repr(BOUNDING_BOX_UM) + '\\n'\n"
        "    + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "    + 'RF_MODE_PORTS = ' + repr(RF_MODE_PORTS) + '\\n'\n"
        "    + 'RF_POWER_MONITORS = ' + repr(RF_POWER_MONITORS) + '\\n'\n"
        "    + 'PRIMARY_KIND = ' + repr(PRIMARY_KIND) + '\\n'\n"
        "    + 'PRIMARY_PARAMS = ' + repr(PRIMARY_PARAMS) + '\\n'\n"
        "    + 'ONE_PORT_KINDS = ' + repr(ONE_PORT_KINDS) + '\\n'\n"
        ")\n"
        "run_remote_checked(_rf_payload + REMOTE_RF_BUILDER, "
        f"'Build {workflow.upper()} RF model', timeout=1800)\n"
    )
    if workflow == "fde":
        run_cell = (
            "# CPU MODE/FDE frequency sweep.\n"
            f"REMOTE_RF_RUNNER = {runner!r}\n"
            "if RUN_SIMULATION:\n"
            "    run_remote_checked(REMOTE_RF_RUNNER, 'RF MODE/FDE frequency sweep [CPU]', timeout=21600)\n"
            "else:\n"
            "    run_remote_checked('RF_RESULT_ARRAYS = {}', 'Initialize unsolved RF result bundle', timeout=120)\n"
        )
    else:
        run_cell = (
            "# 3D CPW discontinuity solve. Geometry was built with CPU threads; the field solve is GPU.\n"
            f"REMOTE_RF_RUNNER = {runner!r}\n"
            "if RUN_SIMULATION:\n"
            "    solve_remote_checked(REMOTE_RF_RUNNER, 'RF 3D FDTD [GPU]', timeout=43200)\n"
            "else:\n"
            "    run_remote_checked('RF_RESULT_ARRAYS = {}', 'Initialize unsolved RF result bundle', timeout=120)\n"
        )
    save_cell = (
        "# Serialize results and save the solved project while the MODE/FDTD owner is alive.\n"
        f"REMOTE_RF_RESULTS_SAVER = {_RF_SAVE_RESULTS_REMOTE!r}\n"
        "run_remote_checked(REMOTE_RF_RESULTS_SAVER, 'Save RF results and summary', timeout=1800)\n"
    )

    if workflow == "fde":
        solver_description = (
            "A 2D Z-normal MODE/FDE cross-section follows the official Ansys CPW and traveling-wave "
            "examples. It runs on CPU and reports complex Z0, neff, ng, and loss versus GHz frequency."
        )
        dimension = "2D Z-normal"
        export_name = "lumerical-rf-mode"
    else:
        solver_description = (
            "A 3D GPU FDTD model follows the official CPW discontinuity expansion workflow: a modal "
            "source plus input/output power and mode-expansion planes report S11, S21, and phase."
        )
        dimension = "3D"
        export_name = "lumerical-rf-fdtd"
    intro = f"""# Max Layout → Lumerical RF notebook

**Component:** {primary_kind}<br>
**Workflow:** {workflow.upper()}<br>
**Frequency:** {settings['frequency_start_ghz']:g}–{settings['frequency_stop_ghz']:g} GHz

{solver_description}

This is an RF-only notebook. It contains no optical wavelength defaults, fiber ports, or automatic optical-port seeding. Geometry is rotated into the selected component's local frame so its RF reference planes remain axis aligned. Manual **RF mode port** and **RF power monitor** objects are authoritative. Endpoint fallback is used only when `rf_port_strategy='component_endpoints'` was explicitly selected.

The licence lifecycle matches the established launcher contract: connect, roam exactly three Shared-Web HPC Packs, build/run, save and fetch every requested artifact, close the live CAD owner, return all three packs, and close SSH. Always run the final release cell after an error or interruption.
"""
    notebook = {
        "cells": [
            _notebook_cell("code", _quick_options(settings, workflow)),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Connect to Lambda\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 2 · Acquire Ansys Shared Web licences\n"),
            _notebook_cell("code", _LICENSE_CHECKOUT_CELL),
            _notebook_cell("markdown", "## 3 · Embedded RF geometry, stack, and reference planes\n"),
            _notebook_cell("code", payload_cell),
            _notebook_cell("markdown", "## 4 · Build the live RF model\n"),
            _notebook_cell("code", remote_build_cell),
            _notebook_cell("markdown", "## 5 · Save the pre-solve inspection project\n"),
            _notebook_cell("code", _inspection_cell(workflow)),
            _notebook_cell("markdown", "## 6 · Run the RF simulation\n"),
            _notebook_cell("code", run_cell),
            _notebook_cell("markdown", "## 7 · Save numerical results and final project\n"),
            _notebook_cell("code", save_cell),
            _notebook_cell("markdown", "## 8 · Fetch artifacts and plot locally on CPU\n"),
            _notebook_cell("code", _fetch_and_plot_cell(workflow)),
            _notebook_cell("markdown", "## 9 · Release MODE/FDTD and all three HPC Packs\n\nAlways run this cell, including after a failed solve.\n"),
            _notebook_cell("code", _RF_RELEASE_CELL),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "max_layout": {
                "export": export_name,
                "domain": "RF",
                "workflow": workflow,
                "units": "um/GHz",
                "dimension": dimension,
                "execution": "lambda-a100-persistent-ssh",
                "license_lifecycle": "shared-web-3-hpc-packs-save-fetch-release",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, warnings


def write_lumerical_rf_notebook(
    path: str | Path,
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> list[str]:
    """Write a dedicated MODE/FDTD RF notebook and return preflight warnings."""
    notebook, warnings = generate_lumerical_rf_notebook(components, configuration)
    Path(path).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return warnings


__all__ = ["generate_lumerical_rf_notebook", "write_lumerical_rf_notebook"]
