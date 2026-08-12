"""Lumerical LumOpt shape-adjoint notebook export.

The ordinary and Cartesian-sweep exporters live in :mod:`max_layout.lumerical`.
This module deliberately keeps the inverse-design path separate because a
LumOpt optimization has a different ownership model: one seed FDTD session is
closed before LumOpt opens its one persistent solver, and that solver is reused
for all forward/adjoint iterations.

Continuous device geometry uses true shape-adjoint gradients.  Grating fiber
alignment (``angle_theta`` and ``fiber_offset``) is co-optimized with bounded
GPU forward solves, then frozen for the adjoint geometry stage.  Process-stack,
mesh, material, and integer/topology controls remain fixed.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import pprint
import re
from typing import Any, Iterable

import numpy as np

from .constants import COMPONENT_SPECS
from .gds.build import component_geometry_arrays
from .lumerical import (
    _BUILD_CELL,
    _LAMBDA_CONNECT_CELL,
    _LICENSE_CHECKOUT_CELL,
    _PIRIS_PATHS_CELL,
    _RELEASE_LICENSES_CELL,
    _item_vertical_reference,
    _notebook_cell,
    _notebook_literal_assignments,
    _quick_run_options_cell,
    _runtime_setup_source,
    _stack_vertical_levels,
    generate_lumerical_notebook,
    sweep_parameter_label,
)


SUPPORTED_ADJOINT_COMPONENT_KINDS = frozenset(
    {"Grating coupler", "GC-SOI", "1x2 MMI"}
)
GRATING_ALIGNMENT_PARAMETERS = frozenset({"angle_theta", "fiber_offset"})


_EXACT_PARAMETER_EXCLUSIONS = frozenset(
    {
        "N",
        "target_length",  # In GC-SOI this only selects ceil(target_length/pitch).
        "output_length",  # Adjusted internally to keep MMI receiver planes fixed.
        "port_sep",  # Would move the fixed upper/lower receiver planes in Y.
        "layer",
        "datatype",
        "slab_layer",
        "slab_datatype",
        "etched_layer",
        "etched_datatype",
        "tolerance",
        "h_total",
        "etch_depth",
        "input_reference_before_taper_um",
        "fdtd_port_clearance_um",
        "fdtd_port_offset_from_waveguide_end_um",
        "waveguide_total_power_before_mode_um",
        "waveguide_monitor_span_um",
        "waveguide_effective_index",
        "waveguide_neff_tolerance",
        "waveguide_mode_search_count",
    }
)


_EXCLUDED_PARAMETER_FRAGMENTS = (
    "fiber",
    "source",
    "monitor",
    "port_",
    "wavelength",
    "material",
    "mesh",
    "index",
    "neff",
    "layer",
    "datatype",
    "tolerance",
    "resolution",
    "points",
    "order",
)


def _is_nonempty_apodization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(len(value))
    except Exception:
        return False


def adjoint_optimizable_component_parameters(
    component: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return continuous JSON geometry parameters accepted by shape adjoint.

    Integer values are intentionally excluded even when their component spec
    presents them as numeric.  Changing a polygon/tooth count makes the shape
    derivative discontinuous.  GC-SOI's period count is therefore derived once
    from the nominal design and frozen by the notebook generator.
    """

    kind = str(component.get("kind", ""))
    if kind not in SUPPORTED_ADJOINT_COMPONENT_KINDS:
        return []
    params = component.get("params", {})
    specs = COMPONENT_SPECS.get(kind, {})
    apodized = _is_nonempty_apodization(params.get("fill_factors"))
    result: list[dict[str, Any]] = []
    for key, value in params.items():
        if isinstance(value, bool) or isinstance(value, int) or not isinstance(value, float):
            continue
        spec = specs.get(key, [])
        if spec and str(spec[0]).lower() != "float":
            continue
        lower = str(key).lower()
        if key in _EXACT_PARAMETER_EXCLUSIONS or lower in {
            str(item).lower() for item in _EXACT_PARAMETER_EXCLUSIONS
        }:
            continue
        # Embedded ``gc_*`` controls belong to optional I/O couplers around an
        # MMI, not to the symmetric MMI being optimized.
        if lower.startswith("gc_"):
            continue
        is_grating_alignment = (
            kind in {"Grating coupler", "GC-SOI"}
            and lower in {"fiber_offset", "angle_theta"}
        )
        if not is_grating_alignment and any(
            fragment in lower for fragment in _EXCLUDED_PARAMETER_FRAGMENTS
        ):
            continue
        if apodized and lower in {"fill_factor", "duty_cycle"}:
            # A non-empty per-tooth array overrides the scalar in both GDS
            # builders.  Advertising the ignored scalar would be misleading.
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        result.append(
            {
                "parameter": str(key),
                "label": sweep_parameter_label(str(key)),
                "value_type": "float",
                "nominal": numeric,
                "initial": numeric,
            }
        )

    priority = {
        "pitch": 0,
        "duty_cycle": 1,
        "fill_factor": 1,
        "angle_theta": 2,
        "fiber_offset": 3,
        "mmi_length": 4,
        "mmi_width": 5,
        "taper_width": 6,
        "taper_L": 7,
        "wg_width": 8,
        "radius": 9,
        "y_span": 10,
        "port_sep": 11,
    }
    return sorted(
        result,
        key=lambda item: (priority.get(str(item["parameter"]), 100), str(item["label"])),
    )


def _validate_parameter_bounds(parameter: str, minimum: float, maximum: float) -> None:
    lower = parameter.lower()
    if maximum <= minimum:
        raise ValueError(f"{sweep_parameter_label(parameter)} maximum must be greater than its minimum.")
    if lower in {"fill_factor", "duty_cycle"} and not (
        0.0 < minimum < maximum < 1.0
    ):
        raise ValueError(
            f"{sweep_parameter_label(parameter)} bounds must remain strictly between 0 and 1."
        )
    positive_tokens = (
        "pitch",
        "width",
        "length",
        "radius",
        "span",
        "gap",
        "sep",
        "taper_l",
    )
    if any(token in lower for token in positive_tokens) and minimum <= 0.0:
        raise ValueError(f"{sweep_parameter_label(parameter)} must remain greater than zero.")
    if lower.endswith("angle_deg") or lower in {"alpha_t"}:
        if minimum <= 0.0 or maximum >= 180.0:
            raise ValueError(
                f"{sweep_parameter_label(parameter)} bounds must remain between 0 and 180 degrees."
            )
    if lower == "angle_theta" and not (0.0 <= minimum < maximum < 90.0):
        raise ValueError(
            "Angle theta bounds must remain at least 0 and below 90 degrees."
        )
    if lower in {"taper_power", "taper_exponent"} and minimum <= 0.0:
        raise ValueError(f"{sweep_parameter_label(parameter)} must remain greater than zero.")


def normalize_lumerical_optimization_spec(
    component: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    *,
    center_wavelength_um: float,
    bandwidth_nm: float,
    wavelength_points: int = 7,
    max_iterations: int = 30,
) -> dict[str, Any]:
    """Validate the first-page optimization choices and return stable JSON.

    The initial value is always read from the component, not trusted from the
    dialog row.  That makes an exported parameter patch directly applicable to
    the same component JSON object.
    """

    kind = str(component.get("kind", ""))
    if kind not in SUPPORTED_ADJOINT_COMPONENT_KINDS:
        raise ValueError(
            "Shape-adjoint optimization supports Grating coupler, GC-SOI, and 1x2 MMI components."
        )
    eligible = {
        str(item["parameter"]): item
        for item in adjoint_optimizable_component_parameters(component)
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        parameter = str(raw.get("parameter", "")).strip()
        if parameter not in eligible:
            raise ValueError(
                f"{parameter or 'Selected parameter'} is not an available continuous adjoint optimization parameter."
            )
        if parameter in seen:
            raise ValueError(f"Optimization parameter {parameter!r} was selected more than once.")
        seen.add(parameter)
        try:
            minimum = float(raw.get("minimum", raw.get("min")))
            maximum = float(raw.get("maximum", raw.get("max")))
        except (TypeError, ValueError):
            raise ValueError(
                f"{eligible[parameter]['label']} needs finite numeric minimum and maximum bounds."
            ) from None
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"{eligible[parameter]['label']} bounds must be finite.")
        _validate_parameter_bounds(parameter, minimum, maximum)
        initial = float(component.get("params", {}).get(parameter))
        if initial < minimum - 1e-12 or initial > maximum + 1e-12:
            raise ValueError(
                f"Current {eligible[parameter]['label']} value {initial:.9g} is outside "
                f"the selected [{minimum:.9g}, {maximum:.9g}] bounds."
            )
        normalized.append(
            {
                "parameter": parameter,
                "label": str(eligible[parameter]["label"]),
                "initial": initial,
                "minimum": minimum,
                "maximum": maximum,
            }
        )
    if not normalized:
        raise ValueError("Select at least one continuous parameter to optimize.")

    if kind == "1x2 MMI":
        # These parameters change the total longitudinal footprint in the GDS
        # builder.  The adjoint parameterization shortens/lengthens the final
        # straight by the opposite amount so fixed upper/lower receiver ports
        # remain embedded in the access guides for every combined bound.
        compensated = {
            "input_length",
            "input_taper_length",
            "mmi_length",
            "output_taper_length",
        }
        maximum_total_increase = sum(
            max(0.0, float(row["maximum"]) - float(row["initial"]))
            for row in normalized
            if str(row["parameter"]) in compensated
        )
        output_length = float(component.get("params", {}).get("output_length", 0.0))
        if output_length - maximum_total_increase <= 0.0:
            raise ValueError(
                "The combined upper bounds would consume the complete MMI output straight while "
                "keeping the receiver planes fixed. Reduce the longitudinal bounds so the "
                "compensated output_length remains greater than zero."
            )

    center = float(center_wavelength_um)
    bandwidth = float(bandwidth_nm)
    if not math.isfinite(center) or center <= 0.0:
        raise ValueError("Center wavelength must be a finite value greater than zero.")
    if not math.isfinite(bandwidth) or bandwidth < 0.0:
        raise ValueError("Optimization bandwidth must be finite and non-negative.")
    points = int(wavelength_points)
    if bandwidth == 0.0:
        points = 1
    elif points < 2:
        raise ValueError("A nonzero optimization bandwidth needs at least two wavelength points.")
    if points < 1 or points > 1001:
        raise ValueError("Objective wavelength points must be between 1 and 1001.")
    iterations = int(max_iterations)
    if iterations < 1 or iterations > 10000:
        raise ValueError("Maximum optimizer iterations must be between 1 and 10000.")
    start_um = center - 0.0005 * bandwidth
    stop_um = center + 0.0005 * bandwidth
    if start_um <= 0.0:
        raise ValueError("The requested bandwidth extends to a non-positive wavelength.")

    params = component.get("params", {})
    if kind == "GC-SOI":
        pitch = float(params.get("pitch", 0.0))
        target_length = float(params.get("target_length", 0.0))
        if pitch <= 0.0 or target_length <= 0.0:
            raise ValueError("GC-SOI pitch and target_length must be positive before optimization.")
        fixed_period_count = int(math.ceil(target_length / pitch))
    elif kind == "Grating coupler":
        fixed_period_count = int(params.get("N", 0))
        if fixed_period_count < 1:
            raise ValueError("The grating coupler must contain at least one fixed period.")
    else:
        fixed_period_count = None

    objective_kind = (
        "grating_coupling_efficiency"
        if kind in {"Grating coupler", "GC-SOI"}
        else "mmi_top_output_over_input"
    )
    objective_description = (
        "Fiber-source to waveguide-receiver coupling efficiency"
        if kind in {"Grating coupler", "GC-SOI"}
        else "Top/upper output branch power divided by input power"
    )
    alignment_parameters = [
        str(row["parameter"])
        for row in normalized
        if str(row["parameter"]) in GRATING_ALIGNMENT_PARAMETERS
    ]
    adjoint_geometry_parameters = [
        str(row["parameter"])
        for row in normalized
        if str(row["parameter"]) not in GRATING_ALIGNMENT_PARAMETERS
    ]
    return {
        "version": 1,
        "engine": "LumOpt",
        "method": (
            "3D GPU alignment + shape adjoint"
            if alignment_parameters else "3D shape adjoint"
        ),
        "component_uid": int(component.get("uid", 0)),
        "component_kind": kind,
        "parameters": normalized,
        "alignment_parameters": alignment_parameters,
        "adjoint_geometry_parameters": adjoint_geometry_parameters,
        "fixed_topology": True,
        "fixed_period_count": fixed_period_count,
        "objective": {
            "kind": objective_kind,
            "description": objective_description,
            "center_wavelength_um": center,
            "bandwidth_nm": bandwidth,
            "wavelength_start_um": start_um,
            "wavelength_stop_um": stop_um,
            "wavelength_points": points,
            "linear_power": True,
            "target": 1.0 if kind in {"Grating coupler", "GC-SOI"} else 0.5,
            "validation_outputs": (
                []
                if kind in {"Grating coupler", "GC-SOI"}
                else [
                    "lower_output_over_input",
                    "total_output_over_input",
                    "upper_lower_imbalance",
                ]
            ),
        },
        "optimizer": {
            "algorithm": "L-BFGS-B",
            "max_iterations": iterations,
            "alignment_algorithm": "L-BFGS-B finite-difference forward solves",
            "alignment_max_iterations": min(iterations, 20),
            "pgtol": 1e-5,
            "ftol": 1e-5,
        },
        "parameterized_geometry": {
            "class": "Parametrization",
            "fallback_class": "ParameterizedGeometry",
            "finite_difference_step": 1e-3,
            "interpolation": "nominal-centered piecewise-linear fixed-topology polygons",
        },
        "optimization_mesh_um": 0.05,
        "opt_fields_monitor": "opt_fields",
        "resource_mode": "GPU",
        "postprocessing_resource": "CPU",
        "store_all_simulations": False,
        "save_each_fsp": False,
    }


def _resample_closed_polygon(vertices: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed polygon boundary to a deterministic vertex count."""

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("Every optimizable polygon must contain at least three XY vertices.")
    if count < 3:
        raise ValueError("The fixed polygon topology needs at least three vertices.")
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= 0.0 or not math.isfinite(perimeter):
        raise ValueError("An optimizable polygon has zero or non-finite perimeter.")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    samples = np.linspace(0.0, perimeter, count, endpoint=False)
    result = np.empty((count, 2), dtype=float)
    segment = 0
    for index, distance in enumerate(samples):
        while segment + 1 < len(cumulative) - 1 and distance >= cumulative[segment + 1]:
            segment += 1
        length = segment_lengths[segment]
        fraction = 0.0 if length <= 0.0 else (distance - cumulative[segment]) / length
        result[index] = closed[segment] + fraction * (closed[segment + 1] - closed[segment])
    return result


def _component_polygons(
    component: dict[str, Any], included_layers: set[tuple[int, int]]
) -> list[tuple[int, int, np.ndarray]]:
    polygons, _ = component_geometry_arrays(component)
    return [
        (int(layer), int(datatype), np.asarray(vertices, dtype=float))
        for vertices, layer, datatype in polygons
        if (int(layer), int(datatype)) in included_layers
    ]


def _mutated_component_for_snapshot(
    component: dict[str, Any],
    parameter: str,
    value: float,
    fixed_period_count: int | None,
) -> dict[str, Any]:
    mutated = deepcopy(component)
    mutated.setdefault("params", {})[parameter] = float(value)
    if str(mutated.get("kind", "")) == "1x2 MMI" and parameter in {
        "input_length",
        "input_taper_length",
        "mmi_length",
        "output_taper_length",
    }:
        nominal_value = float(component.get("params", {}).get(parameter, value))
        nominal_output_length = float(
            component.get("params", {}).get("output_length", 0.0)
        )
        compensated_output_length = nominal_output_length - (float(value) - nominal_value)
        if compensated_output_length <= 0.0:
            raise ValueError(
                f"{sweep_parameter_label(parameter)}={value:.9g} leaves no positive MMI output straight "
                "after fixed-receiver-plane compensation."
            )
        mutated["params"]["output_length"] = compensated_output_length
    if parameter in {"fill_factor", "duty_cycle"}:
        mutated["params"]["fill_factors"] = ""
    if str(mutated.get("kind", "")) == "GC-SOI" and fixed_period_count:
        # ``target_length`` is only used by the GDS builder to calculate the
        # integer period count.  Move that invisible threshold with pitch so
        # all endpoint snapshots retain the nominal, fixed topology.
        pitch = float(mutated["params"].get("pitch", 0.0))
        mutated["params"]["target_length"] = (float(fixed_period_count) - 0.5) * pitch
    return mutated


def _shape_snapshots(
    component: dict[str, Any],
    payload_geometry: list[dict[str, Any]],
    specification: dict[str, Any],
) -> dict[str, Any]:
    target_uid = int(component.get("uid", 0))
    target_payload = [
        item for item in payload_geometry
        if int(item.get("component_uid", -1)) == target_uid
    ]
    if not target_payload:
        raise ValueError(
            "The optimization scope contains no exported polygons for the selected component."
        )
    included_layers = {
        (int(item["layer"]), int(item.get("datatype", 0))) for item in target_payload
    }
    nominal_raw = _component_polygons(component, included_layers)
    if len(nominal_raw) != len(target_payload):
        raise ValueError(
            "The selected component polygon order changed during optimization preflight."
        )
    nominal_vertices: list[np.ndarray] = []
    for raw, payload in zip(nominal_raw, target_payload):
        layer, datatype, vertices = raw
        if (layer, datatype) != (
            int(payload["layer"]), int(payload.get("datatype", 0))
        ):
            raise ValueError("The selected component layer topology is not stable.")
        payload_vertices = np.asarray(payload["vertices_um"], dtype=float)
        nominal_vertices.append(
            _resample_closed_polygon(vertices, len(payload_vertices))
        )

    snapshots: dict[str, Any] = {
        "polygon_names": [str(item["name"]) for item in target_payload],
        "nominal": [
            np.asarray(item["vertices_um"], dtype=float).tolist()
            for item in target_payload
        ],
        "parameters": {},
    }
    fixed_count = specification.get("fixed_period_count")
    for row in specification.get("parameters", []):
        parameter = str(row["parameter"])
        if parameter in GRATING_ALIGNMENT_PARAMETERS:
            # Fiber pose is handled by the synchronized forward-solve
            # alignment stage, not by a material-boundary derivative.
            continue
        endpoint_sets: dict[str, list[list[list[float]]]] = {}
        for endpoint, value in (
            ("minimum", float(row["minimum"])),
            ("maximum", float(row["maximum"])),
        ):
            mutated = _mutated_component_for_snapshot(
                component, parameter, value,
                int(fixed_count) if fixed_count is not None else None,
            )
            raw_polygons = _component_polygons(mutated, included_layers)
            if len(raw_polygons) != len(nominal_raw):
                raise ValueError(
                    f"{sweep_parameter_label(parameter)} bounds change the component polygon count; "
                    "shape adjoint requires fixed topology."
                )
            shifted_polygons: list[list[list[float]]] = []
            for index, (layer, datatype, vertices) in enumerate(raw_polygons):
                nominal_layer, nominal_datatype, _ = nominal_raw[index]
                if (layer, datatype) != (nominal_layer, nominal_datatype):
                    raise ValueError(
                        f"{sweep_parameter_label(parameter)} bounds change the component layer topology."
                    )
                payload_nominal = np.asarray(
                    target_payload[index]["vertices_um"], dtype=float
                )
                sampled = _resample_closed_polygon(vertices, len(payload_nominal))
                shifted = payload_nominal + sampled - nominal_vertices[index]
                shifted_polygons.append(shifted.tolist())
            endpoint_sets[endpoint] = shifted_polygons
        snapshots["parameters"][parameter] = endpoint_sets
    return snapshots


def _stack_geometry_z_bounds(stack: list[dict[str, Any]]) -> tuple[float, float]:
    active = [row for row in stack if float(row.get("thickness_um", 0.0)) > 0.0]
    if not active:
        return -0.5, 0.5
    anchor = next(
        (
            index for index, row in enumerate(active)
            if str(row.get("role", "background")).lower() == "geometry"
        ),
        len(active) // 2,
    )
    ranges: list[tuple[float, float] | None] = [None] * len(active)
    thickness = float(active[anchor]["thickness_um"])
    ranges[anchor] = (-0.5 * thickness, 0.5 * thickness)
    cursor = -0.5 * thickness
    for index in range(anchor - 1, -1, -1):
        thickness = float(active[index]["thickness_um"])
        ranges[index] = (cursor - thickness, cursor)
        cursor -= thickness
    cursor = ranges[anchor][1]  # type: ignore[index]
    for index in range(anchor + 1, len(active)):
        thickness = float(active[index]["thickness_um"])
        ranges[index] = (cursor, cursor + thickness)
        cursor += thickness
    geometry_ranges = [
        ranges[index] for index, row in enumerate(active)
        if str(row.get("role", "background")).lower() == "geometry"
        and ranges[index] is not None
    ]
    if not geometry_ranges:
        return -0.5, 0.5
    return (
        min(float(value[0]) for value in geometry_ranges),
        max(float(value[1]) for value in geometry_ranges),
    )


def _optimization_volume(
    snapshots: dict[str, Any], stack: list[dict[str, Any]], mesh_um: float
) -> list[float]:
    all_vertices: list[np.ndarray] = []
    for family in (snapshots["nominal"],):
        all_vertices.extend(np.asarray(vertices, dtype=float) for vertices in family)
    for endpoint_sets in snapshots["parameters"].values():
        for endpoint in ("minimum", "maximum"):
            all_vertices.extend(
                np.asarray(vertices, dtype=float)
                for vertices in endpoint_sets[endpoint]
            )
    points = np.vstack(all_vertices)
    z_min, z_max = _stack_geometry_z_bounds(stack)
    margin = max(0.1, 2.0 * mesh_um)
    raw = np.asarray(
        [
            float(np.min(points[:, 0])) - margin,
            float(np.max(points[:, 0])) + margin,
            float(np.min(points[:, 1])) - margin,
            float(np.max(points[:, 1])) + margin,
            z_min - margin,
            z_max + margin,
        ],
        dtype=float,
    )
    # Co-locate every opt_fields face with a uniform-mesh plane.
    for index in (0, 2, 4):
        raw[index] = math.floor(raw[index] / mesh_um) * mesh_um
    for index in (1, 3, 5):
        raw[index] = math.ceil(raw[index] / mesh_um) * mesh_um
    return raw.tolist()


def _find_code_cell(notebook: dict[str, Any], needle: str) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if needle in source:
            return source
    raise RuntimeError(f"The base Lumerical notebook is missing the {needle!r} cell.")


def _objective_ports(
    payload: dict[str, Any], specification: dict[str, Any]
) -> dict[str, Any]:
    uid = int(specification["component_uid"])
    kind = str(specification["component_kind"])
    ports = [
        port for port in payload.get("PORTS", [])
        if bool(port.get("enabled", True))
        and int(port.get("parent_component_uid", -1)) == uid
    ]
    if kind in {"Grating coupler", "GC-SOI"}:
        analysis = payload.get("GRATING_ANALYSIS")
        if not analysis:
            raise ValueError(
                "Grating adjoint optimization needs the complete grating simulation setup: "
                "optical source, input-power monitor, passive waveguide receiver, and waveguide total-power monitor."
            )
        excitation_type = str(
            analysis.get("excitation_type", "fiber_mode")
        ).strip().lower()
        source_kind = str(
            analysis.get(
                "source_kind",
                "gaussian" if excitation_type == "gaussian_beam" else "fiber_port",
            )
        ).strip().lower()
        is_gaussian = excitation_type == "gaussian_beam" or source_kind in {
            "gaussian", "gaussian_beam", "gaussian source",
        }
        source_name = str(
            analysis.get("source_name", "")
            or (
                analysis.get("gaussian_source_name", "")
                if is_gaussian
                else analysis.get("fiber_port_name", "")
            )
        )
        receiver_name = str(analysis.get("waveguide_port_name", ""))
        receiver = next(
            (port for port in ports if str(port.get("name", "")) == receiver_name),
            None,
        )
        gaussian_sources = [
            source for source in payload.get("GAUSSIAN_SOURCES", [])
            if int(source.get("parent_component_uid", -1)) == uid
        ]
        gaussian_source = next(
            (
                source for source in gaussian_sources
                if str(source.get("name", "")) == source_name
            ),
            None,
        )
        if not source_name or not receiver_name or receiver is None:
            raise ValueError(
                "Grating adjoint optimization needs an optical source and a waveguide receiver FDTD port."
            )
        if is_gaussian and gaussian_source is None:
            raise ValueError(
                "Gaussian-beam grating optimization could not find the exported independent Gaussian source."
            )
        angle = float(receiver.get("outward_orientation_deg", 0.0)) % 360.0
        nearest = int(round(angle / 90.0) * 90) % 360
        direction = "Forward" if nearest in {0, 90} else "Backward"
        return {
            "kind": "grating coupling efficiency",
            "excitation_type": "gaussian_beam" if is_gaussian else "fiber_mode",
            "source_kind": "gaussian" if is_gaussian else "fiber_port",
            "source_name": source_name,
            "source_port": source_name,
            "source_mode": (
                "" if is_gaussian
                else str(analysis.get("fiber_source_mode", "mode 1"))
            ),
            "monitor_port": receiver_name,
            "direction": direction,
            "mode_number": max(1, int(receiver.get("mode number", 1))),
            "fiber_input_power_monitor": str(analysis["fiber_input_power_monitor_name"]),
            "fiber_input_power_sign": float(analysis.get("fiber_input_power_sign", -1.0)),
            "waveguide_total_power_monitor": str(analysis["waveguide_power_monitor_name"]),
            "waveguide_total_power_sign": float(analysis.get("waveguide_total_power_sign", -1.0)),
            "waveguide_port_expansion_result_name": str(
                analysis.get("waveguide_port_expansion_result_name", "expansion for port monitor")
            ),
            "waveguide_port_modal_direction": str(
                analysis.get("waveguide_port_modal_direction", "T_out")
            ),
            "waveguide_port_modal_sign": float(
                analysis.get(
                    "waveguide_port_modal_sign",
                    analysis.get("waveguide_total_power_sign", -1.0),
                )
            ),
            "normalization": (
                "waveguide fundamental-TE port T_out / measured Gaussian input-monitor power"
                if is_gaussian
                else "waveguide fundamental-TE port T_out / measured fiber input-monitor power"
            ),
        }

    analysis = payload.get("MMI_ANALYSIS")
    if not analysis:
        raise ValueError(
            "MMI adjoint optimization needs the automatic input, upper-output, lower-output ports and input reference monitor."
        )
    output_names = list(analysis.get("output_port_names", []))
    if len(output_names) < 2:
        raise ValueError("MMI optimization could not identify the upper and lower output ports.")
    receiver_name = str(output_names[0])
    receiver = next(
        (port for port in ports if str(port.get("name", "")) == receiver_name),
        None,
    )
    if receiver is None:
        raise ValueError("The MMI upper-output FDTD port is not in the selected export scope.")
    angle = float(receiver.get("outward_orientation_deg", 0.0)) % 360.0
    nearest = int(round(angle / 90.0) * 90) % 360
    direction = "Forward" if nearest in {0, 90} else "Backward"
    return {
        "kind": "MMI upper-branch/input ratio",
        "source_port": str(analysis["input_port_name"]),
        "source_mode": "mode 1",
        "monitor_port": receiver_name,
        "lower_output_port": str(output_names[1]),
        "direction": direction,
        "mode_number": 1,
        "normalization": "upper-output fundamental-TE port power / launched input-port power",
    }


def _fiber_pose_contract(
    payload: dict[str, Any],
    component: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    """Describe synchronized grating source/input-monitor alignment.

    The contract stores absolute simulation coordinates after layout-origin
    translation.  The generated notebook can therefore update the scripted
    fiber group and tilted port, or the independent Gaussian source, together
    with the ordinary input-power monitor without rebuilding the device.
    """

    kind = str(specification.get("component_kind", ""))
    active = [
        str(name) for name in specification.get("alignment_parameters", [])
        if str(name) in GRATING_ALIGNMENT_PARAMETERS
    ]
    if kind not in {"Grating coupler", "GC-SOI"} or not active:
        return {}
    uid = int(specification["component_uid"])
    analysis = dict(payload.get("GRATING_ANALYSIS") or {})
    excitation_type = str(
        analysis.get("excitation_type", "fiber_mode")
    ).strip().lower()
    source_kind = str(
        analysis.get(
            "source_kind",
            "gaussian" if excitation_type == "gaussian_beam" else "fiber_port",
        )
    ).strip().lower()
    is_gaussian = excitation_type == "gaussian_beam" or source_kind in {
        "gaussian", "gaussian_beam", "gaussian source",
    }
    source_name = str(
        analysis.get("source_name", "")
        or (
            analysis.get("gaussian_source_name", "")
            if is_gaussian
            else analysis.get("fiber_port_name", "")
        )
    )
    levels = _stack_vertical_levels(list(payload.get("MATERIAL_STACK", [])))
    input_monitor_name = str(analysis.get("fiber_input_power_monitor_name", ""))
    input_monitors = [
        monitor for monitor in payload.get("MONITORS", [])
        if int(monitor.get("parent_component_uid", -1)) == uid
        and str(monitor.get("monitor_kind", "")) == "Power monitor"
        and str(monitor.get("plane normal", "Z")).upper() == "Z"
        and str(monitor.get("name", "")) == input_monitor_name
    ]
    if len(input_monitors) != 1:
        raise ValueError(
            "Grating alignment optimization needs exactly one ordinary input-power monitor."
        )

    params = component.get("params", {})
    common = {
        "active_parameters": active,
        "component_kind": kind,
        "excitation_type": "gaussian_beam" if is_gaussian else "fiber_mode",
        "source_kind": "gaussian" if is_gaussian else "fiber_port",
        "source_name": source_name,
        "nominal_angle_theta": float(params["angle_theta"]),
        "nominal_fiber_offset": float(params.get("fiber_offset", 0.0)),
        "fiber_tox_offset_um": float(params.get("fiber_tox_offset_um", 0.65)),
        "fiber_power_monitor_below_source_um": max(
            0.001, float(params.get("fiber_power_monitor_below_source_um", 0.1))
        ),
    }

    monitor_contracts = []
    for monitor in input_monitors:
        reference_z_um = _item_vertical_reference(monitor, levels)
        distance_um = float(monitor.get("distance_um", 0.0))
        monitor_contracts.append(
            {
                "name": str(monitor["name"]),
                "role": str(monitor.get("fiber plane role", "input power measurement")),
                "reference_z_um": float(reference_z_um),
                "base_distance_um": distance_um,
                "base_z_um": float(reference_z_um + distance_um),
                "center_um": list(map(float, monitor.get("center", (0.0, 0.0)))),
                "x_span_um": max(
                    1e-6,
                    float(monitor.get("x span", monitor.get("span_um", 4.0))),
                ),
                "y_span_um": max(
                    1e-6,
                    float(monitor.get("y span", monitor.get("span_um", 4.0))),
                ),
                "expected_propagation_sign": float(
                    monitor.get("expected propagation sign", -1.0)
                ),
            }
        )

    if is_gaussian:
        gaussian_sources = [
            source for source in payload.get("GAUSSIAN_SOURCES", [])
            if int(source.get("parent_component_uid", -1)) == uid
            and str(source.get("name", "")) == source_name
        ]
        if not source_name or len(gaussian_sources) != 1:
            raise ValueError(
                "Gaussian alignment optimization needs exactly one independent Gaussian source."
            )
        source = gaussian_sources[0]
        reference_z_um = _item_vertical_reference(source, levels)
        distance_um = float(source.get("distance_um", 0.0))
        source_z_um = reference_z_um + distance_um
        source_span_um = max(
            1e-6,
            float(
                source.get(
                    "span_um",
                    max(source.get("x span", 0.0), source.get("y span", 0.0), 20.0),
                )
            ),
        )
        common.update(
            {
                "phi_deg": float(source.get("angle phi", component.get("orientation_deg", 0.0))),
                "source": {
                    "name": str(source["name"]),
                    "reference_z_um": float(reference_z_um),
                    "base_distance_um": distance_um,
                    "base_z_um": float(source_z_um),
                    "center_um": list(map(float, source.get("center", (0.0, 0.0)))),
                    "span_um": source_span_um,
                    "input_monitor_span_scale": max(
                        1.0,
                        float(source.get("input monitor span scale", 1.2)),
                    ),
                    "polarization_angle_deg": 90.0,
                },
                "ports": [],
                "monitors": monitor_contracts,
            }
        )
        return common

    fibers = [
        fiber for fiber in payload.get("FIBER_GEOMETRIES", [])
        if int(fiber.get("parent_component_uid", -1)) == uid
    ]
    if len(fibers) != 1:
        raise ValueError(
            "Grating alignment optimization needs exactly one automatic fiber geometry."
        )
    fiber = fibers[0]
    tilted_ports = [
        port for port in payload.get("PORTS", [])
        if int(port.get("parent_component_uid", -1)) == uid
        and str(port.get("plane normal", "X")).upper() == "Z"
        and str(port.get("name", "")) == source_name
    ]
    if not source_name or len(tilted_ports) != 1 or len(input_monitors) != 1:
        raise ValueError(
            "Grating alignment optimization needs exactly one tilted fiber source port and one ordinary fiber input-power monitor."
        )
    fiber_reference_z_um = _item_vertical_reference(fiber, levels)
    fiber_z_um = fiber_reference_z_um + float(fiber.get("distance_um", 0.0))
    port_contracts = []
    for port in tilted_ports:
        reference_z_um = _item_vertical_reference(port, levels)
        distance_um = float(port.get("distance_um", 0.0))
        z_um = reference_z_um + distance_um
        port_contracts.append(
            {
                "name": str(port["name"]),
                "role": str(port.get("fiber plane role", "")),
                "is_source": str(port.get("name", "")) == source_name,
                "reference_z_um": float(reference_z_um),
                "base_distance_um": distance_um,
                "base_z_um": float(z_um),
                "base_axis_height_um": float(z_um - fiber_z_um),
                "center_um": list(map(float, port.get("center", (0.0, 0.0)))),
                "span_um": max(1e-6, float(port.get("span_um", 20.0))),
                "mode_number": max(0, int(port.get("mode number", 0))),
                "selected_mode_order": [
                    int(mode_number)
                    for mode_number in port.get("selected mode order", [])
                    if int(mode_number) > 0
                ],
                "candidate_mode_numbers": [
                    int(mode_number)
                    for mode_number in port.get("candidate mode numbers", [1, 2, 3])
                    if int(mode_number) > 0
                ],
                "fiber_target_neff": float(port.get("fiber target neff", 1.44)),
                "mode_degeneracy_tolerance": max(
                    0.0, float(port.get("mode degeneracy tolerance", 0.01))
                ),
                "minimum_local_te_fraction": float(
                    port.get("minimum local TE fraction", 0.8)
                ),
            }
        )
    for monitor in monitor_contracts:
        monitor["base_axis_height_um"] = float(
            monitor["base_z_um"] - fiber_z_um
        )
    common.update({
        "fiber_name": str(fiber["name"]),
        "fiber_center_um": list(map(float, fiber.get("center", (0.0, 0.0)))),
        "fiber_z_um": float(fiber_z_um),
        "phi_deg": float(fiber.get("angle phi", component.get("orientation_deg", 0.0))),
        "core_diameter_um": max(1e-6, float(fiber.get("core diameter_um", 9.0))),
        "ports": port_contracts,
        "monitors": monitor_contracts,
    })
    return common


def _gaussian_alignment_domain_envelope(
    fiber_pose: dict[str, Any], specification: dict[str, Any]
) -> list[float] | None:
    """Union every bounded Gaussian source/input-plane pose in XYZ.

    The adjoint seed FDTD region is fixed before the alignment optimizer
    starts.  Reserve the complete Cartesian endpoint envelope up front so an
    allowed ``fiber_offset`` or ``angle_theta`` value cannot move the source
    or its measured-input plane onto/outside a PML boundary.
    """

    if (
        str(fiber_pose.get("excitation_type", "fiber_mode")) != "gaussian_beam"
        or not fiber_pose.get("source")
        or not fiber_pose.get("monitors")
    ):
        return None
    rows = {
        str(row["parameter"]): row
        for row in specification.get("parameters", [])
    }
    nominal_theta = float(fiber_pose["nominal_angle_theta"])
    nominal_offset = float(fiber_pose.get("nominal_fiber_offset", 0.0))

    def endpoint_values(name: str, nominal: float) -> list[float]:
        row = rows.get(name)
        if row is None:
            return [nominal]
        return list(dict.fromkeys((
            float(row["minimum"]), nominal, float(row["maximum"]),
        )))

    theta_values = endpoint_values("angle_theta", nominal_theta)
    offset_values = endpoint_values("fiber_offset", nominal_offset)
    phi_rad = math.radians(float(fiber_pose.get("phi_deg", 0.0)))
    axis = np.asarray([math.cos(phi_rad), math.sin(phi_rad)], dtype=float)
    source = dict(fiber_pose["source"])
    source_nominal_center = np.asarray(source["center_um"], dtype=float)
    source_half_span = 0.5 * max(1e-6, float(source.get("span_um", 20.0)))
    is_soi = str(fiber_pose.get("component_kind", "")) == "GC-SOI"
    theta_is_active = "angle_theta" in rows
    xy_bounds: list[tuple[float, float, float, float]] = []
    z_positions: list[float] = []

    for theta_deg in theta_values:
        if not 0.0 <= theta_deg < 90.0:
            raise ValueError(
                "Gaussian alignment envelope requires angle_theta in [0, 90) degrees."
            )
        for offset_um in offset_values:
            source_center = (
                source_nominal_center
                + (offset_um - nominal_offset) * axis
            )
            source_distance = float(source["base_distance_um"])
            if is_soi and theta_is_active:
                source_distance = (
                    float(fiber_pose["fiber_tox_offset_um"])
                    * math.cos(math.radians(theta_deg))
                    - 0.35
                )
            source_z = float(source["reference_z_um"]) + source_distance
            xy_bounds.append((
                float(source_center[0]) - source_half_span,
                float(source_center[1]) - source_half_span,
                float(source_center[0]) + source_half_span,
                float(source_center[1]) + source_half_span,
            ))
            z_positions.append(source_z)

            for monitor in fiber_pose.get("monitors", []):
                monitor_distance = float(monitor["base_distance_um"])
                if is_soi and theta_is_active:
                    monitor_distance = source_distance - float(
                        fiber_pose["fiber_power_monitor_below_source_um"]
                    )
                monitor_z = float(monitor["reference_z_um"]) + monitor_distance
                monitor_center = source_center + (
                    (monitor_z - source_z)
                    * math.tan(math.radians(theta_deg))
                    * axis
                )
                # Match ``_fiber_pose_updates`` exactly: the horizontal Pin
                # plane expands by 1/cos(theta) so it still captures the
                # complete oblique Gaussian footprint at every allowed pose.
                projected_span = max(
                    1e-6,
                    float(source.get("span_um", 20.0)),
                    float(source.get("span_um", 20.0))
                    / max(math.cos(math.radians(theta_deg)), 1e-3),
                )
                x_half = 0.5 * projected_span
                y_half = 0.5 * projected_span
                xy_bounds.append((
                    float(monitor_center[0]) - x_half,
                    float(monitor_center[1]) - y_half,
                    float(monitor_center[0]) + x_half,
                    float(monitor_center[1]) + y_half,
                ))
                z_positions.append(monitor_z)

    if not xy_bounds or not z_positions:
        return None
    envelope = [
        min(bound[0] for bound in xy_bounds),
        min(bound[1] for bound in xy_bounds),
        min(z_positions),
        max(bound[2] for bound in xy_bounds),
        max(bound[3] for bound in xy_bounds),
        max(z_positions),
    ]
    if not np.all(np.isfinite(envelope)):
        raise ValueError("Gaussian alignment bounds produced a non-finite FDTD envelope.")
    return list(map(float, envelope))


def _ensure_porttransmission_receiver(
    payload: dict[str, Any], specification: dict[str, Any], warnings: list[str]
) -> None:
    """Verify that adjoint optimization reuses the exported waveguide port."""

    if str(specification.get("component_kind", "")) not in {
        "Grating coupler",
        "GC-SOI",
    }:
        return
    analysis = dict(payload.get("GRATING_ANALYSIS") or {})
    receiver_name = str(analysis.get("waveguide_port_name", ""))
    receiver = next(
        (
            port for port in payload.get("PORTS", [])
            if str(port.get("name", "")) == receiver_name
            and str(port.get("plane normal", "X")).upper() in {"X", "Y"}
        ),
        None,
    )
    if not receiver_name or receiver is None:
        raise ValueError(
            "Grating adjoint optimization requires the exported passive waveguide receiver; "
            "it will not synthesize a different optimization-only port."
        )


_OPT_FIELDS_SETUP_REMOTE = r'''# Add the static, uniform 3D adjoint volume before LumOpt owns the FDTD session.
opt_x_min, opt_x_max, opt_y_min, opt_y_max, opt_z_min, opt_z_max = map(float, OPTIMIZATION_VOLUME_UM)
opt_step_um = float(OPTIMIZATION_SPEC["optimization_mesh_um"])
OPT_LAYER_BUILDER_ORIGIN_M = [
    float(np.asarray(fdtd.getnamed("Max Layout material stack", "x")).squeeze()),
    float(np.asarray(fdtd.getnamed("Max Layout material stack", "y")).squeeze()),
]
for old_name in ("opt_fields", "opt_mesh"):
    try:
        if int(fdtd.getnamednumber(old_name)) > 0:
            fdtd.select(old_name)
            fdtd.delete()
    except Exception:
        pass

fdtd.addpower()
fdtd.set("name", "opt_fields")
fdtd.set("monitor type", "3D")
fdtd.set("x min", opt_x_min * UM)
fdtd.set("x max", opt_x_max * UM)
fdtd.set("y min", opt_y_min * UM)
fdtd.set("y max", opt_y_max * UM)
fdtd.set("z min", opt_z_min * UM)
fdtd.set("z max", opt_z_max * UM)

fdtd.addmesh()
fdtd.set("name", "opt_mesh")
fdtd.set("x min", opt_x_min * UM)
fdtd.set("x max", opt_x_max * UM)
fdtd.set("y min", opt_y_min * UM)
fdtd.set("y max", opt_y_max * UM)
fdtd.set("z min", opt_z_min * UM)
fdtd.set("z max", opt_z_max * UM)
fdtd.set("override x mesh", True)
fdtd.set("override y mesh", True)
fdtd.set("override z mesh", True)
fdtd.set("dx", opt_step_um * UM)
fdtd.set("dy", opt_step_um * UM)
fdtd.set("dz", opt_step_um * UM)
print("Uniform shape-adjoint mesh: %.6g um; monitor: opt_fields; 3D volume: %s" % (
    opt_step_um, OPTIMIZATION_VOLUME_UM,
))
'''


_LUMOPT_RUNTIME_REMOTE = r'''# Run a genuine bundled-LumOpt 3D shape-adjoint optimization.
import inspect
import json
import os
import re
import threading
import time
import numpy as np
import lumapi

UM = 1e-6
BUILD_CPU_THREADS = max(
    1,
    min(int(SETTINGS.get("build_cpu_threads", 30)), os.cpu_count() or 1),
)


def _activate_cpu_model_work(session):
    """Use the clamped CPU row for geometry edits and embedded eigensolvers."""
    session.setresource("FDTD", 1, "active", False)
    session.setresource("FDTD", 2, "device type", "CPU")
    session.setresource("FDTD", 2, "active", True)
    session.setresource("FDTD", 2, "processes", 1)
    session.setresource("FDTD", 2, "threads", BUILD_CPU_THREADS)


def _activate_gpu_solve(session):
    """Activate the GPU only at the point where an FDTD solve is launched."""
    session.setresource("FDTD", 1, "device type", "GPU")
    session.setresource("FDTD", 1, "active", True)
    session.setresource("FDTD", 2, "device type", "CPU")
    session.setresource("FDTD", 2, "active", True)
    session.setresource("FDTD", 2, "processes", 1)
    session.setresource("FDTD", 2, "threads", BUILD_CPU_THREADS)


all_parameter_rows = list(OPTIMIZATION_SPEC["parameters"])
alignment_parameter_names = list(map(str, OPTIMIZATION_SPEC.get("alignment_parameters", [])))
alignment_parameter_rows = [
    row for row in all_parameter_rows
    if str(row["parameter"]) in alignment_parameter_names
]
parameter_rows = [
    row for row in all_parameter_rows
    if str(row["parameter"]) not in alignment_parameter_names
]
parameter_names = [str(row["parameter"]) for row in parameter_rows]
initial_params = np.asarray([float(row["initial"]) for row in parameter_rows], dtype=float)
parameter_bounds = np.asarray([
    [float(row["minimum"]), float(row["maximum"])] for row in parameter_rows
], dtype=float)
alignment_initial_params = np.asarray(
    [float(row["initial"]) for row in alignment_parameter_rows], dtype=float
)
alignment_parameter_bounds = np.asarray([
    [float(row["minimum"]), float(row["maximum"])]
    for row in alignment_parameter_rows
], dtype=float)
if len(all_parameter_rows) < 1:
    raise RuntimeError("No optimization parameters were embedded")


def _numeric_source_mode_label(value):
    """Return canonical ``mode N`` text or fail before any solver launch."""
    text = str(value).strip()
    match = re.fullmatch(r"mode\s+([1-9][0-9]*)", text, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(
            "The resolved optimization source mode must be numeric 'mode N'; received %r. "
            "The 3D seed build must finish its three-candidate fiber-polarization selection first."
            % text
        )
    return "mode %d" % int(match.group(1))


def _require_numeric_source_mode():
    """Validate and canonicalize the fiber-port source-mode label."""
    source_mode = _numeric_source_mode_label(
        OPT_OBJECTIVE_PORTS.get("source_mode", "")
    )
    OPT_OBJECTIVE_PORTS["source_mode"] = source_mode
    return source_mode


def _uses_gaussian_source():
    return (
        str(OPT_OBJECTIVE_PORTS.get("excitation_type", "fiber_mode")).strip().lower()
        == "gaussian_beam"
        or str(OPT_OBJECTIVE_PORTS.get("source_kind", "fiber_port")).strip().lower()
        in {"gaussian", "gaussian_beam", "gaussian source"}
    )


def _require_optimization_source():
    """Validate the independent Gaussian source or numeric fiber-port mode."""
    source_name = str(
        OPT_OBJECTIVE_PORTS.get(
            "source_name", OPT_OBJECTIVE_PORTS.get("source_port", "")
        )
    ).strip()
    if not source_name:
        raise RuntimeError("The optimization source name is empty")
    OPT_OBJECTIVE_PORTS["source_name"] = source_name
    OPT_OBJECTIVE_PORTS["source_port"] = source_name
    if _uses_gaussian_source():
        return source_name, None
    return source_name, _require_numeric_source_mode()


def _configure_optimization_source(fdtd_session):
    """Activate exactly one source while leaving the waveguide port passive."""
    source_name, source_mode = _require_optimization_source()
    fdtd_session.select("FDTD::ports")
    if _uses_gaussian_source():
        # An empty source-port selection makes every FDTD port passive.  The
        # independent Gaussian object then owns the only injected field.
        fdtd_session.set("source port", "")
        _set_named(fdtd_session, "::model::" + source_name, "enabled", True)
    else:
        fdtd_session.set("source port", source_name)
        fdtd_session.set("source mode", source_mode)
    return source_name, source_mode


def _positive_unique_modes(values):
    result = []
    for value in values or []:
        try:
            mode_number = int(value)
        except Exception:
            continue
        if mode_number > 0 and mode_number not in result:
            result.append(mode_number)
    return result


def _resolved_runtime_port_mode(port_contract):
    """Resolve the selected HE11 partner and winner-first retained pair."""
    port_name = str(port_contract["name"])
    selections = globals().get("PORT_MODE_SELECTIONS", {})
    selection = dict(selections.get(port_name, {})) if isinstance(selections, dict) else {}
    ports_by_name = {
        str(port.get("name", "")): port
        for port in globals().get("PORTS", [])
    }
    runtime_port = dict(ports_by_name.get(port_name, {}))

    selected_mode = max(
        0,
        int(selection.get(
            "mode number",
            runtime_port.get("mode number", port_contract.get("mode_number", 0)),
        )),
    )
    selected_order = _positive_unique_modes(
        selection.get(
            "selected mode order",
            runtime_port.get(
                "selected mode order",
                port_contract.get("selected_mode_order", []),
            ),
        )
    )
    candidates = _positive_unique_modes(
        selection.get(
            "candidate mode numbers",
            runtime_port.get(
                "candidate mode numbers",
                port_contract.get("candidate_mode_numbers", [1, 2, 3]),
            ),
        )
    )

    if selected_mode <= 0:
        analysis = globals().get("GRATING_ANALYSIS") or {}
        try:
            selected_mode = int(
                _numeric_source_mode_label(
                    analysis.get("fiber_source_mode", "")
                ).split()[1]
            )
        except Exception:
            selected_mode = 0
    if selected_mode <= 0:
        raise RuntimeError(
            "No resolved numeric local-TE fiber mode is available for optimization port %s"
            % port_name
        )

    retained = [selected_mode]
    retained.extend(
        mode_number
        for mode_number in [*selected_order, *candidates]
        if mode_number != selected_mode and mode_number not in retained
    )
    if len(retained) < 2:
        raise RuntimeError(
            "Optimization port %s must retain both near-degenerate fiber modes; resolved %r"
            % (port_name, retained)
        )
    return int(selected_mode), retained[:2]


def _synchronize_resolved_fiber_mode_contract():
    """Copy seed-build mode choices into the persistent optimization contract."""
    gaussian_source = (
        str(OPT_OBJECTIVE_PORTS.get("excitation_type", "fiber_mode")).strip().lower()
        == "gaussian_beam"
        or str(OPT_OBJECTIVE_PORTS.get("source_kind", "fiber_port")).strip().lower()
        in {"gaussian", "gaussian_beam", "gaussian source"}
    )
    if gaussian_source:
        source_name = str(
            OPT_OBJECTIVE_PORTS.get(
                "source_name", OPT_OBJECTIVE_PORTS.get("source_port", "")
            )
        ).strip()
        if not source_name:
            raise RuntimeError("The optimization Gaussian source name is empty")
        OPT_OBJECTIVE_PORTS["source_name"] = source_name
        OPT_OBJECTIVE_PORTS["source_port"] = source_name
        if globals().get("GRATING_ANALYSIS"):
            GRATING_ANALYSIS["excitation_type"] = "gaussian_beam"
            GRATING_ANALYSIS["source_kind"] = "gaussian"
            GRATING_ANALYSIS["source_name"] = source_name
        return
    if not OPT_FIBER_POSE:
        if str(OPT_OBJECTIVE_PORTS.get("kind", "")).startswith("grating"):
            source_mode_number, selected_order = _resolved_runtime_port_mode({
                "name": str(OPT_OBJECTIVE_PORTS["source_port"]),
                "is_source": True,
                "mode_number": 0,
                "selected_mode_order": [],
                "candidate_mode_numbers": [1, 2, 3],
            })
            source_mode = "mode %d" % source_mode_number
            OPT_OBJECTIVE_PORTS["source_mode"] = source_mode
            if globals().get("GRATING_ANALYSIS"):
                GRATING_ANALYSIS["fiber_source_mode"] = source_mode
                GRATING_ANALYSIS["fiber_source_mode_number"] = source_mode_number
                GRATING_ANALYSIS["fiber_source_selected_mode_order"] = list(
                    selected_order
                )
        _require_numeric_source_mode()
        return

    source_mode_number = 0
    for port_contract in OPT_FIBER_POSE.get("ports", []):
        selected_mode, selected_order = _resolved_runtime_port_mode(port_contract)
        port_contract["mode_number"] = selected_mode
        port_contract["selected_mode_order"] = list(selected_order)
        if bool(port_contract.get("is_source", False)):
            source_mode_number = selected_mode

    if source_mode_number <= 0:
        raise RuntimeError("The optimization fiber-pose contract has no source port")
    source_mode = "mode %d" % source_mode_number
    OPT_OBJECTIVE_PORTS["source_mode"] = source_mode
    if globals().get("GRATING_ANALYSIS"):
        GRATING_ANALYSIS["fiber_source_mode"] = source_mode
        GRATING_ANALYSIS["fiber_source_mode_number"] = source_mode_number
        GRATING_ANALYSIS["fiber_source_selected_mode_order"] = list(
            next(
                port["selected_mode_order"]
                for port in OPT_FIBER_POSE.get("ports", [])
                if bool(port.get("is_source", False))
            )
        )
    _require_numeric_source_mode()

REMOTE_OPT_PROGRESS_FILE = globals().get(
    "REMOTE_OPT_PROGRESS_FILE",
    os.path.join(REMOTE_WORK, "adjoint_live_progress.jsonl"),
)
os.makedirs(os.path.dirname(REMOTE_OPT_PROGRESS_FILE), exist_ok=True)
with open(REMOTE_OPT_PROGRESS_FILE, "w", encoding="utf-8"):
    pass
_OPT_PROGRESS_SEQUENCE = 0


def _selected_parameter_map(alignment_values=None, shape_values=None):
    """Return every selected JSON parameter in the user's displayed order."""
    alignment = (
        alignment_initial_params
        if alignment_values is None
        else np.asarray(alignment_values, dtype=float).ravel()
    )
    shape = (
        initial_params
        if shape_values is None
        else np.asarray(shape_values, dtype=float).ravel()
    )
    values = {
        str(row["parameter"]): float(alignment[index])
        for index, row in enumerate(alignment_parameter_rows)
    }
    values.update({
        str(row["parameter"]): float(shape[index])
        for index, row in enumerate(parameter_rows)
    })
    return {
        str(row["parameter"]): float(values[str(row["parameter"])])
        for row in all_parameter_rows
    }


def _emit_live_progress(stage, iteration, objective_value, parameters):
    """Append one atomic, externally readable completed-iteration record."""
    global _OPT_PROGRESS_SEQUENCE
    try:
        objective_float = float(objective_value)
        parameter_map = {
            str(name): float(value) for name, value in dict(parameters).items()
        }
        if not np.isfinite(objective_float) or not all(
            np.isfinite(value) for value in parameter_map.values()
        ):
            return
        _OPT_PROGRESS_SEQUENCE += 1
        record = {
            "sequence": int(_OPT_PROGRESS_SEQUENCE),
            "stage": str(stage),
            "iteration": int(iteration),
            "objective": objective_float,
            "parameter_names": [str(row["parameter"]) for row in all_parameter_rows],
            "parameters": parameter_map,
            "timestamp": float(time.time()),
        }
        # Open/write/close per record so the second SSH connection immediately
        # observes it on NFS. Reporting must never own or rerun an FDTD solve.
        with open(REMOTE_OPT_PROGRESS_FILE, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
    except Exception:
        # A display failure must never abort a costly optimization.
        return


def _interpolated_target_vertices(parameters):
    values = np.asarray(parameters, dtype=float).ravel()
    if values.size != len(parameter_rows):
        raise RuntimeError("LumOpt supplied %d parameters; expected %d" % (
            values.size, len(parameter_rows),
        ))
    polygons = [np.asarray(vertices, dtype=float).copy() for vertices in OPT_SHAPE_SNAPSHOTS["nominal"]]
    for parameter_index, row in enumerate(parameter_rows):
        name = str(row["parameter"])
        initial = float(row["initial"])
        minimum = float(row["minimum"])
        maximum = float(row["maximum"])
        value = float(values[parameter_index])
        endpoint = OPT_SHAPE_SNAPSHOTS["parameters"][name]
        if value >= initial:
            denominator = max(maximum - initial, 1e-15)
            fraction = (value - initial) / denominator
            destination = endpoint["maximum"]
        else:
            denominator = max(initial - minimum, 1e-15)
            fraction = (initial - value) / denominator
            destination = endpoint["minimum"]
        for polygon_index, destination_vertices in enumerate(destination):
            nominal = np.asarray(OPT_SHAPE_SNAPSHOTS["nominal"][polygon_index], dtype=float)
            destination_array = np.asarray(destination_vertices, dtype=float)
            if destination_array.shape != nominal.shape:
                raise RuntimeError("Fixed-topology polygon shape changed for " + name)
            polygons[polygon_index] += fraction * (destination_array - nominal)
    return polygons


def _layer_builder_geometry_value(parameters, local_origin_m):
    optimized = dict(zip(
        OPT_SHAPE_SNAPSHOTS["polygon_names"],
        _interpolated_target_vertices(parameters),
    ))
    local_origin_m = np.asarray(local_origin_m, dtype=float)
    geometry_by_layer = {}
    for polygon in GEOMETRY:
        name = str(polygon["name"])
        vertices_um = optimized.get(name, np.asarray(polygon["vertices_um"], dtype=float))
        key = "%d:%d" % (int(polygon["layer"]), int(polygon.get("datatype", 0)))
        local_vertices_m = np.asarray(vertices_um, dtype=float) * UM - local_origin_m
        geometry_by_layer.setdefault(key, []).append(local_vertices_m)
    return geometry_by_layer


def update_max_layout_geometry(parameters, fdtd_session, only_update):
    """Legacy ParameterizedGeometry callback; updates Layer Builder in-place."""
    layer_x_m = float(np.asarray(fdtd_session.getnamed("Max Layout material stack", "x")).squeeze())
    layer_y_m = float(np.asarray(fdtd_session.getnamed("Max Layout material stack", "y")).squeeze())
    geometry_by_layer = _layer_builder_geometry_value(parameters, [layer_x_m, layer_y_m])
    fdtd_session.setnamed("Max Layout material stack", "geometry", geometry_by_layer)


def _shape_parameter_map(parameters):
    values = np.asarray(parameters, dtype=float).ravel()
    return {
        name: float(values[index]) for index, name in enumerate(parameter_names)
    }


def _fiber_pose_updates(alignment_parameters, shape_parameters=None):
    """Return one synchronized source and input-measurement pose."""
    if not OPT_FIBER_POSE:
        return {}
    alignment_values = np.asarray(alignment_parameters, dtype=float).ravel()
    alignment_map = {
        str(row["parameter"]): float(alignment_values[index])
        for index, row in enumerate(alignment_parameter_rows)
    }
    shape_map = _shape_parameter_map(
        initial_params if shape_parameters is None else shape_parameters
    )
    theta_deg = float(alignment_map.get(
        "angle_theta", OPT_FIBER_POSE["nominal_angle_theta"]
    ))
    if not 0.0 <= theta_deg < 90.0:
        raise RuntimeError("angle_theta must be at least 0 and below 90 degrees")

    nominal = dict(OPT_COMPONENT_NOMINAL_PARAMS)
    current = dict(nominal)
    current.update(shape_map)
    current.update(alignment_map)

    def _flare_x(values):
        if str(OPT_FIBER_POSE["component_kind"]) == "GC-SOI":
            return float(values.get("wg_length", 0.0)) + float(values.get("radius", 0.0))
        width = float(values.get("wg_width", 0.0))
        alpha = float(values.get("alpha_t", 25.0))
        tangent = np.tan(0.5 * np.deg2rad(alpha))
        if abs(tangent) < 1e-15:
            raise RuntimeError("Grating aperture angle is singular")
        return (
            float(values.get("wg_length", 0.0))
            - 0.5 * width / tangent
            + float(values.get("taper_L", 0.0))
        )

    center_delta_um = 0.0
    if "fiber_offset" in alignment_parameter_names:
        nominal_center_x = _flare_x(nominal) + float(
            OPT_FIBER_POSE["nominal_fiber_offset"]
        )
        current_center_x = _flare_x(current) + float(current["fiber_offset"])
        center_delta_um = current_center_x - nominal_center_x

    phi_deg = float(OPT_FIBER_POSE["phi_deg"])
    phi_rad = np.deg2rad(phi_deg)
    axis_unit = np.asarray([np.cos(phi_rad), np.sin(phi_rad)], dtype=float)
    if str(OPT_FIBER_POSE.get("excitation_type", "fiber_mode")) == "gaussian_beam":
        source_contract = dict(OPT_FIBER_POSE["source"])
        source_center_um = (
            np.asarray(source_contract["center_um"], dtype=float)
            + center_delta_um * axis_unit
        )
        source_distance_um = float(source_contract["base_distance_um"])
        if (
            str(OPT_FIBER_POSE["component_kind"]) == "GC-SOI"
            and "angle_theta" in alignment_parameter_names
        ):
            source_distance_um = (
                float(OPT_FIBER_POSE["fiber_tox_offset_um"])
                * np.cos(np.deg2rad(theta_deg))
                - 0.35
            )
        source_z_um = float(source_contract["reference_z_um"]) + source_distance_um
        updates = {
            "source_kind": "gaussian",
            "source": {
                "name": str(source_contract["name"]),
                "center_um": source_center_um,
                "z_um": source_z_um,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                # S polarization is normal to the grating plane of incidence
                # and therefore remains local TE for every component rotation.
                "polarization_angle_deg": 90.0,
                "span_um": float(source_contract.get("span_um", 20.0)),
            },
            "ports": [],
            "monitors": [],
        }
        for monitor in OPT_FIBER_POSE.get("monitors", []):
            distance_um = float(monitor["base_distance_um"])
            if (
                str(OPT_FIBER_POSE["component_kind"]) == "GC-SOI"
                and "angle_theta" in alignment_parameter_names
            ):
                distance_um = source_distance_um - float(
                    OPT_FIBER_POSE["fiber_power_monitor_below_source_um"]
                )
            z_um = float(monitor["reference_z_um"]) + distance_um
            # The ordinary DFT plane stays horizontal.  Only its center follows
            # the tilted beam axis from the independent source plane.
            center_um = source_center_um + (
                (z_um - source_z_um)
                * np.tan(np.deg2rad(theta_deg))
                * axis_unit
            )
            updates["monitors"].append(
                {
                    "name": str(monitor["name"]),
                    "center_um": center_um,
                    "z_um": z_um,
                    "expected_propagation_sign": float(
                        monitor.get("expected_propagation_sign", -1.0)
                    ),
                    "projected_span_um": float(
                        source_contract.get("input_monitor_span_scale", 1.2)
                    ) * max(
                        float(source_contract.get("span_um", 20.0)),
                        float(source_contract.get("span_um", 20.0))
                        / max(np.cos(np.deg2rad(theta_deg)), 1e-3),
                    ),
                    "role": str(monitor.get("role", "input power measurement")),
                }
            )
        return updates

    bottom_center_um = (
        np.asarray(OPT_FIBER_POSE["fiber_center_um"], dtype=float)
        + center_delta_um * axis_unit
    )
    updates = {
        "source_kind": "fiber_port",
        "fiber": {
            "name": str(OPT_FIBER_POSE["fiber_name"]),
            "center_um": bottom_center_um,
            "theta_deg": theta_deg,
        },
        "ports": [],
        "monitors": [],
    }
    for port in OPT_FIBER_POSE["ports"]:
        distance_um = float(port["base_distance_um"])
        if str(OPT_FIBER_POSE["component_kind"]) == "GC-SOI" and "angle_theta" in alignment_parameter_names:
            source_distance_um = (
                float(OPT_FIBER_POSE["fiber_tox_offset_um"])
                * np.cos(np.deg2rad(theta_deg))
                - 0.35
            )
            distance_um = source_distance_um
        z_um = float(port["reference_z_um"]) + distance_um
        axis_height_um = z_um - float(OPT_FIBER_POSE["fiber_z_um"])
        center_um = bottom_center_um + (
            axis_height_um * np.tan(np.deg2rad(theta_deg)) * axis_unit
        )
        updates["ports"].append({
            "name": str(port["name"]),
            "center_um": center_um,
            "z_um": z_um,
            "theta_deg": theta_deg,
            "phi_deg": float(OPT_FIBER_POSE["phi_deg"]),
            "rotation_offset_um": (
                4.0 * float(OPT_FIBER_POSE["core_diameter_um"])
                * np.tan(np.deg2rad(theta_deg))
            ),
            "mode_number": int(port.get("mode_number", 0)),
            "selected_mode_order": list(port.get("selected_mode_order", [])),
            "candidate_mode_numbers": list(
                port.get("candidate_mode_numbers", [1, 2, 3])
            ),
            "mode_degeneracy_tolerance": float(
                port.get("mode_degeneracy_tolerance", 0.01)
            ),
            "fiber_target_neff": float(port.get("fiber_target_neff", 1.44)),
            "minimum_local_te_fraction": float(
                port.get("minimum_local_te_fraction", 0.8)
            ),
            "is_source": bool(port.get("is_source", False)),
            "role": str(port.get("role", "")),
        })
    for monitor in OPT_FIBER_POSE.get("monitors", []):
        distance_um = float(monitor["base_distance_um"])
        if (
            str(OPT_FIBER_POSE["component_kind"]) == "GC-SOI"
            and "angle_theta" in alignment_parameter_names
        ):
            source_distance_um = (
                float(OPT_FIBER_POSE["fiber_tox_offset_um"])
                * np.cos(np.deg2rad(theta_deg))
                - 0.35
            )
            distance_um = source_distance_um - float(
                OPT_FIBER_POSE["fiber_power_monitor_below_source_um"]
            )
        z_um = float(monitor["reference_z_um"]) + distance_um
        axis_height_um = z_um - float(OPT_FIBER_POSE["fiber_z_um"])
        center_um = bottom_center_um + (
            axis_height_um * np.tan(np.deg2rad(theta_deg)) * axis_unit
        )
        updates["monitors"].append(
            {
                "name": str(monitor["name"]),
                "center_um": center_um,
                "z_um": z_um,
                "expected_propagation_sign": float(
                    monitor.get("expected_propagation_sign", -1.0)
                ),
                "projected_span_um": max(
                    float(OPT_FIBER_POSE["ports"][0].get("span_um", 20.0)),
                    float(OPT_FIBER_POSE["ports"][0].get("span_um", 20.0))
                    / max(np.cos(np.deg2rad(theta_deg)), 1e-3),
                ),
                "role": str(monitor.get("role", "input power measurement")),
            }
        )
    return updates


def _set_named(session, path, property_name, value):
    try:
        session.setnamed(path, property_name, value)
    except Exception:
        session.select(path)
        session.set(property_name, value)


def _apply_fiber_pose_to_session(
    alignment_parameters, fdtd_session, shape_parameters=None, update_modes=True
):
    """Apply angle/offset to the selected source and input monitor atomically."""
    updates = _fiber_pose_updates(alignment_parameters, shape_parameters)
    if not updates:
        return
    fdtd_session.switchtolayout()
    if str(updates.get("source_kind", "fiber_port")) == "gaussian":
        source = updates["source"]
        source_path = "::model::" + str(source["name"])
        _set_named(fdtd_session, source_path, "x", float(source["center_um"][0]) * UM)
        _set_named(fdtd_session, source_path, "y", float(source["center_um"][1]) * UM)
        _set_named(fdtd_session, source_path, "z", float(source["z_um"]) * UM)
        _set_named(fdtd_session, source_path, "angle theta", float(source["theta_deg"]))
        _set_named(fdtd_session, source_path, "angle phi", float(source["phi_deg"]))
        _set_named(
            fdtd_session,
            source_path,
            "polarization angle",
            90.0,
        )
        _set_named(fdtd_session, source_path, "enabled", True)
    else:
        fiber = updates["fiber"]
        fiber_path = "::model::" + str(fiber["name"])
        _set_named(fdtd_session, fiber_path, "x", float(fiber["center_um"][0]) * UM)
        _set_named(fdtd_session, fiber_path, "y", float(fiber["center_um"][1]) * UM)
        _set_named(fdtd_session, fiber_path, "theta", float(fiber["theta_deg"]))
        fdtd_session.runsetup()
    for port in updates["ports"]:
        port_path = "FDTD::ports::" + str(port["name"])
        _set_named(fdtd_session, port_path, "x", float(port["center_um"][0]) * UM)
        _set_named(fdtd_session, port_path, "y", float(port["center_um"][1]) * UM)
        _set_named(fdtd_session, port_path, "z", float(port["z_um"]) * UM)
        _set_named(fdtd_session, port_path, "theta", float(port["theta_deg"]))
        if abs(float(port["phi_deg"])) > 1e-12:
            _set_named(fdtd_session, port_path, "phi", float(port["phi_deg"]))
        _set_named(
            fdtd_session,
            port_path,
            "rotation offset",
            float(port["rotation_offset_um"]) * UM,
        )
        if update_modes:
            fdtd_session.select(port_path)
            selection = None
            selector = globals().get("_select_fiber_local_te_mode")
            if callable(selector):
                selection = dict(selector(
                    fdtd_session,
                    port_path,
                    {
                        "name": str(port["name"]),
                        "angle phi": float(port["phi_deg"]),
                        "candidate mode numbers": list(
                            port.get("candidate_mode_numbers", [1, 2, 3])
                        ),
                        "fiber target neff": float(
                            port.get("fiber_target_neff", 1.44)
                        ),
                        "mode degeneracy tolerance": float(
                            port.get("mode_degeneracy_tolerance", 0.01)
                        ),
                        "minimum local TE fraction": float(
                            port.get("minimum_local_te_fraction", 0.8)
                        ),
                    },
                ))
                mode_number = max(0, int(selection.get("mode number", 0)))
                mode_order = _positive_unique_modes(
                    selection.get("selected mode order", [])
                )
            else:
                # Isolated unit tests and legacy imported runtimes may not
                # contain the build helper.  Preserve the already validated
                # winner-first pair; never invent a fixed mode-2/Ey fallback.
                mode_number = max(0, int(port["mode_number"]))
                mode_order = _positive_unique_modes(
                    port.get("selected_mode_order", [])
                )
            if mode_number <= 0:
                raise RuntimeError(
                    "Fiber port %s has no resolved numeric local-TE mode" % port["name"]
                )
            mode_order = [mode_number] + [
                candidate
                for candidate in (
                    mode_order
                    + _positive_unique_modes(port.get("candidate_mode_numbers", []))
                )
                if candidate != mode_number
            ]
            mode_order = _positive_unique_modes(mode_order)
            if len(mode_order) < 2:
                raise RuntimeError(
                    "Fiber port %s must retain both near-degenerate modes; resolved %r"
                    % (port["name"], mode_order)
                )
            mode_order = mode_order[:2]
            if selection is None:
                update_status = fdtd_session.updateportmodes(
                    np.asarray(mode_order, dtype=int)
                )
                if isinstance(update_status, (bool, np.bool_)) and not bool(update_status):
                    raise RuntimeError(
                        "Lumerical rejected the retained mode pair %r for fiber port %s"
                        % (mode_order, port["name"])
                    )
                selection = {
                    "mode number": mode_number,
                    "selected mode order": list(mode_order),
                    "candidate mode numbers": list(
                        port.get("candidate_mode_numbers", mode_order)
                    ),
                    "polarization": "local TE",
                }
            port["mode_number"] = mode_number
            port["selected_mode_order"] = list(mode_order)
            for contract in OPT_FIBER_POSE.get("ports", []):
                if str(contract.get("name", "")) == str(port["name"]):
                    contract["mode_number"] = mode_number
                    contract["selected_mode_order"] = list(mode_order)
            selections = globals().get("PORT_MODE_SELECTIONS")
            if isinstance(selections, dict):
                selections[str(port["name"])] = dict(selection)
            if globals().get("GRATING_ANALYSIS"):
                GRATING_ANALYSIS["fiber_source_mode"] = "mode %d" % mode_number
                GRATING_ANALYSIS["fiber_source_mode_number"] = mode_number
                GRATING_ANALYSIS["fiber_selected_mode_order"] = list(mode_order)
            if bool(port.get("is_source", False)):
                OPT_OBJECTIVE_PORTS["source_mode"] = "mode %d" % mode_number
    # The input plane is an ordinary DFT power monitor.  Move it with the
    # fiber-axis intersection, but never call a port eigensolver for it.
    for monitor in updates.get("monitors", []):
        monitor_path = "::model::" + str(monitor["name"])
        _set_named(fdtd_session, monitor_path, "x", float(monitor["center_um"][0]) * UM)
        _set_named(fdtd_session, monitor_path, "y", float(monitor["center_um"][1]) * UM)
        _set_named(fdtd_session, monitor_path, "z", float(monitor["z_um"]) * UM)
        _set_named(
            fdtd_session,
            monitor_path,
            "x span",
            float(monitor["projected_span_um"]) * UM,
        )
        _set_named(
            fdtd_session,
            monitor_path,
            "y span",
            float(monitor["projected_span_um"]) * UM,
        )
    fdtd_session.select("FDTD::ports")
    gaussian_source = (
        str(OPT_OBJECTIVE_PORTS.get("excitation_type", "fiber_mode")).strip().lower()
        == "gaussian_beam"
        or str(OPT_OBJECTIVE_PORTS.get("source_kind", "fiber_port")).strip().lower()
        in {"gaussian", "gaussian_beam", "gaussian source"}
    )
    if gaussian_source:
        source_name = str(
            OPT_OBJECTIVE_PORTS.get(
                "source_name", OPT_OBJECTIVE_PORTS.get("source_port", "")
            )
        ).strip()
        if not source_name:
            raise RuntimeError("The optimization Gaussian source name is empty")
        fdtd_session.set("source port", "")
        _set_named(fdtd_session, "::model::" + source_name, "enabled", True)
    else:
        fdtd_session.set("source port", str(OPT_OBJECTIVE_PORTS["source_port"]))
        fdtd_session.set("source mode", _require_numeric_source_mode())


objective = dict(OPTIMIZATION_SPEC["objective"])
optimizer_settings = dict(OPTIMIZATION_SPEC["optimizer"])
REMOTE_OPTIMIZATION_DIR = os.path.join(REMOTE_WORK, "adjoint_optimization")
os.makedirs(REMOTE_OPTIMIZATION_DIR, exist_ok=True)
REMOTE_VALIDATION_FSP = os.path.join(
    REMOTE_OPTIMIZATION_DIR, "_transient_best_validation.fsp"
)


def _history_arrays(history, parameter_count):
    """Read either lumopt2's public history or legacy optimizer histories."""
    fom = np.empty(0, dtype=float)
    params = np.empty((0, parameter_count), dtype=float)
    if isinstance(history, dict):
        for key in ("fom", "fom_history", "objective", "objective_history"):
            if key in history:
                try:
                    fom = np.asarray(history[key], dtype=float).ravel()
                    break
                except Exception:
                    pass
        for key in ("params", "parameters", "parameter_history", "params_history"):
            if key in history:
                try:
                    candidate = np.asarray(history[key], dtype=float)
                    candidate = candidate.reshape((-1, parameter_count))
                    params = candidate
                    break
                except Exception:
                    pass
    return fom, params


def _emit_new_shape_history(history_reader, engine_label, state, alignment_values):
    """Publish newly completed shape iterations without evaluating the FOM."""
    try:
        history = history_reader()
        fom_values, parameter_values = _history_arrays(history, initial_params.size)
        completed = min(fom_values.size, parameter_values.shape[0])
        while int(state["emitted"]) < completed:
            index = int(state["emitted"])
            _emit_live_progress(
                engine_label,
                index,
                fom_values[index],
                _selected_parameter_map(alignment_values, parameter_values[index]),
            )
            state["emitted"] = index + 1
    except Exception:
        # Histories can be momentarily inconsistent while an optimizer appends
        # them. The next callback/poll retries; simulation ownership is untouched.
        return


def _start_shape_history_monitor(history_reader, engine_label, alignment_values):
    """Poll Python-only history while the blocking LumOpt run owns FDTD."""
    stop = threading.Event()
    state = {"emitted": 0}

    def poll():
        _emit_new_shape_history(
            history_reader, engine_label, state, alignment_values
        )

    def worker():
        while not stop.wait(0.5):
            poll()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return {"stop": stop, "thread": thread, "poll": poll, "state": state}


def _stop_shape_history_monitor(monitor):
    monitor["stop"].set()
    monitor["thread"].join(timeout=2.0)
    monitor["poll"]()


ALIGNMENT_EVALUATION_PARAMETERS = []
ALIGNMENT_EVALUATION_FOM = []
ALIGNMENT_ITERATION_PARAMETERS = []
ALIGNMENT_ITERATION_FOM = []


def _alignment_port_score(fdtd_session):
    def spectrum(dataset, key):
        if "lambda" in dataset:
            wavelength_m = np.asarray(dataset["lambda"], dtype=float).squeeze().ravel()
        elif "f" in dataset:
            wavelength_m = 299792458.0 / np.asarray(
                dataset["f"], dtype=float
            ).squeeze().ravel()
        else:
            raise RuntimeError("Power result has neither lambda nor frequency")
        available = {str(name).lower().replace("_", ""): name for name in dataset.keys()}
        resolved = available.get(str(key).lower().replace("_", ""), key)
        values = np.real(np.asarray(dataset[resolved]).squeeze())
        if values.ndim == 0:
            values = np.full(wavelength_m.size, values)
        elif values.ndim > 1:
            axes = [
                axis for axis, size in enumerate(values.shape)
                if size == wavelength_m.size
            ]
            if not axes:
                raise RuntimeError(
                    "Could not align %s shape %s with wavelength" % (resolved, values.shape)
                )
            values = np.moveaxis(values, axes[0], 0).reshape(wavelength_m.size, -1)[:, 0]
        order = np.argsort(wavelength_m)
        return wavelength_m[order], np.asarray(values, dtype=float).ravel()[order]

    input_dataset = fdtd_session.getresult(
        str(OPT_OBJECTIVE_PORTS["fiber_input_power_monitor"]), "T"
    )
    input_wavelength_m, input_signed = spectrum(input_dataset, "T")
    input_power = float(OPT_OBJECTIVE_PORTS["fiber_input_power_sign"]) * input_signed
    expansion_path = "FDTD::ports::" + str(OPT_OBJECTIVE_PORTS["monitor_port"])
    expansion_dataset = fdtd_session.getresult(
        expansion_path,
        str(OPT_OBJECTIVE_PORTS["waveguide_port_expansion_result_name"]),
    )
    wavelength_m, modal_power = spectrum(
        expansion_dataset,
        str(OPT_OBJECTIVE_PORTS["waveguide_port_modal_direction"]),
    )
    modal_power = float(
        OPT_OBJECTIVE_PORTS.get("waveguide_port_modal_sign", 1.0)
    ) * modal_power
    input_power = np.interp(wavelength_m, input_wavelength_m, input_power)
    if np.any(input_power <= 1e-15):
        raise RuntimeError(
            "Alignment input-power monitor has wrong/near-zero signed flux"
        )
    transmission = modal_power / input_power
    finite = transmission[np.isfinite(transmission)]
    if finite.size < 1:
        raise RuntimeError("Alignment solve returned no finite grating coupling spectrum")
    if float(np.min(finite)) < -1e-9 or float(np.max(finite)) > 1.05:
        raise RuntimeError(
            "Alignment solve returned unphysical measured-input-normalized CE: %r"
            % finite.tolist()
        )
    return float(np.mean(finite))


def _alignment_loss(alignment_values, fdtd_session, shape_values):
    values = np.asarray(alignment_values, dtype=float).ravel()
    _activate_cpu_model_work(fdtd_session)
    _apply_fiber_pose_to_session(values, fdtd_session, shape_values, update_modes=True)
    _require_optimization_source()
    _activate_gpu_solve(fdtd_session)
    fdtd_session.run("FDTD", "GPU")
    score = _alignment_port_score(fdtd_session)
    ALIGNMENT_EVALUATION_PARAMETERS.append(values.copy())
    ALIGNMENT_EVALUATION_FOM.append(score)
    return -score


def _nearest_alignment_score(values):
    """Return the already-solved score nearest SciPy's accepted iterate."""
    if not ALIGNMENT_EVALUATION_FOM:
        return float("nan")
    candidate = np.asarray(values, dtype=float).ravel()
    evaluated = np.asarray(ALIGNMENT_EVALUATION_PARAMETERS, dtype=float)
    span = np.maximum(
        alignment_parameter_bounds[:, 1] - alignment_parameter_bounds[:, 0],
        1e-15,
    )
    distance = np.linalg.norm((evaluated - candidate[None, :]) / span[None, :], axis=1)
    return float(ALIGNMENT_EVALUATION_FOM[int(np.nanargmin(distance))])


def _alignment_iteration_callback(intermediate_result):
    """Report one accepted SciPy iterate without launching another solve."""
    values = np.asarray(
        getattr(intermediate_result, "x", intermediate_result), dtype=float
    ).ravel()
    objective_value = getattr(intermediate_result, "fun", None)
    if objective_value is None or not np.isfinite(float(objective_value)):
        score = _nearest_alignment_score(values)
    else:
        score = -float(objective_value)
    ALIGNMENT_ITERATION_PARAMETERS.append(values.copy())
    ALIGNMENT_ITERATION_FOM.append(score)
    _emit_live_progress(
        "fiber alignment",
        len(ALIGNMENT_ITERATION_FOM),
        score,
        _selected_parameter_map(values, initial_params),
    )


def _optimize_fiber_alignment(base_fsp, shape_values, output_fsp, max_iterations):
    """Bounded forward-solve stage for source position and tilt."""
    if not alignment_parameter_rows:
        return alignment_initial_params.copy(), float("nan")
    from scipy.optimize import minimize

    owner = lumapi.FDTD(
        hide=True,
        serverArgs={"threads": str(BUILD_CPU_THREADS)},
    )
    try:
        owner.load(base_fsp)
        _activate_cpu_model_work(owner)
        steps = np.asarray([
            0.05 if str(row["parameter"]) == "angle_theta" else 0.01
            for row in alignment_parameter_rows
        ], dtype=float)
        result = minimize(
            lambda values: _alignment_loss(values, owner, shape_values),
            alignment_initial_params,
            method="L-BFGS-B",
            bounds=[tuple(map(float, bound)) for bound in alignment_parameter_bounds],
            callback=_alignment_iteration_callback,
            options={
                "maxiter": max(1, int(max_iterations)),
                "ftol": float(optimizer_settings.get("ftol", 1e-5)),
                "gtol": float(optimizer_settings.get("pgtol", 1e-5)),
                "eps": steps,
            },
        )
        if ALIGNMENT_EVALUATION_FOM:
            best_index = int(np.nanargmax(np.asarray(ALIGNMENT_EVALUATION_FOM, dtype=float)))
            best_values = np.asarray(
                ALIGNMENT_EVALUATION_PARAMETERS[best_index], dtype=float
            ).copy()
            best_score = float(ALIGNMENT_EVALUATION_FOM[best_index])
        else:
            best_values = np.asarray(result.x, dtype=float).ravel()
            best_score = float("nan")
        _apply_fiber_pose_to_session(
            best_values, owner, shape_values, update_modes=True
        )
        owner.save(output_fsp)
        if not os.path.isfile(output_fsp) or os.path.getsize(output_fsp) <= 0:
            raise RuntimeError("Alignment stage did not save its optimized FSP")
        return best_values, best_score
    finally:
        try:
            owner.close()
        except Exception:
            pass


def _import_lumopt2():
    try:
        import ansys.lumerical.core.lumopt2 as module
        return module, "ansys.lumerical.core.lumopt2"
    except Exception:
        try:
            import lumopt2 as module
            return module, "lumopt2"
        except Exception:
            return None, None


def _run_lumopt2(module):
    """Official v261+ lumopt2 path; Parametrization returns object::property."""
    _require_optimization_source()
    required = (
        "Box", "Parametrization", "PortResults", "Fom", "PNorm", "Project",
        "FdtdSession", "LocalRunner", "ScipyOptimizer", "Optimization",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError("Installed lumopt2 is incomplete; missing: " + ", ".join(missing))

    def parameterization_function(parameters):
        geometry_value = _layer_builder_geometry_value(
            parameters, OPT_LAYER_BUILDER_ORIGIN_M
        )
        return {"Max Layout material stack::geometry": geometry_value}

    region = module.Box(
        x_min=float(OPTIMIZATION_VOLUME_UM[0]) * UM,
        x_max=float(OPTIMIZATION_VOLUME_UM[1]) * UM,
        y_min=float(OPTIMIZATION_VOLUME_UM[2]) * UM,
        y_max=float(OPTIMIZATION_VOLUME_UM[3]) * UM,
        z_min=float(OPTIMIZATION_VOLUME_UM[4]) * UM,
        z_max=float(OPTIMIZATION_VOLUME_UM[5]) * UM,
        mesh_size=float(OPTIMIZATION_SPEC["optimization_mesh_um"]) * UM,
    )
    parametrization = module.Parametrization(
        func=parameterization_function,
        bounds=parameter_bounds,
        initial_params=initial_params,
        optimization_region=region,
        use_jac=False,
    )
    wavelength_array = np.linspace(
        float(objective["wavelength_start_um"]) * UM,
        float(objective["wavelength_stop_um"]) * UM,
        int(objective["wavelength_points"]),
    )
    output = module.PortResults(
        str(OPT_OBJECTIVE_PORTS["monitor_port"]),
        metric="transmission",
        wavelengths=wavelength_array,
    )
    fom = module.Fom(
        output,
        fct=module.PNorm(p=2, target=float(objective["target"])),
    )
    project = module.Project(
        setup=REMOTE_OPTIMIZER_BASE_FSP,
        parametrization=parametrization,
        fom=fom,
        fdtd_session=module.FdtdSession(show_fdtd_cad=False),
        runner=module.LocalRunner(resource="GPU"),
    )
    scipy_optimizer = module.ScipyOptimizer(
        method="L-BFGS-B",
        bounds=parameter_bounds,
        max_iter=int(optimizer_settings["max_iterations"]),
        gtol=float(optimizer_settings.get("pgtol", 1e-5)),
    )
    optimization_kwargs = {
        "project": project,
        "optimizer": scipy_optimizer,
        "callbacks": [],
        "store_all_simulations": False,
    }
    if "initial_params" in inspect.signature(module.Optimization).parameters:
        optimization_kwargs["initial_params"] = initial_params
    lumopt2_optimization = module.Optimization(**optimization_kwargs)
    lumopt2_live_monitor = _start_shape_history_monitor(
        lumopt2_optimization.get_history,
        "shape adjoint (lumopt2)",
        best_alignment_params,
    )

    def lumopt2_iteration_callback(*_args, **_kwargs):
        # The official run callback is invoked at accepted iteration boundaries.
        # It only drains histories already owned by LumOpt2; it never evaluates
        # the FOM, updates geometry, or touches the FDTD session.
        lumopt2_live_monitor["poll"]()

    run_kwargs = {}
    if "callback" in inspect.signature(lumopt2_optimization.run).parameters:
        run_kwargs["callback"] = lumopt2_iteration_callback
    try:
        optimization_result = lumopt2_optimization.run(**run_kwargs)
    finally:
        _stop_shape_history_monitor(lumopt2_live_monitor)
    if isinstance(optimization_result, (tuple, list)) and len(optimization_result) >= 2:
        best_parameters, best_objective = optimization_result[:2]
        result_history = {}
    else:
        best_parameters = getattr(
            optimization_result,
            "optimal_params",
            getattr(
                lumopt2_optimization,
                "best_params",
                getattr(optimization_result, "best_params", None),
            ),
        )
        best_objective = getattr(
            optimization_result,
            "final_fom",
            getattr(
                optimization_result,
                "best_fom",
                getattr(lumopt2_optimization, "best_fom", float("nan")),
            ),
        )
        result_history = getattr(optimization_result, "history", {})
    if best_parameters is None:
        raise RuntimeError("lumopt2 returned no optimal parameter vector")
    best_parameters = np.asarray(best_parameters, dtype=float).ravel()
    if best_parameters.size != initial_params.size:
        raise RuntimeError("lumopt2 returned %d parameters; expected %d" % (
            best_parameters.size, initial_params.size,
        ))
    # lumopt2 needs a file handoff between its owner and the dedicated final
    # validation owner.  Keep that handoff private and transient; the public
    # best FSP is written only after validation when cell 1 requests it.
    project.save_project(REMOTE_VALIDATION_FSP, params=best_parameters)
    try:
        raw_history = lumopt2_optimization.get_history()
    except Exception:
        raw_history = result_history
    fom_history, parameter_history = _history_arrays(raw_history, initial_params.size)

    # Close lumopt2's owner before opening the one validation owner below.
    for candidate in (
        project,
        getattr(project, "fdtd_session", None),
        getattr(project, "session", None),
    ):
        close_method = getattr(candidate, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass
    import gc
    del lumopt2_optimization
    gc.collect()
    validation_owner = lumapi.FDTD(
        hide=True,
        serverArgs={"threads": str(BUILD_CPU_THREADS)},
    )
    validation_owner.load(REMOTE_VALIDATION_FSP)
    _activate_cpu_model_work(validation_owner)
    return (
        validation_owner,
        best_parameters,
        float(best_objective),
        fom_history,
        parameter_history,
        "lumopt2",
    )


def _run_legacy_lumopt():
    """Verified bundled legacy API fallback; still a genuine shape adjoint."""
    _require_optimization_source()
    from lumopt.figures_of_merit.PortTransmission import PortTransmission
    from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
    from lumopt.optimization import Optimization
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers
    from lumopt.utilities.wavelengths import Wavelengths

    shape_geometry = ParameterizedGeometry(
        func=update_max_layout_geometry,
        initial_params=initial_params,
        bounds=parameter_bounds,
        dx=float(OPTIMIZATION_SPEC["parameterized_geometry"]["finite_difference_step"]),
        threads_per_job=BUILD_CPU_THREADS,
        num_jobs=1,
        deps_num_threads=BUILD_CPU_THREADS,
    )
    wavelengths = Wavelengths(
        start=float(objective["wavelength_start_um"]) * UM,
        stop=float(objective["wavelength_stop_um"]) * UM,
        points=int(objective["wavelength_points"]),
    )
    target = float(objective["target"])
    port_objective = PortTransmission(
        monitor_port=str(OPT_OBJECTIVE_PORTS["monitor_port"]),
        mode_number=int(OPT_OBJECTIVE_PORTS.get("mode_number", 1)),
        direction=str(OPT_OBJECTIVE_PORTS["direction"]),
        target_T_fwd=lambda wavelength: np.full(
            np.asarray(wavelength).size, target, dtype=float
        ),
        norm_p=1,
        target_fom=target,
        source_port=str(OPT_OBJECTIVE_PORTS["source_port"]),
    )
    legacy_optimizer = ScipyOptimizers(
        max_iter=int(optimizer_settings["max_iterations"]),
        method="L-BFGS-B",
        scaling_factor=None,
        pgtol=float(optimizer_settings.get("pgtol", 1e-5)),
        ftol=float(optimizer_settings.get("ftol", 1e-5)),
        scale_initial_gradient_to=0.0,
    )
    legacy_live_state = {"emitted": 0}
    legacy_original_callback = legacy_optimizer.callback

    def legacy_reporting_callback(*args, **kwargs):
        # Preserve LumOpt's callback exactly once, then publish the histories it
        # just recorded. Reporting never calls callable_fom/callable_jac.
        result = legacy_original_callback(*args, **kwargs)
        _emit_new_shape_history(
            lambda: {
                "fom": list(getattr(legacy_optimizer, "fom_hist", [])),
                "params": list(getattr(legacy_optimizer, "params_hist", [])),
            },
            "shape adjoint (legacy LumOpt)",
            legacy_live_state,
            best_alignment_params,
        )
        return result

    legacy_optimizer.callback = legacy_reporting_callback
    legacy_optimization = Optimization(
        base_script=REMOTE_OPTIMIZER_BASE_FSP,
        wavelengths=wavelengths,
        fom=port_objective,
        geometry=shape_geometry,
        optimizer=legacy_optimizer,
        use_var_fdtd=False,
        hide_fdtd_cad=True,
        use_deps=True,
        plot_history=False,
        store_all_simulations=False,
        save_global_index=False,
        label="Max Layout 3D shape adjoint",
    )
    run_signature = inspect.signature(legacy_optimization.run).parameters
    if "working_dir" in run_signature:
        optimization_result = legacy_optimization.run(working_dir=REMOTE_OPTIMIZATION_DIR)
    else:
        previous_directory = os.getcwd()
        try:
            os.chdir(REMOTE_OPTIMIZATION_DIR)
            optimization_result = legacy_optimization.run()
        finally:
            os.chdir(previous_directory)
    _emit_new_shape_history(
        lambda: {
            "fom": list(getattr(legacy_optimizer, "fom_hist", [])),
            "params": list(getattr(legacy_optimizer, "params_hist", [])),
        },
        "shape adjoint (legacy LumOpt)",
        legacy_live_state,
        best_alignment_params,
    )
    owner = legacy_optimization.sim.fdtd
    fom_history = np.asarray(getattr(legacy_optimizer, "fom_hist", []), dtype=float).ravel()
    parameter_history = []
    for entry in list(getattr(legacy_optimizer, "params_hist", [])):
        array = np.asarray(entry, dtype=float).ravel()
        if array.size == initial_params.size:
            parameter_history.append(array)
    parameter_history = np.asarray(parameter_history, dtype=float)
    if parameter_history.ndim != 2 or parameter_history.shape[1:] != (initial_params.size,):
        parameter_history = np.empty((0, initial_params.size), dtype=float)
    if fom_history.size and parameter_history.shape[0] == fom_history.size:
        best_index = int(np.nanargmax(fom_history))
        best_parameters = parameter_history[best_index].copy()
        best_objective = float(fom_history[best_index])
    else:
        candidates = [
            getattr(shape_geometry, "current_params", None),
            getattr(legacy_optimizer, "current_params", None),
        ]
        if isinstance(optimization_result, (tuple, list)):
            candidates.extend(list(optimization_result))
        else:
            candidates.append(optimization_result)
        best_parameters = None
        for candidate in candidates:
            try:
                candidate_array = np.asarray(candidate, dtype=float).ravel()
            except Exception:
                continue
            if candidate_array.size == initial_params.size:
                best_parameters = candidate_array
                break
        if best_parameters is None:
            best_parameters = initial_params.copy()
        best_objective = float(np.nanmax(fom_history)) if fom_history.size else float("nan")
    return (
        owner,
        best_parameters,
        best_objective,
        fom_history,
        parameter_history,
        "legacy lumopt",
    )

ADJOINT_FDTD_OWNER = None
try:
    _synchronize_resolved_fiber_mode_contract()
    REMOTE_OPTIMIZER_BASE_FSP = REMOTE_BASE_FSP
    best_alignment_params = alignment_initial_params.copy()
    best_alignment_fom = float("nan")
    if alignment_parameter_rows:
        REMOTE_ALIGNED_BASE_FSP = os.path.join(
            REMOTE_OPTIMIZATION_DIR, "aligned_adjoint_base.fsp"
        )
        best_alignment_params, best_alignment_fom = _optimize_fiber_alignment(
            REMOTE_BASE_FSP,
            initial_params,
            REMOTE_ALIGNED_BASE_FSP,
            optimizer_settings.get("alignment_max_iterations", 20),
        )
        REMOTE_OPTIMIZER_BASE_FSP = REMOTE_ALIGNED_BASE_FSP
        print(
            "Fiber alignment frozen for adjoint stage:",
            {
                str(row["parameter"]): float(best_alignment_params[index])
                for index, row in enumerate(alignment_parameter_rows)
            },
        )

    if parameter_names:
        lumopt2_module, lumopt2_import = _import_lumopt2()
        if lumopt2_module is not None:
            print("Using official", lumopt2_import, "shape-adjoint API.")
            (
                ADJOINT_FDTD_OWNER, best_params, best_fom,
                fom_history, parameter_history, ADJOINT_ENGINE,
            ) = _run_lumopt2(lumopt2_module)
        else:
            print("Official lumopt2 is unavailable; using verified bundled legacy LumOpt shape adjoint.")
            (
                ADJOINT_FDTD_OWNER, best_params, best_fom,
                fom_history, parameter_history, ADJOINT_ENGINE,
            ) = _run_legacy_lumopt()
    else:
        best_params = np.empty(0, dtype=float)
        best_fom = best_alignment_fom
        fom_history = np.empty(0, dtype=float)
        parameter_history = np.empty((0, 0), dtype=float)
        ADJOINT_ENGINE = "GPU forward-solve fiber alignment"
        ADJOINT_FDTD_OWNER = lumapi.FDTD(
            hide=True,
            serverArgs={"threads": str(BUILD_CPU_THREADS)},
        )
        ADJOINT_FDTD_OWNER.load(REMOTE_OPTIMIZER_BASE_FSP)
        _activate_cpu_model_work(ADJOINT_FDTD_OWNER)

    # Preserve the best geometry and run one final forward GPU validation.  It
    # produces the actual best-design broadband upper/lower MMI ratios (or
    # grating CE) rather than treating the optimizer's scalar history as a
    # substitute for a spectral response.  No per-iteration FSP is retained.
    _activate_cpu_model_work(ADJOINT_FDTD_OWNER)
    ADJOINT_FDTD_OWNER.switchtolayout()
    update_max_layout_geometry(best_params, ADJOINT_FDTD_OWNER, True)
    _apply_fiber_pose_to_session(
        best_alignment_params,
        ADJOINT_FDTD_OWNER,
        best_params,
        update_modes=True,
    )
    _configure_optimization_source(ADJOINT_FDTD_OWNER)
    _activate_gpu_solve(ADJOINT_FDTD_OWNER)
    ADJOINT_FDTD_OWNER.run("FDTD", "GPU")

    def _port_transmission(port_name):
        dataset = ADJOINT_FDTD_OWNER.getresult("FDTD::ports::" + str(port_name), "T")
        if "lambda" in dataset:
            wavelength_m = np.asarray(dataset["lambda"], dtype=float).squeeze().ravel()
        elif "f" in dataset:
            wavelength_m = 299792458.0 / np.asarray(dataset["f"], dtype=float).squeeze().ravel()
        else:
            raise RuntimeError("Port %s T result has neither lambda nor frequency" % port_name)
        transmission = np.abs(np.asarray(dataset["T"]).squeeze())
        if transmission.ndim == 0:
            transmission = np.full(wavelength_m.size, transmission)
        elif transmission.ndim > 1:
            wavelength_axes = [
                axis for axis, size in enumerate(transmission.shape)
                if size == wavelength_m.size
            ]
            if not wavelength_axes:
                raise RuntimeError("Could not align port %s T with wavelength" % port_name)
            transmission = np.moveaxis(transmission, wavelength_axes[0], 0).reshape(wavelength_m.size, -1)[:, 0]
        transmission = np.asarray(transmission, dtype=float).ravel()
        order = np.argsort(wavelength_m)
        return wavelength_m[order], transmission[order]

    validation_arrays = {}
    validation_summary = {}
    if OPTIMIZATION_SPEC["component_kind"] == "1x2 MMI":
        validation_wavelength_m, primary_ratio = _port_transmission(
            OPT_OBJECTIVE_PORTS["monitor_port"]
        )
        validation_arrays["validation_wavelength_m"] = validation_wavelength_m
        lower_wavelength_m, lower_ratio = _port_transmission(
            OPT_OBJECTIVE_PORTS["lower_output_port"]
        )
        if lower_wavelength_m.size != validation_wavelength_m.size or not np.allclose(
            lower_wavelength_m, validation_wavelength_m, rtol=1e-9, atol=1e-15
        ):
            lower_ratio = np.interp(validation_wavelength_m, lower_wavelength_m, lower_ratio)
        total_ratio = primary_ratio + lower_ratio
        imbalance = np.abs(primary_ratio - lower_ratio) / np.maximum(total_ratio, 1e-15)
        validation_arrays.update({
            "mmi_top_output_over_input": primary_ratio,
            "mmi_lower_output_over_input": lower_ratio,
            "mmi_total_output_over_input": total_ratio,
            "mmi_upper_lower_imbalance": imbalance,
        })
        validation_summary = {
            "definition": "Top/upper output branch power divided by input power is the optimized objective; lower, total, and imbalance are validation only.",
            "wavelength_m": validation_wavelength_m.tolist(),
            "top_output_over_input": primary_ratio.tolist(),
            "lower_output_over_input": lower_ratio.tolist(),
            "total_output_over_input": total_ratio.tolist(),
            "upper_lower_imbalance": imbalance.tolist(),
        }
    else:
        def _power_spectrum(dataset, value_key):
            if "lambda" in dataset:
                wavelength_m = np.asarray(dataset["lambda"], dtype=float).squeeze().ravel()
            elif "f" in dataset:
                wavelength_m = 299792458.0 / np.asarray(
                    dataset["f"], dtype=float
                ).squeeze().ravel()
            else:
                raise RuntimeError("Power result has neither lambda nor frequency")
            normalized = {
                str(key).lower().replace("_", "").replace(" ", ""): key
                for key in dataset.keys()
            }
            resolved_key = normalized.get(
                str(value_key).lower().replace("_", "").replace(" ", ""),
                value_key,
            )
            values = np.real(np.asarray(dataset[resolved_key]).squeeze())
            if values.ndim == 0:
                values = np.full(wavelength_m.size, values)
            elif values.ndim > 1:
                axes = [
                    axis for axis, size in enumerate(values.shape)
                    if size == wavelength_m.size
                ]
                if not axes:
                    raise RuntimeError(
                        "Could not align %s shape %s with wavelength"
                        % (resolved_key, values.shape)
                    )
                values = np.moveaxis(values, axes[0], 0).reshape(
                    wavelength_m.size, -1
                )[:, 0]
            order = np.argsort(wavelength_m)
            return wavelength_m[order], np.asarray(values, dtype=float).ravel()[order]

        input_dataset = ADJOINT_FDTD_OWNER.getresult(
            str(OPT_OBJECTIVE_PORTS["fiber_input_power_monitor"]), "T"
        )
        input_wavelength_m, input_signed = _power_spectrum(input_dataset, "T")
        input_power = (
            float(OPT_OBJECTIVE_PORTS["fiber_input_power_sign"]) * input_signed
        )
        modal_dataset = ADJOINT_FDTD_OWNER.getresult(
            "FDTD::ports::" + str(OPT_OBJECTIVE_PORTS["monitor_port"]),
            str(OPT_OBJECTIVE_PORTS["waveguide_port_expansion_result_name"]),
        )
        validation_wavelength_m, modal_power = _power_spectrum(
            modal_dataset,
            str(OPT_OBJECTIVE_PORTS["waveguide_port_modal_direction"]),
        )
        modal_power = float(
            OPT_OBJECTIVE_PORTS.get("waveguide_port_modal_sign", 1.0)
        ) * modal_power
        input_power = np.interp(
            validation_wavelength_m, input_wavelength_m, input_power
        )
        total_dataset = ADJOINT_FDTD_OWNER.getresult(
            str(OPT_OBJECTIVE_PORTS["waveguide_total_power_monitor"]), "T"
        )
        total_wavelength_m, total_signed = _power_spectrum(total_dataset, "T")
        total_power = float(
            OPT_OBJECTIVE_PORTS["waveguide_total_power_sign"]
        ) * np.interp(validation_wavelength_m, total_wavelength_m, total_signed)
        if np.any(input_power <= 1e-15):
            raise RuntimeError(
                "Best-design incident input monitor has wrong/near-zero signed flux"
            )
        primary_ratio = modal_power / input_power
        total_ratio = total_power / input_power
        validation_arrays.update(
            {
                "validation_wavelength_m": validation_wavelength_m,
                "grating_fiber_input_power": input_power,
                "grating_waveguide_te_power": modal_power,
                "grating_waveguide_total_power": total_power,
                "grating_waveguide_te_over_input": primary_ratio,
                "grating_waveguide_total_over_input": total_ratio,
            }
        )
        validation_summary = {
            "definition": "Selected-TE waveguide receiver and nearby total-waveguide power, each divided by measured incident input-monitor power.",
            "wavelength_m": validation_wavelength_m.tolist(),
            "fiber_input_power": input_power.tolist(),
            "waveguide_te_over_input": primary_ratio.tolist(),
            "waveguide_total_over_input": total_ratio.tolist(),
        }

    # Forward/adjoint and best-design validation GPU work is complete.  Use
    # CPU for serialization and plotting while retaining the solved project.
    cpu_threads = BUILD_CPU_THREADS
    try:
        ADJOINT_FDTD_OWNER.setresource("FDTD", 1, "active", False)
        ADJOINT_FDTD_OWNER.setresource("FDTD", 2, "device type", "CPU")
        ADJOINT_FDTD_OWNER.setresource("FDTD", 2, "active", True)
        ADJOINT_FDTD_OWNER.setresource("FDTD", 2, "processes", 1)
        ADJOINT_FDTD_OWNER.setresource("FDTD", 2, "threads", cpu_threads)
        print("Adjoint GPU solves complete; post-processing uses CPU: 1 x %d threads." % cpu_threads)
    except Exception as exc:
        print("CPU post-processing resource warning:", str(exc)[:240])

    best_project_name = str(OPTIMIZATION_SPEC["best_project_file"])
    REMOTE_BEST_FSP = os.path.join(REMOTE_WORK, "fsp", best_project_name)
    os.makedirs(os.path.dirname(REMOTE_BEST_FSP), exist_ok=True)
    ADJOINT_FDTD_OWNER.save(REMOTE_BEST_FSP)
    if not os.path.isfile(REMOTE_BEST_FSP) or os.path.getsize(REMOTE_BEST_FSP) <= 0:
        raise RuntimeError("LumOpt did not create the required best-geometry FSP: " + REMOTE_BEST_FSP)
    print("Saved required best-design FSP:", REMOTE_BEST_FSP)

    best_parameters = {
        name: float(best_params[index]) for index, name in enumerate(parameter_names)
    }
    alignment_best_parameters = {
        str(row["parameter"]): float(best_alignment_params[index])
        for index, row in enumerate(alignment_parameter_rows)
    }
    best_parameters.update(alignment_best_parameters)
    patch_parameters = dict(best_parameters)
    derived_parameter_names = []
    component_kind = str(OPTIMIZATION_SPEC["component_kind"])

    def _best_or_nominal(name, default=0.0):
        return float(best_parameters.get(name, OPT_COMPONENT_NOMINAL_PARAMS.get(name, default)))

    if component_kind == "1x2 MMI":
        nominal_output_length = float(OPT_COMPONENT_NOMINAL_PARAMS.get("output_length", 0.0))
        total_longitudinal_change = sum(
            _best_or_nominal(name) - float(OPT_COMPONENT_NOMINAL_PARAMS.get(name, 0.0))
            for name in (
                "input_length", "input_taper_length", "mmi_length", "output_taper_length"
            )
            if name in best_parameters
        )
        patch_parameters["output_length"] = nominal_output_length - total_longitudinal_change
        derived_parameter_names.append("output_length")
    elif component_kind == "GC-SOI":
        fixed_count = int(OPTIMIZATION_SPEC["fixed_period_count"])
        best_pitch = _best_or_nominal("pitch")
        patch_parameters["target_length"] = (float(fixed_count) - 0.5) * best_pitch
        derived_parameter_names.append("target_length")
        nominal_flare = (
            float(OPT_COMPONENT_NOMINAL_PARAMS.get("wg_length", 0.0))
            + float(OPT_COMPONENT_NOMINAL_PARAMS.get("radius", 0.0))
        )
        nominal_fiber_center = nominal_flare + float(
            OPT_COMPONENT_NOMINAL_PARAMS.get("fiber_offset", 0.0)
        )
        best_flare = _best_or_nominal("wg_length") + _best_or_nominal("radius")
        if "fiber_offset" not in best_parameters:
            patch_parameters["fiber_offset"] = nominal_fiber_center - best_flare
            derived_parameter_names.append("fiber_offset")
    elif component_kind == "Grating coupler":
        def _standard_flare(values):
            width = float(values.get("wg_width", 0.0))
            alpha = float(values.get("alpha_t", 25.0))
            focus_offset = 0.5 * width / np.tan(0.5 * np.deg2rad(alpha))
            return float(values.get("wg_length", 0.0)) - focus_offset + float(values.get("taper_L", 0.0))

        nominal_flare = _standard_flare(OPT_COMPONENT_NOMINAL_PARAMS)
        nominal_fiber_center = nominal_flare + float(
            OPT_COMPONENT_NOMINAL_PARAMS.get("fiber_offset", 0.0)
        )
        best_flare_values = dict(OPT_COMPONENT_NOMINAL_PARAMS)
        best_flare_values.update(best_parameters)
        if "fiber_offset" not in best_parameters:
            patch_parameters["fiber_offset"] = nominal_fiber_center - _standard_flare(best_flare_values)
            derived_parameter_names.append("fiber_offset")

    complete_best_parameters = dict(OPT_COMPONENT_NOMINAL_PARAMS)
    complete_best_parameters.update(patch_parameters)

    parameter_patch = {
        "component_uid": int(OPTIMIZATION_SPEC["component_uid"]),
        "component_kind": component_kind,
        "params": patch_parameters,
        "optimized_parameters": list(parameter_names),
        "derived_reproducibility_parameters": derived_parameter_names,
    }
    REMOTE_OPT_TEXT_SUMMARY = os.path.join(REMOTE_WORK, "summary.txt")
    summary = {
        "engine": ADJOINT_ENGINE,
        "method": str(OPTIMIZATION_SPEC.get("method", "3D shape adjoint")),
        "objective": OPT_OBJECTIVE_PORTS,
        "spectral_objective": objective,
        "best_fom": best_fom if np.isfinite(best_fom) else None,
        "best_parameters": best_parameters,
        "complete_best_parameters": complete_best_parameters,
        "best_alignment_parameters": alignment_best_parameters,
        "best_alignment_fom": best_alignment_fom if np.isfinite(best_alignment_fom) else None,
        "editor_parameter_patch": patch_parameters,
        "iterations_recorded": int(fom_history.size),
        "gpu_forward_and_adjoint_solves": True,
        "gpu_best_design_validation_solve": True,
        "best_design_validation": validation_summary,
        "cpu_postprocessing": True,
        "store_all_simulations": False,
        "best_fsp": REMOTE_BEST_FSP,
        "live_iteration_progress": REMOTE_OPT_PROGRESS_FILE,
        "text_summary": REMOTE_OPT_TEXT_SUMMARY,
    }
    REMOTE_OPT_HISTORY = os.path.join(REMOTE_WORK, "adjoint_optimization_history.npz")
    REMOTE_OPT_SUMMARY = os.path.join(REMOTE_WORK, "adjoint_optimization_summary.json")
    REMOTE_PARAMETER_PATCH = os.path.join(REMOTE_WORK, "adjoint_parameter_patch.json")
    REMOTE_OPT_PLOT = os.path.join(REMOTE_WORK, "adjoint_optimization_history.png")
    np.savez_compressed(
        REMOTE_OPT_HISTORY,
        fom=fom_history,
        parameters=parameter_history,
        parameter_names=np.asarray(parameter_names, dtype="U"),
        best_parameters=np.asarray(best_params, dtype=float),
        alignment_fom=np.asarray(ALIGNMENT_EVALUATION_FOM, dtype=float),
        alignment_parameters=np.asarray(ALIGNMENT_EVALUATION_PARAMETERS, dtype=float),
        alignment_iteration_fom=np.asarray(ALIGNMENT_ITERATION_FOM, dtype=float),
        alignment_iteration_parameters=np.asarray(
            ALIGNMENT_ITERATION_PARAMETERS, dtype=float
        ),
        alignment_parameter_names=np.asarray(
            [str(row["parameter"]) for row in alignment_parameter_rows], dtype="U"
        ),
        best_alignment_parameters=np.asarray(best_alignment_params, dtype=float),
        **validation_arrays,
    )
    with open(REMOTE_OPT_SUMMARY, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    with open(REMOTE_PARAMETER_PATCH, "w", encoding="utf-8") as stream:
        json.dump(parameter_patch, stream, indent=2, sort_keys=True)

    def _text_number(value, digits=8):
        try:
            number = float(np.asarray(value).ravel()[0])
        except Exception:
            return str(value)
        return ("%.*g" % (int(digits), number)) if np.isfinite(number) else str(number)

    def _text_json(value):
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

    def _text_section(lines, title):
        if lines:
            lines.append("")
        lines.append(str(title))
        lines.append("-" * len(str(title)))

    _text_parameter_details = {
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
        "fdtd_port_offset_from_waveguide_end_um": ("Waveguide FDTD-port offset from waveguide end", "um"),
        "waveguide_monitor_span_um": ("Waveguide receiver-port transverse span", "um"),
        "waveguide_total_power_before_mode_um": ("Total-power plane distance before receiver port", "um"),
        "waveguide_neff_tolerance": ("Waveguide effective-index tolerance", ""),
        "waveguide_mode_search_count": ("Waveguide eigensolver modes searched", ""),
        "tolerance": ("Geometry build tolerance", "um"),
    }

    def _text_parameter_value(value):
        if isinstance(value, (dict, list, tuple)):
            return _text_json(value)
        if isinstance(value, str):
            return value
        return _text_number(value)

    def _append_text_major_parameters(lines, parameters, prefix="- "):
        parameters = dict(parameters or {})
        found = 0
        shown = set()
        for key, (label, unit) in _text_parameter_details.items():
            if key not in parameters or parameters[key] in (None, ""):
                continue
            suffix = (" " + unit) if unit else ""
            lines.append("%s%s: %s%s" % (prefix, label, _text_parameter_value(parameters[key]), suffix))
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
                    % (prefix, str(key).replace("_", " ").title(), _text_parameter_value(value))
                )
                found += 1

    objective_contract = dict(OPTIMIZATION_SPEC.get("objective", {}))
    center_wavelength_m = float(objective_contract.get("center_wavelength_um", 0.0)) * UM
    center_index = int(np.argmin(np.abs(validation_wavelength_m - center_wavelength_m)))
    validation_peak_index = int(np.nanargmax(primary_ratio))
    text_lines = ["MAX LAYOUT — LUMERICAL ADJOINT OPTIMIZATION SUMMARY"]
    _text_section(text_lines, "PROJECT")
    text_lines.extend([
        "Component: name=%s | kind=%s | UID=%d" % (
            str(OPT_COMPONENT_NOMINAL_PARAMS.get("name", OPTIMIZATION_SPEC["component_kind"])),
            str(OPTIMIZATION_SPEC["component_kind"]),
            int(OPTIMIZATION_SPEC["component_uid"]),
        ),
        "Run status: optimization and best-design validation completed",
    ])

    _text_section(text_lines, "PARAMETERS")
    text_lines.append("Nominal major device parameters (lengths in um; angles in deg):")
    _append_text_major_parameters(text_lines, OPT_COMPONENT_NOMINAL_PARAMS)
    text_lines.append("Exact nominal source parameters (JSON): %s" % _text_json(OPT_COMPONENT_NOMINAL_PARAMS))
    text_lines.append("Complete best-design geometry (optimized, derived, and unchanged nominal parameters):")
    _append_text_major_parameters(text_lines, complete_best_parameters)
    text_lines.append("Complete best-design source parameters (JSON): %s" % _text_json(complete_best_parameters))
    text_lines.append("Parameters changed directly by the optimizer (JSON): %s" % _text_json(best_parameters))
    text_lines.append("Complete editor parameter patch, including derived reproducibility values: %s" % _text_json(patch_parameters))

    _text_section(text_lines, "OBJECTIVE AND BOUNDS")
    text_lines.extend([
        "Objective: %s" % str(OPT_OBJECTIVE_PORTS.get("kind", "linear objective")),
        "Description: %s" % str(objective_contract.get("description", OPT_OBJECTIVE_PORTS.get("description", ""))),
        "Linear-power target: %s" % _text_number(objective_contract.get("target", 1.0)),
        "Spectral contract: center=%s um | bandwidth=%s nm | wavelength range=%s to %s um | %s points"
        % (
            _text_number(objective_contract.get("center_wavelength_um")),
            _text_number(objective_contract.get("bandwidth_nm")),
            _text_number(objective_contract.get("wavelength_start_um")),
            _text_number(objective_contract.get("wavelength_stop_um")),
            str(objective_contract.get("wavelength_points", "")),
        ),
        "Optimization parameter bounds:",
    ])
    for row in all_parameter_rows:
        parameter = str(row["parameter"])
        label, unit = _text_parameter_details.get(
            parameter, (parameter.replace("_", " ").title(), "")
        )
        suffix = (" " + unit) if unit else ""
        text_lines.append(
            "- %s: minimum=%s%s | initial=%s%s | maximum=%s%s"
            % (
                label, _text_number(row["minimum"]), suffix,
                _text_number(row["initial"]), suffix,
                _text_number(row["maximum"]), suffix,
            )
        )

    _text_section(text_lines, "MATERIAL STACK AND MESH")
    text_lines.append("Bottom-to-top layer order; mesh factor means factor x lambda0 / maximum material index.")
    for index, row in enumerate(MATERIAL_STACK, start=1):
        text_lines.append(
            "- %02d %s | material=%s | thickness=%s um | etch=%s um | sidewall=%s deg | mesh_factor=%s | mesh_order=%s | role=%s | conformal=%s | slab_extent=%s | GDS_layers=%s"
            % (
                index, str(row.get("name", "layer")), str(row.get("material", "")),
                _text_number(row.get("thickness_um", 0.0)),
                _text_number(row.get("etch_depth_um", 0.0)),
                _text_number(row.get("sidewall_angle_deg", 90.0)),
                _text_number(row.get("mesh_factor", 0.2)),
                str(row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2)),
                str(row.get("role", "background")), str(bool(row.get("conformal", False))),
                str(row.get("slab_extent", "full FDTD plane")),
                _text_json(row.get("gds_layers", [])),
            )
        )
    domain_um = []
    for property_name in ("x min", "x max", "y min", "y max", "z min", "z max"):
        try:
            domain_um.append(float(np.asarray(ADJOINT_FDTD_OWNER.getnamed("FDTD", property_name)).ravel()[0]) / UM)
        except Exception:
            domain_um.append(None)
    _text_section(text_lines, "SIMULATION SETTINGS")
    text_lines.extend([
        "- Solver: 3D FDTD",
        "- Domain [xmin,xmax,ymin,ymax,zmin,zmax]: %s um" % _text_json(domain_um),
        "- Wavelength sweep: %s to %s um | %s points"
        % (
            _text_number(objective_contract.get("wavelength_start_um")),
            _text_number(objective_contract.get("wavelength_stop_um")),
            str(objective_contract.get("wavelength_points", "")),
        ),
        "- Resources: forward/adjoint solve=GPU | best-design validation=GPU | model-build CPU threads=%s | post-processing=CPU"
        % str(BUILD_CPU_THREADS),
        "- Numerical controls: mesh accuracy=%s | dt factor=%s | PML=%s | geometry/PML overlap=%s um"
        % (
            str(SETTINGS.get("mesh_accuracy", 2)),
            _text_number(SETTINGS.get("dt_stability_factor", 0.99)),
            str(SETTINGS.get("pml_profile", "Standard")),
            _text_number(SETTINGS.get("pml_geometry_overlap_um", 1.0)),
        ),
        "- Time controls: maximum=%s ps (%s fs) | auto shutoff=%s"
        % (
            _text_number(float(SETTINGS.get("simulation_time_fs", 10000.0)) / 1000.0),
            _text_number(SETTINGS.get("simulation_time_fs", 10000.0)),
            _text_number(SETTINGS.get("auto_shutoff_min", 1e-6)),
        ),
        "- TFLN material model: crystal cut=%s | temperature=%s K"
        % (
            str(SETTINGS.get("tfln_crystal_cut", "X")),
            _text_number(SETTINGS.get("tfln_temperature_K", 296.3)),
        ),
    ])
    waveguide_index_estimate = dict(
        globals().get("WAVEGUIDE_INDEX_ESTIMATE", {})
    )
    if waveguide_index_estimate:
        text_lines.append(
            "- Automatic waveguide mode target: core n=%s | adjacent dielectric n=%s | "
            "midpoint neff=%s at %s um | core=%s | surroundings=%s"
            % (
                _text_number(waveguide_index_estimate.get("core_index")),
                _text_number(waveguide_index_estimate.get("surrounding_index")),
                _text_number(waveguide_index_estimate.get("target_neff")),
                _text_number(waveguide_index_estimate.get("wavelength_um")),
                _text_json(waveguide_index_estimate.get("core_materials", [])),
                _text_json(waveguide_index_estimate.get("surrounding_materials", [])),
            )
        )

    _text_section(text_lines, "SOURCES / PORTS / MONITORS")
    text_lines.extend([
        "Source and objective mapping: %s" % _text_json(OPT_OBJECTIVE_PORTS),
        "Fiber geometries: %s" % _text_json([
            {"name": item.get("name"), "center_um": item.get("center"),
             "theta_deg": item.get("angle theta", 0.0), "phi_deg": item.get("angle phi", 0.0),
             "core_diameter_um": item.get("core diameter_um"), "core_index": item.get("core index"),
             "cladding_diameter_um": item.get("cladding diameter_um"), "cladding_index": item.get("cladding index"),
             "length_um": item.get("length_um")}
            for item in FIBER_GEOMETRIES
        ]),
        "FDTD ports: %s" % _text_json([
            {"name": item.get("name"), "normal": item.get("plane normal"),
             "center_um": item.get("center"),
             "spans_um": [item.get("x span"), item.get("y span"), item.get("z span", item.get("z_span_um"))],
             "theta_deg": item.get("angle theta", 0.0), "phi_deg": item.get("angle phi", 0.0),
             "mode": item.get("mode"), "mode_number": item.get("mode number"),
             "polarization": item.get("polarization"), "target_neff": item.get("target neff")}
            for item in PORTS
        ]),
        "Monitors: %s" % _text_json([
            {"name": item.get("name"), "kind": item.get("monitor_kind"),
             "normal": item.get("plane normal"), "center_um": item.get("center"),
             "spans_um": [item.get("x span"), item.get("y span"), item.get("z span")],
             "role": item.get("grating_monitor_role", item.get("parent_port_name")),
             "target_neff": item.get("target neff")}
            for item in MONITORS
        ]),
    ])

    optimizer_contract = dict(OPTIMIZATION_SPEC.get("optimizer", {}))
    parameterized_geometry = dict(OPTIMIZATION_SPEC.get("parameterized_geometry", {}))
    _text_section(text_lines, "OPTIMIZATION SETTINGS")
    text_lines.extend([
        "- Engine/method: %s | %s" % (str(ADJOINT_ENGINE), str(OPTIMIZATION_SPEC.get("method", "3D shape adjoint"))),
        "- Shape optimizer: %s | maximum iterations=%s | pgtol=%s | ftol=%s"
        % (
            str(optimizer_contract.get("algorithm", "L-BFGS-B")),
            str(optimizer_contract.get("max_iterations", "")),
            _text_number(optimizer_contract.get("pgtol", 1e-5)),
            _text_number(optimizer_contract.get("ftol", 1e-5)),
        ),
        "- Alignment optimizer: %s | maximum iterations=%s | evaluations recorded=%d"
        % (
            str(optimizer_contract.get("alignment_algorithm", "not used")),
            str(optimizer_contract.get("alignment_max_iterations", 0)),
            int(np.asarray(ALIGNMENT_EVALUATION_FOM).size),
        ),
        "- Geometry: fixed topology=%s | fixed grating periods=%s | class=%s | finite-difference step=%s"
        % (
            str(bool(OPTIMIZATION_SPEC.get("fixed_topology", True))),
            str(OPTIMIZATION_SPEC.get("fixed_period_count", "n/a")),
            str(parameterized_geometry.get("class", "Parametrization")),
            _text_number(parameterized_geometry.get("finite_difference_step", 1e-3)),
        ),
        "- Optimization mesh: %s um | field monitor=%s | retain every simulation=%s"
        % (
            _text_number(OPTIMIZATION_SPEC.get("optimization_mesh_um", 0.05)),
            str(OPTIMIZATION_SPEC.get("opt_fields_monitor", "opt_fields")),
            str(bool(OPTIMIZATION_SPEC.get("store_all_simulations", False))),
        ),
    ])

    _text_section(text_lines, "RESULTS SUMMARY")
    text_lines.extend([
        "Best optimizer FOM: %s | shape iterations recorded=%d" % (
            _text_number(best_fom), int(fom_history.size)
        ),
        "Best fiber-alignment FOM: %s | best alignment parameters=%s"
        % (_text_number(best_alignment_fom), _text_json(alignment_best_parameters)),
        "Best-design forward validation (linear power): peak=%s (%s%%) at %s nm | center %s nm=%s (%s%%)"
        % (
            _text_number(primary_ratio[validation_peak_index]),
            _text_number(100.0 * primary_ratio[validation_peak_index]),
            _text_number(validation_wavelength_m[validation_peak_index] * 1e9),
            _text_number(validation_wavelength_m[center_index] * 1e9),
            _text_number(primary_ratio[center_index]),
            _text_number(100.0 * primary_ratio[center_index]),
        ),
    ])
    if OPTIMIZATION_SPEC["component_kind"] == "1x2 MMI":
        text_lines.append(
            "MMI validation at center: upper/Pin=%s | lower/Pin=%s | total/Pin=%s | imbalance=%s percentage points"
            % (
                _text_number(primary_ratio[center_index]),
                _text_number(lower_ratio[center_index]),
                _text_number(total_ratio[center_index]),
                _text_number(100.0 * imbalance[center_index]),
            )
        )
    else:
        text_lines.append(
            "Grating validation at center: selected-TE/Pin=%s | total-waveguide/Pin=%s | measured Pin=%s"
            % (
                _text_number(primary_ratio[center_index]),
                _text_number(total_ratio[center_index]),
                _text_number(input_power[center_index]),
            )
        )

    _text_section(text_lines, "OUTPUT FILES")
    text_lines.extend([
        "- Best design FSP: %s" % REMOTE_BEST_FSP,
        "- Optimization history: %s" % REMOTE_OPT_HISTORY,
        "- Machine-readable summary: %s" % REMOTE_OPT_SUMMARY,
        "- Editor parameter patch: %s" % REMOTE_PARAMETER_PATCH,
        "- Live iteration stream: %s" % REMOTE_OPT_PROGRESS_FILE,
    ])

    _text_section(text_lines, "WARNINGS / NOTES")
    export_notes = [str(item) for item in globals().get("EXPORT_WARNINGS", [])]
    if export_notes:
        text_lines.extend("- %s" % note for note in export_notes)
    else:
        text_lines.append("- No export warnings were recorded.")

    _text_section(text_lines, "FSP PROVENANCE")
    text_lines.append(
        "store_all_simulations=false. No per-iteration FSP is retained. The final best-design FSP is always saved. The validated linear response above comes from the dedicated best-design forward GPU solve, not directly from the optimizer FOM history."
    )
    with open(REMOTE_OPT_TEXT_SUMMARY, "w", encoding="utf-8") as stream:
        stream.write("\n".join(text_lines).rstrip() + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if OPTIMIZATION_SPEC["component_kind"] == "1x2 MMI":
        figure, (axis, response_axis) = plt.subplots(2, 1, figsize=(9.0, 8.0))
        response_db_axis = None
    else:
        figure = plt.figure(figsize=(13.0, 8.0))
        grid = figure.add_gridspec(2, 2)
        axis = figure.add_subplot(grid[0, :])
        response_axis = figure.add_subplot(grid[1, 0])
        response_db_axis = figure.add_subplot(grid[1, 1], sharex=response_axis)
    if fom_history.size:
        axis.plot(
            np.arange(1, fom_history.size + 1), fom_history,
            lw=2.2, color="#2563eb", label="shape-adjoint FOM",
        )
    alignment_fom_array = np.asarray(ALIGNMENT_EVALUATION_FOM, dtype=float)
    if alignment_fom_array.size:
        axis.plot(
            np.arange(1, alignment_fom_array.size + 1), alignment_fom_array,
            lw=1.8, color="#059669", label="fiber-alignment CE evaluations",
        )
    if not fom_history.size and not alignment_fom_array.size:
        axis.text(0.5, 0.5, "No readable optimization history", ha="center", va="center")
    if fom_history.size or alignment_fom_array.size:
        axis.legend(loc="best")
    axis.set_xlabel("recorded optimization evaluation / iteration")
    axis.set_ylabel("linear objective")
    axis.set_title(str(OPT_OBJECTIVE_PORTS["kind"]))
    axis.grid(alpha=0.3)
    wavelength_nm = validation_wavelength_m * 1e9
    if OPTIMIZATION_SPEC["component_kind"] == "1x2 MMI":
        response_axis.plot(wavelength_nm, primary_ratio, lw=2.4, label="top/upper output / input (objective)")
        response_axis.plot(wavelength_nm, lower_ratio, lw=2.0, ls="--", label="lower output / input (validation)")
        response_axis.plot(wavelength_nm, total_ratio, lw=1.8, ls=":", label="total output / input (validation)")
        response_axis.set_ylabel("linear branch/input power")
    else:
        response_axis.plot(
            wavelength_nm, primary_ratio, lw=2.4, color="#7c3aed",
            label="selected TE / measured input",
        )
        response_axis.plot(
            wavelength_nm, total_ratio, lw=2.0, color="#f59e0b",
            label="total waveguide / measured input",
        )
        response_axis.set_ylabel("normalized linear power")
        response_db_axis.plot(
            wavelength_nm,
            10.0 * np.log10(np.maximum(primary_ratio, 1e-15)),
            lw=2.4,
            color="#7c3aed",
            label="selected TE / measured input",
        )
        response_db_axis.plot(
            wavelength_nm,
            10.0 * np.log10(np.maximum(total_ratio, 1e-15)),
            lw=2.0,
            color="#f59e0b",
            label="total waveguide / measured input",
        )
        response_db_axis.set_xlabel("wavelength [nm]")
        response_db_axis.set_ylabel("normalized power [dB]")
        response_db_axis.set_title("Best-design validation — dB")
        response_db_axis.grid(alpha=0.3)
        response_db_axis.legend(loc="best")
    response_axis.set_xlabel("wavelength [nm]")
    response_axis.set_title("Best-design forward GPU validation")
    response_axis.grid(alpha=0.3)
    response_axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(REMOTE_OPT_PLOT, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print("Best adjoint FSP:", REMOTE_BEST_FSP)
    print("Best parameters:", best_parameters)
    print("Best objective:", best_fom)
finally:
    if ADJOINT_FDTD_OWNER is not None:
        try:
            ADJOINT_FDTD_OWNER.close()
        except Exception:
            pass
    # The release cell still runs afterward to return all three roamed packs.
    fdtd = None
    # Remove only transient optimizer files after the owner closes.  The
    # inspection FSP is the persistent LumOpt seed and must remain available.
    for _internal_seed in {
        globals().get("REMOTE_ALIGNED_BASE_FSP", ""),
        globals().get("REMOTE_VALIDATION_FSP", ""),
        globals().get("REMOTE_RUNTIME_PROJECT_FILE", ""),
    }:
        if _internal_seed and os.path.isfile(_internal_seed):
            try:
                os.remove(_internal_seed)
                print("Removed transient optimizer seed:", _internal_seed)
            except Exception as _seed_cleanup_exc:
                print("Transient optimizer-seed cleanup warning:", str(_seed_cleanup_exc)[:240])
'''


def generate_lumerical_adjoint_notebook(
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Generate a self-contained, one-owner 3D LumOpt adjoint notebook."""

    specification = deepcopy(spec)
    target_uid = int(specification.get("component_uid", -1))
    target = next(
        (
            component for component in components
            if int(component.get("uid", -2)) == target_uid
        ),
        None,
    )
    if target is None:
        raise ValueError("The optimization component is not present in the selected export scope.")
    if str(target.get("kind", "")) != str(specification.get("component_kind", "")):
        raise ValueError("The optimization specification does not match the selected component kind.")

    # Re-normalize saved/user-provided JSON through the public validator.
    objective = dict(specification.get("objective", {}))
    optimizer = dict(specification.get("optimizer", {}))
    specification = normalize_lumerical_optimization_spec(
        target,
        specification.get("parameters", []),
        center_wavelength_um=float(objective.get("center_wavelength_um", 0.0)),
        bandwidth_nm=float(objective.get("bandwidth_nm", 0.0)),
        wavelength_points=int(objective.get("wavelength_points", 1)),
        max_iterations=int(optimizer.get("max_iterations", 30)),
    )
    best_project = Path(str(configuration.get("project_file", "optimized_component.fsp"))).name
    if not best_project.lower().endswith(".fsp"):
        best_project += ".fsp"
    best_stem = Path(best_project).stem
    specification["best_project_file"] = best_project
    patch_context = deepcopy(target.get("params", {}))

    base_configuration = deepcopy(configuration)
    base_configuration.update(
        {
            "dimension": "3D",
            "resource_mode": "GPU",
            "run_after_build": False,
            "wavelength_start_um": float(specification["objective"]["wavelength_start_um"]),
            "wavelength_stop_um": float(specification["objective"]["wavelength_stop_um"]),
            "frequency_points": int(specification["objective"]["wavelength_points"]),
            "project_file": best_stem + "_adjoint_seed.fsp",
        }
    )
    base_notebook, warnings = generate_lumerical_notebook(components, base_configuration)
    payload = _notebook_literal_assignments(base_notebook)
    warnings = list(warnings)
    _ensure_porttransmission_receiver(payload, specification, warnings)
    fiber_pose = _fiber_pose_contract(payload, target, specification)
    gaussian_alignment_envelope = _gaussian_alignment_domain_envelope(
        fiber_pose, specification
    )
    if gaussian_alignment_envelope is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = (
            gaussian_alignment_envelope
        )
        nominal_bounds = list(map(float, payload["BOUNDING_BOX_UM"]))
        payload["BOUNDING_BOX_UM"] = [
            min(nominal_bounds[0], x_min),
            min(nominal_bounds[1], y_min),
            max(nominal_bounds[2], x_max),
            max(nominal_bounds[3], y_max),
        ]
        payload["SETTINGS"]["fixed_sampling_z_bounds_um"] = [z_min, z_max]
        fiber_pose["fixed_domain_envelope_um"] = list(
            gaussian_alignment_envelope
        )
        warnings.append(
            "The fixed adjoint FDTD region reserves every Gaussian source and "
            "input-power-monitor pose allowed by the selected fiber_offset and "
            "angle_theta bounds."
        )
    shape_snapshots = _shape_snapshots(target, list(payload["GEOMETRY"]), specification)
    mesh_um = max(1e-4, float(specification.get("optimization_mesh_um", 0.05)))
    specification["optimization_mesh_um"] = mesh_um
    volume_um = _optimization_volume(
        shape_snapshots, list(payload["MATERIAL_STACK"]), mesh_um
    )
    objective_ports = _objective_ports(payload, specification)
    specification["objective"]["ports"] = deepcopy(objective_ports)
    gaussian_excitation = (
        str(objective_ports.get("excitation_type", "fiber_mode"))
        == "gaussian_beam"
    )

    warnings.extend(
        [
            "Adjoint shape interpolation is fixed-topology and nominal-centered. Wide bounds on strongly nonlinear geometry can differ from rebuilding the editor component at every intermediate point; inspect the seed and best FSP files.",
            "Device ports, material stack, integer tooth counts, and the optimization volume remain fixed during LumOpt. Selected grating alignment variables are optimized first with synchronized GPU forward solves and then frozen for the shape-adjoint stage.",
            "LumOpt may create transient internal solver files in its working directory, but store_all_simulations is false and no per-iteration FSP is retained or fetched.",
        ]
    )
    if str(target.get("kind")) == "GC-SOI":
        warnings.append(
            f"GC-SOI period count is frozen at {int(specification['fixed_period_count'])}; pitch changes cannot add or remove teeth."
        )

    payload_source = (
        "# Embedded nominal model and fixed-topology adjoint specification.\n"
        + f"EXPORT_SCOPE_LABEL = {payload['EXPORT_SCOPE_LABEL']!r}\n"
        + f"EXPORTED_COMPONENTS = {pprint.pformat(payload['EXPORTED_COMPONENTS'], width=120, sort_dicts=False)}\n"
        + f"SOURCE_COMPONENTS_JSON = {pprint.pformat(payload['SOURCE_COMPONENTS_JSON'], width=160, compact=True, sort_dicts=False)}\n"
        + f"SETTINGS = {pprint.pformat(payload['SETTINGS'], width=120, sort_dicts=False)}\n"
        + f"MATERIAL_STACK = {pprint.pformat(payload['MATERIAL_STACK'], width=120, sort_dicts=False)}\n"
        + f"BOUNDING_BOX_UM = {pprint.pformat(payload['BOUNDING_BOX_UM'])}\n"
        + f"GEOMETRY = {pprint.pformat(payload['GEOMETRY'], width=160, compact=True, sort_dicts=False)}\n"
        + f"PORTS = {pprint.pformat(payload['PORTS'], width=120, sort_dicts=False)}\n"
        + f"FIBER_GEOMETRIES = {pprint.pformat(payload['FIBER_GEOMETRIES'], width=120, sort_dicts=False)}\n"
        + f"GAUSSIAN_SOURCES = {pprint.pformat(payload.get('GAUSSIAN_SOURCES', []), width=120, sort_dicts=False)}\n"
        + f"PORTS_JSON = {pprint.pformat(payload['PORTS_JSON'], width=120, sort_dicts=False)}\n"
        + f"MONITORS = {pprint.pformat(payload['MONITORS'], width=120, sort_dicts=False)}\n"
        + f"GRATING_ANALYSIS = {pprint.pformat(payload['GRATING_ANALYSIS'], width=120, sort_dicts=False)}\n"
        + f"MMI_ANALYSIS = {pprint.pformat(payload['MMI_ANALYSIS'], width=120, sort_dicts=False)}\n"
        + f"OPTIMIZATION_SPEC = {pprint.pformat(specification, width=120, sort_dicts=False)}\n"
        + f"OPT_COMPONENT_NOMINAL_PARAMS = {pprint.pformat(patch_context, width=120, sort_dicts=False)}\n"
        + f"OPT_OBJECTIVE_PORTS = {pprint.pformat(objective_ports, width=120, sort_dicts=False)}\n"
        + f"OPT_FIBER_POSE = {pprint.pformat(fiber_pose, width=140, sort_dicts=False)}\n"
        + f"OPT_SHAPE_SNAPSHOTS = {pprint.pformat(shape_snapshots, width=180, compact=True, sort_dicts=False)}\n"
        + f"OPTIMIZATION_VOLUME_UM = {pprint.pformat(volume_um)}\n"
        + f"EXPORT_WARNINGS = {pprint.pformat(warnings, width=120)}\n"
        + "for warning in EXPORT_WARNINGS:\n    print('Export note:', warning)\n"
        + _runtime_setup_source(_BUILD_CELL)
    )
    build_source = (
        "# Build one GPU-configured seed model in the licensed persistent Lambda session.\n"
        f"REMOTE_MODEL_BUILDER = {repr(_BUILD_CELL)}\n"
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
        "    + 'OPTIMIZATION_SPEC = ' + repr(OPTIMIZATION_SPEC) + '\\n'\n"
        "    + 'OPT_COMPONENT_NOMINAL_PARAMS = ' + repr(OPT_COMPONENT_NOMINAL_PARAMS) + '\\n'\n"
        "    + 'OPT_OBJECTIVE_PORTS = ' + repr(OPT_OBJECTIVE_PORTS) + '\\n'\n"
        "    + 'OPT_FIBER_POSE = ' + repr(OPT_FIBER_POSE) + '\\n'\n"
        "    + 'OPT_SHAPE_SNAPSHOTS = ' + repr(OPT_SHAPE_SNAPSHOTS) + '\\n'\n"
        "    + 'OPTIMIZATION_VOLUME_UM = ' + repr(OPTIMIZATION_VOLUME_UM) + '\\n'\n"
        ")\n"
        "run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER, 'Build 3D adjoint seed model directly', timeout=1800)\n"
        "print('Built the adjoint seed directly in memory.')\n"
    )
    opt_fields_source = (
        f"REMOTE_OPT_FIELDS_SETUP = {repr(_OPT_FIELDS_SETUP_REMOTE)}\n"
        "run_remote_checked(REMOTE_OPT_FIELDS_SETUP, 'Add uniform opt mesh and opt_fields', timeout=600)\n"
    )
    resource_source = (
        _find_code_cell(base_notebook, "REMOTE_RESOURCE_AND_SAVE =")
        + "\n# Use the required inspection FSP as LumOpt's seed; do not write a duplicate seed file.\n"
        + "REMOTE_INTERNAL_SEED_FSP = REMOTE_INSPECTION_PROJECT_FILE\n"
        + "print('LumOpt seed is the saved inspection FSP:', REMOTE_INTERNAL_SEED_FSP)\n"
    )
    close_seed_source = (
        "# End seed ownership before LumOpt opens its single persistent FDTD owner.\n"
        "_close_seed_code = 'fdtd.close()\\ndel fdtd\\nprint(\"Seed FDTD owner closed.\")'\n"
        "if not SETTINGS.get('run_after_build', True):\n"
        "    _close_seed_code += '\\nimport os\\n_p = globals().get(\"REMOTE_RUNTIME_PROJECT_FILE\", \"\")\\nif _p and os.path.isfile(_p): os.remove(_p)'\n"
        "run_remote_checked(_close_seed_code, "
        "'Close seed FDTD owner', timeout=120)\n"
        "REMOTE_BASE_FSP = REMOTE_INTERNAL_SEED_FSP if SETTINGS.get('run_after_build', True) else None\n"
    )
    optimize_source = (
        f"REMOTE_LUMOPT_RUNTIME = {repr(_LUMOPT_RUNTIME_REMOTE)}\n"
        # ``REMOTE_BASE_FSP`` above lives in the local notebook kernel.  The
        # optimization runtime executes in Lambda's persistent Python process,
        # so explicitly inject the resolved remote path into that process.
        # Without this payload the generated notebook fails before LumOpt can
        # open the seed project.
        "REMOTE_OPT_PROGRESS_FILE = REMOTE_WORK + '/adjoint_live_progress.jsonl'\n"
        "_remote_lumopt_payload = (\n"
        "    'REMOTE_BASE_FSP = ' + repr(REMOTE_BASE_FSP) + '\\n'\n"
        "    + 'REMOTE_OPT_PROGRESS_FILE = ' + repr(REMOTE_OPT_PROGRESS_FILE) + '\\n'\n"
        ")\n"
        "if SETTINGS.get('run_after_build', True):\n"
        "    solve_remote_checked(\n"
        "        _remote_lumopt_payload + REMOTE_LUMOPT_RUNTIME,\n"
        "        'GPU fiber alignment + LumOpt 3D shape-adjoint optimization',\n"
        "        timeout=172800,\n"
        "        progress_file=REMOTE_OPT_PROGRESS_FILE,\n"
        "    )\n"
        "else:\n"
        "    print('Optimization disabled by RUN_SIMULATION in cell 1.')\n"
    )
    fetch_source = r'''# Fetch compact CPU-generated optimization artifacts and the always-saved best FSP.
if not SETTINGS.get("run_after_build", True):
    print("Optimization was not run, so there are no optimization artifacts to fetch.")
else:
    REMOTE_OPT_ARTIFACTS = [
        REMOTE_WORK + "/adjoint_optimization_history.npz",
        REMOTE_WORK + "/adjoint_optimization_summary.json",
        REMOTE_WORK + "/adjoint_parameter_patch.json",
        REMOTE_WORK + "/adjoint_optimization_history.png",
        REMOTE_WORK + "/adjoint_live_progress.jsonl",
        REMOTE_WORK + "/summary.txt",
    ]
    REMOTE_OPT_ARTIFACTS.insert(
        0, REMOTE_WORK + "/fsp/" + OPTIMIZATION_SPEC["best_project_file"]
    )
    _status = lam.get("{path: bool(os.path.isfile(path) and os.path.getsize(path) > 0) for path in %r}" % REMOTE_OPT_ARTIFACTS)
    _missing = [path for path in REMOTE_OPT_ARTIFACTS if not _status.get(path, False)]
    if _missing:
        raise RuntimeError("Required optimization artifacts are missing: " + repr(_missing))
    FETCHED_OPTIMIZATION_ARTIFACTS = []
    for remote_path in REMOTE_OPT_ARTIFACTS:
        local_directory = PIRIS_FSP_DIR if remote_path.lower().endswith(".fsp") else PIRIS_RESULTS_DIR
        local_path = local_directory / os.path.basename(remote_path)
        fetched = lam.fetch(remote_path, str(local_path))
        if not local_path.is_file() or local_path.stat().st_size <= 0:
            raise RuntimeError("Required optimization artifact was not fetched: " + str(local_path))
        FETCHED_OPTIMIZATION_ARTIFACTS.append(fetched)
        print("saved ->", fetched)
    from IPython.display import Image, display
    history_plot = PIRIS_RESULTS_DIR / "adjoint_optimization_history.png"
    if history_plot.is_file():
        display(Image(filename=str(history_plot), width=1000))
'''

    parameter_names = ", ".join(
        f"`{row['parameter']}`" for row in specification["parameters"]
    )
    alignment_names = ", ".join(
        f"`{name}`" for name in specification.get("alignment_parameters", [])
    ) or "none"
    shape_names = ", ".join(
        f"`{name}`" for name in specification.get("adjoint_geometry_parameters", [])
    ) or "none"
    objective_text = (
        (
            "selected-TE waveguide power divided by the measured Gaussian input-monitor power"
            if gaussian_excitation
            else "selected-TE waveguide power divided by the measured fiber input-monitor power"
        )
        if specification["component_kind"] in {"Grating coupler", "GC-SOI"}
        else "upper-output fundamental-TE power divided by launched input-port power"
    )
    alignment_label = "source-alignment" if gaussian_excitation else "fiber-alignment"
    alignment_contract_text = (
        "`angle_theta` drives the independent Gaussian source and its local-TE S polarization; "
        "the ordinary horizontal input-power monitor follows the tilted beam-axis intersection. "
        "`fiber_offset` moves the source/monitor assembly along the grating local X axis."
        if gaussian_excitation
        else "`angle_theta` drives the fiber core/cladding, tilted source port, and ordinary "
        "Z-normal input-power monitor together. The monitor follows the tilted-axis intersection "
        "and is never treated as a modal port. `fiber_offset` moves that assembly along the "
        "grating local X axis."
    )
    intro = f"""# Max Layout → Lumerical 3D alignment + LumOpt shape-adjoint optimization

This notebook optimizes **{specification['component_kind']} (UID {specification['component_uid']})**. Continuous device geometry uses a true bundled-LumOpt shape-adjoint stage with official v261+ `lumopt2.Parametrization` when available and verified legacy `ParameterizedGeometry` otherwise. Grating {alignment_label} uses bounded GPU forward solves because moving the excitation/measurement basis is not an ordinary material-boundary adjoint derivative.

**Optimized JSON parameters:** {parameter_names}  
**Synchronized {alignment_label} parameters:** {alignment_names}
**Shape-adjoint geometry parameters:** {shape_names}  
**Objective:** maximize linear {objective_text} from **{specification['objective']['wavelength_start_um']:.9g} µm** to **{specification['objective']['wavelength_stop_um']:.9g} µm** using **{specification['objective']['wavelength_points']}** wavelength sample(s).

- The build stage uses up to 30 CPU threads. Fiber-alignment evaluations, forward solves, adjoint solves, and best-design validation use the saved GPU resource configuration. Plotting, JSON/NPZ serialization, and final artifact handling switch back to CPU.
- During the blocking solve, the cell redraws a live table after every completed {alignment_label} or shape-adjoint iteration. Each row reports the current linear objective and the complete selected JSON parameter vector; no extra FDTD solve is performed for reporting.
- Exactly one FDTD owner exists at a time: the seed builder is closed before LumOpt opens its persistent owner. One Shared Web checkout surrounds the entire build/optimization/save workflow.
- `store_all_simulations=False`: no per-iteration FSP is retained or fetched. Compact history, summary, editor-patch, graph artifacts, the inspection FSP, and the best-geometry FSP are always saved.
- {alignment_contract_text} Both parameters are frozen at their best forward-solve values during the shape-adjoint stage.
- Integer tooth counts, device topology, waveguide receivers, material stack, and process thicknesses remain fixed. GC-SOI keeps exactly **{specification.get('fixed_period_count', 'N/A')}** periods.
- The geometry callback uses exact nominal/minimum/maximum editor-built polygons and fixed-topology, nominal-centered piecewise-linear interpolation. This is genuine adjoint optimization, but it is not a symbolic reimplementation of every editor geometry formula. Keep bounds moderate; inspection and best-design FSP files are stored automatically.
- Always run the final release cell after success, failure, or interruption.
"""

    notebook = {
        "cells": [
            _notebook_cell(
                "code",
                _quick_run_options_cell(
                    {**base_configuration, "run_after_build": True},
                    workflow="adjoint optimization",
                ),
            ),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Connect to Lambda\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 2 · Acquire Ansys Shared Web licences\n"),
            _notebook_cell("code", _LICENSE_CHECKOUT_CELL),
            _notebook_cell("markdown", "## 3 · Embedded fixed-topology model and optimization contract\n"),
            _notebook_cell("code", payload_source),
            _notebook_cell("markdown", "## 4 · Build one 3D seed model\n"),
            _notebook_cell("code", build_source),
            _notebook_cell("markdown", "## 5 · Add co-located uniform optimization mesh and `opt_fields`\n"),
            _notebook_cell("code", opt_fields_source),
            _notebook_cell("markdown", "## 6 · Configure GPU resources and prepare the internal optimizer seed\n"),
            _notebook_cell("code", resource_source),
            _notebook_cell("code", "from IPython.display import FileLink, display\ndisplay(FileLink(str(LOCAL_INSPECTION_PROJECT_FILE)))\n"),
            _notebook_cell("markdown", "## 7 · Transfer ownership from the seed builder to LumOpt\n"),
            _notebook_cell("code", close_seed_source),
            _notebook_cell("markdown", "## 8 · Run bounded 3D GPU shape adjoint\n"),
            _notebook_cell("code", optimize_source),
            _notebook_cell("markdown", "## 9 · Fetch compact history and the required best-design FSP\n"),
            _notebook_cell("code", fetch_source),
            _notebook_cell("markdown", "## 10 · Release FDTD and return all roamed HPC Packs\n\nAlways run this cell, including after an interrupted optimization.\n"),
            _notebook_cell("code", _RELEASE_LICENSES_CELL),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "max_layout": {
                "export": "lumerical-lumopt-shape-adjoint",
                "units": "um",
                "dimension": "3D",
                "objective": specification["objective"]["kind"],
                "optimizer": "L-BFGS-B",
                "resource": "GPU-forward-adjoint-CPU-postprocessing",
                "solver_ownership": "one-fdtd-owner-at-a-time",
                "store_all_simulations": False,
                "license_lifecycle": "shared-web-3-hpc-packs-save-fetch-release",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, warnings


def write_lumerical_adjoint_notebook(
    path: str | Path,
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    """Write a LumOpt shape-adjoint notebook and return export notes."""

    notebook, warnings = generate_lumerical_adjoint_notebook(
        components, configuration, spec
    )
    Path(path).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return warnings


__all__ = [
    "SUPPORTED_ADJOINT_COMPONENT_KINDS",
    "adjoint_optimizable_component_parameters",
    "normalize_lumerical_optimization_spec",
    "generate_lumerical_adjoint_notebook",
    "write_lumerical_adjoint_notebook",
]
