"""Shared RF simulation defaults for CPW MODE/FDTD notebook exports.

The optical material database is deliberately not reused here.  Microwave
permittivity, loss tangent, and conductivity are independent inputs, and the
dedicated RF exporter consumes this small JSON-safe schema directly.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


RF_FDE_COMPONENT_KINDS = {"CPW"}
RF_FDTD_COMPONENT_KINDS = {
    "CPW open",
    "CPW short",
    "Tapered CPW",
    "Symmetric CPW taper",
    "CPW bend",
    "Segmented electrode",
}
RF_SIMULATABLE_COMPONENT_KINDS = RF_FDE_COMPONENT_KINDS | RF_FDTD_COMPONENT_KINDS


RF_STACK_PRESETS: dict[str, list[dict[str, Any]]] = {
    "TFLN CPW": [
        {
            "name": "Si handle",
            "role": "dielectric",
            "thickness_um": 500.0,
            "relative_permittivity": 11.7,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "SiO2 BOX",
            "role": "dielectric",
            "thickness_um": 5.0,
            "relative_permittivity": 3.85,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "X-cut TFLN",
            "role": "dielectric",
            "thickness_um": 0.4,
            # Ansys's X-cut LN electrical-material example uses the
            # extraordinary value on x and the ordinary value on y/z.
            "anisotropic": True,
            "relative_permittivity_x": 27.9,
            "relative_permittivity_y": 44.3,
            "relative_permittivity_z": 44.3,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "SiO2 cladding",
            "role": "dielectric",
            "thickness_um": 1.0,
            "relative_permittivity": 3.85,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "RF metal",
            "role": "metal",
            "thickness_um": 1.0,
            "metal_model": "Conductive 3D",
            "conductivity_s_per_m": 4.10e7,
            "gds_layers": [4, 5],
        },
        {
            "name": "Top air",
            "role": "dielectric",
            "thickness_um": 150.0,
            "relative_permittivity": 1.0,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
    ],
    "Official Ansys conductor-backed FR4": [
        {
            "name": "FR4 substrate",
            "role": "dielectric",
            "thickness_um": 500.0,
            "relative_permittivity": 4.34001,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "RF metal",
            "role": "metal",
            "thickness_um": 0.0,
            "metal_model": "PEC",
            "conductivity_s_per_m": 0.0,
            "gds_layers": [4, 5],
        },
        {
            "name": "Top air",
            "role": "dielectric",
            "thickness_um": 5000.0,
            "relative_permittivity": 1.0,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
    ],
    "Fused-silica CPW": [
        {
            "name": "Fused silica",
            "role": "dielectric",
            "thickness_um": 500.0,
            "relative_permittivity": 3.85,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
        {
            "name": "RF metal",
            "role": "metal",
            "thickness_um": 1.0,
            "metal_model": "Conductive 3D",
            "conductivity_s_per_m": 5.8e7,
            "gds_layers": [4, 5],
        },
        {
            "name": "Top air",
            "role": "dielectric",
            "thickness_um": 150.0,
            "relative_permittivity": 1.0,
            "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        },
    ],
}


def rf_workflow_for_component(kind: str) -> str:
    """Return the physically appropriate official-example workflow."""
    name = str(kind)
    if name in RF_FDE_COMPONENT_KINDS:
        return "fde"
    if name in RF_FDTD_COMPONENT_KINDS:
        return "fdtd"
    raise ValueError(f"{name or 'Selected component'} is not supported by the RF exporter.")


def default_rf_stack(preset: str = "TFLN CPW") -> list[dict[str, Any]]:
    if preset not in RF_STACK_PRESETS:
        raise ValueError(f"Unknown RF stack preset: {preset}")
    return deepcopy(RF_STACK_PRESETS[preset])


def default_rf_configuration(kind: str = "CPW") -> dict[str, Any]:
    workflow = rf_workflow_for_component(kind)
    return {
        "rf_workflow": workflow,
        "rf_stack_preset": "TFLN CPW",
        "material_stack": default_rf_stack("TFLN CPW"),
        "frequency_start_ghz": 1.0,
        "frequency_stop_ghz": 100.0,
        "frequency_points": 25,
        "target_frequency_ghz": 30.0,
        "metal_model": "Conductive 3D",
        "metal_conductivity_s_per_m": 4.10e7,
        "metal_thickness_um": 1.0,
        "substrate_relative_permittivity": 3.85,
        "substrate_loss_tangent": 0.0,
        "substrate_thickness_um": 500.0,
        "backing_ground": False,
        "mesh_edge_um": 0.25,
        "mesh_vertical_um": 0.10,
        "mesh_bulk_um": 5.0,
        "boundary_type": "Metal transverse / PML propagation",
        "pml_layers": 28,
        "port_clearance_wavelengths": 0.25,
        "port_transverse_span_um": 450.0,
        "port_vertical_span_um": 650.0,
        # RF FDTD reference planes are explicit editor objects by default.
        # Endpoint-derived planes remain available only as an opt-in fallback.
        "rf_port_strategy": "manual_only",
        "use_endpoint_reference_planes": False,
        "input_port_inset_um": 0.0,
        "output_port_inset_um": 0.0,
        "multifrequency_mode_injection": True,
        "optimize_for_short_pulse": False,
        "snap_pec_to_yee_cell_boundary": False,
        "simulation_time_ns": 40.0,
        "auto_shutoff": 1e-6,
        "build_cpu_threads": 30,
        "resource_mode": "CPU" if workflow == "fde" else "GPU",
        "run_after_build": True,
        "save_inspection_fsp": True,
        "save_final_fsp": True,
    }


def _positive_finite(configuration: dict[str, Any], key: str) -> float:
    value = float(configuration[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite value.")
    return value


def normalize_rf_configuration(
    kind: str, configuration: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge/validate editor settings without silently changing the solver."""
    expected_workflow = rf_workflow_for_component(kind)
    result = default_rf_configuration(kind)
    if configuration:
        result.update(deepcopy(configuration))
    workflow = str(result.get("rf_workflow", expected_workflow)).strip().lower()
    if workflow != expected_workflow:
        raise ValueError(
            f"{kind} requires the {expected_workflow.upper()} RF workflow; "
            f"{workflow.upper() or 'an empty workflow'} was requested."
        )
    result["rf_workflow"] = workflow
    start = _positive_finite(result, "frequency_start_ghz")
    stop = _positive_finite(result, "frequency_stop_ghz")
    if stop <= start:
        raise ValueError("frequency_stop_ghz must be larger than frequency_start_ghz.")
    target = _positive_finite(result, "target_frequency_ghz")
    if not start <= target <= stop:
        raise ValueError("target_frequency_ghz must lie inside the RF sweep.")
    points = int(result["frequency_points"])
    if points < 2 or points > 10001:
        raise ValueError("frequency_points must be between 2 and 10001.")
    result["frequency_points"] = points
    for key in (
        "mesh_edge_um", "mesh_vertical_um", "mesh_bulk_um",
        "simulation_time_ns", "auto_shutoff",
    ):
        _positive_finite(result, key)
    if int(result.get("pml_layers", 0)) < 1:
        raise ValueError("pml_layers must be at least 1.")
    if int(result.get("build_cpu_threads", 0)) < 1:
        raise ValueError("build_cpu_threads must be at least 1.")
    resource = str(result.get("resource_mode", "")).strip().upper()
    required_resource = "CPU" if workflow == "fde" else "GPU"
    if resource != required_resource:
        raise ValueError(
            f"{kind} uses {required_resource}: MODE/FDE runs on CPU and 3D FDTD runs on GPU."
        )
    result["resource_mode"] = resource
    port_strategy = str(result.get("rf_port_strategy", "manual_only")).strip().lower()
    if port_strategy not in {
        "component_endpoints",
        "manual_only",
        "manual_or_component_endpoints",
        "cross_section_only",
    }:
        raise ValueError("Unknown RF port strategy: " + port_strategy)
    if workflow == "fdtd" and port_strategy == "cross_section_only":
        raise ValueError("A 3D RF FDTD run needs manual RF planes or component endpoint planes.")
    result["rf_port_strategy"] = port_strategy
    stack = list(result.get("material_stack") or [])
    if not stack:
        raise ValueError("The RF material stack cannot be empty.")
    metal_rows = [row for row in stack if str(row.get("role", "")).lower() == "metal"]
    if len(metal_rows) != 1:
        raise ValueError("The RF stack must contain exactly one metal row.")
    for row in stack:
        thickness = float(row.get("thickness_um", 0.0))
        if not math.isfinite(thickness) or thickness < 0.0:
            raise ValueError(f"RF layer {row.get('name', 'unnamed')} has an invalid thickness.")
        if str(row.get("role", "dielectric")).lower() == "dielectric":
            anisotropic = bool(row.get("anisotropic", False))
            keys = (
                ("relative_permittivity_x", "relative_permittivity_y", "relative_permittivity_z")
                if anisotropic else ("relative_permittivity",)
            )
            for key in keys:
                epsilon = float(row.get(key, 0.0))
                if not math.isfinite(epsilon) or epsilon <= 0.0:
                    raise ValueError(
                        f"RF dielectric {row.get('name', 'unnamed')} needs positive {key}."
                    )
    result["material_stack"] = deepcopy(stack)
    return result
