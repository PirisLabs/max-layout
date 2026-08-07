"""Lumerical simulation metadata and self-contained notebook export.

Ports and monitors are explicit simulation-only objects placed from the editor
library.  The GDS builder ignores them, so they never become fabrication
geometry.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
import json
import math
import pprint

import numpy as np

from .constants import LAYER_NAME_MAP, SIMULATION_COMPONENT_KINDS
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


STACK_PRESETS: dict[str, list[dict[str, Any]]] = {
    "TFLN on SiO2": [
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
        {"name": "SiO2 BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 5.0, "role": "background", "gds_layer": 0},
        {"name": "Exported TFLN cross-section", "material": "LiNbO3", "thickness_um": 0.4, "etch_depth_um": 0.2, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1, 2]},
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
        {"name": "Si substrate", "material": "Si (Silicon) - Palik", "thickness_um": 3.0, "role": "background", "gds_layer": 0, "mesh_factor": 0.1},
        {"name": "SiO2 BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "mesh_factor": 0.1},
        {"name": "GC-SOI residual slab", "material": "Si (Silicon) - Palik", "thickness_um": 0.12, "etch_depth_um": 0.12, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [1], "slab_extent": "geometry", "mesh_factor": 0.1},
        {"name": "GC-SOI upper silicon", "material": "Si (Silicon) - Palik", "thickness_um": 0.10, "etch_depth_um": 0.10, "sidewall_angle_deg": 90.0, "role": "geometry", "gds_layers": [2], "slab_extent": "geometry", "mesh_factor": 0.1},
        {"name": "SiO2 TOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 0.48, "role": "background", "gds_layer": 0, "conformal": True, "mesh_factor": 0.1},
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
    return deepcopy(STACK_PRESETS.get(preset, STACK_PRESETS["TFLN on SiO2"]))


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
        params["z reference"] = "top of SiO2 cladding"
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
    local_x_span = max(0.0, float(params.get("x span", 0.0 if local_normal == "X" else legacy_span)))
    local_y_span = max(0.0, float(params.get("y span", 0.0 if local_normal == "Y" else legacy_span)))
    z_span = max(0.0, float(params.get("z span", params.get("z_span_um", 2.0))))
    if global_normal == "X":
        x_span, y_span = 0.0, max(local_x_span, local_y_span, legacy_span)
    elif global_normal == "Y":
        x_span, y_span = max(local_x_span, local_y_span, legacy_span), 0.0
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
    return params


def _collect_export_data(
    components: list[dict[str, Any]], included_layers: set[tuple[int, int]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float], list[str]]:
    geometry: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    fiber_geometries: list[dict[str, Any]] = []
    monitors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for component in components:
        component_kind = str(component.get("kind", ""))
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
        if component_kind in {"Power monitor", "Mode expansion monitor", "Field profile monitor"}:
            monitors.append(_standalone_monitor(component))
            continue
        if component.get("simulation_ports"):
            warnings.append(
                f"UID {component.get('uid')}: legacy embedded simulation ports were ignored; "
                "place ports explicitly from the Ports & monitors library."
            )

    if geometry:
        all_points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in geometry])
        minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
        origin = 0.5 * (minimum + maximum)
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
    else:
        centers = (
            [port.get("center", (0.0, 0.0)) for port in ports]
            + [fiber.get("center", (0.0, 0.0)) for fiber in fiber_geometries]
            + [monitor.get("center", (0.0, 0.0)) for monitor in monitors]
        )
        if centers:
            points = np.asarray(centers, dtype=float)
            origin = 0.5 * (points.min(axis=0) + points.max(axis=0))
            for port in ports:
                port["center"] = [float(port["center"][0] - origin[0]), float(port["center"][1] - origin[1])]
            for monitor in monitors:
                monitor["center"] = [float(monitor["center"][0] - origin[0]), float(monitor["center"][1] - origin[1])]
            for fiber in fiber_geometries:
                fiber["center"] = [float(fiber["center"][0] - origin[0]), float(fiber["center"][1] - origin[1])]
            shifted = points - origin
            bbox = [float(shifted[:, 0].min() - 1.0), float(shifted[:, 1].min() - 1.0),
                    float(shifted[:, 0].max() + 1.0), float(shifted[:, 1].max() + 1.0)]
        else:
            origin = np.array([0.0, 0.0])
            bbox = [-1.0, -1.0, 1.0, 1.0]
        warnings.append(
            "No physical device polygons were selected. This notebook contains only ports/monitors and the background stack; "
            "choose a device-containing scope before solving a component response."
        )
    return geometry, ports, fiber_geometries, monitors, bbox, warnings + [f"Layout origin moved by ({origin[0]:.6g}, {origin[1]:.6g}) µm for simulation."]


def _synchronize_fiber_port_parameters(
    ports: list[dict[str, Any]],
    fiber_geometries: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Make each Z-normal fiber port follow its matching fiber, as in the Ansys 3D example.

    The official grating-coupler model treats the fiber geometry as the source
    of truth for theta and derives the FDTD port rotation offset from that
    angle and the fiber core diameter.  Matching by simulation parent first
    preserves independent manually placed fiber assemblies.
    """
    for port in ports:
        if str(port.get("plane normal", "X")).upper() != "Z" or not fiber_geometries:
            continue
        parent_uid = int(port.get("parent_component_uid", -1))
        matching = [
            fiber for fiber in fiber_geometries
            if int(fiber.get("parent_component_uid", -2)) == parent_uid
        ]
        candidates = matching or fiber_geometries
        port_center = np.asarray(port.get("center", (0.0, 0.0)), dtype=float)
        fiber = min(
            candidates,
            key=lambda item: float(
                np.linalg.norm(np.asarray(item.get("center", (0.0, 0.0)), dtype=float) - port_center)
            ),
        )
        theta_deg = float(fiber.get("angle theta", port.get("angle theta", 7.0)))
        phi_deg = float(fiber.get("angle phi", port.get("angle phi", 0.0)))
        previous_theta = float(port.get("angle theta", theta_deg))
        previous_phi = float(port.get("angle phi", phi_deg))
        core_diameter_um = max(1e-6, float(fiber.get("core diameter_um", 9.0)))
        port["angle theta"] = theta_deg
        port["angle phi"] = phi_deg
        port["rotation offset_um"] = 4.0 * core_diameter_um * math.tan(math.radians(theta_deg))
        if abs(previous_theta - theta_deg) > 1e-9 or abs(previous_phi - phi_deg) > 1e-9:
            warnings.append(
                "Fiber-axis port %s was synchronized to fiber %s: theta %.6g°, phi %.6g°."
                % (port.get("name", ""), fiber.get("name", ""), theta_deg, phi_deg)
            )


def _notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


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
import os
import re
import sys
sys.path.insert(0, os.path.expanduser("~/Desktop/lumerical"))
from lambda_remote import Lambda, _SSH, HOST

_remote_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", PIRIS_SESSION_DIR.name).strip("._") or "max_layout"
REMOTE_WORK = "/lambda/nfs/piris-lumerical/projects/max_layout/" + _remote_slug
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


def solve_remote_checked(code, label, timeout=21600):
    """Keep the live progress display while still detecting a solver traceback."""
    guarded, ok_marker, error_marker = _guard_remote_code(code, label)
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


_LICENSE_CHECKOUT_CELL = r'''import subprocess
from lambda_remote import _SSH, HOST

LIC = "/opt/lumerical/v261/licensingclient/linx64"

# 1. seed the Ansys web sign-in from the shared token (no-op if already seeded)
r = subprocess.run(_SSH + [HOST,
    'test -s ~/.ansys/ansysid/token.json && echo "sign-in already seeded" && exit 0; '
    'test -s ~/remote-token.json || { echo "ERROR: ~/remote-token.json missing on the node"; exit 1; }; '
    f'ANSYS_LICENSING_WEB=1 {LIC}/ansyscl -WebLoginInput ~/remote-token.json && '
    f'{LIC}/LicensingSettings web shared enable --mode user >/dev/null 2>&1; '
    'echo "sign-in seeded from ~/remote-token.json"'],
    capture_output=True, text=True, timeout=180)
print((r.stdout + r.stderr).strip())

# 2. roam 3 HPC Packs to this session for 24 h, exactly like TFLN_GC_1310.ipynb.
r = subprocess.run(_SSH + [HOST,
    f'{LIC}/LicensingSettings web shared products checkout '
    '--name "Ansys HPC Pack - Shared Web" --count 3 --expires "P1D" --mode user'],
    capture_output=True, text=True, timeout=180)
out = (r.stdout + r.stderr).strip()
print("HPC Packs:", "3 roamed to this session for 24 h" if "SUCCESS" in out else out[:400])
'''


_BUILD_CELL = r'''UM = 1e-6

os.makedirs(REMOTE_WORK, exist_ok=True)
REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")
os.makedirs(REMOTE_FSP_DIR, exist_ok=True)
os.chdir(REMOTE_WORK)
for _old_name in {
    os.path.basename(str(SETTINGS.get("project_file", "exported_component.fsp"))),
    "geometry_xyz_projections.png",
    "max_layout_results.npz",
    "max_layout_results.json",
    "grating_response.png",
    "grating_farfield.png",
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


def _layer_builder_geometry():
    """Convert embedded polygons to the struct-of-cell-arrays Layer Builder expects."""
    result = {}
    for polygon in GEOMETRY:
        key = f'{int(polygon["layer"])}:{int(polygon.get("datatype", 0))}'
        result.setdefault(key, []).append(np.asarray(polygon["vertices_um"], dtype=float) * UM)
    return result


def _add_material_stack(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um, pml_geometry_overlap_um):
    """Build films and tapered cross-sections with Lumerical's Layer Builder."""
    fdtd.addlayerbuilder()
    fdtd.set("name", "Max Layout material stack")
    fdtd.set("process name", "Max Layout export")
    fdtd.set("GDS sidewall angle position reference", "middle")
    fdtd.set("x", 0.5 * (bounds[0] + bounds[2]) * UM)
    fdtd.set("y", 0.5 * (bounds[1] + bounds[3]) * UM)
    fdtd.set("z", 0.0)
    fdtd.set("x span", (bounds[2] - bounds[0] + 2.0 * pml_geometry_overlap_um) * UM)
    fdtd.set("y span", (bounds[3] - bounds[1] + 2.0 * pml_geometry_overlap_um) * UM)

    geometry_by_layer = _layer_builder_geometry()
    if geometry_by_layer:
        # Python dict -> Lumerical struct; Python list -> Lumerical cell array.
        fdtd.set("geometry", geometry_by_layer)

    def add_background(name, material, z_min_um, z_max_um):
        if z_max_um <= z_min_um:
            return
        fdtd.addlayer(name)
        fdtd.setlayer(name, "start position", z_min_um * UM)
        fdtd.setlayer(name, "thickness", (z_max_um - z_min_um) * UM)
        fdtd.setlayer(name, "process", "background")
        fdtd.setlayer(name, "background material", material)

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

    def conformal_start(row_number, default_z_min_um):
        """Extend a cladding down through the etched portion of the patterned layer below."""
        consecutive_geometry = []
        for previous_row, previous_z0, previous_z1 in reversed(z_ranges[: row_number - 1]):
            if str(previous_row.get("role", "background")) != "geometry":
                if consecutive_geometry:
                    break
                continue
            consecutive_geometry.append((previous_row, previous_z0, previous_z1))
        if len(consecutive_geometry) > 1:
            # A multi-mask vertical device (for example the official SOI
            # residual-slab + upper-silicon pair) is one physical film.  Its
            # conformal oxide fills beside both masks down to the device base.
            return min(default_z_min_um, min(item[1] for item in consecutive_geometry))
        for previous_row, previous_z0, previous_z1 in consecutive_geometry:
            previous_thickness = previous_z1 - previous_z0
            previous_etch = min(
                previous_thickness,
                max(0.0, float(previous_row.get("etch_depth_um", previous_thickness))),
            )
            return min(default_z_min_um, previous_z1 - previous_etch)
        return default_z_min_um

    for row_index, (row, z0, z1) in enumerate(z_ranges, start=1):
        base_name = f'{row_index:02d} {row["name"]}'
        if row.get("role") != "geometry":
            background_z0 = conformal_start(row_index, z0) if bool(row.get("conformal", False)) else z0
            background_z1 = z1
            if row_index == 1:
                background_z0 = min(background_z0, simulation_z_min_um - pml_geometry_overlap_um)
            if row_index == len(z_ranges):
                background_z1 = max(background_z1, simulation_z_max_um + pml_geometry_overlap_um)
            add_background(base_name, row["material"], background_z0, background_z1)
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
                add_background(f"{base_name} unetched film", row["material"], z0, z1 - etch_depth_um)
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


def _add_layer_mesh_overrides(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um):
    """Resolve dispersive material indices and apply wavelength-scaled layer meshes."""
    wavelength_min_um = min(
        float(SETTINGS.get("wavelength_start_um", 1.25)),
        float(SETTINGS.get("wavelength_stop_um", 1.35)),
    )
    frequency_hz = 299792458.0 / (wavelength_min_um * UM)
    for row_index, (row, z0, z1) in enumerate(z_ranges, start=1):
        mesh_factor = max(0.0, float(row.get("mesh_factor", 0.1)))
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


def _polygon_cross_section(vertices, axis_index, coordinate_um, transverse_um):
    """Return the polygon interval crossed by a line through a manually placed port."""
    points = np.asarray(vertices, dtype=float)
    hits = []
    for index in range(len(points)):
        first = points[index]
        second = points[(index + 1) % len(points)]
        a = float(first[axis_index])
        b = float(second[axis_index])
        if abs(a - b) < 1e-12:
            if abs(coordinate_um - a) < 1e-9:
                hits.extend((float(first[1 - axis_index]), float(second[1 - axis_index])))
            continue
        if coordinate_um < min(a, b) - 1e-12 or coordinate_um > max(a, b) + 1e-12:
            continue
        fraction = (coordinate_um - a) / (b - a)
        if -1e-12 <= fraction <= 1.0 + 1e-12:
            hits.append(float(first[1 - axis_index] + fraction * (second[1 - axis_index] - first[1 - axis_index])))
    hits = sorted(set(round(value, 12) for value in hits))
    for start, stop in zip(hits[0::2], hits[1::2]):
        if start - 1e-9 <= transverse_um <= stop + 1e-9:
            return float(start), float(stop)
    return None


def _add_waveguide_boundary_extensions(fdtd, z_ranges, bounds, boundary_clearance_um, pml_geometry_overlap_um):
    """Continue each manually ported waveguide through its nearby PML, as in the Ansys example."""
    for port_index, port in enumerate(PORTS, start=1):
        if str(port.get("plane normal", "X")).upper() == "Z":
            continue
        actual = float(port.get("outward_orientation_deg", 0.0)) % 360.0
        nearest, axis, _ = _nearest_port_axis(actual)
        if axis not in {"x-axis", "y-axis"}:
            continue
        axis_index = 0 if axis == "x-axis" else 1
        transverse_index = 1 - axis_index
        original_center = np.asarray(port.get("center", (0.0, 0.0)), dtype=float)
        center = original_center.copy()
        distance_um = float(port.get("distance_um", 0.0))
        center += distance_um * np.asarray(
            [np.cos(np.deg2rad(actual)), np.sin(np.deg2rad(actual))], dtype=float
        )
        outward_sign = -1.0 if nearest in (180, 270) else 1.0
        extension_outer = bounds[axis_index] - pml_geometry_overlap_um if outward_sign < 0 else bounds[axis_index + 2] + pml_geometry_overlap_um
        extension_inner = float(center[axis_index]) - outward_sign * boundary_clearance_um
        extension_min = min(extension_outer, extension_inner)
        extension_max = max(extension_outer, extension_inner)

        for row_number, (row, z0, z1) in enumerate(z_ranges, start=1):
            if str(row.get("role", "background")) != "geometry":
                continue
            target_layers = {int(value) for value in row.get("gds_layers", [])}
            interval = None
            for inward_sample_um in (0.001, 0.01, 0.05, 0.2):
                sample_coordinate = float(original_center[axis_index]) - outward_sign * inward_sample_um
                for polygon in GEOMETRY:
                    if int(polygon.get("layer", -1)) not in target_layers:
                        continue
                    interval = _polygon_cross_section(
                        polygon["vertices_um"], axis_index, sample_coordinate, float(center[transverse_index])
                    )
                    if interval is not None:
                        break
                if interval is not None:
                    break
            if interval is None or interval[1] <= interval[0]:
                print("Warning: no exported waveguide cross-section was found for PML extension at port", port.get("name"))
                continue
            thickness_um = z1 - z0
            etch_depth_um = min(thickness_um, max(0.0, float(row.get("etch_depth_um", thickness_um))))
            if etch_depth_um <= 0.0:
                continue
            fdtd.addrect()
            fdtd.set("name", "port PML extension %d row %d" % (port_index, row_number))
            fdtd.set("material", str(row["material"]))
            if axis_index == 0:
                fdtd.set("x", 0.5 * (extension_min + extension_max) * UM)
                fdtd.set("x span", (extension_max - extension_min) * UM)
                fdtd.set("y", 0.5 * (interval[0] + interval[1]) * UM)
                fdtd.set("y span", (interval[1] - interval[0]) * UM)
            else:
                fdtd.set("y", 0.5 * (extension_min + extension_max) * UM)
                fdtd.set("y span", (extension_max - extension_min) * UM)
                fdtd.set("x", 0.5 * (interval[0] + interval[1]) * UM)
                fdtd.set("x span", (interval[1] - interval[0]) * UM)
            fdtd.set("z min", (z1 - etch_depth_um) * UM)
            fdtd.set("z max", z1 * UM)
            print("Extended waveguide at port %s through the %s PML." % (port.get("name"), axis[0].upper()))


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


def _vertical_reference_um(item, device_top_um, stack_top_um, silica_cladding_top_um=None):
    reference = str(item.get("z reference", "device top")).strip().lower()
    if reference in {"top of sio2 cladding", "top of silica cladding", "top cladding"}:
        return device_top_um if silica_cladding_top_um is None else silica_cladding_top_um
    if reference == "top of stack":
        return stack_top_um
    return device_top_um


def _add_fiber_geometries(fdtd, device_top_um, stack_top_um, silica_cladding_top_um):
    """Add only the official tilted core/cladding structure groups; never add a source here."""
    used_names = set()
    for index, fiber in enumerate(FIBER_GEOMETRIES, start=1):
        name = str(fiber.get("name") or f"fiber_{index}")
        if name in used_names:
            name = f"uid_{fiber.get('component_uid', 0)}_{name}"
        used_names.add(name)
        x_um, y_um = map(float, fiber.get("center", (0.0, 0.0)))
        theta_deg = float(fiber.get("angle theta", 10.0))
        phi_deg = float(fiber.get("angle phi", 0.0))
        core_diameter_um = max(1e-6, float(fiber.get("core diameter_um", 9.0)))
        cladding_diameter_um = max(core_diameter_um, float(fiber.get("cladding diameter_um", 50.0)))
        fiber_length_um = max(1e-6, float(fiber.get("fiber length_um", 20.0)))
        bottom_gap_um = float(fiber.get("distance_um", 0.0))
        reference_z_um = _vertical_reference_um(
            fiber, device_top_um, stack_top_um, silica_cladding_top_um
        )
        # Match the official fiber group's meaning of "z span": its requested
        # vertical span is preserved after tilting the cylinder.
        center_z_um = reference_z_um + bottom_gap_um + 0.5 * fiber_length_um

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
        fiber_setup_script = r"""
core_index = %core index%;
cladding_index = %cladding index%;
core_radius = %core diameter%/2.0;
cladding_radius = %cladding diameter%/2.0;
theta_rad = theta*pi/180.0;
L = %z span%/cos(theta_rad);
addcircle;
set("name","cladding");
set("radius",cladding_radius);
set("material","<Object defined dielectric>");
set("index",cladding_index);
set("override mesh order from material database",1);
set("mesh order",3);
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
set("override mesh order from material database",1);
set("mesh order",2);
set("x",0.0);
set("y",0.0);
set("z",0.0);
set("z span",L);
set("first axis","y");
set("rotation 1",theta);
"""
        fdtd.set("script", fiber_setup_script)
        fdtd.runsetup()
        print(
            "Added scripted Ansys fiber property group %s with core/cladding internal offsets (0, 0, 0) um "
            "(no source or port was created)." % name
        )


def _add_ports(fdtd, z_center_um, device_top_um, stack_top_um, silica_cladding_top_um):
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
                port, device_top_um, stack_top_um, silica_cladding_top_um
            ) + distance_um
            direction = "Backward"
        else:
            x_um += distance_um * np.cos(np.deg2rad(actual))
            y_um += distance_um * np.sin(np.deg2rad(actual))
            z_um = z_center_um
        if abs(((actual - nearest + 180.0) % 360.0) - 180.0) > 1e-6:
            print(f"Warning: {name} at {actual:g}° was mapped to the nearest FDTD port axis ({nearest}°).")
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
        fdtd.set("mode selection", str(port.get("mode", "fundamental TE mode")))
        # Force the embedded eigensolver to store modal profiles now. Without
        # this step a programmatically created tilted port can finish an FDTD
        # solve yet expose neither S nor expansion results.
        port_path = "FDTD::ports::" + name
        fdtd.select(port_path)
        mode_update_status = fdtd.updateportmodes()
        if mode_update_status is not None and float(np.asarray(mode_update_status).squeeze()) < 0.0:
            raise RuntimeError("Lumerical could not calculate the selected mode for FDTD port " + name)
        if not bool(fdtd.haveresult(port_path, "mode profiles")):
            raise RuntimeError(
                "FDTD port %s has no mode profile after updateportmodes; enlarge/reposition the port so it crosses its waveguide or fiber core"
                % name
            )
        print("Updated selected modal data for FDTD port " + name)


def _add_monitors(fdtd, z_center_um, device_top_um):
    used_names = set()
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
            z_um = (z_center_um if z_reference == "device center" else device_top_um) + distance_um
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
            fdtd.set("mode selection", str(monitor.get("mode", "fundamental TE mode")))


def _add_grating_analysis_monitor(fdtd, device_top_um, stack_top_um, silica_cladding_top_um):
    """Add the upward near-field plane used for response and far-field projection."""
    if not GRATING_ANALYSIS:
        return
    fdtd.addpower()
    fdtd.set("name", str(GRATING_ANALYSIS["monitor_name"]))
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", float(GRATING_ANALYSIS["center_um"][0]) * UM)
    fdtd.set("y", float(GRATING_ANALYSIS["center_um"][1]) * UM)
    reference_z_um = _vertical_reference_um(
        GRATING_ANALYSIS, device_top_um, stack_top_um, silica_cladding_top_um
    )
    fdtd.set("z", (reference_z_um + float(GRATING_ANALYSIS.get("z_offset_um", 0.25))) * UM)
    fdtd.set("x span", float(GRATING_ANALYSIS["x_span_um"]) * UM)
    fdtd.set("y span", float(GRATING_ANALYSIS["y_span_um"]) * UM)


def build_simulation():
    available_cpu_cores = os.cpu_count() or 1
    build_cpu_threads = max(
        1,
        min(int(SETTINGS.get("build_cpu_threads", 30)), available_cpu_cores),
    )
    fdtd = lumapi.FDTD(
        hide=bool(SETTINGS.get("hide_cad", False)),
        serverArgs={"threads": str(build_cpu_threads)},
    )
    print(
        "Model construction CPU allocation: %d thread%s"
        % (build_cpu_threads, "" if build_cpu_threads == 1 else "s")
    )
    _add_required_materials(fdtd)
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
    boundary_clearance_um = 0.25 * min(
        float(SETTINGS.get("wavelength_start_um", 1.25)),
        float(SETTINGS.get("wavelength_stop_um", 1.35)),
    )
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
    if not z_ranges:
        print("Warning: every stack thickness is zero; no material objects will be added.")
        z_min_um, z_max_um, device_z_um, device_top_um = -1.0, 1.0, 0.0, 0.0
    else:
        z_min_um = z_ranges[0][1]
        z_max_um = z_ranges[-1][2]
        geometry_ranges = [(z0, z1) for row, z0, z1 in z_ranges if row.get("role") == "geometry"]
        device_z_um = sum(geometry_ranges[0]) / 2.0 if geometry_ranges else (z_min_um + z_max_um) / 2.0
        device_top_um = max((z1 for z0, z1 in geometry_ranges), default=device_z_um)
    stack_top_um = z_max_um
    silica_cladding_top_um = _silica_cladding_top_um(z_ranges, device_top_um)

    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
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
    if SETTINGS.get("include_ports", True):
        for port in PORTS:
            if str(port.get("plane normal", "X")).upper() == "Z":
                source_z_um = _vertical_reference_um(
                    port, device_top_um, stack_top_um, silica_cladding_top_um
                ) + float(port.get("distance_um", 0.0))
                z_extent_min_um = min(z_extent_min_um, source_z_um)
                z_extent_max_um = max(z_extent_max_um, source_z_um)
                continue
            plane_normal = str(port.get("plane normal", "")).upper()
            if plane_normal not in {"X", "Y", "Z"}:
                nearest, _, _ = _nearest_port_axis(float(port.get("outward_orientation_deg", 0.0)))
                plane_normal = "X" if nearest in (0, 180) else "Y"
            z_span_um = 0.0 if plane_normal == "Z" else max(0.0, float(port.get("z_span_um", 2.0)))
            z_extent_min_um = min(z_extent_min_um, device_z_um - 0.5 * z_span_um)
            z_extent_max_um = max(z_extent_max_um, device_z_um + 0.5 * z_span_um)
    for monitor in MONITORS:
        geometry_type = str(monitor.get("monitor geometry", "surface")).lower()
        plane_normal = str(monitor.get("plane normal", "")).upper()
        if plane_normal not in {"X", "Y", "Z"}:
            x_span = float(monitor.get("x span", monitor.get("span_um", 4.0)))
            y_span = float(monitor.get("y span", monitor.get("span_um", 4.0)))
            z_span = float(monitor.get("z span", monitor.get("z_span_um", 2.0)))
            plane_normal = min(((abs(x_span), "X"), (abs(y_span), "Y"), (abs(z_span), "Z")))[1]
        if plane_normal == "Z":
            monitor_z_um = device_top_um + float(monitor.get("distance_um", 0.0))
            z_extent_min_um = min(z_extent_min_um, monitor_z_um)
            z_extent_max_um = max(z_extent_max_um, monitor_z_um)
        z_span_um = 0.0
        if geometry_type != "line" and plane_normal != "Z":
            z_span_um = max(0.0, float(monitor.get("z span", monitor.get("z_span_um", 2.0))))
        z_extent_min_um = min(z_extent_min_um, device_z_um - 0.5 * z_span_um)
        z_extent_max_um = max(z_extent_max_um, device_z_um + 0.5 * z_span_um)
    if GRATING_ANALYSIS:
        analysis_z_um = _vertical_reference_um(
            GRATING_ANALYSIS, device_top_um, stack_top_um, silica_cladding_top_um
        ) + float(GRATING_ANALYSIS.get("z_offset_um", 0.25))
        z_extent_min_um = min(z_extent_min_um, analysis_z_um)
        z_extent_max_um = max(z_extent_max_um, analysis_z_um)

    legacy_z_padding = float(SETTINGS.get("z_padding_um", 1.0))
    requested_z_min_padding = float(domain_padding.get("z_min", legacy_z_padding))
    requested_z_max_padding = float(domain_padding.get("z_max", legacy_z_padding))
    # The editor can intentionally shrink the stack into a PML, but sources,
    # ports and monitors must remain strictly inside the solver.  The official
    # grating workflow keeps those sampling planes clear of the boundary.
    z_min_padding = max(boundary_clearance_um, requested_z_min_padding)
    z_max_padding = max(boundary_clearance_um, requested_z_max_padding)
    if z_min_padding != requested_z_min_padding or z_max_padding != requested_z_max_padding:
        print(
            "Adjusted FDTD Z clearance to keep ports/monitors inside the domain: "
            "z min %.6g um, z max %.6g um" % (z_min_padding, z_max_padding)
        )
    simulation_z_min_um = z_extent_min_um - z_min_padding
    simulation_z_max_um = z_extent_max_um + z_max_padding
    if simulation_z_max_um <= simulation_z_min_um:
        raise ValueError("The freely positioned FDTD Z bounds must have a positive span")
    fdtd.set("z min", simulation_z_min_um * UM)
    fdtd.set("z max", simulation_z_max_um * UM)
    fdtd.set("mesh accuracy", int(SETTINGS.get("mesh_accuracy", 2)))
    dt_stability_factor = min(0.99, max(0.1, float(SETTINGS.get("dt_stability_factor", 0.99))))
    fdtd.set("dt stability factor", dt_stability_factor)
    for boundary_property in ("x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"):
        fdtd.set(boundary_property, "PML")
    pml_profile_name = str(SETTINGS.get("pml_profile", "standard")).strip().lower()
    if pml_profile_name not in {"standard", "stabilized"}:
        raise ValueError("pml_profile must be Standard or Stabilized")
    fdtd.set("pml profile", 2 if pml_profile_name == "stabilized" else 1)
    fdtd.set("auto scale pml parameters", False if GRATING_ANALYSIS else True)
    fdtd.set("simulation time", float(SETTINGS.get("simulation_time_fs", 2000.0)) * 1e-15)
    print("FDTD stability: dt factor %.3g, %s PML" % (dt_stability_factor, pml_profile_name))

    if z_ranges:
        _add_material_stack(
            fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um, pml_geometry_overlap_um
        )
        _add_layer_mesh_overrides(fdtd, z_ranges, bounds, simulation_z_min_um, simulation_z_max_um)
        if SETTINGS.get("include_ports", True):
            _add_waveguide_boundary_extensions(fdtd, z_ranges, bounds, boundary_clearance_um, pml_geometry_overlap_um)
    # Port eigensolvers must see the requested wavelength range before their
    # mode profiles are explicitly updated.
    fdtd.setglobalsource("wavelength start", float(SETTINGS.get("wavelength_start_um", 1.25)) * UM)
    fdtd.setglobalsource("wavelength stop", float(SETTINGS.get("wavelength_stop_um", 1.35)) * UM)
    _add_fiber_geometries(fdtd, device_top_um, stack_top_um, silica_cladding_top_um)
    if SETTINGS.get("include_ports", True):
        _add_ports(fdtd, device_z_um, device_top_um, stack_top_um, silica_cladding_top_um)
        if PORTS:
            fdtd.select("FDTD::ports")
            fdtd.set("monitor frequency points", int(SETTINGS.get("frequency_points", 50)))
    _add_monitors(fdtd, device_z_um, device_top_um)
    _add_grating_analysis_monitor(fdtd, device_top_um, stack_top_um, silica_cladding_top_um)
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("frequency points", int(SETTINGS.get("frequency_points", 50)))
    model_bounds_um = [
        float(bounds[0]), float(bounds[1]), float(simulation_z_min_um),
        float(bounds[2]), float(bounds[3]), float(simulation_z_max_um),
    ]
    return fdtd, model_bounds_um


fdtd, MODEL_BOUNDS_UM = build_simulation()
ACTUAL_FDTD_DIMENSION = str(fdtd.getnamed("FDTD", "dimension"))
if ACTUAL_FDTD_DIMENSION.upper() != "3D":
    raise RuntimeError("Max Layout requires a 3D FDTD region, but Lumerical reported " + ACTUAL_FDTD_DIMENSION)
print(f"Built a verified 3D model with {len(GEOMETRY)} polygons, {len(PORTS)} standard FDTD ports, {len(FIBER_GEOMETRIES)} fiber geometry groups, {len(MONITORS)} monitors, and {len(_active_stack(MATERIAL_STACK))} active stack layers.")
'''


_GEOMETRY_PROJECTIONS_REMOTE = r'''# Render the exact embedded XY polygons and their Layer Builder XZ/YZ process projections.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle


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
            draw_z0 = z0
            if bool(row.get("conformal", False)):
                for previous_row, previous_z0, previous_z1 in reversed(z_ranges[:row_index]):
                    if str(previous_row.get("role", "background")) != "geometry":
                        continue
                    previous_thickness = previous_z1 - previous_z0
                    previous_etch = min(
                        previous_thickness,
                        max(0.0, float(previous_row.get("etch_depth_um", previous_thickness))),
                    )
                    draw_z0 = min(z0, previous_z1 - previous_etch)
                    break
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
    for fiber in FIBER_GEOMETRIES:
        x_um, y_um = map(float, fiber.get("center", (0.0, 0.0)))
        start_horizontal = x_um if coordinate_index == 0 else y_um
        start_z = _vertical_reference_um(
            fiber, device_top, stack_top, silica_cladding_top
        ) + float(fiber.get("distance_um", 0.0))
        length = float(fiber.get("fiber length_um", 20.0))
        theta = np.deg2rad(float(fiber.get("angle theta", 10.0)))
        phi = np.deg2rad(float(fiber.get("angle phi", 0.0)))
        horizontal_delta = length * np.sin(theta) * (np.cos(phi) if coordinate_index == 0 else np.sin(phi))
        vertical_delta = length * np.cos(theta)
        axis.plot(
            [start_horizontal, start_horizontal + horizontal_delta],
            [start_z, start_z + vertical_delta],
            color="#0e7490", linewidth=5.0, alpha=0.40, solid_capstyle="round",
        )
        axis.plot(
            [start_horizontal, start_horizontal + horizontal_delta],
            [start_z, start_z + vertical_delta],
            color="#155e75", linewidth=1.4, solid_capstyle="round",
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
    xy_axis.add_patch(plt.Circle((x_um, y_um), cladding_radius, fill=False, edgecolor="#0891b2", linewidth=1.0, linestyle=":"))
    xy_axis.add_patch(plt.Circle((x_um, y_um), core_radius, fill=False, edgecolor="#0e7490", linewidth=1.6))
    xy_axis.annotate(str(fiber.get("name", "fiber geometry")), (x_um, y_um), xytext=(4, 5), textcoords="offset points", fontsize=7, color="#0e7490")

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


_REMOTE_RESOURCE_AND_SAVE = r'''import time

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
elif resource_mode == "CPU":
    fdtd.setresource("FDTD", 1, "active", False)
    fdtd.setresource("FDTD", 2, "device type", "CPU")
    fdtd.setresource("FDTD", 2, "active", True)
    fdtd.setresource("FDTD", 2, "processes", 1)
    fdtd.setresource("FDTD", 2, "threads", TOTAL_CORES)
    print("CPU resource active: 1 process x %d threads" % TOTAL_CORES)
else:
    raise ValueError("resource_mode must be GPU or CPU")

if GRATING_ANALYSIS:
    fdtd.switchtolayout()
    fdtd.select("FDTD::ports")
    fdtd.set("source port", str(GRATING_ANALYSIS["fiber_port_name"]))
    fdtd.set("source mode", "mode 1")
    print("Grating excitation source: FDTD::ports::" + str(GRATING_ANALYSIS["fiber_port_name"]))
    print("Grating excitation direction: Backward along the tilted Z-axis fiber port")
    print("Grating excitation mode: mode 1")
    print("Grating receiver port: FDTD::ports::" + str(GRATING_ANALYSIS["waveguide_port_name"]))
elif MMI_ANALYSIS:
    fdtd.switchtolayout()
    fdtd.select("FDTD::ports")
    fdtd.set("source port", str(MMI_ANALYSIS["input_port_name"]))
    fdtd.set("source mode", "mode 1")
    print("MMI excitation source: FDTD::ports::" + str(MMI_ANALYSIS["input_port_name"]))
    print("MMI excitation mode: mode 1")
    print("MMI output ports:", ", ".join(map(str, MMI_ANALYSIS["output_port_names"])))

_project_name = os.path.basename(str(SETTINGS.get("project_file", "exported_component.fsp")))
if not _project_name.lower().endswith(".fsp"):
    _project_name += ".fsp"
REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")
os.makedirs(REMOTE_FSP_DIR, exist_ok=True)
REMOTE_PROJECT_FILE = os.path.join(REMOTE_FSP_DIR, _project_name)


def save_verified_project():
    """Save to shared storage and do not report success until a non-empty .fsp exists."""
    fdtd.save(REMOTE_PROJECT_FILE)
    for _attempt in range(40):
        if os.path.isfile(REMOTE_PROJECT_FILE) and os.path.getsize(REMOTE_PROJECT_FILE) > 0:
            print("Verified remote project: %s (%d bytes)" % (
                REMOTE_PROJECT_FILE, os.path.getsize(REMOTE_PROJECT_FILE)
            ))
            return REMOTE_PROJECT_FILE
        time.sleep(0.25)
    nearby = sorted(name for name in os.listdir(REMOTE_FSP_DIR) if name.lower().endswith(".fsp"))
    raise RuntimeError(
        "Lumerical save returned without creating %s. Nearby .fsp files: %s"
        % (REMOTE_PROJECT_FILE, nearby)
    )


save_verified_project()
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
    for port in PORTS:
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
            "Mode expansion monitor": ("expansion for port monitor", "mode profiles"),
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
with open(REMOTE_RESULTS_JSON, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "project_file": REMOTE_PROJECT_FILE,
            "simulation_ran": bool(SETTINGS.get("run_after_build", False)),
            "resource_mode": SETTINGS.get("resource_mode"),
            "saved_numeric_keys": sorted(RESULT_ARRAYS),
            "result_notes": RESULT_ERRORS,
            "ports_json": PORTS_JSON,
            "fiber_geometries": FIBER_GEOMETRIES,
            "export_scope": EXPORT_SCOPE_LABEL,
            "exported_components": EXPORTED_COMPONENTS,
            "material_stack": MATERIAL_STACK,
            "grating_analysis": GRATING_ANALYSIS,
            "mmi_analysis": MMI_ANALYSIS,
        },
        stream,
        indent=2,
    )
for required_path in (REMOTE_RESULTS_NPZ, REMOTE_RESULTS_JSON):
    if not os.path.isfile(required_path) or os.path.getsize(required_path) <= 0:
        raise RuntimeError("Required result artifact was not created: " + required_path)
save_verified_project()
print("Saved result bundle:", REMOTE_RESULTS_NPZ)
print("Saved result summary:", REMOTE_RESULTS_JSON)
'''


_GRATING_ANALYSIS_REMOTE = r'''# Waveguide-in grating response and natural upward radiation pattern.
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

if not GRATING_ANALYSIS:
    print("No grating coupler was exported; grating analysis is not required.")
elif not SETTINGS.get("run_after_build", False):
    print("Grating analysis is ready but unsolved. Set SETTINGS['run_after_build'] = True and rerun from section 6.")
else:
    monitor_name = str(GRATING_ANALYSIS["monitor_name"])
    waveguide_port_name = str(GRATING_ANALYSIS["waveguide_port_name"])

    def _one_mode_spectrum(dataset, value_key):
        if "lambda" in dataset:
            wavelength = np.squeeze(np.asarray(dataset["lambda"])).ravel()
        elif "f" in dataset:
            wavelength = 299792458.0 / np.squeeze(np.asarray(dataset["f"])).ravel()
        else:
            raise RuntimeError("The port result has neither lambda nor frequency data")
        values = np.squeeze(np.asarray(dataset[value_key]))
        if values.ndim == 0:
            values = np.full(wavelength.size, values)
        elif values.ndim == 1:
            values = values.ravel()
        else:
            wavelength_axes = [axis for axis, size in enumerate(values.shape) if size == wavelength.size]
            if not wavelength_axes:
                raise RuntimeError(
                    "Could not align port %s data shape %s with %d wavelengths"
                    % (value_key, values.shape, wavelength.size)
                )
            values = np.moveaxis(values, wavelength_axes[0], 0).reshape(wavelength.size, -1)[:, 0]
        if values.size != wavelength.size:
            raise RuntimeError(
                "Port %s returned %d values for %d wavelengths"
                % (value_key, values.size, wavelength.size)
            )
        return wavelength, values

    # Match the official Ansys 3D grating-coupler analysis exactly:
    # T_data = getresult("::model::FDTD::ports::port 2", "T");
    # T = abs(T_data.T); lambda = T_data.lambda;
    receiver_port_path = "::model::FDTD::ports::" + waveguide_port_name
    try:
        T_data = fdtd.getresult(receiver_port_path, "T")
    except Exception as exc:
        raise RuntimeError(
            "The official receiver-port T result is unavailable at %r. "
            "The port must be inside the FDTD region and the simulation must finish before plotting. %s"
            % (receiver_port_path, exc)
        ) from None
    wavelengths_m, fiber_coupling = _one_mode_spectrum(T_data, "T")
    fiber_coupling = np.abs(fiber_coupling)
    print("Grating coupling uses the official Ansys definition: T = abs(T_data.T).")
    order = np.argsort(wavelengths_m)
    wavelengths_m = wavelengths_m[order]
    fiber_coupling = fiber_coupling[order]
    fiber_coupling_db = 10.0 * np.log10(np.maximum(fiber_coupling, 1e-15))

    upward_response = fdtd.getresult(monitor_name, "T")
    upward_wavelengths_m = np.squeeze(np.asarray(upward_response["lambda"])).ravel()
    upward_power = np.abs(np.squeeze(np.asarray(upward_response["T"])).ravel())
    upward_order = np.argsort(upward_wavelengths_m)
    upward_wavelengths_m = upward_wavelengths_m[upward_order]
    upward_power = upward_power[upward_order]
    wavelength_target_m = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    ) * 1e-6
    frequency_index = int(np.argmin(np.abs(upward_wavelengths_m - wavelength_target_m))) + 1

    response_png = os.path.join(REMOTE_WORK, "grating_response.png")
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(wavelengths_m / 1e-9, fiber_coupling_db, lw=2.0, color="#2563eb")
    plt.axvline(wavelength_target_m / 1e-9, color="#dc2626", ls="--", lw=1.0)
    plt.xlabel("wavelength [nm]")
    plt.ylabel("coupling efficiency [dB]")
    plt.title("Grating coupler — fiber port to waveguide port")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(response_png, dpi=160, bbox_inches="tight")
    plt.close()

    analysis_arrays = {
        "wavelength_m": wavelengths_m,
        "fiber_coupling": fiber_coupling,
        "fiber_coupling_db": fiber_coupling_db,
        "upward_wavelength_m": upward_wavelengths_m,
        "upward_power": upward_power,
        "target_wavelength_m": np.asarray([wavelength_target_m]),
    }
    farfield_png = os.path.join(REMOTE_WORK, "grating_farfield.png")
    try:
        resolution = 361
        farfield = np.abs(np.squeeze(np.asarray(
            fdtd.farfield3d(monitor_name, frequency_index, resolution, resolution, 1, 1, 1, 1)
        )))
        ux = np.squeeze(np.asarray(
            fdtd.farfieldux(monitor_name, frequency_index, resolution, resolution, 1)
        )).ravel()
        uy = np.squeeze(np.asarray(
            fdtd.farfielduy(monitor_name, frequency_index, resolution, resolution, 1)
        )).ravel()
        ux_grid, uy_grid = np.meshgrid(ux, uy, indexing="xy")
        if farfield.shape != ux_grid.shape and farfield.T.shape == ux_grid.shape:
            farfield = farfield.T
        radial = np.sqrt(ux_grid**2 + uy_grid**2)
        valid = radial <= 1.0
        projected = np.where(valid, farfield, np.nan)
        peak_flat = int(np.nanargmax(projected))
        peak_row, peak_col = np.unravel_index(peak_flat, projected.shape)
        ux_peak = float(ux_grid[peak_row, peak_col])
        uy_peak = float(uy_grid[peak_row, peak_col])
        theta_peak_deg = float(np.degrees(np.arcsin(np.clip(np.hypot(ux_peak, uy_peak), 0.0, 1.0))))
        phi_peak_deg = float(np.degrees(np.arctan2(uy_peak, ux_peak)))
        print("Natural grating exit: theta = %.3f deg from +Z, phi = %.3f deg" % (theta_peak_deg, phi_peak_deg))

        normalized = projected / np.nanmax(projected)
        theta_grid_rad = np.arcsin(np.clip(radial, 0.0, 1.0))
        theta_grid_deg = np.degrees(theta_grid_rad)
        phi_grid_rad = np.arctan2(uy_grid, ux_grid)

        # Plot the full upper hemisphere in true polar coordinates: azimuth phi
        # is the angular coordinate and elevation from +Z, theta, is the radius.
        plot_stride = max(1, resolution // 180)
        plot_valid = valid[::plot_stride, ::plot_stride]
        plot_phi = phi_grid_rad[::plot_stride, ::plot_stride][plot_valid]
        plot_theta = theta_grid_deg[::plot_stride, ::plot_stride][plot_valid]
        plot_intensity = normalized[::plot_stride, ::plot_stride][plot_valid]
        triangulation = mtri.Triangulation(plot_phi, plot_theta)
        seam_triangles = np.ptp(plot_phi[triangulation.triangles], axis=1) > np.pi
        triangulation.set_mask(seam_triangles)

        fig, axis = plt.subplots(figsize=(8.0, 7.2), subplot_kw={"projection": "polar"})
        levels = np.linspace(0.0, 1.0, 41)
        image = axis.tricontourf(triangulation, plot_intensity, levels=levels, cmap="magma")
        axis.plot(np.radians(phi_peak_deg), theta_peak_deg,
                  marker="x", ms=10, mew=2, color="cyan", label="natural peak")
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(0.0, 90.0)
        axis.set_rticks([0, 15, 30, 45, 60, 75, 90])
        axis.set_rlabel_position(135)
        axis.set_xlabel("azimuth φ; radius θ from +Z [deg]", labelpad=18)
        axis.set_title("Natural grating radiation pattern at %.1f nm" % (wavelength_target_m / 1e-9))
        axis.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10))
        fig.colorbar(image, ax=axis, label="normalized intensity")
        fig.tight_layout()
        fig.savefig(farfield_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        analysis_arrays.update({
            "farfield_intensity": farfield,
            "ux": ux,
            "uy": uy,
            "theta_deg": theta_grid_deg,
            "phi_deg": np.degrees(phi_grid_rad),
            "theta_peak_deg": np.asarray([theta_peak_deg]),
            "phi_peak_deg": np.asarray([phi_peak_deg]),
        })
    except Exception as exc:
        warning = "Far-field projection unavailable: " + str(exc)[:240]
        print(warning)
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.axis("off")
        axis.text(0.5, 0.55, "Far-field projection unavailable", ha="center", va="center", fontsize=15, fontweight="bold")
        axis.text(0.5, 0.40, str(exc)[:220], ha="center", va="center", fontsize=9, wrap=True)
        fig.tight_layout()
        fig.savefig(farfield_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        analysis_arrays["farfield_error"] = np.asarray([str(exc)[:500]])

    grating_npz = os.path.join(REMOTE_WORK, "grating_analysis.npz")
    np.savez_compressed(grating_npz, **analysis_arrays)
    for required_path in (response_png, farfield_png, grating_npz):
        if not os.path.isfile(required_path):
            raise RuntimeError("Required grating artifact was not created: " + required_path)
    print("Saved grating response:", response_png)
    print("Saved grating far field:", farfield_png)
    print("Saved grating analysis:", grating_npz)
'''


_MMI_ANALYSIS_REMOTE = r'''# Mode-1 input to the two MMI output waveguides.
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
    imbalance_db = 10.0 * np.log10(
        np.maximum(output_1_power, 1e-15) / np.maximum(output_2_power, 1e-15)
    )
    output_1_db = 10.0 * np.log10(np.maximum(output_1_over_input, 1e-15))
    output_2_db = 10.0 * np.log10(np.maximum(output_2_over_input, 1e-15))

    target_wavelength_m = 0.5 * (
        float(SETTINGS.get("wavelength_start_um", 1.25))
        + float(SETTINGS.get("wavelength_stop_um", 1.35))
    ) * 1e-6
    target_index = int(np.argmin(np.abs(wavelength_m - target_wavelength_m)))
    print(
        "MMI split at %.3f nm: Pin %.6g, %s %.3f%%, %s %.3f%%, symmetry error %.4f dB, total/Pin %.3f%%"
        % (
            wavelength_m[target_index] * 1e9,
            input_power[target_index],
            output_labels[0], output_1_ratio[target_index] * 100.0,
            output_labels[1], output_2_ratio[target_index] * 100.0,
            imbalance_db[target_index], total_output_over_input[target_index] * 100.0,
        )
    )
    symmetry_error_percent = abs(output_1_ratio[target_index] - 0.5) * 100.0
    symmetry_tolerance_percent = float(MMI_ANALYSIS.get("symmetry_tolerance_percent", 1.0))
    if symmetry_error_percent <= symmetry_tolerance_percent:
        print("Verified symmetric 50/50 MMI within %.3f percentage points." % symmetry_error_percent)
    else:
        print(
            "WARNING: symmetric MMI differs from 50/50 by %.3f percentage points; check mesh and port placement."
            % symmetry_error_percent
        )

    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
    axes[0].plot(wavelength_m * 1e9, output_1_ratio * 100.0, lw=2.0, label=output_labels[0])
    axes[0].plot(wavelength_m * 1e9, output_2_ratio * 100.0, lw=2.0, label=output_labels[1])
    axes[0].axhline(50.0, color="#64748b", ls="--", lw=1.0, label="ideal 50/50")
    axes[0].set_ylabel("normalized output power [%]")
    axes[0].set_title("MMI splitting ratio — mode 1 input")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(wavelength_m * 1e9, output_1_db, lw=2.0, label=output_labels[0])
    axes[1].plot(wavelength_m * 1e9, output_2_db, lw=2.0, label=output_labels[1])
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("output / measured input [dB]")
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
    try:
        field_result = fdtd.getresult("::model::" + field_monitor_name, "E")
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

    def _field_plane(component_name):
        values = np.asarray(field_result.get(component_name, 0.0))
        if field_frequency_hz.size and values.ndim:
            frequency_axes = [axis for axis, size in enumerate(values.shape) if size == field_frequency_hz.size]
            if frequency_axes:
                values = np.take(values, field_frequency_index, axis=frequency_axes[-1])
        values = np.squeeze(values)
        if values.shape == (field_y_m.size, field_x_m.size):
            values = values.T
        if values.shape != (field_x_m.size, field_y_m.size):
            values = values.reshape(field_x_m.size, field_y_m.size)
        return values

    field_intensity = (
        np.abs(_field_plane("Ex")) ** 2
        + np.abs(_field_plane("Ey")) ** 2
        + np.abs(_field_plane("Ez")) ** 2
    )
    field_peak = max(float(np.nanmax(field_intensity)), 1e-30)
    field_intensity_normalized = field_intensity / field_peak
    field_wavelength_m = (
        299792458.0 / field_frequency_hz[field_frequency_index]
        if field_frequency_hz.size else target_wavelength_m
    )
    field_figure, field_axis = plt.subplots(figsize=(11.0, 4.8))
    field_image = field_axis.pcolormesh(
        field_x_m * 1e6,
        field_y_m * 1e6,
        field_intensity_normalized.T,
        shading="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    field_axis.set_aspect("equal", adjustable="box")
    field_axis.set_xlabel("x [um]")
    field_axis.set_ylabel("y [um]")
    field_axis.set_title("MMI fundamental-mode field |E|² at %.3f nm" % (field_wavelength_m * 1e9))
    field_figure.colorbar(field_image, ax=field_axis, label="normalized |E|²")
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
        output_1_over_input=output_1_over_input,
        output_2_over_input=output_2_over_input,
        total_output_over_input=total_output_over_input,
        imbalance_db=imbalance_db,
        total_output_power=total_output_power,
        target_wavelength_m=np.asarray([target_wavelength_m]),
        field_x_m=field_x_m,
        field_y_m=field_y_m,
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


_FETCH_RESULTS_CELL = r'''# Fetch every verified artifact before closing Lumerical or returning the HPC Packs.
REMOTE_ARTIFACTS = [
    REMOTE_PROJECT_FILE,
    REMOTE_WORK + "/geometry_xyz_projections.png",
    REMOTE_WORK + "/max_layout_results.npz",
    REMOTE_WORK + "/max_layout_results.json",
]
if GRATING_ANALYSIS and SETTINGS.get("run_after_build", False):
    REMOTE_ARTIFACTS.extend([
        REMOTE_WORK + "/grating_response.png",
        REMOTE_WORK + "/grating_farfield.png",
        REMOTE_WORK + "/grating_analysis.npz",
    ])
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
for remote_path in REMOTE_ARTIFACTS:
    if not REMOTE_ARTIFACT_STATUS.get(remote_path, False):
        MISSING_REMOTE_ARTIFACTS.append(remote_path)
        print("ERROR — remote stage did not create:", remote_path)
        continue
    local_directory = PIRIS_FSP_DIR if remote_path == REMOTE_PROJECT_FILE else PIRIS_RESULTS_DIR
    local_path = local_directory / os.path.basename(remote_path)
    try:
        fetched = lam.fetch(remote_path, str(local_path))
        FETCHED_RESULTS.append(fetched)
        print("saved ->", fetched)
    except Exception as exc:
        print("TRANSFER ERROR —", os.path.basename(remote_path), str(exc)[:200])
if MISSING_REMOTE_ARTIFACTS:
    print("One or more required artifacts are missing. Keep the output above and run the licence-release cell next.")
'''


_RELEASE_LICENSES_CELL = r'''# Release in the reverse order of acquisition: FDTD, roamed HPC Packs, SSH.
try:
    lam.run("try:\n    fdtd.close()\nexcept Exception:\n    pass", timeout=90)
finally:
    _release = subprocess.run(_SSH + [HOST,
        f'{LIC}/LicensingSettings web shared products checkin '
        '--name "Ansys HPC Pack - Shared Web" --count 3 --mode user'],
        capture_output=True, text=True, timeout=180)
    _release_out = (_release.stdout + _release.stderr).strip()
    print("HPC Packs:", "3 returned to Shared Web" if "SUCCESS" in _release_out else _release_out[:400])
    lam.close()
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
    geometry, ports, fiber_geometries, monitors, bbox, warnings = _collect_export_data(components, included)
    if not bool(configuration.get("include_ports", True)):
        ports = []
    _synchronize_fiber_port_parameters(ports, fiber_geometries, warnings)

    grating_analysis: dict[str, Any] | None = None
    grating_components = [component for component in components if str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"}]
    if grating_components:
        grating = grating_components[0]
        grating_uid = int(grating.get("uid", 0))
        if len(grating_components) > 1:
            warnings.append("Multiple grating couplers were exported; natural-radiation analysis uses the first one only.")
        grating_polygons = [item for item in geometry if int(item.get("component_uid", -1)) == grating_uid]
        waveguide_ports = [
            port for port in ports
            if bool(port.get("enabled", True))
            and str(port.get("domain", "optical")).lower() == "optical"
            and str(port.get("plane normal", "X")).upper() != "Z"
        ]
        matching_waveguide_ports = [
            port for port in waveguide_ports
            if int(port.get("parent_component_uid", -1)) == grating_uid
        ]
        if matching_waveguide_ports:
            waveguide_ports = matching_waveguide_ports
        waveguide_ports.sort(key=lambda port: (float(port.get("order", 9999)), str(port.get("name", ""))))
        fiber_ports = [
            port for port in ports
            if bool(port.get("enabled", True))
            and str(port.get("domain", "optical")).lower() == "optical"
            and str(port.get("plane normal", "X")).upper() == "Z"
        ]
        matching_fiber_ports = [
            port for port in fiber_ports
            if int(port.get("parent_component_uid", -1)) == grating_uid
        ]
        if matching_fiber_ports:
            fiber_ports = matching_fiber_ports
        if not grating_polygons:
            warnings.append("Grating analysis was not added because no polygons from the grating coupler were selected.")
        elif not waveguide_ports:
            warnings.append(
                "Grating analysis was not added because no manually placed waveguide FDTD port was exported. "
                "Add one from Ports & monitors at the waveguide end."
            )
        elif not fiber_ports:
            warnings.append(
                "Grating analysis was not added because no manually placed Ansys-style fiber port was exported. "
                "Add one from Ports & monitors above the grating exit."
            )
        elif not fiber_geometries:
            warnings.append(
                "Grating analysis was not added because no separate fiber geometry group was exported. "
                "Place the Ansys fiber geometry group and put the Fiber-axis FDTD port through its core/cladding."
            )
        else:
            points = np.vstack([np.asarray(item["vertices_um"], dtype=float) for item in grating_polygons])
            minimum, maximum = points.min(axis=0), points.max(axis=0)
            xy_margin_um = max(0.5, float(configuration.get("xy_padding_um", 1.0)))
            z_clearance_um = max(0.1, float(configuration.get("z_padding_um", 1.0)))
            waveguide_receiver_port = deepcopy(waveguide_ports[0])
            exit_center = [float(0.5 * (minimum[0] + maximum[0])), float(0.5 * (minimum[1] + maximum[1]))]
            exit_x_span = float(maximum[0] - minimum[0] + 2.0 * xy_margin_um)
            exit_y_span = float(maximum[1] - minimum[1] + 2.0 * xy_margin_um)
            grating_center = np.asarray(exit_center, dtype=float)
            fiber_port = min(
                fiber_ports,
                key=lambda port: float(
                    np.linalg.norm(np.asarray(port.get("center", (0.0, 0.0)), dtype=float) - grating_center)
                ),
            )
            exit_center = [float(value) for value in fiber_port.get("center", exit_center)]
            fiber_span = max(1.0, float(fiber_port.get("span_um", 20.0)))
            exit_x_span = fiber_span
            exit_y_span = fiber_span
            fiber_port_name = str(fiber_port.get("name", "fiber"))
            fiber_geometry = min(
                fiber_geometries,
                key=lambda fiber: float(
                    np.linalg.norm(
                        np.asarray(fiber.get("center", (0.0, 0.0)), dtype=float)
                        - np.asarray(fiber_port.get("center", (0.0, 0.0)), dtype=float)
                    )
                ),
            )
            fiber_alignment_error_um = float(
                np.linalg.norm(
                    np.asarray(fiber_geometry.get("center", (0.0, 0.0)), dtype=float)
                    - np.asarray(fiber_port.get("center", (0.0, 0.0)), dtype=float)
                )
            )
            if fiber_alignment_error_um > 0.5 * float(fiber_geometry.get("core diameter_um", 9.0)):
                warnings.append(
                    "Grating analysis was not added because the Fiber-axis FDTD port does not pass through "
                    f"the selected fiber core (top-view separation {fiber_alignment_error_um:.6g} µm)."
                )
            else:
                grating_analysis = {
                    "component_uid": grating_uid,
                    "waveguide_port_name": str(waveguide_receiver_port.get("name", f"gc_receiver_uid_{grating_uid}")),
                    "fiber_port_name": fiber_port_name,
                    "fiber_geometry_name": str(fiber_geometry.get("name", "fiber")),
                    "monitor_name": f"grating_up_uid_{grating_uid}",
                    "center_um": exit_center,
                    "x_span_um": exit_x_span,
                    "y_span_um": exit_y_span,
                    "z_offset_um": float(min(0.25, 0.5 * z_clearance_um)),
                    "z reference": str(fiber_port.get("z reference", "device top")),
                    "frequency_points": 50,
                }

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
                    "field_monitor_name": str(field_monitor.get("name", f"uid_{mmi_uid}_mmi_field")),
                    "input_reference_before_taper_um": float(
                        mmi.get("params", {}).get("input_reference_before_taper_um", 2.0)
                    ),
                    "output_port_names": [
                        str(upper_port.get("name", "mmi_upper")),
                        str(lower_port.get("name", "mmi_lower")),
                    ],
                    "output_labels": ["upper output", "lower output"],
                    "ideal_split_percent": [50.0, 50.0],
                    "symmetry_tolerance_percent": 1.0,
                    "frequency_points": int(configuration.get("frequency_points", 50)),
                }
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
        row["mesh_factor"] = max(0.0, float(row.get("mesh_factor", 0.1)))
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
        "simulation_time_fs": float(configuration.get("simulation_time_fs", 2000.0)),
        "frequency_points": int(configuration.get("frequency_points", 50)),
        "build_cpu_threads": max(1, int(configuration.get("build_cpu_threads", 30))),
        "resource_mode": configuration.get("resource_mode", "GPU"),
        "tfln_crystal_cut": str(configuration.get("tfln_crystal_cut", "X")).strip().upper(),
        "tfln_temperature_K": float(configuration.get("tfln_temperature_K", 296.3)),
        "include_ports": bool(configuration.get("include_ports", True)),
        "hide_cad": bool(configuration.get("hide_cad", False)),
        "run_after_build": bool(configuration.get("run_after_build", False)),
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
                "simulation_time_fs": 2000.0,
                "frequency_points": 50,
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
    payload_cell = (
        "# Embedded export data (layout units are micrometres).\n"
        f"EXPORT_SCOPE_LABEL = {export_scope_label!r}\n"
        f"EXPORTED_COMPONENTS = {pprint.pformat(exported_components, width=120, sort_dicts=False)}\n"
        f"SETTINGS = {pprint.pformat(settings, width=120, sort_dicts=False)}\n"
        f"MATERIAL_STACK = {pprint.pformat(stack, width=120, sort_dicts=False)}\n"
        f"BOUNDING_BOX_UM = {pprint.pformat(bbox)}\n"
        f"GEOMETRY = {pprint.pformat(geometry, width=160, compact=True, sort_dicts=False)}\n"
        f"PORTS = {pprint.pformat(ports, width=120, sort_dicts=False)}\n"
        f"FIBER_GEOMETRIES = {pprint.pformat(fiber_geometries, width=120, sort_dicts=False)}\n"
        f"PORTS_JSON = {pprint.pformat(ports_json, width=120, sort_dicts=False)}\n"
        f"MONITORS = {pprint.pformat(monitors, width=120, sort_dicts=False)}\n"
        f"GRATING_ANALYSIS = {pprint.pformat(grating_analysis, width=120, sort_dicts=False)}\n"
        f"MMI_ANALYSIS = {pprint.pformat(mmi_analysis, width=120, sort_dicts=False)}\n"
        f"EXPORT_WARNINGS = {pprint.pformat(warnings, width=120)}\n"
        "for warning in EXPORT_WARNINGS:\n"
        "    print('Export note:', warning)\n"
    )
    remote_builder_source = "import os\nimport json\nimport numpy as np\nimport lumapi\n" + _BUILD_CELL
    remote_build_cell = (
        "# Send the complete self-contained model to the already licensed persistent Lambda session.\n"
        f"REMOTE_MODEL_BUILDER = {repr(remote_builder_source)}\n"
        "_remote_payload = (\n"
        "    'REMOTE_WORK = ' + repr(REMOTE_WORK) + '\\n'\n"
        "    + 'EXPORT_SCOPE_LABEL = ' + repr(EXPORT_SCOPE_LABEL) + '\\n'\n"
        "    + 'EXPORTED_COMPONENTS = ' + repr(EXPORTED_COMPONENTS) + '\\n'\n"
        "    + 'SETTINGS = ' + repr(SETTINGS) + '\\n'\n"
        "    + 'MATERIAL_STACK = ' + repr(MATERIAL_STACK) + '\\n'\n"
        "    + 'BOUNDING_BOX_UM = ' + repr(BOUNDING_BOX_UM) + '\\n'\n"
        "    + 'GEOMETRY = ' + repr(GEOMETRY) + '\\n'\n"
        "    + 'PORTS = ' + repr(PORTS) + '\\n'\n"
        "    + 'FIBER_GEOMETRIES = ' + repr(FIBER_GEOMETRIES) + '\\n'\n"
        "    + 'PORTS_JSON = ' + repr(PORTS_JSON) + '\\n'\n"
        "    + 'MONITORS = ' + repr(MONITORS) + '\\n'\n"
        "    + 'GRATING_ANALYSIS = ' + repr(GRATING_ANALYSIS) + '\\n'\n"
        "    + 'MMI_ANALYSIS = ' + repr(MMI_ANALYSIS) + '\\n'\n"
        "    + 'EXPORT_WARNINGS = ' + repr(EXPORT_WARNINGS) + '\\n'\n"
        ")\n"
        "run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER, 'Build verified 3D model', timeout=1800)\n"
    )
    geometry_projection_cell = (
        "# Build and display XY, XZ, and YZ projections before committing compute time.\n"
        f"REMOTE_GEOMETRY_PROJECTIONS = {repr(_GEOMETRY_PROJECTIONS_REMOTE)}\n"
        "run_remote_checked(REMOTE_GEOMETRY_PROJECTIONS, 'Render 3-axis geometry verification', timeout=1800)\n"
        "GEOMETRY_PROJECTIONS_FILE = REMOTE_WORK.rstrip('/') + '/geometry_xyz_projections.png'\n"
        "lam.show(GEOMETRY_PROJECTIONS_FILE, width=1400)\n"
    )
    resource_save_cell = (
        "# Configure the licensed resource and download a verified pre-solve FSP. This cell does not solve.\n"
        f"REMOTE_RESOURCE_AND_SAVE = {repr(_REMOTE_RESOURCE_AND_SAVE)}\n"
        "run_remote_checked(REMOTE_RESOURCE_AND_SAVE, 'Configure resources and save project', timeout=1800)\n"
        "REMOTE_PROJECT_FILE = lam.get('REMOTE_PROJECT_FILE')\n"
        "PIRIS_FSP_DIR.mkdir(parents=True, exist_ok=True)\n"
        "LOCAL_PROJECT_FILE = PIRIS_FSP_DIR / os.path.basename(REMOTE_PROJECT_FILE)\n"
        "FETCHED_PROJECT_FILE = lam.fetch(REMOTE_PROJECT_FILE, str(LOCAL_PROJECT_FILE))\n"
        "if not LOCAL_PROJECT_FILE.is_file() or LOCAL_PROJECT_FILE.stat().st_size <= 0:\n"
        "    raise RuntimeError('The verified remote .fsp could not be downloaded before solving: ' + str(LOCAL_PROJECT_FILE))\n"
        "print('saved pre-solve project ->', FETCHED_PROJECT_FILE)\n"
    )
    review_project_cell = (
        "# Inspect the exact pre-solve FSP before committing GPU time.\n"
        "from IPython.display import FileLink, display\n"
        "display(FileLink(str(LOCAL_PROJECT_FILE)))\n"
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
        "# Run only after reviewing the downloaded FSP. GPU remains the default for every 3D solve.\n"
        "if SETTINGS.get('run_after_build', False):\n"
        "    _resource_mode = str(SETTINGS.get('resource_mode', 'GPU')).strip().upper()\n"
        "    if _resource_mode == 'GPU':\n"
        "        _solve_code = 'fdtd.run(\"FDTD\", \"GPU\")'\n"
        "    elif _resource_mode == 'CPU':\n"
        "        _solve_code = 'fdtd.run(\"FDTD\", \"CPU\")'\n"
        "    else:\n"
        "        raise ValueError('resource_mode must be GPU or CPU')\n"
        "    solve_remote_checked(_solve_code, label='Max Layout 3D FDTD [' + _resource_mode + ']', timeout=21600)\n"
        "    run_remote_checked('save_verified_project(); print(\"Simulation finished and project re-saved.\")', 'Re-save solved project', timeout=300)\n"
        "else:\n"
        "    print(\"Run is disabled. The reviewed pre-solve .fsp is preserved and will still be fetched.\")\n"
    )
    save_results_cell = (
        "# Serialize model results while the FDTD licence and remote session are still active.\n"
        f"REMOTE_RESULTS_SAVER = {repr(_SAVE_REMOTE_RESULTS)}\n"
        "run_remote_checked(REMOTE_RESULTS_SAVER, 'Save project and numerical result bundle', timeout=1800)\n"
    )
    grating_analysis_cell = (
        "# Plot the grating response and its natural far-field radiation before saving/fetching results.\n"
        f"REMOTE_GRATING_ANALYSIS = {repr(_GRATING_ANALYSIS_REMOTE)}\n"
        "run_remote_checked(REMOTE_GRATING_ANALYSIS, 'Grating response and far-field analysis', timeout=1800)\n"
        "if SETTINGS.get('run_after_build', False):\n"
        "    lam.show(REMOTE_WORK + '/grating_response.png', width=1000)\n"
        "    lam.show(REMOTE_WORK + '/grating_farfield.png', width=900)\n"
    )
    mmi_analysis_cell = (
        "# Plot the symmetric MMI splitting ratio and longitudinal fundamental-mode field.\n"
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
    intro = f"""# Max Layout → Lumerical FDTD notebook

This notebook contains **{len(geometry)} embedded polygons**, **{len(ports)} standard FDTD ports**, **{len(fiber_geometries)} fiber geometry groups**, **{len(monitors)} monitors**, and **{active_count} active material layers**. It is self-contained: no GDS sidecar is required.

**Export scope:** {export_scope_label}  
**Included objects:** {exported_component_text}
{empty_geometry_warning}

The notebook follows the same licence lifecycle as `TFLN_GC_1310.ipynb`: connect to Lambda, seed Ansys Shared Web, roam exactly three HPC Packs, build/run in that licensed session, save and fetch results, close FDTD, return the three packs, and finally close SSH. Lumerical is not required on the local Mac.

- Every exported simulation is a **3D FDTD simulation**. A saved 2D preference is ignored; GPU is the default compute resource and is selected explicitly at solve time.
- Before solving, the notebook renders and displays **XY, XZ, and YZ geometry projections** for visual verification. This common stage is included for every component type and full-layout export.
- Stack rows are ordered bottom-to-top.
- A material thickness of **0 µm means that material is absent**.
- Etch depth **0 µm** keeps an unetched film; etch depth equal to film thickness creates a fully etched patterned layer.
- Exported cross-section rows use Lumerical Layer Builder, including the selected waveguide sidewall angle (90° is vertical).
- A partially etched cross-section can keep its unetched slab across the full FDTD plane or restrict it to the selected GDS geometry footprint.
- Each stack row has a dimensionless mesh factor. The default **0.1** produces an isotropic step of `0.1 × λ₀/nmax` at the shortest simulated wavelength; anisotropic media use their largest index component.
- Surface monitors carry explicit x/y/z spans; their normal-axis span is zero (Z is the into-page axis in the layout view).
- Ports are manual simulation-only objects from the left **Ports & monitors** library. No component, including a grating coupler, creates a default port automatically.
- `PORTS_JSON` uses the exact compact-model structure and field names from the reviewed Lumerical JSON examples: `name`, `dir`, `loc`, `pos`, and `order`.
- Every port becomes a standard Ansys FDTD `addport` object. Fiber is a separate geometry group containing the official example's tilted 9 µm core and 50 µm cladding; a manually placed Z-axis FDTD port passes through that geometry.
- Placed power, mode-expansion, and field-profile monitors become `addpower`, `addmodeexpansion`, and `addprofile` objects.
- The fiber geometry and its Z-axis FDTD port have independent positions and heights above the exported device. Grating-coupler exit analysis is centered on the placed fiber-axis FDTD port.
- A conformal cladding row fills etched openings in the patterned layer immediately below it and then covers the device.
- The FDTD boundary keeps at least λ/4 clearance from ordinary device features. Bottom and top background films extend through their PMLs, and each manually placed lateral waveguide port receives a cross-section-matched continuation through its PML, following the official Ansys example.
- `LiNbO3` is created as a frequency- and temperature-dependent anisotropic sampled material using the Zelmon/Moretti model and the selected X/Y/Z crystal cut.
- Grating-coupler exports follow the official Ansys 3D example: the tilted Z-axis fiber FDTD port is the Backward source and the waveguide FDTD port is the receiver. They plot coupling efficiency in dB versus wavelength and the 3D far field in polar coordinates.
- A 1×2 MMI export launches mode 1 from its input port, measures input power 2 µm before the input taper, plots both output powers relative to that measured input, and plots the normalized longitudinal |E|² distribution through the complete MMI.
- GPU and CPU modes are selectable through `SETTINGS['resource_mode']`; GPU is the default for every 3D export.
- Run the final release cell even after an interrupted simulation so the FDTD licence and roamed HPC Packs are returned.
"""
    notebook = {
        "cells": [
            _notebook_cell("code", _PIRIS_PATHS_CELL),
            _notebook_cell("markdown", intro),
            _notebook_cell("markdown", "## 1 · Connect to Lambda\n"),
            _notebook_cell("code", _LAMBDA_CONNECT_CELL),
            _notebook_cell("markdown", "## 2 · Acquire Ansys Shared Web licences\n\nSeed the headless sign-in and roam exactly three Shared Web HPC Packs for this session.\n"),
            _notebook_cell("code", _LICENSE_CHECKOUT_CELL),
            _notebook_cell("markdown", "## 3 · Embedded layout, stack, ports, and monitors\n"),
            _notebook_cell("code", payload_cell),
            _notebook_cell("markdown", "## 4 · Build the model inside the licensed Lambda session\n"),
            _notebook_cell("code", remote_build_cell),
            _notebook_cell("markdown", "## 5 · Verify the built geometry in XY, XZ, and YZ\n\nThese three projections are generated from the exact polygons and process-stack values passed to the 3D Lumerical model.\n"),
            _notebook_cell("code", geometry_projection_cell),
            _notebook_cell("markdown", "## 6 · Configure resources and save the pre-solve FSP\n\nThis always downloads the exact model into the project's `fsp` folder before any solve begins.\n"),
            _notebook_cell("code", resource_save_cell),
            _notebook_cell("markdown", "## 7 · Inspect the FSP before solving\n\nUse the file link below to open the model in Lumerical FDTD. Lambda normally runs headless; an explicit remote-GUI switch is provided but stays off unless a display is available.\n"),
            _notebook_cell("code", review_project_cell),
            _notebook_cell("markdown", "## 8 · Run the reviewed 3D model\n"),
            _notebook_cell("code", solve_cell),
            *(
                [
                    _notebook_cell("markdown", "## 9 · Grating coupling efficiency and far-field radiation\n\nFollowing the official Ansys 3D grating example, the tilted fiber-side Z port injects Backward toward the chip and the waveguide FDTD port receives the coupled power. The upward monitor is also projected to a polar far-field plot.\n"),
                    _notebook_cell("code", grating_analysis_cell),
                ]
                if grating_analysis
                else []
            ),
            *(
                [
                    _notebook_cell("markdown", "## 9 · Symmetric MMI splitting ratio and field distribution\n\nMode 1 is launched from the input FDTD port. A power monitor 2 µm before the input taper measures the actual incident power, the two output-port powers are reported relative to that reference, and a Z-normal monitor plots normalized |E|² along the complete MMI length. The expected split is 50/50.\n"),
                    _notebook_cell("code", mmi_analysis_cell),
                ]
                if mmi_analysis
                else []
            ),
            _notebook_cell("markdown", "## 10 · Save numerical results before releasing licences\n"),
            _notebook_cell("code", save_results_cell),
            _notebook_cell("markdown", "## 11 · Fetch the verified project, geometry image, and result bundle\n"),
            _notebook_cell("code", _FETCH_RESULTS_CELL),
            _notebook_cell("markdown", "## 12 · Release FDTD and return all roamed HPC Packs\n\nAlways run this cell, including after an interrupted solve.\n"),
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


def write_lumerical_notebook(
    path: str | Path,
    components: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> list[str]:
    """Always write a notebook; questionable settings are recorded as warnings."""
    notebook, warnings = generate_lumerical_notebook(components, configuration)
    Path(path).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return warnings
