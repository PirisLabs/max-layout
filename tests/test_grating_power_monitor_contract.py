from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from max_layout import lumerical
from max_layout.lumerical_optimization import (
    generate_lumerical_adjoint_notebook,
    normalize_lumerical_optimization_spec,
)
from max_layout.ui.window import NativeLayoutWindow


def _assignment_value(notebook: dict, name: str):
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for node in ast.parse("".join(cell.get("source", []))).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"No notebook assignment named {name!r}")


def _all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


class _Factory:
    make_component = NativeLayoutWindow.make_component
    automatic_simulation_companions = (
        NativeLayoutWindow.automatic_simulation_companions
    )
    synchronize_automatic_simulation_companions = (
        NativeLayoutWindow.synchronize_automatic_simulation_companions
    )

    def __init__(self) -> None:
        self.components: list[dict] = []
        self.next_uid = 1


def _grating_components(kind: str = "GC-SOI") -> tuple[dict, list[dict]]:
    factory = _Factory()
    grating = factory.make_component(kind, 0.0, 0.0)
    factory.components.append(grating)
    factory.components.extend(factory.automatic_simulation_companions(grating))
    return grating, factory.components


def _configuration(kind: str = "GC-SOI") -> dict:
    stack_name = (
        "SOI grating coupler (Ansys)" if kind == "GC-SOI" else "TFLN on SiO2"
    )
    return {
        "included_layers": [(1, 0), (2, 0)],
        "material_stack": lumerical.default_stack(stack_name),
        "resource_mode": "GPU",
        "run_after_build": True,
        "project_file": "grating_power_contract.fsp",
    }


class GratingPowerMonitorContractTests(unittest.TestCase):
    def test_single_export_has_two_modal_ports_and_two_power_planes(self) -> None:
        """The input-power plane must never consume a third modal-port solve."""
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                _grating, components = _grating_components(kind)
                notebook, warnings = lumerical.generate_lumerical_notebook(
                    components, _configuration(kind)
                )
                ports = _assignment_value(notebook, "PORTS")
                monitors = _assignment_value(notebook, "MONITORS")
                analysis = _assignment_value(notebook, "GRATING_ANALYSIS")

                self.assertIsNotNone(analysis, warnings)
                self.assertEqual(len(ports), 2)
                source = next(
                    port for port in ports
                    if port["name"] == analysis["fiber_port_name"]
                )
                receiver = next(
                    port for port in ports
                    if port["name"] == analysis["waveguide_port_name"]
                )
                self.assertEqual(str(source["plane normal"]).upper(), "Z")
                self.assertIn(str(receiver["plane normal"]).upper(), {"X", "Y"})
                self.assertNotEqual(source["name"], receiver["name"])
                self.assertFalse(
                    any(
                        port.get("parent_port_name") == "fiber_input_power"
                        or str(port.get("fiber plane role", "")).lower()
                        == "input power measurement"
                        for port in ports
                    )
                )

                input_monitor = next(
                    monitor for monitor in monitors
                    if monitor["name"] == analysis["fiber_input_power_monitor_name"]
                )
                self.assertEqual(input_monitor["monitor_kind"], "Power monitor")
                self.assertEqual(
                    input_monitor.get("parent_port_name"), "fiber_input_power"
                )
                for modal_key in (
                    "mode",
                    "mode number",
                    "candidate mode numbers",
                    "polarization",
                ):
                    self.assertNotIn(modal_key, input_monitor)
                self.assertFalse(
                    any("apod" in str(key).lower() for key in input_monitor)
                )
                output_monitor = next(
                    monitor for monitor in monitors
                    if monitor["name"] == analysis["waveguide_power_monitor_name"]
                )
                self.assertEqual(output_monitor["monitor_kind"], "Power monitor")
                self.assertEqual(
                    output_monitor.get("grating_monitor_role"),
                    "waveguide_total_power",
                )

                # New notebooks must not serialize the removed passive-port
                # result contract, even if readers retain legacy aliases.
                self.assertNotIn("fiber_input_measurement_port_name", analysis)
                self.assertNotIn("fiber_measurement_expansion_result_name", analysis)
                self.assertNotIn("waveguide_mode_monitor_name", analysis)

    def test_fiber_input_power_object_uses_addpower_without_eigensolver(self) -> None:
        """Exercise the generated monitor builder with only the fiber input plane."""
        _grating, components = _grating_components("GC-SOI")
        notebook, _warnings = lumerical.generate_lumerical_notebook(
            components, _configuration("GC-SOI")
        )
        analysis = _assignment_value(notebook, "GRATING_ANALYSIS")
        input_monitor = next(
            monitor
            for monitor in _assignment_value(notebook, "MONITORS")
            if monitor["name"] == analysis["fiber_input_power_monitor_name"]
        )

        add_monitors = next(
            node
            for node in ast.parse(lumerical._BUILD_CELL).body
            if isinstance(node, ast.FunctionDef) and node.name == "_add_monitors"
        )

        class FakeFdtd:
            def __init__(self) -> None:
                self.addpower_calls = 0
                self.settings: list[tuple[str, object]] = []

            def addpower(self) -> None:
                self.addpower_calls += 1

            def addmodeexpansion(self) -> None:  # pragma: no cover - failure path
                raise AssertionError("input power plane became a mode solver")

            def addprofile(self) -> None:  # pragma: no cover - failure path
                raise AssertionError("input power plane became a field monitor")

            def updateportmodes(self, *_args):  # pragma: no cover - failure path
                raise AssertionError("a power monitor ran updateportmodes")

            def updatemodes(self, *_args):  # pragma: no cover - failure path
                raise AssertionError("a power monitor ran updatemodes")

            def set(self, name, value) -> None:
                self.settings.append((str(name), value))

        namespace = {
            "np": np,
            "UM": 1e-6,
            "MONITORS": [deepcopy(input_monitor)],
            "WAVEGUIDE_INDEX_ESTIMATE": {},
            "WAVEGUIDE_MODE_SELECTIONS": {},
            "_nearest_port_axis": lambda _angle: (0.0, "x-axis", "Forward"),
            "_vertical_reference_um": lambda *_args: 1.0,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[deepcopy(add_monitors)], type_ignores=[])
                ),
                "<fiber-input-power-monitor>",
                "exec",
            ),
            namespace,
        )
        fake = FakeFdtd()
        namespace["_add_monitors"](fake, 0.0, 0.2, 1.2, 0.9, 0.55)
        self.assertEqual(fake.addpower_calls, 1)
        self.assertIn(("name", input_monitor["name"]), fake.settings)
        # The frequency range is inherited from the global monitor settings, so
        # setting it per-monitor would raise "requested property is inactive".
        self.assertNotIn(
            "use source limits", [setting_name for setting_name, _ in fake.settings]
        )
        self.assertIn(("output power", True), fake.settings)
        self.assertFalse(
            any("apod" in setting_name.lower() for setting_name, _ in fake.settings)
        )

    def test_single_analysis_normalizes_both_waveguide_measurements_to_pin(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])
        input_power = np.asarray([-0.50, -0.40, -0.25])
        modal_power = np.asarray([0.20, 0.12, 0.05])
        total_power = np.asarray([-0.30, -0.20, -0.10])

        class FakeFdtd:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str | None]] = []

            def getresult(self, path, result_name=None):
                path = str(path)
                self.calls.append((path, result_name))
                if path == "fiber_input" and result_name == "T":
                    return {"lambda": wavelength_m, "T": input_power}
                if path == "waveguide_total" and result_name == "T":
                    return {"lambda": wavelength_m, "T": total_power}
                if path in {
                    "FDTD::ports::waveguide_receiver",
                    "::model::FDTD::ports::waveguide_receiver",
                } and result_name == "expansion for port monitor":
                    return {"lambda": wavelength_m, "T_out": modal_power}
                raise RuntimeError(f"unexpected result request: {path!r}, {result_name!r}")

        with TemporaryDirectory() as temporary_directory:
            fake = FakeFdtd()
            namespace = {
                "np": np,
                "os": __import__("os"),
                "REMOTE_WORK": temporary_directory,
                "SETTINGS": {
                    "run_after_build": True,
                    "wavelength_start_um": 1.50,
                    "wavelength_stop_um": 1.60,
                },
                "GRATING_ANALYSIS": {
                    "fiber_port_name": "fiber_source",
                    "fiber_input_power_monitor_name": "fiber_input",
                    "waveguide_port_name": "waveguide_receiver",
                    "waveguide_port_expansion_result_name": "expansion for port monitor",
                    "waveguide_port_modal_direction": "T_out",
                    "waveguide_power_monitor_name": "waveguide_total",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_total_power_sign": -1.0,
                    "waveguide_target_neff": 2.45,
                },
                "PORT_MODE_SELECTIONS": {
                    "waveguide_receiver": {"mode number": 1, "neff": 2.47}
                },
                "fdtd": fake,
            }
            exec(lumerical._GRATING_ANALYSIS_REMOTE, namespace)
            arrays = namespace["GRATING_RESULT_ARRAYS"]

        np.testing.assert_allclose(arrays["fiber_input_power"], -np.real(input_power))
        np.testing.assert_allclose(
            arrays["coupling_efficiency"],
            modal_power / -np.real(input_power),
        )
        np.testing.assert_allclose(
            arrays["waveguide_total_transmission"],
            -np.real(total_power) / -np.real(input_power),
        )
        np.testing.assert_allclose(arrays["waveguide_mode_power"], modal_power)
        np.testing.assert_allclose(arrays["waveguide_total_power"], -np.real(total_power))
        self.assertIn(
            (
                "FDTD::ports::waveguide_receiver",
                "expansion for port monitor",
            ),
            fake.calls,
        )
        self.assertFalse(
            any("fiber_input" in path and "ports" in path for path, _ in fake.calls)
        )

    def test_sweep_and_multigpu_keep_the_same_pin_normalization_contract(self) -> None:
        grating, components = _grating_components("GC-SOI")
        sweep_spec = lumerical.normalize_lumerical_sweep_spec(
            grating, [{"parameter": "pitch", "values": [0.68, 0.70]}]
        )
        cases = []
        for values in lumerical.expand_lumerical_sweep_points(sweep_spec):
            case_components = deepcopy(components)
            case_grating = next(
                item for item in case_components if item["uid"] == grating["uid"]
            )
            lumerical.apply_lumerical_sweep_values(case_grating, values)
            context = _Factory()
            context.components = case_components
            context.next_uid = 1 + max(item["uid"] for item in case_components)
            context.synchronize_automatic_simulation_companions(case_grating)
            cases.append({"values": values, "components": context.components})

        sequential, _warnings = lumerical.generate_lumerical_sweep_notebook(
            cases, _configuration("GC-SOI"), sweep_spec
        )
        multigpu, _warnings = lumerical.generate_lumerical_multigpu_sweep_notebook(
            cases,
            {
                **_configuration("GC-SOI"),
                "lumerical_multigpu": {"node_count": 2},
            },
            sweep_spec,
        )
        for notebook in (sequential, multigpu):
            source = _all_source(notebook)
            analysis = _assignment_value(notebook, "GRATING_ANALYSIS")
            self.assertIn("fiber_input_power_monitor_name", source)
            self.assertIn("waveguide_port_name", source)
            self.assertIn("waveguide_total_transmission", source)
            self.assertIn("coupling_efficiency", source)
            self.assertNotIn("fiber_input_measurement_port_name", analysis)
            self.assertNotIn("fiber_measurement_expansion_result_name", analysis)

    def test_sweep_extraction_uses_the_same_direction_corrected_ratios(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6])

        class FakeFdtd:
            def __init__(self) -> None:
                self.source_mode = ""

            def select(self, _path) -> None:
                return None

            def set(self, name, value) -> None:
                if str(name) == "source mode":
                    self.source_mode = str(value)

            def get(self, name):
                if str(name) == "source mode":
                    return self.source_mode
                raise RuntimeError(name)

            def getresult(self, path, result_name=None):
                path = str(path)
                if path == "fiber_input" and result_name == "T":
                    return {"lambda": wavelength_m, "T": np.asarray([-0.5, -0.25])}
                if path == "waveguide_total" and result_name == "T":
                    return {"lambda": wavelength_m, "T": np.asarray([-0.3, -0.1])}
                if path in {
                    "FDTD::ports::waveguide_receiver",
                    "::model::FDTD::ports::waveguide_receiver",
                } and result_name == "expansion for port monitor":
                    return {
                        "lambda": wavelength_m,
                        "T_out": np.asarray([0.2, 0.05]),
                    }
                raise RuntimeError(f"unexpected result request: {path!r}, {result_name!r}")

        with TemporaryDirectory() as temporary_directory:
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "power-monitor-contract",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [
                    {
                        "name": "fiber_source",
                        "plane normal": "Z",
                        "candidate mode numbers": [1, 2, 3],
                    }
                ],
                "FIBER_GEOMETRIES": [],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_port_name": "fiber_source",
                    "fiber_input_power_monitor_name": "fiber_input",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_port_name": "waveguide_receiver",
                    "waveguide_port_expansion_result_name": "expansion for port monitor",
                    "waveguide_port_modal_direction": "T_out",
                    "waveguide_power_monitor_name": "waveguide_total",
                    "waveguide_total_power_sign": -1.0,
                },
                "SWEEP_FIBER_MODE_SELECTIONS": {
                    "fiber_source": {
                        "mode number": 2,
                        "selected mode order": [2, 1, 3],
                        "candidate mode numbers": [1, 2, 3],
                    }
                },
                "fdtd": FakeFdtd(),
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            primary_name, returned_wavelength, response, arrays = namespace[
                "_extract_sweep_result"
            ]()

        self.assertEqual(primary_name, "coupling_efficiency")
        np.testing.assert_allclose(returned_wavelength, wavelength_m)
        np.testing.assert_allclose(response, [0.4, 0.2])
        np.testing.assert_allclose(arrays["fiber_input_power"], [0.5, 0.25])
        np.testing.assert_allclose(arrays["waveguide_mode_power"], [0.2, 0.05])
        np.testing.assert_allclose(arrays["waveguide_total_power"], [0.3, 0.1])
        np.testing.assert_allclose(
            arrays["waveguide_total_transmission"], [0.6, 0.4]
        )

    def test_adjoint_reuses_the_exported_waveguide_receiver(self) -> None:
        grating, components = _grating_components("GC-SOI")
        theta = float(grating["params"]["angle_theta"])
        spec = normalize_lumerical_optimization_spec(
            grating,
            [
                {
                    "parameter": "angle_theta",
                    "minimum": theta - 2.0,
                    "maximum": theta + 2.0,
                }
            ],
            center_wavelength_um=1.55,
            bandwidth_nm=0.0,
            max_iterations=2,
        )
        notebook, _warnings = generate_lumerical_adjoint_notebook(
            components, _configuration("GC-SOI"), spec
        )
        ports = _assignment_value(notebook, "PORTS")
        analysis = _assignment_value(notebook, "GRATING_ANALYSIS")
        objective = _assignment_value(notebook, "OPT_OBJECTIVE_PORTS")
        fiber_pose = _assignment_value(notebook, "OPT_FIBER_POSE")

        self.assertEqual(len(ports), 2)
        self.assertEqual(objective["source_port"], analysis["fiber_port_name"])
        self.assertEqual(objective["monitor_port"], analysis["waveguide_port_name"])
        self.assertEqual(len(fiber_pose["ports"]), 1)
        self.assertTrue(fiber_pose["ports"][0]["is_source"])
        self.assertEqual(len(fiber_pose["monitors"]), 1)
        self.assertFalse(any("adjoint_waveguide_receiver" in port["name"] for port in ports))


if __name__ == "__main__":
    unittest.main()
