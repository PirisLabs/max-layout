from __future__ import annotations

import ast
import base64
from copy import deepcopy
import json
import math
import inspect
import unittest
import zlib

import numpy as np

from max_layout import lumerical
from max_layout.constants import (
    CHOICE_PARAMETERS,
    COMPONENT_SPECS,
    DEFAULT_COMPONENT_VALUES,
    SIMULATION_COMPONENT_KINDS,
)
from max_layout.lumerical import (
    expand_lumerical_sweep_points,
    generate_lumerical_multigpu_sweep_notebook,
    generate_lumerical_notebook,
    generate_lumerical_sweep_notebook,
    normalize_lumerical_sweep_spec,
)
from max_layout.ui.window import (
    NativeLayoutWindow,
    grating_fiber_center_local_um,
    migrate_grating_fiber_offset_parameter,
)
from max_layout.ui.lumerical_dialog import ThreeDModelPreview


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


def _assignment(notebook: dict, name: str):
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


def _decoded_sweep_cases(notebook: dict) -> list[dict]:
    encoded = _assignment(notebook, "_SWEEP_CASES_B64")
    return json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))


def _gaussian_setup(kind: str, *, orientation_deg: float = 0.0) -> tuple[_Factory, dict]:
    factory = _Factory()
    grating = factory.make_component(kind, 4.0, -3.0)
    grating["orientation_deg"] = orientation_deg
    grating["params"]["excitation_type"] = "gaussian_beam"
    factory.components.append(grating)
    factory.components.extend(factory.automatic_simulation_companions(grating))
    return factory, grating


def _configuration(kind: str) -> dict:
    preset = "SOI grating coupler (Ansys)" if kind == "GC-SOI" else "TFLN photonics"
    return {
        "included_layers": [[1, 0], [2, 0]],
        "material_stack": lumerical.default_stack(preset),
        "wavelength_start_um": 1.50,
        "wavelength_stop_um": 1.60,
        "frequency_points": 5,
        "resource_mode": "GPU",
        "run_after_build": True,
    }


class GratingExcitationTypeTests(unittest.TestCase):
    def test_3d_preview_keeps_z_planes_horizontal_and_draws_gaussian_k_arrow(self) -> None:
        source = inspect.getsource(ThreeDModelPreview.paintEvent)
        self.assertIn("injection axis = Z", source)
        self.assertIn("(cx - x_half, cy - y_half, port_z)", source)
        self.assertIn('if kind == "Gaussian source"', source)
        self.assertIn("beam_arrow", source)
        self.assertIn("math.sin(theta) * math.cos(phi)", source)
        self.assertNotIn("span * tilted", source)

    def test_excitation_type_is_a_json_backed_choice_for_both_gratings(self) -> None:
        self.assertEqual(
            CHOICE_PARAMETERS["excitation_type"],
            ["fiber_mode", "gaussian_beam"],
        )
        self.assertIn("Gaussian source", SIMULATION_COMPONENT_KINDS)
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    DEFAULT_COMPONENT_VALUES[kind]["excitation_type"], "fiber_mode"
                )
                self.assertEqual(
                    DEFAULT_COMPONENT_VALUES[kind]["gaussian_waist_radius_um"], 4.5
                )
                self.assertEqual(
                    DEFAULT_COMPONENT_VALUES[kind]["gaussian_distance_from_waist_um"],
                    0.0,
                )
                self.assertEqual(
                    DEFAULT_COMPONENT_VALUES[kind]["gaussian_source_span_um"], 20.0
                )
                self.assertEqual(
                    COMPONENT_SPECS[kind]["excitation_type"],
                    [
                        "choice",
                        "fiber_mode",
                        ["fiber_mode", "gaussian_beam"],
                    ],
                )
                for numeric_name in (
                    "gaussian_waist_radius_um",
                    "gaussian_distance_from_waist_um",
                    "gaussian_source_span_um",
                ):
                    self.assertEqual(COMPONENT_SPECS[kind][numeric_name][0], "float")
                saved = json.loads(json.dumps({
                    "kind": kind,
                    "params": {
                        **DEFAULT_COMPONENT_VALUES[kind],
                        "excitation_type": "gaussian_beam",
                    },
                }))
                self.assertEqual(saved["params"]["excitation_type"], "gaussian_beam")

    def test_fiber_mode_remains_the_default_two_modal_port_setup(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                factory = _Factory()
                grating = factory.make_component(kind, 0.0, 0.0)
                factory.components.append(grating)
                companions = factory.automatic_simulation_companions(grating)
                modal_ports = [
                    item for item in companions
                    if item["kind"] in {"FDTD port", "Fiber-axis FDTD port"}
                ]
                self.assertEqual(len(modal_ports), 2)
                self.assertEqual(
                    sum(item["kind"] == "Fiber geometry" for item in companions), 1
                )
                self.assertFalse(any(item["kind"] == "Gaussian source" for item in companions))

    def test_legacy_grating_json_migrates_to_fiber_mode(self) -> None:
        legacy = {
            "kind": "Grating coupler",
            "params": {
                key: deepcopy(value)
                for key, value in DEFAULT_COMPONENT_VALUES["Grating coupler"].items()
                if key
                not in {
                    "excitation_type",
                    "gaussian_waist_radius_um",
                    "gaussian_distance_from_waist_um",
                    "gaussian_source_span_um",
                    "gaussian_multifrequency_points",
                }
            },
        }
        self.assertTrue(migrate_grating_fiber_offset_parameter(legacy))
        self.assertEqual(legacy["params"]["excitation_type"], "fiber_mode")
        self.assertEqual(legacy["params"]["gaussian_waist_radius_um"], 4.5)
        self.assertEqual(legacy["params"]["gaussian_multifrequency_points"], 5)

    def test_gaussian_companions_have_one_receiver_and_two_power_planes(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                factory, grating = _gaussian_setup(kind, orientation_deg=31.0)
                companions = factory.components[1:]
                sources = [item for item in companions if item["kind"] == "Gaussian source"]
                ports = [item for item in companions if item["kind"] == "FDTD port"]
                monitors = [item for item in companions if item["kind"] == "Power monitor"]

                self.assertEqual(len(sources), 1)
                self.assertEqual(len(ports), 1)
                self.assertEqual(len(monitors), 2)
                self.assertFalse(any(item["kind"] == "Fiber geometry" for item in companions))
                self.assertFalse(any(item["kind"] == "Fiber-axis FDTD port" for item in companions))
                self.assertEqual(ports[0]["simulation_parent_port"], "waveguide_point")
                self.assertEqual(ports[0]["params"]["order"], 1)

                source = sources[0]
                source_params = source["params"]
                self.assertEqual(source_params["name"], f"uid_{grating['uid']}_gaussian_source")
                self.assertEqual(source_params["injection axis"], "Z")
                self.assertEqual(source_params["direction"], "Backward")
                self.assertEqual(source_params["polarization"], "local TE")
                self.assertEqual(source_params["polarization angle"], 90.0)
                self.assertEqual(source_params["angle theta"], grating["params"]["angle_theta"])
                self.assertEqual(source_params["angle phi"], 0.0)
                for required in (
                    "waist radius_um",
                    "distance from waist_um",
                    "span_um",
                ):
                    self.assertIn(required, source_params)

                input_power = next(
                    item for item in monitors
                    if item.get("simulation_parent_port") == "fiber_input_power"
                )
                total_output = next(
                    item for item in monitors
                    if item.get("grating_monitor_role") == "waveguide_total_power"
                )
                self.assertEqual(input_power["params"]["plane normal"], "Z")
                self.assertEqual(input_power["params"]["expected propagation sign"], -1.0)
                self.assertEqual(total_output["params"]["monitor geometry"], "surface")

    def test_gaussian_angle_offset_and_local_te_stay_synchronized(self) -> None:
        factory, grating = _gaussian_setup("Grating coupler", orientation_deg=37.0)
        source = next(item for item in factory.components if item["kind"] == "Gaussian source")
        input_power = next(
            item for item in factory.components
            if item["kind"] == "Power monitor"
            and item.get("simulation_parent_port") == "fiber_input_power"
        )

        local_x, local_y = grating_fiber_center_local_um(grating)
        angle = math.radians(37.0)
        self.assertAlmostEqual(
            source["x"], grating["x"] + local_x * math.cos(angle) - local_y * math.sin(angle)
        )
        self.assertAlmostEqual(
            source["y"], grating["y"] + local_x * math.sin(angle) + local_y * math.cos(angle)
        )
        self.assertEqual(source["orientation_deg"], 37.0)
        self.assertEqual(source["params"]["polarization"], "local TE")
        self.assertEqual(source["params"]["polarization angle"], 90.0)

        before = (float(source["x"]), float(source["y"]))
        grating["params"]["fiber_offset"] += 2.25
        grating["params"]["angle_theta"] = 16.0
        self.assertTrue(factory.synchronize_automatic_simulation_companions(grating))
        self.assertAlmostEqual(source["x"] - before[0], 2.25 * math.cos(angle))
        self.assertAlmostEqual(source["y"] - before[1], 2.25 * math.sin(angle))
        self.assertEqual(source["params"]["angle theta"], 16.0)
        self.assertEqual(input_power["params"]["angle theta"], 16.0)
        self.assertEqual(source["params"]["angle phi"], 0.0)
        self.assertEqual(source["params"]["polarization angle"], 90.0)

        below_um = float(grating["params"]["fiber_power_monitor_below_source_um"])
        expected_lateral = -below_um * math.tan(math.radians(16.0))
        self.assertAlmostEqual(
            input_power["x"] - source["x"], expected_lateral * math.cos(angle)
        )
        self.assertAlmostEqual(
            input_power["y"] - source["y"], expected_lateral * math.sin(angle)
        )

    def test_switching_excitation_reconciles_parent_owned_companions(self) -> None:
        factory = _Factory()
        grating = factory.make_component("Grating coupler", 0.0, 0.0)
        factory.components.append(grating)
        factory.components.extend(factory.automatic_simulation_companions(grating))
        input_monitor_uid = next(
            item["uid"] for item in factory.components
            if item.get("simulation_parent_port") == "fiber_input_power"
        )

        grating["params"]["excitation_type"] = "gaussian_beam"
        self.assertTrue(factory.synchronize_automatic_simulation_companions(grating))
        linked = [
            item for item in factory.components
            if item.get("simulation_parent_uid") == grating["uid"]
        ]
        self.assertEqual(sum(item["kind"] == "Gaussian source" for item in linked), 1)
        self.assertFalse(any(item["kind"] == "Fiber geometry" for item in linked))
        self.assertFalse(any(item["kind"] == "Fiber-axis FDTD port" for item in linked))
        self.assertEqual(
            next(item for item in linked if item.get("simulation_parent_port") == "fiber_input_power")["uid"],
            input_monitor_uid,
        )
        self.assertEqual(
            next(item for item in linked if item["kind"] == "FDTD port" and item.get("simulation_parent_port") == "waveguide_point")["params"]["order"],
            1,
        )

        grating["params"]["excitation_type"] = "fiber_mode"
        self.assertTrue(factory.synchronize_automatic_simulation_companions(grating))
        linked = [
            item for item in factory.components
            if item.get("simulation_parent_uid") == grating["uid"]
        ]
        self.assertFalse(any(item["kind"] == "Gaussian source" for item in linked))
        self.assertEqual(sum(item["kind"] == "Fiber geometry" for item in linked), 1)
        self.assertEqual(sum(item["kind"] == "Fiber-axis FDTD port" for item in linked), 1)
        self.assertEqual(
            next(item for item in linked if item["kind"] == "FDTD port" and item.get("simulation_parent_port") == "waveguide_point")["params"]["order"],
            2,
        )

    def test_single_notebook_exports_gaussian_source_without_fiber_mode_objects(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                factory, grating = _gaussian_setup(kind, orientation_deg=23.0)
                notebook, warnings = generate_lumerical_notebook(
                    factory.components, _configuration(kind)
                )
                self.assertFalse(any("not added" in warning.lower() for warning in warnings))
                gaussian_sources = _assignment(notebook, "GAUSSIAN_SOURCES")
                ports = _assignment(notebook, "PORTS")
                fibers = _assignment(notebook, "FIBER_GEOMETRIES")
                monitors = _assignment(notebook, "MONITORS")
                analysis = _assignment(notebook, "GRATING_ANALYSIS")

                self.assertEqual(len(gaussian_sources), 1)
                self.assertEqual(gaussian_sources[0]["name"], f"uid_{grating['uid']}_gaussian_source")
                self.assertEqual(gaussian_sources[0]["angle theta"], grating["params"]["angle_theta"])
                self.assertEqual(gaussian_sources[0]["angle phi"], 23.0)
                self.assertEqual(gaussian_sources[0]["polarization angle"], 90.0)
                self.assertEqual(fibers, [])
                self.assertEqual(len(ports), 1)
                self.assertEqual(ports[0]["parent_port_name"], "waveguide_point")
                self.assertEqual(len(monitors), 2)
                self.assertEqual(analysis["excitation_type"], "gaussian_beam")
                self.assertEqual(analysis["source_kind"], "gaussian")
                self.assertEqual(analysis["source_name"], gaussian_sources[0]["name"])
                self.assertNotIn("fiber_port_name", analysis)
                self.assertNotIn("fiber_source_mode", analysis)

                source = _all_source(notebook)
                self.assertIn("GAUSSIAN_SOURCES", source)
                self.assertIn("fdtd.addgaussian()", source)
                self.assertIn('fdtd.set("injection axis", "z")', source)
                self.assertIn('fdtd.set("direction", "backward")', source)
                self.assertIn('fdtd.set("polarization angle"', source)
                self.assertIn('fdtd.set("waist radius w0"', source)
                self.assertIn('fdtd.set("distance from waist"', source)
                self.assertIn('fdtd.set("source port", "")', source)

    def test_gaussian_multifrequency_profile_sets_samples_first_and_verifies_readback(self) -> None:
        function_node = next(
            node
            for node in ast.parse(lumerical._BUILD_CELL).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_add_gaussian_sources"
        )
        namespace = {
            "np": np,
            "UM": 1e-6,
            "GAUSSIAN_SOURCES": [
                {
                    "name": "test_gaussian",
                    "center": [1.0, 2.0],
                    "span_um": 20.0,
                    "waist radius_um": 4.5,
                    "distance from waist_um": 0.0,
                    "multifrequency beam calculation": True,
                    "frequency points": 7,
                }
            ],
            "_vertical_reference_um": lambda *_args: 0.0,
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[function_node], type_ignores=[])
                ),
                "<gaussian-source-builder>",
                "exec",
            ),
            namespace,
        )

        class FakeFdtd:
            def __init__(self) -> None:
                self.events = []
                self.values = {}

            def addgaussian(self) -> None:
                self.events.append(("addgaussian", None))

            def set(self, property_name, value) -> None:
                self.events.append((str(property_name), value))
                self.values[str(property_name)] = value

            def get(self, property_name):
                self.events.append(("get " + str(property_name), None))
                return self.values[str(property_name)]

        fdtd = FakeFdtd()
        namespace["_add_gaussian_sources"](fdtd, 0.0, 0.0, 0.0, 0.0)
        sample_event = fdtd.events.index(("number of field profile samples", 7))
        activation_event = fdtd.events.index(
            ("multifrequency beam calculation", True)
        )
        self.assertLess(sample_event, activation_event)
        self.assertIn(("get number of field profile samples", None), fdtd.events)
        self.assertIn(("get multifrequency beam calculation", None), fdtd.events)
        self.assertNotIn(("frequency points", 7), fdtd.events)

        class ActivationFirstFdtd(FakeFdtd):
            def __init__(self) -> None:
                super().__init__()
                self.rejected_samples_once = False

            def set(self, property_name, value) -> None:
                if (
                    property_name == "number of field profile samples"
                    and not self.values.get("multifrequency beam calculation", False)
                    and not self.rejected_samples_once
                ):
                    self.events.append((str(property_name), value))
                    self.rejected_samples_once = True
                    raise RuntimeError("property inactive until multifrequency is enabled")
                super().set(property_name, value)

        fallback_fdtd = ActivationFirstFdtd()
        namespace["_add_gaussian_sources"](
            fallback_fdtd, 0.0, 0.0, 0.0, 0.0
        )
        self.assertTrue(fallback_fdtd.rejected_samples_once)
        self.assertEqual(
            fallback_fdtd.values["number of field profile samples"], 7
        )
        self.assertTrue(
            fallback_fdtd.values["multifrequency beam calculation"]
        )

    def test_gaussian_export_filters_stale_parent_owned_fiber_objects(self) -> None:
        factory = _Factory()
        grating = factory.make_component("Grating coupler", 0.0, 0.0)
        factory.components.append(grating)
        # Reproduce a legacy project that changed the parent selection without
        # first reconciling its old fiber companions.
        factory.components.extend(factory.automatic_simulation_companions(grating))
        grating["params"]["excitation_type"] = "gaussian_beam"
        source = factory.make_component("Gaussian source", 5.0, 0.0)
        source["simulation_parent_uid"] = grating["uid"]
        source["simulation_parent_port"] = "gaussian_source"
        source["params"]["name"] = f"uid_{grating['uid']}_gaussian_source"
        factory.components.append(source)

        notebook, warnings = generate_lumerical_notebook(
            factory.components, _configuration("Grating coupler")
        )
        self.assertEqual(_assignment(notebook, "FIBER_GEOMETRIES"), [])
        self.assertEqual(len(_assignment(notebook, "PORTS")), 1)
        self.assertEqual(len(_assignment(notebook, "GAUSSIAN_SOURCES")), 1)
        self.assertTrue(any("removed stale parent-owned" in warning for warning in warnings))

    def test_gaussian_angle_sweep_reserves_every_source_plane_in_fixed_z_domain(self) -> None:
        factory, grating = _gaussian_setup("GC-SOI")
        spec = normalize_lumerical_sweep_spec(
            grating, [{"parameter": "angle_theta", "values": [0.0, 40.0]}]
        )
        cases = []
        for values in expand_lumerical_sweep_points(spec):
            components = deepcopy(factory.components)
            variant = next(item for item in components if item["uid"] == grating["uid"])
            variant["params"].update(values)
            variant_factory = _Factory()
            variant_factory.components = components
            variant_factory.next_uid = 1 + max(item["uid"] for item in components)
            variant_factory.synchronize_automatic_simulation_companions(variant)
            cases.append({"values": values, "components": variant_factory.components})

        notebook, _ = generate_lumerical_sweep_notebook(
            cases, _configuration("GC-SOI"), spec,
            nominal_components=factory.components,
        )
        settings = _assignment(notebook, "SETTINGS")
        z_bounds = settings.get("sweep_sampling_z_bounds_um")
        self.assertIsInstance(z_bounds, list)
        self.assertEqual(len(z_bounds), 2)
        self.assertLess(z_bounds[0], z_bounds[1])
        self.assertFalse(_assignment(notebook, "SWEEP_RECOMPUTE_MODES"))
        self.assertIn("Reserved fixed %s Z envelope", _all_source(notebook))

    def test_gaussian_source_survives_sequential_and_multigpu_sweeps(self) -> None:
        factory, grating = _gaussian_setup("GC-SOI", orientation_deg=19.0)
        spec = normalize_lumerical_sweep_spec(
            grating, [{"parameter": "pitch", "values": [0.66, 0.68]}]
        )
        cases = []
        for values in expand_lumerical_sweep_points(spec):
            components = deepcopy(factory.components)
            variant = next(item for item in components if item["uid"] == grating["uid"])
            variant["params"].update(values)
            variant_factory = _Factory()
            variant_factory.components = components
            variant_factory.next_uid = 1 + max(item["uid"] for item in components)
            self.assertTrue(
                variant_factory.synchronize_automatic_simulation_companions(variant)
            )
            cases.append({"values": values, "components": variant_factory.components})

        configuration = {
            **_configuration("GC-SOI"),
            "multigpu_node_count": 2,
            "simulations_per_gpu": 1,
        }
        sequential, _ = generate_lumerical_sweep_notebook(
            cases, configuration, spec, nominal_components=factory.components
        )
        multigpu, _ = generate_lumerical_multigpu_sweep_notebook(
            cases, configuration, spec, nominal_components=factory.components
        )

        for label, notebook in (("sequential", sequential), ("multigpu", multigpu)):
            with self.subTest(exporter=label):
                payload_sources = _assignment(notebook, "GAUSSIAN_SOURCES")
                self.assertEqual(len(payload_sources), 1)
                self.assertEqual(payload_sources[0]["angle phi"], 19.0)
                self.assertEqual(len(_assignment(notebook, "PORTS")), 1)
                self.assertEqual(len(_assignment(notebook, "MONITORS")), 2)
                remote_cases = _decoded_sweep_cases(notebook)
                self.assertEqual(len(remote_cases), 2)
                for case in remote_cases:
                    self.assertEqual(len(case["gaussian_sources"]), 1)
                    self.assertEqual(case["fiber_geometries"], [])
                    self.assertEqual(len(case["ports"]), 1)
                    self.assertEqual(len(case["monitors"]), 2)
                    self.assertEqual(
                        case["grating_analysis"]["excitation_type"],
                        "gaussian_beam",
                    )
                source = _all_source(notebook)
                self.assertIn("GAUSSIAN_SOURCES", source)
                self.assertIn("gaussian_sources", source)


if __name__ == "__main__":
    unittest.main()
