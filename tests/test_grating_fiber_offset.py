from __future__ import annotations

import ast
import base64
from copy import deepcopy
import inspect
import json
import math
import unittest
import zlib

from max_layout import lumerical
from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.lumerical import (
    expand_lumerical_sweep_points,
    generate_lumerical_multigpu_sweep_notebook,
    generate_lumerical_sweep_notebook,
    normalize_lumerical_sweep_spec,
    sweepable_component_parameters,
)
from max_layout.ui.window import (
    NativeLayoutWindow,
    grating_fiber_center_local_um,
    grating_first_flare_local_x_um,
    migrate_grating_fiber_offset_parameter,
)


_LEGACY_KEYS = (
    "fiber_x_from_grating_start_um",
    "fiber_offset_after_taper_um",
    "fiber_offset_from_flare_um",
)


def _component(kind: str, *, uid: int = 1) -> dict:
    return {
        "uid": uid,
        "kind": kind,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "params": deepcopy(DEFAULT_COMPONENT_VALUES[kind]),
        "attachment": None,
    }


class _Factory:
    make_component = NativeLayoutWindow.make_component
    automatic_simulation_companions = NativeLayoutWindow.automatic_simulation_companions
    synchronize_automatic_simulation_companions = (
        NativeLayoutWindow.synchronize_automatic_simulation_companions
    )

    def __init__(self) -> None:
        self.components: list[dict] = []
        self.next_uid = 1


def _fiber_companions(companions: list[dict]) -> tuple[dict, dict, dict]:
    fiber = next(item for item in companions if item["kind"] == "Fiber geometry")
    source = next(
        item
        for item in companions
        if item["kind"] == "Fiber-axis FDTD port"
        and item.get("simulation_parent_port") != "fiber_input_power"
    )
    power = next(
        item
        for item in companions
        if item["kind"] == "Power monitor"
        and item.get("simulation_parent_port") == "fiber_input_power"
    )
    return fiber, source, power


def _notebook_assignment(notebook: dict, name: str):
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


def _notebook_sweep_cases(notebook: dict) -> list[dict]:
    encoded = _notebook_assignment(notebook, "_SWEEP_CASES_B64")
    return json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))


class GratingFiberOffsetTests(unittest.TestCase):
    def test_new_components_use_canonical_fiber_pose_parent_keys(self) -> None:
        expected_defaults = {
            "Grating coupler": (5.0, 7.0),
            "GC-SOI": (2.74533, 10.0),
        }
        for kind, (expected_offset, expected_theta) in expected_defaults.items():
            with self.subTest(kind=kind):
                params = DEFAULT_COMPONENT_VALUES[kind]
                self.assertEqual(params["fiber_offset"], expected_offset)
                self.assertEqual(params["angle_theta"], expected_theta)
                self.assertNotIn("fiber_tilt_deg", params)
                self.assertTrue(all(key not in params for key in _LEGACY_KEYS))

    def test_legacy_gc_soi_fiber_tilt_migrates_to_angle_theta(self) -> None:
        legacy = _component("GC-SOI")
        legacy["params"].pop("angle_theta")
        legacy["params"]["fiber_tilt_deg"] = 13.25

        self.assertTrue(migrate_grating_fiber_offset_parameter(legacy))
        self.assertEqual(legacy["params"]["angle_theta"], 13.25)
        self.assertNotIn("fiber_tilt_deg", legacy["params"])
        self.assertFalse(migrate_grating_fiber_offset_parameter(legacy))

        canonical = _component("GC-SOI")
        canonical["params"]["angle_theta"] = 8.5
        canonical["params"]["fiber_tilt_deg"] = 17.0
        self.assertTrue(migrate_grating_fiber_offset_parameter(canonical))
        self.assertEqual(canonical["params"]["angle_theta"], 8.5)
        self.assertNotIn("fiber_tilt_deg", canonical["params"])

    def test_exact_first_flare_anchor_and_signed_local_x_center(self) -> None:
        generic = _component("Grating coupler")
        generic["params"].update(
            {
                "wg_length": 7.0,
                "wg_width": 1.4,
                "alpha_t": 28.0,
                "taper_L": 21.0,
                "fiber_offset": -3.25,
            }
        )
        focus_offset = (1.4 / 2.0) / math.tan(math.radians(28.0 / 2.0))
        expected_anchor = 7.0 - focus_offset + 21.0
        self.assertAlmostEqual(
            grating_first_flare_local_x_um(generic), expected_anchor
        )
        self.assertAlmostEqual(
            grating_fiber_center_local_um(generic)[0], expected_anchor - 3.25
        )
        self.assertEqual(grating_fiber_center_local_um(generic)[1], 0.0)

        soi = _component("GC-SOI")
        soi["params"].update(
            {"wg_length": 9.0, "radius": 27.0, "fiber_offset": -4.5}
        )
        self.assertEqual(grating_first_flare_local_x_um(soi), 36.0)
        self.assertEqual(grating_fiber_center_local_um(soi), (31.5, 0.0))

    def test_legacy_keys_migrate_to_canonical_offset_without_moving_fiber(self) -> None:
        legacy_value = 5.0
        for kind in ("Grating coupler", "GC-SOI"):
            for legacy_key in _LEGACY_KEYS:
                with self.subTest(kind=kind, legacy_key=legacy_key):
                    item = _component(kind)
                    params = item["params"]
                    params.pop("fiber_offset")
                    for key in _LEGACY_KEYS:
                        params.pop(key, None)
                    params[legacy_key] = legacy_value

                    old_absolute_x = (
                        float(params["wg_length"])
                        + (
                            float(params["radius"])
                            if kind == "GC-SOI"
                            else float(params["taper_L"])
                        )
                        + legacy_value
                    )
                    self.assertTrue(migrate_grating_fiber_offset_parameter(item))
                    self.assertTrue(all(key not in params for key in _LEGACY_KEYS))
                    self.assertAlmostEqual(
                        grating_fiber_center_local_um(item)[0], old_absolute_x
                    )
                    if kind == "GC-SOI":
                        self.assertEqual(params["fiber_offset"], legacy_value)
                    else:
                        focus_offset = (
                            (float(params["wg_width"]) / 2.0)
                            / math.tan(math.radians(float(params["alpha_t"]) / 2.0))
                        )
                        self.assertAlmostEqual(
                            params["fiber_offset"], legacy_value + focus_offset
                        )
                    self.assertFalse(migrate_grating_fiber_offset_parameter(item))

    def test_canonical_offset_wins_and_removes_stale_legacy_keys(self) -> None:
        item = _component("Grating coupler")
        item["params"].update(
            {
                "fiber_offset": -1.75,
                "fiber_offset_after_taper_um": 99.0,
                "fiber_offset_from_flare_um": 98.0,
                "fiber_x_from_grating_start_um": 97.0,
            }
        )
        self.assertTrue(migrate_grating_fiber_offset_parameter(item))
        self.assertEqual(item["params"]["fiber_offset"], -1.75)
        self.assertTrue(all(key not in item["params"] for key in _LEGACY_KEYS))

    def test_offset_moves_fiber_source_and_power_plane_together_on_local_x(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                factory = _Factory()
                parent = factory.make_component(kind, 12.25, -6.5)
                parent["orientation_deg"] = 0.0
                parent["params"]["fiber_offset"] = -1.25
                factory.components.append(parent)
                companions = factory.automatic_simulation_companions(parent)
                factory.components.extend(companions)
                fiber, source, power = _fiber_companions(companions)

                local_x, local_y = grating_fiber_center_local_um(parent)
                self.assertEqual(local_y, 0.0)
                self.assertAlmostEqual(fiber["x"], parent["x"] + local_x)
                self.assertEqual(fiber["y"], parent["y"])
                self.assertAlmostEqual(source["x"], fiber["x"])
                self.assertEqual(source["y"], fiber["y"])
                self.assertEqual(power["y"], fiber["y"])

                # The lower diagnostic plane has a different top-view X
                # intercept only because it lies farther down the same tilted
                # fiber axis.  This is the concentric tilted-plane relation.
                theta = math.radians(float(source["params"]["angle theta"]))
                source_distance = float(source["params"]["distance_um"])
                power_distance = float(power["params"]["distance_um"])
                self.assertAlmostEqual(
                    power["x"] - source["x"],
                    (power_distance - source_distance) * math.tan(theta),
                )

                before = {
                    int(item["uid"]): (float(item["x"]), float(item["y"]))
                    for item in (fiber, source, power)
                }
                shift = 2.75
                parent["params"]["fiber_offset"] += shift
                self.assertTrue(
                    factory.synchronize_automatic_simulation_companions(parent)
                )
                for item in (fiber, source, power):
                    old_x, old_y = before[int(item["uid"])]
                    self.assertAlmostEqual(float(item["x"]), old_x + shift)
                    self.assertEqual(float(item["y"]), old_y)
                    self.assertEqual(float(item["fiber_offset_um"]), 1.5)

    def test_angle_theta_tilts_fiber_source_and_power_monitor_together(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                factory = _Factory()
                parent = factory.make_component(kind, 8.0, -3.0)
                parent["orientation_deg"] = 29.0
                parent["params"]["angle_theta"] = 5.0
                factory.components.append(parent)
                companions = factory.automatic_simulation_companions(parent)
                factory.components.extend(companions)
                fiber, source, power = _fiber_companions(companions)

                for item in (fiber, source, power):
                    self.assertEqual(float(item["params"]["angle theta"]), 5.0)

                old_power_xy = (float(power["x"]), float(power["y"]))
                parent["params"]["angle_theta"] = 18.0
                self.assertTrue(
                    factory.synchronize_automatic_simulation_companions(parent)
                )

                for item in (fiber, source, power):
                    self.assertEqual(float(item["params"]["angle theta"]), 18.0)
                self.assertAlmostEqual(float(source["x"]), float(fiber["x"]))
                self.assertAlmostEqual(float(source["y"]), float(fiber["y"]))

                source_distance = float(source["params"]["distance_um"])
                power_distance = float(power["params"]["distance_um"])
                lateral = (power_distance - source_distance) * math.tan(
                    math.radians(18.0)
                )
                orientation = math.radians(29.0)
                self.assertAlmostEqual(
                    float(power["x"]) - float(source["x"]),
                    lateral * math.cos(orientation),
                )
                self.assertAlmostEqual(
                    float(power["y"]) - float(source["y"]),
                    lateral * math.sin(orientation),
                )
                self.assertNotEqual(
                    (float(power["x"]), float(power["y"])), old_power_xy
                )

                expected_rotation_offset = 4.0 * float(
                    parent["params"].get("fiber_core_diameter_um", 9.0)
                ) * math.tan(math.radians(18.0))
                self.assertAlmostEqual(
                    float(source["params"]["rotation offset_um"]),
                    expected_rotation_offset,
                )
                self.assertNotIn("rotation offset_um", power["params"])
                self.assertEqual(power["kind"], "Power monitor")
                self.assertEqual(power["params"]["plane normal"], "Z")
                self.assertEqual(
                    power["params"]["fiber plane role"],
                    "input power measurement",
                )
                self.assertEqual(power["params"]["expected propagation sign"], -1.0)
                self.assertGreater(
                    float(power["params"]["x span"]),
                    float(source["params"]["span_um"]),
                )

    def test_rotated_offset_moves_only_fiber_axis_companions(self) -> None:
        factory = _Factory()
        parent = factory.make_component("Grating coupler", 14.0, -9.0)
        angle_deg = 37.0
        parent["orientation_deg"] = angle_deg
        factory.components.append(parent)
        companions = factory.automatic_simulation_companions(parent)
        factory.components.extend(companions)

        fiber, source, power = _fiber_companions(companions)
        waveguide_companions = [
            item
            for item in companions
            if item.get("grating_monitor_role") == "waveguide_total_power"
            or (
                item.get("kind") == "FDTD port"
                and item.get("simulation_parent_port") == "waveguide_point"
            )
        ]
        before = {
            int(item["uid"]): (float(item["x"]), float(item["y"]))
            for item in companions
        }
        offset_delta = 3.125
        parent["params"]["fiber_offset"] += offset_delta
        self.assertTrue(factory.synchronize_automatic_simulation_companions(parent))

        expected_dx = offset_delta * math.cos(math.radians(angle_deg))
        expected_dy = offset_delta * math.sin(math.radians(angle_deg))
        for item in (fiber, source, power):
            old_x, old_y = before[int(item["uid"])]
            self.assertAlmostEqual(float(item["x"]) - old_x, expected_dx)
            self.assertAlmostEqual(float(item["y"]) - old_y, expected_dy)
        for item in waveguide_companions:
            self.assertEqual(
                (float(item["x"]), float(item["y"])), before[int(item["uid"])]
            )

    def test_rotated_fiber_pose_sweep_payload_matches_both_exporters(self) -> None:
        factory = _Factory()
        parent = factory.make_component("Grating coupler", 14.0, -9.0)
        parent["orientation_deg"] = 37.0
        parent["params"]["angle_theta"] = 6.0
        factory.components.append(parent)
        factory.components.extend(factory.automatic_simulation_companions(parent))
        first_offset = float(parent["params"]["fiber_offset"])
        second_offset = first_offset + 2.0
        spec = normalize_lumerical_sweep_spec(
            parent,
            [
                {
                    "parameter": "angle_theta",
                    "values": [6.0, 17.0],
                },
                {
                    "parameter": "fiber_offset",
                    "values": [first_offset, second_offset],
                }
            ],
        )
        sweep_cases = []
        for values in expand_lumerical_sweep_points(spec):
            variant_components = deepcopy(factory.components)
            variant_parent = next(
                item for item in variant_components if item["uid"] == parent["uid"]
            )
            variant_parent["params"].update(values)
            variant_factory = _Factory()
            variant_factory.components = variant_components
            variant_factory.next_uid = 1 + max(
                int(item["uid"]) for item in variant_components
            )
            variant_factory.synchronize_automatic_simulation_companions(
                variant_parent
            )
            sweep_cases.append(
                {"values": values, "components": variant_factory.components}
            )

        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("TFLN on SiO2"),
            "resource_mode": "GPU",
            "run_after_build": True,
        }
        sequential, _ = generate_lumerical_sweep_notebook(
            sweep_cases, configuration, spec
        )
        multigpu, _ = generate_lumerical_multigpu_sweep_notebook(
            sweep_cases, configuration, spec
        )
        sequential_cases = _notebook_sweep_cases(sequential)
        multigpu_cases = _notebook_sweep_cases(multigpu)
        self.assertEqual(sequential_cases, multigpu_cases)
        self.assertEqual(_notebook_assignment(sequential, "SWEEP_SPEC"), spec)
        self.assertEqual(_notebook_assignment(multigpu, "SWEEP_SPEC"), spec)
        self.assertTrue(_notebook_assignment(sequential, "SWEEP_RECOMPUTE_MODES"))
        self.assertTrue(_notebook_assignment(multigpu, "SWEEP_RECOMPUTE_MODES"))

        first_case, second_case, third_case, _fourth_case = sequential_cases
        angle_rad = math.radians(37.0)
        expected_delta = (
            2.0 * math.cos(angle_rad),
            2.0 * math.sin(angle_rad),
        )

        def centers_by_name(case: dict, key: str) -> dict[str, tuple[float, float]]:
            return {
                str(item["name"]): tuple(map(float, item["center"]))
                for item in case[key]
            }

        first_fibers = centers_by_name(first_case, "fiber_geometries")
        second_fibers = centers_by_name(second_case, "fiber_geometries")
        first_ports = {
            str(item["name"]): tuple(map(float, item["center"]))
            for item in first_case["ports"]
            if str(item.get("plane normal", "")).upper() == "Z"
        }
        second_ports = {
            str(item["name"]): tuple(map(float, item["center"]))
            for item in second_case["ports"]
            if str(item.get("plane normal", "")).upper() == "Z"
        }
        first_monitors = centers_by_name(first_case, "monitors")
        second_monitors = centers_by_name(second_case, "monitors")
        for first_centers, second_centers in (
            (first_fibers, second_fibers),
            (first_ports, second_ports),
        ):
            self.assertEqual(first_centers.keys(), second_centers.keys())
            for name in first_centers:
                self.assertAlmostEqual(
                    second_centers[name][0] - first_centers[name][0],
                    expected_delta[0],
                )
                self.assertAlmostEqual(
                    second_centers[name][1] - first_centers[name][1],
                    expected_delta[1],
                )
        input_monitor_name = next(
            name for name in first_monitors if name.endswith("fiber_input_power")
        )
        waveguide_monitor_name = next(
            name for name in first_monitors if name.endswith("waveguide_total_power")
        )
        self.assertEqual(
            second_monitors[waveguide_monitor_name],
            first_monitors[waveguide_monitor_name],
        )
        self.assertAlmostEqual(
            second_monitors[input_monitor_name][0]
            - first_monitors[input_monitor_name][0],
            expected_delta[0],
        )
        self.assertAlmostEqual(
            second_monitors[input_monitor_name][1]
            - first_monitors[input_monitor_name][1],
            expected_delta[1],
        )

        for case, expected_theta in (
            (first_case, 6.0),
            (third_case, 17.0),
        ):
            fibers = case["fiber_geometries"]
            self.assertEqual(len(fibers), 1)
            fiber = fibers[0]
            self.assertEqual(float(fiber["angle theta"]), expected_theta)
            fiber_ports = [
                item
                for item in case["ports"]
                if str(item.get("plane normal", "")).upper() == "Z"
            ]
            self.assertEqual(len(fiber_ports), 1)
            for port in fiber_ports:
                self.assertEqual(float(port["angle theta"]), expected_theta)
                bottom = tuple(map(float, port["fiber bottom center_um"]))
                height = float(port["fiber axis height_um"])
                phi = math.radians(float(port["angle phi"]))
                expected_lateral = height * math.tan(
                    math.radians(expected_theta)
                )
                self.assertAlmostEqual(
                    float(port["center"][0]),
                    bottom[0] + expected_lateral * math.cos(phi),
                )
                self.assertAlmostEqual(
                    float(port["center"][1]),
                    bottom[1] + expected_lateral * math.sin(phi),
                )
                self.assertAlmostEqual(
                    float(port["rotation offset_um"]),
                    4.0
                    * float(fiber["core diameter_um"])
                    * math.tan(math.radians(expected_theta)),
                )
            input_monitors = [
                item
                for item in case["monitors"]
                if item.get("fiber plane role") == "input power measurement"
            ]
            self.assertEqual(len(input_monitors), 1)
            input_monitor = input_monitors[0]
            self.assertEqual(input_monitor["monitor_kind"], "Power monitor")
            self.assertEqual(input_monitor["plane normal"], "Z")
            self.assertEqual(float(input_monitor["angle theta"]), expected_theta)
            self.assertEqual(float(input_monitor["expected propagation sign"]), -1.0)
            self.assertNotIn("rotation offset_um", input_monitor)
            bottom = tuple(map(float, input_monitor["fiber bottom center_um"]))
            height = float(input_monitor["fiber axis height_um"])
            phi = math.radians(float(input_monitor["angle phi"]))
            expected_lateral = height * math.tan(math.radians(expected_theta))
            self.assertAlmostEqual(
                float(input_monitor["center"][0]),
                bottom[0] + expected_lateral * math.cos(phi),
            )
            self.assertAlmostEqual(
                float(input_monitor["center"][1]),
                bottom[1] + expected_lateral * math.sin(phi),
            )

        first_pose_ports = centers_by_name(first_case, "ports")
        third_pose_ports = centers_by_name(third_case, "ports")
        moved_fiber_planes = [
            name
            for name in first_pose_ports
            if "fiber" in name
            and first_pose_ports[name] != third_pose_ports[name]
        ]
        self.assertTrue(moved_fiber_planes)

    def test_gc_soi_fiber_pose_reaches_sequential_and_multigpu_specs(self) -> None:
        factory = _Factory()
        parent = factory.make_component("GC-SOI", 0.0, 0.0)
        parent["params"]["angle_theta"] = 8.0
        factory.components.append(parent)
        factory.components.extend(factory.automatic_simulation_companions(parent))
        nominal_offset = float(parent["params"]["fiber_offset"])
        spec = normalize_lumerical_sweep_spec(
            parent,
            [
                {"parameter": "angle_theta", "values": [8.0, 14.0]},
                {
                    "parameter": "fiber_offset",
                    "values": [nominal_offset, nominal_offset + 1.0],
                },
            ],
        )
        sweep_cases = []
        for values in expand_lumerical_sweep_points(spec):
            components = deepcopy(factory.components)
            variant = next(
                item for item in components if item["uid"] == parent["uid"]
            )
            variant["params"].update(values)
            context = _Factory()
            context.components = components
            context.next_uid = 1 + max(int(item["uid"]) for item in components)
            context.synchronize_automatic_simulation_companions(variant)
            sweep_cases.append({"values": values, "components": components})

        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack(
                "SOI grating coupler (Ansys)"
            ),
            "resource_mode": "GPU",
            "run_after_build": True,
        }
        sequential, _ = generate_lumerical_sweep_notebook(
            sweep_cases, configuration, spec
        )
        multigpu, _ = generate_lumerical_multigpu_sweep_notebook(
            sweep_cases, configuration, spec
        )
        for notebook in (sequential, multigpu):
            self.assertEqual(_notebook_assignment(notebook, "SWEEP_SPEC"), spec)
            self.assertTrue(
                _notebook_assignment(notebook, "SWEEP_RECOMPUTE_MODES")
            )
            cases = _notebook_sweep_cases(notebook)
            self.assertEqual(len(cases), 4)
            for exported, values in zip(cases, expand_lumerical_sweep_points(spec)):
                theta = float(values["angle_theta"])
                self.assertTrue(
                    all(
                        float(fiber["angle theta"]) == theta
                        for fiber in exported["fiber_geometries"]
                    )
                )
                fiber_ports = [
                    port
                    for port in exported["ports"]
                    if str(port.get("plane normal", "")).upper() == "Z"
                ]
                self.assertEqual(len(fiber_ports), 1)
                self.assertTrue(
                    all(float(port["angle theta"]) == theta for port in fiber_ports)
                )
                input_monitors = [
                    monitor
                    for monitor in exported["monitors"]
                    if monitor.get("fiber plane role") == "input power measurement"
                ]
                self.assertEqual(len(input_monitors), 1)
                self.assertEqual(input_monitors[0]["monitor_kind"], "Power monitor")
                self.assertEqual(input_monitors[0]["plane normal"], "Z")
                self.assertEqual(float(input_monitors[0]["angle theta"]), theta)
                self.assertEqual(
                    float(input_monitors[0]["expected propagation sign"]), -1.0
                )

    def test_fiber_pose_is_labeled_and_available_to_sweeps(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                item = _component(kind)
                eligible = {
                    row["parameter"]: row
                    for row in sweepable_component_parameters(item)
                }
                self.assertIn("fiber_offset", eligible)
                self.assertEqual(eligible["fiber_offset"]["label"], "Fiber offset")
                self.assertEqual(eligible["fiber_offset"]["short_name"], "FO")
                self.assertIn("angle_theta", eligible)
                self.assertEqual(eligible["angle_theta"]["label"], "Angle theta")
                self.assertEqual(eligible["angle_theta"]["short_name"], "TH")
                spec = normalize_lumerical_sweep_spec(
                    item,
                    [
                        {"parameter": "angle_theta", "values": [5.0, 12.0]},
                        {"parameter": "fiber_offset", "values": [-1.0, 2.0]},
                    ],
                )
                self.assertEqual(
                    [
                        (axis["parameter"], axis["label"], axis["short_name"])
                        for axis in spec["axes"]
                    ],
                    [
                        ("angle_theta", "Angle theta", "TH"),
                        ("fiber_offset", "Fiber offset", "FO"),
                    ],
                )
                self.assertEqual(spec["point_count"], 4)

        property_source = inspect.getsource(
            NativeLayoutWindow.show_component_properties
        )
        self.assertIn('"Fiber offset (\u00b5m)" if key == "fiber_offset"', property_source)
        self.assertIn('"Angle theta (degrees)" if key == "angle_theta"', property_source)


if __name__ == "__main__":
    unittest.main()
