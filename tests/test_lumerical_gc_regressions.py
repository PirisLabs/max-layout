from __future__ import annotations

import ast
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import numpy as np

from max_layout import lumerical
from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.ui.window import NativeLayoutWindow


def _assignment_value(notebook: dict, name: str):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"No notebook assignment named {name!r}")


def _cell_index(notebook: dict, needle: str) -> int:
    for index, cell in enumerate(notebook["cells"]):
        if needle in "".join(cell.get("source", [])):
            return index
    raise AssertionError(f"No notebook cell contains {needle!r}")


def _runtime_functions(*names: str) -> dict[str, object]:
    wanted = set(names)
    nodes = [
        node
        for node in ast.parse(lumerical._BUILD_CELL).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        missing = wanted - {node.name for node in nodes}
        raise AssertionError(f"Missing runtime helpers: {sorted(missing)}")
    namespace: dict[str, object] = {"np": np}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<lumerical-runtime-helpers>", "exec"), namespace)
    return namespace


class _CompanionFactory:
    make_component = NativeLayoutWindow.make_component
    automatic_simulation_companions = (
        NativeLayoutWindow.automatic_simulation_companions
    )

    def __init__(self) -> None:
        self.components: list[dict] = []
        self.next_uid = 1


def _grating_notebook(angle_theta: float = 13.25) -> tuple[dict, list[str]]:
    factory = _CompanionFactory()
    grating = factory.make_component("GC-SOI", 0.0, 0.0)
    factory.components.append(grating)
    companions = factory.automatic_simulation_companions(grating)
    factory.components.extend(companions)

    grating["params"]["angle_theta"] = angle_theta
    stale_tilts = iter((7.0, 8.0, 9.0))
    for companion in companions:
        if companion["kind"] in {"Fiber geometry", "Fiber-axis FDTD port"}:
            companion["params"]["angle theta"] = next(stale_tilts)

    return lumerical.generate_lumerical_notebook(
        factory.components,
        {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack(
                "SOI grating coupler (Ansys)"
            ),
            "run_after_build": True,
            "project_file": "gc_angle_audit.fsp",
        },
    )


class LumericalGratingRegressionTests(unittest.TestCase):
    def test_parent_angle_is_authoritative_for_fiber_and_both_planes(self) -> None:
        theta_deg = 13.25
        notebook, warnings = _grating_notebook(theta_deg)
        fibers = _assignment_value(notebook, "FIBER_GEOMETRIES")
        ports = _assignment_value(notebook, "PORTS")

        self.assertEqual(len(fibers), 1)
        fiber = fibers[0]
        fiber_ports = [port for port in ports if port["plane normal"] == "Z"]
        self.assertEqual(len(fiber_ports), 2)
        self.assertAlmostEqual(fiber["angle theta"], theta_deg)
        self.assertEqual(
            {round(float(port["angle theta"]), 12) for port in fiber_ports},
            {theta_deg},
        )
        self.assertTrue(
            any("parent angle_theta 13.25" in warning for warning in warnings)
        )

        # Each plane's exported center must lie on the same tilted fiber axis.
        # This also catches using the parent angle for the geometry while a
        # stale 7/10-degree value is still used for a port position.
        for port in fiber_ports:
            axis_height = float(port["fiber axis height_um"])
            expected_axis_offset = axis_height * math.tan(math.radians(theta_deg))
            delta_x = float(port["center"][0]) - float(fiber["center"][0])
            delta_y = float(port["center"][1]) - float(fiber["center"][1])
            self.assertAlmostEqual(math.hypot(delta_x, delta_y), expected_axis_offset)

    def test_three_mode_fiber_search_selects_rotation_aware_gaussian_he11(self) -> None:
        helpers = _runtime_functions(
            "_mode_profile_vector",
            "_fiber_local_te_score",
            "_fiber_gaussian_circular_scores",
            "_fiber_candidate_neff",
            "_select_fiber_local_te_mode",
        )
        select_mode = helpers["_select_fiber_local_te_mode"]

        coordinate = np.linspace(-1.0, 1.0, 31)
        x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="ij")
        circular_gaussian = np.exp(-(x_grid**2 + y_grid**2) / 0.2)
        elongated_field = np.exp(-(x_grid**2 / 0.05 + y_grid**2 / 0.8))

        def field(amplitude: np.ndarray, component: int) -> np.ndarray:
            result = np.zeros((*amplitude.shape, 3), dtype=complex)
            result[..., component] = amplitude
            return result

        profiles = {
            "E1": field(circular_gaussian, 0),
            "E2": field(circular_gaussian, 1),
            # This is Ey-dominant but deliberately not the near-degenerate,
            # circular Gaussian fiber partner.
            "E3": field(elongated_field, 1),
        }

        class FakeFdtd:
            def __init__(self) -> None:
                self.mode_updates: list[list[int]] = []

            def select(self, _path: str) -> None:
                return None

            def updateportmodes(self, modes):
                self.mode_updates.append(np.asarray(modes, dtype=int).tolist())
                return None

            def getresult(self, _path: str, result_name: str):
                if result_name == "mode profiles":
                    return profiles
                if result_name == "neff":
                    return {
                        "neff1": np.asarray([1.4440]),
                        "neff2": np.asarray([1.4450]),
                        "neff3": np.asarray([1.5100]),
                    }
                raise AssertionError(result_name)

        base_port = {
            "name": "fiber_source",
            "candidate mode numbers": [1, 2, 3],
            "fiber target neff": 1.4445,
            "mode degeneracy tolerance": 0.01,
            "minimum local TE fraction": 0.8,
        }
        expectations = ((0.0, 2), (90.0, 1))
        for phi_deg, expected_mode in expectations:
            with self.subTest(phi_deg=phi_deg):
                fdtd = FakeFdtd()
                selected = select_mode(
                    fdtd,
                    "FDTD::ports::fiber_source",
                    {**base_port, "angle phi": phi_deg},
                )
                self.assertEqual(fdtd.mode_updates, [[1, 2, 3]])
                self.assertEqual(selected["candidate mode numbers"], [1, 2, 3])
                self.assertEqual(selected["degenerate mode pair"], [1, 2])
                self.assertEqual(selected["mode number"], expected_mode)
                self.assertGreater(selected["gaussian scores"]["2"], 0.95)
                self.assertGreater(selected["circularity scores"]["2"], 0.95)
                self.assertLess(
                    selected["circularity scores"]["3"],
                    selected["circularity scores"]["2"],
                )

        # The editor defaults and serialized notebook must preserve the same
        # three-candidate contract used by the runtime selector.
        self.assertEqual(
            DEFAULT_COMPONENT_VALUES["Fiber-axis FDTD port"][
                "candidate mode numbers"
            ],
            [1, 2, 3],
        )
        notebook, _warnings = _grating_notebook()
        fiber_ports = [
            port
            for port in _assignment_value(notebook, "PORTS")
            if port["plane normal"] == "Z"
        ]
        self.assertTrue(fiber_ports)
        self.assertTrue(
            all(port["candidate mode numbers"] == [1, 2, 3] for port in fiber_ports)
        )
        self.assertEqual(
            _assignment_value(notebook, "GRATING_ANALYSIS")[
                "fiber_mode_candidates"
            ],
            [1, 2, 3],
        )

    def test_sweep_mode_refresh_keeps_the_three_candidate_contract(self) -> None:
        wanted = {
            "_sweep_mode_profile_vector",
            "_sweep_candidate_neff",
            "_sweep_gaussian_circular_scores",
            "_sweep_reselect_fiber_local_te",
        }
        nodes = [
            node
            for node in ast.parse(lumerical._SWEEP_RUNTIME_REMOTE).body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        self.assertEqual({node.name for node in nodes}, wanted)

        ex_mode = np.zeros((9, 9, 3), dtype=complex)
        ey_mode = np.zeros((9, 9, 3), dtype=complex)
        third_mode = np.zeros((9, 9, 3), dtype=complex)
        ex_mode[..., 0] = 1.0
        ey_mode[..., 1] = 1.0
        third_mode[..., 1] = 1.0

        class FakeFdtd:
            def __init__(self) -> None:
                self.mode_updates: list[list[int]] = []

            def select(self, _path: str) -> None:
                return None

            def updateportmodes(self, modes) -> None:
                self.mode_updates.append(np.asarray(modes, dtype=int).tolist())

            def getresult(self, _path: str, result_name: str):
                if result_name == "mode profiles":
                    return {"E1": ex_mode, "E2": ey_mode, "E3": third_mode}
                if result_name == "neff":
                    return {
                        "neff1": np.asarray([1.4440]),
                        "neff2": np.asarray([1.4450]),
                        "neff3": np.asarray([1.5100]),
                    }
                raise AssertionError(result_name)

        fake_fdtd = FakeFdtd()
        namespace = {"np": np, "fdtd": fake_fdtd}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=nodes, type_ignores=[])
                ),
                "<sweep-three-mode-refresh>",
                "exec",
            ),
            namespace,
        )
        selected = namespace["_sweep_reselect_fiber_local_te"](
            "FDTD::ports::fiber_source",
            {
                "name": "fiber_source",
                "angle phi": 0.0,
                "candidate mode numbers": [1, 2, 3],
                "fiber target neff": 1.4445,
                "mode degeneracy tolerance": 0.01,
                "minimum local TE fraction": 0.8,
            },
            {"candidate mode numbers": [1, 2, 3]},
        )
        self.assertEqual(fake_fdtd.mode_updates, [[1, 2, 3]])
        self.assertEqual(selected["candidate mode numbers"], [1, 2, 3])
        self.assertEqual(selected["degenerate mode pair"], [1, 2])
        self.assertEqual(selected["mode number"], 2)
        self.assertEqual(selected["selected mode order"], [2, 1, 3])

    def test_waveguide_neff_is_derived_from_dispersive_stack_indices(self) -> None:
        notebook, _warnings = _grating_notebook()
        grating_analysis = _assignment_value(notebook, "GRATING_ANALYSIS")
        waveguide_mode_monitor = next(
            monitor
            for monitor in _assignment_value(notebook, "MONITORS")
            if monitor.get("grating_monitor_role") == "waveguide_mode_expansion"
        )
        # Zero is an explicit unresolved sentinel in the serialized payload;
        # the live material database supplies the target after getindex().
        self.assertEqual(grating_analysis["waveguide_target_neff"], 0.0)
        self.assertEqual(waveguide_mode_monitor["target neff"], 0.0)
        self.assertIn("actual core", waveguide_mode_monitor["target neff strategy"])

        helper_nodes = [
            node
            for node in ast.parse(lumerical._BUILD_CELL).body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_maximum_material_index",
                "_derive_waveguide_neff_from_stack",
            }
        ]
        self.assertEqual(len(helper_nodes), 2)

        class FakeFdtd:
            def __init__(self, indices: dict[str, list[float]]) -> None:
                self.indices = indices
                self.calls: list[tuple[str, float]] = []

            def getindex(self, material: str, frequency_hz: float):
                self.calls.append((material, frequency_hz))
                return np.asarray(self.indices[material], dtype=float)

        cases = (
            ("Si", [3.48], 2.46),
            ("LiNbO3", [2.14, 2.21, 2.18], 1.825),
        )
        for core_material, core_indices, expected_target in cases:
            with self.subTest(core_material=core_material):
                namespace = {
                    "np": np,
                    "UM": 1e-6,
                    "SETTINGS": {
                        "wavelength_start_um": 1.25,
                        "wavelength_stop_um": 1.35,
                    },
                    "GRATING_ANALYSIS": {"component_uid": 42},
                    "MMI_ANALYSIS": None,
                    "GEOMETRY": [{"component_uid": 42, "layer": 1}],
                }
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=deepcopy(helper_nodes), type_ignores=[])
                        ),
                        "<dynamic-waveguide-neff>",
                        "exec",
                    ),
                    namespace,
                )
                fdtd = FakeFdtd(
                    {core_material: core_indices, "SiO2": [1.44]}
                )
                z_ranges = [
                    (
                        {
                            "name": "BOX",
                            "material": "SiO2",
                            "role": "background",
                            "gds_layer": 0,
                        },
                        -2.0,
                        0.0,
                    ),
                    (
                        {
                            "name": "device",
                            "material": core_material,
                            "role": "geometry",
                            "gds_layer": 1,
                        },
                        0.0,
                        0.4,
                    ),
                    (
                        {
                            "name": "cladding",
                            "material": "SiO2",
                            "role": "background",
                            "gds_layer": 0,
                        },
                        0.4,
                        1.4,
                    ),
                ]
                result = namespace["_derive_waveguide_neff_from_stack"](
                    fdtd, z_ranges
                )
                self.assertAlmostEqual(result["target_neff"], expected_target)
                self.assertAlmostEqual(
                    result["target_neff"],
                    0.5 * (result["core_index"] + result["surrounding_index"]),
                )
                self.assertEqual(result["surrounding_index"], 1.44)
                self.assertEqual(
                    {material for material, _frequency in fdtd.calls},
                    {core_material, "SiO2"},
                )
                expected_frequency = 299792458.0 / (1.30e-6)
                self.assertTrue(
                    all(
                        math.isclose(frequency, expected_frequency)
                        for _material, frequency in fdtd.calls
                    )
                )

    def test_solved_fsp_is_verified_and_fetched_before_grating_analysis(self) -> None:
        notebook, _warnings = _grating_notebook()
        solve_index = _cell_index(notebook, "solve_remote_checked(_solve_code")
        save_index = _cell_index(notebook, "LOCAL_FINAL_PROJECT_FILE")
        analysis_index = _cell_index(notebook, "REMOTE_GRATING_ANALYSIS")
        fetch_index = _cell_index(notebook, "REMOTE_ARTIFACTS")
        self.assertLess(solve_index, save_index)
        self.assertLess(save_index, analysis_index)
        self.assertLess(save_index, fetch_index)

        save_source = "".join(notebook["cells"][save_index]["source"])
        self.assertIn("save_verified_project(REMOTE_PROJECT_FILE)", save_source)
        self.assertIn("lam.fetch(REMOTE_PROJECT_FILE", save_source)
        self.assertIn("PIRIS_FSP_DIR / os.path.basename(REMOTE_PROJECT_FILE)", save_source)
        result_fetch_source = "".join(notebook["cells"][fetch_index]["source"])
        self.assertNotIn("REMOTE_ARTIFACTS.insert(0, REMOTE_PROJECT_FILE)", result_fetch_source)
        result_save_index = _cell_index(notebook, "REMOTE_RESULTS_SAVER")
        result_save_source = "".join(notebook["cells"][result_save_index]["source"])
        self.assertNotIn("save_verified_project(REMOTE_PROJECT_FILE)", result_save_source)
        self.assertIn("REMOTE_FINAL_FSP_SAVED", result_save_source)
        self.assertIn("os.path.getsize(REMOTE_PROJECT_FILE)", result_save_source)

        inspection_index = _cell_index(notebook, "REMOTE_INSPECTION_PROJECT_FILE")
        inspection_source = "".join(notebook["cells"][inspection_index]["source"])
        self.assertIn('REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")', inspection_source)
        self.assertIn(
            "REMOTE_PROJECT_FILE = os.path.join(REMOTE_FSP_DIR, _project_name)",
            inspection_source,
        )
        self.assertIn(
            "REMOTE_INSPECTION_PROJECT_FILE = os.path.join(", inspection_source
        )

    def test_model_build_clamps_and_applies_cpu_threads_before_geometry(self) -> None:
        build_function = next(
            node
            for node in ast.parse(lumerical._BUILD_CELL).body
            if isinstance(node, ast.FunctionDef) and node.name == "build_simulation"
        )
        prefix: list[ast.stmt] = []
        for statement in build_function.body:
            is_material_start = (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "_add_required_materials"
            )
            if is_material_start:
                break
            prefix.append(deepcopy(statement))
        else:
            self.fail("Could not isolate build resource setup before materials")

        prefix.append(ast.Return(value=ast.Name(id="build_cpu_threads", ctx=ast.Load())))
        probe = ast.FunctionDef(
            name="probe_build_resources",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=prefix,
            decorator_list=[],
        )

        class FakeFdtd:
            def __init__(self) -> None:
                self.resource_calls: list[tuple] = []

            def setresource(self, *args) -> None:
                self.resource_calls.append(args)

        for available_cores, requested, expected in ((8, 30, 8), (64, 30, 30)):
            with self.subTest(
                available_cores=available_cores, requested=requested
            ):
                fake_fdtd = FakeFdtd()
                launch_calls: list[dict] = []

                def launch_fdtd(**kwargs):
                    launch_calls.append(kwargs)
                    return fake_fdtd

                namespace = {
                    "SETTINGS": {
                        "build_cpu_threads": requested,
                        "hide_cad": True,
                    },
                    "os": SimpleNamespace(cpu_count=lambda: available_cores),
                    "time": __import__("time"),
                    "lumapi": SimpleNamespace(FDTD=launch_fdtd),
                }
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=[deepcopy(probe)], type_ignores=[])
                        ),
                        "<cpu-build-resource-probe>",
                        "exec",
                    ),
                    namespace,
                )
                selected_threads = namespace["probe_build_resources"]()
                self.assertEqual(selected_threads, expected)
                self.assertEqual(
                    launch_calls,
                    [{"hide": True, "serverArgs": {"threads": str(expected)}}],
                )
                self.assertEqual(
                    fake_fdtd.resource_calls,
                    [
                        ("FDTD", 1, "device type", "CPU"),
                        ("FDTD", 1, "active", True),
                        ("FDTD", 1, "processes", 1),
                        ("FDTD", 1, "threads", expected),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
