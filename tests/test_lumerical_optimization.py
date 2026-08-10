from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.lumerical import _LAMBDA_CONNECT_CELL, default_stack
from max_layout.lumerical_optimization import (
    _LUMOPT_RUNTIME_REMOTE,
    _mutated_component_for_snapshot,
    adjoint_optimizable_component_parameters,
    generate_lumerical_adjoint_notebook,
    normalize_lumerical_optimization_spec,
    write_lumerical_adjoint_notebook,
)


def component(kind: str, uid: int = 1) -> dict:
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


def mmi_with_simulation_companions() -> tuple[dict, list[dict]]:
    from max_layout.ui.window import NativeLayoutWindow

    class Factory:
        make_component = NativeLayoutWindow.make_component
        automatic_simulation_companions = (
            NativeLayoutWindow.automatic_simulation_companions
        )

        def __init__(self) -> None:
            self.components: list[dict] = []
            self.next_uid = 1

    factory = Factory()
    mmi = factory.make_component("1x2 MMI", 0.0, 0.0)
    mmi["params"]["add_grating_couplers"] = False
    factory.components.append(mmi)
    factory.components.extend(factory.automatic_simulation_companions(mmi))
    return mmi, factory.components


def grating_with_simulation_companions(kind: str) -> tuple[dict, list[dict]]:
    from max_layout.ui.window import NativeLayoutWindow

    class Factory:
        make_component = NativeLayoutWindow.make_component
        automatic_simulation_companions = (
            NativeLayoutWindow.automatic_simulation_companions
        )

        def __init__(self) -> None:
            self.components: list[dict] = []
            self.next_uid = 1

    factory = Factory()
    grating = factory.make_component(kind, 0.0, 0.0)
    factory.components.append(grating)
    factory.components.extend(factory.automatic_simulation_companions(grating))
    return grating, factory.components


def notebook_assignment(notebook: dict, name: str):
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


class LumericalOptimizationTests(unittest.TestCase):
    def test_live_iteration_rows_report_objective_and_all_parameters(self) -> None:
        tree = ast.parse(_LAMBDA_CONNECT_CELL)
        formatter = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_format_live_optimization_rows"
        )
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[formatter], type_ignores=[])
                ),
                "<live progress formatter>",
                "exec",
            ),
            namespace,
        )
        lines = namespace["_format_live_optimization_rows"](
            [
                {
                    "sequence": 1,
                    "stage": "shape adjoint (lumopt2)",
                    "iteration": 3,
                    "objective": 0.4125,
                    "parameter_names": ["pitch", "fill_factor", "angle_theta"],
                    "parameters": {
                        "pitch": 0.8,
                        "fill_factor": 0.5,
                        "angle_theta": 7.0,
                    },
                }
            ]
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("iteration 3", lines[0])
        self.assertIn("objective=0.4125", lines[0])
        self.assertIn("pitch=0.8", lines[0])
        self.assertIn("fill_factor=0.5", lines[0])
        self.assertIn("angle_theta=7", lines[0])

    def test_only_continuous_device_geometry_is_eligible(self) -> None:
        grating = component("GC-SOI")
        eligible = {
            row["parameter"]: row
            for row in adjoint_optimizable_component_parameters(grating)
        }
        keys = set(eligible)
        self.assertIn("pitch", keys)
        self.assertIn("duty_cycle", keys)
        self.assertIn("fiber_offset", keys)
        self.assertIn("angle_theta", keys)
        self.assertEqual(eligible["fiber_offset"]["label"], "Fiber offset")
        self.assertEqual(eligible["angle_theta"]["label"], "Angle theta")
        self.assertNotIn("target_length", keys)
        self.assertNotIn("h_total", keys)
        self.assertNotIn("etch_depth", keys)
        self.assertNotIn("waveguide_monitor_span_um", keys)
        self.assertFalse(any(isinstance(grating["params"][key], int) for key in keys))

        mmi = component("1x2 MMI")
        mmi_keys = {
            row["parameter"]
            for row in adjoint_optimizable_component_parameters(mmi)
        }
        self.assertIn("mmi_length", mmi_keys)
        self.assertIn("mmi_width", mmi_keys)
        self.assertNotIn("port_sep", mmi_keys)
        self.assertNotIn("output_length", mmi_keys)
        self.assertNotIn("gc_pitch", mmi_keys)
        self.assertNotIn("fdtd_port_clearance_um", mmi_keys)

    def test_apodization_hides_ignored_scalar_filling_factor(self) -> None:
        grating = component("GC-SOI")
        grating["params"]["fill_factors"] = "linspace(0.35, 0.45)"
        keys = {
            row["parameter"]
            for row in adjoint_optimizable_component_parameters(grating)
        }
        self.assertNotIn("duty_cycle", keys)

    def test_normalized_mmi_objective_is_top_output_over_input_target_half(self) -> None:
        mmi = component("1x2 MMI")
        spec = normalize_lumerical_optimization_spec(
            mmi,
            [
                {
                    "parameter": "mmi_length",
                    "initial": -123.0,  # ignored; component JSON is authoritative
                    "minimum": 27.0,
                    "maximum": 31.0,
                }
            ],
            center_wavelength_um=1.30,
            bandwidth_nm=100.0,
            wavelength_points=7,
            max_iterations=22,
        )
        self.assertEqual(spec["parameters"][0]["initial"], 29.0)
        self.assertEqual(spec["objective"]["kind"], "mmi_top_output_over_input")
        self.assertEqual(
            spec["objective"]["description"],
            "Top/upper output branch power divided by input power",
        )
        self.assertEqual(spec["objective"]["target"], 0.5)
        self.assertEqual(spec["objective"]["wavelength_start_um"], 1.25)
        self.assertEqual(spec["objective"]["wavelength_stop_um"], 1.35)
        self.assertEqual(spec["optimizer"]["algorithm"], "L-BFGS-B")
        self.assertEqual(spec["optimizer"]["max_iterations"], 22)
        self.assertEqual(
            spec["objective"]["validation_outputs"],
            [
                "lower_output_over_input",
                "total_output_over_input",
                "upper_lower_imbalance",
            ],
        )

    def test_grating_normalization_freezes_period_count_and_targets_one(self) -> None:
        grating = component("GC-SOI")
        spec = normalize_lumerical_optimization_spec(
            grating,
            [{"parameter": "pitch", "minimum": 0.62, "maximum": 0.74}],
            center_wavelength_um=1.55,
            bandwidth_nm=0.0,
            wavelength_points=99,
            max_iterations=3,
        )
        self.assertEqual(
            spec["fixed_period_count"],
            math.ceil(
                grating["params"]["target_length"] / grating["params"]["pitch"]
            ),
        )
        self.assertEqual(spec["objective"]["kind"], "grating_coupling_efficiency")
        self.assertEqual(spec["objective"]["target"], 1.0)
        self.assertEqual(spec["objective"]["wavelength_points"], 1)
        self.assertFalse(spec["store_all_simulations"])
        self.assertFalse(spec["save_each_fsp"])

    def test_rejects_topology_and_non_pose_simulation_controls(self) -> None:
        grating = component("GC-SOI")
        for parameter in ("target_length", "fiber_core_diameter_um"):
            with self.subTest(parameter=parameter):
                with self.assertRaises(ValueError):
                    normalize_lumerical_optimization_spec(
                        grating,
                        [
                            {
                                "parameter": parameter,
                                "minimum": 1.0,
                                "maximum": 2.0,
                            }
                        ],
                        center_wavelength_um=1.55,
                        bandwidth_nm=0.0,
                    )

    def test_grating_pose_parameters_normalize_for_adjoint_alignment(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                grating = component(kind)
                eligible = {
                    row["parameter"]: row
                    for row in adjoint_optimizable_component_parameters(grating)
                }
                self.assertTrue(
                    {"angle_theta", "fiber_offset"}.issubset(eligible)
                )
                theta = float(grating["params"]["angle_theta"])
                offset = float(grating["params"]["fiber_offset"])
                spec = normalize_lumerical_optimization_spec(
                    grating,
                    [
                        {
                            "parameter": "angle_theta",
                            "minimum": max(0.0, theta - 2.0),
                            "maximum": theta + 2.0,
                        },
                        {
                            "parameter": "fiber_offset",
                            "minimum": offset - 1.0,
                            "maximum": offset + 1.0,
                        },
                    ],
                    center_wavelength_um=1.55,
                    bandwidth_nm=20.0,
                )
                self.assertEqual(
                    [row["parameter"] for row in spec["parameters"]],
                    ["angle_theta", "fiber_offset"],
                )
                self.assertEqual(
                    [row["label"] for row in spec["parameters"]],
                    ["Angle theta", "Fiber offset"],
                )
                self.assertEqual(
                    spec["alignment_parameters"],
                    ["angle_theta", "fiber_offset"],
                )
                self.assertEqual(spec["adjoint_geometry_parameters"], [])
                self.assertEqual(
                    [row["initial"] for row in spec["parameters"]],
                    [theta, offset],
                )

        with self.assertRaisesRegex(ValueError, "below 90"):
            normalize_lumerical_optimization_spec(
                component("GC-SOI"),
                [
                    {
                        "parameter": "angle_theta",
                        "minimum": 0.0,
                        "maximum": 90.0,
                    }
                ],
                center_wavelength_um=1.55,
                bandwidth_nm=0.0,
            )

    def test_generated_grating_adjoint_notebook_synchronizes_full_fiber_pose(self) -> None:
        for kind, stack_name in (
            ("Grating coupler", "TFLN on SiO2"),
            ("GC-SOI", "SOI grating coupler (Ansys)"),
        ):
            with self.subTest(kind=kind):
                grating, components = grating_with_simulation_companions(kind)
                theta = float(grating["params"]["angle_theta"])
                offset = float(grating["params"]["fiber_offset"])
                spec = normalize_lumerical_optimization_spec(
                    grating,
                    [
                        {
                            "parameter": "angle_theta",
                            "minimum": max(0.0, theta - 2.0),
                            "maximum": theta + 2.0,
                        },
                        {
                            "parameter": "fiber_offset",
                            "minimum": offset - 1.0,
                            "maximum": offset + 1.0,
                        },
                    ],
                    center_wavelength_um=1.55,
                    bandwidth_nm=0.0,
                    max_iterations=2,
                )
                configuration = {
                    "material_stack": default_stack(stack_name),
                    "included_layers": [(1, 0), (2, 0)],
                    "resource_mode": "GPU",
                    "run_after_build": True,
                    "project_file": f"{kind.lower().replace(' ', '_')}_pose.fsp",
                }
                notebook, _warnings = generate_lumerical_adjoint_notebook(
                    components, configuration, spec
                )
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        compile(
                            "".join(cell["source"]),
                            f"grating-pose-cell-{index}",
                            "exec",
                        )

                pose = notebook_assignment(notebook, "OPT_FIBER_POSE")
                self.assertEqual(
                    pose["active_parameters"],
                    ["angle_theta", "fiber_offset"],
                )
                self.assertEqual(pose["component_kind"], kind)
                self.assertEqual(pose["nominal_angle_theta"], theta)
                self.assertEqual(pose["nominal_fiber_offset"], offset)
                self.assertTrue(pose["fiber_name"])
                self.assertEqual(len(pose["ports"]), 2)
                self.assertEqual(
                    sum(bool(port["is_source"]) for port in pose["ports"]), 1
                )
                for port in pose["ports"]:
                    self.assertEqual(port["candidate_mode_numbers"], [1, 2])

                source = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn("OPT_FIBER_POSE", source)
                self.assertIn("def _fiber_pose_updates", source)
                self.assertIn("_synchronize_resolved_fiber_mode_contract()", source)
                self.assertIn("_require_numeric_source_mode()", source)
                self.assertIn('"selected_mode_order"', source)
                self.assertIn("np.asarray(mode_order, dtype=int)", source)
                self.assertIn('globals().get("_select_fiber_local_te_mode")', source)
                self.assertIn("angle_theta", source)
                self.assertIn("fiber_offset", source)
                for property_name in (
                    '"x"',
                    '"y"',
                    '"z"',
                    '"theta"',
                    '"rotation offset"',
                ):
                    self.assertIn(property_name, source)

    def test_runtime_propagates_resolved_winner_first_fiber_mode_pair(self) -> None:
        helper_names = {
            "_numeric_source_mode_label",
            "_require_numeric_source_mode",
            "_positive_unique_modes",
            "_resolved_runtime_port_mode",
            "_synchronize_resolved_fiber_mode_contract",
        }
        parsed = ast.parse(_LUMOPT_RUNTIME_REMOTE)
        helper_nodes = [
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helper_nodes}, helper_names)
        namespace = {
            "re": re,
            "OPT_OBJECTIVE_PORTS": {
                "kind": "grating coupling efficiency",
                "source_port": "fiber_source",
                "source_mode": "auto local TE",
            },
            "OPT_FIBER_POSE": {
                "ports": [
                    {
                        "name": "fiber_source",
                        "is_source": True,
                        "mode_number": 0,
                        "selected_mode_order": [],
                        "candidate_mode_numbers": [1, 2],
                    },
                    {
                        "name": "fiber_measurement",
                        "is_source": False,
                        "mode_number": 0,
                        "selected_mode_order": [],
                        "candidate_mode_numbers": [1, 2],
                    },
                ]
            },
            "PORT_MODE_SELECTIONS": {
                "fiber_source": {
                    "mode number": 2,
                    "selected mode order": [2, 1],
                    "candidate mode numbers": [1, 2],
                },
                "fiber_measurement": {
                    "mode number": 1,
                    "selected mode order": [1, 2],
                    "candidate mode numbers": [1, 2],
                },
            },
            "PORTS": [],
            "GRATING_ANALYSIS": {"fiber_port_name": "fiber_source"},
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=helper_nodes, type_ignores=[])),
                "<fiber-mode-contract-helpers>",
                "exec",
            ),
            namespace,
        )

        with self.assertRaisesRegex(RuntimeError, "numeric 'mode N'"):
            namespace["_numeric_source_mode_label"]("auto local TE")
        self.assertEqual(namespace["_numeric_source_mode_label"]("Mode 2"), "mode 2")

        namespace["_synchronize_resolved_fiber_mode_contract"]()
        self.assertEqual(
            namespace["OPT_OBJECTIVE_PORTS"]["source_mode"], "mode 2"
        )
        self.assertEqual(
            namespace["OPT_FIBER_POSE"]["ports"][0]["selected_mode_order"],
            [2, 1],
        )
        self.assertEqual(
            namespace["OPT_FIBER_POSE"]["ports"][1]["selected_mode_order"],
            [1, 2],
        )
        self.assertEqual(
            namespace["GRATING_ANALYSIS"]["fiber_source_mode"], "mode 2"
        )
        self.assertEqual(
            namespace["GRATING_ANALYSIS"]["fiber_source_selected_mode_order"],
            [2, 1],
        )

        apply_node = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_fiber_pose_to_session"
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[apply_node], type_ignores=[])),
                "<fiber-pose-mode-refresh>",
                "exec",
            ),
            namespace,
        )

        namespace["UM"] = 1e-6
        namespace["np"] = np
        namespace["_set_named"] = (
            lambda session, path, property_name, value:
            session.setnamed(path, property_name, value)
        )
        namespace["_fiber_pose_updates"] = lambda *_args, **_kwargs: {
            "fiber": {
                "name": "fiber",
                "center_um": [0.0, 0.0],
                "theta_deg": 7.0,
            },
            "ports": [
                {
                    "name": "fiber_source",
                    "center_um": [0.0, 0.0],
                    "z_um": 1.0,
                    "theta_deg": 7.0,
                    "phi_deg": 0.0,
                    "rotation_offset_um": 1.0,
                    "mode_number": 2,
                    "selected_mode_order": [2, 1],
                    "candidate_mode_numbers": [1, 2],
                    "mode_degeneracy_tolerance": 0.01,
                    "minimum_local_te_fraction": 0.8,
                    "is_source": True,
                },
                {
                    "name": "fiber_measurement",
                    "center_um": [0.0, 0.0],
                    "z_um": 0.9,
                    "theta_deg": 7.0,
                    "phi_deg": 0.0,
                    "rotation_offset_um": 1.0,
                    "mode_number": 1,
                    "selected_mode_order": [1, 2],
                    "candidate_mode_numbers": [1, 2],
                    "mode_degeneracy_tolerance": 0.01,
                    "minimum_local_te_fraction": 0.8,
                    "is_source": False,
                },
            ],
        }

        class FakeFdtdSession:
            def __init__(self) -> None:
                self.mode_updates: list[list[int]] = []
                self.group_settings: dict[str, str] = {}

            def switchtolayout(self) -> None:
                pass

            def setnamed(self, _path, _property_name, _value) -> None:
                pass

            def runsetup(self) -> None:
                pass

            def select(self, _path) -> None:
                pass

            def updateportmodes(self, mode_order):
                self.mode_updates.append(np.asarray(mode_order, dtype=int).tolist())
                return None

            def set(self, property_name, value) -> None:
                self.group_settings[str(property_name)] = str(value)

        session = FakeFdtdSession()
        namespace["_apply_fiber_pose_to_session"]([], session, update_modes=True)
        self.assertEqual(session.mode_updates, [[2, 1], [1, 2]])
        self.assertEqual(session.group_settings["source mode"], "mode 2")

        selector_calls = []

        def reselect_after_pose(_session, port_path, port):
            selector_calls.append((port_path, dict(port)))
            if str(port["name"]) == "fiber_source":
                return {
                    "mode number": 1,
                    "selected mode order": [1, 2],
                    "candidate mode numbers": [1, 2],
                    "polarization": "local TE",
                }
            return {
                "mode number": 2,
                "selected mode order": [2, 1],
                "candidate mode numbers": [1, 2],
                "polarization": "local TE",
            }

        namespace["_select_fiber_local_te_mode"] = reselect_after_pose
        rescored_session = FakeFdtdSession()
        namespace["_apply_fiber_pose_to_session"](
            [], rescored_session, update_modes=True
        )
        self.assertEqual(len(selector_calls), 2)
        self.assertEqual(
            namespace["OPT_OBJECTIVE_PORTS"]["source_mode"], "mode 1"
        )
        self.assertEqual(rescored_session.group_settings["source mode"], "mode 1")
        self.assertEqual(
            namespace["OPT_FIBER_POSE"]["ports"][0]["selected_mode_order"],
            [1, 2],
        )
        self.assertEqual(
            namespace["OPT_FIBER_POSE"]["ports"][1]["selected_mode_order"],
            [2, 1],
        )

    def test_mmi_longitudinal_snapshot_keeps_receiver_endpoint_fixed(self) -> None:
        mmi = component("1x2 MMI")
        nominal = mmi["params"]
        nominal_total = sum(
            float(nominal[name])
            for name in (
                "input_length",
                "input_taper_length",
                "mmi_length",
                "output_taper_length",
                "output_length",
            )
        )
        changed = _mutated_component_for_snapshot(
            mmi, "mmi_length", float(nominal["mmi_length"]) + 2.0, None
        )
        changed_total = sum(
            float(changed["params"][name])
            for name in (
                "input_length",
                "input_taper_length",
                "mmi_length",
                "output_taper_length",
                "output_length",
            )
        )
        self.assertAlmostEqual(changed_total, nominal_total)
        self.assertAlmostEqual(
            changed["params"]["output_length"],
            float(nominal["output_length"]) - 2.0,
        )

    def test_generated_notebook_is_true_adjoint_and_code_cells_compile(self) -> None:
        mmi, components = mmi_with_simulation_companions()
        spec = normalize_lumerical_optimization_spec(
            mmi,
            [{"parameter": "mmi_length", "minimum": 27.0, "maximum": 31.0}],
            center_wavelength_um=1.30,
            bandwidth_nm=100.0,
            wavelength_points=5,
            max_iterations=2,
        )
        configuration = {
            "material_stack": default_stack("TFLN on SiO2"),
            "included_layers": [(1, 0)],
            "resource_mode": "GPU",
            "run_after_build": True,
            "project_file": "mmi_optimized.fsp",
        }
        notebook, warnings = generate_lumerical_adjoint_notebook(
            components, configuration, spec
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"optimization-cell-{index}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        first_source = "".join(notebook["cells"][0]["source"])
        self.assertIn("RUN_SIMULATION = True", first_source)
        self.assertIn(
            "One pre-solve inspection FSP and one solved/best FSP are always stored.",
            first_source,
        )
        for removed_name in (
            "MODEL_CACHE_KEY",
            "MODEL_CACHE_HIT",
            "REMOTE_MODEL_CACHE_FSP",
            "REUSE_EXACT_MODEL_CACHE",
            "SAVE_EXACT_MODEL_CACHE_ON_MISS",
        ):
            self.assertNotIn(removed_name, source)
        self.assertIn("SETTINGS['save_inspection_fsp'] = True", source)
        self.assertIn("SETTINGS['save_final_fsp'] = True", source)
        self.assertIn("save_verified_project(REMOTE_INSPECTION_PROJECT_FILE)", source)
        self.assertIn(
            "REMOTE_INTERNAL_SEED_FSP = REMOTE_INSPECTION_PROJECT_FILE", source
        )

        # v261+ official API is primary, with a verified genuine-adjoint legacy
        # fallback.  There is no sweep fallback.
        self.assertIn("ansys.lumerical.core.lumopt2", source)
        self.assertIn("import lumopt2 as module", source)
        self.assertIn("module.Parametrization", source)
        self.assertIn('module.LocalRunner(resource="GPU")', source)
        self.assertIn("ParameterizedGeometry", source)
        self.assertIn("PortTransmission", source)
        self.assertIn('method="L-BFGS-B"', source)
        self.assertIn("store_all_simulations=False", source)
        self.assertNotIn("parameter sweep fallback", source.lower())

        optimize_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "REMOTE_LUMOPT_RUNTIME =" in "".join(cell["source"])
        )
        captured = {}

        def capture_remote(code, label, timeout, progress_file=None):
            captured.update(
                code=code,
                label=label,
                timeout=timeout,
                progress_file=progress_file,
            )

        exec(
            compile(optimize_cell, "<generated adjoint optimize cell>", "exec"),
            {
                "REMOTE_BASE_FSP": "/remote/fsp/seed.fsp",
                "REMOTE_WORK": "/remote/work",
                "SETTINGS": {"run_after_build": True},
                "solve_remote_checked": capture_remote,
            },
        )
        self.assertTrue(
            captured["code"].startswith(
                "REMOTE_BASE_FSP = '/remote/fsp/seed.fsp'\n"
            )
        )
        self.assertIn("REMOTE_OPTIMIZER_BASE_FSP = REMOTE_BASE_FSP", captured["code"])
        self.assertEqual(
            captured["progress_file"],
            "/remote/work/adjoint_live_progress.jsonl",
        )
        self.assertIn("callback=_alignment_iteration_callback", captured["code"])
        self.assertIn("_start_shape_history_monitor", captured["code"])
        self.assertIn("legacy_reporting_callback", captured["code"])
        self.assertIn("_emit_live_progress", captured["code"])
        self.assertNotIn("fdtd_session.run", _LUMOPT_RUNTIME_REMOTE.split(
            "def _alignment_iteration_callback", 1
        )[1].split("def _optimize_fiber_alignment", 1)[0])

        disabled_calls = []
        exec(
            compile(optimize_cell, "<disabled adjoint optimize cell>", "exec"),
            {
                "REMOTE_BASE_FSP": None,
                "REMOTE_WORK": "/remote/work",
                "SETTINGS": {"run_after_build": False},
                "solve_remote_checked": lambda *args, **kwargs: disabled_calls.append(
                    (args, kwargs)
                ),
            },
        )
        self.assertEqual(disabled_calls, [])

        # One seed owner closes before LumOpt.  LumOpt's internal handoff is
        # transient, while the public best-design FSP is always retained.
        self.assertLess(source.index("Seed FDTD owner closed"), source.index("_run_lumopt2"))
        self.assertIn("project.save_project(REMOTE_VALIDATION_FSP, params=best_parameters)", source)
        self.assertIn("ADJOINT_FDTD_OWNER.save(REMOTE_BEST_FSP)", source)
        self.assertIn('"best_fsp": REMOTE_BEST_FSP', source)
        self.assertIn("REMOTE_INTERNAL_SEED_FSP", source)
        self.assertIn("REMOTE_VALIDATION_FSP", source)
        self.assertIn('globals().get("REMOTE_RUNTIME_PROJECT_FILE", "")', source)
        self.assertIn("adjoint_optimization_history.npz", source)
        self.assertIn("adjoint_optimization_summary.json", source)
        self.assertIn('REMOTE_OPT_TEXT_SUMMARY = os.path.join(REMOTE_WORK, "summary.txt")', source)
        self.assertIn("Exact nominal source parameters (JSON)", source)
        self.assertIn("complete_best_parameters = dict(OPT_COMPONENT_NOMINAL_PARAMS)", source)
        self.assertIn("complete_best_parameters.update(patch_parameters)", source)
        self.assertIn("Complete best-design geometry", source)
        self.assertIn("Complete best-design source parameters (JSON)", source)
        self.assertIn('"complete_best_parameters": complete_best_parameters', source)
        self.assertIn('"mmi_width": ("MMI width", "um")', source)
        self.assertIn('"taper_power": ("MMI taper profile exponent", "")', source)
        self.assertIn('"input_reference_before_taper_um": ("Input power-reference distance before taper", "um")', source)
        self.assertIn('"fiber_power_monitor_below_source_um": ("Fiber power-plane distance below source", "um")', source)
        optimization_headings = (
            "PROJECT",
            "PARAMETERS",
            "OBJECTIVE AND BOUNDS",
            "MATERIAL STACK AND MESH",
            "SIMULATION SETTINGS",
            "SOURCES / PORTS / MONITORS",
            "OPTIMIZATION SETTINGS",
            "RESULTS SUMMARY",
            "OUTPUT FILES",
            "WARNINGS / NOTES",
            "FSP PROVENANCE",
        )
        for heading in optimization_headings:
            self.assertIn('"%s"' % heading, source)
        heading_positions = [source.index('"%s"' % heading) for heading in optimization_headings]
        self.assertEqual(heading_positions, sorted(heading_positions))
        self.assertIn("Optimization parameter bounds:", source)
        self.assertIn("Best-design forward validation (linear power)", source)
        self.assertIn("Complete editor parameter patch", source)
        self.assertIn("maximum iterations", source)
        self.assertIn("dt factor", source)
        self.assertIn('REMOTE_WORK + "/summary.txt"', source)
        self.assertIn("adjoint_parameter_patch.json", source)
        self.assertIn("mmi_top_output_over_input", source)
        self.assertIn("mmi_lower_output_over_input", source)
        self.assertIn("mmi_total_output_over_input", source)
        self.assertIn("mmi_upper_lower_imbalance", source)
        self.assertIn('patch_parameters["output_length"]', source)
        self.assertIn('ADJOINT_FDTD_OWNER.run("FDTD", "GPU")', source)
        self.assertIn("'target': 0.5", source)
        self.assertIn("CPU post-processing", source)
        self.assertIn("LicensingSettings web shared products checkin", source)

        metadata = notebook["metadata"]["max_layout"]
        self.assertEqual(metadata["export"], "lumerical-lumopt-shape-adjoint")
        self.assertEqual(metadata["objective"], "mmi_top_output_over_input")
        self.assertFalse(metadata["store_all_simulations"])
        self.assertTrue(any("no per-iteration FSP" in warning for warning in warnings))

    def test_writer_outputs_nbformat_json(self) -> None:
        mmi, components = mmi_with_simulation_companions()
        spec = normalize_lumerical_optimization_spec(
            mmi,
            [{"parameter": "mmi_length", "minimum": 28.0, "maximum": 30.0}],
            center_wavelength_um=1.30,
            bandwidth_nm=0.0,
            wavelength_points=1,
            max_iterations=1,
        )
        configuration = {
            "material_stack": default_stack("TFLN on SiO2"),
            "included_layers": [(1, 0)],
            "resource_mode": "GPU",
            "run_after_build": True,
            "project_file": "mmi_best.fsp",
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mmi_optimization.ipynb"
            warnings = write_lumerical_adjoint_notebook(
                path, components, configuration, spec
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(warnings)
        self.assertEqual(payload["nbformat"], 4)
        self.assertEqual(
            payload["metadata"]["max_layout"]["dimension"], "3D"
        )


if __name__ == "__main__":
    unittest.main()
