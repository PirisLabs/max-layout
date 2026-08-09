from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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

                source = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn("OPT_FIBER_POSE", source)
                self.assertIn("def _fiber_pose_updates", source)
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

        # One seed owner closes before LumOpt; one best design is retained.
        self.assertLess(source.index("Seed FDTD owner closed"), source.index("_run_lumopt2"))
        self.assertIn("project.save_project(REMOTE_BEST_FSP, params=best_parameters)", source)
        self.assertIn("adjoint_optimization_history.npz", source)
        self.assertIn("adjoint_optimization_summary.json", source)
        self.assertIn('REMOTE_OPT_TEXT_SUMMARY = os.path.join(REMOTE_WORK, "summary.txt")', source)
        self.assertIn("Exact nominal source parameters (JSON)", source)
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
