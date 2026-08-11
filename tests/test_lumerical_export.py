from __future__ import annotations

from copy import deepcopy
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.gds.build import component_geometry_arrays, resolve_and_build
from max_layout.gds.couplers import resolve_grating_fill_factors
from max_layout import lumerical
from max_layout.lumerical import (
    apply_lumerical_sweep_values,
    expand_lumerical_sweep_points,
    generate_lumerical_notebook,
    generate_lumerical_sweep_notebook,
    normalize_lumerical_sweep_spec,
    seed_simulation_ports,
    sweepable_component_parameters,
)
from max_layout.ui.lumerical_dialog import (
    CrossSectionDomainPreview,
    _anchored_stack_ranges,
    _conformal_fill_start,
    _sidewall_face_points,
)
from max_layout.utils import parse_sequence


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


def cell_source_containing(notebook: dict, needle: str) -> str:
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if needle in source:
            return source
    raise AssertionError(f"No notebook cell contains {needle!r}")


def assignment_value(notebook: dict, name: str):
    import ast

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"No notebook assignment named {name!r}")


def decoded_sweep_cases(notebook: dict) -> list[dict]:
    encoded = assignment_value(notebook, "_SWEEP_CASES_B64")
    return json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))


def soi_sweep_cases() -> tuple[list[dict], dict]:
    """Build a small, realistic GC-SOI Cartesian sweep export fixture."""
    from max_layout.ui.window import NativeLayoutWindow

    class Factory:
        make_component = NativeLayoutWindow.make_component
        automatic_simulation_companions = (
            NativeLayoutWindow.automatic_simulation_companions
        )
        synchronize_automatic_simulation_companions = (
            NativeLayoutWindow.synchronize_automatic_simulation_companions
        )

        def __init__(self) -> None:
            self.components = []
            self.next_uid = 1

    factory = Factory()
    grating = factory.make_component("GC-SOI", 0.0, 0.0)
    factory.components.append(grating)
    factory.components.extend(factory.automatic_simulation_companions(grating))
    spec = normalize_lumerical_sweep_spec(
        grating,
        [
            {"parameter": "pitch", "values": [0.65, 0.75]},
            {"parameter": "duty_cycle", "values": [0.46, 0.56]},
        ],
    )
    cases = []
    for values in expand_lumerical_sweep_points(spec):
        variant_components = deepcopy(factory.components)
        variant_grating = next(
            item for item in variant_components if item["uid"] == grating["uid"]
        )
        variant_grating["params"].update(values)
        variant_factory = Factory()
        variant_factory.components = variant_components
        variant_factory.next_uid = 1 + max(item["uid"] for item in variant_components)
        variant_factory.synchronize_automatic_simulation_companions(variant_grating)
        cases.append({"values": values, "components": variant_factory.components})
    return cases, spec


def mmi_sweep_cases() -> tuple[list[dict], dict]:
    """Build a two-point 1x2 MMI sweep with its automatic simulation setup."""
    from max_layout.ui.window import NativeLayoutWindow

    class Factory:
        make_component = NativeLayoutWindow.make_component
        automatic_simulation_companions = (
            NativeLayoutWindow.automatic_simulation_companions
        )
        synchronize_automatic_simulation_companions = (
            NativeLayoutWindow.synchronize_automatic_simulation_companions
        )

        def __init__(self) -> None:
            self.components = []
            self.next_uid = 1

    factory = Factory()
    mmi = factory.make_component("1x2 MMI", 0.0, 0.0)
    mmi["params"]["add_grating_couplers"] = False
    factory.components.append(mmi)
    factory.components.extend(factory.automatic_simulation_companions(mmi))
    spec = normalize_lumerical_sweep_spec(
        mmi,
        [{"parameter": "mmi_length", "values": [29.0, 31.0]}],
    )
    cases = []
    for values in expand_lumerical_sweep_points(spec):
        variant_components = deepcopy(factory.components)
        variant_mmi = next(
            item for item in variant_components if item["uid"] == mmi["uid"]
        )
        apply_lumerical_sweep_values(variant_mmi, values)
        variant_factory = Factory()
        variant_factory.components = variant_components
        variant_factory.next_uid = 1 + max(
            item["uid"] for item in variant_components
        )
        variant_factory.synchronize_automatic_simulation_companions(variant_mmi)
        cases.append({"values": values, "components": variant_factory.components})
    return cases, spec


class LumericalExportTests(unittest.TestCase):
    def test_every_lumerical_result_path_writes_and_fetches_summary_txt(self) -> None:
        straight = component("Straight")
        notebook, _ = generate_lumerical_notebook(
            [straight],
            {
                "included_layers": [(1, 0)],
                "material_stack": lumerical.default_stack("TFLN on SiO2"),
                "run_after_build": True,
                "project_file": "straight_summary.fsp",
            },
        )
        save_source = cell_source_containing(notebook, "REMOTE_RESULTS_SAVER")
        fetch_source = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER")
        self.assertIn('REMOTE_TEXT_SUMMARY = os.path.join(REMOTE_WORK, "summary.txt")', save_source)
        self.assertIn("Exact source parameters (JSON)", save_source)
        single_headings = (
            "PROJECT",
            "PARAMETERS",
            "MATERIAL STACK AND MESH",
            "SIMULATION SETTINGS",
            "SOURCES / PORTS / MONITORS",
            "RESULTS SUMMARY",
            "WARNINGS / NOTES",
        )
        for heading in single_headings:
            self.assertIn('"%s"' % heading, save_source)
        self.assertEqual(
            list(map(save_source.index, ('"%s"' % heading for heading in single_headings))),
            sorted(map(save_source.index, ('"%s"' % heading for heading in single_headings))),
        )
        self.assertIn('("pitch", "Pitch", "um")', save_source)
        self.assertIn('("fiber_offset", "Fiber offset", "um")', save_source)
        self.assertIn('("angle_theta", "Fiber angle theta", "deg")', save_source)
        self.assertIn('("fiber_power_monitor_below_source_um", "Horizontal fiber-input monitor distance below source", "um")', save_source)
        self.assertIn('("mmi_length", "MMI length", "um")', save_source)
        self.assertIn('("taper_power", "MMI taper profile exponent", "")', save_source)
        self.assertIn('("input_reference_before_taper_um", "Input power-reference distance before taper", "um")', save_source)
        self.assertIn("slab_extent", save_source)
        self.assertIn("geometry/PML overlap", save_source)
        self.assertIn("Generic numeric providers saved", save_source)
        self.assertIn('REMOTE_WORK + "/summary.txt"', fetch_source)
        self.assertIn("SOURCE_COMPONENTS_JSON = ' + repr(SOURCE_COMPONENTS_JSON)", build_source)

        sweep_cases, sweep_spec = soi_sweep_cases()
        sweep_configuration = {
            "included_layers": [(1, 0), (2, 0)],
            "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            "wavelength_start_um": 1.50,
            "wavelength_stop_um": 1.60,
            "run_after_build": True,
        }
        sequential, _ = generate_lumerical_sweep_notebook(
            sweep_cases, sweep_configuration, sweep_spec
        )
        sequential_source = "\n".join(
            "".join(cell.get("source", [])) for cell in sequential["cells"]
        )
        self.assertIn('SWEEP_TEXT_SUMMARY = os.path.join(results_root, "summary.txt")', sequential_source)
        self.assertIn("Peak-best case:", sequential_source)
        self.assertIn("Peak-best exact source parameters (JSON):", sequential_source)
        self.assertIn("Target-best case at", sequential_source)
        self.assertIn("Target-best major parameters:", sequential_source)
        self.assertIn("Target-best exact source parameters (JSON):", sequential_source)
        self.assertIn("SWEEP DEFINITION", sequential_source)
        self.assertIn("FSP PROVENANCE", sequential_source)
        self.assertIn("per_case_fsp_saved=false", sequential_source)
        self.assertIn("_local_text_summary = PIRIS_RESULTS_DIR / \"summary.txt\"", sequential_source)
        self.assertIn("SWEEP_NOMINAL_PARAMETERS = ' + repr(SWEEP_NOMINAL_PARAMETERS)", sequential_source)

        multigpu, _ = lumerical.generate_lumerical_multigpu_sweep_notebook(
            sweep_cases,
            {**sweep_configuration, "lumerical_multigpu": {"node_count": 2}},
            sweep_spec,
        )
        multigpu_source = "\n".join(
            "".join(cell.get("source", [])) for cell in multigpu["cells"]
        )
        self.assertIn("SWEEP_TEXT_SUMMARY", multigpu_source)
        self.assertIn("'MATERIAL_STACK': MATERIAL_STACK", multigpu_source)
        self.assertIn("'SWEEP_NOMINAL_PARAMETERS': SWEEP_NOMINAL_PARAMETERS", multigpu_source)
        self.assertIn("lumerical_sweep_multithread_results.json", multigpu_source)
        self.assertIn("_local_text_summary = PIRIS_RESULTS_DIR / \"summary.txt\"", multigpu_source)

    def test_sweep_summary_txt_records_complete_winning_source_parameters(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cases = [
                {
                    "values": {"pitch": 0.77},
                    "source_parameters": {
                        "pitch": 0.77, "fill_factor": 0.43, "N": 31,
                        "alpha_t": 22.0, "taper_L": 23.0,
                        "L_extra": 10.0, "wg_width": 1.2, "wg_length": 5.0,
                        "fiber_offset": 4.8, "angle_theta": 7.0,
                        "fiber_power_monitor_below_source_um": 0.1,
                        "waveguide_effective_index": 2.0,
                        "waveguide_neff_tolerance": 0.3,
                        "waveguide_mode_search_count": 20,
                        "tolerance": 0.0005,
                    },
                    "display_label": "P=0.77",
                    "result_stem": "CE-P=0.77",
                    "mmi_analysis": None,
                },
                {
                    "values": {"pitch": 0.81},
                    "source_parameters": {
                        "pitch": 0.81, "fill_factor": 0.43, "N": 31,
                        "alpha_t": 22.0, "taper_L": 23.0,
                        "L_extra": 10.0, "wg_width": 1.2, "wg_length": 5.0,
                        "fiber_offset": 4.8, "angle_theta": 7.0,
                        "fiber_power_monitor_below_source_um": 0.1,
                        "waveguide_effective_index": 2.0,
                        "waveguide_neff_tolerance": 0.3,
                        "waveguide_mode_search_count": 20,
                        "tolerance": 0.0005,
                    },
                    "display_label": "P=0.81",
                    "result_stem": "CE-P=0.81",
                    "mmi_analysis": None,
                },
            ]
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_SHARED_CHECKPOINT_DIR": str(Path(temporary_directory) / "checkpoints"),
                "SWEEP_HASH": "summary-contract",
                "SWEEP_CASES": cases,
                "SWEEP_SPEC": {
                    "component_uid": 9,
                    "component_kind": "Grating coupler",
                    "axes": [{
                        "parameter": "pitch", "short_name": "P",
                        "values": [0.77, 0.81],
                    }],
                },
                "SWEEP_NOMINAL_PARAMETERS": deepcopy(cases[0]["source_parameters"]),
                "SOURCE_COMPONENTS_JSON": [{
                    "uid": 9, "kind": "Grating coupler",
                    "params": deepcopy(cases[0]["source_parameters"]),
                }],
                "SETTINGS": {
                    "wavelength_start_um": 1.25, "wavelength_stop_um": 1.35,
                    "frequency_points": 3, "resource_mode": "GPU",
                },
                "MATERIAL_STACK": lumerical.default_stack("TFLN on SiO2"),
                "BOUNDING_BOX_UM": [-2.0, -5.0, 20.0, 5.0],
                "PORTS": [],
                "FIBER_GEOMETRIES": [],
                "MONITORS": [],
                "GRATING_ANALYSIS": {"component_uid": 9},
                "MMI_ANALYSIS": None,
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            wavelength_m = np.asarray([1.25e-6, 1.30e-6, 1.35e-6])
            def grating_details(response):
                response = np.asarray(response, dtype=float)
                return {
                    "fiber_input_power": np.ones(response.size),
                    "waveguide_mode_power": response,
                    "waveguide_total_power": response + 0.05,
                    "waveguide_total_transmission": response + 0.05,
                }

            first_response = np.asarray([0.20, 0.60, 0.25])
            second_response = np.asarray([0.70, 0.55, 0.40])
            namespace["_save_sweep_case"](
                0, "coupling_efficiency", wavelength_m,
                first_response, grating_details(first_response),
            )
            namespace["_save_sweep_case"](
                1, "coupling_efficiency", wavelength_m,
                second_response, grating_details(second_response),
            )
            namespace["_finalize_sweep_results"]([])
            summary = (Path(temporary_directory) / "summary.txt").read_text(
                encoding="utf-8"
            )

        self.assertIn("completed=2 | failed=0", summary)
        headings = (
            "PROJECT",
            "PARAMETERS",
            "SWEEP DEFINITION",
            "MATERIAL STACK AND MESH",
            "SIMULATION SETTINGS",
            "SOURCES / PORTS / MONITORS",
            "RESULTS SUMMARY",
            "WARNINGS / NOTES",
            "FSP PROVENANCE",
        )
        positions = [summary.index("\n" + heading + "\n") for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Peak-best case: index=2", summary)
        self.assertIn('swept parameters={"pitch":0.81}', summary)
        self.assertIn("Target-best case at 1300 nm: index=1", summary)
        self.assertIn('swept parameters={"pitch":0.77} | value=0.6 (60%)', summary)
        self.assertIn("- Pitch: 0.81 um", summary)
        self.assertIn("- Fill factor: 0.43", summary)
        self.assertIn("- Number of grating periods: 31", summary)
        self.assertIn("- Aperture angle: 22 deg", summary)
        self.assertIn("- Taper length: 23 um", summary)
        self.assertIn("- Thick-end extension: 10 um", summary)
        self.assertIn("- Waveguide width: 1.2 um", summary)
        self.assertIn("- Waveguide length: 5 um", summary)
        self.assertIn("- Fiber offset: 4.8 um", summary)
        self.assertIn("- Fiber angle theta: 7 deg", summary)
        self.assertIn(
            "- Horizontal fiber-input monitor distance below source: 0.1 um",
            summary,
        )
        self.assertNotIn("Waveguide target effective index", summary)
        self.assertNotIn('"waveguide_effective_index"', summary)
        self.assertIn("- Waveguide eigensolver modes searched: 20", summary)
        self.assertIn("- Geometry build tolerance: 0.0005 um", summary)
        self.assertIn("Target-best major parameters:\n- Pitch: 0.77 um", summary)
        self.assertIn("Target-best exact source parameters (JSON):", summary)
        self.assertIn('"fill_factor":0.43', summary)
        self.assertIn('"fiber_offset":4.8', summary)
        self.assertIn('"angle_theta":7.0', summary)
        self.assertIn("peak=0.7 (70%) at 1250 nm", summary)
        self.assertIn("value at target 1300 nm=0.55 (55%)", summary)
        self.assertIn("per_case_fsp_saved=false", summary)

    def test_3d_preview_sidewall_faces_match_middle_reference(self) -> None:
        square = np.asarray([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])

        top_90, bottom_90 = _sidewall_face_points(square, 1.0, 90.0)
        np.testing.assert_allclose(top_90, square)
        np.testing.assert_allclose(bottom_90, square)

        top_45, bottom_45 = _sidewall_face_points(square, 1.0, 45.0)
        np.testing.assert_allclose(top_45.min(axis=0), [-1.5, -1.5])
        np.testing.assert_allclose(top_45.max(axis=0), [1.5, 1.5])
        np.testing.assert_allclose(bottom_45.min(axis=0), [-2.5, -2.5])
        np.testing.assert_allclose(bottom_45.max(axis=0), [2.5, 2.5])

        top_135, bottom_135 = _sidewall_face_points(square, 1.0, 135.0)
        np.testing.assert_allclose(top_135.min(axis=0), [-2.5, -2.5])
        np.testing.assert_allclose(bottom_135.min(axis=0), [-1.5, -1.5])

        reverse_top, reverse_bottom = _sidewall_face_points(square[::-1], 1.0, 45.0)
        np.testing.assert_allclose(reverse_top.min(axis=0), [-1.5, -1.5])
        np.testing.assert_allclose(reverse_bottom.max(axis=0), [2.5, 2.5])

    def test_tfln_automatic_ports_are_twice_width_with_three_um_minimum(self) -> None:
        from max_layout.ui.window import NativeLayoutWindow, automatic_waveguide_port_span_um

        class Factory:
            make_component = NativeLayoutWindow.make_component
            automatic_simulation_companions = NativeLayoutWindow.automatic_simulation_companions

            def __init__(self) -> None:
                self.components = []
                self.next_uid = 1

        self.assertEqual(DEFAULT_COMPONENT_VALUES["Straight"]["width"], 1.2)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Grating coupler"]["wg_width"], 1.2)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["FDTD port"]["span_um"], 3.0)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Grating coupler"]["waveguide_monitor_span_um"], 3.0)

        factory = Factory()
        straight = factory.make_component("Straight", 0.0, 0.0)
        factory.components.append(straight)
        straight_ports = factory.automatic_simulation_companions(straight)
        self.assertEqual([port["params"]["span_um"] for port in straight_ports], [3.0, 3.0])

        taper = factory.make_component("Taper", 0.0, 0.0)
        self.assertEqual(automatic_waveguide_port_span_um(taper, "left"), 3.0)
        self.assertEqual(automatic_waveguide_port_span_um(taper, "right"), 5.0)

        grating = factory.make_component("Grating coupler", 0.0, 0.0)
        generic_companions = factory.automatic_simulation_companions(grating)
        generic_power = next(
            item for item in generic_companions
            if item.get("grating_monitor_role") == "waveguide_total_power"
        )
        generic_receiver = next(
            item for item in generic_companions
            if item["kind"] == "FDTD port"
            and item.get("simulation_parent_port") == "waveguide_point"
        )
        self.assertEqual(generic_power["params"]["y span"], 3.0)
        self.assertEqual(generic_receiver["params"]["span_um"], 3.0)

        soi = factory.make_component("GC-SOI", 0.0, 0.0)
        soi_companions = factory.automatic_simulation_companions(soi)
        soi_power = next(
            item for item in soi_companions
            if item.get("grating_monitor_role") == "waveguide_total_power"
        )
        soi_receiver = next(
            item for item in soi_companions
            if item["kind"] == "FDTD port"
            and item.get("simulation_parent_port") == "waveguide_point"
        )
        self.assertEqual(soi_power["params"]["y span"], 2.5)
        self.assertEqual(soi_receiver["params"]["span_um"], 2.5)

    def test_apodized_grating_rejects_scalar_fill_sweep_and_preserves_pitch_sweep(self) -> None:
        original = component("GC-SOI")
        original["params"]["fill_factors"] = "linspace(0.30, 0.50)"
        temporary = deepcopy(original)
        self.assertNotIn(
            "duty_cycle",
            {
                item["parameter"]
                for item in sweepable_component_parameters(temporary)
            },
        )
        with self.assertRaisesRegex(ValueError, "apodization array"):
            apply_lumerical_sweep_values(temporary, {"duty_cycle": 0.46})
        apply_lumerical_sweep_values(temporary, {"pitch": 0.713})
        self.assertEqual(temporary["params"]["pitch"], 0.713)
        self.assertEqual(
            temporary["params"]["fill_factors"], "linspace(0.30, 0.50)"
        )
        self.assertEqual(original["params"]["fill_factors"], "linspace(0.30, 0.50)")

    def test_grating_geometry_requires_explicit_project_pitch_fill_and_count(self) -> None:
        generic = component("Grating coupler")
        for missing in ("pitch", "fill_factor", "N"):
            broken = deepcopy(generic)
            broken["params"].pop(missing)
            with self.assertRaises((KeyError, ValueError), msg=missing):
                component_geometry_arrays(broken)

        soi = component("GC-SOI")
        for missing in ("pitch", "duty_cycle"):
            broken = deepcopy(soi)
            broken["params"].pop(missing)
            with self.assertRaisesRegex(ValueError, missing):
                component_geometry_arrays(broken)

    def test_nondefault_grating_json_is_embedded_without_geometry_defaults(self) -> None:
        generic = component("Grating coupler")
        generic["params"].update(
            {"pitch": 0.9137, "fill_factor": 0.317, "N": 4}
        )
        generic_notebook, _ = generate_lumerical_notebook(
            [generic], {"included_layers": [[1, 0], [2, 0]]}
        )
        generic_source = assignment_value(
            generic_notebook, "SOURCE_COMPONENTS_JSON"
        )[0]["params"]
        self.assertEqual(generic_source["pitch"], 0.9137)
        self.assertEqual(generic_source["fill_factor"], 0.317)
        self.assertEqual(generic_source["N"], 4)

        soi = component("GC-SOI")
        soi["params"].update(
            {"pitch": 0.7319, "duty_cycle": 0.437, "target_length": 2.5}
        )
        soi_notebook, _ = generate_lumerical_notebook(
            [soi], {"included_layers": [[1, 0], [2, 0]]}
        )
        soi_source = assignment_value(
            soi_notebook, "SOURCE_COMPONENTS_JSON"
        )[0]["params"]
        self.assertEqual(soi_source["pitch"], 0.7319)
        self.assertEqual(soi_source["duty_cycle"], 0.437)
        self.assertEqual(soi_source["target_length"], 2.5)

    def test_sweep_inspection_model_uses_untouched_nominal_json(self) -> None:
        nominal = component("Grating coupler")
        nominal["params"].update(
            {"pitch": 0.9137, "fill_factor": 0.317, "N": 4}
        )
        spec = normalize_lumerical_sweep_spec(
            nominal, [{"parameter": "pitch", "values": [0.78, 0.82]}]
        )
        cases = []
        for values in expand_lumerical_sweep_points(spec):
            variant = deepcopy(nominal)
            apply_lumerical_sweep_values(variant, values)
            cases.append({"values": values, "components": [variant]})
        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("TFLN on SiO2"),
            "run_after_build": True,
            "resource_mode": "GPU",
        }
        single, _ = generate_lumerical_notebook([nominal], configuration)
        sequential, _ = generate_lumerical_sweep_notebook(
            cases,
            configuration,
            spec,
            nominal_components=[deepcopy(nominal)],
        )
        multigpu, _ = lumerical.generate_lumerical_multigpu_sweep_notebook(
            cases,
            configuration,
            spec,
            nominal_components=[deepcopy(nominal)],
        )

        for notebook in (sequential, multigpu):
            self.assertEqual(
                assignment_value(notebook, "SWEEP_NOMINAL_PARAMETERS")["pitch"],
                0.9137,
            )
            self.assertEqual(
                assignment_value(notebook, "SOURCE_COMPONENTS_JSON")[0]["params"]["fill_factor"],
                0.317,
            )
            self.assertEqual(
                assignment_value(notebook, "GEOMETRY"),
                assignment_value(single, "GEOMETRY"),
            )
            remote_cases = decoded_sweep_cases(notebook)
            self.assertEqual(
                [case["source_parameters"]["pitch"] for case in remote_cases],
                [0.78, 0.82],
            )
            self.assertEqual(
                [case["source_parameters"]["fill_factor"] for case in remote_cases],
                [0.317, 0.317],
            )

    def test_grating_and_mmi_neff_targets_are_runtime_derived(self) -> None:
        for kind in ("Grating coupler", "GC-SOI", "1x2 MMI"):
            self.assertNotIn("waveguide_effective_index", component(kind)["params"])

    def test_standard_tfln_grating_and_stack_defaults(self) -> None:
        params = component("Grating coupler")["params"]
        self.assertEqual(params["pitch"], 0.8)
        self.assertEqual(params["fill_factor"], 0.5)

        stack = lumerical.default_stack("TFLN on SiO2")
        cross_section = next(
            row for row in stack if row["name"] == "Exported TFLN cross-section"
        )
        self.assertEqual(cross_section["sidewall_angle_deg"], 79.0)

        from max_layout.ui.window import grating_lumerical_export_settings

        automatic = grating_lumerical_export_settings({})
        self.assertEqual(automatic["stack_preset"], "TFLN on SiO2")
        self.assertTrue(
            all(row["mesh_factor"] == 0.0 for row in automatic["material_stack"])
        )

        legacy = grating_lumerical_export_settings(
            {
                "stack_preset": "TFLN on SiO2",
                "material_stack": lumerical.default_stack("TFLN on SiO2"),
                "frequency_points": 17,
            }
        )
        self.assertTrue(
            all(row["mesh_factor"] == 0.0 for row in legacy["material_stack"])
        )
        self.assertEqual(legacy["frequency_points"], 17)

        customized_stack = lumerical.default_stack("TFLN on SiO2")
        customized_stack[2]["mesh_factor"] = 0.1
        preserved = grating_lumerical_export_settings(
            {
                "stack_preset": "TFLN on SiO2",
                "material_stack": customized_stack,
            }
        )
        self.assertEqual(preserved["material_stack"][2]["mesh_factor"], 0.1)

        dialog_source = Path("src/max_layout/ui/lumerical_dialog.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Use Lumerical automatic mesh", dialog_source)
        self.assertIn("def use_lumerical_automatic_mesh", dialog_source)
        self.assertIn("mesh_factor.setValue(0.0)", dialog_source)

    def test_mmi_default_stack_is_three_micron_bottom_oxide_without_silicon(self) -> None:
        stack = lumerical.default_stack("TFLN MMI (3 um SiO2)")
        self.assertEqual(stack[0]["name"], "Bottom SiO2")
        self.assertEqual(stack[0]["material"], "SiO2 (Glass) - Palik")
        self.assertEqual(stack[0]["thickness_um"], 3.0)
        self.assertFalse(
            any(row["material"] == "Si (Silicon) - Palik" for row in stack)
        )

        from max_layout.ui.window import mmi_lumerical_export_settings

        migrated = mmi_lumerical_export_settings(
            {
                "stack_preset": "TFLN on SiO2",
                "material_stack": lumerical.default_stack("TFLN on SiO2"),
                "frequency_points": 9,
            }
        )
        self.assertEqual(migrated["stack_preset"], "TFLN MMI (3 um SiO2)")
        self.assertEqual(migrated["material_stack"][0]["thickness_um"], 3.0)
        self.assertEqual(migrated["frequency_points"], 9)

        customized_stack = lumerical.default_stack("TFLN on SiO2")
        customized_stack[1]["thickness_um"] = 4.0
        preserved = mmi_lumerical_export_settings(
            {
                "stack_preset": "TFLN on SiO2",
                "material_stack": customized_stack,
            }
        )
        self.assertEqual(preserved["material_stack"][1]["thickness_um"], 4.0)

    def test_fdtd_domain_preview_has_stack_xz_and_true_gds_xy_views(self) -> None:
        source = Path("src/max_layout/ui/lumerical_dialog.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'self.xz_cross_section_preview = CrossSectionDomainPreview(self, "XZ")',
            source,
        )
        self.assertIn(
            'self.xy_top_view_preview = CrossSectionDomainPreview(self, "XY")',
            source,
        )
        self.assertNotIn('CrossSectionDomainPreview(self, "YZ")', source)
        self.assertIn("self.cross_section_previews = (", source)
        self.assertIn("for preview in self.cross_section_previews:", source)
        self.assertNotIn("self.preview_plane = QComboBox()", source)
        self.assertIn('for points, layer in state["polygons"]:', source)
        self.assertIn("painter.drawPolygon(polygon)", source)
        self.assertIn("XY · TOP / GDS view", source)
        self.assertIn("self.domain_center_spins", source)
        self.assertIn("self.domain_size_spins", source)
        self.assertIn("def _set_domain_center_or_size", source)
        self.assertIn(
            "share the same X-min/X-max values",
            source,
        )

        state = {
            "x_base": (-4.0, 8.0),
            "y_base": (-3.0, 5.0),
            "z_base": (-1.0, 2.0),
            "padding": {
                "x_min": 1.0,
                "x_max": 2.0,
                "y_min": 3.0,
                "y_max": 4.0,
                "z_min": 5.0,
                "z_max": 6.0,
            },
        }

        class Projection:
            plane = "XY"

        horizontal, vertical, _h_base, _v_base, xy_domain = (
            CrossSectionDomainPreview._projection_state(Projection(), state)
        )
        self.assertEqual((horizontal, vertical), ("x", "y"))
        self.assertEqual(xy_domain, (-5.0, 10.0, -6.0, 9.0))

        Projection.plane = "XZ"
        horizontal, vertical, _h_base, _v_base, xz_domain = (
            CrossSectionDomainPreview._projection_state(Projection(), state)
        )
        self.assertEqual((horizontal, vertical), ("x", "z"))
        self.assertEqual(xz_domain, (-5.0, 10.0, -6.0, 8.0))

    def test_official_soi_grating_component_and_stack_defaults(self) -> None:
        grating = component("GC-SOI")
        params = grating["params"]
        self.assertEqual(params["pitch"], 0.6713)
        self.assertEqual(params["duty_cycle"], 0.3992)
        self.assertEqual(params["target_length"], 25.0)
        self.assertEqual(params["h_total"], 0.22)
        self.assertEqual(params["etch_depth"], 0.10)
        self.assertEqual(params["angle_theta"], 10.0)
        self.assertNotIn("fiber_tilt_deg", params)
        self.assertEqual(params["fiber_offset"], 2.74533)
        self.assertEqual(params["tolerance"], 0.005)
        self.assertEqual(params["fdtd_port_offset_from_waveguide_end_um"], 2.0)
        self.assertEqual(params["waveguide_monitor_span_um"], 2.5)
        self.assertEqual(params["waveguide_total_power_before_mode_um"], 1.0)
        self.assertNotIn("waveguide_effective_index", params)
        self.assertEqual(params["waveguide_neff_tolerance"], 0.3)
        self.assertEqual(params["waveguide_mode_search_count"], 20)
        polygons, _ = component_geometry_arrays(grating)
        self.assertEqual(len(polygons), 47)
        self.assertLess(sum(len(points) for points, _layer, _datatype in polygons), 30_000)
        legacy_grating = deepcopy(grating)
        legacy_grating["params"]["tolerance"] = 0.0005
        legacy_polygons, _ = component_geometry_arrays(legacy_grating)
        self.assertLess(sum(len(points) for points, _layer, _datatype in legacy_polygons), 30_000)
        self.assertEqual({(layer, datatype) for _points, layer, datatype in polygons}, {(1, 0), (2, 0)})
        all_points = np.vstack([points for points, _layer, _datatype in polygons])
        self.assertAlmostEqual(float(all_points[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(all_points[:, 0].max()), 70.91271704, places=6)

        stack = lumerical.default_stack("SOI grating coupler (Ansys)")
        self.assertEqual([row["thickness_um"] for row in stack], [3.0, 1.0, 0.12, 0.10, 0.7, 0.7])
        self.assertEqual(stack[2]["gds_layers"], [1])
        self.assertEqual(stack[3]["gds_layers"], [2])
        self.assertTrue(stack[4]["conformal"])
        self.assertEqual(stack[5]["material"], "Air")
        self.assertTrue(all(row["mesh_factor"] == 0.0 for row in stack))
        self.assertEqual([row["mesh_order"] for row in stack], [2, 2, 2, 2, 3, 1])

    def test_conformal_cladding_fills_full_device_depth_in_solver_and_previews(self) -> None:
        soi_stack = lumerical.default_stack("SOI grating coupler (Ansys)")
        soi_ranges = _anchored_stack_ranges(soi_stack)
        soi_cladding_index = next(
            index for index, (row, _z0, _z1) in enumerate(soi_ranges)
            if bool(row.get("conformal", False))
        )
        _row, soi_cladding_z0, _z1 = soi_ranges[soi_cladding_index]
        # The 120 nm residual film and 100 nm upper mask are one physical
        # 220 nm silicon film, so oxide must fill down to its -60 nm base.
        self.assertAlmostEqual(
            _conformal_fill_start(soi_ranges, soi_cladding_index, soi_cladding_z0),
            -0.06,
        )

        tfln_stack = [
            {
                "name": "TFLN device",
                "material": "LiNbO3",
                "thickness_um": 0.4,
                "etch_depth_um": 0.2,
                "role": "geometry",
                "gds_layer": 1,
            },
            {
                "name": "SiO2 cladding",
                "material": "SiO2 (Glass) - Palik",
                "thickness_um": 1.0,
                "role": "background",
                "conformal": True,
            },
        ]
        tfln_ranges = _anchored_stack_ranges(tfln_stack)
        _row, tfln_cladding_z0, _z1 = tfln_ranges[1]
        self.assertAlmostEqual(
            _conformal_fill_start(tfln_ranges, 1, tfln_cladding_z0),
            0.0,
        )

        self.assertIn("def _conformal_fill_start", lumerical._BUILD_CELL)
        self.assertIn(
            "_conformal_fill_start(z_ranges, row_index - 1, z0)",
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            "Verified full-domain conformal cladding",
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            "covers every waveguide, grating tooth, flare, terminal arc, and extension",
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            "geometry_bounds_um[0] - pml_geometry_overlap_um",
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            'fdtd.set("x span", (material_x_max_um - material_x_min_um) * UM)',
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            "Keep the FDTD bounds unchanged and independently enlarge the material",
            lumerical._BUILD_CELL,
        )
        self.assertIn(
            "_conformal_fill_start(z_ranges, row_index, z0)",
            lumerical._GEOMETRY_PROJECTIONS_REMOTE,
        )

    def test_material_stack_covers_geometry_outside_clipped_solver_bounds(self) -> None:
        import ast

        helper_names = {
            "_layer_builder_geometry",
            "_conformal_fill_start",
            "_add_material_stack",
        }
        build_tree = ast.parse(lumerical._BUILD_CELL)
        helper_nodes = [
            node
            for node in build_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helper_nodes}, helper_names)

        geometry = [
            {
                "layer": layer,
                "datatype": 0,
                "vertices_um": [
                    [-2.0, -4.0],
                    [2.0, -4.0],
                    [2.0, 4.0],
                    [-2.0, 4.0],
                ],
            }
            for layer in (1, 2)
        ]
        namespace = {"np": np, "UM": 1e-6, "GEOMETRY": geometry}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=helper_nodes, type_ignores=[])
                ),
                "<material-stack-union-extent>",
                "exec",
            ),
            namespace,
        )

        class FakeFDTD:
            def __init__(self) -> None:
                self.layer_builder: dict[str, object] = {}
                self.rectangles: list[dict[str, object]] = []
                self._active = self.layer_builder

            def addlayerbuilder(self):
                self._active = self.layer_builder

            def addrect(self):
                self.rectangles.append({})
                self._active = self.rectangles[-1]

            def set(self, name, value):
                self._active[name] = value

            def addlayer(self, _name):
                return None

            def setlayer(self, _name, _property, _value):
                return None

        solver_bounds_um = [-3.0, 0.0, 3.0, 5.0]
        original_solver_bounds_um = list(solver_bounds_um)
        overlap_um = 1.0
        fake_fdtd = FakeFDTD()
        namespace["_add_material_stack"](
            fake_fdtd,
            _anchored_stack_ranges(
                lumerical.default_stack("SOI grating coupler (Ansys)")
            ),
            solver_bounds_um,
            -4.0,
            2.0,
            overlap_um,
        )

        # The solver clips the negative-Y half of the symmetric geometry, but
        # material objects cover the union of solver and geometry bounds plus
        # the requested overlap: X [-4, 4] um and Y [-5, 6] um.
        expected_center_m = np.asarray([0.0, 0.5]) * 1e-6
        expected_span_m = np.asarray([8.0, 11.0]) * 1e-6
        material_objects = [fake_fdtd.layer_builder, *fake_fdtd.rectangles]
        self.assertGreater(len(fake_fdtd.rectangles), 0)
        self.assertTrue(
            any("SiO2 TOX" in str(item.get("name", "")) for item in fake_fdtd.rectangles)
        )
        for material_object in material_objects:
            self.assertAlmostEqual(material_object["x"], expected_center_m[0], places=12)
            self.assertAlmostEqual(material_object["y"], expected_center_m[1], places=12)
            self.assertAlmostEqual(
                material_object["x span"], expected_span_m[0], places=12
            )
            self.assertAlmostEqual(
                material_object["y span"], expected_span_m[1], places=12
            )
        self.assertEqual(solver_bounds_um, original_solver_bounds_um)

    def test_automatic_soi_grating_uses_one_source_and_waveguide_monitors(self) -> None:
        from max_layout.ui.window import NativeLayoutWindow

        class Factory:
            make_component = NativeLayoutWindow.make_component
            automatic_simulation_companions = NativeLayoutWindow.automatic_simulation_companions

            def __init__(self) -> None:
                self.components = []
                self.next_uid = 1

        factory = Factory()
        grating = factory.make_component("GC-SOI", 0.0, 0.0)
        factory.components.append(grating)
        companions = factory.automatic_simulation_companions(grating)
        orders = {
            item.get("simulation_parent_port", "fiber_source"): int(item["params"]["order"])
            for item in companions
            if item["kind"] in {"FDTD port", "Fiber-axis FDTD port"}
        }
        self.assertEqual(orders["fiber_source"], 1)
        self.assertEqual(orders["waveguide_point"], 2)
        self.assertNotIn("fiber_input_power", orders)
        input_power_monitor = next(
            item for item in companions
            if item["kind"] == "Power monitor"
            and item.get("simulation_parent_port") == "fiber_input_power"
        )
        self.assertEqual(input_power_monitor["params"]["plane normal"], "Z")
        self.assertTrue(input_power_monitor["params"]["align to fiber axis"])
        self.assertEqual(
            input_power_monitor["params"]["fiber plane role"],
            "input power measurement",
        )
        self.assertEqual(input_power_monitor["params"]["expected propagation sign"], -1.0)
        self.assertNotIn("mode number", input_power_monitor["params"])
        self.assertNotIn("polarization", input_power_monitor["params"])
        waveguide_power = next(
            item for item in companions
            if item["kind"] == "Power monitor"
            and item.get("grating_monitor_role") == "waveguide_total_power"
        )
        waveguide_receiver = next(
            item for item in companions
            if item["kind"] == "FDTD port"
            and item.get("simulation_parent_port") == "waveguide_point"
        )
        self.assertEqual(waveguide_power["params"]["y span"], 2.5)
        self.assertEqual(waveguide_receiver["params"]["span_um"], 2.5)
        self.assertEqual(waveguide_receiver["params"]["mode"], "fundamental TE mode")
        self.assertEqual(waveguide_receiver["params"]["target neff"], 0.0)
        self.assertEqual(
            waveguide_receiver["params"]["target neff strategy"],
            "automatic material-index midpoint",
        )
        self.assertEqual(waveguide_receiver["params"]["mode search count"], 20)
        self.assertEqual(waveguide_power["x"] - waveguide_receiver["x"], 1.0)
        fiber_source_component = next(
            item for item in companions
            if item["kind"] == "Fiber-axis FDTD port"
            and item.get("simulation_parent_port") != "fiber_input_power"
        )
        self.assertEqual(fiber_source_component["params"]["mode"], "user select")
        self.assertEqual(fiber_source_component["params"]["mode number"], 0)
        self.assertEqual(fiber_source_component["params"]["polarization"], "local TE")
        self.assertEqual(
            fiber_source_component["params"]["candidate mode numbers"], [1, 2, 3]
        )
        notebook, _warnings = generate_lumerical_notebook(
            [grating, *companions],
            {
                "included_layers": [[1, 0], [2, 0]],
                "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            },
        )
        fibers = assignment_value(notebook, "FIBER_GEOMETRIES")
        ports = assignment_value(notebook, "PORTS")
        monitors = assignment_value(notebook, "MONITORS")
        fiber = fibers[0]
        source = next(port for port in ports if port["name"].endswith("fiber_axis"))
        receiver = next(port for port in ports if port["name"].endswith("waveguide_point"))
        measurement = next(
            monitor for monitor in monitors
            if monitor["name"].endswith("fiber_input_power")
        )
        self.assertEqual(len(ports), 2)
        self.assertEqual(source["name"], "uid_1_fiber_axis")
        self.assertEqual(receiver["name"], "uid_1_waveguide_point")
        self.assertEqual(receiver["mode"], "fundamental TE mode")
        self.assertEqual(measurement["fiber plane role"], "input power measurement")
        self.assertEqual(measurement["monitor_kind"], "Power monitor")
        self.assertEqual(measurement["plane normal"], "Z")
        self.assertEqual(measurement["expected propagation sign"], -1.0)
        self.assertFalse(any(port["name"].endswith("fiber_input_power") for port in ports))
        self.assertTrue(any(monitor["name"].endswith("waveguide_total_power") for monitor in monitors))
        self.assertFalse(any(monitor.get("monitor_kind") == "Mode expansion monitor" for monitor in monitors))
        self.assertEqual(fiber["z reference"], "center of SiO2 cladding")
        self.assertAlmostEqual(
            source["center"][0] - fiber["center"][0],
            0.65 * np.sin(np.deg2rad(10.0)),
        )
        self.assertAlmostEqual(
            measurement["center"][0] - fiber["center"][0],
            (0.65 * np.cos(np.deg2rad(10.0)) - 0.1) * np.tan(np.deg2rad(10.0)),
        )

    def test_fiber_local_te_selects_near_degenerate_partner_after_rotation(self) -> None:
        import ast

        helper_names = {
            "_mode_profile_vector",
            "_fiber_local_te_score",
            "_fiber_gaussian_circular_scores",
            "_fiber_candidate_neff",
            "_select_fiber_local_te_mode",
        }
        build_tree = ast.parse(lumerical._BUILD_CELL)
        helper_nodes = [
            node
            for node in build_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helper_nodes}, helper_names)
        namespace = {"np": np}
        helper_module = ast.Module(body=helper_nodes, type_ignores=[])
        exec(
            compile(helper_module, "<fiber-mode-selection>", "exec"),
            namespace,
        )
        select_fiber_mode = namespace["_select_fiber_local_te_mode"]

        high_neff_mode = np.zeros((3, 4, 3), dtype=complex)
        ey_mode = np.zeros((3, 4, 3), dtype=complex)
        ex_mode = np.zeros((3, 4, 3), dtype=complex)
        high_neff_mode[..., 0] = 1.0
        ey_mode[..., 1] = 1.0
        ex_mode[..., 0] = 1.0

        class FakeFdtd:
            def __init__(self) -> None:
                self.selected = []
                self.mode_updates = []

            def select(self, path):
                self.selected.append(path)

            def updateportmodes(self, modes):
                self.mode_updates.append(np.asarray(modes, dtype=int).tolist())

            def getresult(self, _path, result_name):
                if result_name == "mode profiles":
                    return {"E1": high_neff_mode, "E2": ey_mode, "E3": ex_mode}
                if result_name == "neff":
                    return {
                        "lambda": np.linspace(1.50e-6, 1.60e-6, 4),
                        "n": np.asarray([1, 2, 3]),
                        "neff": np.asarray([
                            [1.72, 1.4440, 1.4445],
                            [1.71, 1.4439, 1.4444],
                            [1.70, 1.4438, 1.4443],
                            [1.69, 1.4437, 1.4442],
                        ]),
                    }
                raise AssertionError(result_name)

        base_port = {
            "name": "fiber_source",
            "candidate mode numbers": [1, 2, 3],
            "minimum local TE fraction": 0.8,
            "mode degeneracy tolerance": 0.01,
        }

        unrotated_fdtd = FakeFdtd()
        unrotated = select_fiber_mode(
            unrotated_fdtd,
            "FDTD::ports::fiber_source",
            {**base_port, "angle phi": 0.0},
        )
        self.assertEqual(unrotated["mode number"], 2)
        self.assertEqual(unrotated["degenerate mode pair"], [2, 3])
        self.assertEqual(unrotated["selected mode order"], [2, 3, 1])
        self.assertEqual(unrotated["target polarization xy"], [0.0, 1.0])
        self.assertAlmostEqual(unrotated["local TE scores"]["1"], 0.0)
        self.assertAlmostEqual(unrotated["local TE scores"]["2"], 1.0)
        self.assertLess(unrotated["neff degeneracy delta"], 0.01)
        self.assertEqual(unrotated_fdtd.mode_updates, [[1, 2, 3]])

        rotated_fdtd = FakeFdtd()
        rotated = select_fiber_mode(
            rotated_fdtd,
            "FDTD::ports::fiber_source",
            {**base_port, "angle phi": 90.0},
        )
        self.assertEqual(rotated["mode number"], 3)
        self.assertEqual(rotated["degenerate mode pair"], [2, 3])
        self.assertEqual(rotated["selected mode order"], [3, 2, 1])
        self.assertAlmostEqual(rotated["target polarization xy"][0], -1.0)
        self.assertAlmostEqual(rotated["target polarization xy"][1], 0.0, places=12)
        self.assertAlmostEqual(rotated["local TE scores"]["3"], 1.0)
        self.assertAlmostEqual(rotated["local TE scores"]["2"], 0.0, places=12)
        self.assertLess(rotated["neff degeneracy delta"], 0.01)
        self.assertEqual(rotated_fdtd.mode_updates, [[1, 2, 3]])

    def test_fiber_port_export_defaults_request_first_three_local_te_candidates(self) -> None:
        defaults = DEFAULT_COMPONENT_VALUES["Fiber-axis FDTD port"]
        self.assertEqual(defaults["mode number"], 0)
        self.assertEqual(defaults["polarization"], "local TE")
        self.assertEqual(defaults["candidate mode numbers"], [1, 2, 3])
        self.assertEqual(defaults["mode degeneracy tolerance"], 0.01)
        self.assertEqual(defaults["minimum local TE fraction"], 0.8)

    def test_fiber_input_plane_is_nonmodal_power_monitor(self) -> None:
        from max_layout.ui.window import NativeLayoutWindow

        class Factory:
            make_component = NativeLayoutWindow.make_component
            automatic_simulation_companions = (
                NativeLayoutWindow.automatic_simulation_companions
            )

            def __init__(self) -> None:
                self.components = []
                self.next_uid = 1

        factory = Factory()
        grating = factory.make_component("Grating coupler", 0.0, 0.0)
        factory.components.append(grating)
        companions = factory.automatic_simulation_companions(grating)
        fiber_ports = [
            item for item in companions if item["kind"] == "Fiber-axis FDTD port"
        ]
        input_monitors = [
            item for item in companions
            if item["kind"] == "Power monitor"
            and item.get("simulation_parent_port") == "fiber_input_power"
        ]
        self.assertEqual(len(fiber_ports), 1)
        self.assertEqual(len(input_monitors), 1)
        self.assertEqual(fiber_ports[0]["params"]["fiber plane role"], "source")
        self.assertEqual(
            input_monitors[0]["params"]["fiber plane role"],
            "input power measurement",
        )
        self.assertEqual(input_monitors[0]["params"]["plane normal"], "Z")
        self.assertEqual(input_monitors[0]["params"]["expected propagation sign"], -1.0)
        self.assertNotIn("mode number", input_monitors[0]["params"])
        self.assertNotIn("candidate mode numbers", input_monitors[0]["params"])
        self.assertIn('fdtd.set("output power", True)', lumerical._BUILD_CELL)
        self.assertNotIn(
            'fdtd.set("use source limits", True)\n            fdtd.set("output power", True)',
            lumerical._BUILD_CELL,
        )

    def test_sweep_fiber_fallback_reselects_pair_and_reaches_multigpu_worker(self) -> None:
        import ast

        helper_names = {
            "_sweep_mode_profile_vector",
            "_sweep_candidate_neff",
            "_sweep_gaussian_circular_scores",
            "_sweep_reselect_fiber_local_te",
        }
        sweep_tree = ast.parse(lumerical._SWEEP_RUNTIME_REMOTE)
        helper_nodes = [
            node
            for node in sweep_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helper_nodes}, helper_names)

        ex_mode = np.zeros((3, 4, 3), dtype=complex)
        ey_mode = np.zeros((3, 4, 3), dtype=complex)
        ex_mode[..., 0] = 1.0
        ey_mode[..., 1] = 1.0

        class FakeFdtd:
            def __init__(self) -> None:
                self.mode_updates = []

            def select(self, _path):
                return None

            def updateportmodes(self, modes):
                self.mode_updates.append(np.asarray(modes, dtype=int).tolist())

            def getresult(self, _path, result_name):
                if result_name == "mode profiles":
                    return {"E1": ex_mode, "E2": ey_mode}
                if result_name == "neff":
                    return {
                        "neff1": np.asarray([1.447000]),
                        "neff2": np.asarray([1.447004]),
                    }
                raise AssertionError(result_name)

        def sweep_selection(phi_deg):
            fake_fdtd = FakeFdtd()
            namespace = {"np": np, "fdtd": fake_fdtd}
            helper_module = ast.Module(body=helper_nodes, type_ignores=[])
            exec(
                compile(helper_module, "<sweep-fiber-mode-selection>", "exec"),
                namespace,
            )
            previous = {
                "mode number": 2,
                "selected mode order": [2, 1],
                "candidate mode numbers": [1, 2],
            }
            selected = namespace["_sweep_reselect_fiber_local_te"](
                "FDTD::ports::fiber_source",
                {
                    "name": "fiber_source",
                    "angle phi": phi_deg,
                    "candidate mode numbers": [1, 2],
                    "minimum local TE fraction": 0.8,
                    "mode degeneracy tolerance": 0.01,
                },
                previous,
            )
            return selected, fake_fdtd.mode_updates

        unrotated, unrotated_updates = sweep_selection(0.0)
        self.assertEqual(unrotated["mode number"], 2)
        self.assertEqual(unrotated["selected mode order"], [2, 1])
        self.assertEqual(unrotated_updates, [[1, 2]])

        rotated, rotated_updates = sweep_selection(90.0)
        self.assertEqual(rotated["mode number"], 1)
        self.assertEqual(rotated["selected mode order"], [1, 2])
        self.assertEqual(rotated_updates, [[1, 2]])

        # The multi-GPU worker uses this fallback without the seed selector.
        # Two modes can be equally local-TE, so it must prefer the circular
        # Gaussian member of the neff-degenerate pair over an elongated field.
        coordinate = np.linspace(-1.0, 1.0, 25)
        grid_x, grid_y = np.meshgrid(coordinate, coordinate, indexing="ij")
        elongated_amplitude = np.exp(
            -0.25 * ((grid_x / 0.75) ** 2 + (grid_y / 0.13) ** 2)
        )
        circular_amplitude = np.exp(
            -0.25 * ((grid_x / 0.28) ** 2 + (grid_y / 0.28) ** 2)
        )
        high_mode = np.zeros((25, 25, 3), dtype=complex)
        elongated_mode = np.zeros_like(high_mode)
        circular_mode = np.zeros_like(high_mode)
        high_mode[..., 0] = circular_amplitude
        elongated_mode[..., 1] = elongated_amplitude
        circular_mode[..., 1] = circular_amplitude

        class ThreeModeFdtd(FakeFdtd):
            def getresult(self, _path, result_name):
                if result_name == "mode profiles":
                    return {"E1": high_mode, "E2": elongated_mode, "E3": circular_mode}
                if result_name == "neff":
                    return {
                        "lambda": np.linspace(1.50e-6, 1.60e-6, 4),
                        "n": np.asarray([1, 2, 3]),
                        "neff": np.asarray([
                            [1.70, 1.4440, 1.4445],
                            [1.69, 1.4439, 1.4444],
                            [1.68, 1.4438, 1.4443],
                            [1.67, 1.4437, 1.4442],
                        ]),
                    }
                raise AssertionError(result_name)

        three_mode_fdtd = ThreeModeFdtd()
        three_mode_namespace = {"np": np, "fdtd": three_mode_fdtd}
        exec(
            compile(
                ast.Module(body=helper_nodes, type_ignores=[]),
                "<sweep-three-mode-selection>",
                "exec",
            ),
            three_mode_namespace,
        )
        selected = three_mode_namespace["_sweep_reselect_fiber_local_te"](
            "FDTD::ports::fiber_source",
            {
                "name": "fiber_source",
                "angle phi": 0.0,
                "candidate mode numbers": [1, 2, 3],
                "fiber target neff": 1.444,
                "minimum local TE fraction": 0.8,
                "mode degeneracy tolerance": 0.01,
            },
            {"mode number": 2, "candidate mode numbers": [1, 2, 3]},
        )
        self.assertEqual(selected["degenerate mode pair"], [2, 3])
        self.assertEqual(selected["mode number"], 3)
        self.assertGreater(
            selected["gaussian scores"]["3"], selected["gaussian scores"]["2"]
        )
        self.assertGreater(
            selected["circularity scores"]["3"], selected["circularity scores"]["2"]
        )

        sweep_cases, spec = soi_sweep_cases()
        notebook, _warnings = lumerical.generate_lumerical_multigpu_sweep_notebook(
            sweep_cases,
            {
                "included_layers": [[1, 0], [2, 0]],
                "material_stack": lumerical.default_stack(
                    "SOI grating coupler (Ansys)"
                ),
                "resource_mode": "GPU",
                "run_after_build": True,
            },
            spec,
        )
        orchestration_cell = cell_source_containing(notebook, "def _worker_payload")
        self.assertIn(
            "+ 'SWEEP_FIBER_MODE_SELECTIONS = ' + "
            "repr(SWEEP_FIBER_MODE_SELECTIONS) + '\\n'",
            orchestration_cell,
        )
        self.assertIn(
            "SWEEP_FIBER_MODE_SELECTIONS = dict("
            "_MULTIGPU_BUILD_STATE.get('port_modes') or {})",
            "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]),
        )

    def test_generic_grating_has_editable_terminal_output_arc(self) -> None:
        grating = component("Grating coupler")
        self.assertEqual(grating["params"]["L_extra"], 10.0)
        self.assertEqual(grating["params"]["fdtd_port_offset_from_waveguide_end_um"], 2.0)
        with_arc, _ = component_geometry_arrays(grating)
        without_arc_component = deepcopy(grating)
        without_arc_component["params"]["L_extra"] = 0.0
        without_arc, _ = component_geometry_arrays(without_arc_component)
        with_arc_xmax = max(float(points[:, 0].max()) for points, _layer, _datatype in with_arc)
        without_arc_xmax = max(float(points[:, 0].max()) for points, _layer, _datatype in without_arc)
        self.assertAlmostEqual(with_arc_xmax - without_arc_xmax, 10.0, places=2)

    def test_apodized_grating_fill_factors_match_n(self) -> None:
        grating = component("Grating coupler")
        grating["params"].update({"N": 5, "fill_factors": "linspace(0.30, 0.50)"})
        resolved = resolve_grating_fill_factors(
            grating["params"], 5, scalar_key="fill_factor"
        )
        np.testing.assert_allclose(resolved, [0.30, 0.35, 0.40, 0.45, 0.50])
        polygons, _ = component_geometry_arrays(grating)
        self.assertGreater(len(polygons), 5)

        explicit = deepcopy(grating)
        explicit["params"]["fill_factors"] = "[0.31, 0.34, 0.37, 0.40, 0.43]"
        explicit_values = resolve_grating_fill_factors(
            explicit["params"], 5, scalar_key="fill_factor"
        )
        np.testing.assert_allclose(explicit_values, [0.31, 0.34, 0.37, 0.40, 0.43])

        mismatch = deepcopy(grating)
        mismatch["params"]["fill_factors"] = "[0.3, 0.4]"
        with self.assertRaisesRegex(ValueError, "Expected 5 values"):
            component_geometry_arrays(mismatch)

    def test_soi_grating_accepts_apodized_fill_factors(self) -> None:
        grating = component("GC-SOI")
        period_count = int(np.ceil(grating["params"]["target_length"] / grating["params"]["pitch"]))
        grating["params"]["fill_factors"] = "linspace(0.30, 0.50)"
        values = resolve_grating_fill_factors(
            grating["params"], period_count, scalar_key="duty_cycle"
        )
        self.assertEqual(values.size, period_count)
        self.assertAlmostEqual(float(values[0]), 0.30)
        self.assertAlmostEqual(float(values[-1]), 0.50)
        polygons, _ = component_geometry_arrays(grating)
        self.assertEqual(len(polygons), 47)

    def test_parse_sequence_symbolic_n(self) -> None:
        self.assertEqual(parse_sequence("linspace(0.2, 0.8)", 4), [0.2, 0.4, 0.6000000000000001, 0.8])
        self.assertEqual(parse_sequence("linspace(0.2, 0.8, N)", 4), [0.2, 0.4, 0.6000000000000001, 0.8])

    def test_lumerical_sweep_parameters_and_cartesian_validation(self) -> None:
        grating = component("GC-SOI")
        eligible = {
            item["parameter"]: item
            for item in sweepable_component_parameters(grating)
        }
        self.assertIn("pitch", eligible)
        self.assertIn("duty_cycle", eligible)
        self.assertEqual(eligible["pitch"]["short_name"], "P")
        self.assertEqual(eligible["duty_cycle"]["short_name"], "F")
        for excluded in (
            "slab_layer", "slab_datatype", "etched_layer", "etched_datatype",
            "tolerance", "h_total", "etch_depth", "waveguide_effective_index",
            "waveguide_mode_search_count", "fiber_core_diameter_um",
        ):
            self.assertNotIn(excluded, eligible)

        spec = normalize_lumerical_sweep_spec(
            grating,
            [
                {"parameter": "pitch", "values": [0.65, 0.67, 0.69]},
                {"parameter": "duty_cycle", "values": [0.38, 0.42]},
            ],
        )
        self.assertEqual(spec["point_count"], 6)
        self.assertFalse(spec["save_each_fsp"])
        self.assertEqual(
            expand_lumerical_sweep_points(spec),
            [
                {"pitch": 0.65, "duty_cycle": 0.38},
                {"pitch": 0.65, "duty_cycle": 0.42},
                {"pitch": 0.67, "duty_cycle": 0.38},
                {"pitch": 0.67, "duty_cycle": 0.42},
                {"pitch": 0.69, "duty_cycle": 0.38},
                {"pitch": 0.69, "duty_cycle": 0.42},
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_lumerical_sweep_spec(
                grating, [{"parameter": "pitch", "values": [0.67, 0.67]}]
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            normalize_lumerical_sweep_spec(
                grating, [{"parameter": "pitch", "values": [0.67]}]
            )
        with self.assertRaisesRegex(ValueError, "limit"):
            normalize_lumerical_sweep_spec(
                grating,
                [
                    {"parameter": "pitch", "values": np.linspace(0.6, 0.8, 11)},
                    {"parameter": "duty_cycle", "values": np.linspace(0.3, 0.5, 10)},
                ],
            )

        generic = component("Grating coupler")
        with self.assertRaisesRegex(ValueError, "whole number"):
            normalize_lumerical_sweep_spec(
                generic, [{"parameter": "N", "values": [10, 11.5]}]
            )

    def test_optimized_lumerical_sweep_notebook_builds_once_and_names_ce_curves(self) -> None:
        from max_layout.ui.window import NativeLayoutWindow

        class Factory:
            make_component = NativeLayoutWindow.make_component
            automatic_simulation_companions = NativeLayoutWindow.automatic_simulation_companions
            synchronize_automatic_simulation_companions = NativeLayoutWindow.synchronize_automatic_simulation_companions

            def __init__(self) -> None:
                self.components = []
                self.next_uid = 1

        factory = Factory()
        grating = factory.make_component("GC-SOI", 0.0, 0.0)
        factory.components.append(grating)
        factory.components.extend(factory.automatic_simulation_companions(grating))
        spec = normalize_lumerical_sweep_spec(
            grating,
            [
                {"parameter": "pitch", "values": [0.65, 0.75]},
                {"parameter": "duty_cycle", "values": [0.46, 0.56]},
            ],
        )
        original = deepcopy(factory.components)
        sweep_cases = []
        for values in expand_lumerical_sweep_points(spec):
            variant_components = deepcopy(factory.components)
            variant_grating = next(item for item in variant_components if item["uid"] == grating["uid"])
            variant_grating["params"].update(values)
            variant_factory = Factory()
            variant_factory.components = variant_components
            variant_factory.next_uid = 1 + max(item["uid"] for item in variant_components)
            variant_factory.synchronize_automatic_simulation_companions(variant_grating)
            sweep_cases.append({"values": values, "components": variant_factory.components})
        self.assertEqual(factory.components, original)

        notebook, warnings = generate_lumerical_sweep_notebook(
            sweep_cases,
            {
                "included_layers": [[1, 0], [2, 0]],
                "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
                "wavelength_start_um": 1.50,
                "wavelength_stop_um": 1.60,
                "frequency_points": 11,
                "resource_mode": "GPU",
                "run_after_build": True,
                "project_file": "gc_soi_sweep.fsp",
            },
            spec,
        )
        self.assertTrue(any("Layout origin moved" in warning for warning in warnings))
        self.assertEqual(notebook["metadata"]["max_layout"]["point_count"], 4)
        self.assertFalse(notebook["metadata"]["max_layout"]["per_point_fsp"])
        first_source = "".join(notebook["cells"][0]["source"])
        self.assertIn("RUN_SIMULATION = True", first_source)
        self.assertIn(
            "One pre-solve inspection FSP and one solved/best FSP are always stored.",
            first_source,
        )
        all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<sweep notebook cell {index}>", "exec")
        # The nominal model is constructed exactly once.  The sweep loop
        # hot-swaps geometry without a cache-load constructor or another FDTD
        # owner.
        self.assertEqual(all_source.count("lumapi.FDTD("), 1)
        self.assertNotIn("MODEL_CACHE_HIT", all_source)
        self.assertNotIn("MODEL_CACHE_KEY", all_source)
        self.assertNotIn("REMOTE_MODEL_CACHE_FSP", all_source)
        self.assertNotIn("REUSE_EXACT_MODEL_CACHE", all_source)
        self.assertNotIn("SAVE_EXACT_MODEL_CACHE_ON_MISS", all_source)
        self.assertIn("SETTINGS['save_inspection_fsp'] = True", all_source)
        self.assertIn("SETTINGS['save_final_fsp'] = True", all_source)
        self.assertIn("save_verified_project(REMOTE_INSPECTION_PROJECT_FILE)", all_source)
        self.assertIn("fdtd.save(REMOTE_BEST_SWEEP_FSP)", all_source)
        self.assertIn("The required winning sweep FSP was not created", all_source)
        self.assertIn("one persistent Lumerical/GPU session", all_source)
        self.assertIn("_layer_builder_geometry(layer_x_um, layer_y_um, GEOMETRY)", all_source)
        self.assertIn('fdtd.run("FDTD", resource_mode)', all_source)
        self.assertNotIn('fdtd.save(_sweep_case_npz', all_source)
        self.assertIn("no per-point FSP is saved", all_source)
        self.assertIn("All sweep solves complete; post-processing resource is CPU", all_source)
        self.assertIn('fdtd.setresource("FDTD", 1, "active", False)', all_source)
        self.assertIn("Sweep was not run, so there are no sweep results to fetch.", all_source)
        self.assertNotIn("lam.fetch(REMOTE_MODEL_CACHE_FSP", all_source)
        self.assertIn("CE-maximum-", all_source)
        self.assertIn("_responses_db", all_source)
        self.assertIn("_maximum_db", all_source)
        self.assertIn("10.0 * np.log10", all_source)
        self.assertIn("maximum_response_db", all_source)
        self.assertIn("target_response_db", all_source)
        self.assertIn("coupling efficiency [dB]", all_source)
        self.assertIn("Maximum CE across wavelength — dB", all_source)
        self.assertIn("Maximum CE for each parameter combination — dB", all_source)
        self.assertIn("result_stems", all_source)
        self.assertIn("sweep_live_progress.jsonl", all_source)
        self.assertIn("progress_mode='sweep'", all_source)
        self.assertIn("current_fraction", all_source)
        self.assertIn('"progress_type": "sweep"', all_source)
        self.assertIn('"completed"', all_source)
        self.assertIn('"peak_response"', all_source)
        self.assertEqual(len(assignment_value(notebook, "SWEEP_CODE_FINGERPRINT")), 64)

        payload_source = cell_source_containing(notebook, "_SWEEP_CASES_B64")
        encoded = assignment_value(notebook, "_SWEEP_CASES_B64")
        decoded_cases = json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))
        self.assertEqual(
            [case["result_stem"] for case in decoded_cases],
            [
                "CE-P=0.65-F=0.46",
                "CE-P=0.65-F=0.56",
                "CE-P=0.75-F=0.46",
                "CE-P=0.75-F=0.56",
            ],
        )
        bbox = assignment_value(notebook, "BOUNDING_BOX_UM")
        for case in decoded_cases:
            points = np.vstack([
                np.asarray(polygon["vertices_um"], dtype=float)
                for polygon in case["target_geometry"]
            ])
            self.assertGreaterEqual(float(points[:, 0].min()), bbox[0] - 1e-9)
            self.assertGreaterEqual(float(points[:, 1].min()), bbox[1] - 1e-9)
            self.assertLessEqual(float(points[:, 0].max()), bbox[2] + 1e-9)
            self.assertLessEqual(float(points[:, 1].max()), bbox[3] + 1e-9)
        self.assertIn("SWEEP_CASES =", payload_source)

        window_source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(window_source.count('addAction("Lumerical run…")'), 2)
        self.assertGreaterEqual(window_source.count('addAction("Lumerical sweep…")'), 2)
        self.assertIn("def export_lumerical_sweep_notebook", window_source)
        self.assertIn(
            'saved_export = copy.deepcopy(target_component.get("lumerical_export_settings", {}))',
            window_source,
        )
        self.assertIn("variant_components = copy.deepcopy(export_components)", window_source)

    def test_multigpu_sweep_is_separate_parallel_resumable_export(self) -> None:
        sweep_cases, spec = soi_sweep_cases()
        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            "wavelength_start_um": 1.50,
            "wavelength_stop_um": 1.60,
            "frequency_points": 11,
            "resource_mode": "GPU",
            "run_after_build": True,
            "project_file": "gc_soi_sweep_multigpu.fsp",
        }

        notebook, warnings = lumerical.generate_lumerical_multigpu_sweep_notebook(
            sweep_cases, configuration, spec
        )
        self.assertTrue(any("Layout origin moved" in warning for warning in warnings))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell["source"]),
                    f"<multi-GPU sweep notebook cell {index}>",
                    "exec",
                )

        metadata = notebook["metadata"]["max_layout"]
        self.assertEqual(metadata["export"], "lumerical-fdtd-sweep-multigpu")
        self.assertEqual(metadata["execution"], "multi-node-parallel-workers")
        self.assertEqual(metadata["point_count"], 4)
        self.assertEqual(metadata["nominal_fsp_count"], 1)
        self.assertFalse(metadata["per_point_fsp"])
        self.assertEqual(
            metadata["lumerical_multigpu"],
            {
                "node_count": 8,
                "simulations_per_gpu": 1,
                "max_parallel_simulations": 8,
            },
        )
        self.assertEqual(
            assignment_value(notebook, "MULTIGPU_SETTINGS"),
            metadata["lumerical_multigpu"],
        )

        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        lower_source = all_source.lower()
        self.assertIn("SHARED_NOMINAL_FSP", all_source)
        self.assertIn("SWEEP_SHARED_CHECKPOINT_DIR", all_source)
        self.assertIn("ThreadPoolExecutor", all_source)
        self.assertIn("as_completed", all_source)
        self.assertRegex(lower_source, r"worker[^\n]{0,100}(work|workspace|remote_work)")
        self.assertRegex(lower_source, r"(worker_index|worker_id|worker_slot)")
        self.assertRegex(
            lower_source,
            r"(os\.path\.join|path\()[^\n]{0,180}worker",
        )
        self.assertIn("Reusing completed sweep checkpoint", all_source)
        self.assertIn("os.path.isfile(case_path)", all_source)
        self.assertIn('fdtd.setresource("FDTD", 1, "active", False)', all_source)
        self.assertRegex(
            all_source,
            r"finally:\n\s+for record in MULTIGPU_WORKER_RECORDS",
        )
        self.assertIn("_cleanup_worker(record)", all_source)
        self.assertIn("FDTD STOP UNCONFIRMED; HPC Packs were NOT returned", all_source)
        self.assertIn("_finalize_sweep_results", all_source)
        self.assertIn("MULTIGPU_FATAL_ERRORS", all_source)
        self.assertIn("checkpoint_schema", all_source)
        self.assertIn("SWEEP_RUNTIME_VERSION", all_source)
        self.assertIn("SWEEP_CODE_FINGERPRINT", all_source)
        self.assertIn("_report_multigpu_progress", all_source)
        self.assertIn("Multi-GPU sweep [", all_source)
        self.assertIn("finished | failed", all_source)
        self.assertIn("peak_response", all_source)
        self.assertIn("socket.gethostname()", all_source)
        self.assertIn("uuid.uuid4().hex", all_source)
        self.assertIn("stop_work_processes(timeout=25)", all_source)
        self.assertIn("_multigpu_run_once_checked", all_source)
        self.assertIn("remaining_pids", all_source)
        self.assertRegex(
            lower_source,
            r"(release|return|check.?in)[^\n]{0,100}(licen[cs]e|hpc pack)",
        )
        self.assertIn("one simulation per gpu", lower_source)
        self.assertIn("same-gpu oversubscription is intentionally disabled", lower_source)
        self.assertIn("Multi-GPU sweep disabled by RUN_SIMULATION in cell 1.", all_source)
        self.assertIn("shutil.copy2(", all_source)
        self.assertIn("WORKER_RUNTIME_FSP", all_source)

        inventory_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "Inventory preflight complete" in "".join(cell.get("source", []))
        )
        orchestration_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "Acquire licences only now" in "".join(cell.get("source", []))
        )
        recovery_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "Idempotent emergency cleanup" in "".join(cell.get("source", []))
        )
        self.assertNotIn("_multigpu_run_checked(\n    lam, MULTIGPU_LICENSE_CHECKOUT_REMOTE", inventory_cell)
        self.assertIn("Lambda.run_once", inventory_cell)
        self.assertIn("Lambda.stop_work_processes", inventory_cell)
        self.assertIn('("run_once", "stop_work_processes")', inventory_cell)
        self.assertIn("MULTIGPU_LICENSE_CHECKOUT_REMOTE", orchestration_cell)
        self.assertIn('"--expires", "__PIRIS_HPC_EXPIRY__"', inventory_cell)
        self.assertIn("HPC_PACK_DURATION_MINUTES", inventory_cell)
        self.assertIn('"PT%dM" % _hpc_duration_minutes', inventory_cell)
        self.assertIn(
            "_multigpu_run_once_checked(client, MULTIGPU_LICENSE_RELEASE_REMOTE",
            orchestration_cell,
        )
        self.assertIn("_client.stop_work_processes(timeout=25)", recovery_cell)
        self.assertIn(
            "_release_output = _multigpu_run_once_checked(", recovery_cell
        )
        self.assertLess(
            recovery_cell.index("_release_output = _multigpu_run_once_checked("),
            recovery_cell.index("_client.close()"),
        )
        shared_fsp_cell_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if "Every worker needs one shared seed" in "".join(cell.get("source", []))
        )
        orchestration_cell_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if "Acquire licences only now" in "".join(cell.get("source", []))
        )
        self.assertLess(shared_fsp_cell_index, orchestration_cell_index)

        # The original one-session exporter remains an independent path.
        sequential, _ = generate_lumerical_sweep_notebook(
            sweep_cases, configuration, spec
        )
        sequential_metadata = sequential["metadata"]["max_layout"]
        self.assertEqual(sequential_metadata["export"], "lumerical-fdtd-sweep")
        self.assertEqual(
            sequential_metadata["execution"], "one-session-layer-builder-hot-swap"
        )
        self.assertNotIn("lumerical_multigpu", sequential_metadata)
        sequential_source = "\n".join(
            "".join(cell.get("source", [])) for cell in sequential["cells"]
        )
        self.assertNotIn("ThreadPoolExecutor", sequential_source)

        window_source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(
            window_source.count('addAction("Lumerical sweep-multithread…")'), 2
        )
        self.assertIn(
            "def export_lumerical_multigpu_sweep_notebook", window_source
        )

    def test_sequential_sweep_schema_failure_cancels_then_resets_cpu_and_raises(self) -> None:
        source = lumerical._SWEEP_RUNNER_REMOTE

        schema_handler = source.index("except SweepResultSchemaError as exc:")
        cancellation = source.index("remaining solves were cancelled", schema_handler)
        loop_break = source.index("        break", cancellation)
        cpu_reset = source.index(
            'fdtd.setresource("FDTD", 2, "device type", "CPU")', loop_break
        )
        final_schema_guard = source.index("if schema_failure is not None:", cpu_reset)
        final_raise = source.index("    raise RuntimeError(", final_schema_guard)

        self.assertLess(schema_handler, cancellation)
        self.assertLess(cancellation, loop_break)
        self.assertLess(loop_break, cpu_reset)
        self.assertLess(cpu_reset, final_schema_guard)
        self.assertLess(final_schema_guard, final_raise)
        self.assertIn(
            "if schema_failure is None:\n    best_sweep_index = _finalize_sweep_results(failures)",
            source,
        )

    def test_multigpu_sweep_preflights_one_result_schema_before_queue_submission(self) -> None:
        sweep_cases, spec = soi_sweep_cases()
        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            "resource_mode": "GPU",
            "run_after_build": True,
        }
        notebook, _ = lumerical.generate_lumerical_multigpu_sweep_notebook(
            sweep_cases, configuration, spec
        )
        orchestration = cell_source_containing(notebook, "Result-schema preflight")

        preflight_solve = orchestration.index("'Result-schema preflight — '")
        queue_filter = orchestration.index(
            "if case_index != schema_preflight_index:", preflight_solve
        )
        queue_fill = orchestration.index("_case_queue.put(case_index)", queue_filter)
        worker_submission = orchestration.index(
            "pool.submit(_worker_loop, record)", queue_fill
        )

        self.assertEqual(orchestration.count("'Result-schema preflight — '"), 1)
        self.assertIn("schema_preflight_index = 0", orchestration)
        self.assertLess(preflight_solve, queue_filter)
        self.assertLess(queue_filter, queue_fill)
        self.assertLess(queue_fill, worker_submission)
        self.assertIn(
            "Result-schema preflight passed; its checkpoint will be reused during aggregation.",
            orchestration,
        )

    def test_write_multigpu_sweep_notebook_preserves_parallel_contract(self) -> None:
        sweep_cases, spec = soi_sweep_cases()
        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            "wavelength_start_um": 1.50,
            "wavelength_stop_um": 1.60,
            "frequency_points": 5,
            "resource_mode": "GPU",
            "run_after_build": True,
            "project_file": "gc_soi_sweep_multigpu.fsp",
            "lumerical_multigpu": {
                "node_count": 4,
                "simulations_per_gpu": 1,
                "max_parallel_simulations": 4,
            },
        }
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "gc-soi_sweep_multigpu.ipynb"
            warnings = lumerical.write_lumerical_multigpu_sweep_notebook(
                output, sweep_cases, configuration, spec
            )
            self.assertTrue(output.is_file())
            self.assertTrue(any("Layout origin moved" in warning for warning in warnings))
            saved = json.loads(output.read_text(encoding="utf-8"))

        metadata = saved["metadata"]["max_layout"]
        self.assertEqual(metadata["export"], "lumerical-fdtd-sweep-multigpu")
        self.assertEqual(
            metadata["lumerical_multigpu"],
            {
                "node_count": 4,
                "simulations_per_gpu": 1,
                "max_parallel_simulations": 4,
            },
        )
        self.assertEqual(saved["nbformat"], 4)

    def test_multigpu_rejects_same_gpu_oversubscription(self) -> None:
        sweep_cases, spec = soi_sweep_cases()
        configuration = {
            "included_layers": [[1, 0], [2, 0]],
            "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
            "resource_mode": "GPU",
            "run_after_build": True,
            "lumerical_multigpu": {"node_count": 4, "simulations_per_gpu": 2},
        }
        with self.assertRaisesRegex(ValueError, "exactly one simulation per GPU"):
            lumerical.generate_lumerical_multigpu_sweep_notebook(
                sweep_cases, configuration, spec
            )

    def test_live_sweep_formatter_reports_overall_progress_eta_and_peak(self) -> None:
        import ast

        tree = ast.parse(lumerical._LAMBDA_CONNECT_CELL)
        formatter = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_format_live_sweep_rows"
        )
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[formatter], type_ignores=[])),
                "<live sweep formatter>",
                "exec",
            ),
            namespace,
        )
        lines = namespace["_format_live_sweep_rows"](
            [
                {
                    "sequence": 0,
                    "status": "running",
                    "case_index": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "total_count": 4,
                    "values": {"pitch": 0.75, "fill_factor": 0.5},
                    "display_label": "P=0.75, F=0.5",
                },
                {
                    "sequence": 1,
                    "status": "completed",
                    "case_index": 0,
                    "completed_count": 1,
                    "failed_count": 0,
                    "total_count": 4,
                    "values": {"pitch": 0.75, "fill_factor": 0.5},
                    "display_label": "P=0.75, F=0.5",
                    "peak_response": 0.437445,
                    "peak_wavelength_nm": 1314.935,
                },
            ],
            elapsed_seconds=120.0,
        )
        rendered = "\n".join(lines)
        self.assertIn("1/4 finished ( 25.00%)", rendered)
        self.assertIn("ETA 6.0 min", rendered)
        self.assertIn("P=0.75, F=0.5", rendered)
        self.assertIn("peak 0.437445 at 1314.935 nm", rendered)

    def test_sweep_checkpoint_rejects_nonfinite_or_wrong_objective(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_directory = Path(temporary_directory) / "checkpoints"
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_SHARED_CHECKPOINT_DIR": str(checkpoint_directory),
                "SWEEP_HASH": "unit-test-sweep-hash",
                "SWEEP_CASES": [{"values": {"pitch": 0.65}, "display_label": "P=0.65"}],
                "SWEEP_SPEC": {
                    "component_uid": 1,
                    "component_kind": "GC-SOI",
                    "axes": [{"parameter": "pitch", "short_name": "P"}],
                },
                "SETTINGS": {"wavelength_start_um": 1.5, "wavelength_stop_um": 1.6},
                "PORTS": [],
                "FIBER_GEOMETRIES": [],
                "MONITORS": [],
                "GRATING_ANALYSIS": {"enabled": True},
                "MMI_ANALYSIS": None,
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            wavelength = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])
            response = np.asarray([0.2, 0.4, 0.3])
            namespace["_save_sweep_case"](
                0,
                "coupling_efficiency",
                wavelength,
                response,
                {
                    "fiber_input_power": np.ones(response.size),
                    "waveguide_mode_power": response,
                    "waveguide_total_power": response + 0.05,
                    "waveguide_total_transmission": response + 0.05,
                },
            )
            self.assertTrue(namespace["_sweep_case_is_complete"](0))
            original_fingerprint = namespace["SWEEP_CODE_FINGERPRINT"]
            namespace["SWEEP_CODE_FINGERPRINT"] = "changed-exporter-code"
            self.assertFalse(namespace["_sweep_case_is_complete"](0))
            namespace["SWEEP_CODE_FINGERPRINT"] = original_fingerprint
            self.assertTrue(namespace["_sweep_case_is_complete"](0))

            checkpoint = Path(namespace["_sweep_case_npz"](0))
            with np.load(checkpoint, allow_pickle=False) as data:
                payload = {key: np.asarray(data[key]) for key in data.files}
            payload["primary_response"] = np.asarray([0.2, np.nan, 0.3])
            np.savez_compressed(checkpoint, **payload)
            self.assertFalse(namespace["_sweep_case_is_complete"](0))

            payload["primary_response"] = response
            payload["primary_name"] = np.asarray(["wrong_objective"])
            np.savez_compressed(checkpoint, **payload)
            self.assertFalse(namespace["_sweep_case_is_complete"](0))

    def test_grating_sweep_resolves_logical_mode_expansion_result_name(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])

        class FakeFDTD:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str | None]] = []

            def getresult(self, path, result_name=None):
                self.calls.append((str(path), result_name))
                if str(path) == "uid_1_fiber_input_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.ones(3)}
                if str(path) == "uid_1_waveguide_total_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.asarray([0.7, 0.8, 0.75])}
                if str(path) in {
                    "FDTD::ports::uid_1_waveguide_receiver",
                    "::model::FDTD::ports::uid_1_waveguide_receiver",
                }:
                    if result_name == "expansion for waveguide_power":
                        return {
                            "lambda": wavelength_m,
                            "T_out": np.asarray([0.30, 0.40, 0.35]),
                        }
                    raise RuntimeError("Can not find result %r" % result_name)
                raise RuntimeError("Can not find result %r" % result_name)

        with TemporaryDirectory() as temporary_directory:
            fake_fdtd = FakeFDTD()
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "logical-expansion-name",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_input_power_monitor_name": "uid_1_fiber_input_power",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_power_monitor_name": "uid_1_waveguide_total_power",
                    "waveguide_total_power_sign": -1.0,
                    "waveguide_port_name": "uid_1_waveguide_receiver",
                    "waveguide_port_expansion_result_name": "expansion for waveguide_power",
                    "waveguide_port_modal_direction": "T_out",
                },
                "SWEEP_PORT_MODE_SELECTIONS": {
                    "uid_1_waveguide_receiver": {
                        "mode number": 1,
                        "selected mode order": [1],
                    }
                },
                "fdtd": fake_fdtd,
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            primary_name, returned_wavelength, response, arrays = namespace[
                "_extract_sweep_result"
            ]()

        self.assertEqual(primary_name, "coupling_efficiency")
        np.testing.assert_allclose(returned_wavelength, wavelength_m)
        np.testing.assert_allclose(response, [0.30, 0.40, 0.35])
        np.testing.assert_allclose(arrays["waveguide_mode_power"], response)
        self.assertIn(
            (
                "FDTD::ports::uid_1_waveguide_receiver",
                "expansion for waveguide_power",
            ),
            fake_fdtd.calls,
        )

    def test_grating_sweep_discovers_available_mode_expansion_result(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6])

        class FakeFDTD:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str | None]] = []

            def getresult(self, path, result_name=None):
                path = str(path)
                self.calls.append((path, result_name))
                if path == "uid_1_fiber_input_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": np.asarray([-0.5, -1.0])}
                if path == "uid_1_waveguide_total_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": np.asarray([-0.4, -0.8])}
                if path in {
                    "FDTD::ports::uid_1_waveguide_receiver",
                    "::model::FDTD::ports::uid_1_waveguide_receiver",
                }:
                    if result_name == "expansion for port monitor":
                        return {
                            "lambda": wavelength_m,
                            "T_out": np.asarray([0.25, 0.50]),
                        }
                    raise RuntimeError("Can not find result %r" % result_name)
                raise RuntimeError("Can not find result %r" % result_name)

        with TemporaryDirectory() as temporary_directory:
            fake_fdtd = FakeFDTD()
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "discover-expansion-name",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_input_power_monitor_name": "uid_1_fiber_input_power",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_power_monitor_name": "uid_1_waveguide_total_power",
                    "waveguide_total_power_sign": -1.0,
                    "waveguide_port_name": "uid_1_waveguide_receiver",
                    # The requested logical result is unavailable, so the
                    # runtime must fall back to the standard port result.
                    "waveguide_port_expansion_result_name": "missing logical result",
                    "waveguide_port_modal_direction": "T_out",
                },
                "SWEEP_PORT_MODE_SELECTIONS": {
                    "uid_1_waveguide_receiver": {
                        "mode number": 1,
                        "selected mode order": [1],
                    }
                },
                "fdtd": fake_fdtd,
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            primary_name, _, response, _ = namespace["_extract_sweep_result"]()

        self.assertEqual(primary_name, "coupling_efficiency")
        np.testing.assert_allclose(response, [0.5, 0.5])
        self.assertIn(
            (
                "FDTD::ports::uid_1_waveguide_receiver",
                "expansion for port monitor",
            ),
            fake_fdtd.calls,
        )

    def test_grating_sweep_extracts_verified_ey_mode_not_first_modal_column(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])

        class FakeFDTD:
            def getresult(self, path, result_name=None):
                path = str(path)
                if path == "uid_1_fiber_input_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.ones(3)}
                if path == "uid_1_waveguide_total_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.asarray([0.31, 0.41, 0.36])}
                if path in {
                    "FDTD::ports::uid_1_waveguide_receiver",
                    "::model::FDTD::ports::uid_1_waveguide_receiver",
                } and result_name == "expansion for port monitor":
                    return {
                        "lambda": wavelength_m,
                        "n": np.asarray([1, 2]),
                        # Mode 1 is the wrong polarization; mode 2 is the
                        # verified local-TE waveguide receiver mode.
                        "T_out": np.asarray([
                            [1e-5, 0.30],
                            [1e-5, 0.40],
                            [1e-5, 0.35],
                        ]),
                    }
                raise RuntimeError("Can not find result %r" % result_name)

        with TemporaryDirectory() as temporary_directory:
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "verified-ey-column",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_input_power_monitor_name": "uid_1_fiber_input_power",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_power_monitor_name": "uid_1_waveguide_total_power",
                    "waveguide_total_power_sign": -1.0,
                    "waveguide_port_name": "uid_1_waveguide_receiver",
                    "waveguide_port_expansion_result_name": "expansion for port monitor",
                    "waveguide_port_modal_direction": "T_out",
                },
                "SWEEP_PORT_MODE_SELECTIONS": {
                    "uid_1_waveguide_receiver": {
                        "mode number": 2,
                        "selected mode order": [2, 1],
                    }
                },
                "fdtd": FakeFDTD(),
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            primary_name, _, response, arrays = namespace["_extract_sweep_result"]()

        self.assertEqual(primary_name, "coupling_efficiency")
        np.testing.assert_allclose(response, [0.30, 0.40, 0.35])
        np.testing.assert_allclose(arrays["fiber_input_power"], 1.0)
        np.testing.assert_allclose(arrays["waveguide_mode_power"], response)

    def test_single_grating_analysis_extracts_verified_ey_mode_by_n_coordinate(self) -> None:
        import ast

        tree = ast.parse(lumerical._GRATING_ANALYSIS_REMOTE)
        wanted = {"_normalized_result_key", "_find_result_key", "_one_spectrum"}
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        }
        self.assertEqual(set(functions), wanted)
        module = ast.Module(
            body=[functions[name] for name in (
                "_normalized_result_key", "_find_result_key", "_one_spectrum"
            )],
            type_ignores=[],
        )
        namespace = {"np": np}
        exec(
            compile(ast.fix_missing_locations(module), "<single-ey-spectrum>", "exec"),
            namespace,
        )
        wavelength_m = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])
        dataset = {
            "lambda": wavelength_m,
            "n": np.asarray([1, 2]),
            "T_in": np.asarray([
                [8e-6, 0.99],
                [9e-6, 1.00],
                [7e-6, 0.98],
            ]),
        }
        returned_wavelength, selected = namespace["_one_spectrum"](
            dataset,
            "T_in",
            selected_mode_number=2,
            selected_mode_order=[2, 1],
        )
        np.testing.assert_allclose(returned_wavelength, wavelength_m)
        np.testing.assert_allclose(selected, [0.99, 1.00, 0.98])

    def test_sweep_reapplies_reselected_ey_source_mode_to_port_group(self) -> None:
        class FakeFDTD:
            def __init__(self) -> None:
                self.selected = None
                self.group = {}

            def select(self, path):
                self.selected = str(path)

            def set(self, name, value):
                self.group[str(name)] = value

            def get(self, name):
                return self.group[str(name)]

        with TemporaryDirectory() as temporary_directory:
            fake_fdtd = FakeFDTD()
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "source-mode-refresh",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [
                    {"name": "fiber_source"},
                    {"name": "fiber_measurement"},
                ],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_port_name": "fiber_source",
                    "fiber_input_measurement_port_name": "fiber_measurement",
                },
                "SWEEP_FIBER_MODE_SELECTIONS": {
                    "fiber_source": {
                        "mode number": 2,
                        "selected mode order": [2, 1],
                        "candidate mode numbers": [1, 2],
                    },
                    "fiber_measurement": {
                        "mode number": 2,
                        "selected mode order": [2, 1],
                        "candidate mode numbers": [1, 2],
                    },
                },
                "fdtd": fake_fdtd,
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            self.assertEqual(fake_fdtd.group["source port"], "fiber_source")
            self.assertEqual(fake_fdtd.group["source mode"], "mode 2")

            namespace["SWEEP_FIBER_MODE_SELECTIONS"]["fiber_source"].update({
                "mode number": 1,
                "selected mode order": [1, 2],
            })
            namespace["_restore_sweep_fiber_mode_contract"]()
            self.assertEqual(fake_fdtd.group["source mode"], "mode 1")

    def test_grating_sweep_expansion_failure_lists_available_results(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6])

        class FakeFDTD:
            def getresult(self, path, result_name=None):
                path = str(path)
                if path == "uid_1_fiber_input_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.ones(2)}
                if path == "uid_1_waveguide_total_power" and result_name == "T":
                    return {"lambda": wavelength_m, "T": -np.ones(2)}
                raise RuntimeError("Can not find result %r" % result_name)

        with TemporaryDirectory() as temporary_directory:
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "missing-expansion-name",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [],
                "MONITORS": [],
                "MMI_ANALYSIS": None,
                "GRATING_ANALYSIS": {
                    "fiber_input_power_monitor_name": "uid_1_fiber_input_power",
                    "fiber_input_power_sign": -1.0,
                    "waveguide_power_monitor_name": "uid_1_waveguide_total_power",
                    "waveguide_total_power_sign": -1.0,
                    "waveguide_port_name": "uid_1_waveguide_receiver",
                    "waveguide_port_expansion_result_name": "waveguide_power",
                    "waveguide_port_modal_direction": "T_out",
                },
                "SWEEP_PORT_MODE_SELECTIONS": {
                    "uid_1_waveguide_receiver": {
                        "mode number": 1,
                        "selected mode order": [1],
                    }
                },
                "fdtd": FakeFDTD(),
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            with self.assertRaises(RuntimeError) as caught:
                namespace["_extract_sweep_result"]()

        message = str(caught.exception)
        self.assertIn("waveguide_power", message)
        self.assertIn("port expansion", message)
        self.assertIn("expansion for port monitor", message)

    def test_mmi_sweep_normalizes_each_output_branch_to_measured_input(self) -> None:
        wavelength_m = np.asarray([1.50e-6, 1.55e-6, 1.60e-6])

        class FakeFDTD:
            def getresult(self, path, result_name=None):
                key = (str(path), result_name)
                results = {
                    ("::model::uid_1_input_reference", "T"): {
                        "lambda": wavelength_m,
                        "T": np.asarray([0.80, 0.80, 0.80]),
                    },
                    ("::model::FDTD::ports::uid_1_upper_right", "T"): {
                        "lambda": wavelength_m,
                        "T": np.asarray([0.32, 0.40, 0.36]),
                    },
                    ("::model::FDTD::ports::uid_1_lower_right", "T"): {
                        "lambda": wavelength_m,
                        "T": np.asarray([0.36, 0.38, 0.40]),
                    },
                }
                if key not in results:
                    raise RuntimeError("Can not find result %r" % (key,))
                return results[key]

        with TemporaryDirectory() as temporary_directory:
            namespace = {
                "REMOTE_WORK": temporary_directory,
                "SWEEP_HASH": "mmi-branch-normalization",
                "SWEEP_CASES": [],
                "SWEEP_SPEC": {},
                "SETTINGS": {},
                "PORTS": [],
                "MONITORS": [],
                "GRATING_ANALYSIS": None,
                "MMI_ANALYSIS": {
                    "input_reference_monitor_name": "uid_1_input_reference",
                    "output_port_names": [
                        "uid_1_upper_right",
                        "uid_1_lower_right",
                    ],
                },
                "fdtd": FakeFDTD(),
            }
            exec(lumerical._SWEEP_RUNTIME_REMOTE, namespace)
            primary_name, returned_wavelength, response, arrays = namespace[
                "_extract_sweep_result"
            ]()
            expected_primary_name = namespace["_sweep_expected_primary_name"]()

        expected_upper_over_input = np.asarray([0.40, 0.50, 0.45])
        expected_lower_over_input = np.asarray([0.45, 0.475, 0.50])
        self.assertEqual(primary_name, "output_1_over_input")
        self.assertEqual(expected_primary_name, "output_1_over_input")
        np.testing.assert_allclose(returned_wavelength, wavelength_m)
        np.testing.assert_allclose(response, expected_upper_over_input)
        np.testing.assert_allclose(
            arrays["output_1_over_input"], expected_upper_over_input
        )
        np.testing.assert_allclose(
            arrays["output_2_over_input"], expected_lower_over_input
        )
        np.testing.assert_allclose(
            arrays["total_output_over_input"],
            expected_upper_over_input + expected_lower_over_input,
        )
        np.testing.assert_allclose(
            arrays["output_1_ratio"] + arrays["output_2_ratio"], 1.0
        )

    def test_layer_builder_is_committed_in_memory_without_a_mode_seed_fsp(self) -> None:
        build_source = lumerical._BUILD_CELL
        self.assertIn("fdtd.runsetup()", build_source)
        self.assertNotIn("fdtd.save(mode_seed_project)", build_source)

    def test_accepting_sweep_parameters_opens_stack_dialog(self) -> None:
        from max_layout.ui import window as window_module

        target = component("GC-SOI")
        spec = normalize_lumerical_sweep_spec(
            target, [{"parameter": "pitch", "values": [0.65, 0.67]}]
        )
        opened: dict[str, object] = {}

        class FakeSweepDialog:
            parameters = ["pitch"]

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def exec(self):
                return window_module.QDialog.DialogCode.Accepted

            def sweep_spec(self):
                return spec

        class FakeExportDialog:
            def __init__(self, _components, _scope, saved=None, parent=None) -> None:
                opened["saved"] = saved
                opened["parent"] = parent

            def setWindowTitle(self, title: str) -> None:
                opened["title"] = title

            def exec(self):
                return window_module.QDialog.DialogCode.Rejected

        class Harness:
            components = [target]

            def selected_components(self):
                return [target]

            def lumerical_scope_options(self, _clicked):
                return [("Clicked component", [int(target["uid"])])]

        original_sweep_dialog = window_module.LumericalSweepDialog
        original_export_dialog = window_module.LumericalExportDialog
        try:
            window_module.LumericalSweepDialog = FakeSweepDialog
            window_module.LumericalExportDialog = FakeExportDialog
            window_module.NativeLayoutWindow.export_lumerical_sweep_notebook(
                Harness(), target
            )
        finally:
            window_module.LumericalSweepDialog = original_sweep_dialog
            window_module.LumericalExportDialog = original_export_dialog

        self.assertEqual(
            opened["title"], "Lumerical sweep — stack, domain, and GPU settings"
        )
        self.assertEqual(opened["saved"]["stack_preset"], "SOI grating coupler (Ansys)")

    def test_accepting_optimization_parameters_opens_locked_gpu_stack_dialog(self) -> None:
        from max_layout.ui import window as window_module

        target = component("GC-SOI")
        spec = {
            "version": 1,
            "method": "shape-adjoint",
            "component_uid": int(target["uid"]),
            "component_kind": "GC-SOI",
            "parameters": [
                {
                    "parameter": "pitch",
                    "initial": 0.6713,
                    "minimum": 0.64,
                    "maximum": 0.70,
                }
            ],
            "objective": {
                "kind": "grating_ce",
                "center_wavelength_um": 1.55,
                "bandwidth_nm": 40.0,
                "wavelength_start_um": 1.53,
                "wavelength_stop_um": 1.57,
                "wavelength_points": 5,
            },
            "optimizer": {"max_iterations": 12},
        }
        opened: dict[str, object] = {}

        class FakeOptimizationDialog:
            parameters = ["pitch"]

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def exec(self):
                return window_module.QDialog.DialogCode.Accepted

            def optimization_spec(self):
                return spec

        class FakeWidget:
            def __init__(self) -> None:
                self.value = None
                self.enabled = True
                self.tooltip = ""

            def setValue(self, value) -> None:
                self.value = value

            def setEnabled(self, enabled: bool) -> None:
                self.enabled = bool(enabled)

            def setToolTip(self, tooltip: str) -> None:
                self.tooltip = str(tooltip)

        class FakeCombo(FakeWidget):
            def setCurrentText(self, value: str) -> None:
                self.value = value

        class FakeCheck(FakeWidget):
            def setChecked(self, value: bool) -> None:
                self.value = bool(value)

        class FakeExportDialog:
            def __init__(self, _components, _scope, saved=None, parent=None) -> None:
                opened["saved"] = saved
                opened["parent"] = parent
                self.wavelength_start = FakeWidget()
                self.wavelength_stop = FakeWidget()
                self.frequency_points = FakeWidget()
                self.resource_mode = FakeCombo()
                self.run_after_build = FakeCheck()
                opened["dialog"] = self

            def setWindowTitle(self, title: str) -> None:
                opened["title"] = title

            def exec(self):
                return window_module.QDialog.DialogCode.Rejected

        class Harness:
            components = [target]

            def selected_components(self):
                return [target]

            def lumerical_scope_options(self, _clicked):
                return [("Clicked component", [int(target["uid"])])]

        original_optimization_dialog = window_module.LumericalOptimizationDialog
        original_export_dialog = window_module.LumericalExportDialog
        try:
            window_module.LumericalOptimizationDialog = FakeOptimizationDialog
            window_module.LumericalExportDialog = FakeExportDialog
            window_module.NativeLayoutWindow.export_lumerical_optimization_notebook(
                Harness(), target
            )
        finally:
            window_module.LumericalOptimizationDialog = original_optimization_dialog
            window_module.LumericalExportDialog = original_export_dialog

        self.assertEqual(
            opened["title"],
            "Lumerical adjoint optimization — stack, domain, and GPU settings",
        )
        self.assertEqual(opened["saved"]["wavelength_start_um"], 1.53)
        self.assertEqual(opened["saved"]["wavelength_stop_um"], 1.57)
        self.assertEqual(opened["saved"]["frequency_points"], 5)
        self.assertEqual(opened["saved"]["resource_mode"], "GPU")
        dialog = opened["dialog"]
        self.assertFalse(dialog.wavelength_start.enabled)
        self.assertFalse(dialog.wavelength_stop.enabled)
        self.assertFalse(dialog.frequency_points.enabled)
        self.assertEqual(dialog.resource_mode.value, "GPU")
        self.assertFalse(dialog.resource_mode.enabled)
        self.assertTrue(dialog.run_after_build.value)
        self.assertFalse(dialog.run_after_build.enabled)

        window_source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(
            window_source.count('addAction("Lumerical optimization…")'), 2
        )
        self.assertIn("def export_lumerical_optimization_notebook", window_source)

    def test_mmi_sweep_keeps_ports_monitors_mesh_orders_in_both_exporters(self) -> None:
        sweep_cases, spec = mmi_sweep_cases()
        eligible = {
            item["parameter"]
            for item in sweepable_component_parameters(sweep_cases[0]["components"][0])
        }
        self.assertTrue(
            {"mmi_length", "mmi_width", "wg_width", "port_sep"}.issubset(eligible)
        )
        self.assertTrue(
            {
                "input_reference_before_taper_um",
                "fdtd_port_clearance_um",
                "taper_points",
            }.isdisjoint(eligible)
        )

        material_stack = lumerical.default_stack("TFLN on SiO2")
        expected_mesh_orders = [6, 5, 2, 3, 1, 4, 7]
        for row, mesh_order in zip(material_stack, expected_mesh_orders):
            row["mesh_order"] = mesh_order
        configuration = {
            "included_layers": [[1, 0]],
            "material_stack": material_stack,
            "wavelength_start_um": 1.25,
            "wavelength_stop_um": 1.35,
            "frequency_points": 5,
            "resource_mode": "GPU",
            "run_after_build": True,
        }
        sequential, _ = generate_lumerical_sweep_notebook(
            sweep_cases, configuration, spec
        )
        multigpu, _ = lumerical.generate_lumerical_multigpu_sweep_notebook(
            sweep_cases, configuration, spec
        )

        self.assertEqual(
            sequential["metadata"]["max_layout"]["export"],
            "lumerical-fdtd-sweep",
        )
        self.assertEqual(
            multigpu["metadata"]["max_layout"]["export"],
            "lumerical-fdtd-sweep-multigpu",
        )
        for label, notebook in (("sequential", sequential), ("multigpu", multigpu)):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile(
                        "".join(cell["source"]),
                        f"<MMI {label} sweep cell {index}>",
                        "exec",
                    )
            self.assertEqual(
                [row["mesh_order"] for row in assignment_value(notebook, "MATERIAL_STACK")],
                expected_mesh_orders,
            )
            remote_cases = decoded_sweep_cases(notebook)
            self.assertEqual(len(remote_cases), 2)
            for remote_case in remote_cases:
                analysis = remote_case["mmi_analysis"]
                self.assertFalse(analysis["include_field_distribution"])
                self.assertNotIn("field_monitor_name", analysis)
                self.assertEqual(analysis["input_port_name"], "uid_1_left_external")
                self.assertEqual(
                    analysis["output_port_names"],
                    ["uid_1_upper_right", "uid_1_lower_right"],
                )
                self.assertEqual(
                    analysis["input_reference_monitor_name"],
                    "uid_1_input_reference",
                )
                self.assertEqual(
                    {
                        port["parent_port_name"]: port["name"]
                        for port in remote_case["ports"]
                    },
                    {
                        "left_external": "uid_1_left_external",
                        "upper_right": "uid_1_upper_right",
                        "lower_right": "uid_1_lower_right",
                    },
                )
                self.assertTrue(
                    all(float(port["span_um"]) >= 3.0 for port in remote_case["ports"])
                )
                self.assertEqual(
                    {float(port["target neff"]) for port in remote_case["ports"]},
                    {0.0},
                )
                self.assertEqual(
                    {str(port["polarization"]) for port in remote_case["ports"]},
                    {"local TE"},
                )
                self.assertEqual(
                    analysis["port_profile_labels"],
                    ["MMI input", "MMI upper output", "MMI lower output"],
                )
                self.assertEqual(
                    [monitor["name"] for monitor in remote_case["monitors"]],
                    ["uid_1_input_reference"],
                )

            first_ports = {
                port["parent_port_name"]: port
                for port in remote_cases[0]["ports"]
            }
            second_ports = {
                port["parent_port_name"]: port
                for port in remote_cases[1]["ports"]
            }
            self.assertEqual(
                first_ports["left_external"]["center"],
                second_ports["left_external"]["center"],
            )
            self.assertNotEqual(
                first_ports["upper_right"]["center"],
                second_ports["upper_right"]["center"],
            )
            runtime_source = cell_source_containing(notebook, "REMOTE_SWEEP_RUNTIME")
            self.assertIn('return "output_1_over_input"', runtime_source)
            self.assertIn('"output_2_over_input": output_2_over_input', runtime_source)
            self.assertIn("SWEEP_CHECKPOINT_SCHEMA = 6", runtime_source)
            self.assertIn('"code_fingerprint"', runtime_source)
            self.assertIn('"mmi_width": ("MMI width", "um")', runtime_source)
            self.assertIn('"mmi_length": ("MMI length", "um")', runtime_source)
            self.assertIn('"taper_power": ("MMI taper profile exponent", "")', runtime_source)
            self.assertIn('"input_reference_before_taper_um": ("Input power-reference distance before taper", "um")', runtime_source)
            self.assertIn("Target-best exact source parameters (JSON):", runtime_source)
            nominal = assignment_value(notebook, "SWEEP_NOMINAL_PARAMETERS")
            self.assertEqual(nominal["mmi_width"], 6.0)
            self.assertEqual(nominal["mmi_length"], 29.0)
            self.assertEqual(nominal["wg_width"], 1.2)
            self.assertEqual(nominal["taper_width"], 2.7)
            self.assertEqual(nominal["input_taper_length"], 10.0)
            self.assertEqual(nominal["output_taper_length"], 10.0)
            self.assertEqual(nominal["input_length"], 6.0)
            self.assertEqual(nominal["output_length"], 6.0)
            self.assertEqual(nominal["port_sep"], 3.25)
            self.assertEqual(nominal["taper_power"], 1.0)
            self.assertEqual(nominal["taper_points"], 41)
            self.assertEqual(nominal["fdtd_port_clearance_um"], 2.0)
            self.assertEqual(nominal["input_reference_before_taper_um"], 2.0)
            preview_source = cell_source_containing(notebook, "REMOTE_PORT_MODE_PROFILES")
            self.assertIn('elif MMI_ANALYSIS:', preview_source)
            self.assertIn('"MMI input"', preview_source)
            self.assertIn('"MMI upper output"', preview_source)
            self.assertIn('"MMI lower output"', preview_source)
            self.assertIn("PORT_MODE_SELECTIONS", preview_source)
            local_results = cell_source_containing(notebook, "_mmi_output_1_over_input")
            self.assertIn('label="upper output / input"', local_results)
            self.assertIn('label="lower output / input"', local_results)
            self.assertIn('label="ideal 50/50"', local_results)

    def test_mmi_export_includes_longitudinal_fundamental_field_plot(self) -> None:
        mmi = component("1x2 MMI")
        items = [mmi]
        for uid, parent_name, x, y, name in (
            (2, "left_external", 2.0, 0.0, "mmi_input"),
            (3, "upper_right", 55.0, 1.625, "mmi_upper"),
            (4, "lower_right", 55.0, -1.625, "mmi_lower"),
        ):
            port = component("FDTD port", uid=uid)
            port.update({"x": x, "y": y, "simulation_parent_uid": 1, "simulation_parent_port": parent_name})
            port["params"]["name"] = name
            items.append(port)
        reference = component("Power monitor", uid=5)
        reference.update({"x": 4.0, "simulation_parent_uid": 1, "simulation_parent_port": "left_external"})
        reference["params"]["name"] = "mmi_input_reference"
        items.append(reference)

        notebook, warnings = generate_lumerical_notebook(items, {"included_layers": [[1, 0]]})
        analysis = assignment_value(notebook, "MMI_ANALYSIS")
        self.assertTrue(analysis["include_field_distribution"])
        self.assertEqual(analysis["field_monitor_name"], "uid_1_mmi_field")
        monitors = assignment_value(notebook, "MONITORS")
        field_monitor = next(monitor for monitor in monitors if monitor["name"] == "uid_1_mmi_field")
        self.assertEqual(field_monitor["plane normal"], "Z")
        self.assertEqual(field_monitor["z reference"], "device center")
        self.assertTrue(any("longitudinal field-profile monitor" in warning for warning in warnings))
        source = cell_source_containing(notebook, "REMOTE_MMI_ANALYSIS =")
        self.assertIn("mmi_field_distribution.png", source)
        self.assertIn("field_intensity_normalized", source)
        self.assertIn("fdtd.getelectric(field_monitor_path, 1)", source)
        self.assertIn('_field_value("E2")', source)
        self.assertIn("vector E attribute", source)
        self.assertNotIn("field_result.get(component_name, 0.0)", source)
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER")
        self.assertIn('for component_name in ("Ex", "Ey", "Ez")', build_source)
        self.assertIn('fdtd.set("output " + component_name, True)', build_source)

    def test_mmi_field_plot_accepts_getelectric_when_component_keys_are_missing(self) -> None:
        wavelengths_m = np.asarray([1.25e-6, 1.35e-6])
        frequencies_hz = 299792458.0 / wavelengths_m
        x_m = np.linspace(0.0, 4.0e-6, 4)
        y_m = np.linspace(-1.0e-6, 1.0e-6, 3)
        electric_intensity = np.arange(1, 25, dtype=float).reshape(4, 3, 1, 2)
        ex_field = np.full((4, 3, 1, 2), 0.1 + 0.2j)
        ey_field = np.arange(1, 25, dtype=float).reshape(4, 3, 1, 2) + 0.5j
        ez_field = np.zeros((4, 3, 1, 2), dtype=complex)

        class FakeFdtd:
            def getresult(self, path, result_name):
                if result_name == "T":
                    if path.endswith("input_reference"):
                        power = np.asarray([1.0, 1.0])
                    else:
                        power = np.asarray([0.5, 0.5])
                    return {"lambda": wavelengths_m, "T": power}
                if result_name == "E":
                    # Reproduce v261 data that has coordinates but no
                    # separately expanded Ex/Ey/Ez attributes.
                    return {"x": x_m, "y": y_m, "f": frequencies_hz}
                raise AssertionError((path, result_name))

            def getelectric(self, path, option):
                self.last_getelectric = (path, option)
                return electric_intensity

            def getdata(self, path, component_name, option):
                self.getdata_calls.append((path, component_name, option))
                return {"Ex": ex_field, "Ey": ey_field, "Ez": ez_field}[component_name]

        fake_fdtd = FakeFdtd()
        fake_fdtd.getdata_calls = []
        with TemporaryDirectory() as temporary:
            namespace = {
                "np": np,
                "os": __import__("os"),
                "fdtd": fake_fdtd,
                "REMOTE_WORK": temporary,
                "SETTINGS": {
                    "run_after_build": True,
                    "wavelength_start_um": 1.25,
                    "wavelength_stop_um": 1.35,
                },
                "PORT_MODE_SELECTIONS": {
                    "input": {"neff": 1.9},
                    "upper": {"neff": 1.9},
                    "lower": {"neff": 1.9},
                },
                "MMI_ANALYSIS": {
                    "input_port_name": "input",
                    "input_reference_monitor_name": "input_reference",
                    "output_port_names": ["upper", "lower"],
                    "output_labels": ["upper output", "lower output"],
                    "field_monitor_name": "mmi_field",
                    "symmetry_tolerance_percent": 1.0,
                    "port_target_neff": 2.0,
                },
            }
            exec(lumerical._MMI_ANALYSIS_REMOTE, namespace)
            result = np.load(Path(temporary) / "mmi_analysis.npz")
            self.assertEqual(result["field_intensity_normalized"].shape, (4, 3))
            self.assertAlmostEqual(
                float(np.max(result["field_intensity_normalized"])), 1.0
            )
            self.assertEqual(result["field_Ex"].shape, (4, 3))
            self.assertEqual(result["field_Ey"].shape, (4, 3))
            self.assertAlmostEqual(
                float(np.max(result["field_Ey_abs_normalized"])), 1.0
            )
            self.assertEqual(
                fake_fdtd.last_getelectric,
                ("::model::mmi_field", 1),
            )
            self.assertEqual(
                [name for _path, name, _option in fake_fdtd.getdata_calls],
                ["Ex", "Ey", "Ez"],
            )

    def test_air_and_separate_official_fiber_objects_are_available(self) -> None:
        self.assertIn("Air", lumerical.MATERIAL_CHOICES)
        self.assertIn("Fiber geometry", DEFAULT_COMPONENT_VALUES)
        self.assertIn("Fiber-axis FDTD port", DEFAULT_COMPONENT_VALUES)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Fiber-axis FDTD port"]["plane normal"], "Z")
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Fiber geometry"]["core diameter_um"], 9.0)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Fiber geometry"]["cladding diameter_um"], 50.0)
        self.assertEqual(DEFAULT_COMPONENT_VALUES["Fiber geometry"]["z reference"], "top of SiO2 cladding")
        self.assertTrue(DEFAULT_COMPONENT_VALUES["Fiber-axis FDTD port"]["align to fiber axis"])
        window_source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        self.assertIn("fiber_input_power", window_source)

    def test_legacy_combined_fiber_never_creates_a_port(self) -> None:
        notebook, warnings = generate_lumerical_notebook(
            [component("Straight"), component("Fiber port", uid=8)],
            {"included_layers": [[1, 0]]},
        )
        self.assertEqual(assignment_value(notebook, "PORTS"), [])
        self.assertEqual(len(assignment_value(notebook, "FIBER_GEOMETRIES")), 1)
        self.assertTrue(any("legacy combined Fiber port" in warning for warning in warnings))

    def test_only_manually_placed_ports_export_and_do_not_change_gds(self) -> None:
        original = component("Straight")
        legacy = deepcopy(original)
        self.assertEqual(seed_simulation_ports(legacy), [])
        placed_port = component("FDTD port", uid=2)

        before = resolve_and_build([original]).cells[0].get_polygons()
        after = resolve_and_build([original, placed_port]).cells[0].get_polygons()
        self.assertEqual(len(before), len(after))
        for left, right in zip(before, after):
            np.testing.assert_allclose(left.points, right.points)
            self.assertEqual((left.layer, left.datatype), (right.layer, right.datatype))

        notebook, _ = generate_lumerical_notebook(
            [original, placed_port], {"included_layers": [[1, 0]]}
        )
        ports = assignment_value(notebook, "PORTS")
        ports_json = assignment_value(notebook, "PORTS_JSON")
        self.assertEqual(len(ports), 1)
        self.assertEqual(ports[0]["component_kind"], "FDTD port")
        self.assertEqual(set(ports_json["opt_1"]), {"name", "dir", "loc", "pos", "order"})

    def test_grating_uses_only_manually_placed_ansys_style_ports(self) -> None:
        grating = component("Grating coupler")
        power_monitor = component("Power monitor", uid=2)
        power_monitor["x"] = -27.0
        power_monitor["simulation_parent_uid"] = 1
        power_monitor["simulation_parent_port"] = "waveguide_point"
        power_monitor["grating_monitor_role"] = "waveguide_total_power"
        power_monitor["params"].update(
            {
                "name": "waveguide_total_power",
                "plane normal": "X",
                "x span": 0.0,
                "y span": 2.5,
                "z span": 2.25,
            }
        )
        waveguide_mode = component("Mode expansion monitor", uid=4)
        waveguide_mode["x"] = -28.0
        waveguide_mode["orientation_deg"] = 180.0
        waveguide_mode["simulation_parent_uid"] = 1
        waveguide_mode["simulation_parent_port"] = "waveguide_point"
        waveguide_mode["grating_monitor_role"] = "waveguide_mode_expansion"
        waveguide_mode["params"].update(
            {
                "name": "waveguide_mode",
                "plane normal": "X",
                "x span": 0.0,
                "y span": 2.5,
                "z span": 2.25,
                "mode": "user select",
                "target neff": 2.5,
                "neff tolerance": 0.3,
                "mode search count": 4,
                "expansion for": "waveguide_total_power",
                "expansion result name": "waveguide_power",
            }
        )
        fiber = component("Fiber geometry", uid=3)
        fiber["x"] = 20.0
        fiber["params"]["name"] = "fiber"
        fiber["params"]["angle theta"] = 10.0
        fiber_port = component("Fiber-axis FDTD port", uid=5)
        fiber_port["x"] = 20.0
        fiber_port["params"]["name"] = "fiber_out"
        fiber_port["params"]["order"] = 2
        fiber_port["params"]["angle theta"] = 7.0
        fiber_input_power = component("Power monitor", uid=6)
        fiber_input_power.update({"x": 20.0, "simulation_parent_uid": 1, "simulation_parent_port": "fiber_input_power"})
        fiber_input_power["params"].update(
            {
                "name": "fiber_input_power",
                "fiber plane role": "input power measurement",
                "plane normal": "Z",
                "z reference": "top of stack",
                "distance_um": -0.1,
                "x span": 20.0,
                "y span": 20.0,
                "z span": 0.0,
                "angle theta": 10.0,
                "angle phi": 0.0,
                "align to fiber axis": True,
            }
        )
        notebook, _ = generate_lumerical_notebook(
            [grating, power_monitor, waveguide_mode, fiber, fiber_port, fiber_input_power],
            {
                "included_layers": [[1, 0], [2, 0]],
                "include_ports": True,
                "material_stack": [
                    {"name": "BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
                    {"name": "Exported cross-section", "material": "LiNbO3", "thickness_um": 0.6, "etch_depth_um": 0.3, "sidewall_angle_deg": 82.0, "slab_extent": "geometry", "mesh_factor": 0.5, "role": "geometry", "gds_layer": 1},
                    {"name": "SiO2 top cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
                    {"name": "Top air", "material": "Air", "thickness_um": 1.0, "role": "background", "gds_layer": 0},
                    {"name": "Absent metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layer": 4},
                ],
            },
        )
        ports = assignment_value(notebook, "PORTS")
        fiber_ports = [port for port in ports if port.get("plane normal") == "Z"]
        monitors = assignment_value(notebook, "MONITORS")
        fiber_geometries = assignment_value(notebook, "FIBER_GEOMETRIES")
        gaussian_sources = assignment_value(notebook, "GAUSSIAN_SOURCES")
        self.assertEqual(len(ports), 2)
        self.assertFalse(any(port.get("auto_generated_for_grating") for port in ports))
        self.assertEqual(len(fiber_ports), 1)
        self.assertEqual(len(fiber_geometries), 1)
        self.assertEqual(gaussian_sources, [])
        source_port = next(port for port in fiber_ports if port["name"] == "fiber_out")
        input_monitor = next(
            monitor for monitor in monitors
            if monitor["name"] == "fiber_input_power"
        )
        self.assertEqual(source_port["angle theta"], fiber_geometries[0]["angle theta"])
        self.assertAlmostEqual(
            source_port["center"][0] - fiber_geometries[0]["center"][0],
            np.tan(np.deg2rad(10.0)),
        )
        self.assertAlmostEqual(
            input_monitor["center"][0] - fiber_geometries[0]["center"][0],
            0.9 * np.tan(np.deg2rad(10.0)),
        )
        self.assertEqual(input_monitor["fiber plane role"], "input power measurement")
        self.assertEqual(input_monitor["monitor_kind"], "Power monitor")
        self.assertNotIn("mode number", input_monitor)
        self.assertNotIn("polarization", input_monitor)
        self.assertNotIn("candidate mode numbers", input_monitor)
        self.assertAlmostEqual(
            source_port["rotation offset_um"],
            4.0 * fiber_geometries[0]["core diameter_um"] * np.tan(np.deg2rad(10.0)),
        )
        json.dumps(notebook)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<notebook cell {index}>", "exec")
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
        self.assertIn("import lumapi", build_source)
        self.assertIn("addport", build_source)
        self.assertIn('serverArgs={"threads": str(build_cpu_threads)}', build_source)
        self.assertIn('SETTINGS.get("build_cpu_threads", 30)', build_source)
        self.assertIn("addgaussian", build_source)
        self.assertIn("addstructuregroup", build_source)
        self.assertIn("addcircle", build_source)
        self.assertIn('"rotation offset"', build_source)
        self.assertIn('fdtd.set("theta", theta_deg)', build_source)
        self.assertIn('fdtd.set("phi", phi_deg)', build_source)
        self.assertIn('fdtd.set("frequency dependent profile", False)', build_source)
        self.assertNotIn('fdtd.set("number of field profile samples", 1)', build_source)
        self.assertNotIn('fdtd.set("auto update", True)', build_source)
        self.assertIn('fdtd.set("angle theta"', build_source)
        self.assertIn("addlayerbuilder", build_source)
        self.assertIn('fdtd.set("base mesh order", layer_builder_mesh_order)', build_source)
        self.assertIn('fdtd.set("background index", 1.0)', build_source)
        self.assertIn('str(row.get("material", "")).strip().lower() == "air"', build_source)
        self.assertIn('fdtd.adduserprop("core mesh order", 0, 4)', build_source)
        self.assertIn('fdtd.adduserprop("cladding mesh order", 0, 5)', build_source)
        self.assertIn('fiber_setup_script = r"""\ndeleteall;', lumerical._BUILD_CELL)
        self.assertIn('set("mesh order",core_mesh_order);', build_source)
        self.assertIn('set("mesh order",cladding_mesh_order);', build_source)
        self.assertNotIn('set("mesh order",4)', build_source)
        self.assertIn("stack_mesh_summary", build_source)
        self.assertIn(
            '"Material mesh orders: %s; fiber core 4; fiber cladding 5."',
            build_source,
        )
        self.assertIn("background_volumes", build_source)
        self.assertIn('fdtd.set("mesh order", mesh_order)', build_source)
        self.assertIn('addmaterial("Sampled 3D data")', build_source)
        self.assertIn('"sampled 3d data"', build_source)
        self.assertIn("_silica_cladding_top_um", build_source)
        self.assertIn("silica_cladding_top_um, silica_cladding_center_um", build_source)
        self.assertIn("center_z_um = reference_z_um + bottom_gap_um", build_source)
        self.assertNotIn("midpoint_shift_um", build_source)
        self.assertIn("through-going cylinders centered on the nominal contact", build_source)
        self.assertIn("mesh precedence clips the effective fiber material", build_source)
        self.assertNotIn('fdtd.set("rotation 1", theta_deg)', build_source)
        self.assertIn('"sidewall angle"', build_source)
        self.assertIn("z_extent_max_um = max(z_extent_max_um, sampling_z_max_um)", build_source)
        self.assertIn("simulation_z_max_um = z_extent_max_um + requested_z_max_padding", build_source)
        self.assertNotIn("max(boundary_clearance_um, requested_z_max_padding)", build_source)
        self.assertNotIn("boundary_clearance_um = 0.25 * min", build_source)
        self.assertIn("domain_padding = dict(SETTINGS.get", build_source)
        self.assertNotIn("_add_waveguide_boundary_extensions", build_source)
        self.assertNotIn("port PML extension", build_source)
        self.assertNotIn("Extended waveguide at port", build_source)
        self.assertIn('SETTINGS.get("pml_geometry_overlap_um", 1.0)', build_source)
        self.assertIn("_add_layer_mesh_overrides", build_source)
        self.assertIn('fdtd.set("override x mesh", True)', build_source)
        self.assertIn('fdtd.set("dx", mesh_step_um * UM)', build_source)
        self.assertIn("fdtd.getindex(material, frequency_hz)", build_source)
        self.assertIn("mesh_factor * wavelength_min_um / maximum_index", build_source)
        self.assertIn("maximum component is deliberately used for anisotropic media", build_source)
        self.assertIn('slab_extent == "geometry"', build_source)
        self.assertIn("Limited unetched slab", build_source)
        self.assertIn("simulation_z_min_um - pml_geometry_overlap_um", build_source)
        self.assertIn("simulation_z_max_um + pml_geometry_overlap_um", build_source)
        self.assertIn("Added scripted Ansys fiber property group", build_source)
        self.assertIn('fdtd.adduserprop("core diameter", 2', build_source)
        self.assertIn('fdtd.adduserprop("cladding diameter", 2', build_source)
        self.assertIn('fdtd.adduserprop("z span", 2', build_source)
        self.assertIn('fdtd.set("script", fiber_setup_script)', build_source)
        self.assertIn('set("x",0.0);', build_source)
        self.assertIn('set("y",0.0);', build_source)
        self.assertIn('set("z",0.0);', build_source)
        self.assertIn('set("alpha",0.03);', build_source)
        self.assertIn('set("alpha",0.35);', build_source)
        self.assertIn("fdtd.updateportmodes()", build_source)
        self.assertIn("fdtd.updateportmodes(requested_mode_number)", build_source)
        self.assertIn("_select_fiber_local_te_mode", build_source)
        self.assertIn("candidate mode numbers", build_source)
        self.assertIn("selected mode order", build_source)
        self.assertIn("Verified fiber/port concentricity", build_source)
        self.assertIn("is not concentric", build_source)
        self.assertIn("no source or port was created", build_source)
        payload = cell_source_containing(notebook, "MATERIAL_STACK =")
        self.assertIn("'dimension': '3D'", payload)
        self.assertIn("'sidewall_angle_deg': 82.0", payload)
        self.assertIn("'slab_extent': 'geometry'", payload)
        self.assertIn("'mesh_factor': 0.5", payload)
        self.assertIn("'conformal': True", payload)
        self.assertIn("GRATING_ANALYSIS =", payload)
        analysis = assignment_value(notebook, "GRATING_ANALYSIS")
        self.assertEqual(analysis["waveguide_power_monitor_name"], "waveguide_total_power")
        self.assertEqual(analysis["waveguide_port_name"], "uid_1_waveguide_point")
        self.assertEqual(
            analysis["waveguide_port_expansion_result_name"],
            "expansion for port monitor",
        )
        self.assertEqual(analysis["waveguide_target_neff"], 0.0)
        self.assertEqual(analysis["waveguide_port_modal_direction"], "T_out")
        self.assertEqual(analysis["fiber_port_name"], "fiber_out")
        self.assertEqual(
            analysis["fiber_input_power_monitor_name"], "fiber_input_power"
        )
        self.assertEqual(analysis["fiber_input_power_sign"], -1.0)
        self.assertNotIn("fiber_input_measurement_port_name", analysis)
        self.assertNotIn("fiber_measurement_expansion_result_name", analysis)
        self.assertEqual(analysis["fiber_source_mode"], "auto local TE")
        self.assertEqual(analysis["fiber_polarization"], "local TE")
        self.assertEqual(analysis["fiber_mode_candidates"], [1, 2, 3])
        self.assertEqual(analysis["frequency_points"], 31)
        settings = assignment_value(notebook, "SETTINGS")
        self.assertEqual(settings["resource_mode"], "GPU")
        self.assertEqual(settings["dt_stability_factor"], 0.99)
        self.assertEqual(settings["pml_profile"], "Standard")
        self.assertEqual(settings["simulation_time_fs"], 10000.0)
        self.assertEqual(settings["auto_shutoff_min"], 1e-6)
        self.assertEqual(settings["frequency_points"], 31)
        self.assertEqual(settings["build_cpu_threads"], 30)
        self.assertEqual(settings["tfln_crystal_cut"], "X")
        self.assertEqual(settings["tfln_temperature_K"], 296.3)
        run_source = cell_source_containing(notebook, "_solve_code")
        self.assertIn('fdtd.run("FDTD", "GPU")', run_source)
        self.assertIn("REMOTE_SWITCH_TO_CPU_ANALYSIS", run_source)
        self.assertIn('fdtd.setresource("FDTD", 1, "active", False)', run_source)
        self.assertIn('fdtd.setresource("FDTD", 2, "threads", analysis_threads)', run_source)
        resource_source = cell_source_containing(notebook, "REMOTE_RESOURCE_AND_SAVE =")
        self.assertIn('getlicenseestimate("FDTD", "1")', resource_source)
        self.assertIn('fdtd.set("source port", str(GRATING_ANALYSIS["fiber_port_name"]))', resource_source)
        self.assertIn('fdtd.set("source mode", fiber_source_mode)', resource_source)
        self.assertIn('GRATING_ANALYSIS.get("fiber_source_mode", "auto local TE")', resource_source)
        self.assertIn(
            "The fiber source mode was not resolved from its near-degenerate pair",
            resource_source,
        )
        self.assertIn("Backward along the tilted Z-axis fiber port", resource_source)
        self.assertIn("LOCAL_INSPECTION_PROJECT_FILE", resource_source)
        self.assertIn("REMOTE_INSPECTION_FSP_SAVED", resource_source)
        self.assertIn("Saved required pre-solve inspection FSP", resource_source)
        self.assertIn(
            "The required pre-solve inspection FSP was not saved",
            resource_source,
        )
        self.assertIn("lam.fetch(REMOTE_INSPECTION_PROJECT_FILE", resource_source)
        self.assertNotIn("_solve_code", resource_source)
        review_source = cell_source_containing(notebook, "OPEN_REMOTE_LUMERICAL_GUI")
        self.assertIn("FileLink", review_source)
        self.assertIn("Lambda is headless", review_source)
        self.assertIn('fdtd.set("dt stability factor", dt_stability_factor)', build_source)
        self.assertIn('fdtd.set("pml profile", 2 if pml_profile_name == "stabilized" else 1)', build_source)
        self.assertIn('fdtd.set("simulation time", simulation_time_fs * 1e-15)', build_source)
        self.assertIn('fdtd.set("auto shutoff min", auto_shutoff_min)', build_source)
        analysis_source = cell_source_containing(notebook, "REMOTE_GRATING_ANALYSIS")
        self.assertNotIn("farfield3d", analysis_source)
        self.assertNotIn("farfieldux", analysis_source)
        self.assertNotIn("farfielduy", analysis_source)
        self.assertIn("grating_response.png", analysis_source)
        self.assertIn("fiber_input_monitor_name", analysis_source)
        self.assertIn('fdtd.getresult(fiber_input_monitor_name, "T")', analysis_source)
        self.assertIn("fiber_input_sign * fiber_input_signed_raw", analysis_source)
        self.assertIn("waveguide_mode_power / np.maximum", analysis_source)
        self.assertIn("waveguide_total_power / np.maximum", analysis_source)
        self.assertIn("no cos(theta) correction", analysis_source)
        self.assertNotIn("passive tilted fiber port", analysis_source.lower())
        self.assertNotIn("grating_field_distribution.png", analysis_source)
        self.assertNotIn("field_intensity_normalized", analysis_source)
        self.assertIn(
            'waveguide_power_data = fdtd.getresult(waveguide_power_monitor_name, "T")',
            analysis_source,
        )
        self.assertIn("_port_expansion(", analysis_source)
        self.assertIn(
            'GRATING_ANALYSIS.get("waveguide_port_modal_direction", "T_out")',
            analysis_source,
        )
        self.assertIn(
            "fiber_coupling = waveguide_mode_power / np.maximum",
            analysis_source,
        )
        self.assertNotIn("np.abs(scattering) ** 2", analysis_source)
        self.assertIn("fiber_coupling", analysis_source)
        self.assertIn("fiber_coupling_db", analysis_source)
        self.assertIn("10.0 * np.log10", analysis_source)
        self.assertIn("normalized power [dB]", analysis_source)
        self.assertIn("plt.subplots(1, 2", analysis_source)
        self.assertIn("Waveguide transmission — linear", analysis_source)
        self.assertIn("Waveguide transmission — dB", analysis_source)
        self.assertIn("measured input", analysis_source)
        self.assertIn("lam.fetch(_remote_grating_npz", analysis_source)
        self.assertIn("display(Image(filename=str(_local_response_png)", analysis_source)
        self.assertIn('fdtd.setexpansion(result_name, input_monitor_name)', build_source)
        self.assertIn('fdtd.set("mode selection", "fundamental mode")', build_source)
        self.assertIn('status = fdtd.updatemodes()', build_source)
        self.assertNotIn('fdtd.seteigensolver("use max index", 1)', build_source)
        self.assertNotIn('fdtd.seteigensolver("number of trial modes", trial_mode_count)', build_source)
        self.assertIn("fdtd.runsetup()", build_source)
        self.assertIn("Committed geometry in memory", build_source)
        self.assertNotIn('fdtd.save(mode_seed_project)', build_source)
        self.assertIn('geometry_by_layer = _layer_builder_geometry(layer_builder_x_um, layer_builder_y_um)', build_source)
        self.assertIn('(global_vertices_um - local_origin_um) * UM', build_source)
        self.assertNotIn('fdtd.seteigensolver("n", target_neff)', build_source)
        self.assertNotIn('fdtd.updatemodes(mode_numbers)', build_source)
        fetch_source = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        self.assertNotIn('REMOTE_WORK + "/grating_analysis.npz"', fetch_source)
        self.assertNotIn('REMOTE_WORK + "/grating_response.png"', fetch_source)
        self.assertIn("geometry_xyz_projections.png", fetch_source)
        self.assertIn('remote_path.lower().endswith(".fsp")', fetch_source)
        paths_source = cell_source_containing(notebook, "PIRIS_PROJECT_ROOT =")
        self.assertIn('PIRIS_FSP_DIR = PIRIS_PROJECT_ROOT / "fsp"', paths_source)
        self.assertIn("PIRIS_FSP_DIR.mkdir", paths_source)
        self.assertIn('REMOTE_FSP_DIR = os.path.join(REMOTE_WORK, "fsp")', resource_source)
        self.assertIn("REMOTE_PROJECT_FILE = os.path.join(REMOTE_FSP_DIR, _project_name)", resource_source)
        projection_source = cell_source_containing(notebook, "REMOTE_GEOMETRY_PROJECTIONS")
        self.assertIn("XY — top view", projection_source)
        self.assertIn("XZ — side view", projection_source)
        self.assertIn("YZ — end view", projection_source)
        self.assertIn(
            'fdtd.getnamed("FDTD", property_name)',
            lumerical._GEOMETRY_PROJECTIONS_REMOTE,
        )
        self.assertIn(
            '("x min", "y min", "z min", "x max", "y max", "z max")',
            lumerical._GEOMETRY_PROJECTIONS_REMOTE,
        )
        self.assertIn(
            "Run the model-build cell before rendering its geometry",
            lumerical._GEOMETRY_PROJECTIONS_REMOTE,
        )
        connection_source = cell_source_containing(notebook, "def solve_remote_checked")
        self.assertIn("file=_ml_sys.stdout", connection_source)
        self.assertIn("Lumerical solver log:", connection_source)
        self.assertIn("lam.show(GEOMETRY_PROJECTIONS_FILE", projection_source)

    def test_official_gc_fast_settings_preserve_exact_domain_and_symmetry(self) -> None:
        grating = component("GC-SOI")
        notebook, _warnings = generate_lumerical_notebook(
            [grating],
            {
                "included_layers": [[1, 0], [2, 0]],
                "material_stack": lumerical.default_stack("SOI grating coupler (Ansys)"),
                "frequency_points": 31,
                "official_gc_domain": True,
                "use_y_antisymmetry": True,
                "antisymmetry_boundary": "y min",
            },
        )
        settings = assignment_value(notebook, "SETTINGS")
        self.assertEqual(settings["frequency_points"], 31)
        self.assertTrue(settings["official_gc_domain"])
        self.assertTrue(settings["use_y_antisymmetry"])
        self.assertEqual(settings["antisymmetry_boundary"], "y min")
        self.assertEqual(settings["simulation_time_fs"], 10000.0)
        self.assertEqual(settings["auto_shutoff_min"], 1e-6)
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER")
        self.assertIn(
            "simulation_z_min_um = z_extent_min_um - requested_z_min_padding",
            build_source,
        )
        self.assertIn('fdtd.set(antisymmetry_boundary + " bc", "Anti-Symmetric")', build_source)
        solve_source = cell_source_containing(notebook, "_solve_code")
        self.assertNotIn("Re-save solved project", solve_source)
        save_source = cell_source_containing(notebook, "REMOTE_RESULTS_SAVER")
        self.assertIn("Reused the already extracted grating spectrum", save_source)

    def test_manual_fdtd_z_bounds_are_exact_and_reject_crossing_sampling_objects(self) -> None:
        build_source = lumerical._BUILD_CELL
        self.assertIn(
            "simulation_z_min_um = z_extent_min_um - requested_z_min_padding",
            build_source,
        )
        self.assertIn(
            "simulation_z_max_um = z_extent_max_um + requested_z_max_padding",
            build_source,
        )
        self.assertNotIn("boundary_clearance_um", build_source)
        self.assertNotIn("Adjusted FDTD Z clearance", build_source)
        self.assertIn("sampling_z_extents = []", build_source)
        self.assertIn(
            "if sample_min <= simulation_z_min_um or sample_max >= simulation_z_max_um",
            build_source,
        )
        self.assertIn(
            "bounds so every sampling object is strictly inside",
            build_source,
        )

    def test_grating_time_and_auto_shutoff_remain_user_editable(self) -> None:
        grating = component("GC-SOI")
        notebook, _warnings = generate_lumerical_notebook(
            [grating],
            {
                "included_layers": [[1, 0], [2, 0]],
                "simulation_time_fs": 12500.0,
                "auto_shutoff_min": 2e-7,
            },
        )
        settings = assignment_value(notebook, "SETTINGS")
        self.assertEqual(settings["simulation_time_fs"], 12500.0)
        self.assertEqual(settings["auto_shutoff_min"], 2e-7)

    def test_symmetric_mmi_uses_pre_taper_input_reference_and_two_outputs(self) -> None:
        mmi = component("1x2 MMI")
        mmi["params"]["add_grating_couplers"] = False
        mmi["params"]["input_reference_before_taper_um"] = 2.0
        self.assertEqual(mmi["params"]["fdtd_port_clearance_um"], 2.0)

        input_port = component("FDTD port", uid=2)
        input_port["params"].update({"name": "mmi_input", "order": 1, "pos": "Left"})
        input_port["simulation_parent_uid"] = 1
        input_port["simulation_parent_port"] = "left_external"

        upper_port = component("FDTD port", uid=3)
        upper_port["params"].update({"name": "mmi_upper", "order": 2, "pos": "Right"})
        upper_port["simulation_parent_uid"] = 1
        upper_port["simulation_parent_port"] = "upper_right"

        lower_port = component("FDTD port", uid=4)
        lower_port["params"].update({"name": "mmi_lower", "order": 3, "pos": "Right"})
        lower_port["simulation_parent_uid"] = 1
        lower_port["simulation_parent_port"] = "lower_right"

        reference = component("Power monitor", uid=5)
        reference["params"]["name"] = "mmi_input_reference"
        reference["simulation_parent_uid"] = 1
        reference["simulation_parent_port"] = "left_external"

        notebook, warnings = generate_lumerical_notebook(
            [mmi, input_port, upper_port, lower_port, reference],
            {"included_layers": [[1, 0]], "include_ports": True, "run_after_build": True},
        )
        self.assertFalse(any("MMI splitting analysis was not added" in warning for warning in warnings))
        analysis = assignment_value(notebook, "MMI_ANALYSIS")
        self.assertEqual(analysis["input_port_name"], "mmi_input")
        self.assertEqual(analysis["input_reference_monitor_name"], "mmi_input_reference")
        self.assertEqual(analysis["input_reference_before_taper_um"], 2.0)
        self.assertEqual(analysis["output_port_names"], ["mmi_upper", "mmi_lower"])
        self.assertEqual(analysis["ideal_split_percent"], [50.0, 50.0])

        resource_source = cell_source_containing(notebook, "REMOTE_RESOURCE_AND_SAVE")
        self.assertIn('fdtd.set("source port", str(MMI_ANALYSIS["input_port_name"]))', resource_source)
        mmi_source = cell_source_containing(notebook, "REMOTE_MMI_ANALYSIS")
        self.assertIn('input_reference_monitor_name = str(MMI_ANALYSIS["input_reference_monitor_name"])', mmi_source)
        self.assertIn("output_1_over_input", mmi_source)
        self.assertIn("output / measured input (linear)", mmi_source)
        self.assertIn("MMI branch transmission", mmi_source)
        self.assertIn("Secondary symmetry diagnostic", mmi_source)
        self.assertIn("%s/Pin %.3f%%", mmi_source)
        self.assertIn(
            'axes[0].plot(wavelength_m * 1e9, output_1_over_input',
            mmi_source,
        )
        self.assertIn(
            'axes[1].plot(wavelength_m * 1e9, output_1_ratio',
            mmi_source,
        )
        self.assertIn("output_1_split_fraction", mmi_source)
        self.assertIn("symmetry_error_percent", mmi_source)
        self.assertNotIn("imbalance_db", mmi_source)
        self.assertNotIn("np.log10", mmi_source)
        self.assertIn("ideal 50/50", mmi_source)
        self.assertIn("Verified symmetric 50/50 MMI", mmi_source)
        fetch_source = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        self.assertIn("mmi_splitting_ratio.png", fetch_source)
        self.assertIn("mmi_analysis.npz", fetch_source)

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<mmi notebook cell {index}>", "exec")

    def test_standalone_library_ports_and_monitors_export_but_never_enter_gds(self) -> None:
        straight = component("Straight")
        placed = [
            {"uid": 2, "kind": "FDTD port", "x": 0.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["FDTD port"]), "attachment": None},
            {"uid": 3, "kind": "Fiber geometry", "x": 25.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["Fiber geometry"]), "attachment": None},
            {"uid": 7, "kind": "Fiber-axis FDTD port", "x": 25.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["Fiber-axis FDTD port"]), "attachment": None},
            {"uid": 4, "kind": "Power monitor", "x": 50.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["Power monitor"]), "attachment": None},
            {"uid": 5, "kind": "Mode expansion monitor", "x": 45.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["Mode expansion monitor"]), "attachment": None},
            {"uid": 6, "kind": "Field profile monitor", "x": 40.0, "y": 0.0, "orientation_deg": 0.0, "mirrored": False, "params": deepcopy(DEFAULT_COMPONENT_VALUES["Field profile monitor"]), "attachment": None},
        ]
        before = resolve_and_build([straight]).cells[0].get_polygons()
        after = resolve_and_build([straight, *placed]).cells[0].get_polygons()
        self.assertEqual(len(before), len(after))
        for left, right in zip(before, after):
            np.testing.assert_allclose(left.points, right.points)

        notebook, _ = generate_lumerical_notebook(
            [straight, *placed],
            {
                "included_layers": [[1, 0]],
                "material_stack": [
                    {"name": "Device", "material": "Si (Silicon) - Palik", "thickness_um": 0.22, "role": "geometry", "gds_layers": [1]}
                ],
            },
        )
        payload = cell_source_containing(notebook, "MATERIAL_STACK =")
        build = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
        self.assertIn("MONITORS =", payload)
        self.assertIn("power_monitor", payload)
        self.assertIn("addpower", build)
        self.assertIn("addmodeexpansion", build)
        self.assertIn("addprofile", build)
        self.assertIn("addgaussian", build)
        self.assertEqual(assignment_value(notebook, "GAUSSIAN_SOURCES"), [])
        self.assertIn("addstructuregroup", build)
        self.assertIn("addcircle", build)

    def test_surface_monitor_has_one_zero_span_and_horizontal_option(self) -> None:
        horizontal = component("Power monitor", uid=10)
        horizontal["params"]["plane normal"] = "Y"
        horizontal["params"]["x span"] = 6.0
        horizontal["params"]["y span"] = 0.0
        horizontal["params"]["z span"] = 3.0
        notebook, _ = generate_lumerical_notebook(
            [component("Straight"), horizontal],
            {"included_layers": [[1, 0]]},
        )
        payload = cell_source_containing(notebook, "MATERIAL_STACK =")
        build = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
        self.assertIn("'plane normal': 'Y'", payload)
        self.assertIn("'y span': 0.0", payload)
        self.assertIn('f"2D {plane_normal}-normal"', build)

    def test_license_save_fetch_release_lifecycle_matches_reference(self) -> None:
        notebook, _ = generate_lumerical_notebook(
            [component("Straight")],
            {"included_layers": [[1, 0]], "resource_mode": "GPU", "run_after_build": True},
        )
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]

        def index(needle: str) -> int:
            return next(i for i, source in enumerate(sources) if needle in source)

        checkout = index("products checkout")
        build = index("run_remote_checked(_remote_payload + REMOTE_MODEL_BUILDER")
        resource_configuration = index("REMOTE_RESOURCE_AND_SAVE =")
        port_mode_preview = index("REMOTE_PORT_MODE_PROFILES")
        project_review = index("OPEN_REMOTE_LUMERICAL_GUI")
        solve = index("_solve_code")
        save = index("REMOTE_RESULTS_SAVER")
        fetch = index("REMOTE_ARTIFACTS")
        checkin = index("products checkin")
        close = index("lam.close()")
        self.assertLess(checkout, build)
        self.assertLess(build, port_mode_preview)
        self.assertLess(port_mode_preview, resource_configuration)
        self.assertLess(build, resource_configuration)
        self.assertLess(resource_configuration, project_review)
        self.assertLess(project_review, solve)
        self.assertLess(solve, save)
        self.assertLess(save, fetch)
        self.assertLess(fetch, checkin)
        self.assertLessEqual(checkin, close)

        checkout_source = sources[checkout]
        release_source = sources[checkin]
        self.assertIn('max(0, 3 - _existing_count)', checkout_source)
        self.assertIn('--expires "{HPC_PACK_EXPIRY}"', checkout_source)
        self.assertIn('HPC_PACK_EXPIRY = "PT%dM" % HPC_PACK_DURATION_MINUTES', checkout_source)
        self.assertIn('--licenseModel "Shared Web" --mode user', checkout_source)
        self.assertNotIn('--expires "{HPC_PACK_EXPIRY}" --type roaming', checkout_source)
        self.assertIn('--type roaming', release_source)
        self.assertIn('--licenseModel "Shared Web" --mode user', release_source)
        self.assertNotIn('--count 3 --mode user', release_source)
        self.assertIn('json.JSONDecoder()', checkout_source)
        self.assertIn('json.JSONDecoder()', release_source)
        self.assertIn('require_usage=True', release_source)
        self.assertIn("_fdtd_owner.close()", release_source)
        self.assertIn("max_layout_results.npz", cell_source_containing(notebook, "REMOTE_RESULTS_SAVER"))
        self.assertEqual(
            notebook["metadata"]["max_layout"]["license_lifecycle"],
            "shared-web-3-hpc-packs-save-fetch-release",
        )

    def test_first_cell_exposes_run_switches_and_mandatory_fsp_contract(self) -> None:
        notebook, _ = generate_lumerical_notebook(
            [component("Straight")],
            {
                "included_layers": [[1, 0]],
                "run_after_build": True,
                "reuse_model_cache": True,
                "save_model_cache_on_miss": False,
                "save_inspection_fsp": False,
                "save_final_fsp": False,
            },
        )
        first_source = "".join(notebook["cells"][0]["source"])
        self.assertTrue(first_source.startswith("# ==="))
        self.assertIn("# QUICK RUN OPTIONS", first_source)
        self.assertIn("RUN_SIMULATION = True", first_source)
        self.assertIn(
            "One pre-solve inspection FSP and one solved/best FSP are always stored.",
            first_source,
        )
        self.assertIn(
            "Project-file saving is always enabled: inspection plus solved/best FSP.",
            first_source,
        )
        self.assertIn("SHOW_GEOMETRY_PREVIEW = True", first_source)
        self.assertIn("SHOW_PORT_MODE_PREVIEW = True", first_source)
        self.assertIn("RUN_GPU_SYSTEM_CHECK = False", first_source)
        self.assertIn("HPC_PACK_DURATION_MINUTES = 30", first_source)

        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for removed_name in (
            "REUSE_EXACT_MODEL_CACHE",
            "SAVE_EXACT_MODEL_CACHE_ON_MISS",
            "MODEL_CACHE_KEY",
            "MODEL_CACHE_HIT",
            "REMOTE_MODEL_CACHE_FSP",
            "_save_model_cache_on_miss",
        ):
            self.assertNotIn(removed_name, all_source)
        self.assertIn("SETTINGS['save_inspection_fsp'] = True", all_source)
        self.assertIn("SETTINGS['save_final_fsp'] = True", all_source)
        self.assertIn("save_verified_project(REMOTE_INSPECTION_PROJECT_FILE)", all_source)
        self.assertIn("save_verified_project(REMOTE_PROJECT_FILE)", all_source)
        self.assertIn("REMOTE_INSPECTION_FSP_SAVED = True", all_source)
        self.assertIn("REMOTE_FINAL_FSP_SAVED = True", all_source)
        self.assertIn("lam.fetch(REMOTE_INSPECTION_PROJECT_FILE", all_source)
        self.assertIn("lam.fetch(REMOTE_PROJECT_FILE", all_source)
        self.assertNotIn("REMOTE_ARTIFACTS.insert(0, REMOTE_PROJECT_FILE)", all_source)
        self.assertIn("if SHOW_GEOMETRY_PREVIEW:", all_source)
        self.assertIn("if SHOW_PORT_MODE_PREVIEW:", all_source)
        self.assertIn('if bool(SETTINGS.get("run_gpu_system_check", False)):', all_source)

    def test_default_path_builds_once_without_a_reusable_model_cache(self) -> None:
        notebook, _ = generate_lumerical_notebook(
            [component("Straight")],
            {"included_layers": [[1, 0]], "run_after_build": True},
        )
        first_source = "".join(notebook["cells"][0]["source"])
        self.assertNotIn("REUSE_EXACT_MODEL_CACHE", first_source)
        self.assertNotIn("SAVE_EXACT_MODEL_CACHE_ON_MISS", first_source)
        self.assertIn("SHOW_GEOMETRY_PREVIEW = True", first_source)
        self.assertIn("SHOW_PORT_MODE_PREVIEW = True", first_source)
        self.assertIn("RUN_GPU_SYSTEM_CHECK = False", first_source)
        self.assertIn("HPC_PACK_DURATION_MINUTES = 30", first_source)
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
        self.assertNotIn("_save_model_cache_on_miss", build_source)
        self.assertNotIn("MODEL_CACHE", build_source)
        self.assertEqual(build_source.count("lumapi.FDTD("), 1)
        self.assertEqual(lumerical._BUILD_CELL.count("fdtd.runsetup()"), 1)
        self.assertIn("Closed the previous live FDTD model", lumerical._BUILD_CELL)

    def test_legacy_cache_and_save_flags_are_ignored(self) -> None:
        base_configuration = {
            "included_layers": [[1, 0]],
            "wavelength_start_um": 1.25,
            "wavelength_stop_um": 1.35,
            "mesh_accuracy": 2,
            "resource_mode": "GPU",
            "run_after_build": True,
            "project_file": "one.fsp",
        }
        legacy_disabled = {
            **base_configuration,
            "reuse_model_cache": True,
            "save_model_cache_on_miss": False,
            "save_inspection_fsp": False,
            "save_final_fsp": False,
        }
        legacy_enabled = {
            **base_configuration,
            "reuse_model_cache": False,
            "save_model_cache_on_miss": True,
            "save_inspection_fsp": True,
            "save_final_fsp": True,
        }
        disabled_notebook, _ = generate_lumerical_notebook(
            [component("Straight")], legacy_disabled
        )
        enabled_notebook, _ = generate_lumerical_notebook(
            [component("Straight")], legacy_enabled
        )
        self.assertEqual(disabled_notebook, enabled_notebook)
        settings = assignment_value(disabled_notebook, "SETTINGS")
        self.assertTrue(settings["save_inspection_fsp"])
        self.assertTrue(settings["save_final_fsp"])
        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in disabled_notebook["cells"]
        )
        for removed_name in (
            "MODEL_CACHE_KEY",
            "MODEL_CACHE_HIT",
            "REMOTE_MODEL_CACHE_FSP",
            "REUSE_EXACT_MODEL_CACHE",
            "SAVE_EXACT_MODEL_CACHE_ON_MISS",
        ):
            self.assertNotIn(removed_name, all_source)

    def test_every_component_export_is_3d_and_gets_three_axis_verification(self) -> None:
        for kind in ("Straight", "Taper", "1x2 MMI", "CPW", "Tapered CPW"):
            with self.subTest(kind=kind):
                notebook, _ = generate_lumerical_notebook(
                    [component(kind)],
                    {"dimension": "2D", "included_layers": [[1, 0], [2, 0], [10, 0]]},
                )
                payload = cell_source_containing(notebook, "MATERIAL_STACK =")
                builder = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
                projections = cell_source_containing(notebook, "REMOTE_GEOMETRY_PROJECTIONS =")
                self.assertIn("'dimension': '3D'", payload)
                self.assertIn("'resource_mode': 'GPU'", payload)
                self.assertNotIn("'dimension': '2D'", payload)
                self.assertEqual(assignment_value(notebook, "PORTS"), [])
                self.assertIn('fdtd.set("dimension", "3D")', builder)
                self.assertIn("geometry_xyz_projections.png", projections)
                self.assertEqual(notebook["metadata"]["max_layout"]["dimension"], "3D")

    def test_port_modes_are_visualized_and_checked_before_solve(self) -> None:
        grating = component("GC-SOI", uid=1)
        fiber_port = component("Fiber-axis FDTD port", uid=2)
        fiber_port["params"].update({
            "name": "fiber_source",
            "mode": "user select",
            "mode number": 2,
            "polarization": "Ey",
        })
        waveguide_port = component("FDTD port", uid=3)
        waveguide_port["params"]["name"] = "waveguide_receiver"
        notebook, _ = generate_lumerical_notebook(
            [grating, fiber_port, waveguide_port],
            {"included_layers": [[1, 0]], "run_after_build": True},
        )
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        preview_index = next(i for i, source in enumerate(sources) if "REMOTE_PORT_MODE_PROFILES" in source)
        solve_index = next(i for i, source in enumerate(sources) if "_solve_code" in source)
        preview = sources[preview_index]
        self.assertLess(preview_index, solve_index)
        self.assertIn('getresult(result_path, "mode profiles")', preview)
        self.assertIn("electric = np.asarray(mode_profile[candidate])", preview)
        self.assertNotIn('if "E" in mode_profile', preview)
        self.assertIn('vector_candidates.append("E%d" % int(preferred_mode_number))', preview)
        self.assertIn("key[0] == \"E\" and key[1:].isdigit()", preview)
        self.assertIn("preferred_mode_number = max(0, int(profile_object.get(\"mode number\", 0)))", preview)
        self.assertIn('"|Ex|"', preview)
        self.assertIn('"|Ey|"', preview)
        self.assertIn("PORT_POLARIZATION_VALID", preview)
        self.assertIn("PORT_MODE_CONFINEMENT_VALID", preview)
        self.assertIn("edge_fraction > 0.05", preview)
        self.assertIn("PORT_MODE_VALID", preview)
        self.assertIn("not Ey-dominant", preview)
        self.assertIn('str(GRATING_ANALYSIS["waveguide_port_name"])', preview)
        self.assertIn('"Waveguide receiver fundamental mode"', preview)
        self.assertIn('PORT_MODE_SELECTIONS.get(object_name, {})', preview)
        self.assertIn("lam.show(PORT_MODE_PROFILES_FILE", preview)
        fetch = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        self.assertIn("port_mode_Ex_Ey.png", fetch)

    def test_embedded_remote_programs_compile(self) -> None:
        for name in (
            "_BUILD_CELL",
            "_GEOMETRY_PROJECTIONS_REMOTE",
            "_PORT_MODE_PROFILES_REMOTE",
            "_REMOTE_RESOURCE_AND_SAVE",
            "_SAVE_REMOTE_RESULTS",
            "_GRATING_ANALYSIS_REMOTE",
        ):
            with self.subTest(name=name):
                compile(getattr(lumerical, name), "<%s>" % name, "exec")


if __name__ == "__main__":
    unittest.main()
