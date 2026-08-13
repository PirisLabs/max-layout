from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.lumerical_rf import (
    build_lumerical_rf_preview_state,
    write_lumerical_rf_notebook,
)
from max_layout.rf_defaults import default_rf_configuration


def component(kind: str, uid: int = 1, x: float = 0.0, y: float = 0.0) -> dict:
    return {
        "uid": uid,
        "kind": kind,
        "x": x,
        "y": y,
        "orientation_deg": 0.0,
        "mirrored": False,
        "params": deepcopy(DEFAULT_COMPONENT_VALUES[kind]),
        "attachment": None,
    }


def write_notebook(components: list[dict], configuration: dict) -> tuple[dict, list[str]]:
    with TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "rf_export.ipynb"
        warnings = write_lumerical_rf_notebook(path, components, configuration)
        return json.loads(path.read_text(encoding="utf-8")), warnings


def notebook_sources(notebook: dict) -> list[str]:
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def complete_source(notebook: dict) -> str:
    return "\n".join(notebook_sources(notebook))


def source_index(sources: list[str], needle: str) -> int:
    try:
        return next(index for index, source in enumerate(sources) if needle in source)
    except StopIteration as exc:
        raise AssertionError(f"No notebook cell contains {needle!r}") from exc


def assignment_value(notebook: dict, variable_name: str):
    for source in notebook_sources(notebook):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment found for {variable_name!r}")


class LumericalRFExportTests(unittest.TestCase):
    def test_rf_material_stack_editor_matches_photonic_readability_contract(self) -> None:
        source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        dialog_start = source.index("class RFLumericalExportDialog")
        dialog_end = source.index("class NativeLayoutWindow", dialog_start)
        dialog_source = source[dialog_start:dialog_end]

        self.assertIn("self.resize(1420, 860)", dialog_source)
        self.assertIn("self.setMinimumSize(1180, 720)", dialog_source)
        self.assertIn("self.material_table.setMinimumHeight(430)", dialog_source)
        self.assertIn(
            "self.material_table.verticalHeader().setDefaultSectionSize(48)",
            dialog_source,
        )
        self.assertIn(
            "self.material_table.horizontalHeader().setMinimumSectionSize(120)",
            dialog_source,
        )
        self.assertIn(
            "(205, 215, 165, 120, 120, 120, 120, 155, 205, 145, 185)",
            dialog_source,
        )
        self.assertIn("QHeaderView.ResizeMode.Interactive", dialog_source)
        self.assertIn("role.setMinimumSize(190, 38)", dialog_source)
        self.assertIn("metal_model.setMinimumSize(165, 38)", dialog_source)

    def test_rf_dialog_exposes_photonic_style_interactive_3d_preview(self) -> None:
        source = Path("src/max_layout/ui/window.py").read_text(encoding="utf-8")
        dialog_start = source.index("class RFLumericalExportDialog")
        dialog_end = source.index("class NativeLayoutWindow", dialog_start)
        dialog_source = source[dialog_start:dialog_end]

        self.assertIn('self.tabs.addTab(page, "RF 3D preview")', dialog_source)
        self.assertIn("Show me a 3D version of the RF file I have built", dialog_source)
        self.assertIn("build_lumerical_rf_preview_state", dialog_source)
        self.assertIn("ThreeDModelPreview(self)", dialog_source)
        self.assertIn('("RF material stack", "show_stack")', dialog_source)
        self.assertIn('("RF ports & monitors", "show_ports")', dialog_source)
        self.assertIn('("Solver region", "show_fdtd")', dialog_source)
        self.assertIn("preview.hidden_stack_rows", dialog_source)

    def test_straight_cpw_preview_matches_mode_cross_section_stack(self) -> None:
        configuration = default_rf_configuration("CPW")
        configuration.update(
            {
                "scope_uids": [1],
                "primary_component_uid": 1,
            }
        )
        state = build_lumerical_rf_preview_state(
            [component("CPW")], configuration
        )

        self.assertEqual(state["workflow"], "fde")
        self.assertIn("MODE/FDE 2D cross-section", state["solver_label"])
        self.assertEqual(len(state["polygons"]), 3)
        self.assertEqual(state["components"], [])
        self.assertEqual(state["solver_bounds_um"][2:4], (-0.5, 0.5))
        self.assertEqual(state["padding"], {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
        })
        metal_rows = [
            (row, z0, z1)
            for row, z0, z1 in state["stack_ranges"]
            if row.get("role") == "geometry"
        ]
        self.assertEqual(len(metal_rows), 1)
        metal, z0, z1 = metal_rows[0]
        self.assertEqual(metal["material"], "RF metal")
        self.assertEqual(z0, 0.0)
        self.assertGreater(z1, z0)

    def test_3d_rf_preview_uses_manual_planes_and_frequency_domain_clearance(self) -> None:
        taper = component("Tapered CPW", uid=1)
        source_port = component("RF mode port", uid=2, x=20.0)
        source_port["params"].update(
            {"name": "manual_rf_source", "rf role": "Source", "span_um": 400.0}
        )
        output_monitor = component("RF power monitor", uid=3, x=480.0)
        output_monitor["params"].update(
            {"name": "manual_rf_output", "rf role": "Output", "span_um": 400.0}
        )
        configuration = default_rf_configuration("Tapered CPW")
        configuration.update(
            {
                "scope_uids": [1, 2, 3],
                "primary_component_uid": 1,
                "manual_rf_object_uids": [2, 3],
                "rf_port_strategy": "manual_only",
                "use_endpoint_reference_planes": False,
                "frequency_stop_ghz": 40.0,
                "mesh_bulk_um": 5.0,
                "port_clearance_wavelengths": 0.25,
            }
        )
        state = build_lumerical_rf_preview_state(
            [taper, source_port, output_monitor], configuration
        )

        self.assertEqual(state["workflow"], "fdtd")
        self.assertIn("3D RF FDTD", state["solver_label"])
        self.assertEqual(
            {plane["kind"] for plane in state["components"]},
            {"RF mode port", "RF power monitor"},
        )
        names = {plane["params"]["name"] for plane in state["components"]}
        self.assertEqual(
            names,
            {
                "manual_rf_source",
                "manual_rf_source_reference",
                "manual_rf_output",
            },
        )
        input_reference = next(
            plane
            for plane in state["components"]
            if plane["params"]["name"] == "manual_rf_source_reference"
        )
        self.assertEqual(input_reference["kind"], "RF power monitor")
        # At 40 GHz the requested quarter-wavelength clearance is about
        # 1873.7 µm, larger than the 25 µm five-cell minimum.
        expected_clearance_um = 0.25 * 299792458.0 / 40.0e9 / 1.0e-6
        geometry_x_min = min(float(points[:, 0].min()) for points, _layer in state["polygons"])
        geometry_x_max = max(float(points[:, 0].max()) for points, _layer in state["polygons"])
        bounds = state["solver_bounds_um"]
        self.assertAlmostEqual(geometry_x_min - bounds[0], expected_clearance_um)
        self.assertAlmostEqual(bounds[1] - geometry_x_max, expected_clearance_um)

    def test_rf_preview_includes_backing_ground_and_rejects_mixed_solvers(self) -> None:
        configuration = default_rf_configuration("CPW")
        configuration.update(
            {
                "scope_uids": [1],
                "primary_component_uid": 1,
                "backing_ground": True,
            }
        )
        state = build_lumerical_rf_preview_state(
            [component("CPW")], configuration
        )
        self.assertEqual(len(state["polygons"]), 4)
        backing_rows = [
            (row, z0, z1)
            for row, z0, z1 in state["stack_ranges"]
            if row.get("name") == "Backing ground"
        ]
        self.assertEqual(len(backing_rows), 1)
        backing_row, backing_z0, backing_z1 = backing_rows[0]
        self.assertEqual(backing_row["role"], "geometry")
        self.assertEqual(backing_row["material"], "RF metal")
        self.assertGreater(backing_z1, backing_z0)
        self.assertEqual(backing_z0, state["solver_bounds_um"][4])

        mixed_configuration = default_rf_configuration("CPW")
        mixed_configuration.update(
            {
                "scope_uids": [1, 2],
                "primary_component_uid": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "separately"):
            build_lumerical_rf_preview_state(
                [component("CPW", uid=1), component("Tapered CPW", uid=2)],
                mixed_configuration,
            )

    def test_rf_metal_row_filters_nonmetal_gds_layers_in_preview_and_notebook(self) -> None:
        electrode = component("Segmented electrode", uid=1)
        electrode["params"].update(
            {
                "include_oxide_masks": True,
                "layer": 4,
                "oxide_layer": 3,
            }
        )
        configuration = default_rf_configuration("Segmented electrode")
        configuration.update(
            {
                "scope_uids": [1],
                "primary_component_uid": 1,
                "rf_port_strategy": "component_endpoints",
                "use_endpoint_reference_planes": True,
            }
        )
        metal_row = next(
            row
            for row in configuration["material_stack"]
            if row["role"] == "metal"
        )
        metal_row["gds_layers"] = [4]

        state = build_lumerical_rf_preview_state([electrode], configuration)
        self.assertEqual({layer for _points, layer in state["polygons"]}, {4})

        notebook, _warnings = write_notebook([electrode], configuration)
        geometry = assignment_value(notebook, "GEOMETRY")
        self.assertTrue(geometry)
        self.assertEqual({item["layer"] for item in geometry}, {4})

        metal_row["gds_layers"] = [99]
        with self.assertRaisesRegex(ValueError, "GDS layer"):
            build_lumerical_rf_preview_state([electrode], configuration)

    def test_straight_cpw_uses_official_mode_fde_cpu_workflow(self) -> None:
        configuration = default_rf_configuration("CPW")
        configuration.update(
            {
                "frequency_start_ghz": 12.5,
                "frequency_stop_ghz": 67.5,
                "frequency_points": 7,
                "target_frequency_ghz": 30.0,
                "run_after_build": True,
                "save_inspection_fsp": True,
                "save_final_fsp": True,
                "project_file": "straight_cpw.lms",
            }
        )
        notebook, warnings = write_notebook([component("CPW")], configuration)
        source = complete_source(notebook)

        self.assertEqual(warnings, [])
        self.assertEqual(notebook["metadata"]["max_layout"]["workflow"], "fde")
        self.assertEqual(
            notebook["metadata"]["max_layout"]["dimension"], "2D Z-normal"
        )
        self.assertIn("lumapi.MODE(", source)
        self.assertIn("addfde", source.lower())
        self.assertIn("2d z normal", source.lower())
        self.assertIn("CPU", source)
        self.assertNotIn("lumapi.FDTD(", source)
        self.assertIn("12.5", source)
        self.assertIn("67.5", source)
        self.assertIn("1e9", source.lower().replace("+", ""))

        # The official CPW cross-section workflow reports electrical modal
        # quantities, not an optical transmission spectrum.
        for result_name in ("Z0", "neff", "ng", "loss"):
            self.assertIn(result_name, source)
        self.assertTrue("findmodes" in source.lower() or "run(\"FDE\"" in source)
        self.assertTrue("mode profile" in source.lower() or "field" in source.lower())

    def test_cpw_taper_uses_3d_gpu_s_parameters_and_manual_rf_planes(self) -> None:
        taper = component("Tapered CPW", uid=1)
        source_port = component("RF mode port", uid=2, x=20.0)
        source_port["params"].update(
            {"name": "manual_rf_source", "rf role": "Source", "order": 1}
        )
        output_monitor = component("RF power monitor", uid=3, x=480.0)
        output_monitor["params"].update(
            {
                "name": "manual_rf_output",
                "rf role": "Output",
                "expansion port": "manual_rf_source",
            }
        )
        configuration = default_rf_configuration("Tapered CPW")
        configuration.update(
            {
                "frequency_start_ghz": 2.0,
                "frequency_stop_ghz": 42.0,
                "frequency_points": 11,
                "target_frequency_ghz": 20.0,
                "simulation_time_ns": 17.0,
                "rf_port_strategy": "manual_only",
                "use_endpoint_reference_planes": False,
                "run_after_build": True,
                "save_inspection_fsp": True,
                "save_final_fsp": True,
                "project_file": "cpw_taper.fsp",
            }
        )
        notebook, warnings = write_notebook(
            [taper, source_port, output_monitor], configuration
        )
        source = complete_source(notebook)
        lower = source.lower()

        self.assertEqual(warnings, [])
        self.assertEqual(notebook["metadata"]["max_layout"]["workflow"], "fdtd")
        self.assertEqual(notebook["metadata"]["max_layout"]["dimension"], "3D")
        self.assertIn("lumapi.FDTD(", source)
        self.assertIn("GPU", source)
        self.assertIn("3d", lower)
        self.assertIn("17.0", source)
        self.assertIn("1e-9", lower)
        self.assertIn("manual_rf_source", source)
        self.assertIn("manual_rf_output", source)
        self.assertIn("manual_only", source)
        self.assertIn("use_endpoint_reference_planes", lower)

        # A discontinuity run must use modal reference planes and extract
        # complex microwave scattering data.
        self.assertIn("addmode", lower)
        self.assertTrue("addpower" in lower or "adddftmonitor" in lower)
        self.assertIn("modeexpansion", lower)
        for result_name in ("S11", "S21", "phase"):
            self.assertIn(result_name, source)
        self.assertTrue("np.angle" in source or "angle(" in lower)
        self.assertIn("pml", lower)

    def test_rf_notebooks_do_not_reintroduce_optical_defaults_or_auto_ports(self) -> None:
        configurations = [
            ("CPW", [component("CPW")]),
            (
                "Tapered CPW",
                [
                    component("Tapered CPW"),
                    component("RF mode port", uid=2, x=10.0),
                    component("RF power monitor", uid=3, x=490.0),
                ],
            ),
        ]
        for kind, components in configurations:
            with self.subTest(kind=kind):
                configuration = default_rf_configuration(kind)
                if kind != "CPW":
                    configuration.update(
                        {
                            "rf_port_strategy": "manual_only",
                            "use_endpoint_reference_planes": False,
                        }
                    )
                notebook, _ = write_notebook(components, configuration)
                source = complete_source(notebook)
                lower = source.lower()
                self.assertNotIn("Fiber-axis FDTD port", source)
                self.assertNotIn("fiber geometry", lower)
                self.assertNotIn("seed_simulation_ports", source)
                self.assertNotIn("wavelength_start_um", source)
                self.assertNotIn("wavelength_stop_um", source)
                self.assertNotIn("1.25e-6", lower)
                self.assertNotIn("1.35e-6", lower)

    def test_license_save_fetch_release_and_rf_artifact_contract(self) -> None:
        for kind, extension in (("CPW", ".lms"), ("Tapered CPW", ".fsp")):
            with self.subTest(kind=kind):
                components = [component(kind)]
                configuration = default_rf_configuration(kind)
                configuration.update(
                    {
                        "save_inspection_fsp": True,
                        "save_final_fsp": True,
                        "project_file": f"rf_project{extension}",
                    }
                )
                if kind != "CPW":
                    rf_source = component("RF mode port", uid=2, x=10.0)
                    rf_source["params"].update(
                        {"name": "rf_input", "rf role": "Source"}
                    )
                    rf_output = component("RF power monitor", uid=3, x=490.0)
                    rf_output["params"].update(
                        {"name": "rf_output", "rf role": "Output"}
                    )
                    components.extend([rf_source, rf_output])
                    configuration.update(
                        {
                            "rf_port_strategy": "manual_only",
                            "use_endpoint_reference_planes": False,
                        }
                    )
                notebook, _ = write_notebook(components, configuration)
                sources = notebook_sources(notebook)
                source = "\n".join(sources)

                checkout = source_index(sources, "products checkout")
                build = source_index(sources, "lumapi.")
                save = source_index(sources, "rf_results.npz")
                fetch = source_index(sources, "lam.fetch")
                release = source_index(sources, "products checkin")
                close = source_index(sources, "lam.close()")
                self.assertLess(checkout, build)
                self.assertLess(build, save)
                self.assertLess(save, fetch)
                self.assertLess(fetch, release)
                self.assertLessEqual(release, close)

                self.assertIn('max(0, HPC_PACK_COUNT - _existing_count)', sources[checkout])
                self.assertIn('--expires "{HPC_PACK_EXPIRY}"', sources[checkout])
                self.assertIn("HPC_PACK_DURATION_MINUTES = 30", sources[0])
                self.assertIn("HPC_PACK_COUNT = 3", sources[0])
                self.assertIn('--licenseModel "Shared Web" --mode user', sources[checkout])
                self.assertIn("--type roaming", sources[release])
                self.assertIn('--licenseModel "Shared Web" --mode user', sources[release])
                self.assertNotIn("--count 3 --mode user", sources[release])
                self.assertIn("json.JSONDecoder()", sources[release])
                self.assertIn("require_usage=True", sources[release])
                self.assertIn("close()", sources[release])
                self.assertIn("rf_results.npz", source)
                self.assertIn("rf_results.json", source)
                self.assertIn("summary.txt", source)
                self.assertIn(extension, source.lower())
                self.assertIn("fsp", source.lower())
                for heading in (
                    "GEOMETRY",
                    "RF MATERIAL STACK",
                    "SIMULATION SETTINGS",
                    "RESULTS SUMMARY",
                ):
                    self.assertIn(heading, source)

    def test_every_generated_rf_notebook_code_cell_is_valid_python(self) -> None:
        cases = []
        cpw_configuration = default_rf_configuration("CPW")
        cases.append(([component("CPW")], cpw_configuration))

        taper_configuration = default_rf_configuration("Tapered CPW")
        taper_configuration.update(
            {
                "rf_port_strategy": "manual_only",
                "use_endpoint_reference_planes": False,
            }
        )
        source_port = component("RF mode port", uid=2, x=10.0)
        source_port["params"].update({"name": "rf_input", "rf role": "Source"})
        output_monitor = component("RF power monitor", uid=3, x=490.0)
        output_monitor["params"].update({"name": "rf_output", "rf role": "Output"})
        cases.append(
            ([component("Tapered CPW"), source_port, output_monitor], taper_configuration)
        )

        for components, configuration in cases:
            notebook, _ = write_notebook(components, configuration)
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                with self.subTest(kind=components[0]["kind"], cell=index):
                    ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
