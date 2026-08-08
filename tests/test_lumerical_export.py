from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.gds.build import component_geometry_arrays, resolve_and_build
from max_layout.gds.couplers import resolve_grating_fill_factors
from max_layout import lumerical
from max_layout.lumerical import generate_lumerical_notebook, seed_simulation_ports
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


class LumericalExportTests(unittest.TestCase):
    def test_grating_platforms_have_distinct_neff_validation_defaults(self) -> None:
        self.assertEqual(component("Grating coupler")["params"]["waveguide_effective_index"], 2.0)
        self.assertEqual(component("GC-SOI")["params"]["waveguide_effective_index"], 2.5)

    def test_official_soi_grating_component_and_stack_defaults(self) -> None:
        grating = component("GC-SOI")
        params = grating["params"]
        self.assertEqual(params["pitch"], 0.6713)
        self.assertEqual(params["duty_cycle"], 0.3992)
        self.assertEqual(params["target_length"], 25.0)
        self.assertEqual(params["h_total"], 0.22)
        self.assertEqual(params["etch_depth"], 0.10)
        self.assertEqual(params["fiber_tilt_deg"], 10.0)
        self.assertEqual(params["fiber_x_from_grating_start_um"], 2.74533)
        self.assertEqual(params["tolerance"], 0.005)
        self.assertEqual(params["fdtd_port_offset_from_waveguide_end_um"], 2.0)
        self.assertEqual(params["waveguide_monitor_span_um"], 2.5)
        self.assertEqual(params["waveguide_total_power_before_mode_um"], 1.0)
        self.assertEqual(params["waveguide_effective_index"], 2.5)
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
        self.assertEqual(orders["fiber_input_power"], 2)
        self.assertNotIn("waveguide_point", orders)
        diagnostic_port = next(
            item for item in companions
            if item["kind"] == "Fiber-axis FDTD port"
            and item.get("simulation_parent_port") == "fiber_input_power"
        )
        self.assertEqual(diagnostic_port["params"]["plane normal"], "Z")
        self.assertTrue(diagnostic_port["params"]["align to fiber axis"])
        self.assertEqual(diagnostic_port["params"]["fiber plane role"], "input power measurement")
        self.assertEqual(diagnostic_port["params"]["mode number"], 2)
        self.assertEqual(diagnostic_port["params"]["polarization"], "Ey")
        waveguide_power = next(
            item for item in companions
            if item["kind"] == "Power monitor"
            and item.get("grating_monitor_role") == "waveguide_total_power"
        )
        waveguide_mode = next(
            item for item in companions
            if item["kind"] == "Mode expansion monitor"
            and item.get("grating_monitor_role") == "waveguide_mode_expansion"
        )
        self.assertEqual(waveguide_power["params"]["y span"], 2.5)
        self.assertEqual(waveguide_mode["params"]["y span"], 2.5)
        self.assertEqual(waveguide_mode["params"]["mode"], "fundamental mode")
        self.assertEqual(waveguide_mode["params"]["target neff"], 2.5)
        self.assertEqual(waveguide_mode["params"]["mode search count"], 20)
        self.assertEqual(waveguide_power["x"] - waveguide_mode["x"], 1.0)
        fiber_source_component = next(
            item for item in companions
            if item["kind"] == "Fiber-axis FDTD port"
            and item.get("simulation_parent_port") != "fiber_input_power"
        )
        self.assertEqual(fiber_source_component["params"]["mode"], "user select")
        self.assertEqual(fiber_source_component["params"]["mode number"], 2)
        self.assertEqual(fiber_source_component["params"]["polarization"], "Ey")
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
        measurement = next(port for port in ports if port["name"].endswith("fiber_input_power"))
        self.assertEqual(len(ports), 2)
        self.assertEqual(source["name"], "uid_1_fiber_axis")
        self.assertEqual(measurement["fiber plane role"], "input power measurement")
        self.assertEqual(measurement["dir"], "Backward")
        self.assertFalse(any(monitor["name"].endswith("fiber_input_power") for monitor in monitors))
        self.assertTrue(any(monitor["name"].endswith("waveguide_total_power") for monitor in monitors))
        self.assertTrue(any(monitor["name"].endswith("waveguide_mode") for monitor in monitors))
        self.assertEqual(fiber["z reference"], "center of SiO2 cladding")
        self.assertAlmostEqual(
            source["center"][0] - fiber["center"][0],
            0.65 * np.sin(np.deg2rad(10.0)),
        )
        self.assertAlmostEqual(
            measurement["center"][0] - fiber["center"][0],
            (0.65 * np.cos(np.deg2rad(10.0)) - 0.1) * np.tan(np.deg2rad(10.0)),
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
        self.assertEqual(analysis["field_monitor_name"], "uid_1_mmi_field")
        monitors = assignment_value(notebook, "MONITORS")
        field_monitor = next(monitor for monitor in monitors if monitor["name"] == "uid_1_mmi_field")
        self.assertEqual(field_monitor["plane normal"], "Z")
        self.assertEqual(field_monitor["z reference"], "device center")
        self.assertTrue(any("longitudinal field-profile monitor" in warning for warning in warnings))
        source = cell_source_containing(notebook, "REMOTE_MMI_ANALYSIS =")
        self.assertIn("mmi_field_distribution.png", source)
        self.assertIn("field_intensity_normalized", source)

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
        self.assertEqual(len(ports), 2)
        self.assertFalse(any(port.get("auto_generated_for_grating") for port in ports))
        self.assertEqual(len(fiber_ports), 2)
        self.assertEqual(len(fiber_geometries), 1)
        source_port = next(port for port in fiber_ports if port["name"] == "fiber_out")
        input_port = next(port for port in fiber_ports if port["name"] == "fiber_input_power")
        self.assertEqual(source_port["angle theta"], fiber_geometries[0]["angle theta"])
        self.assertAlmostEqual(
            source_port["center"][0] - fiber_geometries[0]["center"][0],
            np.tan(np.deg2rad(10.0)),
        )
        self.assertAlmostEqual(
            input_port["center"][0] - fiber_geometries[0]["center"][0],
            0.9 * np.tan(np.deg2rad(10.0)),
        )
        self.assertEqual(input_port["fiber plane role"], "passive fiber measurement")
        self.assertEqual(input_port["mode number"], 2)
        self.assertEqual(input_port["polarization"], "Ey")
        self.assertFalse(any(monitor["name"] == "fiber_input_power" for monitor in monitors))
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
        self.assertNotIn("addgaussian", build_source)
        self.assertIn("addstructuregroup", build_source)
        self.assertIn("addcircle", build_source)
        self.assertIn('"rotation offset"', build_source)
        self.assertIn('fdtd.set("theta", theta_deg)', build_source)
        self.assertIn('fdtd.set("phi", phi_deg)', build_source)
        self.assertIn('fdtd.set("frequency dependent profile", False)', build_source)
        self.assertNotIn('fdtd.set("number of field profile samples", 1)', build_source)
        self.assertNotIn('fdtd.set("auto update", True)', build_source)
        self.assertNotIn('fdtd.set("angle theta"', build_source)
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
        self.assertIn("Material mesh orders: grating 2", build_source)
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
        self.assertIn("z_extent_max_um = max(z_extent_max_um, device_z_um + 0.5 * z_span_um)", build_source)
        self.assertIn("simulation_z_max_um = z_extent_max_um + z_max_padding", build_source)
        self.assertIn("z_max_padding = max(boundary_clearance_um, requested_z_max_padding)", build_source)
        self.assertIn("boundary_clearance_um = 0.25 * min", build_source)
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
        self.assertEqual(analysis["waveguide_mode_monitor_name"], "waveguide_mode")
        self.assertEqual(analysis["waveguide_target_neff"], 2.5)
        self.assertEqual(analysis["waveguide_modal_direction"], "Tbackward")
        self.assertEqual(analysis["fiber_port_name"], "fiber_out")
        self.assertEqual(analysis["fiber_input_measurement_port_name"], "fiber_input_power")
        self.assertEqual(analysis["fiber_measurement_expansion_result_name"], "expansion for port monitor")
        self.assertIsNone(analysis["fiber_input_monitor_name"])
        self.assertEqual(analysis["fiber_source_mode"], "mode 2")
        self.assertEqual(analysis["fiber_polarization"], "Ey")
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
        resource_source = cell_source_containing(notebook, "saved pre-solve project ->")
        self.assertIn('getlicenseestimate("FDTD", "1")', resource_source)
        self.assertIn('fdtd.set("source port", str(GRATING_ANALYSIS["fiber_port_name"]))', resource_source)
        self.assertIn('fdtd.set("source mode", fiber_source_mode)', resource_source)
        self.assertIn('GRATING_ANALYSIS.get("fiber_source_mode", "mode 2")', resource_source)
        self.assertIn('GRATING_ANALYSIS.get("fiber_polarization", "Ey")', resource_source)
        self.assertIn("Backward along the tilted Z-axis fiber port", resource_source)
        self.assertIn("PIRIS_FSP_DIR / os.path.basename(REMOTE_PROJECT_FILE)", resource_source)
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
        self.assertIn("Passive tilted fiber port — forward T_in", analysis_source)
        self.assertIn("Passive tilted fiber-port power accounting", analysis_source)
        self.assertIn("Source-normalized waveguide total power", analysis_source)
        self.assertIn("Source-normalized selected waveguide-mode power", analysis_source)
        self.assertIn("_waveguide_total_power", analysis_source)
        self.assertIn("_waveguide_mode_power", analysis_source)
        self.assertIn("_fiber_forward", analysis_source)
        self.assertIn("_fiber_reflected", analysis_source)
        self.assertIn("_fiber_net", analysis_source)
        self.assertIn("passive tilted fiber port", analysis_source.lower())
        self.assertNotIn("grating_field_distribution.png", analysis_source)
        self.assertNotIn("field_intensity_normalized", analysis_source)
        self.assertIn('waveguide_power_data = fdtd.getresult(waveguide_power_monitor_name, "T")', analysis_source)
        self.assertIn('expansion_data = fdtd.getresult(waveguide_mode_monitor_name, expansion_result_name)', analysis_source)
        self.assertIn('modal_direction_key = str(GRATING_ANALYSIS.get("waveguide_modal_direction", "Tbackward"))', analysis_source)
        self.assertIn('_fiber_port_expansion(', analysis_source)
        self.assertIn('_find_result_key(fiber_expansion, "T_in", "Tin", "T in")', analysis_source)
        self.assertIn("fiber_coupling = waveguide_mode_power_source_normalized / np.maximum", analysis_source)
        self.assertNotIn("np.abs(scattering) ** 2", analysis_source)
        self.assertIn("fiber_coupling", analysis_source)
        self.assertNotIn("fiber_coupling_db", analysis_source)
        self.assertNotIn("coupling efficiency [dB]", analysis_source)
        self.assertIn("Incident-normalized waveguide coupling efficiency (linear)", analysis_source)
        self.assertIn("source-normalized linear power", analysis_source)
        self.assertIn("lam.fetch(_remote_grating_npz", analysis_source)
        self.assertIn("display(Image(filename=str(_local_response_png)", analysis_source)
        self.assertIn('fdtd.setexpansion(result_name, input_monitor_name)', build_source)
        self.assertIn('fdtd.set("mode selection", "fundamental mode")', build_source)
        self.assertIn('status = fdtd.updatemodes()', build_source)
        self.assertNotIn('fdtd.seteigensolver("use max index", 1)', build_source)
        self.assertNotIn('fdtd.seteigensolver("number of trial modes", trial_mode_count)', build_source)
        self.assertIn('mode_seed_project = os.path.join(REMOTE_WORK, "_max_layout_mode_seed.fsp")', build_source)
        self.assertIn('geometry_by_layer = _layer_builder_geometry(layer_builder_x_um, layer_builder_y_um)', build_source)
        self.assertIn('(global_vertices_um - local_origin_um) * UM', build_source)
        self.assertNotIn('fdtd.seteigensolver("n", target_neff)', build_source)
        self.assertNotIn('fdtd.updatemodes(mode_numbers)', build_source)
        fetch_source = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        self.assertNotIn('REMOTE_WORK + "/grating_analysis.npz"', fetch_source)
        self.assertNotIn('REMOTE_WORK + "/grating_response.png"', fetch_source)
        self.assertIn("geometry_xyz_projections.png", fetch_source)
        self.assertIn("PIRIS_FSP_DIR if remote_path == REMOTE_PROJECT_FILE", fetch_source)
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
        self.assertIn('SETTINGS.get("official_gc_domain", False)', build_source)
        self.assertIn('fdtd.set(antisymmetry_boundary + " bc", "Anti-Symmetric")', build_source)
        solve_source = cell_source_containing(notebook, "_solve_code")
        self.assertNotIn("Re-save solved project", solve_source)
        save_source = cell_source_containing(notebook, "REMOTE_RESULTS_SAVER")
        self.assertIn("Reused the already extracted grating spectrum", save_source)

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
        self.assertIn("normalized power (linear)", mmi_source)
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
        self.assertNotIn("addgaussian", build)
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
        project_prefetch = index("saved pre-solve project ->")
        port_mode_preview = index("REMOTE_PORT_MODE_PROFILES")
        project_review = index("OPEN_REMOTE_LUMERICAL_GUI")
        solve = index("_solve_code")
        save = index("REMOTE_RESULTS_SAVER")
        fetch = index("REMOTE_ARTIFACTS")
        checkin = index("products checkin")
        close = index("lam.close()")
        self.assertLess(checkout, build)
        self.assertLess(build, port_mode_preview)
        self.assertLess(port_mode_preview, project_prefetch)
        self.assertLess(build, project_prefetch)
        self.assertLess(project_prefetch, project_review)
        self.assertLess(project_review, solve)
        self.assertLess(solve, save)
        self.assertLess(save, fetch)
        self.assertLess(fetch, checkin)
        self.assertLessEqual(checkin, close)

        checkout_source = sources[checkout]
        release_source = sources[checkin]
        self.assertIn('--count 3 --expires "P1D" --mode user', checkout_source)
        self.assertIn('--count 3 --mode user', release_source)
        self.assertIn("fdtd.close()", release_source)
        self.assertIn("max_layout_results.npz", cell_source_containing(notebook, "REMOTE_RESULTS_SAVER"))
        self.assertEqual(
            notebook["metadata"]["max_layout"]["license_lifecycle"],
            "shared-web-3-hpc-packs-save-fetch-release",
        )

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
        self.assertIn('str(GRATING_ANALYSIS["waveguide_mode_monitor_name"])', preview)
        self.assertIn('"Waveguide fundamental mode"', preview)
        self.assertIn('WAVEGUIDE_MODE_SELECTIONS.get(object_name, {})', preview)
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
