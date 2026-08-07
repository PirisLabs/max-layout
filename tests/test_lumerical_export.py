from __future__ import annotations

from copy import deepcopy
import json
import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.gds.build import component_geometry_arrays, resolve_and_build
from max_layout import lumerical
from max_layout.lumerical import generate_lumerical_notebook, seed_simulation_ports


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
        polygons, _ = component_geometry_arrays(grating)
        self.assertEqual(len(polygons), 47)
        self.assertEqual({(layer, datatype) for _points, layer, datatype in polygons}, {(1, 0), (2, 0)})
        all_points = np.vstack([points for points, _layer, _datatype in polygons])
        self.assertAlmostEqual(float(all_points[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(all_points[:, 0].max()), 70.91271704, places=6)

        stack = lumerical.default_stack("SOI grating coupler (Ansys)")
        self.assertEqual([row["thickness_um"] for row in stack], [3.0, 1.0, 0.12, 0.10, 0.48])
        self.assertEqual(stack[2]["gds_layers"], [1])
        self.assertEqual(stack[3]["gds_layers"], [2])
        self.assertTrue(stack[4]["conformal"])
        self.assertTrue(all(row["mesh_factor"] == 0.1 for row in stack))

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
        fiber = component("Fiber geometry", uid=3)
        fiber["x"] = 20.0
        fiber["params"]["name"] = "fiber"
        fiber["params"]["angle theta"] = 10.0
        fiber_port = component("Fiber-axis FDTD port", uid=5)
        fiber_port["x"] = 20.0
        fiber_port["params"]["name"] = "fiber_out"
        fiber_port["params"]["order"] = 2
        fiber_port["params"]["angle theta"] = 7.0
        waveguide = component("FDTD port", uid=4)
        waveguide["x"] = -27.0
        waveguide["params"]["name"] = "waveguide_in"
        waveguide["params"]["order"] = 1
        notebook, _ = generate_lumerical_notebook(
            [grating, power_monitor, fiber, fiber_port, waveguide],
            {
                "included_layers": [[1, 0], [2, 0]],
                "include_ports": True,
                "material_stack": [
                    {"name": "BOX", "material": "SiO2 (Glass) - Palik", "thickness_um": 2.0, "role": "background", "gds_layer": 0},
                    {"name": "Exported cross-section", "material": "LiNbO3", "thickness_um": 0.6, "etch_depth_um": 0.3, "sidewall_angle_deg": 82.0, "slab_extent": "geometry", "mesh_factor": 0.5, "role": "geometry", "gds_layer": 1},
                    {"name": "SiO2 top cladding", "material": "SiO2 (Glass) - Palik", "thickness_um": 1.0, "role": "background", "gds_layer": 0, "conformal": True},
                    {"name": "Absent metal", "material": "Au (Gold) - CRC", "thickness_um": 0.0, "role": "geometry", "gds_layer": 4},
                ],
            },
        )
        ports = assignment_value(notebook, "PORTS")
        fiber_ports = [port for port in ports if port.get("plane normal") == "Z"]
        fiber_geometries = assignment_value(notebook, "FIBER_GEOMETRIES")
        self.assertEqual(len(ports), 2)
        self.assertFalse(any(port.get("auto_generated_for_grating") for port in ports))
        self.assertEqual(len(fiber_ports), 1)
        self.assertEqual(len(fiber_geometries), 1)
        self.assertEqual(fiber_ports[0]["name"], "fiber_out")
        self.assertEqual(fiber_ports[0]["angle theta"], fiber_geometries[0]["angle theta"])
        self.assertAlmostEqual(
            fiber_ports[0]["rotation offset_um"],
            4.0 * fiber_geometries[0]["core diameter_um"] * np.tan(np.deg2rad(10.0)),
        )
        json.dumps(notebook)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<notebook cell {index}>", "exec")
        build_source = cell_source_containing(notebook, "REMOTE_MODEL_BUILDER =")
        self.assertIn("addport", build_source)
        self.assertIn('serverArgs={"threads": str(build_cpu_threads)}', build_source)
        self.assertIn('SETTINGS.get("build_cpu_threads", 30)', build_source)
        self.assertNotIn("addgaussian", build_source)
        self.assertIn("addstructuregroup", build_source)
        self.assertIn("addcircle", build_source)
        self.assertIn('"rotation offset"', build_source)
        self.assertIn('fdtd.set("theta", theta_deg)', build_source)
        self.assertIn('fdtd.set("phi", phi_deg)', build_source)
        self.assertNotIn('fdtd.set("angle theta"', build_source)
        self.assertIn("addlayerbuilder", build_source)
        self.assertIn('addmaterial("Sampled 3D data")', build_source)
        self.assertIn('"sampled 3d data"', build_source)
        self.assertIn("_silica_cladding_top_um", build_source)
        self.assertIn("fiber, device_top_um, stack_top_um, silica_cladding_top_um", build_source)
        self.assertIn("center_z_um = reference_z_um + bottom_gap_um", build_source)
        self.assertIn('"sidewall angle"', build_source)
        self.assertIn("z_extent_max_um = max(z_extent_max_um, device_z_um + 0.5 * z_span_um)", build_source)
        self.assertIn("simulation_z_max_um = z_extent_max_um + z_max_padding", build_source)
        self.assertIn("z_max_padding = max(boundary_clearance_um, requested_z_max_padding)", build_source)
        self.assertIn("boundary_clearance_um = 0.25 * min", build_source)
        self.assertIn("domain_padding = dict(SETTINGS.get", build_source)
        self.assertIn("_add_waveguide_boundary_extensions", build_source)
        self.assertIn("Extended waveguide at port", build_source)
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
        self.assertIn("bounds[axis_index] - pml_geometry_overlap_um", build_source)
        self.assertIn("Added scripted Ansys fiber property group", build_source)
        self.assertIn('fdtd.adduserprop("core diameter", 2', build_source)
        self.assertIn('fdtd.adduserprop("cladding diameter", 2', build_source)
        self.assertIn('fdtd.adduserprop("z span", 2', build_source)
        self.assertIn('fdtd.set("script", fiber_setup_script)', build_source)
        self.assertIn('set("x",0.0);', build_source)
        self.assertIn('set("y",0.0);', build_source)
        self.assertIn('set("z",0.0);', build_source)
        self.assertIn("fdtd.updateportmodes()", build_source)
        self.assertIn("no source or port was created", build_source)
        payload = cell_source_containing(notebook, "MATERIAL_STACK =")
        self.assertIn("'dimension': '3D'", payload)
        self.assertIn("'sidewall_angle_deg': 82.0", payload)
        self.assertIn("'slab_extent': 'geometry'", payload)
        self.assertIn("'mesh_factor': 0.5", payload)
        self.assertIn("'conformal': True", payload)
        self.assertIn("GRATING_ANALYSIS =", payload)
        analysis = assignment_value(notebook, "GRATING_ANALYSIS")
        self.assertEqual(analysis["waveguide_port_name"], "waveguide_in")
        self.assertEqual(analysis["fiber_port_name"], "fiber_out")
        self.assertEqual(analysis["center_um"], fiber_ports[0]["center"])
        self.assertEqual(analysis["frequency_points"], 50)
        settings = assignment_value(notebook, "SETTINGS")
        self.assertEqual(settings["resource_mode"], "GPU")
        self.assertEqual(settings["dt_stability_factor"], 0.99)
        self.assertEqual(settings["pml_profile"], "Standard")
        self.assertEqual(settings["simulation_time_fs"], 2000.0)
        self.assertEqual(settings["frequency_points"], 50)
        self.assertEqual(settings["build_cpu_threads"], 30)
        self.assertEqual(settings["tfln_crystal_cut"], "X")
        self.assertEqual(settings["tfln_temperature_K"], 296.3)
        run_source = cell_source_containing(notebook, "_solve_code")
        self.assertIn('fdtd.run("FDTD", "GPU")', run_source)
        resource_source = cell_source_containing(notebook, "saved pre-solve project ->")
        self.assertIn('getlicenseestimate("FDTD", "1")', resource_source)
        self.assertIn('fdtd.set("source port", str(GRATING_ANALYSIS["fiber_port_name"]))', resource_source)
        self.assertIn('fdtd.set("source mode", "mode 1")', resource_source)
        self.assertIn("Backward along the tilted Z-axis fiber port", resource_source)
        self.assertIn("PIRIS_FSP_DIR / os.path.basename(REMOTE_PROJECT_FILE)", resource_source)
        self.assertNotIn("_solve_code", resource_source)
        review_source = cell_source_containing(notebook, "OPEN_REMOTE_LUMERICAL_GUI")
        self.assertIn("FileLink", review_source)
        self.assertIn("Lambda is headless", review_source)
        self.assertIn('fdtd.set("dt stability factor", dt_stability_factor)', build_source)
        self.assertIn('fdtd.set("pml profile", 2 if pml_profile_name == "stabilized" else 1)', build_source)
        analysis_source = cell_source_containing(notebook, "REMOTE_GRATING_ANALYSIS")
        self.assertIn("farfield3d", analysis_source)
        self.assertIn("farfieldux", analysis_source)
        self.assertIn("farfielduy", analysis_source)
        self.assertIn("grating_response.png", analysis_source)
        self.assertIn("grating_farfield.png", analysis_source)
        self.assertIn('receiver_port_path = "::model::FDTD::ports::" + waveguide_port_name', analysis_source)
        self.assertIn('T_data = fdtd.getresult(receiver_port_path, "T")', analysis_source)
        self.assertIn('fiber_coupling = np.abs(fiber_coupling)', analysis_source)
        self.assertNotIn("np.abs(scattering) ** 2", analysis_source)
        self.assertIn("fiber port to waveguide port", analysis_source)
        self.assertIn('subplot_kw={"projection": "polar"}', analysis_source)
        self.assertIn("axis.set_theta_zero_location", analysis_source)
        self.assertIn("theta_grid_deg", analysis_source)
        self.assertIn("phi_grid_rad", analysis_source)
        self.assertIn("fiber_coupling_db", analysis_source)
        self.assertIn("coupling efficiency [dB]", analysis_source)
        fetch_source = cell_source_containing(notebook, "REMOTE_ARTIFACTS")
        self.assertIn("grating_analysis.npz", fetch_source)
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
        connection_source = cell_source_containing(notebook, "def solve_remote_checked")
        self.assertIn("file=_ml_sys.stdout", connection_source)
        self.assertIn("Lumerical solver log:", connection_source)
        self.assertIn("lam.show(GEOMETRY_PROJECTIONS_FILE", projection_source)

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
        project_review = index("OPEN_REMOTE_LUMERICAL_GUI")
        solve = index("_solve_code")
        save = index("REMOTE_RESULTS_SAVER")
        fetch = index("REMOTE_ARTIFACTS")
        checkin = index("products checkin")
        close = index("lam.close()")
        self.assertLess(checkout, build)
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

    def test_embedded_remote_programs_compile(self) -> None:
        for name in (
            "_BUILD_CELL",
            "_GEOMETRY_PROJECTIONS_REMOTE",
            "_REMOTE_RESOURCE_AND_SAVE",
            "_SAVE_REMOTE_RESULTS",
            "_GRATING_ANALYSIS_REMOTE",
        ):
            with self.subTest(name=name):
                compile(getattr(lumerical, name), "<%s>" % name, "exec")


if __name__ == "__main__":
    unittest.main()
