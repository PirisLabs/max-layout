from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES, PHOTONIC_LAYER, SLAB_LAYER
from max_layout.gds.build import _canonicalize_component_layers, component_geometry_arrays
from max_layout.geometry.linear_slab_etch import LINEAR_SLAB_ETCH_KIND, resolve_linear_slab_etch
from max_layout.ports import component_local_ports
from max_layout.ui.window import NativeLayoutWindow, linear_slab_etch_lumerical_export_settings


def component(parameters: dict | None = None) -> dict:
    return {
        "uid": 1,
        "kind": LINEAR_SLAB_ETCH_KIND,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": deepcopy(parameters or DEFAULT_COMPONENT_VALUES[LINEAR_SLAB_ETCH_KIND]),
    }


class Factory:
    make_component = NativeLayoutWindow.make_component
    automatic_simulation_companions = NativeLayoutWindow.automatic_simulation_companions
    synchronize_automatic_simulation_companions = NativeLayoutWindow.synchronize_automatic_simulation_companions

    def __init__(self) -> None:
        self.components = []
        self.next_uid = 1


class LinearSlabEtchTests(unittest.TestCase):
    def test_defaults_match_supplied_reference_gds(self) -> None:
        defaults = DEFAULT_COMPONENT_VALUES[LINEAR_SLAB_ETCH_KIND]
        expected = {
            "slab_tip_width": 0.2,
            "slab_tip_taper_length": 100.0,
            "slab_tip_length": 6.0,
            "slab_wg_width": 32.0,
            "slab_wg_length": 11.0,
            "rib_tip_width": 0.3,
            "rib_taper_length": 60.0,
            "rib_wg_width": 1.2,
            "rib_wg_length": 11.0,
            "slab_tfln_thickness": 0.2,
            "rib_tfln_thickness": 0.2,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(defaults[key], value)

        polygons, labels = component_geometry_arrays(component())
        self.assertFalse(labels)
        self.assertEqual({layer for _points, layer, _datatype in polygons}, {PHOTONIC_LAYER, SLAB_LAYER})
        slab_points = np.vstack([points for points, layer, _datatype in polygons if layer == SLAB_LAYER])
        rib_points = np.vstack([points for points, layer, _datatype in polygons if layer == PHOTONIC_LAYER])
        np.testing.assert_allclose(slab_points.min(axis=0), [0.0, -16.0])
        np.testing.assert_allclose(slab_points.max(axis=0), [117.0, 16.0])
        np.testing.assert_allclose(rib_points.min(axis=0), [46.0, -0.6])
        np.testing.assert_allclose(rib_points.max(axis=0), [117.0, 0.6])

        q = resolve_linear_slab_etch(defaults)
        np.testing.assert_allclose(q["slab_taper_x"], [6, 16, 31, 46, 61, 76, 91, 106])
        np.testing.assert_allclose(q["slab_taper_width"], [0.2, 0.34, 0.5, 0.8, 1.2, 2.5, 8.0, 32.0])
        np.testing.assert_allclose(q["rib_taper_x"], [46, 61, 76, 91, 106])
        np.testing.assert_allclose(q["rib_taper_width"], [0.3, 0.5, 0.8, 1.1, 1.2])

    def test_ports_are_input_clearance_and_output_straight_midpoint(self) -> None:
        ports = component_local_ports(component())
        self.assertEqual(ports["input_fiber"]["center"], (-1.0, 0.0))
        self.assertEqual(ports["input_fiber"]["outward_orientation_deg"], 180.0)
        self.assertEqual(ports["waveguide_point"]["center"], (111.5, 0.0))
        self.assertEqual(ports["waveguide_point"]["outward_orientation_deg"], 0.0)
        parameters = deepcopy(DEFAULT_COMPONENT_VALUES[LINEAR_SLAB_ETCH_KIND])
        parameters["rib_wg_length"] = 20.0
        self.assertEqual(component_local_ports(component(parameters))["waveguide_point"]["center"], (116.0, 0.0))

    def test_layers_are_canonicalized_to_wg_and_slab(self) -> None:
        item = component()
        item["params"].update({"layer": 99, "slab_layer": 98})
        _canonicalize_component_layers(item)
        self.assertEqual(item["params"]["layer"], PHOTONIC_LAYER)
        self.assertEqual(item["params"]["slab_layer"], SLAB_LAYER)

    def test_automatic_simulation_ports_follow_component_ports(self) -> None:
        factory = Factory()
        item = factory.make_component(LINEAR_SLAB_ETCH_KIND, 0.0, 0.0)
        factory.components.append(item)
        companions = factory.automatic_simulation_companions(item)
        self.assertEqual([entry["kind"] for entry in companions], ["FDTD port", "FDTD port"])
        by_name = {entry["simulation_parent_port"]: entry for entry in companions}
        self.assertEqual((by_name["input_fiber"]["x"], by_name["input_fiber"]["y"]), (-1.0, 0.0))
        self.assertEqual((by_name["waveguide_point"]["x"], by_name["waveguide_point"]["y"]), (111.5, 0.0))
        self.assertEqual(by_name["input_fiber"]["params"]["order"], 1)
        self.assertEqual(by_name["waveguide_point"]["params"]["order"], 2)

    def test_lumerical_stack_matches_requested_cross_section(self) -> None:
        item = component()
        settings = linear_slab_etch_lumerical_export_settings({}, item)
        self.assertEqual(settings["stack_preset"], "Linear Slab Etch TFLN")
        stack = settings["material_stack"]
        self.assertEqual([row["thickness_um"] for row in stack], [2.0, 12.0, 0.4, 5.0, 1.0])
        cross_section = stack[2]
        self.assertEqual(cross_section["material"], "LiNbO3")
        self.assertEqual(cross_section["etch_depth_um"], 0.2)
        self.assertEqual(cross_section["gds_layers"], [PHOTONIC_LAYER])
        self.assertEqual(cross_section["slab_gds_layers"], [SLAB_LAYER])
        self.assertEqual(cross_section["slab_extent"], "geometry")


if __name__ == "__main__":
    unittest.main()
