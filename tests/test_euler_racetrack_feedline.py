from __future__ import annotations

import unittest

from max_layout.constants import DEFAULT_COMPONENT_VALUES, FEEDLINE_LAYER
from max_layout.gds.build import component_geometry_arrays, euler_racetrack_metrics, resolve_and_build
from max_layout.geometry.landmarks import (
    resolved_feedline_s_bend_length,
    smoothstep_s_bend_minimum_radius,
)
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


class EulerRacetrackFeedlineTests(unittest.TestCase):
    def test_25_ghz_round_trip_length_and_euler_straights(self) -> None:
        c_m_per_s = 299_792_458.0
        group_index = 2.21
        target_fsr_hz = 25.0e9
        total_um = c_m_per_s / (group_index * target_fsr_hz) * 1.0e6
        straight, bend_arc, _height = euler_racetrack_metrics(total_um, 100.0)

        self.assertAlmostEqual(total_um, 5426.107837104072, places=9)
        self.assertAlmostEqual(bend_arc, 628.318530717959, places=9)
        self.assertAlmostEqual(straight, 2084.735387834077, places=9)
        reconstructed = 2.0 * straight + 2.0 * bend_arc
        self.assertAlmostEqual(reconstructed, total_um, places=9)
        self.assertAlmostEqual(
            c_m_per_s / (group_index * reconstructed * 1.0e-6) / 1.0e9,
            25.0,
            places=9,
        )

    def test_euler_racetrack_component_builds_from_round_trip_length(self) -> None:
        component = {
            "uid": 1,
            "kind": "Euler racetrack",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": safe_json_copy(DEFAULT_COMPONENT_VALUES["Euler racetrack"]),
        }
        polygons, labels = component_geometry_arrays(component)
        self.assertGreaterEqual(len(polygons), 1)
        self.assertEqual(labels, [])
        self.assertEqual(component_local_ports(component)["center"]["center"], (0, 0))

    def test_feedline_can_end_after_one_s_bend(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"])
        parameters.update(
            {
                "input_straight_length": 10.0,
                "s_bend_length": 20.0,
                "offset": 5.0,
                "Lc": 100.0,
                "output_straight_length": 10.0,
                "add_grating_couplers": False,
                "add_output_s_bend": False,
                "input_s_bend_direction": "up",
                "output_s_bend_direction": "down",
            }
        )
        component = {
            "uid": 2,
            "kind": "Feedline",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": parameters,
        }
        ports = component_local_ports(component)
        self.assertEqual(ports["first_s_bend_end"]["center"], (30.0, 5.0))
        self.assertEqual(ports["Lc_end"]["center"], (130.0, 5.0))
        self.assertEqual(ports["second_s_bend_end"]["center"], (130.0, 5.0))
        self.assertEqual(ports["right_gc_point"]["center"], (140.0, 5.0))
        polygons, labels = component_geometry_arrays(component)
        self.assertGreaterEqual(len(polygons), 4)
        self.assertEqual(
            {layer for _points, layer, _datatype in polygons},
            {FEEDLINE_LAYER},
        )
        self.assertEqual(labels, [])

    def test_saved_feedline_is_canonicalized_to_dedicated_layer(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"])
        parameters.update(
            {
                "layer": 1,
                "add_grating_couplers": False,
                "add_output_s_bend": False,
            }
        )
        component = {
            "uid": 20,
            "kind": "Feedline",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": parameters,
        }
        library = resolve_and_build([component])
        layers = {
            polygon.layer
            for polygon in library.top_level()[0].get_polygons(
                apply_repetitions=True,
                include_paths=True,
            )
        }
        self.assertEqual(layers, {FEEDLINE_LAYER})

    def test_legacy_second_s_bend_remains_available(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"])
        parameters.update(
            {
                "input_straight_length": 10.0,
                "s_bend_length": 20.0,
                "offset": 5.0,
                "Lc": 100.0,
                "output_straight_length": 10.0,
                "add_grating_couplers": False,
                "add_output_s_bend": True,
                "input_s_bend_direction": "up",
                "output_s_bend_direction": "down",
            }
        )
        component = {
            "uid": 3,
            "kind": "Feedline",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": parameters,
        }
        ports = component_local_ports(component)
        self.assertEqual(ports["right_gc_point"]["center"], (160.0, 0.0))

    def test_radius_controlled_500_um_feedline_keeps_bend_near_bottom_gc(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"])
        parameters.update(
            {
                "input_straight_length": 50.0,
                "s_bend_length": 1.0,
                "s_bend_radius": 50.0,
                "offset": 20.0,
                "pre_coupling_straight_length": 100.0,
                "Lc": 100.0,
                "add_output_s_bend": False,
            }
        )
        bend_length = resolved_feedline_s_bend_length(parameters)
        parameters["output_straight_length"] = (
            500.0
            - parameters["input_straight_length"]
            - bend_length
            - parameters["pre_coupling_straight_length"]
            - parameters["Lc"]
        )
        component = {
            "uid": 4,
            "kind": "Feedline",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": parameters,
        }
        ports = component_local_ports(component)
        self.assertAlmostEqual(bend_length, 73.6179551838218, places=8)
        self.assertAlmostEqual(
            smoothstep_s_bend_minimum_radius(bend_length, 20.0, 200001),
            50.0,
            places=6,
        )
        self.assertEqual(ports["left_straight_end"]["center"], (50.0, 0.0))
        self.assertAlmostEqual(ports["coupling_straight_start"]["center"][0], 223.6179551838218, places=8)
        self.assertAlmostEqual(ports["right_gc_point"]["center"][0], 500.0, places=8)
        self.assertEqual(ports["right_gc_point"]["center"][1], 20.0)


if __name__ == "__main__":
    unittest.main()
