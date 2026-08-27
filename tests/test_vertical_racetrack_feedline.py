from __future__ import annotations

import unittest

from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    GC_COMPOSITE_KINDS,
    PHOTONIC_COMPONENT_KINDS,
    VERTICAL_RACETRACK_FEEDLINE_KIND,
)
from max_layout.gds.build import component_geometry_arrays
from max_layout.geometry.landmarks import feedline_landmarks
from max_layout.geometry.vertical_racetrack_feedline import (
    resolve_vertical_racetrack_feedline,
)
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


def component(parameters: dict | None = None, mirrored: bool = False) -> dict:
    return {
        "uid": 1,
        "kind": VERTICAL_RACETRACK_FEEDLINE_KIND,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": mirrored,
        "attachment": None,
        "params": safe_json_copy(
            parameters
            or DEFAULT_COMPONENT_VALUES[VERTICAL_RACETRACK_FEEDLINE_KIND]
        ),
    }


class VerticalRacetrackFeedlineTests(unittest.TestCase):
    def test_component_is_available_in_resonator_library(self) -> None:
        self.assertIn(VERTICAL_RACETRACK_FEEDLINE_KIND, PHOTONIC_COMPONENT_KINDS)
        self.assertIn(VERTICAL_RACETRACK_FEEDLINE_KIND, GC_COMPOSITE_KINDS)
        defaults = DEFAULT_COMPONENT_VALUES[VERTICAL_RACETRACK_FEEDLINE_KIND]
        self.assertEqual(defaults["feedline_length"], 2300.0)
        self.assertEqual(defaults["s_bend_length"], 200.0)
        self.assertEqual(defaults["offset"], 200.0)
        self.assertEqual(defaults["pitch"], 0.83)
        self.assertEqual(defaults["fill_factor"], 0.47)

    def test_two_half_s_bends_keep_coupling_at_feedline_midpoint(self) -> None:
        parameters = safe_json_copy(
            DEFAULT_COMPONENT_VALUES[VERTICAL_RACETRACK_FEEDLINE_KIND]
        )
        layout = resolve_vertical_racetrack_feedline(parameters)
        feed = layout["feed_params"]
        landmarks = feedline_landmarks(feed, False)

        self.assertEqual(layout["each_s_bend_length"], 100.0)
        self.assertEqual(layout["each_lateral_offset"], 100.0)
        self.assertEqual(feed["pre_coupling_straight_length"], 950.0)
        self.assertEqual(feed["post_coupling_straight_length"], 950.0)
        self.assertEqual(feed["output_straight_length"], 50.0)
        self.assertTrue(feed["add_output_s_bend"])
        self.assertEqual(landmarks["input"][0], 0.0)
        self.assertEqual(landmarks["output"][0], 2300.0)
        self.assertEqual(
            (landmarks["coupling_straight_start"][0] + landmarks["lc_end"][0])
            / 2.0,
            1150.0,
        )

        ports = component_local_ports(component(parameters))
        self.assertAlmostEqual(ports["input_gc_point"]["center"][1], -1150.0)
        self.assertAlmostEqual(ports["output_gc_point"]["center"][1], 1150.0)
        self.assertEqual(ports["coupling_point"]["center"][1], 0.0)
        coupling_x = ports["coupling_point"]["center"][0]
        input_x = ports["input_gc_point"]["center"][0]
        output_x = ports["output_gc_point"]["center"][0]
        self.assertAlmostEqual((input_x + output_x) / 2.0, coupling_x)
        self.assertAlmostEqual(abs(input_x - output_x), 200.0)

    def test_requested_gap_is_edge_to_edge_at_curved_end(self) -> None:
        parameters = safe_json_copy(
            DEFAULT_COMPONENT_VALUES[VERTICAL_RACETRACK_FEEDLINE_KIND]
        )
        layout = resolve_vertical_racetrack_feedline(parameters)
        coupling_x = float(layout["coupling_center"][0])
        edge_gap = (
            coupling_x
            - float(layout["racetrack_half_x"])
            - (float(parameters["racetrack_width"]) + float(parameters["wg_width"]))
            / 2.0
        )
        self.assertAlmostEqual(edge_gap, parameters["coupling_gap"], places=9)

    def test_parameter_changes_preserve_symmetry(self) -> None:
        parameters = safe_json_copy(
            DEFAULT_COMPONENT_VALUES[VERTICAL_RACETRACK_FEEDLINE_KIND]
        )
        parameters.update(
            {
                "feedline_length": 2600.0,
                "input_straight_length": 60.0,
                "s_bend_length": 300.0,
                "offset": 240.0,
                "Lc": 120.0,
            }
        )
        layout = resolve_vertical_racetrack_feedline(parameters)
        feed = layout["feed_params"]
        self.assertEqual(layout["each_s_bend_length"], 150.0)
        self.assertEqual(layout["each_lateral_offset"], 120.0)
        self.assertEqual(feed["pre_coupling_straight_length"], 1030.0)
        self.assertEqual(feed["post_coupling_straight_length"], 1030.0)
        ports = component_local_ports(component(parameters))
        self.assertAlmostEqual(ports["input_gc_point"]["center"][1], -1300.0)
        self.assertAlmostEqual(ports["output_gc_point"]["center"][1], 1300.0)
        self.assertEqual(ports["coupling_point"]["center"][1], 0.0)

    def test_component_builds_flattened_waveguide_and_grating_geometry(self) -> None:
        polygons, labels = component_geometry_arrays(component())
        self.assertGreater(len(polygons), 10)
        self.assertEqual(labels, [])
        self.assertIn(1, {layer for _points, layer, _datatype in polygons})


if __name__ == "__main__":
    unittest.main()
