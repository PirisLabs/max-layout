from __future__ import annotations

from copy import deepcopy
import unittest

import gdstk

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.gds.build import component_geometry_arrays
from max_layout.gds.couplers import add_three_euler_inward_gc
from max_layout.ports import component_local_ports


KIND = "MZI vertical GC"


def component(parameters: dict | None = None) -> dict:
    return {
        "uid": 1,
        "kind": KIND,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": deepcopy(parameters or DEFAULT_COMPONENT_VALUES[KIND]),
    }


class MziVerticalGcStaggerTests(unittest.TestCase):
    def assert_point_close(self, actual, expected, places: int = 8) -> None:
        self.assertAlmostEqual(float(actual[0]), float(expected[0]), places=places)
        self.assertAlmostEqual(float(actual[1]), float(expected[1]), places=places)

    def test_default_and_legacy_projects_remain_vertically_aligned(self) -> None:
        self.assertEqual(DEFAULT_COMPONENT_VALUES[KIND]["gc_horizontal_stagger"], 0.0)
        parameters = deepcopy(DEFAULT_COMPONENT_VALUES[KIND])
        parameters.pop("gc_horizontal_stagger")
        ports = component_local_ports(component(parameters))
        self.assertAlmostEqual(
            ports["left_gc_waveguide"]["center"][0],
            ports["right_gc_waveguide"]["center"][0],
        )

    def test_positive_stagger_moves_upper_left_and_lower_right(self) -> None:
        parameters = deepcopy(DEFAULT_COMPONENT_VALUES[KIND])
        parameters.update({"arm_length": 10000.0, "gc_horizontal_stagger": 100.0})
        ports = component_local_ports(component(parameters))
        center_x = ports["center"]["center"][0]
        upper = ports["left_gc_waveguide"]["center"]
        lower = ports["right_gc_waveguide"]["center"]
        self.assert_point_close(upper, (center_x - 100.0, abs(upper[1])))
        self.assert_point_close(lower, (center_x + 100.0, -abs(lower[1])))
        self.assertAlmostEqual(lower[0] - upper[0], 200.0)

    def test_built_route_endpoints_match_the_reported_ports(self) -> None:
        parameters = deepcopy(DEFAULT_COMPONENT_VALUES[KIND])
        parameters.update({"arm_length": 12000.0, "gc_horizontal_stagger": 100.0})
        ports = component_local_ports(component(parameters))
        total_length = ports["right_external"]["center"][0]
        cell = gdstk.Cell("MZI_VERTICAL_GC_STAGGER_TEST")
        left, left_orientation = add_three_euler_inward_gc(
            cell, parameters, (0.0, 0.0), 180.0, True, 1.2, 1, 0, "LEFT_GC"
        )
        right, right_orientation = add_three_euler_inward_gc(
            cell,
            parameters,
            (total_length, 0.0),
            0.0,
            False,
            1.2,
            1,
            0,
            "RIGHT_GC",
        )
        self.assert_point_close(left, ports["left_gc_waveguide"]["center"])
        self.assert_point_close(right, ports["right_gc_waveguide"]["center"])
        self.assertAlmostEqual(left_orientation, 90.0)
        self.assertAlmostEqual(right_orientation, 270.0)
        self.assertGreater(len(cell.polygons), 0)

    def test_staggered_component_builds_and_rejects_an_impossible_offset(self) -> None:
        parameters = deepcopy(DEFAULT_COMPONENT_VALUES[KIND])
        parameters["gc_horizontal_stagger"] = 100.0
        polygons, labels = component_geometry_arrays(component(parameters))
        self.assertGreater(len(polygons), 0)
        self.assertEqual(labels, [])

        parameters["gc_horizontal_stagger"] = 100000.0
        with self.assertRaisesRegex(ValueError, "straight lengths cannot be negative"):
            component_geometry_arrays(component(parameters))


if __name__ == "__main__":
    unittest.main()
