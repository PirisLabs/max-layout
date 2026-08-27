from __future__ import annotations

import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.gds.build import component_geometry_arrays
from max_layout.utils import safe_json_copy


def wafer(parameters: dict | None = None) -> dict:
    return {
        "uid": 1,
        "kind": "4-inch wafer outline",
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": safe_json_copy(
            parameters or DEFAULT_COMPONENT_VALUES["4-inch wafer outline"]
        ),
    }


class WaferOutlineTests(unittest.TestCase):
    def test_default_four_inch_wafer_includes_three_inch_boundary(self) -> None:
        defaults = DEFAULT_COMPONENT_VALUES["4-inch wafer outline"]
        self.assertEqual(defaults["diameter"], 100000.0)
        self.assertIs(defaults["show_3_inch_boundary"], True)
        self.assertEqual(defaults["three_inch_boundary_diameter"], 76200.0)

        polygons, labels = component_geometry_arrays(wafer())
        self.assertEqual(labels, [])
        self.assertGreaterEqual(len(polygons), 2)
        self.assertEqual({layer for _points, layer, _datatype in polygons}, {100})
        radii = np.concatenate(
            [np.linalg.norm(points, axis=1) for points, _layer, _datatype in polygons]
        )
        self.assertLess(float(np.min(np.abs(radii - 38100.0))), 1.0)
        self.assertLess(float(np.min(np.abs(radii - 38000.0))), 1.0)
        self.assertLess(float(np.min(np.abs(radii - 50000.0))), 1.0)

    def test_three_inch_boundary_can_be_hidden(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["4-inch wafer outline"])
        parameters["show_3_inch_boundary"] = 0
        polygons, _labels = component_geometry_arrays(wafer(parameters))
        radii = np.concatenate(
            [np.linalg.norm(points, axis=1) for points, _layer, _datatype in polygons]
        )
        self.assertGreater(float(np.min(np.abs(radii - 38100.0))), 1000.0)

    def test_three_inch_boundary_must_fit_inside_outer_wafer(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["4-inch wafer outline"])
        parameters["three_inch_boundary_diameter"] = parameters["diameter"]
        with self.assertRaisesRegex(ValueError, "boundary diameter < wafer diameter"):
            component_geometry_arrays(wafer(parameters))


if __name__ == "__main__":
    unittest.main()
