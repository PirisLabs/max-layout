from __future__ import annotations

import unittest

import numpy as np

from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    LAYER_NAME_MAP,
    PHOTONIC_COMPONENT_KINDS,
)
from max_layout.gds.build import component_geometry_arrays
from max_layout.geometry.slab_taper import slab_taper_profile
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


class SlabTaperTests(unittest.TestCase):
    def component(self, parameters: dict | None = None) -> dict:
        return {
            "uid": 1,
            "kind": "SLAB taper",
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "attachment": None,
            "params": safe_json_copy(
                parameters or DEFAULT_COMPONENT_VALUES["SLAB taper"]
            ),
        }

    def test_defaults_and_library_membership(self) -> None:
        defaults = DEFAULT_COMPONENT_VALUES["SLAB taper"]
        self.assertIn("SLAB taper", PHOTONIC_COMPONENT_KINDS)
        self.assertEqual(defaults["initial_width"], 1.2)
        self.assertEqual(defaults["final_width"], 0.3)
        self.assertEqual(defaults["exponential_factor"], 1.3)
        self.assertEqual(defaults["layer"], 8)
        self.assertEqual(LAYER_NAME_MAP[8], "SLAB")

    def test_exponential_profile_has_exact_endpoints(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES["SLAB taper"])
        parameters.update(
            {
                "initial_width": 4.0,
                "final_width": 0.5,
                "length": 600.0,
                "exponential_factor": 1.7,
                "points": 101,
            }
        )
        xs, widths = slab_taper_profile(parameters)
        self.assertEqual(len(xs), 101)
        self.assertAlmostEqual(xs[0], 0.0)
        self.assertAlmostEqual(xs[-1], 600.0)
        self.assertAlmostEqual(widths[0], 4.0)
        self.assertAlmostEqual(widths[-1], 0.5)
        expected_mid = 4.0 * (0.5 / 4.0) ** (0.5**1.7)
        self.assertAlmostEqual(widths[50], expected_mid)

    def test_geometry_is_only_on_slab_layer_and_ports_match_length(self) -> None:
        component = self.component()
        polygons, _labels = component_geometry_arrays(component)
        self.assertTrue(polygons)
        self.assertEqual({layer for _points, layer, _datatype in polygons}, {8})
        points = np.vstack([points for points, _layer, _datatype in polygons])
        self.assertAlmostEqual(float(points[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(points[:, 0].max()), 365.0)
        start_y = points[np.isclose(points[:, 0], 0.0), 1]
        self.assertAlmostEqual(float(start_y.min()), -0.6)
        self.assertAlmostEqual(float(start_y.max()), 0.6)
        ports = component_local_ports(component)
        self.assertEqual(ports["left"]["center"], (0.0, 0.0))
        self.assertEqual(ports["right"]["center"], (365.0, 0.0))


if __name__ == "__main__":
    unittest.main()
