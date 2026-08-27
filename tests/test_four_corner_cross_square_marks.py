from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from max_layout.constants import (
    COMPONENT_SPECS,
    DEFAULT_COMPONENT_VALUES,
    MARKER_COMPONENT_KINDS,
)
from max_layout.gds.build import component_geometry_arrays
from max_layout.params import resize_component_parameters
from max_layout.ports import component_local_ports


KIND = "Four-corner cross + squares marks"


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


def polygon_signatures(polygons):
    signatures = []
    for points, layer, datatype in polygons:
        points = np.asarray(points, dtype=float)
        low = points.min(axis=0)
        high = points.max(axis=0)
        signatures.append(
            (
                int(layer),
                int(datatype),
                tuple(np.round(low, 9)),
                tuple(np.round(high, 9)),
                round(
                    abs(
                        float(
                            (points[1, 0] - points[0, 0])
                            * (points[2, 1] - points[0, 1])
                            - (points[1, 1] - points[0, 1])
                            * (points[2, 0] - points[0, 0])
                        )
                    ),
                    9,
                ),
            )
        )
    return sorted(signatures)


class FourCornerCrossSquareMarksTests(unittest.TestCase):
    def test_component_is_visible_and_all_dimensions_are_editable(self) -> None:
        self.assertIn(KIND, MARKER_COMPONENT_KINDS)
        defaults = DEFAULT_COMPONENT_VALUES[KIND]
        self.assertEqual(defaults["x_spacing"], 1000.0)
        self.assertEqual(defaults["y_spacing"], 1000.0)
        for key in (
            "size",
            "bar_width",
            "square_size",
            "square_gap",
            "x_spacing",
            "y_spacing",
        ):
            self.assertIn(key, COMPONENT_SPECS[KIND])
        self.assertEqual(component_local_ports(component(KIND))["center"]["center"], (0, 0))

    def test_geometry_matches_four_explicit_marks_at_center_spaced_corners(self) -> None:
        mark_set = component(KIND)
        mark_set["params"].update({"x_spacing": 840.0, "y_spacing": 520.0})
        combined, combined_labels = component_geometry_arrays(mark_set)
        self.assertEqual(combined_labels, [])

        explicit = []
        for index, (x, y) in enumerate(
            ((-420.0, -260.0), (420.0, -260.0), (-420.0, 260.0), (420.0, 260.0)),
            start=10,
        ):
            child = component("Cross + squares mark", uid=index)
            child["x"] = x
            child["y"] = y
            child["params"].update(
                {
                    key: mark_set["params"][key]
                    for key in ("size", "bar_width", "square_size", "square_gap")
                }
            )
            polygons, labels = component_geometry_arrays(child)
            explicit.extend(polygons)
            self.assertEqual(labels, [])

        self.assertEqual(len(combined), 24)
        self.assertTrue(all(layer == 3 and datatype == 0 for _points, layer, datatype in combined))
        self.assertEqual(polygon_signatures(combined), polygon_signatures(explicit))

    def test_rotation_preserves_center_to_center_spacing(self) -> None:
        mark_set = component(KIND)
        mark_set["orientation_deg"] = 90.0
        mark_set["params"].update({"x_spacing": 800.0, "y_spacing": 400.0})
        polygons, _labels = component_geometry_arrays(mark_set)
        points = np.vstack([points for points, _layer, _datatype in polygons])
        extent = points.max(axis=0) - points.min(axis=0)
        size = float(mark_set["params"]["size"])
        np.testing.assert_allclose(extent, [400.0 + size, 800.0 + size], atol=1e-9)

    def test_corner_resize_scales_spacing_by_axis_and_mark_uniformly(self) -> None:
        resized = resize_component_parameters(
            KIND,
            DEFAULT_COMPONENT_VALUES[KIND],
            scale_x=2.0,
            scale_y=0.5,
        )
        self.assertEqual(resized["x_spacing"], 2000.0)
        self.assertEqual(resized["y_spacing"], 500.0)
        # Uniform mark dimensions use the geometric-mean scale (one here).
        self.assertEqual(resized["size"], DEFAULT_COMPONENT_VALUES[KIND]["size"])

    def test_nonpositive_spacing_is_rejected(self) -> None:
        mark_set = component(KIND)
        mark_set["params"]["x_spacing"] = 0.0
        with self.assertRaisesRegex(ValueError, "positive mark size"):
            component_geometry_arrays(mark_set)


if __name__ == "__main__":
    unittest.main()
