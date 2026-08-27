from __future__ import annotations

import unittest

import numpy as np

from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    GLOBAL_MARKER_LAYER,
    LAYER_NAME_MAP,
)
from max_layout.gds.build import (
    _canonicalize_component_layers,
    component_geometry_arrays,
)
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


def chip_outline(parameters: dict | None = None) -> dict:
    values = safe_json_copy(DEFAULT_COMPONENT_VALUES["Chip outline"])
    if parameters:
        values.update(parameters)
    return {
        "uid": 1,
        "kind": "Chip outline",
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "params": values,
    }


class ChipOutlineGlobalMarkerTests(unittest.TestCase):
    def test_default_frame_contains_three_50_um_squares_per_side(self) -> None:
        polygons, _labels = component_geometry_arrays(chip_outline())
        marker_polygons = [
            points
            for points, layer, datatype in polygons
            if layer == GLOBAL_MARKER_LAYER and datatype == 0
        ]
        self.assertEqual(len(marker_polygons), 12)

        measured = set()
        for points in marker_polygons:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            center = (minimum + maximum) / 2.0
            size = maximum - minimum
            measured.add(
                (
                    float(center[0]),
                    float(center[1]),
                    float(size[0]),
                    float(size[1]),
                )
            )
        self.assertEqual(
            measured,
            {
                (-200.0, 5965.0, 50.0, 50.0),
                (0.0, 5965.0, 50.0, 50.0),
                (200.0, 5965.0, 50.0, 50.0),
                (-200.0, -5965.0, 50.0, 50.0),
                (0.0, -5965.0, 50.0, 50.0),
                (200.0, -5965.0, 50.0, 50.0),
                (-6965.0, -200.0, 50.0, 50.0),
                (-6965.0, 0.0, 50.0, 50.0),
                (-6965.0, 200.0, 50.0, 50.0),
                (6965.0, -200.0, 50.0, 50.0),
                (6965.0, 0.0, 50.0, 50.0),
                (6965.0, 200.0, 50.0, 50.0),
            },
        )

    def test_marker_ports_follow_frame_size_and_clearance(self) -> None:
        ports = component_local_ports(
            chip_outline(
                {
                    "width": 1000.0,
                    "height": 800.0,
                    "line_width": 20.0,
                    "ebeam_global_marker_size": 60.0,
                    "ebeam_global_marker_edge_clearance": 15.0,
                    "ebeam_global_marker_count": 1,
                }
            )
        )
        self.assertEqual(ports["ebeam_global_top"]["center"], (0.0, 335.0))
        self.assertEqual(ports["ebeam_global_bottom"]["center"], (0.0, -335.0))
        self.assertEqual(ports["ebeam_global_left"]["center"], (-435.0, 0.0))
        self.assertEqual(ports["ebeam_global_right"]["center"], (435.0, 0.0))

    def test_three_marker_ports_are_centered_and_spaced_per_side(self) -> None:
        ports = component_local_ports(chip_outline())
        self.assertEqual(ports["ebeam_global_top_1"]["center"], (-200.0, 5965.0))
        self.assertEqual(ports["ebeam_global_top_2"]["center"], (0.0, 5965.0))
        self.assertEqual(ports["ebeam_global_top_3"]["center"], (200.0, 5965.0))
        self.assertEqual(ports["ebeam_global_left_1"]["center"], (-6965.0, -200.0))
        self.assertEqual(ports["ebeam_global_left_3"]["center"], (-6965.0, 200.0))

    def test_marker_layer_is_always_g_markers(self) -> None:
        component = chip_outline(
            {
                "ebeam_global_marker_layer": 99,
                "ebeam_global_marker_datatype": 7,
            }
        )
        _canonicalize_component_layers(component)
        self.assertEqual(
            component["params"]["ebeam_global_marker_layer"], GLOBAL_MARKER_LAYER
        )
        self.assertEqual(component["params"]["ebeam_global_marker_datatype"], 0)
        self.assertEqual(LAYER_NAME_MAP[GLOBAL_MARKER_LAYER], "G-markers")

    def test_markers_must_fit_inside_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not fit"):
            component_geometry_arrays(
                chip_outline(
                    {
                        "width": 100.0,
                        "height": 100.0,
                        "line_width": 10.0,
                        "ebeam_global_marker_size": 80.0,
                    }
                )
            )

    def test_dicing_square_is_canonicalized_to_global_marker_layer(self) -> None:
        component = {
            "uid": 2,
            "kind": "Square mark",
            "x": 100.0,
            "y": 200.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "params": {
                "size": 10.0,
                "line_width": 1.0,
                "filled": 1,
                "dicing_marker": True,
                "layer": 3,
                "datatype": 7,
            },
        }
        _canonicalize_component_layers(component)
        self.assertEqual(component["params"]["layer"], GLOBAL_MARKER_LAYER)
        self.assertEqual(component["params"]["datatype"], 0)
        polygons, _labels = component_geometry_arrays(component)
        self.assertEqual(len(polygons), 1)
        points, layer, datatype = polygons[0]
        self.assertEqual((layer, datatype), (GLOBAL_MARKER_LAYER, 0))
        np.testing.assert_allclose(np.ptp(points, axis=0), [10.0, 10.0])


if __name__ == "__main__":
    unittest.main()
