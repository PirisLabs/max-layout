from __future__ import annotations

import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES, RF_COMPONENT_KINDS
from max_layout.gds.build import component_geometry_arrays
from max_layout.params import resize_component_parameters
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


KIND = "modulator_RF2"
BRIDGE_KIND = "modulator_RF2_bridge"


def component(parameters: dict | None = None, kind: str = KIND) -> dict:
    return {
        "uid": 1,
        "kind": kind,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": safe_json_copy(parameters or DEFAULT_COMPONENT_VALUES[kind]),
    }


class ModulatorRF2Tests(unittest.TestCase):
    def test_component_is_available_in_rf_library_with_requested_defaults(self) -> None:
        self.assertIn(KIND, RF_COMPONENT_KINDS)
        self.assertEqual(
            DEFAULT_COMPONENT_VALUES[KIND],
            {
                "length": 2000.0,
                "signal_width": 130.0,
                "ground_width": 130.0,
                "gap": 5.0,
                "layer": 4,
                "datatype": 0,
            },
        )

    def test_bridge_variant_is_a_separate_rf_library_component(self) -> None:
        self.assertIn(BRIDGE_KIND, RF_COMPONENT_KINDS)
        self.assertEqual(
            DEFAULT_COMPONENT_VALUES[BRIDGE_KIND],
            {
                "length": 2000.0,
                "signal_width": 130.0,
                "ground_width": 130.0,
                "gap": 5.0,
                "microstrip_separation": 100.0,
                "center_strip_width": 10.0,
                "outer_gnd_pad_width": 130.0,
                "layer": 4,
                "datatype": 0,
            },
        )

    def test_two_parallel_strips_have_no_center_bridge(self) -> None:
        polygons, labels = component_geometry_arrays(component())
        self.assertEqual(labels, [])
        self.assertEqual(len(polygons), 2)
        self.assertEqual({(layer, datatype) for _, layer, datatype in polygons}, {(4, 0)})

        bounds = []
        for points, _layer, _datatype in polygons:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            bounds.append(tuple(float(value) for value in (*minimum, *maximum)))
        self.assertEqual(
            bounds,
            [
                (0.0, 2.5, 2000.0, 132.5),
                (0.0, -132.5, 2000.0, -2.5),
            ],
        )

    def test_bridge_variant_has_two_microstrip_pairs_and_joins_inner_conductors(self) -> None:
        polygons, labels = component_geometry_arrays(component(kind=BRIDGE_KIND))
        self.assertEqual(labels, [])
        self.assertEqual(len(polygons), 7)
        self.assertEqual({(layer, datatype) for _, layer, datatype in polygons}, {(4, 0)})

        bounds = []
        for points, _layer, _datatype in polygons:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            bounds.append(tuple(float(value) for value in (*minimum, *maximum)))
        self.assertEqual(
            bounds,
            [
                (0.0, 185.0, 2000.0, 315.0),
                (0.0, 50.0, 2000.0, 180.0),
                (0.0, -180.0, 2000.0, -50.0),
                (0.0, -315.0, 2000.0, -185.0),
                (995.0, -50.0, 1005.0, 50.0),
                (995.0, 315.0, 1005.0, 445.0),
                (995.0, -445.0, 1005.0, -315.0),
            ],
        )

        bridge_minimum = np.min(polygons[4][0], axis=0)
        bridge_maximum = np.max(polygons[4][0], axis=0)
        bottom_inner_maximum = np.max(polygons[2][0], axis=0)
        top_inner_minimum = np.min(polygons[1][0], axis=0)
        self.assertEqual(float(bridge_minimum[1]), float(bottom_inner_maximum[1]))
        self.assertEqual(float(bridge_maximum[1]), float(top_inner_minimum[1]))
        top_outer_maximum = np.max(polygons[0][0], axis=0)
        top_pad_minimum = np.min(polygons[5][0], axis=0)
        bottom_outer_minimum = np.min(polygons[3][0], axis=0)
        bottom_pad_maximum = np.max(polygons[6][0], axis=0)
        self.assertEqual(float(top_pad_minimum[1]), float(top_outer_maximum[1]))
        self.assertEqual(float(bottom_pad_maximum[1]), float(bottom_outer_minimum[1]))

    def test_bridge_variant_crosses_saved_signal_and_ground_widths_for_geometry(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[BRIDGE_KIND])
        parameters.update({"signal_width": 80.0, "ground_width": 140.0})
        polygons, _labels = component_geometry_arrays(component(parameters, BRIDGE_KIND))

        bounds = []
        for points, _layer, _datatype in polygons:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            bounds.append(tuple(float(value) for value in (*minimum, *maximum)))
        self.assertEqual(
            bounds,
            [
                (0.0, 135.0, 2000.0, 275.0),
                (0.0, 50.0, 2000.0, 130.0),
                (0.0, -190.0, 2000.0, -50.0),
                (0.0, -275.0, 2000.0, -195.0),
                (995.0, -50.0, 1005.0, 50.0),
                (995.0, 275.0, 1005.0, 405.0),
                (995.0, -405.0, 1005.0, -275.0),
            ],
        )

        ports = component_local_ports(component(parameters, BRIDGE_KIND))
        self.assertEqual(ports["top_signal_left"]["center"], (0.0, 205.0))
        self.assertEqual(ports["top_ground_left"]["center"], (0.0, 90.0))
        self.assertEqual(ports["bottom_signal_left"]["center"], (0.0, -120.0))
        self.assertEqual(ports["bottom_ground_left"]["center"], (0.0, -235.0))
        self.assertEqual(ports["top_gap_left"]["center"], (0.0, 132.5))
        self.assertEqual(ports["bottom_gap_left"]["center"], (0.0, -192.5))

    def test_rf_and_alignment_ports_describe_both_separate_strips(self) -> None:
        ports = component_local_ports(component())
        self.assertEqual(ports["signal_left"]["center"], (0.0, 67.5))
        self.assertEqual(ports["signal_right"]["center"], (2000.0, 67.5))
        self.assertEqual(ports["ground_left"]["center"], (0.0, -67.5))
        self.assertEqual(ports["ground_right"]["center"], (2000.0, -67.5))
        self.assertEqual(ports["gap_center_left"]["center"], (0.0, 0.0))
        self.assertEqual(ports["gap_center_right"]["center"], (2000.0, 0.0))
        self.assertNotIn("bridge_center", ports)
        self.assertEqual(ports["center"]["center"], (1000.0, 0.0))

        bridge_ports = component_local_ports(component(kind=BRIDGE_KIND))
        self.assertEqual(bridge_ports["top_signal_left"]["center"], (0.0, 250.0))
        self.assertEqual(bridge_ports["top_ground_left"]["center"], (0.0, 115.0))
        self.assertEqual(bridge_ports["bottom_signal_left"]["center"], (0.0, -115.0))
        self.assertEqual(bridge_ports["bottom_ground_left"]["center"], (0.0, -250.0))
        self.assertEqual(bridge_ports["top_gap_left"]["center"], (0.0, 182.5))
        self.assertEqual(bridge_ports["bottom_gap_left"]["center"], (0.0, -182.5))
        self.assertEqual(bridge_ports["bridge_center"]["center"], (1000.0, 0.0))
        self.assertEqual(bridge_ports["top_outer_gnd_pad_center"]["center"], (1000.0, 380.0))
        self.assertEqual(bridge_ports["bottom_outer_gnd_pad_center"]["center"], (1000.0, -380.0))
        self.assertEqual(bridge_ports["center"]["center"], (1000.0, 0.0))

    def test_invalid_dimensions_are_rejected(self) -> None:
        for update in (
            {"length": 0.0},
            {"signal_width": 0.0},
            {"ground_width": 0.0},
            {"gap": -0.1},
        ):
            parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[KIND])
            parameters.update(update)
            with self.assertRaisesRegex(ValueError, "modulator_RF2"):
                component_geometry_arrays(component(parameters))

        for update in (
            {"center_strip_width": 0.0},
            {"center_strip_width": 2001.0},
            {"outer_gnd_pad_width": 0.0},
            {"microstrip_separation": -0.1},
        ):
            parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[BRIDGE_KIND])
            parameters.update(update)
            with self.assertRaisesRegex(ValueError, "modulator_RF2_bridge"):
                component_geometry_arrays(component(parameters, BRIDGE_KIND))

    def test_editor_resize_scales_length_and_transverse_dimensions(self) -> None:
        resized = resize_component_parameters(
            KIND, DEFAULT_COMPONENT_VALUES[KIND], scale_x=2.0, scale_y=0.5
        )
        self.assertEqual(resized["length"], 4000.0)
        self.assertEqual(resized["signal_width"], 65.0)
        self.assertEqual(resized["ground_width"], 65.0)
        self.assertEqual(resized["gap"], 2.5)

        bridge_resized = resize_component_parameters(
            BRIDGE_KIND,
            DEFAULT_COMPONENT_VALUES[BRIDGE_KIND],
            scale_x=2.0,
            scale_y=0.5,
        )
        self.assertEqual(bridge_resized["length"], 4000.0)
        self.assertEqual(bridge_resized["center_strip_width"], 20.0)
        self.assertEqual(bridge_resized["signal_width"], 65.0)
        self.assertEqual(bridge_resized["ground_width"], 65.0)
        self.assertEqual(bridge_resized["gap"], 2.5)
        self.assertEqual(bridge_resized["microstrip_separation"], 50.0)
        self.assertEqual(bridge_resized["outer_gnd_pad_width"], 65.0)


if __name__ == "__main__":
    unittest.main()
