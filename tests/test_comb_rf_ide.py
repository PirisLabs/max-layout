from __future__ import annotations

import unittest

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES, RF_COMPONENT_KINDS
from max_layout.gds.build import component_geometry_arrays
from max_layout.geometry.comb_rf_ide import COMB_RF_IDE_KIND, resolve_comb_rf_ide
from max_layout.params import resize_component_parameters
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


def component(parameters: dict | None = None) -> dict:
    return {
        "uid": 1,
        "kind": COMB_RF_IDE_KIND,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": safe_json_copy(parameters or DEFAULT_COMPONENT_VALUES[COMB_RF_IDE_KIND]),
    }


class CombRFIDETests(unittest.TestCase):
    def test_component_is_available_with_paper_informed_defaults(self) -> None:
        self.assertIn(COMB_RF_IDE_KIND, RF_COMPONENT_KINDS)
        self.assertEqual(
            DEFAULT_COMPONENT_VALUES[COMB_RF_IDE_KIND],
            {
                "length": 500.0,
                "signal_width": 130.0,
                "gap": 3.0,
                "ground_width": 130.0,
                "finger_width": 3.0,
                "finger_length": 33.5,
                "finger_gap": 2.0,
                "IDE_Offset": 400.0,
                "layer": 4,
                "datatype": 0,
            },
        )

    def test_fingers_pack_top_down_and_alternate_between_signal_buses(self) -> None:
        q = resolve_comb_rf_ide(DEFAULT_COMPONENT_VALUES[COMB_RF_IDE_KIND])
        self.assertEqual(q["finger_count"], 26)
        self.assertEqual(q["finger_rectangles"][0], {"index": 0, "bus": "left", "bounds": (400.0, 62.0, 433.5, 65.0)})
        self.assertEqual(q["finger_rectangles"][1], {"index": 1, "bus": "right", "bounds": (403.0, 57.0, 436.5, 60.0)})
        self.assertEqual(q["finger_rectangles"][-1], {"index": 25, "bus": "right", "bounds": (403.0, -63.0, 436.5, -60.0)})

        for upper, lower in zip(q["finger_rectangles"], q["finger_rectangles"][1:]):
            self.assertAlmostEqual(upper["bounds"][1] - lower["bounds"][3], 2.0)
        self.assertAlmostEqual(q["right_signal_bus"][0] - q["finger_rectangles"][0]["bounds"][2], 3.0)
        self.assertAlmostEqual(q["finger_rectangles"][1]["bounds"][0] - q["left_signal_bus"][2], 3.0)

    def test_geometry_contains_two_buses_two_grounds_and_all_fingers(self) -> None:
        polygons, labels = component_geometry_arrays(component())
        self.assertEqual(labels, [])
        self.assertEqual(len(polygons), 30)
        self.assertEqual({(layer, datatype) for _, layer, datatype in polygons}, {(4, 0)})
        bounds = []
        for points, _layer, _datatype in polygons[:6]:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            bounds.append(tuple(float(v) for v in (*minimum, *maximum)))
        self.assertEqual(
            bounds,
            [
                (0.0, -65.0, 400.0, 65.0),
                (436.5, -65.0, 500.0, 65.0),
                (0.0, 68.0, 500.0, 198.0),
                (0.0, -198.0, 500.0, -68.0),
                (400.0, 62.0, 433.5, 65.0),
                (403.0, 57.0, 436.5, 60.0),
            ],
        )

    def test_ports_expose_cpw_ends_and_ide_buses(self) -> None:
        ports = component_local_ports(component())
        self.assertEqual(ports["signal_left"]["center"], (0.0, 0.0))
        self.assertEqual(ports["signal_right"]["center"], (500.0, 0.0))
        self.assertEqual(ports["ide_left_bus"]["center"], (400.0, 0.0))
        self.assertEqual(ports["ide_right_bus"]["center"], (436.5, 0.0))
        self.assertEqual(ports["ide_center"]["center"], (418.25, 0.0))
        self.assertEqual(ports["upper_ground_left"]["center"], (0.0, 133.0))
        self.assertEqual(ports["lower_ground_right"]["center"], (500.0, -133.0))

    def test_invalid_dimensions_are_rejected(self) -> None:
        for update in (
            {"length": 0.0},
            {"signal_width": 0.0},
            {"ground_width": 0.0},
            {"finger_width": 0.0},
            {"finger_length": 0.0},
            {"gap": -0.1},
            {"finger_gap": -0.1},
            {"IDE_Offset": 0.0},
            {"IDE_Offset": 464.0},
            {"finger_width": 131.0},
        ):
            parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[COMB_RF_IDE_KIND])
            parameters.update(update)
            with self.assertRaisesRegex(ValueError, "comb_RF_IDE"):
                component_geometry_arrays(component(parameters))

    def test_editor_resize_scales_longitudinal_and_transverse_dimensions(self) -> None:
        resized = resize_component_parameters(
            COMB_RF_IDE_KIND,
            DEFAULT_COMPONENT_VALUES[COMB_RF_IDE_KIND],
            scale_x=2.0,
            scale_y=0.5,
        )
        self.assertEqual(resized["length"], 1000.0)
        self.assertEqual(resized["IDE_Offset"], 800.0)
        self.assertEqual(resized["finger_length"], 67.0)
        self.assertEqual(resized["signal_width"], 65.0)
        self.assertEqual(resized["ground_width"], 65.0)
        self.assertEqual(resized["gap"], 1.5)
        self.assertEqual(resized["finger_width"], 1.5)
        self.assertEqual(resized["finger_gap"], 1.0)


if __name__ == "__main__":
    unittest.main()
