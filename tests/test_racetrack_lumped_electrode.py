from __future__ import annotations

import unittest

import gdstk
import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES, PHOTONIC_COMPONENT_KINDS
from max_layout.gds.build import component_geometry_arrays, resolve_and_build
from max_layout.geometry.racetrack_electrode import (
    RACETRACK_LUMPED_ELECTRODE_KIND,
    resolve_racetrack_lumped_electrode,
)
from max_layout.ports import component_local_ports
from max_layout.utils import safe_json_copy


def component(parameters: dict | None = None, mirrored: bool = False) -> dict:
    return {
        "uid": 1,
        "kind": RACETRACK_LUMPED_ELECTRODE_KIND,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": mirrored,
        "attachment": None,
        "params": safe_json_copy(
            parameters or DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        ),
    }


class RacetrackLumpedElectrodeTests(unittest.TestCase):
    def test_component_is_available_with_paper_inspired_defaults(self) -> None:
        self.assertIn(RACETRACK_LUMPED_ELECTRODE_KIND, PHOTONIC_COMPONENT_KINDS)
        defaults = DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        self.assertEqual(defaults["electrode_length_mode"], "fit straight")
        self.assertEqual(defaults["target_frequency_ghz"], 30.925)
        self.assertEqual(defaults["center_connection"], "inner join")

    def test_fitted_electrode_uses_symmetric_straight_clearance(self) -> None:
        layout = resolve_racetrack_lumped_electrode(
            DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        )
        self.assertAlmostEqual(
            layout["electrode_length"],
            layout["straight_length"] - 200.0,
            places=9,
        )
        self.assertAlmostEqual(layout["electrode_x0"], -layout["electrode_x1"], places=12)

    def test_optical_straights_are_centered_in_both_rf_gaps(self) -> None:
        ports = component_local_ports(component())
        self.assertEqual(
            ports["upper_optical_interaction_center"]["center"],
            ports["upper_electrode_gap_center"]["center"],
        )
        self.assertEqual(
            ports["lower_optical_interaction_center"]["center"],
            ports["lower_electrode_gap_center"]["center"],
        )

    def test_geometry_uses_only_wg_and_rf_layers_without_overlap(self) -> None:
        polygons, labels = component_geometry_arrays(component())
        self.assertEqual(labels, [])
        self.assertEqual({layer for _points, layer, _datatype in polygons}, {1, 4})
        optical = [gdstk.Polygon(points) for points, layer, _datatype in polygons if layer == 1]
        rf = [gdstk.Polygon(points) for points, layer, _datatype in polygons if layer == 4]
        overlap = gdstk.boolean(optical, rf, "and", precision=1e-4) or []
        self.assertLessEqual(sum(polygon.area() for polygon in overlap), 1e-8)

    def test_center_bus_touches_inner_conductors_without_area_overlap(self) -> None:
        layout = resolve_racetrack_lumped_electrode(
            DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        )
        polygons = [gdstk.Polygon(points) for points in layout["rf_polygons"]]
        bus = polygons[-1]
        for inner in (polygons[1], polygons[2]):
            overlap = gdstk.boolean([bus], [inner], "and", precision=1e-5) or []
            self.assertLessEqual(sum(polygon.area() for polygon in overlap), 1e-8)
        united = gdstk.boolean(polygons, [], "or", precision=1e-5) or []
        self.assertEqual(len(united), 3)

    def test_manual_electrode_cannot_enter_bends(self) -> None:
        parameters = safe_json_copy(
            DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        )
        parameters.update({"electrode_length_mode": "manual", "electrode_length": 5000.0})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            resolve_racetrack_lumped_electrode(parameters)

    def test_mirroring_swaps_gap_ports_and_preserves_layer_bounds(self) -> None:
        normal_ports = component_local_ports(component())
        mirrored_ports = component_local_ports(component(mirrored=True))
        self.assertEqual(
            normal_ports["upper_electrode_gap_center"]["center"][1],
            -mirrored_ports["upper_electrode_gap_center"]["center"][1],
        )
        normal_polygons, _ = component_geometry_arrays(component())
        mirror_polygons, _ = component_geometry_arrays(component(mirrored=True))
        for layer in (1, 4):
            normal_points = np.vstack([points for points, value, _ in normal_polygons if value == layer])
            mirror_points = np.vstack([points for points, value, _ in mirror_polygons if value == layer])
            self.assertAlmostEqual(float(normal_points[:, 0].min()), float(mirror_points[:, 0].min()))
            self.assertAlmostEqual(float(normal_points[:, 0].max()), float(mirror_points[:, 0].max()))
            self.assertAlmostEqual(float(normal_points[:, 1].min()), -float(mirror_points[:, 1].max()))
            self.assertAlmostEqual(float(normal_points[:, 1].max()), -float(mirror_points[:, 1].min()))

    def test_saved_layers_are_canonicalized(self) -> None:
        parameters = safe_json_copy(
            DEFAULT_COMPONENT_VALUES[RACETRACK_LUMPED_ELECTRODE_KIND]
        )
        parameters.update({"layer": 99, "rf_layer": 98})
        library = resolve_and_build([component(parameters)])
        layers = {
            polygon.layer
            for polygon in library.top_level()[0].get_polygons(
                apply_repetitions=True, include_paths=True
            )
        }
        self.assertEqual(layers, {1, 4})


if __name__ == "__main__":
    unittest.main()
