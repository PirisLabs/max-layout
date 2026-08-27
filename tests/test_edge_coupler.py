from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QFormLayout

from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    EDGE_COUPLER_COMPONENT_KINDS,
    EXPONENTIAL_EDGE_COUPLER_FEED_KIND,
    LAYER_NAME_MAP,
    LINEAR_EDGE_COUPLER_FEED_KIND,
    LOCAL_MARKER_LAYER,
    PHOTONIC_COMPONENT_KINDS,
    WG_TIP_LAYER,
    edge_coupler_total_count,
    edge_coupler_layer_name_map,
    edge_coupler_local_marker_layer,
    edge_coupler_slab_layer,
)
from max_layout.gds.build import _canonicalize_component_layers, component_geometry_arrays
from max_layout.geometry.edge_coupler import (
    EDGE_COUPLER_COMMON_FRONT_PARAMETERS,
    EDGE_COUPLER_FRONT_PARAMETERS,
    assign_edge_coupler_layers,
    edge_coupler_front_parameters,
    edge_coupler_slab_profile,
    migrate_edge_coupler_parameters,
)
from max_layout.ports import component_local_ports
from max_layout.ui.window import NativeLayoutWindow
from max_layout.utils import safe_json_copy


EXPONENTIAL = "Exponential edge coupler"
LINEAR = "Linear edge coupler"
EXPONENTIAL_FEED = EXPONENTIAL_EDGE_COUPLER_FEED_KIND
LINEAR_FEED = LINEAR_EDGE_COUPLER_FEED_KIND


def edge_coupler(kind: str, parameters: dict | None = None) -> dict:
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


def endpoint_width(points: np.ndarray, x: float) -> float:
    y_values = points[np.isclose(points[:, 0], x, atol=1e-9), 1]
    if not len(y_values):
        raise AssertionError(f"No polygon vertices found at x={x}")
    return float(y_values.max() - y_values.min())


class EdgeCouplerGeometryTests(unittest.TestCase):
    def test_intuitive_defaults_and_library_membership(self) -> None:
        exponential = DEFAULT_COMPONENT_VALUES[EXPONENTIAL]
        linear = DEFAULT_COMPONENT_VALUES[LINEAR]
        self.assertEqual(tuple(exponential)[:11], EDGE_COUPLER_FRONT_PARAMETERS)
        self.assertEqual(tuple(linear)[:10], EDGE_COUPLER_COMMON_FRONT_PARAMETERS)
        self.assertEqual(exponential["top_taper_length"], 10.0)
        self.assertEqual(exponential["top_start_offset"], 10.0)
        self.assertEqual(exponential["narrow_slab_straight_length"], 100.0)
        self.assertEqual(exponential["wide_slab_straight_length"], 10.0)
        self.assertEqual(exponential["slab_start_width"], 0.5)
        self.assertEqual(exponential["slab_end_width"], 5.0)
        self.assertEqual(exponential["top_tip_width"], 0.2)
        self.assertEqual(exponential["top_end_width"], 1.2)
        self.assertEqual(exponential["slab_exponential_factor"], 1.3)
        self.assertEqual(exponential["local_alignment_mark_size"], 50.0)
        self.assertEqual(
            exponential["local_alignment_mark_vertical_offset"], 100.0
        )
        self.assertEqual(exponential["local_alignment_mark_name"], "L1")
        self.assertEqual(
            exponential["local_alignment_mark_layer"], LOCAL_MARKER_LAYER
        )
        self.assertNotIn("slab_exponential_factor", linear)
        self.assertIn(EXPONENTIAL, PHOTONIC_COMPONENT_KINDS)
        self.assertIn(LINEAR, PHOTONIC_COMPONENT_KINDS)
        self.assertEqual(LAYER_NAME_MAP[1], "WG")
        self.assertEqual(LAYER_NAME_MAP[WG_TIP_LAYER], "Wg_tip")
        self.assertEqual(LAYER_NAME_MAP[8], "SLAB")
        self.assertEqual(LAYER_NAME_MAP[LOCAL_MARKER_LAYER], "L-markers")
        self.assertEqual(exponential["layer"], WG_TIP_LAYER)
        self.assertEqual(linear["layer"], WG_TIP_LAYER)
        dynamic_names = edge_coupler_layer_name_map(2)
        self.assertEqual(dynamic_names[edge_coupler_slab_layer(1)], "SLAB 1")
        self.assertEqual(dynamic_names[edge_coupler_slab_layer(2)], "SLAB 2")
        self.assertNotIn(LOCAL_MARKER_LAYER, dynamic_names)
        self.assertEqual(edge_coupler_local_marker_layer(1), LOCAL_MARKER_LAYER)
        self.assertEqual(edge_coupler_local_marker_layer(2), LOCAL_MARKER_LAYER)

    def test_exponential_slab_spans_offset_plus_top_taper(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[EXPONENTIAL])
        parameters.update({"top_start_offset": 20.0, "top_taper_length": 10.0})
        polygons, _labels = component_geometry_arrays(
            edge_coupler(EXPONENTIAL, parameters)
        )
        self.assertEqual(
            {layer for _points, layer, _datatype in polygons},
            {WG_TIP_LAYER, edge_coupler_slab_layer(1), LOCAL_MARKER_LAYER},
        )
        slab_layer = edge_coupler_slab_layer(1)
        by_layer = {
            layer: np.vstack(
                [points for points, polygon_layer, _datatype in polygons if polygon_layer == layer]
            )
            for layer in (WG_TIP_LAYER, slab_layer)
        }
        # Narrow straight: 0..100; taper: 100..130; wide straight: 130..140.
        # The top WG remains confined to L1..L1+L2 inside the taper.
        self.assertAlmostEqual(float(by_layer[slab_layer][:, 0].min()), 0.0)
        self.assertAlmostEqual(float(by_layer[slab_layer][:, 0].max()), 140.0)
        self.assertAlmostEqual(float(by_layer[WG_TIP_LAYER][:, 0].min()), 120.0)
        self.assertAlmostEqual(float(by_layer[WG_TIP_LAYER][:, 0].max()), 130.0)
        self.assertAlmostEqual(endpoint_width(by_layer[slab_layer], 0.0), 0.5)
        self.assertAlmostEqual(endpoint_width(by_layer[slab_layer], 100.0), 0.5)
        self.assertAlmostEqual(endpoint_width(by_layer[slab_layer], 130.0), 5.0)
        self.assertAlmostEqual(endpoint_width(by_layer[slab_layer], 140.0), 5.0)
        self.assertAlmostEqual(endpoint_width(by_layer[WG_TIP_LAYER], 120.0), 0.2)
        self.assertAlmostEqual(endpoint_width(by_layer[WG_TIP_LAYER], 130.0), 1.2)
        local_marks = [
            points
            for points, layer, _datatype in polygons
            if layer == LOCAL_MARKER_LAYER
        ]
        self.assertEqual(len(local_marks), 2)
        centers = sorted(
            (
                float((points[:, 0].min() + points[:, 0].max()) / 2.0),
                float((points[:, 1].min() + points[:, 1].max()) / 2.0),
                float(points[:, 0].max() - points[:, 0].min()),
                float(points[:, 1].max() - points[:, 1].min()),
            )
            for points in local_marks
        )
        self.assertEqual(centers, [(120.0, -100.0, 50.0, 50.0), (120.0, 100.0, 50.0, 50.0)])

    def test_exponential_and_linear_slab_profiles_differ_only_in_shape(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[EXPONENTIAL])
        parameters.update(
            {"top_start_offset": 20.0, "top_taper_length": 10.0, "points": 101}
        )
        exp_x, exp_width = edge_coupler_slab_profile(parameters, EXPONENTIAL)
        linear_x, linear_width = edge_coupler_slab_profile(parameters, LINEAR)
        self.assertAlmostEqual(exp_x[-1], 30.0)
        np.testing.assert_allclose(exp_x, linear_x)
        expected_exp_mid = 0.5 * (5.0 / 0.5) ** (0.5**1.3)
        self.assertAlmostEqual(exp_width[50], expected_exp_mid)
        self.assertAlmostEqual(linear_width[50], 2.75)
        self.assertNotAlmostEqual(exp_width[50], linear_width[50])

    def test_ports_match_l1_and_l2_landmarks(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[EXPONENTIAL])
        parameters.update({"top_start_offset": 20.0, "top_taper_length": 10.0})
        ports = component_local_ports(edge_coupler(EXPONENTIAL, parameters))
        self.assertEqual(ports["slab_start"]["center"], (0, 0))
        self.assertEqual(ports["slab_taper_start"]["center"], (100.0, 0))
        self.assertEqual(ports["tip"]["center"], (120.0, 0))
        self.assertEqual(ports["slab_taper_end"]["center"], (130.0, 0))
        self.assertEqual(ports["slab_end"]["center"], (140.0, 0))
        self.assertEqual(ports["taper_end"]["center"], (130.0, 0))
        self.assertEqual(ports["waveguide_end"]["center"], (130.0, 0))
        self.assertEqual(ports["local_alignment_top"]["center"], (120.0, 100.0))
        self.assertEqual(ports["local_alignment_bottom"]["center"], (120.0, -100.0))

    def test_legacy_edge_coupler_migrates_to_exponential_variant(self) -> None:
        component = {
            "uid": 1,
            "kind": "Edge coupler",
            "params": {
                "tip_width": 0.25,
                "wg_width": 1.4,
                "taper_length": 250.0,
                "wg_straight_length": 30.0,
                "layer": 99,
                "datatype": 4,
            },
        }
        self.assertTrue(migrate_edge_coupler_parameters(component))
        self.assertEqual(component["kind"], EXPONENTIAL)
        parameters = component["params"]
        self.assertEqual(parameters["top_tip_width"], 0.25)
        self.assertEqual(parameters["top_end_width"], 1.4)
        self.assertEqual(parameters["top_taper_length"], 250.0)
        self.assertEqual(parameters["top_start_offset"], 30.0)
        self.assertEqual(parameters["narrow_slab_straight_length"], 100.0)
        self.assertEqual(parameters["wide_slab_straight_length"], 10.0)
        self.assertNotIn("tip_width", parameters)
        _canonicalize_component_layers(component)
        self.assertEqual(parameters["layer"], WG_TIP_LAYER)
        self.assertEqual(parameters["slab_layer"], edge_coupler_slab_layer(1))

    def test_multiple_edge_couplers_receive_unique_slabs_and_shared_local_layer(self) -> None:
        couplers = [
            {**edge_coupler(EXPONENTIAL), "uid": 30},
            {**edge_coupler(LINEAR), "uid": 10},
            {**edge_coupler(EXPONENTIAL), "uid": 20},
        ]
        numbered = assign_edge_coupler_layers(couplers)
        self.assertEqual([component["uid"] for component in numbered], [10, 20, 30])
        for number, component in enumerate(numbered, start=1):
            self.assertEqual(component["edge_coupler_number"], number)
            self.assertEqual(
                component["params"]["slab_layer"],
                edge_coupler_slab_layer(number),
            )
            self.assertEqual(
                component["params"]["local_alignment_mark_layer"],
                LOCAL_MARKER_LAYER,
            )
            self.assertEqual(
                component["params"]["local_alignment_mark_name"], f"L{number}"
            )

    def test_pair_feed_has_two_edge_couplers_and_a_2500_um_straight(self) -> None:
        for kind in (EXPONENTIAL_FEED, LINEAR_FEED):
            parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[kind])
            front = edge_coupler_front_parameters(kind)
            self.assertEqual(tuple(parameters)[: len(front)], front)
            self.assertEqual(parameters["Feed_length"], 2500.0)
            pair = edge_coupler(kind, parameters)
            self.assertEqual(edge_coupler_total_count([pair]), 2)
            polygons, _labels = component_geometry_arrays(pair)
            self.assertEqual(
                {layer for _points, layer, _datatype in polygons},
                {
                    1,
                    WG_TIP_LAYER,
                    LOCAL_MARKER_LAYER,
                    edge_coupler_slab_layer(1),
                    edge_coupler_slab_layer(2),
                },
            )
            ports = component_local_ports(pair)
            self.assertEqual(ports["left_tip"]["center"], (110.0, 0.0))
            self.assertEqual(ports["feed_left"]["center"], (120.0, 0.0))
            self.assertEqual(ports["feed_right"]["center"], (2620.0, 0.0))
            self.assertEqual(ports["right_tip"]["center"], (2630.0, 0.0))
            self.assertEqual(ports["right_slab_facet"]["center"], (2740.0, 0.0))

    def test_pair_feed_consumes_two_slabs_and_shared_local_layer(self) -> None:
        pair = {**edge_coupler(EXPONENTIAL_FEED), "uid": 10}
        single = {**edge_coupler(EXPONENTIAL), "uid": 20}
        assign_edge_coupler_layers([single, pair])
        self.assertEqual(pair["edge_coupler_numbers"], [1, 2])
        self.assertEqual(pair["params"]["slab_layer"], 12)
        self.assertEqual(pair["params"]["right_slab_layer"], 13)
        self.assertEqual(pair["params"]["local_alignment_mark_layer"], 11)
        self.assertEqual(pair["params"]["right_local_alignment_mark_layer"], 11)
        self.assertEqual(single["edge_coupler_numbers"], [3])
        self.assertEqual(single["params"]["slab_layer"], 14)


class EdgeCouplerPropertyPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = NativeLayoutWindow()

    def tearDown(self) -> None:
        self.window.close()

    def property_labels(self, kind: str) -> list[str]:
        component = self.window.make_component(kind, 0.0, 0.0)
        self.window.components = [component]
        self.window.show_component_properties(component)
        labels: list[str] = []
        for row in range(self.window.properties_form.rowCount()):
            item = self.window.properties_form.itemAt(
                row, QFormLayout.ItemRole.LabelRole
            )
            widget = item.widget() if item is not None else None
            labels.append(widget.text() if widget is not None else "")
        return labels

    def test_exponential_main_controls_are_first(self) -> None:
        labels = self.property_labels(EXPONENTIAL)
        self.assertEqual(
            labels[1:12],
            [
                "Top WG taper length (µm)",
                "Top WG start offset from SLAB taper start (µm)",
                "Narrow-side SLAB straight length (µm)",
                "Wide-side SLAB straight length (µm)",
                "SLAB start width (µm)",
                "SLAB end width (µm)",
                "Top WG tip width (µm)",
                "Top WG end width (µm)",
                "SLAB exponential factor",
                "LOCAL square size (µm)",
                "LOCAL marker offset above/below WG (µm)",
            ],
        )
        self.assertEqual(labels[12:16], ["x", "y", "orientation_deg", "mirrored"])

    def test_linear_main_controls_are_first_without_exponential_factor(self) -> None:
        labels = self.property_labels(LINEAR)
        self.assertEqual(
            labels[1:11],
            [
                "Top WG taper length (µm)",
                "Top WG start offset from SLAB taper start (µm)",
                "Narrow-side SLAB straight length (µm)",
                "Wide-side SLAB straight length (µm)",
                "SLAB start width (µm)",
                "SLAB end width (µm)",
                "Top WG tip width (µm)",
                "Top WG end width (µm)",
                "LOCAL square size (µm)",
                "LOCAL marker offset above/below WG (µm)",
            ],
        )
        self.assertEqual(labels[11:15], ["x", "y", "orientation_deg", "mirrored"])

    def test_pair_feed_leads_with_feed_length(self) -> None:
        for kind in (EXPONENTIAL_FEED, LINEAR_FEED):
            labels = self.property_labels(kind)
            self.assertEqual(labels[1], "Feed length (µm)")

    def test_top_waveguide_layer_is_named_wg_tip(self) -> None:
        labels = self.property_labels(EXPONENTIAL)
        self.assertIn("Top taper layer (Wg_tip)", labels)

    def test_library_has_separate_edge_couplers_category(self) -> None:
        photonic = next(
            self.window.library_tree.topLevelItem(index)
            for index in range(self.window.library_tree.topLevelItemCount())
            if self.window.library_tree.topLevelItem(index).text(0) == "Photonic"
        )
        groups = {
            photonic.child(index).text(0): photonic.child(index)
            for index in range(photonic.childCount())
        }
        self.assertIn("Edge couplers", groups)
        edge_group = groups["Edge couplers"]
        self.assertEqual(
            {edge_group.child(index).text(0) for index in range(edge_group.childCount())},
            {
                "Exponential",
                "Linear",
                "Exponential pair + straight feed",
                "Linear pair + straight feed",
            },
        )
        grating_names = {
            groups["Grating couplers"].child(index).text(0)
            for index in range(groups["Grating couplers"].childCount())
        }
        self.assertNotIn("Exponential", grating_names)
        self.assertNotIn("Linear", grating_names)


if __name__ == "__main__":
    unittest.main()
