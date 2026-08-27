from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QCheckBox, QFormLayout

from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    LAYER_NAME_MAP,
    MODULATOR_COMPONENT_KINDS,
    PHOTONIC_COMPONENT_KINDS,
    WG_TIP_LAYER,
    edge_coupler_local_marker_layer,
    edge_coupler_slab_layer,
)
from max_layout.gds.build import component_geometry_arrays
from max_layout.geometry.edge_coupler import assign_edge_coupler_layers
from max_layout.geometry.modulator import resolve_modulator_layout
from max_layout.ports import component_global_ports, component_local_ports
from max_layout.ui.window import NativeLayoutWindow
from max_layout.utils import safe_json_copy


CONTINUOUS = "MZI + CPW modulator"
SEGMENTED = "MZI + segmented-electrode modulator"


class _Factory:
    make_component = NativeLayoutWindow.make_component
    automatic_simulation_companions = (
        NativeLayoutWindow.automatic_simulation_companions
    )

    def __init__(self) -> None:
        self.components: list[dict] = []
        self.next_uid = 1


def component(kind: str, parameters: dict | None = None) -> dict:
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


class ModulatorBlockTests(unittest.TestCase):
    def assert_point_close(self, actual, expected, places: int = 9) -> None:
        self.assertAlmostEqual(float(actual[0]), float(expected[0]), places=places)
        self.assertAlmostEqual(float(actual[1]), float(expected[1]), places=places)

    def test_both_modulators_are_photonic_library_blocks_with_major_keys_first(self) -> None:
        self.assertEqual(MODULATOR_COMPONENT_KINDS, {CONTINUOUS, SEGMENTED})
        for kind in MODULATOR_COMPONENT_KINDS:
            self.assertIn(kind, PHOTONIC_COMPONENT_KINDS)
            self.assertEqual(
                list(DEFAULT_COMPONENT_VALUES[kind])[:6],
                [
                    "initial_gap",
                    "final_gap",
                    "signal_width",
                    "taper_length",
                    "profile",
                    "mzi_length",
                ],
            )
            self.assertFalse(DEFAULT_COMPONENT_VALUES[kind]["add_grating_couplers"])

    def test_reference_center_electrode_lengths_are_preserved(self) -> None:
        continuous = resolve_modulator_layout(
            CONTINUOUS, DEFAULT_COMPONENT_VALUES[CONTINUOUS]
        )
        segmented = resolve_modulator_layout(
            SEGMENTED, DEFAULT_COMPONENT_VALUES[SEGMENTED]
        )
        self.assertEqual(continuous["electrode_length"], 11500.0)
        self.assertEqual(segmented["electrode_length"], 11480.0)

    def test_gap_is_shared_by_mzi_electrode_bends_and_taper_starts(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[kind])
            parameters.update(
                {
                    "initial_gap": 7.5,
                    "final_gap": 18.0,
                    "signal_width": 130.0,
                    "taper_length": 725.0,
                    "profile": "exponential",
                }
            )
            layout = resolve_modulator_layout(kind, parameters)
            self.assertEqual(layout["arm_separation"], 137.5)
            self.assertEqual(layout["mzi_parameters"]["arm_separation"], 137.5)
            self.assertEqual(layout["bend_parameters"]["gap"], 7.5)
            self.assertEqual(layout["bend_parameters"]["layer"], 9)
            self.assertEqual(layout["taper_parameters"]["initial_gap"], 7.5)
            self.assertEqual(layout["taper_parameters"]["final_gap"], 18.0)
            self.assertEqual(layout["taper_parameters"]["length"], 725.0)
            self.assertEqual(layout["taper_parameters"]["profile"], "exponential")
            self.assertEqual(layout["taper_parameters"]["layer"], 9)
            electrode = layout["electrode_parameters"]
            self.assertEqual(electrode["layer"], 4)
            if kind == CONTINUOUS:
                self.assertEqual(electrode["gap"], 7.5)
            else:
                self.assertEqual(electrode["gap"], 7.5)
                self.assertEqual(electrode["end_gap"], 7.5)

    def test_rf_routes_are_exact_point_symmetric(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            layout = resolve_modulator_layout(kind, DEFAULT_COMPONENT_VALUES[kind])
            center_x = float(layout["electrode_center"][0])
            paired_points = (
                (layout["left_bend_start"], layout["right_bend_start"]),
                (layout["left_bend_end"], layout["right_bend_end"]),
                (layout["left_rf_external"], layout["right_rf_external"]),
                (layout["optical_left"], layout["optical_right"]),
                (layout["optical_active_start"], layout["optical_active_end"]),
            )
            for left, right in paired_points:
                self.assertAlmostEqual(float(left[0]) + float(right[0]), 2.0 * center_x)
                self.assertAlmostEqual(float(left[1]) + float(right[1]), 0.0)

    def test_bends_overlap_electrode_horizontally_and_tapers_snap_exactly(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            layout = resolve_modulator_layout(kind, DEFAULT_COMPONENT_VALUES[kind])
            overlap = float(layout["electrode_bend_overlap"])
            self.assertEqual(overlap, 2.0)
            electrode = component(
                str(layout["electrode_kind"]), layout["electrode_parameters"]
            )
            electrode_ports = component_global_ports(electrode)

            left_bend = component("CPW bend", layout["bend_parameters"])
            left_bend.update(
                {
                    "x": float(layout["left_bend_start"][0]),
                    "y": float(layout["left_bend_start"][1]),
                    "orientation_deg": 180.0,
                }
            )
            left_bend_ports = component_global_ports(left_bend)
            self.assertAlmostEqual(
                left_bend_ports["signal_start"]["center"][0]
                - electrode_ports["signal_left"]["center"][0],
                overlap,
            )
            self.assertAlmostEqual(
                left_bend_ports["signal_start"]["center"][1],
                electrode_ports["signal_left"]["center"][1],
            )

            right_bend = component("CPW bend", layout["bend_parameters"])
            right_bend.update(
                {
                    "x": float(layout["right_bend_start"][0]),
                    "y": float(layout["right_bend_start"][1]),
                    "orientation_deg": 0.0,
                }
            )
            right_bend_ports = component_global_ports(right_bend)
            self.assertAlmostEqual(
                electrode_ports["signal_right"]["center"][0]
                - right_bend_ports["signal_start"]["center"][0],
                overlap,
            )
            self.assertAlmostEqual(
                right_bend_ports["signal_start"]["center"][1],
                electrode_ports["signal_right"]["center"][1],
            )

            for side, bend_ports in (
                ("left", left_bend_ports),
                ("right", right_bend_ports),
            ):
                taper = component("Tapered CPW", layout["taper_parameters"])
                taper.update(
                    {
                        "x": float(layout[f"{side}_bend_end"][0]),
                        "y": float(layout[f"{side}_bend_end"][1]),
                        "orientation_deg": float(layout[f"{side}_taper_angle"]),
                    }
                )
                taper_ports = component_global_ports(taper)
                self.assert_point_close(
                    bend_ports["signal_end"]["center"],
                    taper_ports["signal_left"]["center"],
                )

    def test_external_ports_follow_taper_final_gap(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[CONTINUOUS])
        parameters["final_gap"] = 22.0
        ports = component_local_ports(component(CONTINUOUS, parameters))
        for side in ("left", "right"):
            upper = ports[f"upper_gap_{side}"]["center"]
            lower = ports[f"lower_gap_{side}"]["center"]
            separation = math.dist(upper, lower)
            self.assertAlmostEqual(separation, parameters["signal_width"] + 22.0)

    def test_optional_edge_couplers_extend_optical_ports_to_facet_width(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[CONTINUOUS])
        parameters.update(
            {
                "add_edge_coupler_tapers": True,
                "edge_coupler_tip_width": 0.3,
                "left_edge_coupler_taper_length": 115.0,
                "right_edge_coupler_taper_length": 615.0,
            }
        )
        layout = resolve_modulator_layout(CONTINUOUS, parameters)
        self.assertEqual(layout["left_edge_taper_parameters"]["width_start"], 1.2)
        self.assertEqual(layout["left_edge_taper_parameters"]["width_end"], 0.3)
        self.assertEqual(layout["right_edge_taper_parameters"]["width_start"], 1.2)
        self.assertEqual(layout["right_edge_taper_parameters"]["width_end"], 0.3)
        self.assertAlmostEqual(
            layout["optical_mzi_left"][0] - layout["optical_left"][0], 115.0
        )
        self.assertAlmostEqual(
            layout["optical_right"][0] - layout["optical_mzi_right"][0], 615.0
        )
        ports = component_local_ports(component(CONTINUOUS, parameters))
        self.assert_point_close(ports["optical_left"]["center"], layout["optical_left"])
        self.assert_point_close(ports["optical_right"]["center"], layout["optical_right"])

    def test_edge_coupler_choice_attaches_two_complete_numbered_couplers(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            for profile in ("exponential", "linear"):
                parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[kind])
                parameters.update(
                    {
                        "add_edge_coupler_tapers": True,
                        "edge_coupler_profile": profile,
                    }
                )
                modulator = component(kind, parameters)
                assign_edge_coupler_layers([modulator])
                self.assertEqual(modulator["edge_coupler_numbers"], [1, 2])
                self.assertEqual(
                    modulator["params"]["left_edge_coupler_slab_layer"],
                    edge_coupler_slab_layer(1),
                )
                self.assertEqual(
                    modulator["params"]["right_edge_coupler_slab_layer"],
                    edge_coupler_slab_layer(2),
                )
                polygons, _labels = component_geometry_arrays(modulator)
                layers = {layer for _points, layer, _datatype in polygons}
                self.assertTrue(
                    {
                        edge_coupler_slab_layer(1),
                        edge_coupler_local_marker_layer(1),
                        edge_coupler_slab_layer(2),
                        edge_coupler_local_marker_layer(2),
                    }.issubset(layers)
                )
                self.assertIn(WG_TIP_LAYER, layers)
                layout = resolve_modulator_layout(kind, modulator["params"])
                self.assertEqual(layout["mzi_parameters"]["layer"], 1)
                self.assertEqual(
                    layout["left_edge_coupler_parameters"]["layer"],
                    WG_TIP_LAYER,
                )
                self.assertEqual(
                    layout["edge_coupler_kind"],
                    "Linear edge coupler"
                    if profile == "linear"
                    else "Exponential edge coupler",
                )
                ports = component_local_ports(modulator)
                marker_offset = parameters[
                    "edge_coupler_local_marker_vertical_offset"
                ]
                self.assertEqual(
                    ports["left_local_alignment_top"]["center"],
                    (layout["optical_left"][0], marker_offset),
                )
                self.assertEqual(
                    ports["right_local_alignment_bottom"]["center"],
                    (layout["optical_right"][0], -marker_offset),
                )

    def test_edge_and_grating_couplers_cannot_both_be_enabled(self) -> None:
        parameters = safe_json_copy(DEFAULT_COMPONENT_VALUES[CONTINUOUS])
        parameters["add_edge_coupler_tapers"] = True
        parameters["add_grating_couplers"] = True
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            resolve_modulator_layout(CONTINUOUS, parameters)

    def test_automatic_optical_ports_are_snapped_to_rotated_modulator_endpoints(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            factory = _Factory()
            modulator = factory.make_component(kind, 123.0, -456.0)
            modulator["orientation_deg"] = 37.0
            modulator["mirrored"] = True
            factory.components.append(modulator)
            expected = component_global_ports(modulator)
            companions = factory.automatic_simulation_companions(modulator)
            self.assertEqual(len(companions), 2)
            for companion in companions:
                port_name = companion["simulation_parent_port"]
                self.assertIn(port_name, {"optical_left", "optical_right"})
                self.assertAlmostEqual(companion["x"], expected[port_name]["center"][0])
                self.assertAlmostEqual(companion["y"], expected[port_name]["center"][1])
                self.assertAlmostEqual(
                    companion["orientation_deg"] % 360.0,
                    expected[port_name]["outward_orientation_deg"] % 360.0,
                )

    def test_default_geometry_separates_center_rf_from_rf_bends(self) -> None:
        for kind in MODULATOR_COMPONENT_KINDS:
            polygons, _labels = component_geometry_arrays(component(kind))
            layers = {layer for _points, layer, _datatype in polygons}
            self.assertIn(1, layers)
            self.assertIn(4, layers)
            self.assertIn(9, layers)
            self.assertNotIn(2, layers)

    def test_custom_layers_are_named(self) -> None:
        self.assertEqual(LAYER_NAME_MAP[7], "Wg_tip")
        self.assertEqual(LAYER_NAME_MAP[8], "SLAB")
        self.assertEqual(LAYER_NAME_MAP[9], "RFbends")


class ModulatorPropertyPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = NativeLayoutWindow()
        self.modulator = self.window.make_component(CONTINUOUS, 0.0, 0.0)
        self.window.components = [self.modulator]

    def tearDown(self) -> None:
        self.window.close()

    def test_major_linked_controls_are_the_first_editable_rows(self) -> None:
        self.window.show_component_properties(self.modulator)
        labels: list[str] = []
        for row in range(self.window.properties_form.rowCount()):
            item = self.window.properties_form.itemAt(
                row, QFormLayout.ItemRole.LabelRole
            )
            widget = item.widget() if item is not None else None
            labels.append(widget.text() if widget is not None else "")
        self.assertEqual(
            labels[2:8],
            [
                "Interaction / bend / taper-start gap (µm)",
                "External taper-end gap (µm)",
                "RF signal width (µm)",
                "Symmetric taper length (µm)",
                "Symmetric taper form",
                "MZI straight-arm length (µm)",
            ],
        )
        self.assertEqual(labels[8], "Horizontal CPW-to-bend overlap (µm)")

    def test_edge_and_grating_checkboxes_are_front_loaded_and_exclusive(self) -> None:
        self.window.show_component_properties(self.modulator)
        edge_widget = self.window.parameter_widgets[
            "params.add_edge_coupler_tapers"
        ][0]
        grating_widget = self.window.parameter_widgets[
            "params.add_grating_couplers"
        ][0]
        self.assertIsInstance(edge_widget, QCheckBox)
        self.assertIsInstance(grating_widget, QCheckBox)
        edge_widget.setChecked(True)
        self.assertTrue(edge_widget.isChecked())
        self.assertFalse(grating_widget.isChecked())
        grating_widget.setChecked(True)
        self.assertTrue(grating_widget.isChecked())
        self.assertFalse(edge_widget.isChecked())

        labels = []
        for row in range(self.window.properties_form.rowCount()):
            item = self.window.properties_form.itemAt(
                row, QFormLayout.ItemRole.LabelRole
            )
            widget = item.widget() if item is not None else None
            labels.append(widget.text() if widget is not None else "")
        self.assertLess(
            labels.index("Edge coupler (two optical ends)"),
            labels.index("x"),
        )
        self.assertLess(
            labels.index("Grating coupler (two optical ends)"),
            labels.index("x"),
        )

    def test_normal_click_anywhere_opens_the_shared_modulator_page(self) -> None:
        calls: list[dict] = []
        self.window.show_component_properties = calls.append
        fake_item = type("FakeComponentItem", (), {"uid": self.modulator["uid"]})()
        self.window.show_properties_for_scene_click(fake_item, QPointF())
        self.assertEqual(calls, [self.modulator])


if __name__ == "__main__":
    unittest.main()
