from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from max_layout.constants import DEFAULT_COMPONENT_VALUES, EBEAM_LAYER
from max_layout.gds.build import (
    _canonicalize_component_layers,
    component_geometry_arrays,
    resolve_and_build,
    test_block_device_placements as photonic_test_block_placements,
)
from max_layout.gds.ebeam import multipass_field_layout
from max_layout.ui import items as canvas_items
from max_layout.ui.window import NativeLayoutWindow


def component(kind: str, uid: int, x: float = 0.0, y: float = 0.0) -> dict:
    return {
        "uid": uid,
        "kind": kind,
        "x": float(x),
        "y": float(y),
        "orientation_deg": 0.0,
        "mirrored": False,
        "params": deepcopy(DEFAULT_COMPONENT_VALUES[kind]),
        "attachment": None,
    }


def layer_polygons(library, layer: int) -> list[np.ndarray]:
    return [
        np.asarray(polygon.points, dtype=float)
        for polygon in library.cells[0].get_polygons(
            apply_repetitions=True, include_paths=True
        )
        if int(polygon.layer) == int(layer)
    ]


class EditorGdsParityTests(unittest.TestCase):
    def test_dense_photonic_test_block_preview_uses_every_export_polygon(self) -> None:
        block = component("Photonic test block", 1, x=123.0, y=-47.0)
        block["orientation_deg"] = 31.0
        block["params"].update(
            {
                "include_ebeam_fields": False,
                "photonic_component_kind": "Grating coupler",
                "photonic_base_params": deepcopy(
                    DEFAULT_COMPONENT_VALUES["Grating coupler"]
                ),
                "sweep_parameters": ["pitch"],
                # More than 500 GC polygons on one layer exercises the old
                # canvas sampling threshold.
                "sweep_ranges": {
                    "pitch": {
                        "values": [0.70 + 0.002 * index for index in range(18)]
                    }
                },
                "grid_columns": 6,
            }
        )

        previous = canvas_items._APPROXIMATE_PREVIEW
        canvas_items._APPROXIMATE_PREVIEW = False
        canvas_items.clear_preview_caches()
        try:
            preview = canvas_items.component_local_polygons(block)
        finally:
            canvas_items._APPROXIMATE_PREVIEW = previous
            canvas_items.clear_preview_caches()

        local = deepcopy(block)
        local.update({"x": 0.0, "y": 0.0, "orientation_deg": 0.0, "attachment": None})
        _canonicalize_component_layers(local)
        exported, _labels = component_geometry_arrays(local)

        self.assertGreater(len(exported), 500)
        self.assertEqual(len(preview), len(exported))
        for preview_polygon, export_polygon in zip(preview, exported):
            self.assertEqual(preview_polygon[1:], export_polygon[1:])
            np.testing.assert_allclose(
                preview_polygon[0], export_polygon[0], rtol=0.0, atol=1e-12
            )


class PhotonicTestBlockEbeamTests(unittest.TestCase):
    def test_primary_sweep_variable_can_advance_along_x_or_y(self) -> None:
        block = component("Photonic test block", 1)
        block["params"].update(
            {
                "photonic_component_kind": "Straight",
                "photonic_base_params": deepcopy(DEFAULT_COMPONENT_VALUES["Straight"]),
                "sweep_parameters": ["length"],
                "sweep_ranges": {"length": {"values": [20.0, 30.0, 40.0]}},
                "primary_sweep_parameter": "length",
                "edge_spacing": 25.0,
            }
        )

        def placement_centers(axis: str) -> list[tuple[float, float]]:
            block["params"]["primary_sweep_axis"] = axis
            centers = []
            for _index, polygons, shift in photonic_test_block_placements(block):
                points = np.vstack([np.asarray(polygon.points, dtype=float) for polygon in polygons])
                low, high = points.min(axis=0) + shift, points.max(axis=0) + shift
                centers.append(tuple((low + high) / 2.0))
            return centers

        x_centers = placement_centers("x")
        y_centers = placement_centers("y")
        self.assertGreater(np.ptp([center[0] for center in x_centers]), 0.0)
        self.assertAlmostEqual(np.ptp([center[1] for center in x_centers]), 0.0)
        self.assertAlmostEqual(np.ptp([center[0] for center in y_centers]), 0.0)
        self.assertGreater(np.ptp([center[1] for center in y_centers]), 0.0)

    def test_beamer_ftext_extracts_all_layers_before_mapping(self) -> None:
        flow = NativeLayoutWindow.beamer_flow_template(None, "", 1.8, 1.8, 1.8)
        import_index = flow.index("NODE Import ()")
        extract_index = flow.index("NODE Extract ()")
        mapping_index = flow.index("NODE Mapping ()")
        self.assertLess(import_index, extract_index)
        self.assertLess(extract_index, mapping_index)
        self.assertIn("LAYERSET = *", flow[extract_index:mapping_index])
        self.assertIn("OUT_PORT[0] = 5, Extract%20Layers, 0", flow)
        self.assertIn("IN_PORT[0] = 5, Extract%20Layers, 0", flow[mapping_index:])

    def test_new_block_has_no_automatic_write_fields(self) -> None:
        self.assertFalse(
            DEFAULT_COMPONENT_VALUES["Photonic test block"]["include_ebeam_fields"]
        )
        block = component("Photonic test block", 1)
        block["params"]["sweep_parameters"] = []
        polygons, _labels = component_geometry_arrays(block)
        self.assertNotIn(EBEAM_LAYER, {layer for _points, layer, _datatype in polygons})

        missing_legacy_key = deepcopy(block)
        missing_legacy_key["params"].pop("include_ebeam_fields")
        polygons, _labels = component_geometry_arrays(missing_legacy_key)
        self.assertNotIn(EBEAM_LAYER, {layer for _points, layer, _datatype in polygons})

    def test_explicit_legacy_true_still_exports_write_fields(self) -> None:
        block = component("Photonic test block", 1)
        block["params"].update(
            {"sweep_parameters": [], "include_ebeam_fields": True}
        )
        polygons, _labels = component_geometry_arrays(block)
        self.assertIn(EBEAM_LAYER, {layer for _points, layer, _datatype in polygons})


class IndependentEbeamMovementTests(unittest.TestCase):
    def test_ebeam_gds_has_independent_fields_without_printed_numbers(self) -> None:
        field = component("E-beam multipass", 2)
        field["params"].update(
            {
                "field_size": 100.0,
                "target_width": 250.0,
                "target_height": 100.0,
                # Old projects may retain this value.  It must no longer add
                # polygon text to the fabrication layer.
                "show_order": True,
            }
        )

        polygons, _labels = component_geometry_arrays(field)
        ebeam_polygons = [
            np.asarray(points, dtype=float)
            for points, layer, _datatype in polygons
            if int(layer) == EBEAM_LAYER
        ]
        expected_fields = len(multipass_field_layout(field["params"])["fields"])

        # One separate 4-corner rectangle per field: no Boolean group and no
        # extra polygons forming printed field-order digits.
        self.assertEqual(len(ebeam_polygons), expected_fields)
        self.assertTrue(all(len(points) == 4 for points in ebeam_polygons))

    def test_exported_field_group_moves_without_source_gds(self) -> None:
        source = component("Straight", 1, x=25.0, y=-10.0)
        field = component("E-beam multipass", 2, x=25.0, y=-10.0)
        field["params"].update(
            {"target_width": 200.0, "target_height": 100.0, "manual_layout_locked": True}
        )
        field["coverage_source_uids"] = [1]

        before = resolve_and_build([source, field])
        source_before = layer_polygons(before, 1)
        fields_before = layer_polygons(before, EBEAM_LAYER)

        field["x"] += 750.0
        field["y"] += 125.0
        after = resolve_and_build([source, field])
        source_after = layer_polygons(after, 1)
        fields_after = layer_polygons(after, EBEAM_LAYER)

        self.assertEqual(len(source_before), len(source_after))
        self.assertEqual(len(fields_before), len(fields_after))
        for first, second in zip(source_before, source_after):
            np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)
        for first, second in zip(fields_before, fields_after):
            np.testing.assert_allclose(
                second - first,
                np.tile([750.0, 125.0], (len(first), 1)),
                rtol=0.0,
                atol=1e-12,
            )

    def test_absolute_ebeam_command_changes_only_selected_field_components(self) -> None:
        source = component("Straight", 1, x=11.0, y=12.0)
        field = component("E-beam multipass", 2, x=10.0, y=20.0)
        field["coverage_source_uids"] = [1]

        class StatusBar:
            def showMessage(self, *_args, **_kwargs) -> None:
                pass

        class FakeWindow:
            def __init__(self) -> None:
                self.rebuilt = []

            def selected_components(self):
                return [field]

            def components_world_bounds(self, _components):
                return (0.0, 0.0, 520.0, 520.0)

            def snapshot(self):
                return json.dumps([source, field], sort_keys=True)

            def commit_interaction_snapshot(self, _snapshot):
                pass

            def rebuild_scene(self, select_uids=None):
                self.rebuilt = list(select_uids or [])

            def statusBar(self):
                return StatusBar()

        fake = FakeWindow()
        with patch(
            "max_layout.ui.window.QInputDialog.getDouble",
            side_effect=[(1000.0, True), (2000.0, True)],
        ):
            NativeLayoutWindow.position_selected_ebeam_blocks(fake)

        # The old field-set center was (260, 260), so this is a translation
        # of (+740, +1740) applied to the E-beam component only.
        self.assertEqual((field["x"], field["y"]), (750.0, 1760.0))
        self.assertEqual((source["x"], source["y"]), (11.0, 12.0))
        self.assertEqual(field["coverage_source_uids"], [1])
        self.assertTrue(field["params"]["manual_layout_locked"])
        self.assertEqual(fake.rebuilt, [2])

    def test_reset_fields_returns_a_manual_group_to_source_tracking(self) -> None:
        field = component("E-beam multipass", 2)
        field["params"].update(
            {
                "manual_layout_locked": True,
                "manual_field_offsets": {"c1_r1": [20.0, 30.0]},
                "manual_field_order": {"c1_r1": 3},
                "removed_field_keys": ["c2_r1"],
                "auto_pruned_field_keys": ["c3_r1"],
            }
        )

        class FakeWindow:
            def selected_components(self):
                return [field]

            def snapshot(self):
                return "before"

            def component_by_uid(self, _uid):
                return None

            def prune_ebeam_component(self, _component):
                pass

            def commit_interaction_snapshot(self, _snapshot):
                pass

            def rebuild_scene(self, select_uids=None):
                self.selected = list(select_uids or [])

        fake = FakeWindow()
        NativeLayoutWindow.reset_selected_ebeam_fields(fake)
        self.assertFalse(field["params"]["manual_layout_locked"])
        self.assertEqual(field["params"]["manual_field_offsets"], {})
        self.assertEqual(field["params"]["manual_field_order"], {})
        self.assertEqual(field["params"]["removed_field_keys"], [])
        self.assertEqual(field["params"]["auto_pruned_field_keys"], [])
        self.assertEqual(fake.selected, [2])

    def test_user_selected_top_center_field_anchors_geometry_order(self) -> None:
        params = dict(DEFAULT_COMPONENT_VALUES["E-beam multipass"])
        params.update(
            {
                "field_size": 100.0,
                "edge_clearance": 0.0,
                "target_width": 500.0,
                "target_height": 300.0,
                "start_corner": "top-left",
                "primary_axis": "x",
                "serpentine": True,
                # A device can begin at the top center rather than a corner.
                "manual_field_order": {"c3_r1": 1},
            }
        )
        fields = multipass_field_layout(params)["fields"]
        self.assertEqual(fields[0]["field_key"], "c3_r1")
        # The automatic continuation follows touching fields through the
        # geometry; it must not jump back to the configured top-left corner.
        for previous, current in zip(fields, fields[1:]):
            dc = abs(int(previous["column"]) - int(current["column"]))
            dr = abs(int(previous["row"]) - int(current["row"]))
            self.assertEqual(dc + dr, 1)

    def test_user_selected_first_explicit_field_orders_by_geometry(self) -> None:
        params = dict(DEFAULT_COMPONENT_VALUES["E-beam multipass"])
        params.update(
            {
                "explicit_fields": [
                    {"field_key": "left", "bounds": [-150, -50, -50, 50]},
                    {"field_key": "far", "bounds": [150, -50, 250, 50]},
                    {"field_key": "center", "bounds": [-50, -50, 50, 50]},
                    {"field_key": "right", "bounds": [50, -50, 150, 50]},
                ],
                "manual_field_order": {"center": 1},
            }
        )
        fields = multipass_field_layout(params)["fields"]
        self.assertEqual(
            [field["field_key"] for field in fields],
            ["center", "left", "right", "far"],
        )

    def test_assigning_new_first_field_clears_old_fixed_slots(self) -> None:
        field = component("E-beam multipass", 2)
        field["params"].update(
            {
                "field_size": 100.0,
                "edge_clearance": 0.0,
                "target_width": 300.0,
                "target_height": 200.0,
                "manual_field_order": {"c1_r1": 1, "c2_r2": 4},
            }
        )

        class FakeWindow:
            def component_by_uid(self, uid):
                return field if int(uid) == 2 else None

            def field_order_state(self):
                return {}, {2: (1, 6)}

        NativeLayoutWindow.set_field_global_order(
            FakeWindow(), 2, "c2_r1", 1
        )
        self.assertEqual(field["params"]["manual_field_order"], {"c2_r1": 1})
        self.assertEqual(
            multipass_field_layout(field["params"])["fields"][0]["field_key"],
            "c2_r1",
        )


if __name__ == "__main__":
    unittest.main()
