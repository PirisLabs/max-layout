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
)
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


if __name__ == "__main__":
    unittest.main()
