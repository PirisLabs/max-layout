from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from max_layout.ui.dialogs import ArrayEditDialog
from max_layout.ui.window import NativeLayoutWindow


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Text:
    def __init__(self, value: str):
        self._value = value

    def text(self) -> str:
        return self._value


class _Choice:
    def __init__(self, value: str):
        self._value = value

    def currentText(self) -> str:
        return self._value


class _Checked:
    def __init__(self, value: bool):
        self._value = value

    def isChecked(self) -> bool:
        return self._value


class _CreateArrayDialog:
    """Non-interactive 2 x 2 length sweep used by the behavior tests."""

    def __init__(self, _param_names, _parent=None):
        self.nx = _Value(2)
        self.ny = _Value(2)
        self.dx = _Text("100")
        self.dy = _Text("200")
        self.x_param = _Choice("length")
        self.y_param = _Choice("")
        self.x_values = _Text("[10, 20]")
        self.y_values = _Text("")
        self.auto_label = _Checked(False)
        self.label_prefix = _Text("A")
        self.label_offset_x = _Value(20.0)
        self.label_offset_y = _Value(20.0)

    def exec(self):
        return QDialog.DialogCode.Accepted


class EditableArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = NativeLayoutWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def _create_array(self):
        source = self.window.make_component("Straight", 10.0, 20.0)
        source["params"]["width"] = 1.2
        self.window.components = [source]
        self.window.rebuild_scene(select_uids=[int(source["uid"])])
        with patch("max_layout.ui.window.ArrayDialog", _CreateArrayDialog):
            self.window.create_array()
        array_id = str(source["array_group_id"])
        members = [
            component
            for component in self.window.components
            if str(component.get("array_group_id", "")) == array_id
            and component.get("kind") == "Straight"
        ]
        self.assertEqual(len(members), 4)
        return array_id, sorted(members, key=lambda item: tuple(reversed(item["array_index"])))

    def test_every_member_exposes_array_editor_and_normalized_recipe(self) -> None:
        array_id, members = self._create_array()
        for member in members:
            self.window.scene.clearSelection()
            self.window.items_by_uid[int(member["uid"])].setSelected(True)
            self.app.processEvents()
            # isVisible() also depends on the hidden off-screen parent window;
            # isHidden() records whether the property panel exposed the button.
            self.assertFalse(self.window.edit_array_button.isHidden())
            self.assertTrue(self.window.edit_array_button.isEnabled())
            spec = self.window.array_spec_for_member(member)
            self.assertEqual(spec["nx"], 2)
            self.assertEqual(spec["ny"], 2)
            self.assertEqual(spec["x_parameter"], "length")
            self.assertEqual(spec["x_values"], [10.0, 20.0])
            self.assertEqual(spec["x_spacings"], [100.0])
            self.assertEqual(spec["y_spacings"], [200.0])
            self.assertEqual(str(member["array_group_id"]), array_id)

    def test_regeneration_updates_shared_base_preserves_sweep_and_other_components(self) -> None:
        array_id, members = self._create_array()
        unrelated = self.window.make_component("Straight", 999.0, 888.0)
        unrelated["params"].update({"length": 777.0, "width": 9.0})
        unrelated_before = json.loads(json.dumps(unrelated))
        self.window.components.append(unrelated)

        spec = self.window.array_spec_for_member(members[-1])
        spec.update(
            {
                "nx": 3,
                "ny": 2,
                "x_spacings": [150.0, 250.0],
                "y_spacings": [75.0],
                "x_values": [11.0, 22.0, 33.0],
            }
        )
        template_id = str(members[-1]["array_template_id"])
        returned_uids = self.window.regenerate_array(
            array_id,
            spec,
            {
                template_id: {
                    # Even if the dialog happens to return the displayed swept
                    # value, the array recipe remains authoritative for length.
                    "params": {"width": 2.4, "length": 999.0}
                }
            },
        )

        generated = [
            component
            for component in self.window.components
            if str(component.get("array_group_id", "")) == array_id
            and component.get("kind") == "Straight"
        ]
        self.assertEqual(len(generated), 6)
        self.assertEqual(set(returned_uids), {int(item["uid"]) for item in generated})
        by_index = {tuple(item["array_index"]): item for item in generated}
        for row in range(2):
            for col, swept_length in enumerate((11.0, 22.0, 33.0)):
                self.assertEqual(by_index[(col, row)]["params"]["length"], swept_length)
                self.assertEqual(by_index[(col, row)]["params"]["width"], 2.4)
        self.assertEqual(
            sorted({round(float(item["x"]), 9) for item in generated}),
            [10.0, 160.0, 410.0],
        )
        self.assertEqual(
            sorted({round(float(item["y"]), 9) for item in generated}),
            [20.0, 95.0],
        )
        self.assertEqual(unrelated, unrelated_before)

    def test_array_editor_exposes_shared_parameters_but_protects_swept_parameter(self) -> None:
        array_id, members = self._create_array()
        spec = self.window.array_spec_for_member(members[0])
        templates = self.window.array_templates_for_member(members[0], spec)
        dialog = ArrayEditDialog(self.window, array_id, spec, templates)
        try:
            template_id = str(members[0]["array_template_id"])
            inputs = dialog.template_inputs[template_id]
            self.assertIn("params.width", inputs)
            self.assertIn("params.length", inputs)
            width_widget, _ = inputs["params.width"]
            length_widget, _ = inputs["params.length"]
            self.assertTrue(width_widget.isEnabled())
            self.assertFalse(length_widget.isEnabled())
            width_widget.setValue(3.4)
            updates = dialog.template_updates()[template_id]["params"]
            self.assertEqual(updates["width"], 3.4)
            self.assertNotIn("length", updates)
        finally:
            dialog.close()

    def test_array_edit_is_undoable_and_recipe_survives_json_roundtrip(self) -> None:
        array_id, members = self._create_array()
        before = self.window.snapshot()
        spec = self.window.array_spec_for_member(members[0])
        spec.update({"nx": 3, "x_spacings": [125.0, 125.0], "x_values": [10.0, 15.0, 20.0]})
        self.window.regenerate_array(array_id, spec)
        self.window.commit_interaction_snapshot(before)
        self.assertEqual(
            len([item for item in self.window.components if item.get("array_group_id") == array_id]),
            6,
        )
        self.window.undo()
        restored = [item for item in self.window.components if item.get("array_group_id") == array_id]
        self.assertEqual(len(restored), 4)
        self.assertEqual(self.window.array_spec_for_member(restored[-1])["nx"], 2)

        payload = json.loads(json.dumps(self.window.project_payload()))
        second = NativeLayoutWindow()
        try:
            second.components = payload["components"]
            second.next_uid = payload["next_uid"]
            second.next_group_id = payload["next_group_id"]
            second.next_array_id = payload["next_array_id"]
            second.next_module_instance_id = payload["next_module_instance_id"]
            second.rebuild_scene()
            member = next(item for item in second.components if item.get("array_group_id") == array_id)
            roundtrip_spec = second.array_spec_for_member(member)
            self.assertEqual(roundtrip_spec["x_parameter"], "length")
            self.assertEqual(roundtrip_spec["x_values"], [10.0, 20.0])
            self.assertEqual(roundtrip_spec["x_spacings"], [100.0])
        finally:
            second.close()

    def test_legacy_array_is_inferred_from_geometry_and_parameter_grid(self) -> None:
        legacy = []
        for row, width in enumerate((1.0, 2.0)):
            for col, length in enumerate((10.0, 20.0, 30.0)):
                component = self.window.make_component(
                    "Straight", 5.0 + 100.0 * col, 7.0 + 50.0 * row
                )
                component["params"].update({"length": length, "width": width})
                component["array_group_id"] = "Alegacy"
                component["array_index"] = [col, row]
                legacy.append(component)
        self.window.components = legacy
        self.window.rebuild_scene()

        spec = self.window.array_spec_for_member(legacy[-1])
        self.assertEqual(spec["nx"], 3)
        self.assertEqual(spec["ny"], 2)
        self.assertEqual(spec["x_spacings"], [100.0, 100.0])
        self.assertEqual(spec["y_spacings"], [50.0])
        self.assertEqual(spec["x_parameter"], "length")
        self.assertEqual(spec["x_values"], [10.0, 20.0, 30.0])
        self.assertEqual(spec["y_parameter"], "width")
        self.assertEqual(spec["y_values"], [1.0, 2.0])
        self.window.show_component_properties(legacy[0])
        self.app.processEvents()
        self.assertFalse(self.window.edit_array_button.isHidden())


if __name__ == "__main__":
    unittest.main()
