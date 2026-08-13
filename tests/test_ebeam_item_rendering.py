from __future__ import annotations

from copy import deepcopy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QGraphicsSimpleTextItem

from max_layout.constants import DEFAULT_COMPONENT_VALUES
from max_layout.ui.items import EbeamContainerItem


def _ebeam_component(*, show_order: bool) -> dict:
    params = deepcopy(DEFAULT_COMPONENT_VALUES["E-beam multipass"])
    params.update(
        {
            "show_order": show_order,
            "field_size": 100.0,
            "target_width": 300.0,
            "target_height": 300.0,
        }
    )
    return {
        "uid": 1,
        "kind": "E-beam multipass",
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "params": params,
    }


class EbeamItemRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_hidden_field_numbers_create_no_text_items(self) -> None:
        container = EbeamContainerItem(object(), _ebeam_component(show_order=False))

        self.assertGreater(len(container.field_items), 1)
        self.assertTrue(
            all(field.order_label is None for field in container.field_items.values())
        )
        self.assertFalse(
            any(
                isinstance(child, QGraphicsSimpleTextItem)
                for field in container.field_items.values()
                for child in field.childItems()
            )
        )

    def test_order_labels_are_created_and_updated_only_when_requested(self) -> None:
        component = _ebeam_component(show_order=False)
        container = EbeamContainerItem(object(), component)
        field = next(iter(container.field_items.values()))

        component["params"]["show_order"] = True
        field.set_global_order(27)
        self.assertIsNotNone(field.order_label)
        self.assertTrue(field.order_label.isVisible())
        self.assertEqual(field.order_label.text(), "27")

        component["params"]["show_order"] = False
        field.set_global_order(28)
        self.assertIsNone(field.order_label)
        self.assertIn("Write field 28", field.toolTip())


if __name__ == "__main__":
    unittest.main()
