from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QGraphicsItem

from max_layout.ui.items import ComponentGraphicsItem


def marker(*, dicing: bool) -> dict:
    return {
        "uid": 1,
        "kind": "Square mark",
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "attachment": None,
        "params": {
            "size": 100.0,
            "width": 100.0,
            "height": 20.0,
            "line_width": 1.0,
            "filled": 1,
            "dicing_marker": dicing,
            "layer": 10,
            "datatype": 0,
        },
    }


class DicingMarkerPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_dicing_marker_has_non_fabricated_screen_locator(self) -> None:
        item = ComponentGraphicsItem(None, marker(dicing=True))
        locators = [
            child
            for child in item.childItems()
            if child.data(2) == "dicing_preview_locator"
        ]
        self.assertEqual(len(locators), 1)
        self.assertTrue(
            bool(
                locators[0].flags()
                & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
        )

    def test_regular_square_has_no_dicing_locator(self) -> None:
        item = ComponentGraphicsItem(None, marker(dicing=False))
        self.assertFalse(
            any(
                child.data(2) == "dicing_preview_locator"
                for child in item.childItems()
            )
        )


if __name__ == "__main__":
    unittest.main()
