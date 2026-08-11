from __future__ import annotations

import os
import unittest


# Exercise real widget geometry without requiring a desktop session in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QScrollArea

from max_layout.ui.lumerical_dialog import LumericalExportDialog


class LumericalExportDialogLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.dialog = LumericalExportDialog(
            all_components=[],
            scope_options=[("All layout geometry", [])],
        )

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_export_actions_remain_visible_in_a_short_window(self) -> None:
        self.dialog.resize(1024, 640)
        self.dialog.show()
        self.app.processEvents()

        action_boxes = [
            box
            for box in self.dialog.findChildren(QDialogButtonBox)
            if box.button(QDialogButtonBox.StandardButton.Save) is not None
        ]
        self.assertEqual(len(action_boxes), 1)
        actions = action_boxes[0]
        save = actions.button(QDialogButtonBox.StandardButton.Save)
        cancel = actions.button(QDialogButtonBox.StandardButton.Cancel)

        self.assertLessEqual(self.dialog.height(), 640)
        self.assertTrue(actions.isVisibleTo(self.dialog))
        self.assertTrue(save.isVisibleTo(self.dialog))
        self.assertTrue(cancel.isVisibleTo(self.dialog))
        self.assertTrue(self.dialog.rect().contains(actions.geometry()))
        self.assertLess(actions.geometry().bottom(), self.dialog.tabs.geometry().top())

    def test_fdtd_settings_scroll_in_a_short_window(self) -> None:
        self.dialog.resize(1024, 640)
        solver_index = next(
            index
            for index in range(self.dialog.tabs.count())
            if self.dialog.tabs.tabText(index) == "FDTD and compute"
        )
        self.dialog.tabs.setCurrentIndex(solver_index)
        self.dialog.show()
        self.app.processEvents()

        solver_tab = self.dialog.tabs.widget(solver_index)
        scroll_areas = solver_tab.findChildren(QScrollArea)
        self.assertEqual(len(scroll_areas), 1)
        settings_scroll = scroll_areas[0]

        self.assertTrue(settings_scroll.widgetResizable())
        self.assertTrue(settings_scroll.isVisibleTo(self.dialog))
        self.assertGreater(settings_scroll.verticalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
