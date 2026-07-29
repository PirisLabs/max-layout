"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ..ui.theme import DarkPopupEventFilter
from ..ui.window import NativeLayoutWindow


def native_main() -> None:
    # Native macOS popup surfaces ignore application stylesheets.
    for attribute_name in ("AA_DontUseNativeMenuBar", "AA_DontUseNativeDialogs"):
        attribute = getattr(Qt.ApplicationAttribute, attribute_name, None)
        if attribute is not None:
            QApplication.setAttribute(attribute, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Photonic Layout Editor Native")
    app.setOrganizationName("PirisLabs")
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 10))
    dark_popup_filter = DarkPopupEventFilter(app)
    app.installEventFilter(dark_popup_filter)
    app._dark_popup_filter = dark_popup_filter
    app.setStyleSheet(
        """
        QMainWindow, QDialog {
            background: #111722;
            color: #eaf1f8;
        }
        QWidget {
            color: #eaf1f8;
            font-size: 12px;
        }
        QMenuBar {
            background: #151d29;
            color: #ffffff;
            border-bottom: 1px solid #2a3545;
            padding: 3px;
        }
        QMenuBar::item {
            background: #151d29;
            color: #ffffff;
            padding: 6px 10px;
            border-radius: 5px;
        }
        QMenuBar::item:selected {
            background: #263448;
            color: #ffffff;
        }
        QMenu {
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #4b5563;
            padding: 5px;
        }
        QMenu::item {
            background-color: #000000;
            color: #ffffff;
            padding: 7px 28px 7px 12px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }
        QMenu::item:disabled {
            color: #7d8590;
            background-color: #000000;
        }
        QMenu::separator {
            height: 1px;
            background: #374151;
            margin: 4px 8px;
        }
        QComboBox QAbstractItemView,
        QComboBoxPrivateContainer,
        QComboBoxPrivateContainer QWidget,
        QAbstractItemView[showDropIndicator="true"] {
            background-color: #000000;
            color: #ffffff;
            selection-background-color: #2563eb;
            selection-color: #ffffff;
            border: 1px solid #4b5563;
            outline: 0;
        }
        QComboBox QAbstractItemView::item,
        QComboBoxPrivateContainer QAbstractItemView::item {
            background-color: #000000;
            color: #ffffff;
            padding: 5px;
        }
        QComboBox QAbstractItemView::item:selected,
        QComboBoxPrivateContainer QAbstractItemView::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }
        QToolTip {
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #6b7280;
            padding: 4px;
        }
        QToolBar {
            background: #151d29;
            border: none;
            border-bottom: 1px solid #283445;
            spacing: 5px;
            padding: 5px 6px;
        }
        QToolBar::separator {
            background: #344256;
            width: 1px;
            margin: 4px 6px;
        }
        QToolButton {
            background: #202b3a;
            border: 1px solid #334359;
            border-radius: 6px;
            padding: 6px 9px;
            color: #edf4fb;
        }
        QToolButton:hover {
            background: #2b3a4f;
            border-color: #4c6586;
        }
        QToolButton:pressed, QToolButton:checked {
            background: #2d6cdf;
            border-color: #6fa0ff;
        }
        QDockWidget {
            color: #eaf1f8;
            font-weight: 600;
        }
        QDockWidget::title {
            background: #182231;
            border-bottom: 1px solid #314055;
            padding: 9px 10px;
            text-align: left;
        }
        QDockWidget#propertiesDock {
            color: #111827;
            font-weight: 800;
        }
        QDockWidget#propertiesDock::title {
            background: #f59e0b;
            color: #111827;
            border-bottom: 1px solid #b45309;
            font-weight: 900;
        }
        QWidget#propertiesOuter,
        QWidget#componentPropertiesContent,
        QScrollArea#componentPropertiesScroll,
        QScrollArea#componentPropertiesScroll > QWidget > QWidget {
            background: #f59e0b;
            color: #111827;
        }
        QWidget#componentPropertiesContent QLabel,
        QWidget#propertiesOuter QLabel,
        QWidget#componentPropertiesContent QCheckBox,
        QWidget#propertiesOuter QCheckBox {
            color: #111827;
            font-weight: 800;
        }
        QWidget#componentPropertiesContent QLineEdit,
        QWidget#componentPropertiesContent QSpinBox,
        QWidget#componentPropertiesContent QDoubleSpinBox,
        QWidget#componentPropertiesContent QComboBox {
            background: #fff7ed;
            color: #111827;
            border: 1px solid #7c2d12;
            font-weight: 800;
            selection-background-color: #fdba74;
            selection-color: #111827;
        }
        QWidget#componentPropertiesContent QLineEdit:focus,
        QWidget#componentPropertiesContent QSpinBox:focus,
        QWidget#componentPropertiesContent QDoubleSpinBox:focus,
        QWidget#componentPropertiesContent QComboBox:focus {
            border: 2px solid #111827;
            background: #ffffff;
        }
        QWidget#propertiesOuter QPushButton {
            background: #fdba74;
            color: #111827;
            border: 1px solid #7c2d12;
            font-weight: 900;
        }
        QWidget#propertiesOuter QPushButton:hover {
            background: #fed7aa;
            border-color: #111827;
        }
        QTreeWidget, QListWidget, QPlainTextEdit, QScrollArea {
            background: #101720;
            border: 1px solid #2d3a4b;
            border-radius: 7px;
            alternate-background-color: #151e29;
        }
        QTreeWidget::item {
            padding: 5px 3px;
            border-radius: 4px;
        }
        QTreeWidget::item:hover {
            background: #1d2b3d;
        }
        QTreeWidget::item:selected {
            background: #2d6cdf;
            color: white;
        }
        QHeaderView::section {
            background: #1a2533;
            color: #cbd8e6;
            border: none;
            border-right: 1px solid #2b394a;
            border-bottom: 1px solid #2b394a;
            padding: 6px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #0d141d;
            border: 1px solid #35465c;
            border-radius: 6px;
            padding: 6px 7px;
            selection-background-color: #2d6cdf;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #5e95ff;
        }
        QPushButton {
            background: #253247;
            border: 1px solid #3b4d68;
            border-radius: 7px;
            padding: 7px 11px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #30425d;
            border-color: #5d79a4;
        }
        QPushButton:pressed {
            background: #2d6cdf;
        }
        QPushButton:disabled {
            color: #748091;
            background: #1a222e;
            border-color: #293443;
        }
        QCheckBox {
            spacing: 7px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #51647e;
            border-radius: 4px;
            background: #0d141d;
        }
        QCheckBox::indicator:checked {
            background: #2d6cdf;
            border-color: #79a4ff;
        }
        QStatusBar {
            background: #131b26;
            border-top: 1px solid #2d3a4c;
            color: #aebed0;
        }
        QLabel#statusPill {
            background: #202b3a;
            border: 1px solid #34445a;
            border-radius: 6px;
            padding: 3px 8px;
            margin: 2px;
            color: #dce9f5;
        }
        QSplitter::handle {
            background: #263244;
        }
        QSplitter::handle:hover {
            background: #3b506d;
        }
        QScrollBar:vertical {
            background: #111822;
            width: 11px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #3a4b63;
            min-height: 25px;
            border-radius: 5px;
        }
        QScrollBar:horizontal {
            background: #111822;
            height: 11px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #3a4b63;
            min-width: 25px;
            border-radius: 5px;
        }
        QProgressDialog {
            min-width: 420px;
        }
        """
    )
    window = NativeLayoutWindow()
    window.show()
    raise SystemExit(app.exec())
