"""Dark palette, popup styling, and per-layer colours."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QAbstractItemView, QMenu, QWidget

DARK_POPUP_STYLE = r"""
QWidget {
    background-color: #000000;
    color: #ffffff;
}
QMenu {
    background-color: #000000;
    color: #ffffff;
    border: 1px solid #6b7280;
    padding: 5px;
}
QMenu::item {
    background-color: #000000;
    color: #ffffff;
    padding: 7px 30px 7px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QMenu::item:disabled {
    background-color: #000000;
    color: #7d8590;
}
QMenu::item:checked {
    background-color: #111827;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background-color: #374151;
    margin: 4px 8px;
}
QAbstractItemView {
    background-color: #000000;
    color: #ffffff;
    alternate-background-color: #090909;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    border: 1px solid #6b7280;
    outline: 0;
}
QAbstractItemView::item {
    background-color: #000000;
    color: #ffffff;
    padding: 5px;
}
QAbstractItemView::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #000000;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #4b5563;
    border-radius: 4px;
}
"""


def _force_dark_popup(widget: QWidget) -> None:
    """Force menus and popup lists to a black background with white text."""
    try:
        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(9, 9, 9))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(37, 99, 235))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)
        widget.setStyleSheet(DARK_POPUP_STYLE)
        for view in widget.findChildren(QAbstractItemView):
            view.setPalette(palette)
            view.setAutoFillBackground(True)
            view.setStyleSheet(DARK_POPUP_STYLE)
        widget.update()
    except RuntimeError:
        pass


class DarkPopupEventFilter(QObject):
    """Reapplies the dark popup style when Qt creates a menu dynamically."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (
            QEvent.Type.Polish,
            QEvent.Type.Show,
            QEvent.Type.ShowToParent,
        ):
            if isinstance(watched, QMenu):
                _force_dark_popup(watched)
            elif isinstance(watched, QAbstractItemView):
                window = watched.window()
                if window is not None and bool(window.windowFlags() & Qt.WindowType.Popup):
                    _force_dark_popup(watched)
                    if window is not watched:
                        _force_dark_popup(window)
            elif isinstance(watched, QWidget) and bool(
                watched.windowFlags() & Qt.WindowType.Popup
            ):
                _force_dark_popup(watched)
        return False

LAYER_COLORS = {
    0: QColor("#f472b6"),
    1: QColor("#40c8ff"),
    2: QColor("#ff9f43"),
    3: QColor("#ffe66d"),
    4: QColor("#2563eb"),
    5: QColor("#55efc4"),
    6: QColor("#fef08a"),
}


def color_for_layer(layer: int, alpha: int = 150) -> QColor:
    color = QColor(LAYER_COLORS.get(int(layer), QColor("#b2bec3")))
    color.setAlpha(alpha)
    return color
