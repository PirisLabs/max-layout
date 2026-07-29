"""The main application window."""

from __future__ import annotations

from ..acceleration import CPU_COUNT, DEFAULT_THREADS, configure as configure_acceleration
from ..runtime import launcher_path
from functools import partial
from pathlib import Path
from typing import Any
import copy
import json
import math
import os
import re
import shutil
import sys
import tempfile

import gdstk
import numpy as np

from PySide6.QtCore import QPoint, QPointF, QProcess, QRectF, QSettings, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QImageReader, QKeySequence, QPainterPath, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsScene, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea, QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..constants import CHOICE_PARAMETERS, COMPONENT_SPECS, DEFAULT_COMPONENT_VALUES, EBEAM_LAYER, GC_LAYER, LAYER_NAME_MAP, MARKER_COMPONENT_KINDS, MARKER_LAYER, NATIVE_APP_VERSION, PHOTONIC_LAYER, RF_COMPONENT_KINDS, RF_LAYER, component_display_name
from ..gds.build import _add_component_geometry_to_cell, _canonicalize_component_layers, library_bbox_and_center, resolve_and_build, rotate_components_layout
from ..gds.ebeam import multipass_field_layout
from ..geometry.shapes import mmi_total_length
from ..geometry.transforms import scene_to_world_point, transform_points, transformed_local_points, world_to_scene_point
from ..modules_db import load_native_modules, save_native_modules
from ..params import resize_component_parameters
from ..ports import PORT_ALIASES, component_global_ports, component_local_ports, solve_attachment
from ..ui.dialogs import ArrayDialog, EbeamDialog, ModuleVariablesDialog
from ..ui.items import ComponentGraphicsItem, EbeamContainerItem, LayoutView, WriteFieldItem
from ..ui.theme import _force_dark_popup, color_for_layer
from ..utils import inclusive_sweep, parse_sequence, safe_json_copy


class NativeLayoutWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # The native macOS menu bar ignores Qt stylesheets, so use the Qt menu bar.
        self.menuBar().setNativeMenuBar(False)
        self.menuBar().setStyleSheet(
            "QMenuBar { background:#151d29; color:#ffffff; }"
            "QMenuBar::item { background:#151d29; color:#ffffff; padding:6px 10px; }"
            "QMenuBar::item:selected { background:#263448; color:#ffffff; }"
        )
        self.setWindowTitle(f"Max Layout — Photonic + RF — {NATIVE_APP_VERSION}")
        self.resize(1680, 1000)
        self.settings = QSettings("PirisLabs", "PhotonicLayoutNative")
        self.layout_threads = max(1, min(CPU_COUNT, int(self.settings.value("layout/cpu_threads", DEFAULT_THREADS))))
        configure_acceleration(self.layout_threads)
        self.components: list[dict[str, Any]] = []
        self.items_by_uid: dict[int, QGraphicsItem] = {}
        self.next_uid = 1
        self.next_group_id = 1
        self.next_array_id = 1
        self.next_module_instance_id = 1
        self.project_path: Path | None = None
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self._restoring = False
        self._group_move_guard = False
        self.active_field: tuple[int, str] | None = None
        self.custom_modules = load_native_modules()
        self.parameter_widgets: dict[str, tuple[QWidget, str]] = {}
        self.export_process: QProcess | None = None
        self.export_temp_file: Path | None = None
        self.progress_dialog: QProgressDialog | None = None
        self.snap_enabled = True
        self.auto_connect_input_enabled = True
        self.snap_distance_pixels = 32.0
        self.show_ports_enabled = True
        self.field_play_timer = QTimer(self)
        self.field_play_timer.timeout.connect(self.advance_writefield_playback)
        self.field_play_sequence: list[tuple[int, str]] = []
        self.field_play_index = -1
        self.llm_process: QProcess | None = None
        self.llm_request_file: Path | None = None
        self.llm_response_file: Path | None = None

        self.scene = QGraphicsScene(self)
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)
        self.view = LayoutView(self.scene)
        self.setCentralWidget(self.view)

        self.setDockNestingEnabled(True)
        self.layer_visibility = {layer: True for layer in LAYER_NAME_MAP}
        self.create_library_dock()
        self.create_properties_dock()
        self.create_project_dock()
        self.create_layers_dock()
        self.create_llm_dock()
        self.splitDockWidget(self.library_dock, self.project_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self.properties_dock, self.layers_dock, Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.layers_dock, self.llm_dock)
        self.layers_dock.raise_()
        self.create_actions_and_toolbars()
        self.create_status_bar()
        QTimer.singleShot(0,self.apply_default_dock_proportions)
        self.view.cursorWorldChanged.connect(self.update_cursor_status)
        self.view.zoomChanged.connect(self.update_zoom_status)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_canvas_context_menu)
        self.statusBar().showMessage(
            "Native Qt editor ready. Middle-drag pans; wheel zooms; F fits all; 0 centers the origin."
        )

        self.new_project(push_undo=False)

    def apply_default_dock_proportions(self) -> None:
        """Allocate 80% of the right dock height to Properties and 20% to Layers/LLM."""
        # A slight title-bar compensation yields an approximately 80/20
        # visible-area split rather than an 80/20 outer-frame split.
        self.resizeDocks([self.properties_dock,self.layers_dock],[770,230],Qt.Orientation.Vertical)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def create_library_dock(self) -> None:
        dock = QDockWidget("Component library", self)
        dock.setObjectName("componentLibraryDock")
        container = QWidget()
        layout = QVBoxLayout(container)
        self.library_filter = QLineEdit()
        self.library_filter.setPlaceholderText("Filter components…")
        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderHidden(True)
        self.library_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.library_filter)
        layout.addWidget(self.library_tree)
        self.add_component_button = QPushButton("Add selected component")
        layout.addWidget(self.add_component_button)
        dock.setWidget(container)
        dock.setMinimumWidth(270)
        self.library_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.library_filter.textChanged.connect(self.populate_library)
        self.library_tree.itemDoubleClicked.connect(lambda *_: self.add_selected_library_component())
        self.add_component_button.clicked.connect(self.add_selected_library_component)
        self.populate_library()

    def populate_library(self) -> None:
        filter_text = self.library_filter.text().strip().lower()
        self.library_tree.clear()
        photonic_groups: dict[str, list[str]] = {
            "Waveguides & bends": [],
            "Grating couplers": [],
            "MMI": [],
            "MZI": [],
            "Resonators": [],
            "Photonic crystals": [],
            "Photonic test blocks": [],
            "Other photonic": [],
        }
        rf_groups: dict[str,list[str]]={
            "CPW transmission lines":[],
            "RF tapers":[],
            "RF bends":[],
            "Electrodes & modulators":[],
            "Calibration structures":[],
            "Other RF":[],
        }
        categories: dict[str, list[str]] = {
            "Alignment / labels": [],
            "E-beam": [],
            "Chip / utility": [],
            "User modules": [],
        }
        for kind in DEFAULT_COMPONENT_VALUES:
            if filter_text and filter_text not in kind.lower():
                continue
            if kind in RF_COMPONENT_KINDS or kind == "MZI + CPW module":
                lower=kind.lower()
                if "open" in lower or "short" in lower:group="Calibration structures"
                elif "electrode" in lower or "mzi" in lower:group="Electrodes & modulators"
                elif "taper" in lower:group="RF tapers"
                elif "bend" in lower:group="RF bends"
                elif "cpw" in lower:group="CPW transmission lines"
                else:group="Other RF"
                rf_groups[group].append(kind)
            elif kind in MARKER_COMPONENT_KINDS:
                categories["Alignment / labels"].append(kind)
            elif kind == "E-beam multipass":
                categories["E-beam"].append(kind)
            elif kind == "Chip outline":
                categories["Chip / utility"].append(kind)
            else:
                lower=kind.lower()
                if "test block" in lower:group="Photonic test blocks"
                elif "photonic crystal" in lower:group="Photonic crystals"
                elif "mzi" in lower:group="MZI"
                elif "mmi" in lower:group="MMI"
                elif any(token in lower for token in ("ring", "racetrack", "resonator", "loopback")):group="Resonators"
                elif "grating" in lower or "edge coupler" in lower:group="Grating couplers"
                elif any(token in lower for token in ("straight", "taper", "bend", "feedline", "waveguide")):group="Waveguides & bends"
                else:group="Other photonic"
                photonic_groups[group].append(kind)
        for module_name in sorted(self.custom_modules):
            if not filter_text or filter_text in module_name.lower():
                categories["User modules"].append(module_name)
        populated_photonic={name:values for name,values in photonic_groups.items() if values}
        if populated_photonic:
            photonic_parent=QTreeWidgetItem(["Photonic"]);photonic_parent.setFlags(photonic_parent.flags()&~Qt.ItemFlag.ItemIsSelectable);self.library_tree.addTopLevelItem(photonic_parent)
            for group,names in populated_photonic.items():
                group_item=QTreeWidgetItem([group]);group_item.setFlags(group_item.flags()&~Qt.ItemFlag.ItemIsSelectable);photonic_parent.addChild(group_item)
                if group=="Photonic test blocks":
                    test_groups={"Grating":[],"MMI":[],"MZI":[],"Resonator":[],"Other":[]}
                    for name in names:
                        lower=name.lower()
                        if "grating" in lower:target="Grating"
                        elif "mzi" in lower:target="MZI"
                        elif "mmi" in lower:target="MMI"
                        elif "ring" in lower or "resonator" in lower or "racetrack" in lower:target="Resonator"
                        else:target="Other"
                        test_groups[target].append(name)
                    for test_group,test_names in test_groups.items():
                        if not test_names:continue
                        test_parent=QTreeWidgetItem([test_group]);test_parent.setFlags(test_parent.flags()&~Qt.ItemFlag.ItemIsSelectable);group_item.addChild(test_parent)
                        for name in sorted(test_names):
                            child=QTreeWidgetItem([component_display_name(name)]);child.setData(0,Qt.ItemDataRole.UserRole,("component",name));test_parent.addChild(child)
                        test_parent.setExpanded(bool(filter_text))
                else:
                    for name in sorted(names):
                        child=QTreeWidgetItem([component_display_name(name)]);child.setData(0,Qt.ItemDataRole.UserRole,("component",name));group_item.addChild(child)
                group_item.setExpanded(bool(filter_text))
            photonic_parent.setExpanded(True)
        populated_rf={name:values for name,values in rf_groups.items() if values}
        if populated_rf:
            rf_parent=QTreeWidgetItem(["RF"]);rf_parent.setFlags(rf_parent.flags()&~Qt.ItemFlag.ItemIsSelectable);self.library_tree.addTopLevelItem(rf_parent)
            for group,names in populated_rf.items():
                group_item=QTreeWidgetItem([group]);group_item.setFlags(group_item.flags()&~Qt.ItemFlag.ItemIsSelectable);rf_parent.addChild(group_item)
                for name in sorted(names):
                    child=QTreeWidgetItem([component_display_name(name)]);child.setData(0,Qt.ItemDataRole.UserRole,("component",name));group_item.addChild(child)
                group_item.setExpanded(bool(filter_text))
            rf_parent.setExpanded(True)
        for category, names in categories.items():
            if not names:
                continue
            parent = QTreeWidgetItem([category])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.library_tree.addTopLevelItem(parent)
            for name in sorted(names):
                child = QTreeWidgetItem([name])
                child.setData(0, Qt.ItemDataRole.UserRole, ("module" if category == "User modules" else "component", name))
                parent.addChild(child)
            parent.setExpanded(True)

    def create_properties_dock(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("propertiesDock")
        outer = QWidget()
        outer.setObjectName("propertiesOuter")
        outer_layout = QVBoxLayout(outer)
        self.properties_scroll = QScrollArea()
        self.properties_scroll.setObjectName("componentPropertiesScroll")
        self.properties_scroll.setWidgetResizable(True)
        self.properties_content = QWidget()
        self.properties_content.setObjectName("componentPropertiesContent")
        self.properties_form = QFormLayout(self.properties_content)
        self.properties_scroll.setWidget(self.properties_content)
        outer_layout.addWidget(self.properties_scroll)
        button_row = QHBoxLayout()
        self.apply_properties_button = QPushButton("Apply")
        self.update_ebeam_button = QPushButton("Update / prune fields")
        self.update_ebeam_button.setVisible(False)
        self.module_variables_button = QPushButton("Module variables…")
        self.module_variables_button.setVisible(False)
        button_row.addWidget(self.apply_properties_button)
        button_row.addWidget(self.update_ebeam_button)
        button_row.addWidget(self.module_variables_button)
        outer_layout.addLayout(button_row)
        dock.setWidget(outer)
        dock.setMinimumWidth(315)
        self.properties_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.apply_properties_button.clicked.connect(self.apply_properties)
        self.update_ebeam_button.clicked.connect(self.update_selected_ebeam)
        self.module_variables_button.clicked.connect(self.open_module_variables)
        self.show_no_selection_properties()

    def create_project_dock(self) -> None:
        dock = QDockWidget("Product library / Project", self)
        dock.setObjectName("projectDock")
        container = QWidget()
        layout = QVBoxLayout(container)
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabels(["Name", "UID", "Layer"])
        self.project_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.project_tree.itemSelectionChanged.connect(self.project_tree_selection_changed)
        self.project_tree.itemDoubleClicked.connect(self.project_tree_double_clicked)
        self.project_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self.show_project_context_menu)
        layout.addWidget(self.project_tree)
        dock.setWidget(container)
        dock.setMinimumWidth(270)
        self.project_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def create_layers_dock(self) -> None:
        dock = QDockWidget("Layers", self)
        dock.setObjectName("layersDock")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        self.layer_checkboxes: dict[int, QCheckBox] = {}
        for layer, name in sorted(LAYER_NAME_MAP.items()):
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            color = color_for_layer(layer, 255)
            swatch.setStyleSheet(
                f"background:{color.name()}; border-radius:4px; border:1px solid rgba(255,255,255,0.35);"
            )
            checkbox = QCheckBox(f"{layer}  {name}")
            checkbox.setChecked(True)
            checkbox.toggled.connect(partial(self.set_layer_visible, layer))
            self.layer_checkboxes[layer] = checkbox
            row.addWidget(swatch)
            row.addWidget(checkbox, 1)
            layout.addLayout(row)
        layout.addSpacing(6)
        show_all = QPushButton("Show all layers")
        show_all.clicked.connect(self.show_all_layers)
        layout.addWidget(show_all)
        layout.addStretch(1)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.Shape.NoFrame);scroll.setWidget(content)
        dock.setWidget(scroll)
        dock.setMinimumHeight(100)
        dock.setMinimumWidth(315)
        self.layers_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def create_llm_dock(self) -> None:
        dock = QDockWidget("ChatGPT / LLM", self)
        dock.setObjectName("llmDock")
        content = QWidget()
        layout = QVBoxLayout(content)

        form = QFormLayout()
        self.llm_scope = QComboBox()
        self.llm_scope.addItem("Modify layout", "layout")
        self.llm_scope.addItem("Improve source code", "source")
        self.llm_mode = QComboBox()
        self.llm_mode.addItem("Local commands", "local")
        self.llm_mode.addItem("Codex CLI", "codex")
        self.llm_mode.addItem("OpenAI cloud", "openai")
        self.llm_model = QLineEdit("gpt-5.6")
        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_api_key.setPlaceholderText("Uses OPENAI_API_KEY if blank")
        form.addRow("Task", self.llm_scope)
        form.addRow("Mode", self.llm_mode)
        form.addRow("Model", self.llm_model)
        form.addRow("API key", self.llm_api_key)
        layout.addLayout(form)

        self.llm_chat_log = QPlainTextEdit()
        self.llm_chat_log.setReadOnly(True)
        self.llm_chat_log.setPlaceholderText(
            "Describe layout changes or request a source-code update."
        )
        self.llm_chat_log.setPlainText(
            "Assistant: Local layout commands work without an API key. "
            "Cloud source mode writes a syntax-checked updated copy and does "
            "not overwrite the running editor."
        )
        layout.addWidget(self.llm_chat_log, 1)

        self.llm_prompt = QPlainTextEdit()
        self.llm_prompt.setMaximumHeight(110)
        self.llm_prompt.setPlaceholderText(
            "Example: add a CPW at 0,0 and align its input signal point to the closest RF point."
        )
        layout.addWidget(self.llm_prompt)

        buttons = QHBoxLayout()
        self.llm_send_button = QPushButton("Apply instruction")
        self.llm_send_button.setObjectName("primaryButton")
        self.llm_clear_button = QPushButton("Clear")
        buttons.addWidget(self.llm_send_button)
        buttons.addWidget(self.llm_clear_button)
        layout.addLayout(buttons)

        note = QLabel(
            "The API key is used for the request only and is not stored in the project."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.llm_send_button.clicked.connect(self.run_llm_assistant)
        self.llm_clear_button.clicked.connect(
            lambda: self.llm_chat_log.setPlainText("Chat cleared. The layout is unchanged.")
        )
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.Shape.NoFrame);scroll.setWidget(content)
        dock.setWidget(scroll)
        dock.setMinimumHeight(100)
        dock.setMinimumWidth(315)
        self.llm_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def create_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self.selection_status_label = QLabel("Selection: 0")
        self.cursor_status_label = QLabel("X 0.000 µm   Y 0.000 µm")
        self.zoom_status_label = QLabel("Zoom 100%")
        self.opengl_status_label = QLabel("OpenGL" if self.view.opengl_enabled else "CPU canvas")
        self.license_status_label = QLabel("© 2026 Ali Khalatpour · Piris Labs — Provided free to use and share with attribution · No warranty")
        self.license_status_label.setStyleSheet("font-size: 9px; color: #8492a6; padding: 0 6px;")
        self.license_status_label.setToolTip("Copyright retained by Ali Khalatpour / Piris Labs. The software is provided free of charge to use and share with attribution, without warranty.")
        bar.addPermanentWidget(self.license_status_label, 1)
        for label in (
            self.selection_status_label,
            self.cursor_status_label,
            self.zoom_status_label,
            self.opengl_status_label,
        ):
            label.setObjectName("statusPill")
            bar.addPermanentWidget(label)
        self.update_zoom_status(self.view.current_zoom_percent())

    def create_actions_and_toolbars(self) -> None:
        self.actions: dict[str, QAction] = {}
        file_toolbar = QToolBar("File", self)
        edit_toolbar = QToolBar("Edit", self)
        view_toolbar = QToolBar("View", self)
        layout_toolbar = QToolBar("Layout", self)
        ebeam_toolbar = QToolBar("E-beam", self)
        for toolbar in (file_toolbar, edit_toolbar, view_toolbar, layout_toolbar, ebeam_toolbar):
            toolbar.setMovable(False)
            toolbar.setIconSize(QSize(18, 18))
            toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.addToolBar(toolbar)

        file_menu = self.menuBar().addMenu("&File")
        edit_menu = self.menuBar().addMenu("&Edit")
        view_menu = self.menuBar().addMenu("&View")
        layout_menu = self.menuBar().addMenu("&Layout")
        ebeam_menu = self.menuBar().addMenu("&E-beam")
        for top_level_menu in (file_menu, edit_menu, view_menu, layout_menu, ebeam_menu):
            _force_dark_popup(top_level_menu)

        def action(
            key: str,
            text: str,
            slot,
            shortcut: str | None = None,
            toolbar: QToolBar | None = None,
            menu: QMenu | None = None,
            checkable: bool = False,
            checked: bool = False,
            status_tip: str | None = None,
        ) -> QAction:
            item = QAction(text, self)
            item.setCheckable(checkable)
            if checkable:
                item.setChecked(checked)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            if status_tip:
                item.setStatusTip(status_tip)
                item.setToolTip(status_tip)
            item.triggered.connect(slot)
            if toolbar:
                toolbar.addAction(item)
            if menu:
                menu.addAction(item)
            self.actions[key] = item
            return item

        action("new", "New", self.new_project, "Ctrl+N", file_toolbar, file_menu)
        action("open", "Open", self.open_project, "Ctrl+O", file_toolbar, file_menu)
        action("save", "Save", self.save_project, "Ctrl+S", file_toolbar, file_menu)
        action("save_as", "Save As", self.save_project_as, "Ctrl+Shift+S", None, file_menu)
        file_toolbar.addSeparator()
        file_menu.addSeparator()
        action("export_gds", "Export GDS", self.export_gds, None, file_toolbar, file_menu)
        action("export_python", "Export Python", self.export_python, None, None, file_menu)
        action("export_field", "Export Field TXT", self.export_field_txt, None, None, file_menu)
        action("export_ftext", "Export BEAMER FTEXT", self.export_beamer_ftext, None, file_toolbar, file_menu)
        action("import_field", "Import Field TXT", self.import_field_txt, None, None, file_menu)
        action("image_to_gds", "Import Image → GDS…", self.import_image_as_gds, None, file_toolbar, file_menu, status_tip="Vectorize a photograph or drawing into scalable GDS polygons.")

        action("undo", "Undo", self.undo, "Ctrl+Z", edit_toolbar, edit_menu)
        action("redo", "Redo", self.redo, "Ctrl+Y", edit_toolbar, edit_menu)
        edit_toolbar.addSeparator()
        edit_menu.addSeparator()
        action("delete", "🗑 Delete Selected", self.delete_selected, "Delete", edit_toolbar, edit_menu)
        action("duplicate", "Duplicate", self.duplicate_selected, "Ctrl+D", edit_toolbar, edit_menu)
        action("group", "Group", self.group_selected, "Ctrl+G", edit_toolbar, edit_menu)
        action("ungroup", "Ungroup", self.ungroup_selected, "Ctrl+Shift+G", None, edit_menu)
        action("save_module", "Save module", self.save_selection_as_module, None, None, edit_menu)

        action(
            "fit_all",
            "Fit All",
            self.fit_layout,
            "F",
            view_toolbar,
            view_menu,
            status_tip="Fit the complete project in the canvas.",
        )
        action(
            "fit_design",
            "Fit Design",
            self.fit_design,
            "D",
            view_toolbar,
            view_menu,
            status_tip="Fit design geometry while ignoring the chip outline.",
        )
        action(
            "fit_selection",
            "Fit Selection",
            self.fit_selection,
            "Shift+F",
            view_toolbar,
            view_menu,
        )
        view_toolbar.addSeparator()
        view_menu.addSeparator()
        action("zoom_in", "Zoom +", self.zoom_in, "+", view_toolbar, view_menu)
        action("zoom_out", "Zoom −", self.zoom_out, "-", view_toolbar, view_menu)
        action("one_to_one", "1:1", self.one_to_one_view, "1", view_toolbar, view_menu)
        action(
            "center_origin",
            "Origin",
            self.center_origin,
            "0",
            view_toolbar,
            view_menu,
            status_tip="Center the canvas on coordinate (0, 0).",
        )
        action(
            "selection_zero",
            "Selection → 0",
            self.move_selection_to_origin,
            "Ctrl+0",
            view_toolbar,
            view_menu,
            status_tip="Move the center of the selected geometry to coordinate (0, 0).",
        )
        view_toolbar.addSeparator()
        view_menu.addSeparator()
        action(
            "show_grid",
            "Grid",
            self.toggle_grid,
            None,
            view_toolbar,
            view_menu,
            checkable=True,
            checked=True,
        )
        action(
            "show_axes",
            "Axes",
            self.toggle_axes,
            None,
            view_toolbar,
            view_menu,
            checkable=True,
            checked=True,
        )
        action(
            "show_rulers",
            "Rulers",
            self.toggle_rulers,
            None,
            view_toolbar,
            view_menu,
            checkable=True,
            checked=True,
            status_tip="Show X/Y coordinate rulers in micrometres.",
        )
        action("measure_ruler", "Measure", self.toggle_measure_ruler, "M", view_toolbar, view_menu, checkable=True, checked=False, status_tip="Click A–B for distance, then C for the included angle ABC.")
        action("smart_sketch", "Smart Sketch", self.toggle_smart_sketch, "P", layout_toolbar, layout_menu, checkable=True, checked=False, status_tip="Draw one or more rough strokes; turn Smart Sketch off to recognize and build editable layout components.")
        action("layout_threads", "Performance / CPU Threads…", self.open_layout_thread_settings, "Ctrl+Shift+T", None, layout_menu, status_tip="Choose the CPU thread count used by numerical geometry operations and background exports.")
        view_menu.addSeparator()
        view_menu.addAction(self.library_dock.toggleViewAction())
        view_menu.addAction(self.project_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        view_menu.addAction(self.layers_dock.toggleViewAction())
        view_menu.addAction(self.llm_dock.toggleViewAction())

        action("rotate_ccw", "Rotate +90°", lambda: self.rotate_selected(90), "Ctrl+R", layout_toolbar, layout_menu)
        action("rotate_cw", "Rotate −90°", lambda: self.rotate_selected(-90), None, None, layout_menu)
        action("mirror", "Mirror", self.mirror_selected, "Ctrl+M", layout_toolbar, layout_menu)
        layout_toolbar.addSeparator()
        layout_menu.addSeparator()
        action("layout_zero", "All GDS lower-left → (0, 0)", self.move_entire_layout_to_origin, "Ctrl+Shift+0", layout_toolbar, layout_menu, status_tip="Move every component together so the complete GDS bounding-box lower-left corner is exactly (0, 0).")
        action("rotate_all", "Rotate Entire GDS…", self.rotate_entire_layout_dialog, "Ctrl+Shift+R", layout_toolbar, layout_menu, status_tip="Rotate every component around the complete-layout center; no selection required.")
        action("rotate_all_90", "Rotate Entire GDS +90°", lambda: self.rotate_entire_layout(90), None, None, layout_menu)
        action("rotate_all_minus90", "Rotate Entire GDS −90°", lambda: self.rotate_entire_layout(-90), None, None, layout_menu)
        layout_toolbar.addSeparator()
        layout_menu.addSeparator()
        action("connect_nearest", "Connect Nearest", self.connect_selected_nearest, "C", layout_toolbar, layout_menu)
        action(
            "snap_ports",
            "Snap Ports",
            self.toggle_snap_ports,
            None,
            layout_toolbar,
            layout_menu,
            checkable=True,
            checked=True,
            status_tip="Snap the closest compatible connection points when a component is released.",
        )
        action(
            "auto_input",
            "Auto-connect Input",
            self.toggle_auto_connect_input,
            None,
            layout_toolbar,
            layout_menu,
            checkable=True,
            checked=True,
            status_tip="Prefer the dragged component input port and attach it to the closest compatible point.",
        )
        action(
            "show_ports",
            "Show Points",
            self.toggle_show_ports,
            None,
            layout_toolbar,
            layout_menu,
            checkable=True,
            checked=True,
        )
        layout_toolbar.addSeparator()
        layout_menu.addSeparator()
        action("align_vertical", "Align vertical", lambda: self.align_selected("x"), None, layout_toolbar, layout_menu)
        action("align_horizontal", "Align horizontal", lambda: self.align_selected("y"), None, layout_toolbar, layout_menu)
        action("distribute_x", "Distribute X", lambda: self.distribute_selected("x"), None, None, layout_menu)
        action("distribute_y", "Distribute Y", lambda: self.distribute_selected("y"), None, None, layout_menu)
        action("array", "Array", self.create_array, None, layout_toolbar, layout_menu)
        action("position_array", "Position Entire Array…", self.position_entire_array, None, layout_toolbar, layout_menu, status_tip="Move every component and linked E-beam cover in the selected array to an absolute center X/Y.")

        action("cover_selection", "Cover selection", self.create_ebeam_coverage, None, ebeam_toolbar, ebeam_menu)
        action("update_fields", "Update / prune", self.update_selected_ebeam, None, ebeam_toolbar, ebeam_menu)
        action("reset_fields", "Reset fields", self.reset_selected_ebeam_fields, None, None, ebeam_menu)
        action("remove_field", "Remove field", self.remove_active_field, None, None, ebeam_menu)
        action("field_order", "Assign order", self.assign_active_field_order, None, ebeam_toolbar, ebeam_menu)
        action("move_ebeam_block", "Move Entire E-beam Block…", self.position_entire_array, None, ebeam_toolbar, ebeam_menu, status_tip="Position a complete generated E-beam array or premade test block by its absolute center.")
        action("field_earlier", "Field Earlier", lambda: self.shift_active_field_order(-1), None, None, ebeam_menu)
        action("field_later", "Field Later", lambda: self.shift_active_field_order(1), None, None, ebeam_menu)
        ebeam_toolbar.addSeparator()
        action("play_fields", "▶ Play Fields", self.play_writefields, None, ebeam_toolbar, ebeam_menu)
        action("step_fields", "Step Field", self.step_writefields, None, ebeam_toolbar, ebeam_menu)
        action("stop_fields", "■ Stop", self.stop_writefields, None, ebeam_toolbar, ebeam_menu)

    def open_layout_thread_settings(self) -> None:
        dialog=QDialog(self);dialog.setWindowTitle("Layout Performance / CPU Threads");layout=QVBoxLayout(dialog);form=QFormLayout();threads=QSpinBox();threads.setRange(1,CPU_COUNT);threads.setValue(self.layout_threads);form.addRow(f"CPU threads (1–{CPU_COUNT})",threads);layout.addLayout(form)
        note=QLabel("Controls NumPy/BLAS geometry work and is inherited by background GDS/Python export workers. Qt scene updates remain on the GUI thread for stability.");note.setWordWrap(True);layout.addWidget(note);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        self.layout_threads=threads.value();self.settings.setValue("layout/cpu_threads",self.layout_threads);configure_acceleration(self.layout_threads);self.opengl_status_label.setText(("OpenGL" if self.view.opengl_enabled else "CPU canvas")+f" • {self.layout_threads} threads");self.statusBar().showMessage(f"Layout CPU thread limit set to {self.layout_threads}.",5000)

    # Project / undo
    # ------------------------------------------------------------------
    def snapshot(self) -> str:
        return json.dumps(
            {
                "components": self.components,
                "next_uid": self.next_uid,
                "next_group_id": self.next_group_id,
                "next_array_id": self.next_array_id,
                "next_module_instance_id": self.next_module_instance_id,
            },
            sort_keys=True,
        )

    def restore_snapshot(self, payload: str) -> None:
        data = json.loads(payload)
        self._restoring = True
        try:
            self.components = data["components"]
            self.next_uid = int(data.get("next_uid", 1))
            self.next_group_id = int(data.get("next_group_id", 1))
            self.next_array_id = int(data.get("next_array_id", 1))
            self.next_module_instance_id = int(data.get("next_module_instance_id", 1))
            self.rebuild_scene()
        finally:
            self._restoring = False

    def push_undo(self) -> None:
        if self._restoring:
            return
        current = self.snapshot()
        if not self.undo_stack or self.undo_stack[-1] != current:
            self.undo_stack.append(current)
            self.undo_stack = self.undo_stack[-60:]
        self.redo_stack.clear()

    def commit_interaction_snapshot(self, before: str) -> None:
        after = self.snapshot()
        if before != after:
            self.undo_stack.append(before)
            self.undo_stack = self.undo_stack[-60:]
            self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        current = self.snapshot()
        previous = self.undo_stack.pop()
        self.redo_stack.append(current)
        self.restore_snapshot(previous)

    def redo(self) -> None:
        if not self.redo_stack:
            return
        current = self.snapshot()
        upcoming = self.redo_stack.pop()
        self.undo_stack.append(current)
        self.restore_snapshot(upcoming)

    def new_project(self, checked: bool = False, push_undo: bool = True) -> None:
        if push_undo and self.components:
            answer = QMessageBox.question(
                self,
                "New project",
                "Clear the current project?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.components = []
        self.next_uid = 1
        self.next_group_id = 1
        self.next_array_id = 1
        self.next_module_instance_id = 1
        self.project_path = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.rebuild_scene()
        self.view.centerOn(0.0,0.0)
        self.update_zoom_status(self.view.current_zoom_percent())
        self.statusBar().showMessage("New blank project created. Add a chip outline manually from Chip / utility if needed.")

    def project_payload(self) -> dict[str, Any]:
        return {
            "format": "photonic-layout-native",
            "version": 1,
            "components": self.components,
            "next_uid": self.next_uid,
            "next_group_id": self.next_group_id,
            "next_array_id": self.next_array_id,
            "next_module_instance_id": self.next_module_instance_id,
            "layer_map": LAYER_NAME_MAP,
        }

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        self.project_path.write_text(json.dumps(self.project_payload(), indent=2))
        self.statusBar().showMessage(f"Saved project: {self.project_path}")

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(self.project_path or Path.home() / "photonic_layout_project.json"),
            "Layout project (*.json)",
        )
        if not path:
            return
        self.project_path = Path(path)
        self.save_project()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            str(Path.home()),
            "Layout project (*.json)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            components = data.get("components", data)
            if not isinstance(components, list):
                raise ValueError("The selected file does not contain a component list.")
            self.components = components
            self.next_uid = int(data.get("next_uid", max([int(c.get("uid", 0)) for c in components] + [0]) + 1))
            self.next_group_id = int(data.get("next_group_id", 1))
            self.next_array_id = int(data.get("next_array_id", 1))
            self.next_module_instance_id = int(data.get("next_module_instance_id", 1))
            self.project_path = Path(path)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.rebuild_scene()
            self.fit_layout()
            self.statusBar().showMessage(f"Opened project: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    # ------------------------------------------------------------------
    # Components / scene
    # ------------------------------------------------------------------
    def make_component(self, kind: str, x: float, y: float) -> dict[str, Any]:
        params = safe_json_copy(DEFAULT_COMPONENT_VALUES[kind])
        component = {
            "uid": self.next_uid,
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "orientation_deg": 0.0,
            "mirrored": False,
            "params": params,
            "attachment": None,
        }
        self.next_uid += 1
        return component

    def configure_test_block_sweeps(self, component: dict[str, Any]) -> bool:
        kind=str(component.get("kind",""));p=component["params"];options={
            "Double-ring test block":[("coupling_gap","Ring coupling gap",.5,1.,.1),("ring_radius","Ring radius",20.,200.,30.)],
            "Grating test block":[("pitch","Grating pitch",.73,.77,.005),("fill_factor","Fill factor",.47,.67,.05)],
            "Grating angle-taper test block":[("alpha_t","Aperture angle",22.,28.,1.),("taper_L","Taper length",20.,24.,1.)],
            "MMI + Reference test block":[("mmi_length","MMI length",26.,33.,1.),("taper_width","Waveguide taper width",2.5,3.1,.1)],
            "MMI split-combine test block":[("taper_length","MMI taper length",8.,12.,1.),("taper_width","Waveguide taper width",2.5,3.1,.1)],
            "Vertical-GC MZI test block":[("mmi_length","MMI length",27.,31.,1.)],
            "Vertical-GC MZI + CPW test block":[("mmi_length","MMI length",27.,31.,1.)],
            "Vertical-GC MZI + segmented electrode test block":[("mmi_length","MMI length",27.,31.,1.)],
        }.get(kind,[])
        if not options:return True
        dialog=QDialog(self);dialog.setWindowTitle(f"Configure parameter sweeps — {kind}");dialog.resize(1040,max(330,210+70*len(options)));dialog.setMinimumWidth(920);layout=QVBoxLayout(dialog);layout.setContentsMargins(18,18,18,18);layout.setSpacing(12);label=QLabel("Select each major parameter to sweep, then enter its inclusive start, stop, and step. Unchecked parameters remain nominal.");label.setWordWrap(True);label.setMinimumHeight(42);layout.addWidget(label)
        table=QTableWidget(len(options),5);table.setHorizontalHeaderLabels(["Sweep","Parameter","Start","Stop","Step"]);header=table.horizontalHeader();header.setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents);header.setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        for column in (2,3,4):header.setSectionResizeMode(column,QHeaderView.ResizeMode.Fixed);table.setColumnWidth(column,205)
        table.verticalHeader().setDefaultSectionSize(54);table.setMinimumWidth(880);editors=[]
        for row,(key,title,start,stop,step) in enumerate(options):
            check=QCheckBox();check.setChecked(True);check.setMinimumSize(44,38);table.setCellWidget(row,0,check);parameter_item=QTableWidgetItem(title);parameter_item.setToolTip(title);table.setItem(row,1,parameter_item);spins=[]
            for col,value in zip((2,3,4),(start,stop,step)):
                spin=QDoubleSpinBox();spin.setRange(-1e9,1e9);spin.setDecimals(6);spin.setMinimumWidth(190);spin.setMinimumHeight(40);spin.setAlignment(Qt.AlignmentFlag.AlignRight);spin.setKeyboardTracking(False);spin.setValue(value);spin.setToolTip(f"{title} — {table.horizontalHeaderItem(col).text()}");table.setCellWidget(row,col,spin);spins.append(spin)
            editors.append((key,check,*spins))
        table.setMinimumHeight(105+58*len(options));layout.addWidget(table);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.setMinimumHeight(44);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return False
        active=[]
        for key,check,start_box,stop_box,step_box in editors:
            if not check.isChecked():continue
            start,stop,step=start_box.value(),stop_box.value(),step_box.value()
            if step<=0 or stop<start:QMessageBox.critical(self,"Invalid sweep",f"{key}: stop must be ≥ start and step must be positive.");return False
            values=inclusive_sweep(start,stop,step);active.append(key)
            mapping={"pitch":("pitch_start","pitch_stop","pitch_step"),"fill_factor":("fill_start","fill_stop","fill_step"),"alpha_t":("angle_start_deg","angle_stop_deg","angle_step_deg"),"taper_L":("taper_length_start","taper_length_stop","taper_length_step"),"mmi_length":("mmi_length_start","mmi_length_stop","mmi_length_step"),"taper_length":("taper_length_start","taper_length_stop","taper_length_step"),"taper_width":("taper_width_start","taper_width_stop","taper_width_step")}
            if key=="coupling_gap":p["gap_values"]=",".join(f"{v:g}" for v in values)
            elif key=="ring_radius":p["radius_values"]=",".join(f"{v:g}" for v in values)
            else:
                names=mapping[key]
                for name,value in zip(names,(start,stop,step)):p[name]=value
            if key=="mmi_length" and kind.startswith("Vertical-GC"):p["mzi_count"]=len(values)
        p["sweep_parameters"]=active
        return True

    def configure_photonic_crystal(self, component: dict[str,Any]) -> bool:
        p=component["params"];dialog=QDialog(self);dialog.setWindowTitle("New photonic crystal");form=QFormLayout(dialog)
        combos={}
        for key,label,values in (("slab_shape","Slab shape",CHOICE_PARAMETERS["slab_shape"]),("device_type","Structure",CHOICE_PARAMETERS["device_type"]),("mask_tone","Mask output",CHOICE_PARAMETERS["mask_tone"]),("lattice","Lattice",CHOICE_PARAMETERS["lattice"]),("hole_shape","Hole shape",CHOICE_PARAMETERS["hole_shape"])):
            box=QComboBox();box.addItems(values);box.setCurrentText(str(p[key]));form.addRow(label,box);combos[key]=box
        numbers={}
        for key,label in (("length","Crystal/slab length (µm)"),("width","Crystal/slab width (µm)"),("pitch_x","Pitch X (µm)"),("pitch_y","Pitch Y (µm)"),("hole_radius_x","Hole radius X (µm)"),("hole_radius_y","Hole radius Y (µm)")):
            box=QDoubleSpinBox();box.setRange(.000001,1e7);box.setDecimals(6);box.setValue(float(p[key]));form.addRow(label,box);numbers[key]=box
        integers={}
        for key,label in (("columns","Hole columns"),("rows","Hole rows"),("defect_rows","Waveguide defect rows")):
            box=QSpinBox();box.setRange(1,10000);box.setValue(int(p[key]));form.addRow(label,box);integers[key]=box
        hint=QLabel("Positive outputs the hole mask only. Negative performs the Boolean subtraction and outputs the remaining slab. Line-defect waveguide removes the selected center rows.");hint.setWordWrap(True);form.addRow(hint)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return False
        for key,box in combos.items():p[key]=box.currentText()
        for key,box in numbers.items():p[key]=box.value()
        for key,box in integers.items():p[key]=box.value()
        return True

    def add_selected_library_component(self) -> None:
        item = self.library_tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        item_type, name = data
        if item_type == "module":
            self.add_saved_module(name)
            return
        center = scene_to_world_point(self.view.mapToScene(self.view.viewport().rect().center()))
        snapshot = self.snapshot()
        component = self.make_component(name, *center)
        if name=="Photonic crystal" and not self.configure_photonic_crystal(component):
            self.next_uid=max(1,self.next_uid-1);return
        if name.endswith("test block") and not self.configure_test_block_sweeps(component):
            self.next_uid=max(1,self.next_uid-1);return
        self.components.append(component)
        self.commit_interaction_snapshot(snapshot)
        self.scene.clearSelection()
        self.add_component_scene_item(component, selected=True)
        self.refresh_project_tree()
        self.on_scene_selection_changed()
        self.statusBar().showMessage(f"Added {name}.")

    def add_component_scene_item(self, component: dict[str, Any], selected: bool = False) -> QGraphicsItem:
        if component.get("kind") == "E-beam multipass":
            item: QGraphicsItem = EbeamContainerItem(self, component)
        else:
            item = ComponentGraphicsItem(self, component)
        uid = int(component["uid"])
        item.setData(10, uid)
        self.scene.addItem(item)
        self.items_by_uid[uid] = item
        item.setSelected(bool(selected))
        layer = self.default_display_layer(component)
        if not self.layer_visibility.get(layer, True):
            item.setVisible(False)
        return item

    def refresh_component_scene_item(self, uid: int, selected: bool = True) -> None:
        """Replace one changed component without clearing and rebuilding the scene."""
        component = self.component_by_uid(uid)
        if component is None:
            return
        old = self.items_by_uid.pop(int(uid), None)
        if old is not None and old.scene() is self.scene:
            self.scene.removeItem(old)
        self.add_component_scene_item(component, selected=selected)
        if not self.show_ports_enabled:
            item = self.items_by_uid.get(int(uid))
            if isinstance(item, ComponentGraphicsItem):
                for port in item.port_items:port.setVisible(False)
        self.refresh_project_tree()
        self.on_scene_selection_changed()

    def rebuild_scene(
        self,
        preserve_selection: bool = False,
        select_uids: list[int] | None = None,
    ) -> None:
        selected = (
            [int(item.data(10)) for item in self.scene.selectedItems() if item.data(10) is not None]
            if preserve_selection
            else []
        )
        if select_uids is not None:
            selected = [int(uid) for uid in select_uids]
        self.scene.clear()
        self.items_by_uid.clear()
        for component in self.components:
            try:
                self.add_component_scene_item(component)
            except Exception as exc:
                self.statusBar().showMessage(f"Preview error for UID {component.get('uid')}: {exc}")
        # Enforce geometry intersection every time a project is rebuilt or
        # reopened.  This prevents stale/manual write fields with no covered
        # geometry from surviving into the GUI, FTXT, or GDS exports.
        for component in self.components:
            if component.get("kind")!="E-beam multipass":continue
            before=set(map(str,component.get("params",{}).get("auto_pruned_field_keys",[])));self.prune_ebeam_component(component);after=set(map(str,component.get("params",{}).get("auto_pruned_field_keys",[])))
            if after!=before:
                item=self.items_by_uid.get(int(component["uid"]))
                if isinstance(item,EbeamContainerItem):item.rebuild_fields()
        for uid in selected:
            if uid in self.items_by_uid:
                self.items_by_uid[uid].setSelected(True)
        self.refresh_field_numbers()
        self.refresh_project_tree()
        for layer, visible in self.layer_visibility.items():
            if not visible:
                self.set_layer_visible(layer, False)
        if not self.show_ports_enabled:
            self.toggle_show_ports(False)
        self.on_scene_selection_changed()

    def selected_component_items(self) -> list[QGraphicsItem]:
        result = []
        seen: set[int] = set()
        for item in self.scene.selectedItems():
            current = item
            while current is not None and current.data(10) is None:
                current = current.parentItem()
            if current is None:
                continue
            uid = int(current.data(10))
            if uid not in seen and uid in self.items_by_uid:
                seen.add(uid)
                result.append(self.items_by_uid[uid])
        return result

    def selected_components(self) -> list[dict[str, Any]]:
        selected_uids = {int(item.data(10)) for item in self.selected_component_items()}
        return [component for component in self.components if int(component["uid"]) in selected_uids]

    def component_by_uid(self, uid: int) -> dict[str, Any] | None:
        return next((component for component in self.components if int(component["uid"]) == int(uid)), None)

    def select_group(self, group_id: str) -> None:
        for component in self.components:
            if str(component.get("group_id", "")) == str(group_id):
                item = self.items_by_uid.get(int(component["uid"]))
                if item:
                    item.setSelected(True)

    def move_group_with_primary(self, primary: ComponentGraphicsItem) -> None:
        group_id = primary.component.get("group_id")
        if not group_id or self._group_move_guard:
            return
        previous = primary.component.pop("_last_group_position", None)
        current = (float(primary.component["x"]), float(primary.component["y"]))
        if previous is None:
            primary.component["_last_group_position"] = list(current)
            return
        dx = current[0] - float(previous[0])
        dy = current[1] - float(previous[1])
        primary.component["_last_group_position"] = list(current)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return
        self._group_move_guard = True
        try:
            for component in self.components:
                if component is primary.component:
                    continue
                if str(component.get("group_id", "")) != str(group_id):
                    continue
                component["x"] = float(component.get("x", 0.0)) + dx
                component["y"] = float(component.get("y", 0.0)) + dy
                item = self.items_by_uid.get(int(component["uid"]))
                if item:
                    item.setPos(float(component["x"]), -float(component["y"]))
        finally:
            self._group_move_guard = False

    def on_scene_selection_changed(self) -> None:
        self.refresh_project_tree_selection()
        items = self.selected_component_items()
        if hasattr(self, "selection_status_label"):
            selected_fields = sum(isinstance(item, WriteFieldItem) for item in self.scene.selectedItems())
            suffix = f" + {selected_fields} field" if selected_fields == 1 else (f" + {selected_fields} fields" if selected_fields else "")
            self.selection_status_label.setText(f"Selection: {len(items)}{suffix}")
        if len(items) == 1:
            self.show_component_properties(self.component_by_uid(int(items[0].data(10))))
        elif len(items) > 1:
            self.show_multi_selection_properties(len(items))
        else:
            self.show_no_selection_properties()

    # ------------------------------------------------------------------
    # Property editor
    # ------------------------------------------------------------------
    def clear_properties(self) -> None:
        while self.properties_form.rowCount():
            self.properties_form.removeRow(0)
        self.parameter_widgets.clear()
        self.update_ebeam_button.setVisible(False)
        self.module_variables_button.setVisible(False)

    def show_no_selection_properties(self) -> None:
        self.clear_properties()
        label = QLabel(
            "Select a component or write field.\n\n"
            "Native rendering uses QGraphicsView with a stable raster viewport. "
            "GDS generation and flattening run in a separate CPU process."
        )
        label.setWordWrap(True)
        self.properties_form.addRow(label)
        self.apply_properties_button.setEnabled(False)

    def show_multi_selection_properties(self, count: int) -> None:
        self.clear_properties()
        self.properties_form.addRow(QLabel(f"{count} components selected."))
        self.properties_form.addRow(QLabel("Use Group, Align, Distribute, Array, Rotate, or Mirror."))
        self.apply_properties_button.setEnabled(False)

    def make_parameter_widget(self, key: str, spec: list[Any], value: Any) -> QWidget:
        kind = spec[0]
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            return widget
        if kind == "choice":
            widget = QComboBox()
            widget.addItems([str(choice) for choice in spec[2]])
            widget.setCurrentText(str(value))
            return widget
        if kind == "int":
            widget = QSpinBox()
            widget.setRange(-2_000_000_000, 2_000_000_000)
            widget.setValue(int(value))
            return widget
        if kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(9)
            widget.setSingleStep(max(abs(float(value)) * 0.02, 0.001))
            widget.setValue(float(value))
            return widget
        widget = QLineEdit(str(value))
        return widget

    def read_parameter_widget(self, widget: QWidget, spec_type: str) -> Any:
        if spec_type == "bool":
            return bool(widget.isChecked())
        if spec_type == "choice":
            return str(widget.currentText())
        if spec_type == "int":
            return int(widget.value())
        if spec_type == "float":
            return float(widget.value())
        return str(widget.text())

    def show_component_properties(self, component: dict[str, Any] | None) -> None:
        self.clear_properties()
        if component is None:
            self.show_no_selection_properties()
            return
        title = QLabel(f"<b>{component.get('kind')}</b> · UID {component.get('uid')}")
        self.properties_form.addRow(title)
        for key, value in (
            ("x", component.get("x", 0.0)),
            ("y", component.get("y", 0.0)),
            ("orientation_deg", component.get("orientation_deg", 0.0)),
        ):
            spec = ["float", value]
            widget = self.make_parameter_widget(key, spec, value)
            self.properties_form.addRow(key, widget)
            self.parameter_widgets[key] = (widget, "float")
        mirrored = QCheckBox()
        mirrored.setChecked(bool(component.get("mirrored", False)))
        self.properties_form.addRow("mirrored", mirrored)
        self.parameter_widgets["mirrored"] = (mirrored, "bool")

        if component.get("kind") == "E-beam multipass":
            self.update_ebeam_button.setVisible(True)
            field_info = QLabel(
                "Field defaults: 520 µm, 10 µm clearance. "
                "Click an individual red field to move or renumber it."
            )
            field_info.setWordWrap(True)
            self.properties_form.addRow(field_info)

        specs = COMPONENT_SPECS.get(component.get("kind"), {})
        for key, value in component.get("params", {}).items():
            if key in {
                "manual_field_offsets",
                "manual_field_order",
                "removed_field_keys",
                "auto_pruned_field_keys",
                "explicit_fields",
                "sweep_parameters",
                "polygons",
            }:
                continue
            spec = specs.get(key)
            if spec is None:
                if isinstance(value, bool):
                    spec = ["bool", value]
                elif isinstance(value, int):
                    spec = ["int", value]
                elif isinstance(value, float):
                    spec = ["float", value]
                else:
                    spec = ["string", value]
            widget = self.make_parameter_widget(key, spec, value)
            self.properties_form.addRow(key, widget)
            self.parameter_widgets[f"params.{key}"] = (widget, spec[0])

        active = self.active_field
        if component.get("kind") == "E-beam multipass" and active and int(active[0]) == int(component["uid"]):
            item = self.items_by_uid.get(int(component["uid"]))
            if isinstance(item, EbeamContainerItem) and active[1] in item.field_items:
                field_item = item.field_items[active[1]]
                order = self.global_field_order(int(component["uid"]), active[1])
                label = QLabel(f"<b>Selected field:</b> {order} · {active[1]}")
                self.properties_form.addRow(label)
                center_world = scene_to_world_point(field_item.mapToScene(QPointF(0, 0)))
                x_widget = self.make_parameter_widget("field_x", ["float", center_world[0]], center_world[0])
                y_widget = self.make_parameter_widget("field_y", ["float", center_world[1]], center_world[1])
                order_widget = self.make_parameter_widget("field_order", ["int", order], order)
                self.properties_form.addRow("field center X", x_widget)
                self.properties_form.addRow("field center Y", y_widget)
                self.properties_form.addRow("global field order", order_widget)
                self.parameter_widgets["active_field_x"] = (x_widget, "float")
                self.parameter_widgets["active_field_y"] = (y_widget, "float")
                self.parameter_widgets["active_field_order"] = (order_widget, "int")

        module_instance = component.get("module_instance_id")
        if module_instance:
            self.module_variables_button.setVisible(True)
        self.apply_properties_button.setEnabled(True)

    def apply_properties(self) -> None:
        components = self.selected_components()
        if len(components) != 1:
            return
        component = components[0]
        snapshot = self.snapshot()
        try:
            for key, (widget, spec_type) in self.parameter_widgets.items():
                value = self.read_parameter_widget(widget, spec_type)
                if key.startswith("params."):
                    component["params"][key.split(".", 1)[1]] = value
                elif key == "active_field_x" or key == "active_field_y" or key == "active_field_order":
                    continue
                else:
                    component[key] = value
            if (
                component.get("kind") == "E-beam multipass"
                and self.active_field
                and int(self.active_field[0]) == int(component["uid"])
            ):
                item = self.items_by_uid.get(int(component["uid"]))
                if isinstance(item, EbeamContainerItem):
                    field_item = item.field_items.get(self.active_field[1])
                    if field_item:
                        field_x = float(self.read_parameter_widget(*self.parameter_widgets["active_field_x"]))
                        field_y = float(self.read_parameter_widget(*self.parameter_widgets["active_field_y"]))
                        local_scene = item.mapFromScene(world_to_scene_point((field_x, field_y)))
                        field_item.setPos(local_scene)
                        desired = int(self.read_parameter_widget(*self.parameter_widgets["active_field_order"]))
                        self.set_field_global_order(int(component["uid"]), self.active_field[1], desired)
            self.commit_interaction_snapshot(snapshot)
            if component.get("kind") == "E-beam multipass":
                self.rebuild_scene(preserve_selection=True)
            else:
                self.refresh_component_scene_item(int(component["uid"]), selected=True)
            self.statusBar().showMessage("Properties updated.")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid parameter", str(exc))

    # ------------------------------------------------------------------
    # Project tree
    # ------------------------------------------------------------------
    def refresh_project_tree(self) -> None:
        selected_uids = {int(item.data(10)) for item in self.selected_component_items()}
        self.project_tree.blockSignals(True)
        self.project_tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for component in self.components:
            group_id = component.get("group_id")
            if group_id:
                group_key = str(group_id)
                parent = groups.get(group_key)
                if parent is None:
                    parent = QTreeWidgetItem([f"Group {group_key}", "", ""])
                    groups[group_key] = parent
                    self.project_tree.addTopLevelItem(parent)
            else:
                parent = self.project_tree.invisibleRootItem()
            layer = self.default_display_layer(component)
            child = QTreeWidgetItem(
                [
                    str(component.get("module_name") or component.get("kind")),
                    str(component.get("uid")),
                    f"{layer}: {LAYER_NAME_MAP.get(layer, '')}",
                ]
            )
            child.setData(0, Qt.ItemDataRole.UserRole, int(component["uid"]))
            parent.addChild(child)
            if int(component["uid"]) in selected_uids:
                child.setSelected(True)
            if group_id:
                parent.setExpanded(True)
        self.project_tree.blockSignals(False)

    def default_display_layer(self, component: dict[str, Any]) -> int:
        kind = component.get("kind")
        if kind == "E-beam multipass":
            return EBEAM_LAYER
        if kind in RF_COMPONENT_KINDS or kind == "MZI + CPW module":
            return RF_LAYER
        if kind in MARKER_COMPONENT_KINDS:
            return MARKER_LAYER
        if kind == "Grating coupler":
            return GC_LAYER
        return PHOTONIC_LAYER

    def refresh_project_tree_selection(self) -> None:
        selected_uids = {int(item.data(10)) for item in self.selected_component_items()}
        self.project_tree.blockSignals(True)
        iterator = self.project_tree.invisibleRootItem()
        stack = [iterator.child(i) for i in range(iterator.childCount())]
        while stack:
            item = stack.pop()
            uid = item.data(0, Qt.ItemDataRole.UserRole)
            if uid is not None:
                item.setSelected(int(uid) in selected_uids)
            stack.extend(item.child(i) for i in range(item.childCount()))
        self.project_tree.blockSignals(False)

    def project_tree_selection_changed(self) -> None:
        uids = {
            int(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.project_tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole) is not None
        }
        self.scene.blockSignals(True)
        try:
            for uid, graphics_item in self.items_by_uid.items():
                graphics_item.setSelected(uid in uids)
        finally:
            self.scene.blockSignals(False)
        self.on_scene_selection_changed()

    def project_tree_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        uid = item.data(0, Qt.ItemDataRole.UserRole)
        if uid is None or int(uid) not in self.items_by_uid:
            return
        graphics_item = self.items_by_uid[int(uid)]
        self.view.centerOn(graphics_item)

    def show_project_context_menu(self, position: QPoint) -> None:
        clicked = self.project_tree.itemAt(position)
        if clicked is not None:
            uid = clicked.data(0, Qt.ItemDataRole.UserRole)
            if uid is not None and not clicked.isSelected():
                self.project_tree.clearSelection()
                clicked.setSelected(True)
                self.project_tree_selection_changed()

        menu = QMenu(self)
        _force_dark_popup(menu)
        go_action = menu.addAction("Go to object")
        duplicate_action = menu.addAction("Duplicate")
        array_action = menu.addAction("Make an array")
        lattice_action = menu.addAction("Create photonic-crystal lattice…")
        boolean_menu=menu.addMenu("Boolean operation")
        boolean_actions={boolean_menu.addAction(label):op for label,op in (("Union","union"),("Difference (first minus rest)","difference"),("Intersection","intersection"),("XOR","xor"))}
        save_module_action = menu.addAction("Add selection to User modules…")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        delete_action.setProperty("danger", True)
        has_selection = bool(self.selected_components())
        for action_item in (go_action, duplicate_action, array_action, lattice_action, save_module_action, delete_action):
            action_item.setEnabled(has_selection)
        boolean_menu.setEnabled(len(self.selected_components())>=2)

        chosen = menu.exec(self.project_tree.viewport().mapToGlobal(position))
        if chosen is go_action:
            self.fit_selection()
        elif chosen is duplicate_action:
            self.duplicate_selected()
        elif chosen is array_action:
            self.create_array()
        elif chosen is lattice_action:
            self.create_photonic_crystal_lattice()
        elif chosen in boolean_actions:
            self.boolean_selected_geometry(boolean_actions[chosen])
        elif chosen is save_module_action:
            self.save_selection_as_module()
        elif chosen is delete_action:
            self.delete_selected()

    # ------------------------------------------------------------------
    # Edit / layout operations
    # ------------------------------------------------------------------
    def delete_selected(self) -> None:
        if any(isinstance(item, WriteFieldItem) for item in self.scene.selectedItems()):
            if self.active_field:
                self.remove_active_field()
            return
        selected = {int(c["uid"]) for c in self.selected_components()}
        if not selected and self.active_field:
            self.remove_active_field()
            return
        if not selected:
            return
        snapshot = self.snapshot()
        self.components = [component for component in self.components if int(component["uid"]) not in selected]
        for component in self.components:
            if component.get("kind") == "E-beam multipass":
                sources = [int(uid) for uid in component.get("coverage_source_uids", []) if int(uid) not in selected]
                component["coverage_source_uids"] = sources
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene()

    def duplicate_selected(self) -> None:
        originals = self.selected_components()
        if not originals:
            return
        snapshot = self.snapshot()
        uid_map: dict[int, int] = {}
        duplicates: list[dict[str, Any]] = []
        for original in originals:
            duplicate = safe_json_copy(original)
            old_uid = int(original["uid"])
            duplicate["uid"] = self.next_uid
            self.next_uid += 1
            uid_map[old_uid] = int(duplicate["uid"])
            duplicate["x"] = float(duplicate.get("x", 0.0)) + 100.0
            duplicate["y"] = float(duplicate.get("y", 0.0)) - 100.0
            duplicate.pop("group_id", None)
            duplicate.pop("array_group_id", None)
            duplicate.pop("_last_group_position", None)
            duplicates.append(duplicate)
        for duplicate in duplicates:
            if duplicate.get("coverage_source_uids"):
                duplicate["coverage_source_uids"] = [
                    uid_map.get(int(uid), int(uid)) for uid in duplicate["coverage_source_uids"]
                ]
        self.components.extend(duplicates)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(c["uid"]) for c in duplicates])

    def create_photonic_crystal_lattice(self) -> None:
        originals=[c for c in self.selected_components() if c.get("kind")!="E-beam multipass"]
        if not originals:
            QMessageBox.information(self,"Photonic-crystal lattice","Select one or more primitive structures first.");return
        dialog=QDialog(self);dialog.setWindowTitle("Create photonic-crystal lattice");form=QFormLayout(dialog)
        pattern=QComboBox();pattern.addItems(["Triangular / hexagonal","Square","FCC (111) projection","FCC (100) projection","BCC (110) projection"])
        columns=QSpinBox();columns.setRange(1,500);columns.setValue(20);rows=QSpinBox();rows.setRange(1,500);rows.setValue(20)
        pitch=QDoubleSpinBox();pitch.setRange(.000001,1e6);pitch.setDecimals(6);pitch.setValue(.42);pitch.setSuffix(" µm")
        form.addRow("Lattice",pattern);form.addRow("Columns",columns);form.addRow("Rows",rows);form.addRow("Lattice constant",pitch)
        note=QLabel("The selected structure is used as the lattice basis. FCC/BCC choices add the appropriate projected basis sites.");note.setWordWrap(True);form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        mode=pattern.currentText();nx=columns.value();ny=rows.value();a=pitch.value();snapshot=self.snapshot();created=[];group_id=f"G{self.next_group_id}";self.next_group_id+=1
        if mode in {"Triangular / hexagonal","FCC (111) projection"}:dx=a;dy=a*math.sqrt(3)/2;basis=[(0.0,0.0)]
        elif mode=="Square":dx=dy=a;basis=[(0.0,0.0)]
        elif mode=="FCC (100) projection":dx=dy=a;basis=[(0.0,0.0),(.5,.5)]
        else:dx=a;dy=a/math.sqrt(2);basis=[(0.0,0.0),(.5,.5)]
        for row in range(ny):
            row_shift=.5 if mode in {"Triangular / hexagonal","FCC (111) projection"} and row%2 else 0.0
            for col in range(nx):
                for bx,by in basis:
                    if row==0 and col==0 and bx==0 and by==0:
                        for original in originals:original["group_id"]=group_id
                        continue
                    for original in originals:
                        clone=copy.deepcopy(original);clone["uid"]=self.next_uid;self.next_uid+=1;clone["x"]=float(original["x"])+(col+row_shift+bx)*dx;clone["y"]=float(original["y"])+(row+by)*dy;clone["group_id"]=group_id;clone["attachment"]=None;clone.pop("array_group_id",None);created.append(clone)
        self.components.extend(created);self.commit_interaction_snapshot(snapshot);self.rebuild_scene(select_uids=[int(c["uid"]) for c in originals+created]);self.statusBar().showMessage(f"Created {mode} lattice with {len(originals)+len(created)} elements.",7000)

    def boolean_selected_geometry(self, operation: str) -> None:
        selected=[c for c in self.selected_components() if c.get("kind")!="E-beam multipass"]
        if len(selected)<2:
            QMessageBox.information(self,"Boolean operation","Select at least two geometry components.");return
        operands=[]
        try:
            for index,component in enumerate(selected):
                cell=gdstk.Cell(f"BOOL_SOURCE_{index}");_add_component_geometry_to_cell(component,cell);polys=list(cell.polygons)
                for path in cell.paths:polys.extend(path.to_polygons())
                operands.append(polys)
            if operation=="union":result=gdstk.boolean([p for group in operands for p in group],[],"or",layer=PHOTONIC_LAYER,datatype=0)
            elif operation=="difference":result=gdstk.boolean(operands[0],[p for group in operands[1:] for p in group],"not",layer=PHOTONIC_LAYER,datatype=0)
            elif operation=="intersection":
                result=operands[0]
                for group in operands[1:]:result=gdstk.boolean(result,group,"and",layer=PHOTONIC_LAYER,datatype=0)
            else:result=gdstk.boolean(operands[0],[p for group in operands[1:] for p in group],"xor",layer=PHOTONIC_LAYER,datatype=0)
            if not result:raise ValueError("The operation produced no geometry.")
        except Exception as exc:
            QMessageBox.critical(self,"Boolean operation failed",str(exc));return
        snapshot=self.snapshot();selected_ids={int(c["uid"]) for c in selected};self.components=[c for c in self.components if int(c["uid"]) not in selected_ids];points=[np.asarray(poly.points,float).tolist() for poly in result];component=self.make_component("Boolean geometry",0.0,0.0);component["params"].update({"polygons":points,"operation":operation});self.components.append(component);self.commit_interaction_snapshot(snapshot);self.rebuild_scene(select_uids=[int(component["uid"])]);self.statusBar().showMessage(f"Boolean {operation} created {len(points)} polygon(s).",6000)

    def import_image_as_gds(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"Import image as GDS",str(Path.home()),"Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:return
        reader=QImageReader(path);reader.setAutoTransform(True);decoded=reader.read();pixmap=QPixmap.fromImage(decoded)
        if pixmap.isNull():QMessageBox.critical(self,"Image import","The selected image could not be decoded.");return
        dialog=QDialog(self);dialog.setWindowTitle("Image → GDS vectorization");form=QFormLayout(dialog)
        width_box=QDoubleSpinBox();width_box.setRange(.001,1e7);width_box.setDecimals(3);width_box.setValue(500);width_box.setSuffix(" µm")
        preset=QComboBox();preset.addItems(["Photo edges — recommended","Dark silhouette / logo","Light silhouette / logo"]);detail=QSpinBox();detail.setRange(24,384);detail.setValue(192);threshold=QSpinBox();threshold.setRange(0,255);threshold.setValue(38);invert=QCheckBox("Invert foreground/background");simplify=QCheckBox("Merge adjacent pixels into polygons");simplify.setChecked(True)
        form.addRow("Conversion preset",preset);form.addRow("Final GDS width",width_box);form.addRow("Raster detail (max pixels)",detail);form.addRow("Threshold / edge strength",threshold);form.addRow(invert);form.addRow(simplify)
        note=QLabel("Photographs are converted to a one-bit engraving mask. Higher detail resembles the source more closely but produces larger GDS files.");note.setWordWrap(True);form.addRow(note);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        image=pixmap.toImage();maximum=detail.value();image=image.scaled(maximum,maximum,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation);w,h=image.width(),image.height();scale=width_box.value()/max(w,1);rectangles=[];cut=threshold.value();inverse=invert.isChecked();values=np.empty((h,w),dtype=np.uint8)
        for row in range(h):
            for col in range(w):values[row,col]=image.pixelColor(col,row).value()
        edge_mode=preset.currentIndex()==0
        if edge_mode:
            gx=np.zeros_like(values,dtype=float);gy=np.zeros_like(values,dtype=float);gx[:,1:-1]=values[:,2:].astype(float)-values[:,:-2].astype(float);gy[1:-1,:]=values[2:,:].astype(float)-values[:-2,:].astype(float);mask=np.hypot(gx,gy)>cut
        else:
            mask=values<cut
            if preset.currentIndex()==2:mask=~mask
        if inverse:mask=~mask
        for row in range(h):
            run_start=None
            for col in range(w+1):
                foreground=False
                if col<w:
                    foreground=bool(mask[row,col])
                if foreground and run_start is None:run_start=col
                elif not foreground and run_start is not None:
                    x0=(run_start-w/2)*scale;x1=(col-w/2)*scale;y0=(h/2-row-1)*scale;y1=(h/2-row)*scale;rectangles.append(gdstk.rectangle((x0,y0),(x1,y1)));run_start=None
        if not rectangles:QMessageBox.information(self,"Image import","No foreground pixels were found. Adjust threshold or inversion.");return
        try:result=gdstk.boolean(rectangles,[],"or",layer=PHOTONIC_LAYER,datatype=0) if simplify.isChecked() else rectangles
        except Exception as exc:QMessageBox.critical(self,"Image vectorization",str(exc));return
        center=scene_to_world_point(self.view.mapToScene(self.view.viewport().rect().center()));component=self.make_component("Boolean geometry",*center);component["params"].update({"polygons":[np.asarray(poly.points,float).tolist() for poly in result],"operation":"image vectorization","source_image":Path(path).name,"image_preset":preset.currentText(),"image_detail":detail.value(),"image_threshold":threshold.value()});snapshot=self.snapshot();self.components.append(component);self.commit_interaction_snapshot(snapshot);self.rebuild_scene(select_uids=[int(component["uid"])]);self.statusBar().showMessage(f"Vectorized {Path(path).name} into {len(result)} GDS polygon(s) at {width_box.value():g} µm width.",10000)

    def group_selected(self) -> None:
        selected = self.selected_components()
        if len(selected) < 2:
            return
        snapshot = self.snapshot()
        group_id = f"G{self.next_group_id}"
        self.next_group_id += 1
        for component in selected:
            component["group_id"] = group_id
            component["_last_group_position"] = [component["x"], component["y"]]
        self.commit_interaction_snapshot(snapshot)
        self.refresh_project_tree()
        self.statusBar().showMessage(f"Grouped {len(selected)} components as {group_id}.")

    def ungroup_selected(self) -> None:
        selected = self.selected_components()
        if not selected:
            return
        snapshot = self.snapshot()
        group_ids = {component.get("group_id") for component in selected if component.get("group_id")}
        for component in self.components:
            if component.get("group_id") in group_ids:
                component.pop("group_id", None)
                component.pop("_last_group_position", None)
        self.commit_interaction_snapshot(snapshot)
        self.refresh_project_tree()

    def rotate_selected(self, angle: float) -> None:
        selected = self.selected_components()
        if not selected:
            return
        snapshot = self.snapshot()
        for component in selected:
            component["orientation_deg"] = (float(component.get("orientation_deg", 0.0)) + angle) % 360.0
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)

    def move_entire_layout_to_origin(self) -> None:
        if not self.components:
            return
        try:
            library=resolve_and_build(safe_json_copy(self.components));bbox,_=library_bbox_and_center(library);dx,dy=-float(bbox[0][0]),-float(bbox[0][1])
            snapshot=self.snapshot()
            for component in self.components:
                component['x']=float(component.get('x',0))+dx;component['y']=float(component.get('y',0))+dy
            self.commit_interaction_snapshot(snapshot);self.rebuild_scene();self.center_origin();self.statusBar().showMessage(f'Moved entire GDS by ({dx:.6g}, {dy:.6g}) µm; lower-left is now (0, 0).',8000)
        except Exception as exc:
            QMessageBox.critical(self,'Could not zero complete GDS',str(exc))

    def rotate_entire_layout_dialog(self) -> None:
        angle,ok=QInputDialog.getDouble(self,'Rotate Entire GDS','Rotation angle (degrees, counter-clockwise):',90.0,-360000.0,360000.0,6)
        if ok:self.rotate_entire_layout(angle)

    def rotate_entire_layout(self, angle: float) -> None:
        if not self.components:return
        try:
            snapshot=self.snapshot();rotated,pivot=rotate_components_layout(self.components,float(angle),'center');self.components=rotated;self.commit_interaction_snapshot(snapshot);self.rebuild_scene();self.fit_layout();self.statusBar().showMessage(f'Rotated entire GDS {float(angle):g}° around ({pivot[0]:.6g}, {pivot[1]:.6g}) µm.',8000)
        except Exception as exc:
            QMessageBox.critical(self,'Could not rotate complete GDS',str(exc))

    def mirror_selected(self) -> None:
        selected = self.selected_components()
        if not selected:
            return
        snapshot = self.snapshot()
        for component in selected:
            component["mirrored"] = not bool(component.get("mirrored", False))
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)

    def align_selected(self, axis: str) -> None:
        selected = self.selected_components()
        if len(selected) < 2:
            return
        snapshot = self.snapshot()
        target = float(selected[0][axis])
        for component in selected[1:]:
            component[axis] = target
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)

    def distribute_selected(self, axis: str) -> None:
        selected = self.selected_components()
        if len(selected) < 3:
            return
        snapshot = self.snapshot()
        selected.sort(key=lambda component: float(component[axis]))
        start = float(selected[0][axis])
        stop = float(selected[-1][axis])
        spacing = (stop - start) / (len(selected) - 1)
        for index, component in enumerate(selected):
            component[axis] = start + index * spacing
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)

    def fit_rect(self, bounds: QRectF, margin: float = 80.0) -> None:
        if bounds.isValid() and not bounds.isNull() and bounds.width() > 0 and bounds.height() > 0:
            self.view.fitInView(
                bounds.adjusted(-margin, -margin, margin, margin),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self.view.emit_zoom()

    def fit_layout(self) -> None:
        self.fit_rect(self.scene.itemsBoundingRect(), 100.0)

    def fit_design(self) -> None:
        bounds: QRectF | None = None
        for component in self.components:
            if component.get("kind") == "Chip outline":
                continue
            item = self.items_by_uid.get(int(component["uid"]))
            if item is None or not item.isVisible():
                continue
            rect = item.sceneBoundingRect()
            bounds = rect if bounds is None else bounds.united(rect)
        if bounds is None:
            self.fit_layout()
        else:
            self.fit_rect(bounds, 80.0)

    def fit_selection(self) -> None:
        selected = self.scene.selectedItems()
        if not selected:
            self.fit_design()
            return
        bounds: QRectF | None = None
        for item in selected:
            rect = item.sceneBoundingRect()
            bounds = rect if bounds is None else bounds.united(rect)
        if bounds is not None:
            self.fit_rect(bounds, 50.0)

    def zoom_in(self) -> None:
        self.view.zoom_by(1.25)

    def zoom_out(self) -> None:
        self.view.zoom_by(0.8)

    def one_to_one_view(self) -> None:
        self.view.reset_one_to_one()

    def center_origin(self) -> None:
        self.view.centerOn(0.0, 0.0)
        self.statusBar().showMessage("View centered on coordinate (0, 0).")

    def move_selection_to_origin(self) -> None:
        selected = self.selected_components()
        if not selected:
            self.center_origin()
            return
        snapshot = self.snapshot()
        items = [self.items_by_uid[int(component["uid"])] for component in selected if int(component["uid"]) in self.items_by_uid]
        bounds: QRectF | None = None
        for item in items:
            rect = item.sceneBoundingRect()
            bounds = rect if bounds is None else bounds.united(rect)
        if bounds is None:
            return
        world_center = scene_to_world_point(bounds.center())
        for component in selected:
            component["x"] = float(component.get("x", 0.0)) - world_center[0]
            component["y"] = float(component.get("y", 0.0)) - world_center[1]
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"]) for component in selected])
        self.center_origin()
        self.statusBar().showMessage("Moved selected geometry center to coordinate (0, 0).")

    def toggle_grid(self, enabled: bool) -> None:
        self.view.show_grid = bool(enabled)
        self.view.viewport().update()

    def toggle_axes(self, enabled: bool) -> None:
        self.view.show_axes = bool(enabled)
        self.view.viewport().update()

    def toggle_rulers(self, enabled: bool) -> None:
        self.view.show_rulers = bool(enabled)
        self.view.viewport().update()

    def toggle_measure_ruler(self, enabled: bool) -> None:
        self.view.measure_mode=bool(enabled)
        if not enabled:self.view.measure_start=None;self.view.measure_end=None;self.view.measure_third=None
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor);self.view.viewport().update();self.statusBar().showMessage("Measurement ruler: click A–B for distance, then C for angle ABC." if enabled else "Measurement ruler cleared.",5000)

    def toggle_smart_sketch(self, enabled: bool) -> None:
        if enabled:
            self.view.measure_mode=False
            if "measure_ruler" in self.actions:self.actions["measure_ruler"].setChecked(False)
            self.view.sketch_strokes=[];self.view._active_sketch=None;self.view.sketch_mode=True;self.view.setCursor(Qt.CursorShape.CrossCursor);self.statusBar().showMessage("Smart Sketch: draw rough geometry with one or more strokes. Turn Smart Sketch off (or press P) to recognize it.",8000)
        else:
            self.view.sketch_mode=False;self.view.setCursor(Qt.CursorShape.ArrowCursor);self.recognize_smart_sketch()
        self.view.viewport().update()

    def recognize_smart_sketch(self) -> None:
        strokes=self.view.sketch_strokes;self.view.sketch_strokes=[]
        if not strokes:return
        arrays=[np.array([[point.x(),-point.y()] for point in stroke],float) for stroke in strokes if len(stroke)>=2]
        if not arrays:return
        def rdp(points:np.ndarray,epsilon:float)->np.ndarray:
            if len(points)<3:return points
            start,end=points[0],points[-1];line=end-start;den=float(np.linalg.norm(line))
            distances=np.linalg.norm(points-start,axis=1) if den<1e-12 else np.abs(line[0]*(start[1]-points[:,1])-(start[0]-points[:,0])*line[1])/den
            index=int(np.argmax(distances));maximum=float(distances[index])
            if maximum<=epsilon:return np.vstack((start,end))
            return np.vstack((rdp(points[:index+1],epsilon)[:-1],rdp(points[index:],epsilon)))
        snapshot=self.snapshot();created=[];descriptions=[]
        for raw in arrays:
            extent=max(float(np.ptp(raw[:,0])),float(np.ptp(raw[:,1])),1e-6);points=raw
            if len(points)>5:
                padded=np.pad(points,((2,2),(0,0)),mode="edge");points=np.vstack([padded[i:i+5].mean(axis=0) for i in range(len(points))]);points[0]=raw[0];points[-1]=raw[-1]
            path_len=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum());closed=np.linalg.norm(points[-1]-points[0])<.07*max(path_len,1e-9)
            simplified=rdp(points,extent*.018)
            if closed:
                simplified[-1]=simplified[0];split=int(np.argmax(np.linalg.norm(points-points[0],axis=1)));simplified=np.vstack((rdp(points[:split+1],extent*.018)[:-1],rdp(points[split:],extent*.018)));simplified[-1]=simplified[0]
                unique=max(0,len(simplified)-1);xmin,ymin=points.min(axis=0);xmax,ymax=points.max(axis=0);rx=max((xmax-xmin)/2,.001);ry=max((ymax-ymin)/2,.001);center=((xmin+xmax)/2,(ymin+ymax)/2)
                if unique<=4:
                    component=self.make_component("Grating coupler",xmin,center[1]);component["orientation_deg"]=0.;description="triangle → grating coupler"
                elif max(rx,ry)/min(rx,ry)>1.18:
                    component=self.make_component("Elliptical ring",*center);component["params"].update({"radius_x":rx,"radius_y":ry});description="elliptical ring"
                else:
                    component=self.make_component("Ring",*center);component["params"]["radius"]=(rx+ry)/2;description="circular ring"
            else:
                chord_vec=points[-1]-points[0];chord=float(np.linalg.norm(chord_vec));ratio=chord/max(path_len,1e-9);sample=max(1,min(5,len(points)//4));v0=points[sample]-points[0];v1=points[-1]-points[-sample-1];a0=math.atan2(v0[1],v0[0]);a1=math.atan2(v1[1],v1[0]);delta=(math.degrees(a1-a0)+180)%360-180;orientation=math.degrees(a0);forward=np.array([math.cos(a0),math.sin(a0)]);normal=np.array([-forward[1],forward[0]]);longitudinal=float(np.dot(chord_vec,forward));offset=float(np.dot(chord_vec,normal))
                if ratio>.975:
                    component=self.make_component("Straight",*points[0]);component["params"]["length"]=chord;component["orientation_deg"]=math.degrees(math.atan2(chord_vec[1],chord_vec[0]));description="straight"
                elif abs(delta)<28 and abs(offset)>.04*max(abs(longitudinal),1e-9):
                    component=self.make_component("S-bend",*points[0]);component["params"].update({"length":max(abs(longitudinal),chord*.5),"offset":offset});component["orientation_deg"]=orientation;description="S-bend"
                else:
                    if abs(delta)<12:
                        cross=np.cross(chord_vec,points[len(points)//2]-points[0]);delta=90 if cross>0 else -90
                    component=self.make_component("Euler bend",*points[0]);component["params"].update({"radius":max(1.,min(1e5,max(float(np.ptp(points[:,0])),float(np.ptp(points[:,1]))))),"bend_angle_deg":max(-180.,min(180.,delta))});component["orientation_deg"]=orientation;description="Euler bend"
            self.components.append(component);created.append(int(component["uid"]));descriptions.append(description)
        self.commit_interaction_snapshot(snapshot);self.rebuild_scene(select_uids=created);self.statusBar().showMessage("Smart Sketch created clean standard components: "+", ".join(descriptions)+". Click any component to correct its parameters.",10000)

    def update_cursor_status(self, x: float, y: float) -> None:
        if hasattr(self, "cursor_status_label"):
            self.cursor_status_label.setText(f"X {x:,.3f} µm   Y {y:,.3f} µm")

    def update_zoom_status(self, percent: float) -> None:
        if hasattr(self, "zoom_status_label"):
            self.zoom_status_label.setText(f"Zoom {percent:,.1f}%")

    def set_layer_visible(self, layer: int, visible: bool) -> None:
        self.layer_visibility[int(layer)] = bool(visible)
        for component in self.components:
            item = self.items_by_uid.get(int(component["uid"]))
            if item is None:
                continue
            if isinstance(item, EbeamContainerItem):
                if int(layer) == EBEAM_LAYER:
                    item.setVisible(bool(visible))
                continue
            if isinstance(item, ComponentGraphicsItem):
                child_layers = []
                for child in item.childItems():
                    child_layer = child.data(0)
                    if child_layer is not None:
                        child_layers.append(int(child_layer))
                        child.setVisible(self.layer_visibility.get(int(child_layer), True))
                if child_layers:
                    item.setVisible(any(self.layer_visibility.get(child_layer, True) for child_layer in child_layers))
        self.view.viewport().update()

    def show_all_layers(self) -> None:
        for layer, checkbox in self.layer_checkboxes.items():
            checkbox.setChecked(True)
        self.statusBar().showMessage("All mapped layers are visible.")

    def resize_component_from_handle(
        self,
        uid: int,
        initial_parameters: dict[str, Any],
        scale_x: float,
        scale_y: float,
        snapshot_before: str,
    ) -> None:
        component = self.component_by_uid(uid)
        if component is None:
            return
        component["params"] = resize_component_parameters(
            str(component.get("kind", "")),
            initial_parameters,
            scale_x,
            scale_y,
        )
        _canonicalize_component_layers(component)
        self.commit_interaction_snapshot(snapshot_before)
        self.rebuild_scene(select_uids=[uid])
        self.statusBar().showMessage(
            f"Resized {component.get('kind')} from its corner handle."
        )

    def toggle_snap_ports(self, enabled: bool) -> None:
        self.snap_enabled = bool(enabled)
        self.statusBar().showMessage(
            "Port snapping enabled." if self.snap_enabled else "Port snapping disabled."
        )

    def toggle_auto_connect_input(self, enabled: bool) -> None:
        self.auto_connect_input_enabled = bool(enabled)
        self.statusBar().showMessage(
            "Input-port auto-connect enabled."
            if self.auto_connect_input_enabled
            else "Input-port auto-connect disabled."
        )

    def toggle_show_ports(self, enabled: bool) -> None:
        self.show_ports_enabled = bool(enabled)
        for item in self.items_by_uid.values():
            if isinstance(item, ComponentGraphicsItem):
                for port_item in item.port_items:
                    port_item.setVisible(self.show_ports_enabled)
        self.statusBar().showMessage(
            "Connection and center points visible."
            if self.show_ports_enabled
            else "Connection and center points hidden."
        )

    def preferred_input_port_name(self, component: dict[str, Any]) -> str | None:
        aliases = PORT_ALIASES.get(str(component.get("kind", "")), {})
        alias = aliases.get("input")
        ports = component_local_ports(component)
        if alias in ports:
            return str(alias)
        for candidate in (
            "left",
            "start",
            "left_external",
            "signal_left",
            "signal_start",
            "waveguide_point",
            "tip",
            "left_gc_point",
            "center",
        ):
            if candidate in ports:
                return candidate
        return next(iter(ports), None)

    @staticmethod
    def compatible_port_domains(domain_a: str, domain_b: str) -> bool:
        a = str(domain_a or "alignment")
        b = str(domain_b or "alignment")
        return a == b or a == "alignment" or b == "alignment"

    def port_screen_distance(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        screen_a = self.view.mapFromScene(world_to_scene_point(point_a))
        screen_b = self.view.mapFromScene(world_to_scene_point(point_b))
        return math.hypot(screen_a.x() - screen_b.x(), screen_a.y() - screen_b.y())

    def nearest_port_pair(
        self,
        component: dict[str, Any],
        input_only: bool,
    ) -> dict[str, Any] | None:
        own_ports = component_global_ports(component)
        if input_only:
            preferred = self.preferred_input_port_name(component)
            own_names = [preferred] if preferred in own_ports else []
        else:
            own_names = list(own_ports)
        if not own_names:
            return None

        best: dict[str, Any] | None = None
        for target_component in self.components:
            if int(target_component["uid"]) == int(component["uid"]):
                continue
            if target_component.get("kind") == "E-beam multipass":
                continue
            for target_name, target_port in component_global_ports(target_component).items():
                for own_name in own_names:
                    own_port = own_ports[own_name]
                    if not self.compatible_port_domains(
                        str(own_port.get("domain")),
                        str(target_port.get("domain")),
                    ):
                        continue
                    distance_pixels = self.port_screen_distance(
                        own_port["center"],
                        target_port["center"],
                    )
                    if best is None or distance_pixels < best["distance_pixels"]:
                        best = {
                            "distance_pixels": distance_pixels,
                            "own_name": own_name,
                            "own_port": own_port,
                            "target_uid": int(target_component["uid"]),
                            "target_name": target_name,
                            "target_port": target_port,
                        }
        return best

    def apply_port_pair(
        self,
        component: dict[str, Any],
        pair: dict[str, Any],
    ) -> None:
        component["attachment"] = {
            "target_uid": int(pair["target_uid"]),
            "target_port": str(pair["target_name"]),
            "own_port": str(pair["own_name"]),
        }
        solve_attachment(component, pair["target_port"])
        item = self.items_by_uid.get(int(component["uid"]))
        if isinstance(item, ComponentGraphicsItem):
            item.sync_transform()

    def snap_component_after_move(self, uid: int, force: bool = False) -> bool:
        if not force and not self.snap_enabled:
            return False
        component = self.component_by_uid(uid)
        if component is None or component.get("kind") == "E-beam multipass":
            return False

        pair = self.nearest_port_pair(component, input_only=False)
        if self.auto_connect_input_enabled:
            input_pair = self.nearest_port_pair(component, input_only=True)
            # Prefer the input only when it is effectively tied with the truly
            # closest input/output/gap/center point.  A visibly closer output or
            # center must always win.
            if input_pair is not None and (
                pair is None
                or float(input_pair["distance_pixels"]) <= float(pair["distance_pixels"]) + 1.5
            ):
                pair = input_pair
        if pair is None:
            return False
        if not force and float(pair["distance_pixels"]) > float(self.snap_distance_pixels):
            return False

        self.apply_port_pair(component, pair)
        self.statusBar().showMessage(
            f"Connected {component.get('kind')}:{pair['own_name']} to "
            f"UID {pair['target_uid']}:{pair['target_name']}."
        )
        return True

    def connect_selected_nearest(self) -> None:
        selected = self.selected_components()
        if not selected:
            QMessageBox.information(self, "Connect nearest", "Select a component first.")
            return
        component = selected[-1]
        snapshot = self.snapshot()
        if self.snap_component_after_move(int(component["uid"]), force=True):
            self.commit_interaction_snapshot(snapshot)
            self.rebuild_scene(select_uids=[int(component["uid"])])
        else:
            QMessageBox.information(
                self,
                "Connect nearest",
                "No compatible connection point was found.",
            )

    # ------------------------------------------------------------------
    # Arrays / modules
    # ------------------------------------------------------------------
    def create_array(self) -> None:
        selected = self.selected_components()
        if not selected:
            QMessageBox.information(self, "Array", "Select one or more components first.")
            return
        selected_uids = {int(component["uid"]) for component in selected}
        linked_ebeam = [
            component
            for component in self.components
            if component.get("kind") == "E-beam multipass"
            and selected_uids.intersection({int(uid) for uid in component.get("coverage_source_uids", [])})
            and int(component["uid"]) not in selected_uids
        ]
        originals = selected + linked_ebeam
        param_names = sorted(
            {
                key
                for component in selected
                for key in component.get("params", {})
                if key not in {"manual_field_offsets", "manual_field_order", "removed_field_keys", "auto_pruned_field_keys", "explicit_fields"}
                and isinstance(component["params"][key], (int, float))
            }
        )
        dialog = ArrayDialog(param_names, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        nx, ny = dialog.nx.value(), dialog.ny.value()
        try:
            x_spacings = parse_sequence(dialog.dx.text(), max(nx - 1, 1))
            y_spacings = parse_sequence(dialog.dy.text(), max(ny - 1, 1))
            x_positions = [0.0]
            for value in x_spacings[: nx - 1]:
                x_positions.append(x_positions[-1] + value)
            y_positions = [0.0]
            for value in y_spacings[: ny - 1]:
                y_positions.append(y_positions[-1] + value)
            x_values = parse_sequence(dialog.x_values.text(), nx) if dialog.x_param.currentText() else [None] * nx
            y_values = parse_sequence(dialog.y_values.text(), ny) if dialog.y_param.currentText() else [None] * ny
        except Exception as exc:
            QMessageBox.critical(self, "Array values", str(exc))
            return

        snapshot = self.snapshot()
        array_id = f"A{self.next_array_id}"
        self.next_array_id += 1
        for component in originals:
            component["array_group_id"] = array_id
            component["array_index"] = [0, 0]
        # deepcopy is materially faster than a JSON serialize/parse roundtrip
        # for large arrays while preserving the independent nested parameters.
        base_order = [copy.deepcopy(component) for component in originals]
        created_uids = [int(component["uid"]) for component in originals]

        for row in range(ny):
            for col in range(nx):
                if row == 0 and col == 0:
                    if dialog.x_param.currentText():
                        for component in selected:
                            if dialog.x_param.currentText() in component.get("params", {}):
                                component["params"][dialog.x_param.currentText()] = x_values[col]
                    if dialog.y_param.currentText():
                        for component in selected:
                            if dialog.y_param.currentText() in component.get("params", {}):
                                component["params"][dialog.y_param.currentText()] = y_values[row]
                    continue
                uid_map: dict[int, int] = {}
                cell_components: list[dict[str, Any]] = []
                for base in base_order:
                    duplicate = copy.deepcopy(base)
                    old_uid = int(base["uid"])
                    duplicate["uid"] = self.next_uid
                    self.next_uid += 1
                    uid_map[old_uid] = int(duplicate["uid"])
                    duplicate["x"] = float(base["x"]) + x_positions[col]
                    duplicate["y"] = float(base["y"]) + y_positions[row]
                    duplicate["array_group_id"] = array_id
                    duplicate["array_index"] = [col, row]
                    duplicate.pop("_last_group_position", None)
                    if dialog.x_param.currentText() and dialog.x_param.currentText() in duplicate.get("params", {}):
                        duplicate["params"][dialog.x_param.currentText()] = x_values[col]
                    if dialog.y_param.currentText() and dialog.y_param.currentText() in duplicate.get("params", {}):
                        duplicate["params"][dialog.y_param.currentText()] = y_values[row]
                    cell_components.append(duplicate)
                for duplicate in cell_components:
                    if duplicate.get("coverage_source_uids"):
                        duplicate["coverage_source_uids"] = [
                            uid_map.get(int(uid), int(uid)) for uid in duplicate["coverage_source_uids"]
                        ]
                    attachment = duplicate.get("attachment")
                    if attachment and int(attachment.get("target_uid", -1)) in uid_map:
                        attachment["target_uid"] = uid_map[int(attachment["target_uid"])]
                self.components.extend(cell_components)
                created_uids.extend(int(component["uid"]) for component in cell_components)

                if dialog.auto_label.isChecked():
                    cell_non_ebeam = [component for component in cell_components if component.get("kind") != "E-beam multipass"]
                    if cell_non_ebeam:
                        bounds = self.components_world_bounds(cell_non_ebeam)
                        fragments = []
                        if dialog.x_param.currentText():
                            fragments.append(f"{dialog.x_param.currentText()}={x_values[col]:g}")
                        if dialog.y_param.currentText():
                            fragments.append(f"{dialog.y_param.currentText()}={y_values[row]:g}")
                        prefix = dialog.label_prefix.text().strip()
                        text_value = f"{prefix}-" if prefix else ""
                        text_value += ", ".join(fragments) if fragments else f"{col},{row}"
                        label = self.make_component(
                            "Text / Number",
                            bounds[0] + dialog.label_offset_x.value(),
                            bounds[3] + dialog.label_offset_y.value(),
                        )
                        label["params"]["text"] = text_value
                        label["params"]["layer"] = MARKER_LAYER
                        label["params"]["datatype"] = 0
                        label["array_group_id"] = array_id
                        label["array_index"] = [col, row]
                        self.components.append(label)
                        created_uids.append(int(label["uid"]))

        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=created_uids)
        self.statusBar().showMessage(
            f"Created {nx} × {ny} array. E-beam fields replicate with the same array; numbering remains continuous."
        )

    def position_entire_array(self) -> None:
        selected=self.selected_components()
        if not selected:QMessageBox.information(self,"Position Entire Array","Select any member of the array or its E-beam coverage first.");return
        chosen=selected[-1];array_id=chosen.get("array_group_id");is_test_block=str(chosen.get("kind","")).endswith("test block")
        if not array_id and not is_test_block:QMessageBox.information(self,"Position Entire Array","Select a generated-array member, linked E-beam cover, or premade test block.");return
        members=[component for component in self.components if component.get("array_group_id")==array_id] if array_id else [chosen]
        bounds=self.components_world_bounds(members);cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2
        x,ok=QInputDialog.getDouble(self,"Position Entire Array","Target array center X (µm):",cx,-1e9,1e9,6)
        if not ok:return
        y,ok=QInputDialog.getDouble(self,"Position Entire Array","Target array center Y (µm):",cy,-1e9,1e9,6)
        if not ok:return
        snapshot=self.snapshot();dx=x-cx;dy=y-cy
        for component in members:component["x"]=float(component.get("x",0))+dx;component["y"]=float(component.get("y",0))+dy
        label=f"array {array_id}" if array_id else str(chosen.get("kind"));self.commit_interaction_snapshot(snapshot);self.rebuild_scene(select_uids=[int(component["uid"]) for component in members]);self.statusBar().showMessage(f"Positioned complete {label} at ({x:g}, {y:g}) µm, including linked E-beam coverage.",8000)

    def components_world_bounds(self, components: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        rects = []
        for component in components:
            item = self.items_by_uid.get(int(component["uid"]))
            if item:
                rect = item.sceneBoundingRect()
                rects.append((rect.left(), -rect.bottom(), rect.right(), -rect.top()))
        if not rects:
            return (0, 0, 0, 0)
        return (
            min(rect[0] for rect in rects),
            min(rect[1] for rect in rects),
            max(rect[2] for rect in rects),
            max(rect[3] for rect in rects),
        )

    def save_selection_as_module(self) -> None:
        selected = [component for component in self.selected_components() if component.get("kind") != "E-beam multipass"]
        if not selected:
            QMessageBox.information(self, "Save module", "Select one or more non-Ebeam components.")
            return
        name, accepted = QInputDialog.getText(self, "Save module", "Module name:")
        if not accepted or not name.strip():
            return
        bounds = self.components_world_bounds(selected)
        center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        module_components = []
        alias_counts: dict[str, int] = {}
        selected_uids = {int(component["uid"]) for component in selected}
        for component in selected:
            stored = safe_json_copy(component)
            stored["x"] = float(stored["x"]) - center[0]
            stored["y"] = float(stored["y"]) - center[1]
            alias_counts[stored["kind"]] = alias_counts.get(stored["kind"], 0) + 1
            stored["module_component_alias"] = f"{stored['kind'].replace(' ', '_')}_{alias_counts[stored['kind']]}"
            stored.pop("uid", None)
            stored.pop("group_id", None)
            stored.pop("array_group_id", None)
            stored.pop("module_instance_id", None)
            if stored.get("attachment") and int(stored["attachment"].get("target_uid", -1)) not in selected_uids:
                stored["attachment"] = None
            module_components.append(stored)
        self.custom_modules[name.strip()] = {
            "name": name.strip(),
            "components": module_components,
        }
        save_native_modules(self.custom_modules)
        self.populate_library()
        self.statusBar().showMessage(f"Saved module: {name.strip()}")

    def add_saved_module(self, name: str) -> None:
        module = self.custom_modules.get(name)
        if not module:
            return
        center = scene_to_world_point(self.view.mapToScene(self.view.viewport().rect().center()))
        snapshot = self.snapshot()
        instance_id = f"M{self.next_module_instance_id}"
        self.next_module_instance_id += 1
        group_id = f"G{self.next_group_id}"
        self.next_group_id += 1
        uid_map: dict[int, int] = {}
        created: list[dict[str, Any]] = []
        for index, stored in enumerate(module["components"], start=1):
            component = safe_json_copy(stored)
            old_uid = int(stored.get("uid", -index))
            component["uid"] = self.next_uid
            self.next_uid += 1
            uid_map[old_uid] = int(component["uid"])
            component["x"] = float(component.get("x", 0.0)) + center[0]
            component["y"] = float(component.get("y", 0.0)) + center[1]
            component["group_id"] = group_id
            component["module_instance_id"] = instance_id
            component["module_name"] = name
            component.setdefault("module_component_alias", f"component_{index}")
            created.append(component)
        for component in created:
            attachment = component.get("attachment")
            if attachment and int(attachment.get("target_uid", -1)) in uid_map:
                attachment["target_uid"] = uid_map[int(attachment["target_uid"])]
            elif attachment:
                component["attachment"] = None
        self.components.extend(created)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"]) for component in created])
        self.statusBar().showMessage(f"Added module: {name}")

    def open_module_variables(self) -> None:
        selected = self.selected_components()
        if len(selected) != 1 or not selected[0].get("module_instance_id"):
            return
        instance_id = selected[0]["module_instance_id"]
        members = [component for component in self.components if component.get("module_instance_id") == instance_id]
        ModuleVariablesDialog(self, members).exec()

    # ------------------------------------------------------------------
    # E-beam fields
    # ------------------------------------------------------------------
    def ebeam_source_polygons(self, components: list[dict[str, Any]]) -> list[gdstk.Polygon]:
        """Flatten exact world-space geometry eligible for E-beam coverage (layers 1/2/3 only)."""
        eligible={PHOTONIC_LAYER,GC_LAYER,MARKER_LAYER};result=[]
        for source in components:
            if source.get("kind")=="E-beam multipass":continue
            library=gdstk.Library(unit=1e-6,precision=1e-9);cell=library.new_cell(f"COVER_SOURCE_{int(source.get('uid',0))}")
            _add_component_geometry_to_cell(safe_json_copy(source),cell)
            result.extend(polygon for polygon in cell.get_polygons(apply_repetitions=True,include_paths=True) if int(polygon.layer) in eligible)
        return result

    @staticmethod
    def polygon_collection_bounds(polygons: list[gdstk.Polygon]) -> tuple[float,float,float,float]:
        if not polygons:raise ValueError("No layer 1, 2, or 3 geometry was found in the selection.")
        points=np.vstack([np.asarray(polygon.points,float) for polygon in polygons]);return (float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max()))

    def create_ebeam_coverage(self) -> None:
        sources = [component for component in self.selected_components() if component.get("kind") != "E-beam multipass"]
        if not sources:
            QMessageBox.information(self, "E-beam coverage", "Select the geometry to cover first.")
            return
        source_polygons=self.ebeam_source_polygons(sources)
        if not source_polygons:
            QMessageBox.information(self,"E-beam coverage","The selection contains no WG, GC, or marker geometry on layers 1, 2, or 3. RF-only geometry is intentionally ignored.");return
        dialog = EbeamDialog(DEFAULT_COMPONENT_VALUES["E-beam multipass"], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        bounds = self.polygon_collection_bounds(source_polygons)
        center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        snapshot = self.snapshot()
        component = self.make_component("E-beam multipass", *center)
        params = component["params"]
        params.update(
            {
                "field_size": dialog.field_size.value(),
                "edge_clearance": dialog.clearance.value(),
                "overlap_x_enabled": dialog.enable_x.isChecked(),
                "overlap_y_enabled": dialog.enable_y.isChecked(),
                "overlap_x_percent": dialog.overlap_x.value(),
                "overlap_y_percent": dialog.overlap_y.value(),
                "target_width": bounds[2] - bounds[0],
                "target_height": bounds[3] - bounds[1],
                "start_corner": dialog.start_corner.currentText(),
                "primary_axis": dialog.primary_axis.currentText(),
                "serpentine": dialog.serpentine.isChecked(),
                "preserve_manual_grid_position": True,
                "field_layer": EBEAM_LAYER,
                "field_datatype": 0,
                "beamer_wg_dose": dialog.wg_dose.value(),
                "beamer_gc_dose": dialog.gc_dose.value(),
                "beamer_marker_dose": dialog.marker_dose.value(),
                "beamer_region_layer": EBEAM_LAYER,
                "manual_field_offsets": {},
                "manual_field_order": {},
                "removed_field_keys": [],
                "auto_pruned_field_keys": [],
            }
        )
        component["coverage_source_uids"] = [int(source["uid"]) for source in sources]
        self.components.append(component)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"])])
        self.prune_ebeam_component(component)
        self.rebuild_scene(select_uids=[int(component["uid"])])
        self.statusBar().showMessage("Created E-beam coverage and removed fields with no geometry overlap.")

    def source_shape_path(self, source_item: QGraphicsItem) -> QPainterPath:
        try:
            return source_item.mapToScene(source_item.shape())
        except Exception:
            path = QPainterPath()
            path.addRect(source_item.sceneBoundingRect())
            return path

    def prune_ebeam_component(self, component: dict[str, Any]) -> None:
        params = component["params"]
        params["auto_pruned_field_keys"] = []
        layout = multipass_field_layout(params)
        source_uids=[int(uid) for uid in component.get("coverage_source_uids",[])]
        if not source_uids:return
        source_components=[self.component_by_uid(uid) for uid in source_uids];source_components=[source for source in source_components if source is not None]
        source_polygons=self.ebeam_source_polygons(source_components);source_boxes=[]
        for polygon in source_polygons:
            box=polygon.bounding_box();source_boxes.append((polygon,float(box[0][0]),float(box[0][1]),float(box[1][0]),float(box[1][1])))
        pruned: list[str] = []
        for field in layout["fields"]:
            rect_data = field.get("rect")
            if isinstance(rect_data, (list, tuple)) and len(rect_data) == 4:
                x0, y0, x1, y1 = map(float, rect_data)
            else:
                cx, cy = map(float, field.get("center", (0.0, 0.0)))
                width = float(field.get("width", layout.get("field_size", params.get("field_size", 520.0))))
                height = float(field.get("height", layout.get("field_size", params.get("field_size", 520.0))))
                x0, y0, x1, y1 = (
                    cx - width / 2.0,
                    cy - height / 2.0,
                    cx + width / 2.0,
                    cy + height / 2.0,
                )
                field["width"] = width
                field["height"] = height
                field["rect"] = (x0, y0, x1, y1)
            corners=transform_points(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float),(float(component["x"]),float(component["y"])),float(component.get("orientation_deg",0)));field_polygon=gdstk.Polygon(corners);fb=field_polygon.bounding_box();fx0,fy0=float(fb[0][0]),float(fb[0][1]);fx1,fy1=float(fb[1][0]),float(fb[1][1]);candidates=[polygon for polygon,bx0,by0,bx1,by1 in source_boxes if min(fx1,bx1)>max(fx0,bx0)+1e-9 and min(fy1,by1)>max(fy0,by0)+1e-9]
            if not candidates or not gdstk.boolean([field_polygon],candidates,"and",precision=1e-6):pruned.append(str(field["field_key"]))
        params["auto_pruned_field_keys"] = pruned

    def deoverlap_ebeam_fields(self, component: dict[str, Any], preferred_key: str) -> int:
        """Move a manually dragged field by the minimum amount needed to abut its neighbors."""
        params=component["params"]
        if bool(params.get("overlap_x_enabled",False)) or bool(params.get("overlap_y_enabled",False)):
            return 0
        layout=multipass_field_layout(params);fields={str(field["field_key"]):field for field in layout["fields"]}
        moving=fields.get(str(preferred_key))
        if moving is None:return 0
        offsets=dict(params.get("manual_field_offsets",{}));moves=0
        for _ in range(max(1,len(fields)*2)):
            collision=None
            ax0,ay0,ax1,ay1=map(float,moving["rect"])
            for key,other in fields.items():
                if key==str(preferred_key):continue
                bx0,by0,bx1,by1=map(float,other["rect"]);ox=min(ax1,bx1)-max(ax0,bx0);oy=min(ay1,by1)-max(ay0,by0)
                if ox>1e-9 and oy>1e-9:collision=(other,ox,oy);break
            if collision is None:break
            other,ox,oy=collision;acx=(ax0+ax1)/2;acy=(ay0+ay1)/2;bcx=(float(other["rect"][0])+float(other["rect"][2]))/2;bcy=(float(other["rect"][1])+float(other["rect"][3]))/2
            dx=dy=0.0
            if ox<=oy:dx=(-ox if acx<bcx else ox)
            else:dy=(-oy if acy<bcy else oy)
            base=moving.get("base_center",moving["center"]);new_center=(float(moving["center"][0])+dx,float(moving["center"][1])+dy);offsets[str(preferred_key)]=[new_center[0]-float(base[0]),new_center[1]-float(base[1])];moving["center"]=new_center;moving["rect"]=(ax0+dx,ay0+dy,ax1+dx,ay1+dy);moves+=1
        params["manual_field_offsets"]=offsets
        return moves

    def finish_individual_field_move(self, uid: int, field_key: str, snapshot: str) -> None:
        component=self.component_by_uid(uid)
        if component is None:return
        moved=self.deoverlap_ebeam_fields(component,field_key);self.prune_ebeam_component(component);self.commit_interaction_snapshot(snapshot)
        QTimer.singleShot(0,lambda:self.rebuild_scene(select_uids=[uid]));self.statusBar().showMessage(f"Write field updated; {moved} overlap correction(s) applied and empty fields removed.",7000)

    def finish_ebeam_group_move(self, uid: int, snapshot: str) -> None:
        component=self.component_by_uid(uid)
        if component is None:return
        self.prune_ebeam_component(component);self.commit_interaction_snapshot(snapshot)
        QTimer.singleShot(0,lambda:self.rebuild_scene(select_uids=[uid]));self.statusBar().showMessage("Moved the complete write-field set; empty fields were removed.",7000)

    def update_selected_ebeam(self) -> None:
        components = [component for component in self.selected_components() if component.get("kind") == "E-beam multipass"]
        if not components:
            QMessageBox.information(self, "Update fields", "Select an E-beam field component.")
            return
        snapshot = self.snapshot()
        for component in components:
            source_components = [
                self.component_by_uid(int(uid))
                for uid in component.get("coverage_source_uids", [])
                if self.component_by_uid(int(uid)) is not None
            ]
            if source_components:
                source_polygons=self.ebeam_source_polygons(source_components)
                if not source_polygons:
                    component["params"]["auto_pruned_field_keys"]=[str(field["field_key"]) for field in multipass_field_layout(component["params"])["fields"]];continue
                bounds = self.polygon_collection_bounds(source_polygons)
                params = component["params"]
                params["target_width"] = bounds[2] - bounds[0]
                params["target_height"] = bounds[3] - bounds[1]
                if not bool(params.get("preserve_manual_grid_position", True)):
                    component["x"] = (bounds[0] + bounds[2]) / 2
                    component["y"] = (bounds[1] + bounds[3]) / 2
            self.prune_ebeam_component(component)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"]) for component in components])
        self.statusBar().showMessage("Updated fields, preserved manual positions, and removed fields with no geometry overlap.")

    def reset_selected_ebeam_fields(self) -> None:
        components = [component for component in self.selected_components() if component.get("kind") == "E-beam multipass"]
        if not components:
            return
        snapshot = self.snapshot()
        for component in components:
            component["params"]["manual_field_offsets"] = {}
            component["params"]["manual_field_order"] = {}
            component["params"]["removed_field_keys"] = []
            component["params"]["auto_pruned_field_keys"] = []
            self.prune_ebeam_component(component)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"]) for component in components])

    def remove_active_field(self) -> None:
        if not self.active_field:
            return
        component = self.component_by_uid(self.active_field[0])
        if not component or component.get("kind") != "E-beam multipass":
            return
        snapshot = self.snapshot()
        removed = set(map(str, component["params"].get("removed_field_keys", [])))
        removed.add(str(self.active_field[1]))
        component["params"]["removed_field_keys"] = sorted(removed)
        self.active_field = None
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"])])

    def field_order_state(self) -> tuple[dict[tuple[int, str], int], dict[int, tuple[int, int]]]:
        mapping: dict[tuple[int, str], int] = {}
        ranges: dict[int, tuple[int, int]] = {}
        order = 1
        for component in self.components:
            if component.get("kind") != "E-beam multipass":
                continue
            fields = multipass_field_layout(component["params"])["fields"]
            start = order
            for field in fields:
                mapping[(int(component["uid"]), str(field["field_key"]))] = order
                order += 1
            ranges[int(component["uid"])] = (start, order - 1)
        return mapping, ranges

    def global_field_order(self, uid: int, field_key: str) -> int:
        mapping, _ = self.field_order_state()
        return mapping.get((int(uid), str(field_key)), 0)

    def set_field_global_order(self, uid: int, field_key: str, desired_global: int) -> None:
        component = self.component_by_uid(uid)
        if component is None:
            return
        _, ranges = self.field_order_state()
        start, end = ranges.get(int(uid), (1, 0))
        if desired_global < start or desired_global > end:
            raise ValueError(f"This array cell accepts field numbers {start} through {end}.")
        desired_local = desired_global - start + 1
        orders = dict(component["params"].get("manual_field_order", {}))
        for key, value in list(orders.items()):
            if str(key) != str(field_key) and int(round(float(value))) == desired_local:
                orders.pop(key, None)
        orders[str(field_key)] = desired_local
        component["params"]["manual_field_order"] = orders

    def assign_active_field_order(self) -> None:
        if not self.active_field:
            QMessageBox.information(self, "Field order", "Click an individual field first.")
            return
        _, ranges = self.field_order_state()
        start, end = ranges.get(int(self.active_field[0]), (1, 0))
        desired, accepted = QInputDialog.getInt(
            self,
            "Assign field order",
            f"Global field number ({start}–{end}):",
            self.global_field_order(*self.active_field),
            start,
            end,
        )
        if not accepted:
            return
        snapshot = self.snapshot()
        self.set_field_global_order(self.active_field[0], self.active_field[1], desired)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(self.active_field[0])])

    def refresh_field_numbers(self) -> None:
        mapping, _ = self.field_order_state()
        for uid, item in self.items_by_uid.items():
            if not isinstance(item, EbeamContainerItem):
                continue
            for key, field_item in item.field_items.items():
                field_item.set_global_order(mapping.get((uid, key), 0))
        self.update_writefield_playback_visuals()

    def collect_field_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for component in self.components:
            kind=component.get("kind")
            if kind == "E-beam multipass":
                layout = multipass_field_layout(component["params"])
                for field in layout["fields"]:
                    x0, y0, x1, y1 = map(float, field["rect"])
                    points = transformed_local_points(
                        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                        component,
                    )
                    records.append(
                        {
                            "xmin": float(points[:, 0].min()),
                            "ymin": float(points[:, 1].min()),
                            "xmax": float(points[:, 0].max()),
                            "ymax": float(points[:, 1].max()),
                        }
                    )
            elif kind in {"Double-ring test block", "Grating test block", "Grating angle-taper test block", "MMI + Reference test block", "MMI split-combine test block", "Long MZI test block", "Vertical-GC MZI test block", "Vertical-GC MZI + CPW test block", "Vertical-GC MZI + segmented electrode test block"}:
                # Compound test blocks draw each field as four thin polygons
                # on layer 6.  Rebuild the component, retain only those bars,
                # and union every four consecutive bars into its field box.
                library=gdstk.Library(unit=1e-6,precision=1e-9);cell=library.new_cell(f"FTXT_FIELDS_{int(component.get('uid',0))}");_add_component_geometry_to_cell(safe_json_copy(component),cell)
                bars=[]
                for polygon in cell.get_polygons(apply_repetitions=True,include_paths=True):
                    if int(polygon.layer)!=EBEAM_LAYER:continue
                    box=polygon.bounding_box();width=float(box[1][0]-box[0][0]);height=float(box[1][1]-box[0][1])
                    # Boundary bars are long and thin; layer-6 field-number
                    # text is deliberately excluded from FTXT rectangles.
                    if max(width,height)>=100 and max(width,height)>10*max(min(width,height),1e-12):bars.append(polygon)
                if len(bars)%4:
                    raise ValueError(f"{kind} produced {len(bars)} E-beam boundary polygons; expected groups of four.")
                for offset in range(0,len(bars),4):
                    points=np.vstack([np.asarray(polygon.points,float) for polygon in bars[offset:offset+4]])
                    records.append({"xmin":float(points[:,0].min()),"ymin":float(points[:,1].min()),"xmax":float(points[:,0].max()),"ymax":float(points[:,1].max())})
        for index, record in enumerate(records, start=1):
            record["order"] = index
            record["region_name"] = f"R{index}"
        return records

    def shift_active_field_order(self, delta: int) -> None:
        if not self.active_field:
            QMessageBox.information(self, "Field order", "Click an individual write field first.")
            return
        uid, field_key = self.active_field
        current = self.global_field_order(uid, field_key)
        _, ranges = self.field_order_state()
        start, end = ranges.get(int(uid), (current, current))
        desired = max(start, min(end, current + int(delta)))
        if desired == current:
            return
        snapshot = self.snapshot()
        self.set_field_global_order(uid, field_key, desired)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[uid])
        self.active_field = (uid, field_key)
        self.statusBar().showMessage(f"Moved write field to global order {desired}.")

    def ordered_writefield_sequence(self) -> list[tuple[int, str]]:
        mapping, _ = self.field_order_state()
        return [
            key
            for key, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]

    def update_writefield_playback_visuals(self) -> None:
        index_by_key = {
            key: index for index, key in enumerate(self.field_play_sequence)
        }
        for uid, item in self.items_by_uid.items():
            if not isinstance(item, EbeamContainerItem):
                continue
            for field_key, field_item in item.field_items.items():
                sequence_index = index_by_key.get((uid, field_key))
                if sequence_index is None or self.field_play_index < 0:
                    state = "future"
                elif sequence_index < self.field_play_index:
                    state = "complete"
                elif sequence_index == self.field_play_index:
                    state = "active"
                else:
                    state = "future"
                field_item.set_playback_state(state)

    def play_writefields(self) -> None:
        self.field_play_sequence = self.ordered_writefield_sequence()
        if not self.field_play_sequence:
            QMessageBox.information(self, "Write-field playback", "No active write fields exist.")
            return
        self.field_play_index = -1
        self.advance_writefield_playback()
        self.field_play_timer.start(350)
        self.statusBar().showMessage(
            f"Playing {len(self.field_play_sequence)} write fields in final order."
        )

    def step_writefields(self) -> None:
        self.field_play_timer.stop()
        if not self.field_play_sequence:
            self.field_play_sequence = self.ordered_writefield_sequence()
            self.field_play_index = -1
        if not self.field_play_sequence:
            return
        self.advance_writefield_playback()

    def advance_writefield_playback(self) -> None:
        if not self.field_play_sequence:
            self.field_play_timer.stop()
            return
        self.field_play_index += 1
        if self.field_play_index >= len(self.field_play_sequence):
            self.field_play_timer.stop()
            self.field_play_index = len(self.field_play_sequence)
            self.update_writefield_playback_visuals()
            self.statusBar().showMessage("Write-field playback complete.")
            return
        self.update_writefield_playback_visuals()
        uid, field_key = self.field_play_sequence[self.field_play_index]
        order = self.global_field_order(uid, field_key)
        item = self.items_by_uid.get(uid)
        if isinstance(item, EbeamContainerItem):
            field_item = item.field_items.get(field_key)
            if field_item is not None:
                self.view.centerOn(field_item)
        self.statusBar().showMessage(
            f"Write-field playback: {order} of {len(self.field_play_sequence)}."
        )

    def stop_writefields(self) -> None:
        self.field_play_timer.stop()
        self.field_play_sequence = []
        self.field_play_index = -1
        self.update_writefield_playback_visuals()
        self.statusBar().showMessage("Write-field playback stopped.")

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_gds(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export flattened GDS",
            str(Path.home() / "photonic_layout.gds"),
            "GDSII (*.gds)",
        )
        if not path:
            return
        self.start_worker_export("--worker-export-gds", path, "Exporting flattened GDS…")

    def export_python(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export runnable Python",
            str(Path.home() / "photonic_layout_export.py"),
            "Python (*.py)",
        )
        if not path:
            return
        self.start_worker_export("--worker-export-python", path, "Exporting runnable Python…")

    def start_worker_export(self, mode: str, output_path: str, label: str) -> None:
        if self.export_process is not None:
            QMessageBox.information(self, "Export", "An export is already running.")
            return
        temp = Path(tempfile.mkstemp(prefix="photonic_native_", suffix=".json")[1])
        temp.write_text(json.dumps(self.project_payload()))
        self.export_temp_file = temp
        self.progress_dialog = QProgressDialog(label, "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        process = QProcess(self)
        self.export_process = process
        process.setProgram(sys.executable)
        launcher = launcher_path()
        process.setArguments([launcher, mode, str(temp), output_path])
        process.readyReadStandardOutput.connect(
            lambda: self.statusBar().showMessage(bytes(process.readAllStandardOutput()).decode(errors="replace"))
        )
        process.finished.connect(partial(self.worker_export_finished, output_path))
        process.errorOccurred.connect(lambda error: self.worker_export_failed(process.errorString()))
        self.progress_dialog.canceled.connect(process.kill)
        process.start()

    def worker_export_finished(self, output_path: str, exit_code: int, exit_status) -> None:
        process = self.export_process
        stderr = bytes(process.readAllStandardError()).decode(errors="replace") if process else ""
        if self.progress_dialog:
            self.progress_dialog.close()
        if self.export_temp_file:
            self.export_temp_file.unlink(missing_ok=True)
        self.export_process = None
        self.export_temp_file = None
        if exit_code == 0:
            self.statusBar().showMessage(f"Exported: {output_path}")
        else:
            QMessageBox.critical(self, "Export failed", stderr or f"Worker exited with code {exit_code}.")

    def worker_export_failed(self, message: str) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Export failed", message)

    def export_field_txt(self) -> None:
        records = self.collect_field_records()
        if not records:
            QMessageBox.information(self, "Field TXT", "No E-beam fields exist.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export manual field control",
            str(Path.home() / "manual_field_control.txt"),
            "Text (*.txt)",
        )
        if not path:
            return
        lines = ["# MANUAL FIELD CONTROL FILE"]
        for record in records:
            lines.append(
                "True\t"
                f"{record['xmin']:.12g}\t{record['ymin']:.12g}\t"
                f"{record['xmax']:.12g}\t{record['ymax']:.12g}\t"
                f"{record['region_name']}\t"
            )
        Path(path).write_text("\n".join(lines) + "\n")
        self.statusBar().showMessage(f"Exported {len(records)} ordered regions: {path}")

    def import_field_txt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import manual field control",
            str(Path.home()),
            "Text (*.txt);;All files (*)",
        )
        if not path:
            return
        records = []
        try:
            for raw_line in Path(path).read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 6 or parts[0].lower() not in {"true", "1", "yes"}:
                    continue
                xmin, ymin, xmax, ymax = map(float, parts[1:5])
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                if ymax < ymin:
                    ymin, ymax = ymax, ymin
                records.append((xmin, ymin, xmax, ymax, parts[5]))
            if not records:
                raise ValueError("No active field records were found.")
            xmin = min(record[0] for record in records)
            ymin = min(record[1] for record in records)
            xmax = max(record[2] for record in records)
            ymax = max(record[3] for record in records)
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            snapshot = self.snapshot()
            component = self.make_component("E-beam multipass", *center)
            component["params"].update(
                {
                    "target_width": xmax - xmin,
                    "target_height": ymax - ymin,
                    "edge_clearance": 0.0,
                    "manual_field_offsets": {},
                    "manual_field_order": {},
                    "removed_field_keys": [],
                    "auto_pruned_field_keys": [],
                    "explicit_fields": [
                        {
                            "field_key": f"import_{index}",
                            "region_name": region,
                            "bounds": [
                                x0 - center[0],
                                y0 - center[1],
                                x1 - center[0],
                                y1 - center[1],
                            ],
                        }
                        for index, (x0, y0, x1, y1, region) in enumerate(records, start=1)
                    ],
                }
            )
            component["coverage_source_uids"] = []
            self.components.append(component)
            self.commit_interaction_snapshot(snapshot)
            self.rebuild_scene(select_uids=[int(component["uid"])])
            self.fit_layout()
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    def export_beamer_ftext(self) -> None:
        records = self.collect_field_records()
        if not records:
            QMessageBox.information(self, "BEAMER FTEXT", "No E-beam fields exist.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export BEAMER FTEXT",
            str(Path.home() / "photonic_layout.ftxt"),
            "BEAMER FTEXT (*.ftxt);;Text (*.txt)",
        )
        if not path:
            return
        ebeam_components = [component for component in self.components if component.get("kind") == "E-beam multipass"]
        params = ebeam_components[0]["params"] if ebeam_components else {}
        wg_dose = float(params.get("beamer_wg_dose", 1.8))
        gc_dose = float(params.get("beamer_gc_dose", 1.8))
        marker_dose = float(params.get("beamer_marker_dose", 1.8))
        region_lines = []
        for record in records:
            region_lines.extend(
                [
                    "REGION_ACTIVE = False",
                    f"REGION_NAME = {record['region_name']}",
                    f"START_X = {record['xmin']:.12g}",
                    f"START_Y = {record['ymin']:.12g}",
                    f"STOP_X = {record['xmax']:.12g}",
                    f"STOP_Y = {record['ymax']:.12g}",
                ]
            )
        flow = self.beamer_flow_template(
            "\n".join(region_lines),
            wg_dose,
            gc_dose,
            marker_dose,
        )
        Path(path).write_text(flow)
        self.statusBar().showMessage(f"Exported BEAMER FTEXT with {len(records)} regions: {path}")

    def beamer_flow_template(
        self,
        region_lines: str,
        wg_dose: float,
        gc_dose: float,
        marker_dose: float,
    ) -> str:
        return f"""FLOW photonic_layout
PROGRAM_VERSION BEAMER_Revision_Number_7.4.0_(a48748191b),_Jul_21_2025,_08:56:31
OPERATING_SYSTEM Microsoft_Windows_8
ZOOM_LEVEL 1.000000
LIB_COMMENT Layer_1_WG%3BLayer_2_GC%3BLayer_3_Marker%3BLayer_4_RF%3BLayer_5_Probe%3BLayer_6_Ebeam
SHOW_LIB_COMMENT false
 ()
NODE Import ()
ID       = 1
VERSION    = 2
SHOWCOMMENT    = false
COMMENTSIZE = 206, 40
LABEL    = photonic_layout
POSITION = 420, 47
COLLECTFORLOOP = false
DISABLED = false
OUT_PORT[0] = 4, Mapping, 0

FILE_NAME = .%5Cphotonic_layout.gds
FILE_TYPE = 1
LAYERSET = *
SINGLE_PATH_IMPORT = false
ZERO_PATH_WIDTH = 0.000000
BOXES_IMPORT = true
KEEPELEMENTORDER = false
FLATTENLAYOUT = false
OVERLAP_AND_GAP_DETECTION = false
OVERLAP_AND_GAP_DETECTION_MIN_SIZE = 0.000000
LoadTextElements = false
ConvertTextElementsToPolys = false
CONVERTED_TEXT_SIZE = 1.000000
IMPORT_SHAPE_CIRCLE = false
IMPORT_SHAPE_RING = false
IMPORT_SHAPE_ARCBOW = false
IMPORT_SHAPE_SECTOR = false
IMPORT_SHAPE_ELLIPSE = false
IMPORT_SHAPE_ROTATED_RECTANGLE = false
IMPORT_SHAPE_PARALLELOGRAM = false
MAXIMUM_CIRCLE_ERROR = 0.001000
CURVE_DETECTION = false
QAP_START
FileName
QAP_END

ENDNODE

NODE FDA ()
ID       = 2
VERSION    = 2
SHOWCOMMENT    = false
COMMENTSIZE = 206, 40
LABEL    = FDA
POSITION = 415, 260
COLLECTFORLOOP = false
DISABLED = false
IN_PORT[0] = 4, Mapping, 0
OUT_PORT[0] = 3, EBPG%20GPF, 0
QAP_START
QAP_END

UseUserDefinedFractureGrid = false
UserDefinedFractureGrid = 0.010000
UserDefinedMinFigSize = 0.100000

ASSIGN_METHOD = SET_VALUE
ASSIGN_MODE = BYLAYER
ASSIGNMENT = WG%20%3A%20{wg_dose:.6f}
ASSIGNMENT = GC%20%3A%20{gc_dose:.6f}
ASSIGNMENT = MARKER%20%3A%20{marker_dose:.6f}
ENDNODE

NODE Export ()
ID       = 3
VERSION    = 2
SHOWCOMMENT    = false
COMMENTSIZE = 206, 40
LABEL    = EBPG%20GPF
POSITION = 390, 366
COLLECTFORLOOP = false
DISABLED = false
IN_PORT[0] = 2, FDA, 0
QAP_START
FileName
QAP_END


FILE_NAME = .%5Cphotonic_layout.gpf
FILE_TYPE = 7
EXTENT_AUTOMATIC

FORMAT_TYPE = 5200%20%2F%205000%2B%2020bit%20HS%20UPG%20100kV%201.46
FORMAT_VERSION = 1.46-UPG
TENSION = 100
MAINFIELD_DAC_BITS = 20
MAINFIELD_RESOLUTION_MIN = 0.160000000
MAINFIELD_RESOLUTION_MAX = 1.000000000
MINIMUM_MAINFIELD_SIZE = 10.00000
MAXIMUM_MAINFIELD_SIZE = 1048.57600
NUMBER_SUBFIELD_BITS = 14
SUBFIELD_RESOLUTION_MIN = 0.080000000
SUBFIELD_RESOLUTION_MAX = 0.500000000
MAXIMUM_SUBFIELD_MSF = 8192
MINIMUM_SUBFIELD_SIZE = 0.00125
MAXIMUM_SUBFIELD_SIZE = 4.52500
SYSTEM_TYPE = HS
RESOLUTION_CONNECTION = Advanced
RESOLUTION = 0.001
BEAM_STEP_SIZE = 0.004
MAIN_FIELD_RESOLUTIONX = 0.000500000000
MAIN_FIELD_RESOLUTIONY = 0.000500000000
MAIN_FIELD_MSF = 2
SUB_FIELD_RESOLUTIONX = 0.000200000000
SUB_FIELD_RESOLUTIONY = 0.000200000000
SUB_FIELD_MSF = 20
MAIN_FIELD_SIZE_X = 520.000000
MAIN_FIELD_SIZE_Y = 520.000000
MAIN_FIELD_PLACEMENT = RegionLayer
FIXEDFIELDTRAVERSAL = MeanderX
FIXEDFIELDALIGNMENT = Lower%20Left
SUB_FIELD_SIZE_X = 3.260000
SUB_FIELD_SIZE_Y = 3.260000
COMPACTION_REGION_SIZE = 4.525000
SUBFIELD_OVERLAP_X = 0.100000
SUBFIELD_OVERLAP_Y = 0.100000
REGION_TRAVERSAL_MODE = MeanderX
FEATURE_ORDERING_TYPE = NoCompaction
FEATURE_ORDERING_START_POSITION_TYPE = Automatic
SORTED_ORDER_LAYER = *
DOSE_ORDERING_TYPE = AscendingDose
SHOT_FILLING_MODE = HighResolutionMode
BEAM_STEP_SIZE_FRACTURING = false
USE_FIELD_SCALING = false
FIELD_SCALING_LAYER = *
CURVE_TOLERANCE = 1.000000
USE_CIRCLE_SHAPE = true
USE_ELLIPSE_SHAPE = true
USE_CUBE_SHAPE = true
USE_POLYGON_SHAPE = false
NUMBER_OF_SLEEVES = 1
SLEEVING_BULK_OVERLAP = 0.000000
SLEEVING_BSS = 0.010000
SYMMETRIC_FRACTURING = false
SHAPE_PROCESSING = Jump
Y_TRAPEZIAS = true
DIAGONAL_LINE_COMPACTION = true
TRAPEZOID_DENSITY_CORRECTION = false
NORMALIZE_DOSE_RANGE = false
AREA_SELECTION = SelectedFieldsOnly
FIELD_OVERLAP_BEHAVIOUR = KeepFieldSize
FRACTURE_MODE = LRFT
{region_lines}
REGION_LAYER = Ebeam

FIELD_OVERLAP_X = 2
FIELD_OVERLAP_Y = 2
SORT_METHOD = Fracture
OVERLAP_METHOD = Share%20between%20Fields
INTERLEAVING_SIZE = 0.000000
INTERLOCK_LAYER = *
MULTIPASS_MODE = Two%20Passes
MULTIPASS_FIELD_ARRANGEMENT = Shortest%20Path
MAINFIELD_OFFSET_X = 0.000000
MAINFIELD_OFFSET_Y = 0.000000
MAINFIELD_OFFSET_ABSOLUTE = false
SUBFIELD_OFFSET_X = 0.000000
SUBFIELD_OFFSET_Y = 0.000000
SUBFIELD_OFFSET_ABSOLUTE = false
MULTIPASS_LAYER = *
ENDNODE

NODE Mapping ()
ID       = 4
VERSION    = 2
SHOWCOMMENT    = false
COMMENTSIZE = 206, 40
LABEL    = Mapping
POSITION = 417, 125
COLLECTFORLOOP = false
DISABLED = false
IN_PORT[0] = 1, photonic_layout, 0
OUT_PORT[0] = 2, FDA, 0
QAP_START
QAP_END


LAYER_MAPPING = 1(0)%20%3A%20WG%20%3A%20Wg
LAYER_MAPPING = 2(0)%20%3A%20GC%20%3A%20Gc
LAYER_MAPPING = 3(0)%20%3A%20MARKER%20%3A%20marker
LAYER_MAPPING = 4(0)%20%3A%20RF%20%3A%20RF
LAYER_MAPPING = 5(0)%20%3A%20probe%20%3A%20probe
LAYER_MAPPING = 6(0)%20%3A%20Ebeam%20%3A%20Ebeam
ENDNODE

ENDFLOW
"""

    def append_llm_chat(self, role: str, message: str) -> None:
        prefix = {
            "user": "You",
            "assistant": "Assistant",
            "error": "Error",
        }.get(role, role.title())
        current = self.llm_chat_log.toPlainText().rstrip()
        entry = f"{prefix}: {message}"
        self.llm_chat_log.setPlainText(f"{current}\n\n{entry}" if current else entry)
        scroll_bar = self.llm_chat_log.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def compact_layout_summary_native(self) -> list[dict[str, Any]]:
        selected = {int(component["uid"]) for component in self.selected_components()}
        summary = []
        for component in self.components:
            try:
                ports = component_global_ports(component)
                compact_ports = {
                    name: {
                        "center": [float(value) for value in port["center"]],
                        "domain": port.get("domain"),
                    }
                    for name, port in ports.items()
                }
            except Exception:
                compact_ports = {}
            summary.append(
                {
                    "uid": int(component["uid"]),
                    "kind": component.get("kind"),
                    "x": float(component.get("x", 0.0)),
                    "y": float(component.get("y", 0.0)),
                    "orientation_deg": float(component.get("orientation_deg", 0.0)),
                    "mirrored": bool(component.get("mirrored", False)),
                    "selected": int(component["uid"]) in selected,
                    "params": component.get("params", {}),
                    "connection_points": compact_ports,
                }
            )
        return summary

    def canonical_native_kind(self, raw: str) -> str | None:
        query = str(raw or "").strip().lower()
        aliases = {
            "straight": "Straight",
            "taper": "Taper",
            "s bend": "S-bend",
            "sbend": "S-bend",
            "euler": "Euler bend",
            "gc": "Grating coupler",
            "grating coupler": "Grating coupler",
            "mmi": "1x2 MMI",
            "1x2 mmi": "1x2 MMI",
            "mmi reference": "MMI + Reference",
            "mmi + reference": "MMI + Reference",
            "mzi": "MZI",
            "cpw": "CPW",
            "cpw open": "CPW open",
            "cpw short": "CPW short",
            "cpw bend": "CPW bend",
            "segmented electrode": "Segmented electrode",
            "marker": "Cross mark",
        }
        if query in aliases:
            return aliases[query]
        return next(
            (kind for kind in DEFAULT_COMPONENT_VALUES if kind.lower() == query),
            None,
        )

    @staticmethod
    def parse_native_parameter_pairs(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            r"(\"[^\"]*\"|'[^']*'|[-+0-9.eE]+|[A-Za-z_]+)"
        )
        for match in pattern.finditer(text):
            value: Any = match.group(2)
            if (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]
            else:
                try:
                    value = float(value)
                except ValueError:
                    lowered = value.lower()
                    if lowered in {"true", "false"}:
                        value = lowered == "true"
            result[match.group(1)] = value
        return result

    def local_native_assistant_plan(self, prompt: str) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        parts = [
            part.strip()
            for part in re.split(r"[;\n]+", str(prompt))
            if part.strip()
        ]
        for part in parts:
            low = part.lower()
            if low.startswith("add "):
                body = re.sub(r"^add\s+", "", part, flags=re.I)
                at_match = re.search(
                    r"\bat\s*\(?\s*([-+0-9.eE]+)\s*[, ]\s*([-+0-9.eE]+)\s*\)?",
                    body,
                    flags=re.I,
                )
                rotation_match = re.search(
                    r"\b(?:rotate|orientation)\s*=?\s*([-+0-9.eE]+)",
                    body,
                    flags=re.I,
                )
                kind_text = re.split(
                    r"\bat\b|\brotate\b|\borientation\b|\bwith\b",
                    body,
                    maxsplit=1,
                    flags=re.I,
                )[0].strip()
                kind = self.canonical_native_kind(kind_text)
                if not kind:
                    raise ValueError(f"Unknown component: {kind_text}")
                actions.append(
                    {
                        "type": "add",
                        "kind": kind,
                        "x": float(at_match.group(1)) if at_match else None,
                        "y": float(at_match.group(2)) if at_match else None,
                        "orientation_deg": float(rotation_match.group(1)) if rotation_match else 0.0,
                        "params": self.parse_native_parameter_pairs(body),
                    }
                )
                continue
            match = re.match(
                r"^move\s+(?:selected\s+)?to\s*\(?\s*([-+0-9.eE]+)\s*[, ]\s*([-+0-9.eE]+)",
                part,
                flags=re.I,
            )
            if match:
                actions.append(
                    {
                        "type": "move_selected",
                        "x": float(match.group(1)),
                        "y": float(match.group(2)),
                    }
                )
                continue
            match = re.match(
                r"^move\s+(?:selected\s+)?by\s*\(?\s*([-+0-9.eE]+)\s*[, ]\s*([-+0-9.eE]+)",
                part,
                flags=re.I,
            )
            if match:
                actions.append(
                    {
                        "type": "move_selected_by",
                        "dx": float(match.group(1)),
                        "dy": float(match.group(2)),
                    }
                )
                continue
            match = re.match(
                r"^rotate(?:\s+selected)?(?:\s+by)?\s*([-+0-9.eE]+)?",
                part,
                flags=re.I,
            )
            if match and low.startswith("rotate"):
                actions.append(
                    {
                        "type": "rotate_selected",
                        "angle": float(match.group(1)) if match.group(1) else 90.0,
                    }
                )
                continue
            if low.startswith("mirror"):
                actions.append({"type": "mirror_selected"})
                continue
            if "connect" in low and "nearest" in low:
                actions.append({"type": "connect_nearest"})
                continue
            if low.startswith("delete"):
                actions.append({"type": "delete_selected"})
                continue
            if low.startswith("duplicate"):
                values = self.parse_native_parameter_pairs(part)
                actions.append(
                    {
                        "type": "duplicate_selected",
                        "dx": float(values.get("dx", 15.0)),
                        "dy": float(values.get("dy", -15.0)),
                    }
                )
                continue
            if low.startswith("select all"):
                actions.append({"type": "select_all"})
                continue
            if low.startswith("select "):
                kind = self.canonical_native_kind(
                    re.sub(r"^select\s+", "", part, flags=re.I)
                )
                actions.append({"type": "select_kind", "kind": kind})
                continue
            if low.startswith("set "):
                actions.append(
                    {
                        "type": "set_params",
                        "params": self.parse_native_parameter_pairs(part),
                    }
                )
                continue
            if "center" in low:
                actions.append({"type": "center_layout"})
                continue
            if low.startswith("fit"):
                actions.append({"type": "fit"})
                continue
            raise ValueError(f"Local mode did not understand: {part}")
        return {
            "message": f"Prepared {len(actions)} action(s).",
            "actions": actions,
        }

    def apply_native_assistant_actions(self, actions: list[dict[str, Any]]) -> None:
        if not actions:
            return
        snapshot = self.snapshot()
        selected_ids = {int(component["uid"]) for component in self.selected_components()}
        fit_after = False

        def selected_components_local() -> list[dict[str, Any]]:
            return [
                component
                for component in self.components
                if int(component["uid"]) in selected_ids
            ]

        for action in actions:
            action_type = str(action.get("type") or "")
            if action_type == "add":
                kind = self.canonical_native_kind(str(action.get("kind") or ""))
                if not kind:
                    raise ValueError(f"Unknown component: {action.get('kind')}")
                center = scene_to_world_point(
                    self.view.mapToScene(self.view.viewport().rect().center())
                )
                x = float(action["x"]) if action.get("x") is not None else center[0]
                y = float(action["y"]) if action.get("y") is not None else center[1]
                component = self.make_component(kind, x, y)
                component["orientation_deg"] = float(action.get("orientation_deg", 0.0))
                component["mirrored"] = bool(action.get("mirrored", False))
                for key, value in dict(action.get("params") or {}).items():
                    if key in component["params"]:
                        component["params"][key] = value
                _canonicalize_component_layers(component)
                self.components.append(component)
                selected_ids = {int(component["uid"])}
            elif action_type == "select_all":
                selected_ids = {int(component["uid"]) for component in self.components}
            elif action_type == "select_kind":
                kind = self.canonical_native_kind(str(action.get("kind") or ""))
                selected_ids = {
                    int(component["uid"])
                    for component in self.components
                    if component.get("kind") == kind
                }
            elif action_type == "move_selected":
                selected = selected_components_local()
                if selected:
                    anchor = selected[-1]
                    dx = float(action.get("x", anchor["x"])) - float(anchor["x"])
                    dy = float(action.get("y", anchor["y"])) - float(anchor["y"])
                    for component in selected:
                        component["attachment"] = None
                        component["x"] = float(component["x"]) + dx
                        component["y"] = float(component["y"]) + dy
            elif action_type == "move_selected_by":
                for component in selected_components_local():
                    component["attachment"] = None
                    component["x"] = float(component["x"]) + float(action.get("dx", 0.0))
                    component["y"] = float(component["y"]) + float(action.get("dy", 0.0))
            elif action_type == "rotate_selected":
                for component in selected_components_local():
                    component["attachment"] = None
                    component["orientation_deg"] = (
                        float(component.get("orientation_deg", 0.0))
                        + float(action.get("angle", 90.0))
                    ) % 360.0
            elif action_type == "mirror_selected":
                for component in selected_components_local():
                    component["mirrored"] = not bool(component.get("mirrored", False))
            elif action_type == "connect_nearest":
                selected = selected_components_local()
                if selected:
                    pair = self.nearest_port_pair(
                        selected[-1],
                        input_only=self.auto_connect_input_enabled,
                    )
                    if pair is None:
                        pair = self.nearest_port_pair(selected[-1], input_only=False)
                    if pair is not None:
                        self.apply_port_pair(selected[-1], pair)
            elif action_type == "duplicate_selected":
                source = selected_components_local()
                copies: list[dict[str, Any]] = []
                uid_map: dict[int, int] = {}
                for component in source:
                    duplicate = safe_json_copy(component)
                    old_uid = int(component["uid"])
                    duplicate["uid"] = self.next_uid
                    self.next_uid += 1
                    uid_map[old_uid] = int(duplicate["uid"])
                    duplicate["x"] = float(duplicate["x"]) + float(action.get("dx", 15.0))
                    duplicate["y"] = float(duplicate["y"]) + float(action.get("dy", -15.0))
                    duplicate["attachment"] = None
                    copies.append(duplicate)
                self.components.extend(copies)
                selected_ids = {int(component["uid"]) for component in copies}
            elif action_type == "delete_selected":
                self.components = [
                    component
                    for component in self.components
                    if int(component["uid"]) not in selected_ids
                ]
                selected_ids.clear()
            elif action_type == "set_params":
                values = dict(action.get("params") or {})
                for component in selected_components_local():
                    for key, value in values.items():
                        if key in component.get("params", {}):
                            component["params"][key] = value
                    _canonicalize_component_layers(component)
            elif action_type == "center_layout":
                if self.components:
                    library = resolve_and_build(self.components)
                    _, center = library_bbox_and_center(library)
                    for component in self.components:
                        component["x"] = float(component["x"]) - center[0]
                        component["y"] = float(component["y"]) - center[1]
            elif action_type == "fit":
                fit_after = True

        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=sorted(selected_ids))
        if fit_after:
            self.fit_design()

    def run_llm_assistant(self) -> None:
        prompt = self.llm_prompt.toPlainText().strip()
        if not prompt:
            return
        task = str(self.llm_scope.currentData() or "layout")
        mode = str(self.llm_mode.currentData() or "local")
        self.llm_prompt.clear()
        self.append_llm_chat("user", prompt)

        if mode == "local":
            try:
                if task == "source":
                    raise ValueError("Source-code updates require OpenAI cloud mode.")
                plan = self.local_native_assistant_plan(prompt)
                self.apply_native_assistant_actions(plan.get("actions", []))
                self.append_llm_chat(
                    "assistant",
                    str(plan.get("message") or "Layout actions applied."),
                )
            except Exception as exc:
                self.append_llm_chat("error", str(exc))
            return

        if self.llm_process is not None:
            self.append_llm_chat("error", "An LLM request is already running.")
            return

        request_fd, request_name = tempfile.mkstemp(
            prefix="photonic_llm_request_",
            suffix=".json",
        )
        response_fd, response_name = tempfile.mkstemp(
            prefix="photonic_llm_response_",
            suffix=".json",
        )
        os.close(request_fd)
        os.close(response_fd)
        self.llm_request_file = Path(request_name)
        self.llm_response_file = Path(response_name)
        self.llm_response_file.write_text("")
        if mode == "codex":
            codex_program = shutil.which("codex") or "/Applications/ChatGPT.app/Contents/Resources/codex"
            if not Path(codex_program).exists():
                self.append_llm_chat("error", "Codex CLI was not found. Install or sign in to Codex first.")
                self.llm_request_file.unlink(missing_ok=True); self.llm_response_file.unlink(missing_ok=True)
                self.llm_request_file = None; self.llm_response_file = None
                return
            if task == "layout":
                codex_prompt = (
                    "You are the embedded assistant for a photonic/RF layout editor. "
                    "Return ONLY JSON with a plan object containing message and actions. "
                    "Allowed action types are add, move_selected, move_selected_by, rotate_selected, "
                    "mirror_selected, connect_nearest, delete_selected, duplicate_selected, set_selected, fit. "
                    f"Current layout: {json.dumps(self.compact_layout_summary_native())}\nUser request: {prompt}"
                )
            else:
                codex_prompt = (
                    "Give concise source-code debugging guidance for this photonic and RF layout editor. "
                    "Return ONLY JSON with keys message and output_file, with output_file empty. "
                    f"User request: {prompt}"
                )
            process = QProcess(self); self.llm_process = process; self.llm_send_button.setEnabled(False)
            process.setProgram(codex_program)
            process.setArguments(["exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "workspace-write", "-C", str(Path.cwd()), "-o", str(self.llm_response_file), codex_prompt])
            process.finished.connect(self.llm_request_finished); process.start(); self.statusBar().showMessage("Codex assistant working…")
            return
        self.llm_request_file.write_text(
            json.dumps(
                {
                    "task": task,
                    "prompt": prompt,
                    "model": self.llm_model.text().strip() or "gpt-5.6",
                    "api_key": self.llm_api_key.text().strip(),
                    "layout": self.compact_layout_summary_native(),
                }
            )
        )

        process = QProcess(self)
        self.llm_process = process
        self.llm_send_button.setEnabled(False)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                launcher_path(),
                "--worker-llm",
                str(self.llm_request_file),
                str(self.llm_response_file),
            ]
        )
        process.finished.connect(self.llm_request_finished)
        process.start()
        self.statusBar().showMessage("LLM assistant working…")

    def llm_request_finished(self, exit_code: int, exit_status) -> None:
        process = self.llm_process
        stderr = (
            bytes(process.readAllStandardError()).decode(errors="replace")
            if process is not None
            else ""
        )
        try:
            if exit_code != 0:
                raise ValueError(stderr or f"LLM worker exited with code {exit_code}.")
            if self.llm_response_file is None:
                raise ValueError("The LLM worker produced no response file.")
            response = json.loads(self.llm_response_file.read_text())
            if "plan" in response:
                plan = response["plan"]
                self.apply_native_assistant_actions(plan.get("actions", []))
                self.append_llm_chat(
                    "assistant",
                    str(plan.get("message") or "Layout actions applied."),
                )
            else:
                message = str(response.get("message") or "Source update created.")
                output_file = str(response.get("output_file") or "")
                self.append_llm_chat(
                    "assistant",
                    f"{message}\nSaved: {output_file}",
                )
            self.statusBar().showMessage("LLM request complete.")
        except Exception as exc:
            self.append_llm_chat("error", str(exc))
            self.statusBar().showMessage("LLM request failed.")
        finally:
            if self.llm_request_file:
                self.llm_request_file.unlink(missing_ok=True)
            if self.llm_response_file:
                self.llm_response_file.unlink(missing_ok=True)
            self.llm_request_file = None
            self.llm_response_file = None
            self.llm_process = None
            self.llm_send_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------
    def contextual_parameter_sections(
        self,
        component: dict[str, Any],
        local_point: QPointF,
        clicked_layer: int | None,
    ) -> list[tuple[str, list[str]]]:
        """Return the most likely clicked section first, followed by alternatives."""
        kind = str(component.get("kind", ""))
        p = component.get("params", {})
        groups: list[tuple[str, list[str]]] = []

        def add(name: str, keys: tuple[str, ...]) -> None:
            available = [key for key in keys if key in p]
            if available and not any(existing == name for existing, _ in groups):
                groups.append((name, available))

        gc_keys = (
            "pitch", "fill_factor", "N", "alpha_t", "taper_L", "wg_width", "wg_length",
            "gc_pitch", "gc_fill_factor", "gc_N", "gc_alpha_t", "gc_taper_L", "gc_wg_length",
        )
        mmi_keys = (
            "mmi_width", "mmi_length", "wg_width", "taper_width", "input_taper_length",
            "output_taper_length", "input_length", "output_length", "port_sep",
            "taper_power", "taper_points",
        )
        arm_keys = ("arm_length", "arm_separation", "wg_width")
        sbend_keys = ("s_bend_length", "arm_separation", "wg_width")
        vertical_route_keys = (
            "gc_prebend_straight", "gc_vertical_run", "gc_inward_run_fraction",
            "gc_align_gc_to_mzi_center", "gc_inward_run", "gc_euler_radius",
            "gc_euler_fraction", "gc_vertical_side",
        )
        cpw_keys = (
            "cpw_taper_length", "cpw_s_bend_clearance", "cpw_signal_width",
            "cpw_ground_width", "cpw_end_gap", "cpw_middle_gap", "cpw_profile",
            "cpw_target_s11_db", "cpw_exponential_factor", "cpw_points",
        )
        segmented_keys = (
            "mzi_total_length", "seg_taper_length", "seg_end_flat_length", "seg_inner_flat_length", "seg_s_bend_clearance", "seg_signal_width", "seg_end_gap", "seg_gap",
            "seg_ground_width", "seg_t_top_width", "seg_t_top_length", "seg_t_neck_width",
            "seg_t_neck_length", "seg_segment_spacing", "seg_segment_count", "seg_auto_segment_count", "seg_include_oxide_masks",
        )
        field_keys = ("include_ebeam_fields", "ebeam_field_size", "ebeam_edge_clearance", "parameter_text_height")

        # The actual polygon layer is the strongest indication of what was clicked.
        if clicked_layer == GC_LAYER:
            add("Grating coupler", gc_keys)
        elif clicked_layer == RF_LAYER:
            if "segmented" in kind.lower():
                add("T-segmented electrode", segmented_keys)
            else:
                add("CPW taper", cpw_keys)
        elif clicked_layer == EBEAM_LAYER:
            add("E-beam write fields", field_keys)

        if kind == "Grating coupler":
            add("Grating coupler", gc_keys)
        elif kind in {"Straight", "Taper", "S-bend", "Euler bend"}:
            add(kind, tuple(p.keys()))
        elif "MZI" in kind or kind in {"1x2 MMI", "MMI + Reference", "MMI + Reference test block", "MMI split-combine test block"}:
            x = float(local_point.x())
            y = -float(local_point.y())
            mlen = mmi_total_length(p) if all(key in p for key in ("input_length", "input_taper_length", "mmi_length", "output_taper_length", "output_length")) else 0.0
            sb = float(p.get("s_bend_length", 0.0))
            total = float(p.get("mzi_total_length", 2 * mlen + 2 * sb + float(p.get("arm_length", 0.0))))
            # For test arrays, fold the click into the closest row's device coordinates.
            if "test block" in kind and total > 0:
                x = x % max(total, 1.0)
            if abs(y) > max(float(p.get("arm_separation", 0.0)), float(p.get("port_sep", 0.0))) + 10.0:
                add("Vertical Euler / GC route", vertical_route_keys)
            elif mlen and (x <= mlen or x >= total - mlen):
                add("MMI body and tapers", mmi_keys)
            elif sb and (x <= mlen + sb or x >= total - mlen - sb):
                add("Optical S-bends", sbend_keys)
            else:
                add("Straight MZI arms", arm_keys)
            add("MMI body and tapers", mmi_keys)
            add("Optical S-bends", sbend_keys)
            add("Straight MZI arms", arm_keys)
            add("Vertical Euler / GC route", vertical_route_keys)
            add("Grating coupler", gc_keys)
            add("CPW taper", cpw_keys)
            add("T-segmented electrode", segmented_keys)
            add("E-beam write fields", field_keys)
        else:
            add(kind or "Device", tuple(key for key in p.keys() if key not in {"polygons","manual_field_offsets","manual_field_order","explicit_fields"}))
        return groups

    def show_contextual_component_properties(
        self,
        component: dict[str, Any],
        section_title: str,
        keys: list[str],
    ) -> None:
        """Populate the dock with only the controls belonging to a clicked section."""
        self.clear_properties()
        title = QLabel(
            f"<b>{section_title}</b><br>"
            f"{component.get('kind')} · UID {component.get('uid')}"
        )
        title.setWordWrap(True)
        self.properties_form.addRow(title)
        hint = QLabel("Local section controls — changes rebuild the complete parent device.")
        hint.setWordWrap(True)
        self.properties_form.addRow(hint)
        specs = COMPONENT_SPECS.get(component.get("kind"), {})
        for key in keys:
            if key in {"polygons","manual_field_offsets","manual_field_order","explicit_fields"}:continue
            if key not in component.get("params", {}):
                continue
            value = component["params"][key]
            spec = specs.get(key)
            if spec is None:
                spec = ["bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float" if isinstance(value, float) else "string", value]
            widget = self.make_parameter_widget(key, spec, value)
            self.properties_form.addRow(key.replace("_", " "), widget)
            self.parameter_widgets[f"params.{key}"] = (widget, spec[0])
        all_button = QPushButton("Show all component parameters…")
        all_button.clicked.connect(lambda: self.show_component_properties(component))
        self.properties_form.addRow(all_button)
        self.apply_properties_button.setEnabled(True)
        self.properties_scroll.verticalScrollBar().setValue(0)
        self.properties_dock.raise_()

    def show_properties_for_scene_click(
        self,
        component_item: ComponentGraphicsItem,
        scene_position: QPointF,
    ) -> None:
        """Resolve the exact polygon under a normal click and show its local controls."""
        hit_item = self.scene.itemAt(scene_position, self.view.transform())
        clicked_layer = None
        current = hit_item
        while current is not None and current is not component_item:
            if current.data(0) is not None:
                try:
                    clicked_layer = int(current.data(0))
                    break
                except (TypeError, ValueError):
                    pass
            current = current.parentItem()
        component = self.component_by_uid(component_item.uid)
        if component is None:
            return
        local = component_item.mapFromScene(scene_position)
        sections = self.contextual_parameter_sections(component, local, clicked_layer)
        if sections:
            self.show_contextual_component_properties(component, sections[0][0], sections[0][1])

    def edit_component_section(self, component: dict[str, Any], title: str, keys: list[str]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{component.get('kind')} — {title}")
        dialog.setMinimumWidth(430)
        outer = QVBoxLayout(dialog)
        note = QLabel(f"<b>{title}</b><br>Only parameters for the clicked section are shown.")
        note.setWordWrap(True)
        outer.addWidget(note)
        form = QFormLayout()
        widgets: dict[str, tuple[QWidget, str]] = {}
        specs = COMPONENT_SPECS.get(component.get("kind"), {})
        for key in keys:
            if key in {"polygons","manual_field_offsets","manual_field_order","explicit_fields"}:continue
            value = component.get("params", {}).get(key)
            spec = specs.get(key)
            if spec is None:
                spec = ["bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float" if isinstance(value, float) else "string", value]
            widget = self.make_parameter_widget(key, spec, value)
            form.addRow(key.replace("_", " "), widget)
            widgets[key] = (widget, spec[0])
        outer.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        outer.addWidget(buttons)
        applied = {"snapshot": None}

        def apply_section() -> None:
            if applied["snapshot"] is None:
                applied["snapshot"] = self.snapshot()
            for key, (widget, spec_type) in widgets.items():
                component["params"][key] = self.read_parameter_widget(widget, spec_type)
            self.refresh_component_scene_item(int(component["uid"]), selected=True)
            self.statusBar().showMessage(f"{title} updated; parent device rebuilt.")

        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply_section)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            apply_section()
        if applied["snapshot"] is not None:
            self.commit_interaction_snapshot(applied["snapshot"])

    def show_canvas_context_menu(self, viewport_position: QPoint) -> None:
        menu = QMenu(self)
        _force_dark_popup(menu)
        scene_position = self.view.mapToScene(viewport_position)
        hit_item = self.scene.itemAt(scene_position, self.view.transform())
        clicked_layer = None
        component_item = hit_item
        if hit_item is not None and hit_item.data(0) is not None:
            try:
                clicked_layer = int(hit_item.data(0))
            except (TypeError, ValueError):
                clicked_layer = None
        while component_item is not None and component_item.data(10) is None:
            component_item = component_item.parentItem()
        section_actions: dict[QAction, tuple[dict[str, Any], str, list[str]]] = {}
        if component_item is not None and component_item.data(10) is not None:
            component = self.component_by_uid(int(component_item.data(10)))
            if component is not None:
                component_item.setSelected(True)
                local = component_item.mapFromScene(scene_position)
                sections = self.contextual_parameter_sections(component, local, clicked_layer)
                if sections:
                    primary_title, primary_keys = sections[0]
                    primary_action = menu.addAction(f"Edit {primary_title}…")
                    font = primary_action.font()
                    font.setBold(True)
                    primary_action.setFont(font)
                    section_actions[primary_action] = (component, primary_title, primary_keys)
                    if len(sections) > 1:
                        section_menu = menu.addMenu("Edit another section")
                        for section_title, section_keys in sections[1:]:
                            action = section_menu.addAction(section_title)
                            section_actions[action] = (component, section_title, section_keys)
                    menu.addSeparator()
        save_module_action = menu.addAction("Add selection to User modules…")
        save_module_action.setEnabled(bool([component for component in self.selected_components() if component.get("kind") != "E-beam multipass"]))
        lattice_action=menu.addAction("Create photonic-crystal lattice…");lattice_action.setEnabled(save_module_action.isEnabled())
        boolean_menu=menu.addMenu("Boolean operation");boolean_actions={boolean_menu.addAction(label):op for label,op in (("Union","union"),("Difference (first minus rest)","difference"),("Intersection","intersection"),("XOR","xor"))};boolean_menu.setEnabled(len(self.selected_components())>=2)
        delete_action = menu.addAction("Delete Selected")
        delete_action.setEnabled(bool(self.scene.selectedItems()) or bool(self.active_field))
        menu.addSeparator()
        menu.addAction("Fit Selection", self.fit_selection)
        menu.addAction("Fit Design", self.fit_design)
        chosen = menu.exec(self.view.viewport().mapToGlobal(viewport_position))
        if chosen in section_actions:
            self.edit_component_section(*section_actions[chosen])
        elif chosen is save_module_action:
            self.save_selection_as_module()
        elif chosen is lattice_action:
            self.create_photonic_crystal_lattice()
        elif chosen in boolean_actions:
            self.boolean_selected_geometry(boolean_actions[chosen])
        elif chosen is delete_action:
            self.delete_selected()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.export_process is not None:
            self.export_process.kill()
        if self.llm_process is not None:
            self.llm_process.kill()
        self.field_play_timer.stop()
        event.accept()
