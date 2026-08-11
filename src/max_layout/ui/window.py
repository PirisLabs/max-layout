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
from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsScene, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea, QSpinBox, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..constants import CHOICE_PARAMETERS, COMPONENT_SPECS, DEFAULT_COMPONENT_VALUES, EBEAM_LAYER, GC_LAYER, LAYER_NAME_MAP, LEGACY_PHOTONIC_TEST_BLOCK_KINDS, MARKER_COMPONENT_KINDS, MARKER_LAYER, NATIVE_APP_VERSION, PHOTONIC_COMPONENT_KINDS, PHOTONIC_LAYER, RF_COMPONENT_KINDS, RF_LAYER, SIMULATION_COMPONENT_KINDS, SIMULATION_LAYER, component_display_name
from ..gds.build import _add_component_geometry_to_cell, _canonicalize_component_layers, component_geometry_arrays, library_bbox_and_center, recenter_components_at_origin, resolve_and_build, rotate_components_layout, test_block_device_placements
from ..gds.ebeam import multipass_field_layout
from ..geometry.shapes import mmi_total_length
from ..geometry.rf_taper import synchronize_rf_taper_points
from ..geometry.transforms import scene_to_world_point, transform_points, transformed_local_points, world_to_scene_point
from ..modules_db import load_native_modules, save_native_modules
from ..lumerical import (
    apply_lumerical_sweep_values,
    default_stack,
    expand_lumerical_sweep_points,
    write_lumerical_multigpu_sweep_notebook,
    write_lumerical_notebook,
    write_lumerical_sweep_notebook,
)
from ..lumerical_optimization import write_lumerical_adjoint_notebook
from ..lumerical_rf import build_lumerical_rf_preview_state, write_lumerical_rf_notebook
from ..params import resize_component_parameters
from ..ports import PORT_ALIASES, component_global_ports, component_local_ports, solve_attachment
from ..rf_defaults import RF_SIMULATABLE_COMPONENT_KINDS, RF_STACK_PRESETS, default_rf_configuration, normalize_rf_configuration
from ..ui.dialogs import ArrayDialog, EbeamDialog, ModuleVariablesDialog
from ..ui.items import ComponentGraphicsItem, EbeamContainerItem, LayoutView, WriteFieldItem, clear_preview_caches
from ..ui.lumerical_dialog import (
    LumericalExportDialog,
    LumericalMultigpuSweepDialog,
    LumericalOptimizationDialog,
    LumericalSweepDialog,
    ThreeDModelPreview,
)
from ..ui.theme import _force_dark_popup, color_for_layer
from ..utils import inclusive_sweep, numeric_list, parse_sequence, safe_json_copy


CHIP_BOUNDARY_MARGIN_UM = 50.0
WAFER_BOUNDARY_MARGIN_UM = 5000.0  # 0.5 cm keep-out from the wafer edge
BOUNDARY_MARGINS_UM = {"Chip outline": CHIP_BOUNDARY_MARGIN_UM, "4-inch wafer outline": WAFER_BOUNDARY_MARGIN_UM}
BOUNDARY_COMPONENT_KINDS = set(BOUNDARY_MARGINS_UM)
MMI_STACK_PRESET = "TFLN MMI (3 um SiO2)"
MMI_STACK_VERSION = 1
GRATING_AUTO_MESH_VERSION = 1


def _material_stack_signature(stack: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    """Comparable stack signature used only for safe default migration."""
    signature = []
    for row in stack:
        role = "geometry" if str(row.get("role", "background")).lower() == "geometry" else "background"
        thickness_um = float(row.get("thickness_um", 0.0))
        raw_layers = row.get("gds_layers", [row.get("gds_layer", 0)])
        if isinstance(raw_layers, (str, int, float)):
            raw_layers = [raw_layers]
        signature.append(
            (
                str(row.get("name", "")),
                str(row.get("material", "")),
                thickness_um,
                float(row.get("etch_depth_um", thickness_um if role == "geometry" else 0.0)),
                float(row.get("sidewall_angle_deg", 90.0)),
                role,
                tuple(int(value) for value in raw_layers),
                "geometry" if str(row.get("slab_extent", "full")).lower() == "geometry" else "full",
                float(row.get("mesh_factor", 0.2)),
                int(row.get("mesh_order", 3 if bool(row.get("conformal", False)) else 2)),
                bool(row.get("conformal", False)),
            )
        )
    return tuple(signature)


def mmi_lumerical_export_settings(saved: dict[str, Any] | None) -> dict[str, Any]:
    """Use the oxide-only MMI stack unless the user already customized it."""
    settings = copy.deepcopy(saved or {})
    saved_stack = settings.get("material_stack")
    legacy_default = (
        int(settings.get("mmi_stack_version", 0)) < MMI_STACK_VERSION
        and str(settings.get("stack_preset", "TFLN on SiO2")) == "TFLN on SiO2"
        and (
            not saved_stack
            or _material_stack_signature(list(saved_stack))
            == _material_stack_signature(default_stack("TFLN on SiO2"))
        )
    )
    if not settings or legacy_default:
        settings["stack_preset"] = MMI_STACK_PRESET
        settings["material_stack"] = default_stack(MMI_STACK_PRESET)
    settings["mmi_stack_version"] = MMI_STACK_VERSION
    return settings


def grating_lumerical_export_settings(saved: dict[str, Any] | None) -> dict[str, Any]:
    """Default a TFLN grating coupler to Lumerical's automatic mesh.

    A zero per-layer mesh factor means that no layer mesh-override object is
    created, leaving the 3D FDTD mesh accuracy setting in control.  Migrate
    only the former untouched TFLN preset so user-edited stacks remain intact.
    """
    settings = copy.deepcopy(saved or {})
    saved_stack = settings.get("material_stack")
    legacy_default = (
        int(settings.get("grating_auto_mesh_version", 0)) < GRATING_AUTO_MESH_VERSION
        and str(settings.get("stack_preset", "TFLN on SiO2")) == "TFLN on SiO2"
        and (
            not saved_stack
            or _material_stack_signature(list(saved_stack))
            == _material_stack_signature(default_stack("TFLN on SiO2"))
        )
    )
    if not settings or legacy_default:
        automatic_stack = default_stack("TFLN on SiO2")
        for row in automatic_stack:
            row["mesh_factor"] = 0.0
        settings["stack_preset"] = "TFLN on SiO2"
        settings["material_stack"] = automatic_stack
    settings["grating_auto_mesh_version"] = GRATING_AUTO_MESH_VERSION
    return settings


def automatic_waveguide_port_span_um(
    component: dict[str, Any], port_name: str = ""
) -> float:
    """Platform-aware transverse span for automatically placed waveguide planes.

    The current general photonic library is TFLN-first, so its automatic
    planes are at least 3 um and at least twice the endpoint waveguide width.
    GC-SOI remains tied to the separate official-example span.
    """
    kind = str(component.get("kind", ""))
    params = component.get("params", {})
    if kind == "GC-SOI":
        return max(0.5, float(params.get("waveguide_monitor_span_um", 2.5)))
    if kind == "Taper":
        width_um = float(
            params.get("width_start", 1.2)
            if str(port_name) == "left"
            else params.get("width_end", params.get("width_start", 1.2))
        )
    elif kind in {"Grating coupler", "1x2 MMI", "Cascaded MMI", "MMI + Reference"}:
        width_um = float(params.get("wg_width", params.get("width", 1.2)))
    else:
        width_um = float(params.get("width", params.get("wg_width", 1.2)))
    requested_um = float(
        params.get(
            "waveguide_monitor_span_um",
            params.get("waveguide_port_span_um", 0.0),
        )
    )
    return max(3.0, 2.0 * abs(width_um), requested_um)


def _standard_grating_focus_offset_um(params: dict[str, Any]) -> float:
    """Centerline shift between nominal taper length and its actual focus."""
    aperture_deg = float(params.get("alpha_t", 25.0))
    half_angle_rad = 0.5 * math.radians(aperture_deg)
    tangent = math.tan(half_angle_rad)
    if not 0.0 < aperture_deg < 180.0 or abs(tangent) < 1e-15:
        raise ValueError("Grating aperture angle must be between 0 and 180 degrees")
    return 0.5 * float(params.get("wg_width", 1.2)) / tangent


def migrate_grating_fiber_offset_parameter(component: dict[str, Any]) -> bool:
    """Normalize legacy grating alignment names to canonical parent keys.

    The old standard-GC reference omitted the focusing taper's focal shift.
    Its value is converted so an existing layout keeps the same physical fiber
    center.  The canonical value is always a signed micrometre distance along
    local X and never introduces a local-Y displacement.  ``angle_theta`` is
    likewise the single parent value shared by the fiber structure, source
    port, and passive power plane; old SOI projects used ``fiber_tilt_deg``.
    """
    kind = str(component.get("kind", ""))
    if kind not in {"Grating coupler", "GC-SOI"}:
        return False
    params = component.setdefault("params", {})
    changed = False
    legacy_keys = (
        (
            "fiber_x_from_grating_start_um",
            "fiber_offset_after_taper_um",
            "fiber_offset_from_flare_um",
        )
        if kind == "GC-SOI"
        else (
            "fiber_offset_after_taper_um",
            "fiber_offset_from_flare_um",
            "fiber_x_from_grating_start_um",
        )
    )
    if "fiber_offset" not in params:
        default_value = 2.74533 if kind == "GC-SOI" else 5.0
        legacy_value = next(
            (params[key] for key in legacy_keys if key in params),
            None,
        )
        if legacy_value is None:
            params["fiber_offset"] = default_value
        elif kind == "GC-SOI":
            params["fiber_offset"] = float(legacy_value)
        else:
            # Old x = wg_length + taper_L + legacy.  New x uses the
            # geometry-exact flare boundary, so add the omitted focal shift.
            params["fiber_offset"] = (
                float(legacy_value) + _standard_grating_focus_offset_um(params)
            )
        changed = True
    for key in legacy_keys:
        if key in params:
            params.pop(key, None)
            changed = True
    if "angle_theta" not in params:
        legacy_theta = params.get("fiber_tilt_deg")
        params["angle_theta"] = float(
            legacy_theta if legacy_theta is not None
            else (10.0 if kind == "GC-SOI" else 7.0)
        )
        changed = True
    if "fiber_tilt_deg" in params:
        params.pop("fiber_tilt_deg", None)
        changed = True
    return changed


def grating_angle_theta_deg(component: dict[str, Any]) -> float:
    """Return the one parent-controlled fiber/source tilt in degrees."""
    kind = str(component.get("kind", ""))
    if kind not in {"Grating coupler", "GC-SOI"}:
        raise ValueError("angle_theta is defined only for grating couplers")
    # This migration writes the editable UI default or a legacy fiber_tilt_deg
    # into the parent JSON once.  Every companion reads that one stored value.
    migrate_grating_fiber_offset_parameter(component)
    params = component.get("params", {})
    if "angle_theta" not in params:
        raise ValueError("Grating parent JSON is missing its authoritative angle_theta")
    value = float(params["angle_theta"])
    if not math.isfinite(value) or value < 0.0 or value >= 90.0:
        raise ValueError("Grating fiber angle_theta must be at least 0 and below 90 degrees")
    return value


def grating_first_flare_local_x_um(component: dict[str, Any]) -> float:
    """Nominal first-flare plane in the grating component's local X frame."""
    kind = str(component.get("kind", ""))
    params = component.get("params", {})
    if kind == "GC-SOI":
        return float(params.get("wg_length", 10.0)) + float(params.get("radius", 25.0))
    if kind == "Grating coupler":
        return (
            float(params.get("wg_length", 5.0))
            - _standard_grating_focus_offset_um(params)
            + float(params.get("taper_L", 22.0))
        )
    raise ValueError("Fiber offset is defined only for grating couplers")


def grating_fiber_center_local_um(component: dict[str, Any]) -> tuple[float, float]:
    """Fiber bottom center: first flare plus ``fiber_offset`` on local X only."""
    params = component.get("params", {})
    if "fiber_offset" in params:
        offset_um = float(params["fiber_offset"])
    else:
        kind = str(component.get("kind", ""))
        legacy_keys = (
            ("fiber_x_from_grating_start_um", "fiber_offset_after_taper_um", "fiber_offset_from_flare_um")
            if kind == "GC-SOI"
            else ("fiber_offset_after_taper_um", "fiber_offset_from_flare_um", "fiber_x_from_grating_start_um")
        )
        offset_um = float(next(
            (params[key] for key in legacy_keys if key in params),
            2.74533 if kind == "GC-SOI" else 5.0,
        ))
        if kind == "Grating coupler" and any(key in params for key in legacy_keys):
            offset_um += _standard_grating_focus_offset_um(params)
    return grating_first_flare_local_x_um(component) + offset_um, 0.0


def add_fixed_default_row(table: QTableWidget, row: int, key: str, value: Any) -> tuple[str, Any, bool, Any]:
    """Render a non-scanned parameter as a single editable default cell and return its reader spec."""
    table.setItem(row, 0, QTableWidgetItem(key.replace("_", " ")))
    for column in (1, 2, 3):
        table.setItem(row, column, QTableWidgetItem("—"))
    if key in CHOICE_PARAMETERS or isinstance(value, bool):
        allowed = CHOICE_PARAMETERS.get(key, [False, True] if isinstance(value, bool) else [value])
        box = QComboBox();box.addItems([str(option).lower() if isinstance(option, bool) else str(option) for option in allowed])
        box.setCurrentText(str(value).lower() if isinstance(value, bool) else str(value));box.setMinimumSize(290, 42)
        table.setCellWidget(row, 4, box);return ("choice", box, isinstance(value, bool), value)
    if isinstance(value, str):
        entry = QLineEdit(str(value));entry.setMinimumSize(290, 42)
        table.setCellWidget(row, 4, entry);return ("text", entry, False, value)
    box = QDoubleSpinBox();box.setRange(-1e9, 1e9);box.setDecimals(0 if isinstance(value, int) else 9)
    box.setValue(float(value));box.setMinimumSize(290, 42)
    table.setCellWidget(row, 4, box);return ("number", box, isinstance(value, int), value)


def read_fixed_default(spec: tuple[str, Any, bool, Any]) -> Any:
    """Pull the edited default back out of the widget created by :func:`add_fixed_default_row`."""
    mode, widget, flag, original = spec
    if mode == "choice":
        text = widget.currentText()
        return text.lower() in {"1", "true", "yes", "on"} if flag else text
    if mode == "text":
        return widget.text()
    # An untouched spin box must not quantize a value carrying more digits than the box displays.
    if abs(widget.value() - float(original)) <= 0.5 * 10 ** -widget.decimals():
        return original
    return int(round(widget.value())) if flag else float(widget.value())


RF_SIMULATION_OBJECT_KINDS = {"RF mode port", "RF power monitor"}


def _rf_spin(
    value: float,
    minimum: float,
    maximum: float,
    decimals: int = 6,
    step: float | None = None,
) -> QDoubleSpinBox:
    """Create a consistently sized RF-settings numeric control."""
    widget = QDoubleSpinBox()
    widget.setRange(float(minimum), float(maximum))
    widget.setDecimals(int(decimals))
    widget.setValue(float(value))
    if step is not None:
        widget.setSingleStep(float(step))
    widget.setMinimumWidth(190)
    return widget


class RFLumericalExportDialog(QDialog):
    """RF-specific MODE/FDTD setup following the official CPW examples.

    Optical FDTD ports are intentionally ignored.  A 3D RF run uses explicit
    ``RF mode port`` / ``RF power monitor`` objects from the selected scope,
    with a clearly labelled component-endpoint fallback for early layouts.
    """

    MATERIAL_HEADERS = (
        "Name",
        "Role",
        "Thickness (µm)",
        "εr",
        "εx",
        "εy",
        "εz",
        "loss tan δ",
        "conductivity (S/m)",
        "GDS layer(s)",
        "Metal model",
    )

    def __init__(
        self,
        components: list[dict[str, Any]],
        target_component: dict[str, Any],
        scope_options: list[tuple[str, list[int]]],
        saved: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.components = components
        self.target_component = target_component
        self.kind = str(target_component.get("kind", ""))
        defaults = default_rf_configuration(self.kind)
        self.values = copy.deepcopy(defaults)
        self.values.update(copy.deepcopy(saved or {}))
        # Solver and hardware are determined by the official-example
        # workflow, never inherited from an older optical export.
        self.values["rf_workflow"] = defaults["rf_workflow"]
        self.values["resource_mode"] = defaults["resource_mode"]
        self._active_rf_preview_state: dict[str, Any] | None = None
        self.setWindowTitle(f"Lumerical RF — {component_display_name(self.kind)}")
        self.resize(1420, 860)
        self.setMinimumSize(1180, 720)

        root = QVBoxLayout(self)
        title = QLabel(
            "Straight CPW uses the official 2D MODE/FDE impedance workflow. "
            "Tapers, bends, opens, shorts, and segmented electrodes use 3D FDTD S-parameters."
        )
        title.setWordWrap(True)
        root.addWidget(title)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_run_tab(scope_options)
        self._build_material_tab()
        self._build_mesh_tab()
        self._build_preview_tab()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Export RF notebook")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_run_tab(self, scope_options: list[tuple[str, list[int]]]) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        scope_group = QGroupBox("Geometry and RF planes")
        scope_form = QFormLayout(scope_group)
        self.scope_combo = QComboBox()
        for label, uids in scope_options:
            self.scope_combo.addItem(str(label), list(map(int, uids)))
        scope_form.addRow("Export geometry", self.scope_combo)
        self.scope_report = QLabel()
        self.scope_report.setWordWrap(True)
        scope_form.addRow("Detected RF objects", self.scope_report)
        self.endpoint_fallback = QCheckBox(
            "If manual RF planes are missing, use the component input/output endpoints"
        )
        self.endpoint_fallback.setChecked(
            bool(self.values.get("use_endpoint_reference_planes", True))
        )
        self.endpoint_fallback.setToolTip(
            "This creates RF reference planes in the notebook only. It does not create optical ports or GDS geometry."
        )
        scope_form.addRow("Endpoint fallback", self.endpoint_fallback)
        note = QLabel(
            "Preferred: place RF mode port and RF power monitor objects from the left Ports & monitors section, "
            "attach/select them with this CPW, and position them manually. Optical FDTD ports are never reused."
        )
        note.setWordWrap(True)
        scope_form.addRow("Port policy", note)
        layout.addWidget(scope_group)

        sweep_group = QGroupBox("RF frequency sweep")
        sweep_form = QFormLayout(sweep_group)
        workflow = "2D MODE/FDE · quasi-TEM Z₀ and propagation" if self.values["rf_workflow"] == "fde" else "3D FDTD · S11, S21, insertion loss, and phase"
        self.workflow_label = QLabel(workflow)
        sweep_form.addRow("Official-example workflow", self.workflow_label)
        self.frequency_start = _rf_spin(self.values.get("frequency_start_ghz", 1.0), 1e-6, 1e6, 6, 1.0)
        self.frequency_stop = _rf_spin(self.values.get("frequency_stop_ghz", 100.0), 1e-6, 1e6, 6, 1.0)
        self.target_frequency = _rf_spin(self.values.get("target_frequency_ghz", 30.0), 1e-6, 1e6, 6, 1.0)
        self.frequency_points = QSpinBox()
        self.frequency_points.setRange(2, 10001)
        self.frequency_points.setValue(int(self.values.get("frequency_points", 25)))
        self.frequency_points.setMinimumWidth(190)
        sweep_form.addRow("Start frequency (GHz)", self.frequency_start)
        sweep_form.addRow("Stop frequency (GHz)", self.frequency_stop)
        sweep_form.addRow("Frequency samples", self.frequency_points)
        sweep_form.addRow("Reported target frequency (GHz)", self.target_frequency)
        layout.addWidget(sweep_group)

        port_group = QGroupBox("RF plane defaults")
        port_form = QFormLayout(port_group)
        self.input_inset = _rf_spin(self.values.get("input_port_inset_um", 0.0), -1e7, 1e7, 6, 1.0)
        self.output_inset = _rf_spin(self.values.get("output_port_inset_um", 0.0), -1e7, 1e7, 6, 1.0)
        self.port_transverse_span = _rf_spin(self.values.get("port_transverse_span_um", 450.0), 1e-6, 1e7, 6, 10.0)
        self.port_vertical_span = _rf_spin(self.values.get("port_vertical_span_um", 650.0), 1e-6, 1e7, 6, 10.0)
        self.multifrequency_injection = QCheckBox("Calculate/inject the quasi-TEM mode across the frequency sweep")
        self.multifrequency_injection.setChecked(bool(self.values.get("multifrequency_mode_injection", True)))
        port_form.addRow("Input plane inset (µm)", self.input_inset)
        port_form.addRow("Output plane inset (µm)", self.output_inset)
        port_form.addRow("Transverse span (µm)", self.port_transverse_span)
        port_form.addRow("Vertical span (µm)", self.port_vertical_span)
        port_form.addRow("Broadband mode", self.multifrequency_injection)
        if self.values["rf_workflow"] == "fde":
            self.endpoint_fallback.setChecked(False)
            self.endpoint_fallback.setEnabled(False)
            for widget in (self.input_inset, self.output_inset):
                widget.setEnabled(False)
            self.scope_report.setToolTip("A uniform CPW FDE run solves one transverse cross-section and does not need longitudinal ports.")
        layout.addWidget(port_group)
        layout.addStretch(1)
        self.tabs.addTab(page, "Run & ports")
        self.scope_combo.currentIndexChanged.connect(self._update_scope_report)
        self._update_scope_report()

    def _build_material_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("RF stack preset"))
        self.stack_preset = QComboBox()
        self.stack_preset.addItems(list(RF_STACK_PRESETS))
        preset = str(self.values.get("rf_stack_preset", "TFLN CPW"))
        if self.stack_preset.findText(preset) >= 0:
            self.stack_preset.setCurrentText(preset)
        reload_button = QPushButton("Load preset values")
        reload_button.clicked.connect(self._load_selected_stack_preset)
        preset_row.addWidget(self.stack_preset, 1)
        preset_row.addWidget(reload_button)
        layout.addLayout(preset_row)

        help_label = QLabel(
            "RF material values are relative permittivity and loss tangent, not optical refractive index. "
            "For anisotropic media leave εr blank and fill εx, εy, and εz. The metal row maps the selected GDS layers to PEC or a finite-conductivity volume."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.material_table = QTableWidget(0, len(self.MATERIAL_HEADERS))
        self.material_table.setHorizontalHeaderLabels(self.MATERIAL_HEADERS)
        self.material_table.setAlternatingRowColors(True)
        self.material_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.material_table.setMinimumHeight(430)
        self.material_table.verticalHeader().setDefaultSectionSize(48)
        self.material_table.horizontalHeader().setMinimumSectionSize(120)
        for column, width in enumerate((205, 215, 165, 120, 120, 120, 120, 155, 205, 145, 185)):
            self.material_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            self.material_table.setColumnWidth(column, width)
        self.material_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.material_table, 1)

        controls = QHBoxLayout()
        add_dielectric = QPushButton("Add dielectric")
        add_metal = QPushButton("Add metal")
        remove = QPushButton("Remove selected row")
        preview_button = QPushButton("Show RF 3D preview")
        add_dielectric.clicked.connect(lambda: self._append_material_row({
            "name": "New dielectric", "role": "dielectric", "thickness_um": 1.0,
            "relative_permittivity": 1.0, "loss_tangent": 0.0,
            "conductivity_s_per_m": 0.0,
        }))
        add_metal.clicked.connect(lambda: self._append_material_row({
            "name": "RF metal", "role": "metal", "thickness_um": 1.0,
            "metal_model": "Conductive 3D", "conductivity_s_per_m": 4.1e7,
            "gds_layers": [4, 5],
        }))
        remove.clicked.connect(self._remove_selected_material_rows)
        preview_button.clicked.connect(self._show_rf_3d_preview)
        controls.addWidget(add_dielectric)
        controls.addWidget(add_metal)
        controls.addWidget(remove)
        controls.addWidget(preview_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._set_material_stack(list(self.values.get("material_stack") or []))
        self.tabs.addTab(page, "RF material stack")

    def _build_mesh_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        mesh_group = QGroupBox("Mesh")
        mesh_form = QFormLayout(mesh_group)
        self.mesh_edge = _rf_spin(self.values.get("mesh_edge_um", 0.25), 1e-6, 1e7, 6, 0.05)
        self.mesh_vertical = _rf_spin(self.values.get("mesh_vertical_um", 0.10), 1e-6, 1e7, 6, 0.05)
        self.mesh_bulk = _rf_spin(self.values.get("mesh_bulk_um", 5.0), 1e-6, 1e7, 6, 0.5)
        mesh_form.addRow("Maximum metal-edge cell (µm)", self.mesh_edge)
        mesh_form.addRow("Maximum vertical metal cell (µm)", self.mesh_vertical)
        mesh_form.addRow("Maximum bulk dielectric cell (µm)", self.mesh_bulk)
        mesh_note = QLabel("The metal-edge override controls CPW gaps and conductor corners; the bulk value controls the surrounding dielectric and air.")
        mesh_note.setWordWrap(True)
        mesh_form.addRow("Meaning", mesh_note)
        layout.addWidget(mesh_group)

        boundary_group = QGroupBox("Boundaries and run length")
        boundary_form = QFormLayout(boundary_group)
        self.boundary_type = QComboBox()
        self.boundary_type.addItems([
            "Metal transverse / PML propagation",
            "PML all sides",
            "PEC transverse / PML propagation",
        ])
        current_boundary = str(self.values.get("boundary_type", "Metal transverse / PML propagation"))
        if self.boundary_type.findText(current_boundary) < 0:
            self.boundary_type.addItem(current_boundary)
        self.boundary_type.setCurrentText(current_boundary)
        self.pml_layers = QSpinBox()
        self.pml_layers.setRange(1, 256)
        self.pml_layers.setValue(int(self.values.get("pml_layers", 28)))
        self.port_clearance = _rf_spin(self.values.get("port_clearance_wavelengths", 0.25), 0.0, 100.0, 6, 0.05)
        self.simulation_time = _rf_spin(self.values.get("simulation_time_ns", 40.0), 1e-6, 1e9, 6, 1.0)
        self.auto_shutoff = _rf_spin(self.values.get("auto_shutoff", 1e-6), 1e-15, 1.0, 12)
        self.backing_ground = QCheckBox("Add a conductor-backed ground plane below the RF stack")
        self.backing_ground.setChecked(bool(self.values.get("backing_ground", False)))
        self.snap_pec = QCheckBox("Snap zero-thickness PEC sheets to a Yee-cell boundary")
        self.snap_pec.setChecked(bool(self.values.get("snap_pec_to_yee_cell_boundary", False)))
        self.short_pulse = QCheckBox("Use the short-pulse RF FDTD optimization")
        self.short_pulse.setChecked(bool(self.values.get("optimize_for_short_pulse", False)))
        boundary_form.addRow("Boundary preset", self.boundary_type)
        boundary_form.addRow("PML layers", self.pml_layers)
        boundary_form.addRow("Port/discontinuity clearance (λ)", self.port_clearance)
        boundary_form.addRow("Maximum simulation time (ns)", self.simulation_time)
        boundary_form.addRow("Auto shutoff", self.auto_shutoff)
        boundary_form.addRow("Backing conductor", self.backing_ground)
        boundary_form.addRow("PEC mesh alignment", self.snap_pec)
        boundary_form.addRow("Pulse optimization", self.short_pulse)
        layout.addWidget(boundary_group)

        execution_group = QGroupBox("Execution and saved files")
        execution_form = QFormLayout(execution_group)
        self.build_threads = QSpinBox()
        self.build_threads.setRange(1, max(256, CPU_COUNT))
        self.build_threads.setValue(int(self.values.get("build_cpu_threads", min(30, CPU_COUNT))))
        self.resource_mode = QComboBox()
        self.resource_mode.addItems(["CPU", "GPU"])
        self.resource_mode.setCurrentText(str(self.values.get("resource_mode", "CPU")))
        self.resource_mode.setEnabled(False)
        self.run_after_build = QCheckBox("Run immediately after constructing the model")
        self.run_after_build.setChecked(bool(self.values.get("run_after_build", True)))
        project_suffix = ".lms" if self.values["rf_workflow"] == "fde" else ".fsp"
        execution_form.addRow("Model-build CPU threads", self.build_threads)
        execution_form.addRow("Solver resource", self.resource_mode)
        execution_form.addRow("Solve", self.run_after_build)
        resource_note = QLabel(
            f"MODE/FDE is CPU-only. Three-dimensional RF FDTD is configured for GPU; plotting and report generation return to CPU. "
            f"The inspection and final {project_suffix} projects are always saved."
        )
        resource_note.setWordWrap(True)
        execution_form.addRow("Resource policy", resource_note)
        layout.addWidget(execution_group)
        layout.addStretch(1)
        self.tabs.addTab(page, "Mesh & boundaries")

    def _build_preview_tab(self) -> None:
        """Add the same interactive pre-export inspection used by photonics."""
        page = QWidget()
        layout = QVBoxLayout(page)
        description = QLabel(
            "Inspect the RF material volumes, exact conductor polygons, manual or endpoint RF planes, "
            "and solver region before exporting. Every visible value is rebuilt from the controls in "
            "this window, so changing the stack, frequency range, mesh, or port spans changes this view."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        workflow_note = QLabel(
            "Straight CPW uses a two-dimensional MODE/FDE cross-section; the preview gives that section "
            "a 1 µm visual extrusion only. CPW tapers, bends, opens, shorts, and segmented electrodes show "
            "the actual three-dimensional RF FDTD region."
        )
        workflow_note.setWordWrap(True)
        workflow_note.setStyleSheet("color:#0f766e; font-weight:600;")
        layout.addWidget(workflow_note)

        show_button = QPushButton("Show me a 3D version of the RF file I have built")
        show_button.setMinimumHeight(48)
        show_button.clicked.connect(self._show_rf_3d_preview)
        layout.addWidget(show_button)
        layout.addStretch(1)
        self.tabs.addTab(page, "RF 3D preview")

    def preview_state(self) -> dict[str, Any]:
        """Return one stable state while the interactive preview is open."""
        if self._active_rf_preview_state is not None:
            return self._active_rf_preview_state
        return build_lumerical_rf_preview_state(
            self.components,
            self.configuration(validate_ports=False),
        )

    def _show_rf_3d_preview(self) -> None:
        """Open an orbitable RF stack/conductor/plane/solver preview."""
        try:
            state = build_lumerical_rf_preview_state(
                self.components,
                self.configuration(validate_ports=False),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not build RF 3D preview", str(exc))
            return

        self._active_rf_preview_state = state
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Pre-export 3D Lumerical RF model")
        preview_dialog.resize(1250, 860)
        preview_dialog.setMinimumSize(1100, 760)
        preview_layout = QVBoxLayout(preview_dialog)

        visibility_row = QHBoxLayout()
        visibility_row.addWidget(QLabel("Show/hide:"))
        preview = ThreeDModelPreview(self)
        preview.show_fiber = False
        for label, attribute in (
            ("RF metal geometry", "show_device"),
            ("RF material stack", "show_stack"),
            ("RF ports & monitors", "show_ports"),
            ("Solver region", "show_fdtd"),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, name=attribute: (
                    setattr(preview, name, bool(checked)),
                    preview.update(),
                )
            )
            visibility_row.addWidget(checkbox)
        visibility_row.addStretch(1)
        reset_view = QPushButton("Reset 3D view")
        reset_view.clicked.connect(
            lambda: (
                setattr(preview, "azimuth_deg", 35.0),
                setattr(preview, "elevation_deg", 20.0),
                setattr(preview, "zoom_factor", 1.0),
                setattr(preview, "pan", QPointF(0.0, 0.0)),
                preview.update(),
            )
        )
        visibility_row.addWidget(reset_view)
        preview_layout.addLayout(visibility_row)

        bounds = state["solver_bounds_um"]
        bounds_label = QLabel(
            "%s · X %.6g to %.6g µm · Y %.6g to %.6g µm · Z %.6g to %.6g µm"
            % (state["solver_label"], *map(float, bounds))
        )
        bounds_label.setWordWrap(True)
        preview_layout.addWidget(bounds_label)

        warnings = list(state.get("warnings", []))
        if warnings:
            warning_label = QLabel("Preview notes: " + "  •  ".join(map(str, warnings)))
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("color:#b45309;")
            preview_layout.addWidget(warning_label)

        layer_grid = QGridLayout()
        layer_grid.addWidget(QLabel("Individual RF stack rows:"), 0, 0, 1, 3)

        def set_layer_visible(row_id: int, visible: bool) -> None:
            if visible:
                preview.hidden_stack_rows.discard(row_id)
            else:
                preview.hidden_stack_rows.add(row_id)
            preview.update()

        for index, (row, _z0, _z1) in enumerate(state["stack_ranges"]):
            row_id = int(row.get("_preview_id", index))
            layer_checkbox = QCheckBox(
                str(
                    row.get(
                        "_preview_label",
                        f"{row.get('name', 'Layer')} — {row.get('material', '')}",
                    )
                )
            )
            layer_checkbox.setChecked(True)
            layer_checkbox.toggled.connect(
                lambda checked, rid=row_id: set_layer_visible(rid, checked)
            )
            layer_grid.addWidget(layer_checkbox, 1 + index // 3, index % 3)
        preview_layout.addLayout(layer_grid)
        preview_layout.addWidget(preview, 1)

        close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_buttons.rejected.connect(preview_dialog.reject)
        preview_layout.addWidget(close_buttons)
        try:
            preview_dialog.exec()
        finally:
            self._active_rf_preview_state = None

    @staticmethod
    def _cell_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _append_material_row(self, material: dict[str, Any]) -> None:
        row = self.material_table.rowCount()
        self.material_table.insertRow(row)
        self.material_table.setItem(row, 0, QTableWidgetItem(str(material.get("name", "Layer"))))
        role = QComboBox()
        role.addItem("Dielectric / background", "dielectric")
        role.addItem("Metal from GDS", "metal")
        role_value = "metal" if str(material.get("role", "dielectric")).lower() == "metal" else "dielectric"
        role.setCurrentIndex(max(0, role.findData(role_value)))
        role.setMinimumSize(190, 38)
        self.material_table.setCellWidget(row, 1, role)
        self.material_table.setItem(row, 2, QTableWidgetItem(f"{float(material.get('thickness_um', 0.0)):g}"))
        anisotropic = bool(material.get("anisotropic", False))
        isotropic_epsilon = "" if anisotropic or role_value == "metal" else f"{float(material.get('relative_permittivity', 1.0)):g}"
        self.material_table.setItem(row, 3, QTableWidgetItem(isotropic_epsilon))
        for column, key in zip((4, 5, 6), ("relative_permittivity_x", "relative_permittivity_y", "relative_permittivity_z")):
            value = material.get(key)
            self.material_table.setItem(row, column, QTableWidgetItem("" if value is None else f"{float(value):g}"))
        self.material_table.setItem(row, 7, QTableWidgetItem(f"{float(material.get('loss_tangent', 0.0)):g}"))
        self.material_table.setItem(row, 8, QTableWidgetItem(f"{float(material.get('conductivity_s_per_m', 0.0)):g}"))
        raw_layers = material.get("gds_layers", [])
        if isinstance(raw_layers, (int, float, str)):
            raw_layers = [raw_layers]
        layers_text = ", ".join(str(int(float(value))) for value in raw_layers) if raw_layers else ""
        self.material_table.setItem(row, 9, QTableWidgetItem(layers_text))
        metal_model = QComboBox()
        metal_model.addItems(["Conductive 3D", "PEC"])
        requested_model = str(material.get("metal_model", "Conductive 3D"))
        if metal_model.findText(requested_model) < 0:
            metal_model.addItem(requested_model)
        metal_model.setCurrentText(requested_model)
        metal_model.setMinimumSize(165, 38)
        self.material_table.setCellWidget(row, 10, metal_model)

    def _set_material_stack(self, stack: list[dict[str, Any]]) -> None:
        self.material_table.setRowCount(0)
        for material in stack:
            self._append_material_row(dict(material))

    def _load_selected_stack_preset(self) -> None:
        preset = self.stack_preset.currentText()
        self._set_material_stack(copy.deepcopy(RF_STACK_PRESETS[preset]))
        self.backing_ground.setChecked(preset == "Official Ansys conductor-backed FR4")

    def _remove_selected_material_rows(self) -> None:
        rows = sorted({index.row() for index in self.material_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.material_table.removeRow(row)

    def _material_stack(self) -> list[dict[str, Any]]:
        stack: list[dict[str, Any]] = []
        for row in range(self.material_table.rowCount()):
            name = self._cell_text(self.material_table, row, 0) or f"Layer {row + 1}"
            role_widget = self.material_table.cellWidget(row, 1)
            role = str(role_widget.currentData()) if isinstance(role_widget, QComboBox) else "dielectric"
            thickness = float(self._cell_text(self.material_table, row, 2) or 0.0)
            material: dict[str, Any] = {
                "name": name,
                "role": role,
                "thickness_um": thickness,
                "loss_tangent": float(self._cell_text(self.material_table, row, 7) or 0.0),
                "conductivity_s_per_m": float(self._cell_text(self.material_table, row, 8) or 0.0),
            }
            if role == "metal":
                model_widget = self.material_table.cellWidget(row, 10)
                material["metal_model"] = model_widget.currentText() if isinstance(model_widget, QComboBox) else "Conductive 3D"
                layer_values = numeric_list(self._cell_text(self.material_table, row, 9))
                if not layer_values:
                    raise ValueError(f"{name}: enter at least one GDS layer for the RF metal.")
                if any(
                    not math.isfinite(value) or value < 0 or abs(value - round(value)) > 1e-12
                    for value in layer_values
                ):
                    raise ValueError(f"{name}: GDS layers must be nonnegative whole numbers.")
                material["gds_layers"] = [int(round(value)) for value in layer_values]
            else:
                axes = [self._cell_text(self.material_table, row, column) for column in (4, 5, 6)]
                if any(axes):
                    if not all(axes):
                        raise ValueError(f"{name}: fill all of εx, εy, and εz, or leave all three blank.")
                    material.update({
                        "anisotropic": True,
                        "relative_permittivity_x": float(axes[0]),
                        "relative_permittivity_y": float(axes[1]),
                        "relative_permittivity_z": float(axes[2]),
                    })
                else:
                    epsilon = self._cell_text(self.material_table, row, 3)
                    if not epsilon:
                        raise ValueError(f"{name}: enter εr or all three anisotropic permittivities.")
                    material["relative_permittivity"] = float(epsilon)
            stack.append(material)
        return stack

    def _scope_uids(self) -> list[int]:
        raw = self.scope_combo.currentData()
        return list(map(int, raw or []))

    def _manual_rf_objects(self) -> list[dict[str, Any]]:
        scope_uids = set(self._scope_uids())
        return [
            component for component in self.components
            if int(component.get("uid", -1)) in scope_uids
            and str(component.get("kind", "")) in RF_SIMULATION_OBJECT_KINDS
        ]

    def _update_scope_report(self, *_args) -> None:
        objects = self._manual_rf_objects()
        ports = sum(str(component.get("kind", "")) == "RF mode port" for component in objects)
        monitors = sum(str(component.get("kind", "")) == "RF power monitor" for component in objects)
        if objects:
            self.scope_report.setText(
                f"{ports} RF mode port(s), {monitors} RF power monitor(s). These manual planes take precedence over endpoint defaults."
            )
        elif self.values["rf_workflow"] == "fde":
            self.scope_report.setText("No RF planes selected; a uniform CPW FDE solve uses its transverse cross-section directly.")
        else:
            self.scope_report.setText("No manual RF planes selected. Keep endpoint fallback enabled or add/select RF plane objects before export.")

    def configuration(self, *, validate_ports: bool = True) -> dict[str, Any]:
        stack = self._material_stack()
        objects = self._manual_rf_objects()
        manual_uids = [int(component["uid"]) for component in objects]
        manual_ports = [component for component in objects if component.get("kind") == "RF mode port"]
        use_endpoints = bool(self.endpoint_fallback.isChecked())
        if (
            validate_ports
            and self.values["rf_workflow"] == "fdtd"
            and not manual_ports
            and not use_endpoints
        ):
            raise ValueError(
                "A 3D RF FDTD run needs a selected RF mode port or the explicitly enabled component-endpoint fallback."
            )
        if manual_uids:
            port_strategy = "manual_or_component_endpoints" if use_endpoints else "manual_only"
        else:
            if use_endpoints:
                port_strategy = "component_endpoints"
            elif not validate_ports and self.values["rf_workflow"] == "fdtd":
                # Previewing unported geometry is useful while the user is
                # still arranging explicit RF planes.  ``manual_only`` is a
                # valid 3D strategy with an empty plane list; the preview
                # builder then presents the missing-source/output warnings.
                port_strategy = "manual_only"
            else:
                port_strategy = "cross_section_only"
        configuration = copy.deepcopy(self.values)
        configuration.update({
            "rf_workflow": self.values["rf_workflow"],
            "scope_uids": self._scope_uids(),
            "primary_component_uid": int(self.target_component["uid"]),
            "primary_component_kind": self.kind,
            "manual_rf_object_uids": manual_uids,
            "rf_port_strategy": port_strategy,
            "use_endpoint_reference_planes": use_endpoints,
            "frequency_start_ghz": self.frequency_start.value(),
            "frequency_stop_ghz": self.frequency_stop.value(),
            "frequency_points": self.frequency_points.value(),
            "target_frequency_ghz": self.target_frequency.value(),
            "material_stack": stack,
            "rf_stack_preset": self.stack_preset.currentText(),
            "input_port_inset_um": self.input_inset.value(),
            "output_port_inset_um": self.output_inset.value(),
            "port_transverse_span_um": self.port_transverse_span.value(),
            "port_vertical_span_um": self.port_vertical_span.value(),
            "multifrequency_mode_injection": self.multifrequency_injection.isChecked(),
            "mesh_edge_um": self.mesh_edge.value(),
            "mesh_vertical_um": self.mesh_vertical.value(),
            "mesh_bulk_um": self.mesh_bulk.value(),
            "boundary_type": self.boundary_type.currentText(),
            "pml_layers": self.pml_layers.value(),
            "port_clearance_wavelengths": self.port_clearance.value(),
            "simulation_time_ns": self.simulation_time.value(),
            "auto_shutoff": self.auto_shutoff.value(),
            "backing_ground": self.backing_ground.isChecked(),
            "snap_pec_to_yee_cell_boundary": self.snap_pec.isChecked(),
            "optimize_for_short_pulse": self.short_pulse.isChecked(),
            "build_cpu_threads": self.build_threads.value(),
            "resource_mode": self.resource_mode.currentText(),
            "run_after_build": self.run_after_build.isChecked(),
            "save_inspection_fsp": True,
            "save_final_fsp": True,
        })
        metal = next((row for row in stack if row.get("role") == "metal"), None)
        if metal is not None:
            configuration["metal_model"] = metal.get("metal_model", "Conductive 3D")
            configuration["metal_conductivity_s_per_m"] = float(metal.get("conductivity_s_per_m", 0.0))
            configuration["metal_thickness_um"] = float(metal.get("thickness_um", 0.0))
        substrate = next(
            (
                row for row in stack
                if row.get("role") == "dielectric"
                and (
                    bool(row.get("anisotropic", False))
                    or float(row.get("relative_permittivity", 1.0)) > 1.000001
                )
            ),
            None,
        )
        if substrate is not None:
            if bool(substrate.get("anisotropic", False)):
                configuration["substrate_relative_permittivity"] = max(
                    float(substrate["relative_permittivity_x"]),
                    float(substrate["relative_permittivity_y"]),
                    float(substrate["relative_permittivity_z"]),
                )
            else:
                configuration["substrate_relative_permittivity"] = float(
                    substrate.get("relative_permittivity", 1.0)
                )
            configuration["substrate_loss_tangent"] = float(
                substrate.get("loss_tangent", 0.0)
            )
            configuration["substrate_thickness_um"] = float(
                substrate.get("thickness_um", 0.0)
            )
        return normalize_rf_configuration(self.kind, configuration)

    def accept(self) -> None:
        try:
            self.configuration()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical RF settings", str(exc))
            return
        super().accept()


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
        # This is an editor: items are added and moved frequently.  Maintaining
        # a BSP index for a dense scene makes each mutation unnecessarily
        # expensive; linear lookup is faster for the typical component count.
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)
        saved_opengl = str(self.settings.value("layout/use_opengl", "true")).strip().lower() in {"1","true","yes","on"}
        self.view = LayoutView(self.scene, use_opengl=saved_opengl)
        self.view.regionFitRequested.connect(self.fit_drawn_region)
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
            "RF test blocks":[],
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
        simulation_names: list[str] = []
        for kind in DEFAULT_COMPONENT_VALUES:
            if kind in LEGACY_PHOTONIC_TEST_BLOCK_KINDS:
                continue
            if kind == "Fiber port":
                continue
            if filter_text and filter_text not in kind.lower():
                continue
            if kind in SIMULATION_COMPONENT_KINDS:
                simulation_names.append(kind)
            elif kind in RF_COMPONENT_KINDS or kind == "MZI + CPW module":
                lower=kind.lower()
                if kind == "RF test block":group="RF test blocks"
                elif "open" in lower or "short" in lower:group="Calibration structures"
                elif kind == "Segmented electrode":group="CPW transmission lines"
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
            elif kind in {"Chip outline", "4-inch wafer outline"}:
                categories["Chip / utility"].append(kind)
            else:
                lower=kind.lower()
                if "test block" in lower:group="Photonic test blocks"
                elif "photonic crystal" in lower:group="Photonic crystals"
                elif "mzi" in lower:group="MZI"
                elif "mmi" in lower:group="MMI"
                elif any(token in lower for token in ("ring", "racetrack", "resonator", "loopback")):group="Resonators"
                elif kind == "GC-SOI" or "grating" in lower or "edge coupler" in lower:group="Grating couplers"
                elif any(token in lower for token in ("straight", "taper", "bend", "feedline", "waveguide")):group="Waveguides & bends"
                else:group="Other photonic"
                photonic_groups[group].append(kind)
        for module_name in sorted(self.custom_modules):
            if not filter_text or filter_text in module_name.lower():
                categories["User modules"].append(module_name)
        if simulation_names:
            simulation_parent = QTreeWidgetItem(["Ports & monitors"])
            simulation_parent.setFlags(simulation_parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.library_tree.addTopLevelItem(simulation_parent)
            for name in sorted(simulation_names):
                child = QTreeWidgetItem([component_display_name(name)])
                child.setData(0, Qt.ItemDataRole.UserRole, ("component", name))
                simulation_parent.addChild(child)
            simulation_parent.setExpanded(True)
        populated_photonic={name:values for name,values in photonic_groups.items() if values}
        if populated_photonic:
            photonic_parent=QTreeWidgetItem(["Photonic"]);photonic_parent.setFlags(photonic_parent.flags()&~Qt.ItemFlag.ItemIsSelectable);self.library_tree.addTopLevelItem(photonic_parent)
            for group,names in populated_photonic.items():
                group_item=QTreeWidgetItem([group]);group_item.setFlags(group_item.flags()&~Qt.ItemFlag.ItemIsSelectable);photonic_parent.addChild(group_item)
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
                child = QTreeWidgetItem([name if category == "User modules" else component_display_name(name)])
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
        self.llm_test_button = QPushButton("Test API")
        self.llm_clear_button = QPushButton("Clear")
        buttons.addWidget(self.llm_send_button)
        buttons.addWidget(self.llm_test_button)
        buttons.addWidget(self.llm_clear_button)
        layout.addLayout(buttons)

        note = QLabel(
            "The API key is used for the request only and is not stored in the project."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.llm_send_button.clicked.connect(self.run_llm_assistant)
        self.llm_test_button.clicked.connect(self.run_llm_api_test)
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
        action("refresh_json_gui", "↻ Refresh GUI from JSON", self.refresh_gui_from_json, "F5", file_toolbar, file_menu, status_tip="Clear derived previews and rebuild the canvas from the current JSON project data without changing export geometry.")
        file_toolbar.addSeparator()
        file_menu.addSeparator()
        action("export_gds", "Export GDS", self.export_gds, None, file_toolbar, file_menu)
        action("export_python", "Export Python", self.export_python, None, None, file_menu)
        action(
            "export_lumerical",
            "Lumerical Run…",
            self.export_lumerical_notebook,
            None,
            None,
            file_menu,
            status_tip="Export the selected optical device or route a CPW device to the dedicated RF MODE/FDTD notebook workflow.",
        )
        action(
            "export_lumerical_rf",
            "Lumerical RF Run…",
            self.export_lumerical_rf_notebook,
            None,
            None,
            file_menu,
            status_tip="Export a straight CPW with MODE/FDE or an RF taper/discontinuity with 3D FDTD S-parameters.",
        )
        action(
            "export_lumerical_sweep",
            "Lumerical Sweep…",
            self.export_lumerical_sweep_notebook,
            None,
            None,
            file_menu,
            status_tip="Sweep selected component parameters in one persistent Lumerical/GPU session with minimal model-build and FSP overhead.",
        )
        action(
            "export_lumerical_multigpu_sweep",
            "Lumerical sweep-multithread…",
            self.export_lumerical_multigpu_sweep_notebook,
            None,
            None,
            file_menu,
            status_tip="Distribute sweep points across independent A100 workers while preserving the existing sequential sweep export.",
        )
        action(
            "export_lumerical_optimization",
            "Lumerical Optimization…",
            self.export_lumerical_optimization_notebook,
            None,
            None,
            file_menu,
            status_tip="Export a 3D GPU shape-adjoint notebook for a grating coupler or symmetric 1x2 MMI.",
        )
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
        action("layout_center", "All GDS center → (0, 0)", self.center_entire_layout_at_origin, None, layout_toolbar, layout_menu, status_tip="Move every component together so the complete GDS bounding-box center is exactly (0, 0).")
        action("push_within_chip", "Push within chip/wafer boundary", self.remove_outside_chip_outline, None, layout_toolbar, layout_menu, status_tip=f"Delete every component whose geometry reaches outside the Chip outline or 4-inch wafer outline, keeping {CHIP_BOUNDARY_MARGIN_UM:g} µm clear of a chip edge and {WAFER_BOUNDARY_MARGIN_UM/1000.0:g} mm clear of a wafer edge.")
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
        action("move_ebeam_block", "Move Entire E-beam Block…", self.position_selected_ebeam_blocks, None, ebeam_toolbar, ebeam_menu, status_tip="Move the selected E-beam field group without moving its covered GDS geometry.")
        action("field_earlier", "Field Earlier", lambda: self.shift_active_field_order(-1), None, None, ebeam_menu)
        action("field_later", "Field Later", lambda: self.shift_active_field_order(1), None, None, ebeam_menu)
        ebeam_toolbar.addSeparator()
        action("play_fields", "▶ Play Fields", self.play_writefields, None, ebeam_toolbar, ebeam_menu)
        action("step_fields", "Step Field", self.step_writefields, None, ebeam_toolbar, ebeam_menu)
        action("stop_fields", "■ Stop", self.stop_writefields, None, ebeam_toolbar, ebeam_menu)

    def open_layout_thread_settings(self) -> None:
        dialog=QDialog(self);dialog.setWindowTitle("Layout Performance / CPU, Cache, and GPU");layout=QVBoxLayout(dialog);form=QFormLayout();threads=QSpinBox();threads.setRange(1,CPU_COUNT);threads.setValue(self.layout_threads);form.addRow(f"CPU threads (1–{CPU_COUNT})",threads);gpu_canvas=QCheckBox("Use GPU-accelerated OpenGL canvas when supported");gpu_canvas.setChecked(self.view.opengl_enabled);form.addRow("Interactive canvas",gpu_canvas);layout.addLayout(form)
        note=QLabel("Repeated preview geometry is cached automatically. OpenGL accelerates painting and dragging; CPU threads and a separate worker process handle geometry and full-resolution GDS export. Disable OpenGL if a graphics driver shows artifacts.");note.setWordWrap(True);layout.addWidget(note);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        self.layout_threads=threads.value();self.settings.setValue("layout/cpu_threads",self.layout_threads);configure_acceleration(self.layout_threads);opengl_enabled=self.view.set_opengl_enabled(gpu_canvas.isChecked());self.settings.setValue("layout/use_opengl",opengl_enabled);self.opengl_status_label.setText(("OpenGL GPU" if opengl_enabled else "CPU canvas")+f" • {self.layout_threads} threads");self.statusBar().showMessage(f"Performance updated: {self.layout_threads} CPU threads, {'OpenGL GPU canvas' if opengl_enabled else 'CPU canvas'}, preview cache enabled.",5000)

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
            for component in components:
                synchronize_rf_taper_points(component)
                if component.get("kind") == "Fiber geometry" and bool(component.get("auto_placed", False)):
                    component.setdefault("params", {})["z reference"] = "top of SiO2 cladding"
                if component.get("kind") == "GC-SOI":
                    params = component.setdefault("params", {})
                    migrate_grating_fiber_offset_parameter(component)
                    params.setdefault("fill_factors", "")
                    params.setdefault("tooth_shape", "curved")
                    if abs(float(params.get("tolerance", 0.005)) - 0.0005) <= 1e-12:
                        params["tolerance"] = 0.005
                    if abs(float(params.get("fdtd_port_offset_from_waveguide_end_um", 2.0)) - 3.0) <= 1e-12:
                        params["fdtd_port_offset_from_waveguide_end_um"] = 2.0
                elif component.get("kind") == "Grating coupler":
                    params = component.setdefault("params", {})
                    migrate_grating_fiber_offset_parameter(component)
                    params.setdefault("fill_factors", "")
                    params.setdefault("tooth_shape", "curved")
                    if abs(float(params.get("waveguide_monitor_span_um", 3.0)) - 2.5) <= 1e-12:
                        params["waveguide_monitor_span_um"] = 3.0
                    if abs(float(params.get("fdtd_port_offset_from_waveguide_end_um", 2.0)) - 3.0) <= 1e-12:
                        params["fdtd_port_offset_from_waveguide_end_um"] = 2.0
            self.components = components
            self.next_uid = int(data.get("next_uid", max([int(c.get("uid", 0)) for c in components] + [0]) + 1))
            for component in self.components:
                if component.get("kind") in {"Grating coupler", "GC-SOI"}:
                    self.synchronize_automatic_simulation_companions(component)
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

    def refresh_gui_from_json(self) -> None:
        """Rebuild every visual item from a clean JSON round-trip and empty caches."""
        selected = [
            int(item.data(10))
            for item in self.scene.selectedItems()
            if item.data(10) is not None
        ]
        # The round-trip deliberately removes any GUI-only Python object state
        # while preserving the exact serializable project used by GDS export.
        self.components = json.loads(json.dumps(self.components))
        for component in self.components:
            synchronize_rf_taper_points(component)
        clear_preview_caches()
        self.rebuild_scene(select_uids=selected)
        self.view.viewport().update()
        self.statusBar().showMessage(
            "GUI refreshed from current JSON data; preview caches were rebuilt.",
            6000,
        )

    # ------------------------------------------------------------------
    # Components / scene
    # ------------------------------------------------------------------
    def make_component(self, kind: str, x: float, y: float) -> dict[str, Any]:
        params = safe_json_copy(DEFAULT_COMPONENT_VALUES[kind])
        if kind in {"FDTD port", "Fiber-axis FDTD port"}:
            order = 1 + sum(
                component.get("kind") in {"FDTD port", "Fiber-axis FDTD port"}
                for component in self.components
            )
            params["name"] = f"opt_{order}"
            params["order"] = order
        elif kind in {"Power monitor", "Mode expansion monitor", "Field profile monitor"}:
            prefix = {
                "Power monitor": "power_monitor",
                "Mode expansion monitor": "mode_expansion",
                "Field profile monitor": "field_profile",
            }[kind]
            count = 1 + sum(component.get("kind") == kind for component in self.components)
            params["name"] = f"{prefix}_{count}"
        elif kind == "RF mode port":
            used_orders = {
                int(component.get("params", {}).get("order", 0))
                for component in self.components
                if component.get("kind") == "RF mode port"
            }
            order = 1
            while order in used_orders:
                order += 1
            params["name"] = f"rf_port_{order}"
            params["order"] = order
        elif kind == "RF power monitor":
            used_names = {
                str(component.get("params", {}).get("name", ""))
                for component in self.components
                if component.get("kind") == "RF power monitor"
            }
            count = 1
            while f"rf_power_{count}" in used_names:
                count += 1
            params["name"] = f"rf_power_{count}"
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
        synchronize_rf_taper_points(component)
        self.next_uid += 1
        return component

    def automatic_simulation_companions(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Create movable Ansys-style ports for common optical components.

        The planes sit at the physical optical endpoints. The export dialog's
        2 µm default domain clearance and the notebook's boundary-extension
        geometry continue each ported waveguide through the corresponding PML.
        """
        kind = str(component.get("kind", ""))
        component_params = component.get("params", {})
        mmi_neff_tolerance = max(
            0.0, float(component_params.get("waveguide_neff_tolerance", 0.3))
        )
        mmi_mode_search_count = max(
            1, int(component_params.get("waveguide_mode_search_count", 20))
        )
        requested_ports = {
            "Straight": ["left", "right"],
            "Taper": ["left", "right"],
            "S-bend": ["left", "right"],
            "Euler bend": ["start", "end"],
            "1x2 MMI": ["left_external", "upper_right", "lower_right"],
            "Cascaded MMI": ["input", "output"],
            "MMI + Reference": ["left_external", "upper_right", "lower_right", "reference_left_external", "reference_right"],
            "Grating coupler": ["waveguide_point"],
            "GC-SOI": ["waveguide_point"],
        }.get(kind)
        if not requested_ports:
            return []

        global_ports = component_global_ports(component)
        companions: list[dict[str, Any]] = []
        next_order = 1 + sum(
            existing.get("kind") in {"FDTD port", "Fiber-axis FDTD port"}
            for existing in self.components
        )
        automatic_order_base = next_order
        for port_name in requested_ports:
            port = global_ports.get(port_name)
            if not port or str(port.get("domain", "optical")) != "optical":
                continue
            outward = float(port["outward_orientation_deg"]) % 360.0
            center = tuple(map(float, port["center"]))
            port_offset_um = 0.0
            if kind in {"Grating coupler", "GC-SOI"} and port_name == "waveguide_point":
                port_offset_um = max(
                    0.0,
                    float(component.get("params", {}).get(
                        "fdtd_port_offset_from_waveguide_end_um", 2.0
                    )),
                )
                inward_angle = math.radians(outward + 180.0)
                center = (
                    center[0] + port_offset_um * math.cos(inward_angle),
                    center[1] + port_offset_um * math.sin(inward_angle),
                )
            elif kind == "1x2 MMI":
                port_offset_um = max(
                    0.0,
                    float(component.get("params", {}).get("fdtd_port_clearance_um", 2.0)),
                )
                inward_angle = math.radians(outward + 180.0)
                center = (
                    center[0] + port_offset_um * math.cos(inward_angle),
                    center[1] + port_offset_um * math.sin(inward_angle),
                )
            if kind in {"Grating coupler", "GC-SOI"} and port_name == "waveguide_point":
                grating_params = component.get("params", {})
                monitor_span_um = automatic_waveguide_port_span_um(component, port_name)
                before_mode_um = max(
                    0.0,
                    float(grating_params.get("waveguide_total_power_before_mode_um", 1.0)),
                )
                inward_angle = math.radians(outward + 180.0)
                total_center = (
                    center[0] + before_mode_um * math.cos(inward_angle),
                    center[1] + before_mode_um * math.sin(inward_angle),
                )
                power_name = f"uid_{int(component['uid'])}_waveguide_total_power"
                total_power = self.make_component(
                    "Power monitor", float(total_center[0]), float(total_center[1])
                )
                total_power["orientation_deg"] = outward
                total_power["auto_placed"] = True
                total_power["simulation_parent_uid"] = int(component["uid"])
                total_power["simulation_parent_port"] = port_name
                total_power["grating_monitor_role"] = "waveguide_total_power"
                total_power["waveguide_end_offset_um"] = port_offset_um + before_mode_um
                total_power["params"].update(
                    {
                        "name": power_name,
                        "monitor geometry": "surface",
                        "plane normal": "X",
                        "distance_um": 0.0,
                        "x span": 0.0,
                        "y span": monitor_span_um,
                        "z span": 2.25,
                    }
                )
                companions.append(total_power)

            placed = self.make_component("FDTD port", float(center[0]), float(center[1]))
            placed["orientation_deg"] = outward
            placed["auto_placed"] = True
            placed["simulation_parent_uid"] = int(component["uid"])
            placed["simulation_parent_port"] = port_name
            placed["waveguide_end_offset_um"] = port_offset_um
            placed["params"].update(
                {
                    "name": f"uid_{int(component['uid'])}_{port_name}",
                    "order": next_order,
                    "pos": "Right",
                    "plane normal": "X",
                    "distance_um": 0.0,
                    "span_um": automatic_waveguide_port_span_um(component, port_name),
                }
            )
            if kind == "1x2 MMI":
                # The three access waveguides have the same cross-section, so
                # their ports must select the same Ey-dominant fundamental TE
                # family around one platform effective-index target.
                placed["params"].update(
                    {
                        "mode": "fundamental TE mode",
                        "polarization": "local TE",
                        "target neff": 0.0,
                        "target neff strategy": "automatic material-index midpoint",
                        "neff tolerance": mmi_neff_tolerance,
                        "mode search count": mmi_mode_search_count,
                    }
                )
            elif kind in {"Grating coupler", "GC-SOI"} and port_name == "waveguide_point":
                placed["params"].update(
                    {
                        "mode": "fundamental TE mode",
                        "polarization": "local TE",
                        "target neff": 0.0,
                        "target neff strategy": "automatic material-index midpoint",
                        "neff tolerance": max(
                            0.0, float(component_params.get("waveguide_neff_tolerance", 0.3))
                        ),
                        "mode search count": max(
                            1, int(component_params.get("waveguide_mode_search_count", 20))
                        ),
                    }
                )
            next_order += 1
            companions.append(placed)

        if kind == "1x2 MMI":
            input_port = global_ports.get("left_external")
            taper_start = global_ports.get("left_straight_end")
            if input_port and taper_start:
                outward = float(input_port["outward_orientation_deg"]) % 360.0
                input_straight_um = max(0.1, float(component.get("params", {}).get("input_length", 6.0)))
                before_taper_um = min(
                    max(0.1, float(component.get("params", {}).get("input_reference_before_taper_um", 2.0))),
                    max(0.1, input_straight_um - 0.1),
                )
                toward_input_angle = math.radians(outward)
                taper_x, taper_y = map(float, taper_start["center"])
                reference_x = taper_x + before_taper_um * math.cos(toward_input_angle)
                reference_y = taper_y + before_taper_um * math.sin(toward_input_angle)
                reference = self.make_component("Power monitor", reference_x, reference_y)
                reference["orientation_deg"] = outward
                reference["auto_placed"] = True
                reference["simulation_parent_uid"] = int(component["uid"])
                reference["simulation_parent_port"] = "left_external"
                reference["mmi_input_reference_before_taper_um"] = before_taper_um
                reference["params"].update(
                    {
                        "name": f"uid_{int(component['uid'])}_input_reference",
                        "monitor geometry": "surface",
                        "plane normal": "X",
                        "distance_um": 0.0,
                        "x span": 0.0,
                        "y span": automatic_waveguide_port_span_um(
                            component, "left_external"
                        ),
                        "z span": 2.25,
                    }
                )
                companions.append(reference)

            # A Z-normal longitudinal profile through the device center shows
            # how the launched fundamental mode expands and interferes along
            # the input taper, MMI body, and output tapers.
            total_length_um = mmi_total_length(component.get("params", {}))
            local_center = np.asarray([0.5 * total_length_um, 0.0], dtype=float)
            device_angle = float(component.get("orientation_deg", 0.0))
            world_center = np.asarray(
                [float(component.get("x", 0.0)), float(component.get("y", 0.0))],
                dtype=float,
            ) + np.asarray(
                [
                    local_center[0] * math.cos(math.radians(device_angle)),
                    local_center[0] * math.sin(math.radians(device_angle)),
                ]
            )
            profile = self.make_component("Field profile monitor", float(world_center[0]), float(world_center[1]))
            profile["orientation_deg"] = device_angle
            profile["auto_placed"] = True
            profile["simulation_parent_uid"] = int(component["uid"])
            profile["simulation_parent_port"] = "mmi_longitudinal_field"
            profile["params"].update(
                {
                    "name": f"uid_{int(component['uid'])}_mmi_field",
                    "monitor geometry": "surface",
                    "plane normal": "Z",
                    "z reference": "device center",
                    "distance_um": 0.0,
                    "x span": total_length_um,
                    "y span": max(float(component.get("params", {}).get("mmi_width", 6.0)) + 2.0, float(component.get("params", {}).get("port_sep", 3.25)) + 4.0),
                    "z span": 0.0,
                }
            )
            companions.append(profile)

        if kind in {"Grating coupler", "GC-SOI"}:
            # Place the fiber on the grating side: straight lead + complete
            # flare anchor + the user-controlled local-X fiber offset.
            grating_params = component.get("params", {})
            migrate_grating_fiber_offset_parameter(component)
            local_x, local_y = grating_fiber_center_local_um(component)
            fiber_offset_um = float(component["params"]["fiber_offset"])
            fiber_angle_deg = grating_angle_theta_deg(component)
            angle = math.radians(float(component.get("orientation_deg", 0.0)))
            fiber_x = (
                float(component.get("x", 0.0))
                + local_x * math.cos(angle)
                - local_y * math.sin(angle)
            )
            fiber_y = (
                float(component.get("y", 0.0))
                + local_x * math.sin(angle)
                + local_y * math.cos(angle)
            )
            fiber = self.make_component("Fiber geometry", fiber_x, fiber_y)
            fiber["orientation_deg"] = float(component.get("orientation_deg", 0.0))
            fiber["auto_placed"] = True
            fiber["simulation_parent_uid"] = int(component["uid"])
            fiber["fiber_offset_um"] = fiber_offset_um
            fiber["params"].update(
                {
                    "name": f"uid_{int(component['uid'])}_fiber",
                    "angle theta": fiber_angle_deg,
                    "angle phi": 0.0,
                    "distance_um": 0.0,
                    "z reference": (
                        "center of SiO2 cladding" if kind == "GC-SOI"
                        else "top of SiO2 cladding"
                    ),
                    "core diameter_um": float(grating_params.get("fiber_core_diameter_um", 9.0)),
                    "core index": float(grating_params.get("fiber_core_index", 1.44427)),
                    "cladding diameter_um": float(grating_params.get("fiber_cladding_diameter_um", 50.0)),
                    "cladding index": float(grating_params.get("fiber_cladding_index", 1.43482)),
                    "fiber length_um": float(grating_params.get("fiber_length_um", 20.0)),
                }
            )
            companions.append(fiber)

            port_local_x = local_x
            port_distance_um = 0.0
            if kind == "GC-SOI":
                tox_offset_um = float(grating_params.get("fiber_tox_offset_um", 0.65))
                # Official model: TOX center + 0.65 cos(theta), with the port
                # measured here from the 0.70 um TOX top. The fiber and port
                # intentionally retain the same top-view center.
                port_distance_um = tox_offset_um * math.cos(math.radians(fiber_angle_deg)) - 0.35
            fiber_port_x = float(component.get("x", 0.0)) + port_local_x * math.cos(angle)
            fiber_port_y = float(component.get("y", 0.0)) + port_local_x * math.sin(angle)
            fiber_port = self.make_component("Fiber-axis FDTD port", fiber_port_x, fiber_port_y)
            fiber_port["orientation_deg"] = float(component.get("orientation_deg", 0.0))
            fiber_port["auto_placed"] = True
            fiber_port["simulation_parent_uid"] = int(component["uid"])
            fiber_port["fiber_offset_um"] = fiber_offset_um
            fiber_port["params"].update(
                {
                    "name": f"uid_{int(component['uid'])}_fiber_axis",
                    # Match the official Ansys grating model: fiber is the
                    # first/source port and waveguide is the second/receiver.
                    "order": automatic_order_base,
                    "dir": "Backward",
                    "fiber plane role": "source",
                    "angle theta": fiber_angle_deg,
                    "angle phi": 0.0,
                    "align to fiber axis": True,
                    "rotation offset_um": 4.0 * float(grating_params.get("fiber_core_diameter_um", 9.0)) * math.tan(math.radians(fiber_angle_deg)),
                    "distance_um": port_distance_um,
                    "z reference": "top of SiO2 cladding" if kind == "GC-SOI" else "top of stack",
                }
            )
            for companion in companions:
                if companion.get("kind") == "FDTD port":
                    companion.setdefault("params", {}).update(
                        {
                            "order": automatic_order_base + 1,
                            "mode": "fundamental TE mode",
                            "span_um": automatic_waveguide_port_span_um(
                                component, "waveguide_point"
                            ),
                        }
                    )
            companions.append(fiber_port)

            # A non-modal Z-normal power monitor measures the actual incident
            # flux below the tilted source.  Lumerical DFT monitors cannot be
            # rotated; the monitor is centered on the tilted fiber-axis
            # intersection and made wide enough to capture its full projected
            # beam.  Its signed T is negative for the downward (-Z) source.
            monitor_below_source_um = max(
                0.001,
                float(grating_params.get("fiber_power_monitor_below_source_um", 0.1)),
            )
            measurement_phi_rad = angle + math.radians(
                float(fiber_port["params"].get("angle phi", 0.0))
            )
            measurement_lateral_um = monitor_below_source_um * math.tan(
                math.radians(fiber_angle_deg)
            )
            projected_monitor_span_um = float(
                fiber_port["params"].get("span_um", 20.0)
            ) / max(math.cos(math.radians(fiber_angle_deg)), 1e-3)
            fiber_power_x = fiber_port_x - measurement_lateral_um * math.cos(measurement_phi_rad)
            fiber_power_y = fiber_port_y - measurement_lateral_um * math.sin(measurement_phi_rad)
            fiber_power = self.make_component("Power monitor", fiber_power_x, fiber_power_y)
            fiber_power["orientation_deg"] = float(component.get("orientation_deg", 0.0))
            fiber_power["auto_placed"] = True
            fiber_power["simulation_parent_uid"] = int(component["uid"])
            fiber_power["simulation_parent_port"] = "fiber_input_power"
            fiber_power["fiber_offset_um"] = fiber_offset_um
            fiber_power["params"].update(
                {
                    "name": f"uid_{int(component['uid'])}_fiber_input_power",
                    "fiber plane role": "input power measurement",
                    "monitor geometry": "surface",
                    "plane normal": "Z",
                    "z reference": fiber_port["params"]["z reference"],
                    "distance_um": port_distance_um - monitor_below_source_um,
                    "x span": projected_monitor_span_um,
                    "y span": projected_monitor_span_um,
                    "z span": 0.0,
                    "angle theta": fiber_angle_deg,
                    "angle phi": float(fiber_port["params"].get("angle phi", 0.0)),
                    "align to fiber axis": True,
                    "expected propagation sign": -1.0,
                }
            )
            companions.append(fiber_power)
        return companions

    def synchronize_automatic_simulation_companions(self, component: dict[str, Any]) -> bool:
        """Keep automatically placed ports/fiber aligned after parent edits."""
        parent_uid = int(component.get("uid", 0))
        companions = [
            candidate for candidate in self.components
            if int(candidate.get("simulation_parent_uid", -1)) == parent_uid
            and bool(candidate.get("auto_placed", False))
        ]
        if not companions:
            return False
        if str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"}:
            grating_params = component.get("params", {})
            monitor_span_um = automatic_waveguide_port_span_um(component, "waveguide_point")
            waveguide_power_name = f"uid_{parent_uid}_waveguide_total_power"
            # Restore the passive waveguide receiver in projects created
            # during the temporary mode-expansion-only implementation.
            for candidate in companions:
                if (
                    candidate.get("kind") == "Mode expansion monitor"
                    and candidate.get("simulation_parent_port") == "waveguide_point"
                ):
                    old = candidate.get("params", {})
                    converted = safe_json_copy(DEFAULT_COMPONENT_VALUES["FDTD port"])
                    converted.update(
                        {
                            "name": f"uid_{parent_uid}_waveguide_point",
                            "plane normal": "X",
                            "distance_um": 0.0,
                            "span_um": monitor_span_um,
                            "z_span_um": float(old.get("z span", 2.25)),
                            "mode": "fundamental TE mode",
                            "polarization": "local TE",
                            "target neff": 0.0,
                            "target neff strategy": "automatic material-index midpoint",
                            "neff tolerance": max(0.0, float(grating_params.get("waveguide_neff_tolerance", 0.3))),
                            "mode search count": max(1, int(grating_params.get("waveguide_mode_search_count", 20))),
                            "order": 2,
                        }
                    )
                    candidate["kind"] = "FDTD port"
                    candidate.pop("grating_monitor_role", None)
                    candidate["params"] = converted
            has_waveguide_receiver = any(
                candidate.get("kind") == "FDTD port"
                and candidate.get("simulation_parent_port") == "waveguide_point"
                for candidate in companions
            )
            if not has_waveguide_receiver:
                global_waveguide = component_global_ports(component).get("waveguide_point")
                if global_waveguide is not None:
                    outward = float(global_waveguide["outward_orientation_deg"]) % 360.0
                    center_x, center_y = map(float, global_waveguide["center"])
                    clearance_um = max(
                        0.0,
                        float(grating_params.get("fdtd_port_offset_from_waveguide_end_um", 2.0)),
                    )
                    inward = math.radians(outward + 180.0)
                    receiver = self.make_component(
                        "FDTD port",
                        center_x + clearance_um * math.cos(inward),
                        center_y + clearance_um * math.sin(inward),
                    )
                    receiver["orientation_deg"] = outward
                    receiver["auto_placed"] = True
                    receiver["simulation_parent_uid"] = parent_uid
                    receiver["simulation_parent_port"] = "waveguide_point"
                    receiver["params"].update(
                        {
                            "name": f"uid_{parent_uid}_waveguide_point",
                            "order": 2,
                            "plane normal": "X",
                            "distance_um": 0.0,
                            "span_um": monitor_span_um,
                            "z_span_um": 2.25,
                            "mode": "fundamental TE mode",
                            "polarization": "local TE",
                            "target neff": 0.0,
                            "target neff strategy": "automatic material-index midpoint",
                            "neff tolerance": max(0.0, float(grating_params.get("waveguide_neff_tolerance", 0.3))),
                            "mode search count": max(1, int(grating_params.get("waveguide_mode_search_count", 20))),
                        }
                    )
                    self.components.append(receiver)
                    companions.append(receiver)
            has_waveguide_power = any(
                candidate.get("kind") == "Power monitor"
                and candidate.get("grating_monitor_role") == "waveguide_total_power"
                for candidate in companions
            )
            if not has_waveguide_power:
                receiver = next(
                    (
                        candidate for candidate in companions
                        if candidate.get("kind") == "FDTD port"
                        and candidate.get("simulation_parent_port") == "waveguide_point"
                    ),
                    None,
                )
                if receiver is not None:
                    total_power = self.make_component(
                        "Power monitor",
                        float(receiver.get("x", 0.0)),
                        float(receiver.get("y", 0.0)),
                    )
                    total_power["orientation_deg"] = float(receiver.get("orientation_deg", 0.0))
                    total_power["auto_placed"] = True
                    total_power["simulation_parent_uid"] = parent_uid
                    total_power["simulation_parent_port"] = "waveguide_point"
                    total_power["grating_monitor_role"] = "waveguide_total_power"
                    total_power["params"].update(
                        {
                            "name": waveguide_power_name,
                            "monitor geometry": "surface",
                            "plane normal": "X",
                            "distance_um": 0.0,
                            "x span": 0.0,
                            "y span": monitor_span_um,
                            "z span": 2.25,
                        }
                    )
                    self.components.append(total_power)
                    companions.append(total_power)
            # Demote any legacy passive fiber port back to the intended
            # non-modal input-power monitor.
            fiber_source = next(
                (
                    candidate for candidate in companions
                    if candidate.get("kind") == "Fiber-axis FDTD port"
                    and candidate.get("simulation_parent_port") != "fiber_input_power"
                ),
                None,
            )
            source_params = fiber_source.get("params", {}) if fiber_source is not None else {}
            for candidate in companions:
                if (
                    candidate.get("kind") == "Fiber-axis FDTD port"
                    and candidate.get("simulation_parent_port") == "fiber_input_power"
                ):
                    old = candidate.get("params", {})
                    converted = safe_json_copy(DEFAULT_COMPONENT_VALUES["Power monitor"])
                    span_um = max(
                        float(old.get("span_um", 0.0)),
                        float(old.get("x span", 0.0)),
                        float(old.get("y span", 0.0)),
                        float(source_params.get("span_um", 20.0)),
                    )
                    converted.update(
                        {
                            "name": str(old.get("name", f"uid_{parent_uid}_fiber_input_power")),
                            "fiber plane role": "input power measurement",
                            "monitor geometry": "surface",
                            "plane normal": "Z",
                            "z reference": str(old.get("z reference", "top of stack")),
                            "distance_um": float(old.get("distance_um", 0.0)),
                            "x span": span_um,
                            "y span": span_um,
                            "z span": 0.0,
                            "angle theta": grating_angle_theta_deg(component),
                            "angle phi": float(source_params.get("angle phi", old.get("angle phi", 0.0))),
                            "align to fiber axis": True,
                            "expected propagation sign": -1.0,
                        }
                    )
                    candidate["kind"] = "Power monitor"
                    candidate["params"] = converted
            has_input_power = any(
                candidate.get("kind") == "Power monitor"
                and candidate.get("simulation_parent_port") == "fiber_input_power"
                for candidate in companions
            )
            if fiber_source is not None and not has_input_power:
                params = component.get("params", {})
                below_source_um = max(
                    0.001,
                    float(params.get("fiber_power_monitor_below_source_um", 0.1)),
                )
                source_params = fiber_source.get("params", {})
                source_theta_deg = grating_angle_theta_deg(component)
                source_phi_rad = math.radians(
                    float(fiber_source.get("orientation_deg", 0.0))
                    + float(source_params.get("angle phi", 0.0))
                )
                lateral_um = below_source_um * math.tan(math.radians(source_theta_deg))
                fiber_power = self.make_component(
                    "Power monitor",
                    float(fiber_source.get("x", 0.0)) - lateral_um * math.cos(source_phi_rad),
                    float(fiber_source.get("y", 0.0)) - lateral_um * math.sin(source_phi_rad),
                )
                fiber_power["orientation_deg"] = float(fiber_source.get("orientation_deg", 0.0))
                fiber_power["auto_placed"] = True
                fiber_power["simulation_parent_uid"] = parent_uid
                fiber_power["simulation_parent_port"] = "fiber_input_power"
                fiber_power["params"].update(
                    {
                        "name": f"uid_{parent_uid}_fiber_input_power",
                        "fiber plane role": "input power measurement",
                        "monitor geometry": "surface",
                        "plane normal": "Z",
                        "z reference": str(source_params.get("z reference", "top of stack")),
                        "distance_um": float(source_params.get("distance_um", 0.0)) - below_source_um,
                        "x span": float(source_params.get("span_um", 20.0)),
                        "y span": float(source_params.get("span_um", 20.0)),
                        "z span": 0.0,
                        "angle theta": source_theta_deg,
                        "angle phi": float(source_params.get("angle phi", 0.0)),
                        "align to fiber axis": True,
                        "expected propagation sign": -1.0,
                    }
                )
                self.components.append(fiber_power)
                companions.append(fiber_power)
        global_ports = component_global_ports(component)
        for companion in companions:
            parent_port_name = companion.get("simulation_parent_port")
            if parent_port_name:
                port = global_ports.get(str(parent_port_name))
                if port:
                    outward = float(port["outward_orientation_deg"]) % 360.0
                    center_x, center_y = map(float, port["center"])
                    port_offset_um = 0.0
                    if companion.get("kind") == "FDTD port":
                        companion.setdefault("params", {})["span_um"] = automatic_waveguide_port_span_um(
                            component, str(parent_port_name)
                        )
                    if str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"} and str(parent_port_name) == "waveguide_point":
                        port_offset_um = max(
                            0.0,
                            float(component.get("params", {}).get(
                                "fdtd_port_offset_from_waveguide_end_um", 2.0
                            )),
                        )
                        inward_angle = math.radians(outward + 180.0)
                        center_x += port_offset_um * math.cos(inward_angle)
                        center_y += port_offset_um * math.sin(inward_angle)
                        role = str(companion.get("grating_monitor_role", ""))
                        monitor_span_um = automatic_waveguide_port_span_um(
                            component, str(parent_port_name)
                        )
                        if role == "waveguide_total_power":
                            before_mode_um = max(
                                0.0,
                                float(component.get("params", {}).get("waveguide_total_power_before_mode_um", 1.0)),
                            )
                            center_x += before_mode_um * math.cos(inward_angle)
                            center_y += before_mode_um * math.sin(inward_angle)
                            port_offset_um += before_mode_um
                            companion.setdefault("params", {}).update(
                                {
                                    "x span": 0.0,
                                    "y span": monitor_span_um,
                                    "z span": 2.25,
                                }
                            )
                    elif (
                        str(component.get("kind", "")) == "1x2 MMI"
                        and companion.get("kind") == "FDTD port"
                    ):
                        companion.setdefault("params", {}).update(
                            {
                                "mode": "fundamental TE mode",
                                "polarization": "local TE",
                                "target neff": 0.0,
                                "target neff strategy": "automatic material-index midpoint",
                                "neff tolerance": max(
                                    0.0,
                                    float(
                                        component.get("params", {}).get(
                                            "waveguide_neff_tolerance", 0.3
                                        )
                                    ),
                                ),
                                "mode search count": max(
                                    1,
                                    int(
                                        component.get("params", {}).get(
                                            "waveguide_mode_search_count", 20
                                        )
                                    ),
                                ),
                            }
                        )
                        port_offset_um = max(
                            0.0,
                            float(component.get("params", {}).get("fdtd_port_clearance_um", 2.0)),
                        )
                        inward_angle = math.radians(outward + 180.0)
                        center_x += port_offset_um * math.cos(inward_angle)
                        center_y += port_offset_um * math.sin(inward_angle)
                    if (
                        str(component.get("kind", "")) == "1x2 MMI"
                        and str(parent_port_name) == "left_external"
                        and companion.get("kind") == "Power monitor"
                    ):
                        companion.setdefault("params", {}).update(
                            {
                                "x span": 0.0,
                                "y span": automatic_waveguide_port_span_um(
                                    component, "left_external"
                                ),
                                "z span": 2.25,
                            }
                        )
                        taper_start = global_ports.get("left_straight_end")
                        input_straight_um = max(
                            0.1, float(component.get("params", {}).get("input_length", 6.0))
                        )
                        before_taper_um = min(
                            max(
                                0.1,
                                float(component.get("params", {}).get("input_reference_before_taper_um", 2.0)),
                            ),
                            max(0.1, input_straight_um - 0.1),
                        )
                        if taper_start:
                            center_x, center_y = map(float, taper_start["center"])
                        toward_input_angle = math.radians(outward)
                        center_x += before_taper_um * math.cos(toward_input_angle)
                        center_y += before_taper_um * math.sin(toward_input_angle)
                        port_offset_um = before_taper_um
                    companion["x"], companion["y"] = center_x, center_y
                    companion["orientation_deg"] = outward
                    companion["waveguide_end_offset_um"] = port_offset_um
            if str(component.get("kind", "")) == "1x2 MMI" and companion.get("simulation_parent_port") == "mmi_longitudinal_field":
                total_length_um = mmi_total_length(component.get("params", {}))
                angle_rad = math.radians(float(component.get("orientation_deg", 0.0)))
                companion["x"] = float(component.get("x", 0.0)) + 0.5 * total_length_um * math.cos(angle_rad)
                companion["y"] = float(component.get("y", 0.0)) + 0.5 * total_length_um * math.sin(angle_rad)
                companion["orientation_deg"] = float(component.get("orientation_deg", 0.0))
                companion.setdefault("params", {}).update(
                    {
                        "x span": total_length_um,
                        "y span": max(float(component.get("params", {}).get("mmi_width", 6.0)) + 2.0, float(component.get("params", {}).get("port_sep", 3.25)) + 4.0),
                        "z reference": "device center",
                    }
                )
            if (
                str(component.get("kind", "")) in {"Grating coupler", "GC-SOI"}
                and (
                    companion.get("kind") in {"Fiber geometry", "Fiber-axis FDTD port"}
                    or (
                        companion.get("kind") == "Power monitor"
                        and companion.get("simulation_parent_port") == "fiber_input_power"
                    )
                )
            ):
                migrate_grating_fiber_offset_parameter(component)
                params = component.get("params", {})
                is_soi = str(component.get("kind", "")) == "GC-SOI"
                local_x, local_y = grating_fiber_center_local_um(component)
                offset_um = float(params["fiber_offset"])
                theta_deg = grating_angle_theta_deg(component)
                angle = math.radians(float(component.get("orientation_deg", 0.0)))
                companion["x"] = (
                    float(component.get("x", 0.0))
                    + local_x * math.cos(angle)
                    - local_y * math.sin(angle)
                )
                companion["y"] = (
                    float(component.get("y", 0.0))
                    + local_x * math.sin(angle)
                    + local_y * math.cos(angle)
                )
                companion["orientation_deg"] = float(component.get("orientation_deg", 0.0))
                companion.pop("grating_taper_end_offset_um", None)
                companion["fiber_offset_um"] = offset_um
                companion.setdefault("params", {})["angle theta"] = theta_deg
                source_distance_um = 0.0
                source_z_reference = "top of stack"
                if is_soi:
                    tox_offset_um = float(params.get("fiber_tox_offset_um", 0.65))
                    source_distance_um = tox_offset_um * math.cos(math.radians(theta_deg)) - 0.35
                    source_z_reference = "top of SiO2 cladding"
                if companion.get("kind") == "Fiber geometry":
                    # Migrate automatic fibers created before the SiO2-specific
                    # vertical reference was introduced. Manual fibers retain
                    # their independently selected reference.
                    companion.setdefault("params", {})["z reference"] = (
                        "center of SiO2 cladding" if is_soi
                        else "top of SiO2 cladding"
                    )
                    companion["params"]["distance_um"] = 0.0
                    if is_soi:
                        companion["params"].update(
                            {
                                "core diameter_um": float(params.get("fiber_core_diameter_um", 9.0)),
                                "core index": float(params.get("fiber_core_index", 1.44427)),
                                "cladding diameter_um": float(params.get("fiber_cladding_diameter_um", 50.0)),
                                "cladding index": float(params.get("fiber_cladding_index", 1.43482)),
                                "fiber length_um": float(params.get("fiber_length_um", 20.0)),
                            }
                        )
                elif companion.get("kind") == "Fiber-axis FDTD port":
                    rotation_offset_um = 4.0 * float(
                        params.get("fiber_core_diameter_um", 9.0)
                    ) * math.tan(math.radians(theta_deg))
                    companion["params"].update(
                        {
                            "fiber plane role": "source",
                            "port geometry": "surface",
                            "plane normal": "Z",
                            "z reference": source_z_reference,
                            "distance_um": source_distance_um,
                            "span_um": float(companion["params"].get("span_um", 20.0)),
                            "z_span_um": 0.0,
                            "angle theta": theta_deg,
                            "angle phi": float(companion["params"].get("angle phi", 0.0)),
                            "align to fiber axis": True,
                            "rotation offset_um": rotation_offset_um,
                        }
                    )
                elif companion.get("kind") == "Power monitor":
                    below_source_um = max(
                        0.001,
                        float(params.get("fiber_power_monitor_below_source_um", 0.1)),
                    )
                    source_port = next(
                        (
                            item for item in companions
                            if item.get("kind") == "Fiber-axis FDTD port"
                            and item.get("simulation_parent_port") != "fiber_input_power"
                        ),
                        None,
                    )
                    source_port_params = source_port.get("params", {}) if source_port is not None else {}
                    span_um = float(source_port_params.get("span_um", 20.0)) / max(
                        math.cos(math.radians(theta_deg)), 1e-3
                    )
                    companion["params"].update(
                        {
                            "fiber plane role": "input power measurement",
                            "monitor geometry": "surface",
                            "plane normal": "Z",
                            "z reference": source_z_reference,
                            "distance_um": source_distance_um - below_source_um,
                            "x span": span_um,
                            "y span": span_um,
                            "z span": 0.0,
                            "angle theta": theta_deg,
                            "angle phi": float(source_port_params.get("angle phi", 0.0)),
                            "align to fiber axis": True,
                            "expected propagation sign": -1.0,
                        }
                    )
                    phi_rad = angle + math.radians(
                        float(source_port_params.get("angle phi", 0.0))
                    )
                    lateral_um = below_source_um * math.tan(math.radians(theta_deg))
                    source_x = float(source_port.get("x", companion["x"]) if source_port else companion["x"])
                    source_y = float(source_port.get("y", companion["y"]) if source_port else companion["y"])
                    companion["x"] = source_x - lateral_um * math.cos(phi_rad)
                    companion["y"] = source_y - lateral_um * math.sin(phi_rad)
        return True

    def configure_test_block_sweeps(self, component: dict[str, Any]) -> bool:
        kind=str(component.get("kind",""));p=component["params"];options={
            "Double-ring test block":[("coupling_gap","Ring coupling gap",.5,1.,.1),("ring_radius","Ring radius",20.,200.,30.)],
            "Grating test block":[("pitch","Grating pitch",.73,.77,.005),("fill_factor","Fill factor",.47,.67,.05)],
            "Grating angle-taper test block":[("alpha_t","Aperture angle",22.,28.,1.),("taper_L","Taper length",20.,24.,1.)],
            "MMI + Reference test block":[("mmi_length","MMI length",26.,33.,1.),("taper_width","Waveguide taper width",2.5,3.1,.1)],
            "MMI split-combine test block":[("taper_length","MMI taper length",8.,12.,1.),("taper_width","Waveguide taper width",2.5,3.1,.1)],
            "Vertical-GC MZI test block":[("mmi_length","MMI length",25.,33.,2.)],
            "Vertical-GC MZI + CPW test block":[("mmi_length","MMI length",25.,33.,2.)],
            "Vertical-GC MZI + segmented electrode test block":[("mmi_length","MMI length",25.,33.,2.)],
            "Straight-GC MZI + segmented RF bends test block":[("mmi_length","MMI length",25.,33.,2.)],
            "Straight-GC MZI + CPW RF bends test block":[("mmi_length","MMI length",25.,33.,2.)],
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
            if key=="mmi_length" and kind in {"Vertical-GC MZI test block","Vertical-GC MZI + CPW test block","Vertical-GC MZI + segmented electrode test block","Straight-GC MZI + segmented RF bends test block","Straight-GC MZI + CPW RF bends test block"}:p["mzi_count"]=len(values)
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

    def test_block_scan_dialog(self, family: str, title: str, base: dict[str, Any], keys: list[str], selected: list[str], saved_ranges: dict[str, Any], edge_spacing: float, explicit_defaults: dict[str, str] | None = None):
        """Scan ranges plus fixed defaults for one test block.

        Shared by the final step of both wizards and by the right-click editor.  Returns
        ``(sweep_ranges, base_with_defaults_applied, edge_spacing)``, or None when the dialog is
        cancelled or the entries do not validate.
        """
        fixed_keys=[key for key in keys if key not in set(selected)]
        ranges_dialog=QDialog(self);ranges_dialog.setWindowTitle(title);ranges_dialog.resize(1420,min(940,max(500,290+60*(len(selected)+min(len(fixed_keys),8)))));ranges_dialog.setMinimumWidth(1180);ranges_layout=QVBoxLayout(ranges_dialog)
        values_hint=QLabel("Scanned parameters are listed first — enter explicit comma-separated values (for example: 500, 1000, 2000), or leave that field blank to use Start / Stop / Step. Every remaining parameter follows, where the last column sets the fixed default used by all devices in the block.");values_hint.setWordWrap(True);ranges_layout.addWidget(values_hint)
        table=QTableWidget(len(selected)+len(fixed_keys),5);table.setHorizontalHeaderLabels(["Parameter","Start","Stop","Step","Explicit values / default"]);header=table.horizontalHeader();header.setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
        for column in (1,2,3):header.setSectionResizeMode(column,QHeaderView.ResizeMode.Fixed);table.setColumnWidth(column,205)
        header.setSectionResizeMode(4,QHeaderView.ResizeMode.Stretch);table.setColumnWidth(4,320);table.verticalHeader().setDefaultSectionSize(54);editors={};fixed_editors={}
        for offset,key in enumerate(fixed_keys):fixed_editors[key]=add_fixed_default_row(table,len(selected)+offset,key,base[key])
        for row,key in enumerate(selected):
            table.setItem(row,0,QTableWidgetItem(f"● {key.replace('_',' ')}"));value=base[key];saved=saved_ranges.get(key) or {}
            if key in CHOICE_PARAMETERS or isinstance(value,(str,bool)):
                values=list(saved["values"]) if saved.get("values") else CHOICE_PARAMETERS.get(key,[False,True] if isinstance(value,bool) else [value]);entry=QLineEdit(", ".join(str(v).lower() if isinstance(v,bool) else str(v) for v in values));entry.setMinimumSize(290,42)
                for column in (1,2,3):table.setItem(row,column,QTableWidgetItem("—"))
                table.setCellWidget(row,4,entry);editors[key]=("values",entry)
            else:
                nominal=float(value);span=max(abs(nominal)*0.2,1.0);start=max(0.0,nominal-span);stop=nominal+span;step=max((stop-start)/4.0,1.0 if isinstance(value,int) else 0.001);widgets=[]
                if "start" in saved:start,stop,step=float(saved.get("start",start)),float(saved.get("stop",stop)),float(saved.get("step",step))
                for column,initial in zip((1,2,3),(start,stop,step)):
                    box=QDoubleSpinBox();box.setRange(-1e9,1e9);box.setDecimals(0 if isinstance(value,int) else 6);box.setValue(initial);box.setMinimumSize(185,42);table.setCellWidget(row,column,box);widgets.append(box)
                explicit=QLineEdit();explicit.setPlaceholderText("e.g. 500, 1000, 2000");explicit.setMinimumSize(290,42)
                if saved.get("values"):explicit.setText(", ".join(f"{float(v):g}" for v in saved["values"]))
                elif not saved and key in (explicit_defaults or {}):explicit.setText((explicit_defaults or {})[key])
                table.setCellWidget(row,4,explicit);editors[key]=("numeric",widgets,isinstance(value,int),explicit)
        ranges_layout.addWidget(table)
        packing_hint=QLabel("The length-like scan parameter forms columns. Every combination of gap, width, and other selected values forms rows. True device and label bounds determine the compact row and column spacing.");packing_hint.setWordWrap(True);ranges_layout.addWidget(packing_hint)
        options=QFormLayout();edge_box=QDoubleSpinBox();edge_box.setRange(0,1e7);edge_box.setDecimals(3);edge_box.setValue(float(edge_spacing));edge_box.setMinimumSize(230,40);options.addRow("Minimum edge-to-edge spacing (µm)",edge_box);ranges_layout.addLayout(options)
        range_buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);range_buttons.accepted.connect(ranges_dialog.accept);range_buttons.rejected.connect(ranges_dialog.reject);ranges_layout.addWidget(range_buttons)
        if ranges_dialog.exec()!=QDialog.DialogCode.Accepted:return None
        sweep_ranges={};combination_count=1;updated=safe_json_copy(base)
        try:
            for key,editor in editors.items():
                if editor[0]=="values":
                    values=[token.strip() for token in editor[1].text().split(",") if token.strip()];allowed=CHOICE_PARAMETERS.get(key)
                    if isinstance(base[key],bool):values=[value.lower() in {"1","true","yes","on"} for value in values]
                    elif allowed:
                        invalid=[value for value in values if value not in allowed]
                        if invalid:raise ValueError(f"Invalid {key} value(s): {', '.join(invalid)}")
                    if not values:raise ValueError(f"Enter at least one value for {key}.")
                    sweep_ranges[key]={"values":values};combination_count*=len(values)
                else:
                    boxes,is_integer,explicit=editor[1],editor[2],editor[3];custom=explicit.text().strip()
                    if custom:
                        values=numeric_list(custom)
                        if is_integer:
                            if any(abs(value-round(value))>1e-9 for value in values):raise ValueError(f"{key} requires whole-number values.")
                            values=[int(round(value)) for value in values]
                        sweep_ranges[key]={"values":values}
                    else:
                        start,stop,step=(box.value() for box in boxes);values=inclusive_sweep(start,stop,step);sweep_ranges[key]={"start":start,"stop":stop,"step":step}
                    combination_count*=len(values)
            if combination_count>500:raise ValueError(f"This scan creates {combination_count} devices. Reduce it to 500 or fewer.")
            for key,spec in fixed_editors.items():updated[key]=read_fixed_default(spec)
        except Exception as exc:
            QMessageBox.critical(self,f"Invalid {family} scan",str(exc));return None
        return sweep_ranges,updated,float(edge_box.value())

    def edit_test_block_scan(self, component: dict[str, Any]) -> None:
        """Right-click entry point: re-open the ranges-and-defaults table for an existing block."""
        is_rf=component.get("kind")=="RF test block";family="RF" if is_rf else "photonic"
        params=component["params"];base_key="rf_base_params" if is_rf else "photonic_base_params"
        source_kind=str(params.get("rf_component_kind" if is_rf else "photonic_component_kind") or ("CPW" if is_rf else "Straight"))
        if source_kind not in DEFAULT_COMPONENT_VALUES:
            QMessageBox.critical(self,"Edit scan",f"Unknown component kind '{source_kind}'.");return
        base=safe_json_copy(DEFAULT_COMPONENT_VALUES[source_kind])
        base.update({key:safe_json_copy(value) for key,value in (params.get(base_key) or {}).items() if key in base})
        excluded={"layer","datatype","points","oxide_layer","oxide_datatype"} if is_rf else {"layer","datatype","gc_layer","gc_datatype","waveguide_layer","waveguide_datatype","resonator_layer","resonator_datatype","hole_layer","hole_datatype","points","taper_points"}
        keys=[key for key in base if key not in excluded and (is_rf or ("tolerance" not in key and not key.endswith("_points")))]
        selected=[str(key) for key in (params.get("sweep_parameters") or []) if str(key) in keys]
        if not selected:
            QMessageBox.warning(self,"Edit scan","This block has no scan parameters recorded. Re-create it to choose which parameters to scan.");return
        result=self.test_block_scan_dialog(family,f"{component.get('kind')} — ranges, defaults and spacing ({component_display_name(source_kind)})",base,keys,selected,params.get("sweep_ranges") or {},float(params.get("edge_spacing",300.0)))
        if result is None:return
        sweep_ranges,updated_base,edge_spacing=result
        snapshot=self.snapshot()
        params[base_key]=updated_base;params["sweep_ranges"]=sweep_ranges;params["edge_spacing"]=edge_spacing
        self.commit_interaction_snapshot(snapshot);self.rebuild_scene();self.show_component_properties(component);self.statusBar().showMessage(f"Updated {component.get('kind')} scan ranges and defaults.",8000)

    def configure_rf_test_block(self, component: dict[str, Any]) -> bool:
        """Three-step RF component, parameter, and range selection wizard."""
        source_kinds = sorted(RF_COMPONENT_KINDS - {"RF test block"})
        choose = QDialog(self);choose.setWindowTitle("RF test block — 1 of 3: component and label");choose.resize(650,520);form=QFormLayout(choose)
        component_box=QComboBox()
        for kind in source_kinds:component_box.addItem(component_display_name(kind),kind)
        saved_kind=str(component["params"].get("rf_component_kind","") or "")
        if saved_kind in source_kinds:component_box.setCurrentIndex(source_kinds.index(saved_kind))
        form.addRow("RF component to scan",component_box)
        label_box=QLineEdit(str(component["params"].get("device_label_prefix") or component_box.currentData()));label_box.setMinimumWidth(320);label_box.setToolTip("Prefix added to every generated device label, for example CP.");form.addRow("Device label prefix",label_box)
        component_box.currentIndexChanged.connect(lambda *_:label_box.setText(str(component_box.currentData())))
        label_height_box=QDoubleSpinBox();label_height_box.setRange(0.1,10000.0);label_height_box.setDecimals(3);label_height_box.setValue(float(component["params"].get("label_height",20.0)));label_height_box.setMinimumSize(220,38);form.addRow("Label text height (µm)",label_height_box)
        label_x_box=QDoubleSpinBox();label_x_box.setRange(-1e7,1e7);label_x_box.setDecimals(3);label_x_box.setValue(float(component["params"].get("label_offset_x",0.0)));label_x_box.setMinimumSize(220,38);form.addRow("Label X offset from top-left (µm)",label_x_box)
        label_y_box=QDoubleSpinBox();label_y_box.setRange(-1e7,1e7);label_y_box.setDecimals(3);label_y_box.setValue(float(component["params"].get("label_offset_y",10.0)));label_y_box.setMinimumSize(220,38);form.addRow("Label Y offset from top-left (µm)",label_y_box)
        taper_center_box=QComboBox();taper_center_box.addItems(["CPW","T electrode"]);taper_center_box.setCurrentText(str(component["params"].get("taper_test_center","CPW")));form.addRow("Taper test center section",taper_center_box)
        probe_cpw_box=QDoubleSpinBox();probe_cpw_box.setRange(0.0,1e7);probe_cpw_box.setDecimals(3);probe_cpw_box.setValue(float(component["params"].get("probe_cpw_length",100.0)));probe_cpw_box.setMinimumSize(220,38);form.addRow("Probe CPW length, each end (µm)",probe_cpw_box)
        input_transition_box=QDoubleSpinBox();input_transition_box.setRange(0.0,1e7);input_transition_box.setDecimals(3);input_transition_box.setValue(float(component["params"].get("input_transition_length",0.0)));input_transition_box.setMinimumSize(220,38);form.addRow("Input transition before taper (µm)",input_transition_box)
        output_transition_box=QDoubleSpinBox();output_transition_box.setRange(0.0,1e7);output_transition_box.setDecimals(3);output_transition_box.setValue(float(component["params"].get("output_transition_length",0.0)));output_transition_box.setMinimumSize(220,38);form.addRow("Output transition after taper (µm)",output_transition_box)
        t_transition_box=QDoubleSpinBox();t_transition_box.setRange(0.0,1e7);t_transition_box.setDecimals(3);t_transition_box.setValue(float(component["params"].get("t_electrode_transition_length",0.0)));t_transition_box.setMinimumSize(220,38);form.addRow("T-electrode transition, each side (µm)",t_transition_box)
        taper_widgets=(taper_center_box,probe_cpw_box,input_transition_box,output_transition_box,t_transition_box)
        def update_taper_test_fields(*_):
            is_taper=str(component_box.currentData()) in {"Tapered CPW","Symmetric CPW taper"}
            for widget in taper_widgets:
                widget.setVisible(is_taper);label=form.labelForField(widget)
                if label is not None:label.setVisible(is_taper)
            t_visible=is_taper and taper_center_box.currentText()=="T electrode";t_transition_box.setVisible(t_visible);t_label=form.labelForField(t_transition_box)
            if t_label is not None:t_label.setVisible(t_visible)
        component_box.currentIndexChanged.connect(update_taper_test_fields);taper_center_box.currentIndexChanged.connect(update_taper_test_fields);update_taper_test_fields()
        hint=QLabel("Choose a component from the RF library. The next window selects its scan parameters.");hint.setWordWrap(True);form.addRow(hint)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next");buttons.accepted.connect(choose.accept);buttons.rejected.connect(choose.reject);form.addRow(buttons)
        if choose.exec()!=QDialog.DialogCode.Accepted:return False
        source_kind=str(component_box.currentData());base=safe_json_copy(DEFAULT_COMPONENT_VALUES[source_kind])
        if source_kind==saved_kind:base.update({key:safe_json_copy(value) for key,value in (component["params"].get("rf_base_params") or {}).items() if key in base})
        excluded={"layer","datatype","points","oxide_layer","oxide_datatype"}
        keys=[key for key in base if key not in excluded]
        common={
            "CPW":["signal_width","ground_width","length","gap"],
            "CPW open":["signal_width","ground_width","length","gap","signal_recess"],
            "CPW short":["signal_width","ground_width","length","gap","bridge_length"],
            "Tapered CPW":["length","final_gap","initial_gap","signal_width","ground_width","profile","exponential_factor","target_s11_db"],
            "Symmetric CPW taper":["taper_length","middle_gap","initial_gap","signal_width","ground_width","end_straight_length","middle_straight_length","profile","exponential_factor","target_s11_db"],
            "CPW bend":["R_eff","bend_angle_deg","signal_width","ground_width","gap"],
            "Segmented electrode":["signal_width","ground_width","segment_count","gap","end_gap","transition_length","t_top_width","t_top_length","t_neck_width","t_neck_length","segment_spacing"],
        }.get(source_kind,keys[:6])
        default_selected={
            "CPW":{"signal_width","length"},
            "Tapered CPW":{"length","final_gap","profile"},
            "Symmetric CPW taper":{"taper_length","middle_gap","profile"},
            "Segmented electrode":{"signal_width","ground_width","segment_count"},
        }.get(source_kind,set(common[:3]))
        saved_selected=[str(key) for key in (component["params"].get("sweep_parameters") or [])] if source_kind==saved_kind else []
        default_selected=set(key for key in saved_selected if key in keys) or default_selected
        select=QDialog(self);select.setWindowTitle("RF test block — 2 of 3: parameters");select.resize(560,560);layout=QVBoxLayout(select)
        title=QLabel(f"Select parameters to scan for <b>{component_display_name(source_kind)}</b>.");title.setWordWrap(True);layout.addWidget(title)
        show_all=QCheckBox("Show full physical parameter list");layout.addWidget(show_all)
        scroll=QScrollArea();scroll.setWidgetResizable(True);holder=QWidget();holder_layout=QVBoxLayout(holder);checks={}
        for key in keys:
            check=QCheckBox(key.replace("_"," "));check.setChecked(key in default_selected);check.setVisible(key in common or key in default_selected);holder_layout.addWidget(check);checks[key]=check
        holder_layout.addStretch(1);scroll.setWidget(holder);layout.addWidget(scroll,1)
        show_all.toggled.connect(lambda enabled:[check.setVisible(enabled or key in common or key in default_selected) for key,check in checks.items()])
        select_buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);select_buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next");select_buttons.accepted.connect(select.accept);select_buttons.rejected.connect(select.reject);layout.addWidget(select_buttons)
        if select.exec()!=QDialog.DialogCode.Accepted:return False
        selected=[key for key,check in checks.items() if check.isChecked()]
        if not selected:QMessageBox.warning(self,"RF test block","Select at least one parameter to scan.");return False

        rf_explicit_defaults={
            "CPW":{"signal_width":"10, 11, 12, 13, 14, 15","length":"500, 1000, 2000"},
            "Tapered CPW":{"length":"500, 1000, 2000"},
            "Symmetric CPW taper":{"taper_length":"500, 1000, 2000"},
        }
        result=self.test_block_scan_dialog("RF","RF test block — 3 of 3: ranges, defaults and spacing",base,keys,selected,(component["params"].get("sweep_ranges") or {}) if source_kind==saved_kind else {},float(component["params"].get("edge_spacing",300.0)),rf_explicit_defaults.get(source_kind,{}))
        if result is None:return False
        sweep_ranges,base,edge_spacing=result
        component["params"].pop("columns",None);component["params"].update({"rf_component_kind":source_kind,"rf_base_params":base,"device_label_prefix":label_box.text().strip() or source_kind,"label_height":label_height_box.value(),"label_offset_x":label_x_box.value(),"label_offset_y":label_y_box.value(),"taper_test_structure":source_kind in {"Tapered CPW","Symmetric CPW taper"},"taper_test_center":taper_center_box.currentText(),"probe_cpw_length":probe_cpw_box.value(),"input_transition_length":input_transition_box.value(),"output_transition_length":output_transition_box.value(),"t_electrode_transition_length":t_transition_box.value(),"sweep_parameters":selected,"sweep_ranges":sweep_ranges,"edge_spacing":edge_spacing})
        return True

    def configure_photonic_test_block(self, component: dict[str, Any]) -> bool:
        """Choose a photonic component, its scan parameters, and scan ranges."""
        source_kinds=sorted(PHOTONIC_COMPONENT_KINDS)
        choose=QDialog(self);choose.setWindowTitle("Photonic test block — 1 of 3: component and label");choose.resize(650,320);form=QFormLayout(choose)
        component_box=QComboBox()
        for kind in source_kinds:component_box.addItem(component_display_name(kind),kind)
        saved_kind=str(component["params"].get("photonic_component_kind","") or "")
        if saved_kind in source_kinds:component_box.setCurrentIndex(source_kinds.index(saved_kind))
        form.addRow("Photonic component to scan",component_box)
        label_box=QLineEdit(str(component["params"].get("device_label_prefix") or component_box.currentData()));label_box.setMinimumWidth(320);label_box.setToolTip("Prefix added to every generated device label, for example MZI.");form.addRow("Device label prefix",label_box)
        component_box.currentIndexChanged.connect(lambda *_:label_box.setText(str(component_box.currentData())))
        label_height_box=QDoubleSpinBox();label_height_box.setRange(0.1,10000.0);label_height_box.setDecimals(3);label_height_box.setValue(float(component["params"].get("label_height",20.0)));label_height_box.setMinimumSize(220,38);form.addRow("Label text height (µm)",label_height_box)
        label_x_box=QDoubleSpinBox();label_x_box.setRange(-1e7,1e7);label_x_box.setDecimals(3);label_x_box.setValue(float(component["params"].get("label_offset_x",0.0)));label_x_box.setMinimumSize(220,38);form.addRow("Label X offset from top-left (µm)",label_x_box)
        label_y_box=QDoubleSpinBox();label_y_box.setRange(-1e7,1e7);label_y_box.setDecimals(3);label_y_box.setValue(float(component["params"].get("label_offset_y",10.0)));label_y_box.setMinimumSize(220,38);form.addRow("Label Y offset from top-left (µm)",label_y_box)
        hint=QLabel("Choose a component from the photonic library. The next window selects its scan parameters.");hint.setWordWrap(True);form.addRow(hint)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next");buttons.accepted.connect(choose.accept);buttons.rejected.connect(choose.reject);form.addRow(buttons)
        if choose.exec()!=QDialog.DialogCode.Accepted:return False
        source_kind=str(component_box.currentData());base=safe_json_copy(DEFAULT_COMPONENT_VALUES[source_kind])
        if source_kind==saved_kind:base.update({key:safe_json_copy(value) for key,value in (component["params"].get("photonic_base_params") or {}).items() if key in base})
        excluded={"layer","datatype","gc_layer","gc_datatype","waveguide_layer","waveguide_datatype","resonator_layer","resonator_datatype","hole_layer","hole_datatype","points","taper_points"}
        keys=[key for key in base if key not in excluded and "tolerance" not in key and not key.endswith("_points")]
        common={
            "Straight":["length","width"],
            "Taper":["length","width_start","width_end"],
            "S-bend":["length","offset","width"],
            "Euler bend":["radius","bend_angle_deg","width","euler_fraction"],
            "Grating coupler":["pitch","fill_factor","N","alpha_t","taper_L","L_extra","wg_width","wg_length"],
            "1x2 MMI":["mmi_length","mmi_width","taper_width","wg_width","port_sep","input_taper_length","output_taper_length"],
            "Cascaded MMI":["N_levels","mmi_length","mmi_width","taper_width","wg_width","s_bend_length","output_gc_spacing"],
            "MMI + Reference":["mmi_length","mmi_width","taper_width","wg_width","reference_dy","reference_branch"],
            "MMI split-combine cascade":["cascade_count","mmi_length","mmi_width","taper_width","wg_width","interconnect_length"],
            "MZI":["mmi_length","mmi_width","taper_width","wg_width","arm_separation","s_bend_length","arm_length"],
            "MZI vertical GC":["mmi_length","mmi_width","taper_width","wg_width","arm_separation","arm_length","gc_vertical_run"],
            "Ring":["radius","width"],
            "Elliptical ring":["radius_x","radius_y","width"],
            "Racetrack":["radius","coupling_length","width"],
            "Ring + two feedlines":["ring_radius","ring_width","coupling_gap","feedline_width","feedline_length","grating_coupler_separation"],
            "Edge coupler":["tip_width","wg_width","taper_length","wg_straight_length"],
            "Loopback mirror":["Lc","gap","s_bend_length","arc_radius","width"],
            "Feedline":["wg_width","Lc","offset","s_bend_length","pitch","fill_factor"],
            "Ring + feedline":["ring_radius","ring_width","coupling_gap","resonator_count","resonator_spacing","wg_width","Lc"],
            "Racetrack + feedline":["racetrack_length","racetrack_radius","racetrack_width","Lc","coupling_gap","resonator_count","resonator_spacing","wg_width"],
            "Photonic crystal":["pitch_x","pitch_y","hole_radius_x","hole_radius_y","columns","rows","defect_rows","lattice","hole_shape"],
        }.get(source_kind,keys[:6])
        common=[key for key in common if key in keys]
        saved_selected=[str(key) for key in (component["params"].get("sweep_parameters") or [])] if source_kind==saved_kind else []
        default_selected=set(key for key in saved_selected if key in keys) or set(common[:3])
        select=QDialog(self);select.setWindowTitle("Photonic test block — 2 of 3: parameters");select.resize(560,560);layout=QVBoxLayout(select)
        title=QLabel(f"Select parameters to scan for <b>{component_display_name(source_kind)}</b>.");title.setWordWrap(True);layout.addWidget(title)
        show_all=QCheckBox("Show full physical parameter list");layout.addWidget(show_all)
        scroll=QScrollArea();scroll.setWidgetResizable(True);holder=QWidget();holder_layout=QVBoxLayout(holder);checks={}
        for key in keys:
            check=QCheckBox(key.replace("_"," "));check.setChecked(key in default_selected);check.setVisible(key in common or key in default_selected);holder_layout.addWidget(check);checks[key]=check
        holder_layout.addStretch(1);scroll.setWidget(holder);layout.addWidget(scroll,1)
        show_all.toggled.connect(lambda enabled:[check.setVisible(enabled or key in common or key in default_selected) for key,check in checks.items()])
        select_buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);select_buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next");select_buttons.accepted.connect(select.accept);select_buttons.rejected.connect(select.reject);layout.addWidget(select_buttons)
        if select.exec()!=QDialog.DialogCode.Accepted:return False
        selected=[key for key,check in checks.items() if check.isChecked()]
        if not selected:QMessageBox.warning(self,"Photonic test block","Select at least one parameter to scan.");return False

        result=self.test_block_scan_dialog("photonic","Photonic test block — 3 of 3: ranges, defaults and spacing",base,keys,selected,(component["params"].get("sweep_ranges") or {}) if source_kind==saved_kind else {},float(component["params"].get("edge_spacing",300.0)))
        if result is None:return False
        sweep_ranges,base,edge_spacing=result
        component["params"].pop("columns",None);component["params"].update({"photonic_component_kind":source_kind,"photonic_base_params":base,"device_label_prefix":label_box.text().strip() or source_kind,"label_height":label_height_box.value(),"label_offset_x":label_x_box.value(),"label_offset_y":label_y_box.value(),"sweep_parameters":selected,"sweep_ranges":sweep_ranges,"edge_spacing":edge_spacing})
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
        rf_parent: dict[str, Any] | None = None
        if name in RF_SIMULATION_OBJECT_KINDS:
            selected_rf_parents = [
                candidate for candidate in self.selected_components()
                if str(candidate.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
            ]
            if len(selected_rf_parents) == 1:
                rf_parent = selected_rf_parents[0]
                component["simulation_parent_uid"] = int(rf_parent["uid"])
                component["orientation_deg"] = float(rf_parent.get("orientation_deg", 0.0))
        if name=="RF test block" and not self.configure_rf_test_block(component):
            self.next_uid=max(1,self.next_uid-1);return
        if name=="Photonic test block" and not self.configure_photonic_test_block(component):
            self.next_uid=max(1,self.next_uid-1);return
        if name=="Photonic crystal" and not self.configure_photonic_crystal(component):
            self.next_uid=max(1,self.next_uid-1);return
        if name in LEGACY_PHOTONIC_TEST_BLOCK_KINDS and not self.configure_test_block_sweeps(component):
            self.next_uid=max(1,self.next_uid-1);return
        self.components.append(component)
        companions = self.automatic_simulation_companions(component)
        if companions:
            group_id = f"G{self.next_group_id}"
            self.next_group_id += 1
            component["group_id"] = group_id
            for companion in companions:
                companion["group_id"] = group_id
                self.components.append(companion)
        self.commit_interaction_snapshot(snapshot)
        self.scene.clearSelection()
        self.add_component_scene_item(component, selected=True)
        for companion in companions:
            self.add_component_scene_item(companion, selected=False)
        self.refresh_project_tree()
        self.on_scene_selection_changed()
        if companions:
            self.statusBar().showMessage(f"Added {name} with {len(companions)} movable FDTD/fiber simulation object(s).")
        elif rf_parent is not None:
            self.statusBar().showMessage(
                f"Added movable {name} and attached it to {rf_parent.get('kind')} UID {rf_parent.get('uid')}."
            )
        else:
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
        # Auto-generated coverage follows its source until a user moves or
        # rearranges the field set.  A manually locked layout is intentionally
        # independent and must survive scene rebuilds and project reopen.
        for component in self.components:
            if component.get("kind")!="E-beam multipass":continue
            if bool(component.get("params",{}).get("manual_layout_locked",False)):continue
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
        if primary.component.get("kind") in SIMULATION_COMPONENT_KINDS:
            return
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
        if component.get("kind") in {"Grating coupler", "GC-SOI"}:
            migrate_grating_fiber_offset_parameter(component)
            grating_params = component.setdefault("params", {})
            grating_params.setdefault("fill_factors", "")
            grating_params.setdefault("tooth_shape", "curved")
            if (
                component.get("kind") == "Grating coupler"
                and abs(float(grating_params.get("waveguide_monitor_span_um", 3.0)) - 2.5) <= 1e-12
            ):
                grating_params["waveguide_monitor_span_um"] = 3.0
            grating_params.setdefault(
                "waveguide_monitor_span_um",
                float(
                    grating_params.pop(
                        "waveguide_port_span_um",
                        2.5 if component.get("kind") == "GC-SOI" else 3.0,
                    )
                ),
            )
            grating_params.setdefault("waveguide_total_power_before_mode_um", 1.0)
            # Legacy projects serialized a guessed platform-specific neff.
            # Remove it from the editable model: export derives the validation
            # target from the actual dispersive stack at the center wavelength.
            grating_params.pop("waveguide_effective_index", None)
            grating_params.setdefault("waveguide_neff_tolerance", 0.3)
            grating_params.setdefault("waveguide_mode_search_count", 20)
        if component.get("kind") == "1x2 MMI":
            # Migrate older saved MMIs to the shared three-port modal
            # validation settings without changing their physical geometry.
            mmi_params = component.setdefault("params", {})
            mmi_params.pop("waveguide_effective_index", None)
            mmi_params.setdefault("waveguide_neff_tolerance", 0.3)
            mmi_params.setdefault("waveguide_mode_search_count", 20)
        if component.get("kind") == "GC-SOI":
            params = component.get("params", {})
            pitch = float(params.get("pitch", 0.6713))
            target_length = float(params.get("target_length", 25.0))
            derived_count = int(math.ceil(target_length / pitch)) if pitch > 0.0 else 0
            count_label = QLabel(str(derived_count))
            count_label.setToolTip("Automatically calculated as ceil(target length / pitch).")
            self.properties_form.addRow("Derived grating tooth count (N)", count_label)
        if component.get("kind") == "Grating coupler":
            # Older saved layouts predate the editable waveguide-side offset.
            component.setdefault("params", {})
            component["params"].setdefault("fdtd_port_offset_from_waveguide_end_um", 2.0)
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
            if component.get("kind") in {"RF test block","Photonic test block"} and key=="columns":
                continue
            if key in {
                "manual_field_offsets",
                "manual_field_order",
                "removed_field_keys",
                "auto_pruned_field_keys",
                "explicit_fields",
                "sweep_parameters",
                "sweep_ranges",
                "rf_base_params",
                "photonic_base_params",
                "polygons",
            }:
                continue
            if (
                key == "angle theta"
                and bool(component.get("auto_placed", False))
                and component.get("simulation_parent_uid") is not None
            ):
                controlled = QLabel(
                    f"{float(value):.9g}° — controlled by the parent grating angle_theta"
                )
                controlled.setToolTip(
                    "Edit Angle theta on the parent grating coupler. The fiber geometry, "
                    "source plane, and passive measurement plane update together."
                )
                self.properties_form.addRow("Angle theta (parent controlled)", controlled)
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
            if key == "fill_factors" and isinstance(widget, QLineEdit):
                widget.setPlaceholderText("Uniform when blank; e.g. linspace(0.35, 0.55)")
                widget.setToolTip(
                    "Enter one fill factor per grating tooth as a list, or use "
                    "linspace(start, stop). The component supplies the tooth count automatically."
                )
            parameter_label = (
                "Fiber offset (µm)" if key == "fiber_offset"
                else "Angle theta (degrees)" if key == "angle_theta"
                else "Horizontal fiber-input monitor below source (µm)" if key == "fiber_power_monitor_below_source_um"
                else "Waveguide mode-monitor offset from end (µm)" if key == "fdtd_port_offset_from_waveguide_end_um"
                else "Waveguide monitor span (µm)" if key == "waveguide_monitor_span_um"
                else "Total-power monitor before receiver port (µm)" if key == "waveguide_total_power_before_mode_um"
                else "Allowed effective-index difference" if key == "waveguide_neff_tolerance"
                else "Waveguide modes to search" if key == "waveguide_mode_search_count"
                else "Apodized fill factors (one per tooth)" if key == "fill_factors"
                else "Grating tooth geometry" if key == "tooth_shape"
                else "Number of grating teeth (N)" if key == "N" and component.get("kind") == "Grating coupler"
                else "Uniform fill factor" if key == "fill_factor" and component.get("kind") == "Grating coupler"
                else "Uniform duty cycle" if key == "duty_cycle" and component.get("kind") == "GC-SOI"
                else "GDS curve tolerance (µm)" if key == "tolerance" and component.get("kind") in {"Grating coupler", "GC-SOI"}
                else "Grating straight waveguide length (µm)" if key == "wg_length" and component.get("kind") in {"Grating coupler", "GC-SOI"}
                else "GC straight lead length (µm)" if key == "gc_wg_length"
                else "CPW taper model" if key == "cpw_profile"
                else key
            )
            self.properties_form.addRow(parameter_label, widget)
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
                component.get("kind") in {"Power monitor", "Mode expansion monitor", "Field profile monitor"}
                and str(component.get("params", {}).get("monitor geometry", "surface")).lower() == "surface"
            ):
                params = component["params"]
                normal = str(params.get("plane normal", "X")).upper()
                transverse = max(
                    0.001,
                    abs(float(params.get("x span", 0.0))),
                    abs(float(params.get("y span", 0.0))),
                    abs(float(params.get("span_um", 4.0))),
                )
                raw_depth = abs(float(params.get("z span", params.get("z_span_um", 2.0))))
                depth = raw_depth if raw_depth > 0.0 else 2.0
                if normal == "Y":
                    params.update({"x span": transverse, "y span": 0.0, "z span": depth})
                elif normal == "Z":
                    params.update({"x span": transverse, "y span": transverse, "z span": 0.0})
                else:
                    params.update({"plane normal": "X", "x span": 0.0, "y span": transverse, "z span": depth})
            synchronize_rf_taper_points(component)
            companions_changed = self.synchronize_automatic_simulation_companions(component)
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
            if component.get("kind") == "E-beam multipass" or companions_changed:
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
        if kind in SIMULATION_COMPONENT_KINDS:
            return SIMULATION_LAYER
        if kind == "E-beam multipass":
            return EBEAM_LAYER
        if kind in RF_COMPONENT_KINDS or kind == "MZI + CPW module":
            return RF_LAYER
        if kind in MARKER_COMPONENT_KINDS:
            return MARKER_LAYER
        if kind in {"Grating coupler", "GC-SOI"}:
            return GC_LAYER
        if kind in {"Chip outline", "4-inch wafer outline"}:
            return int(component.get("params", {}).get("layer", 100))
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
        menu.addSeparator()
        export_lumerical_action = menu.addAction("Lumerical run…")
        sweep_lumerical_action = menu.addAction("Lumerical sweep…")
        multigpu_sweep_lumerical_action = menu.addAction("Lumerical sweep-multithread…")
        optimize_lumerical_action = menu.addAction("Lumerical optimization…")
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
        selected_components = self.selected_components()
        selected_rf_targets = [
            component for component in selected_components
            if str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
        ]
        valid_rf_selection = (
            len(selected_rf_targets) == 1
            and all(
                str(component.get("kind", ""))
                in RF_SIMULATABLE_COMPONENT_KINDS | RF_SIMULATION_OBJECT_KINDS
                for component in selected_components
            )
        )
        selected_rf_target = (
            selected_rf_targets[0]
            if valid_rf_selection
            else self.rf_lumerical_target_component(
                selected_components[0] if len(selected_components) == 1 else None
            )
        )
        if selected_rf_target is not None:
            export_lumerical_action.setText("Lumerical RF run…")
            export_lumerical_action.setToolTip(
                "Use MODE/FDE for a uniform CPW or 3D FDTD S-parameters for an RF discontinuity."
            )
        export_lumerical_action.setEnabled(len(selected_components) == 1 or valid_rf_selection)
        optical_sweep_enabled = len(selected_components) == 1 and selected_rf_target is None
        sweep_lumerical_action.setEnabled(optical_sweep_enabled)
        multigpu_sweep_lumerical_action.setEnabled(optical_sweep_enabled)
        optimize_lumerical_action.setEnabled(optical_sweep_enabled)
        chosen = menu.exec(self.project_tree.viewport().mapToGlobal(position))
        if chosen is go_action:
            self.fit_selection()
        elif chosen is duplicate_action:
            self.duplicate_selected()
        elif chosen is array_action:
            self.create_array()
        elif chosen is lattice_action:
            self.create_photonic_crystal_lattice()
        elif chosen is export_lumerical_action:
            self.export_lumerical_notebook(self.selected_components()[0])
        elif chosen is sweep_lumerical_action:
            self.export_lumerical_sweep_notebook(self.selected_components()[0])
        elif chosen is multigpu_sweep_lumerical_action:
            self.export_lumerical_multigpu_sweep_notebook(self.selected_components()[0])
        elif chosen is optimize_lumerical_action:
            self.export_lumerical_optimization_notebook(self.selected_components()[0])
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

    def component_world_bbox(self, component: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """World-space bounding box of one component, or None when it draws nothing."""
        polygons,_labels=component_geometry_arrays(component)
        if not polygons:return None
        stacked=np.vstack([points for points,_layer,_datatype in polygons])
        low=stacked.min(axis=0);high=stacked.max(axis=0)
        return (float(low[0]),float(low[1])),(float(high[0]),float(high[1]))

    def test_block_devices_outside(self, component: dict[str, Any], contains):
        """(kept indices, excluded indices) for one test block, or None if it builds nothing.

        Judged per generated device rather than per block, so a block straddling the edge keeps the
        devices that fit instead of being deleted whole. Every device is re-tested from scratch, so
        a previously excluded one comes back if the block is later moved inside."""
        try:
            placements=test_block_device_placements(component)
        except Exception:
            return None
        if not placements:return None
        origin=np.array([float(component["x"]),float(component["y"])],float);orientation=float(component.get("orientation_deg",0.0))
        kept=[];excluded=[]
        for index,polygons,shift in placements:
            points=[np.asarray(polygon.points,float)+shift for polygon in polygons]
            if not points:continue
            world=transform_points(np.vstack(points),origin,orientation)
            low=world.min(axis=0);high=world.max(axis=0)
            if all(contains(px,py) for px in (low[0],high[0]) for py in (low[1],high[1])):kept.append(index)
            else:excluded.append(index)
        return kept,excluded

    def boundary_containment_test(self):
        """Return (contains(x, y), description) for the layout's Chip outline or wafer outline,
        inset by the edge margin. Dimensions come from the params rather than the drawn bounds,
        because these outlines also render dimension text well outside the boundary itself.
        Both shapes are convex, so a bounding box fits exactly when its four corners do."""
        boundaries=[component for component in self.components if component.get("kind") in BOUNDARY_COMPONENT_KINDS]
        if not boundaries:raise ValueError("This layout has no Chip outline or 4-inch wafer outline component; add one to define the boundary.")
        if len(boundaries)>1:raise ValueError("This layout has "+str(len(boundaries))+" boundary components ("+", ".join(sorted({str(component.get("kind")) for component in boundaries}))+"); keep exactly one.")
        boundary=boundaries[0];p=boundary.get("params",{});kind=str(boundary.get("kind"))
        cx,cy=float(boundary["x"]),float(boundary["y"]);margin=BOUNDARY_MARGINS_UM[kind]
        orientation=float(boundary.get("orientation_deg",0.0))
        if kind=="Chip outline":
            squared=orientation%180.0
            if min(squared,abs(squared-90.0))>1e-9:raise ValueError(f"The chip outline is rotated {orientation:g}°; only multiples of 90° are supported.")
            width=float(p["width"]);height=float(p["height"])
            if abs(squared-90.0)<=1e-9:width,height=height,width
            x0,y0,x1,y1=cx-width/2.0+margin,cy-height/2.0+margin,cx+width/2.0-margin,cy+height/2.0-margin
            if x1<=x0 or y1<=y0:raise ValueError(f"A {margin:g} µm margin leaves no usable area inside a {width:g} × {height:g} µm chip outline.")
            return (lambda px,py:x0-1e-9<=px<=x1+1e-9 and y0-1e-9<=py<=y1+1e-9),f"{width:g} × {height:g} µm chip outline ({margin:g} µm edge margin)"
        radius=float(p.get("diameter",100000.0))/2.0;flat_length=float(p.get("primary_flat_length",32500.0))
        half_flat=flat_length/2.0
        if radius<=0 or half_flat<=0 or half_flat>=radius:raise ValueError("The wafer outline needs 0 < primary flat length < diameter.")
        keep_radius=radius-margin;flat_y=-math.sqrt(max(0.0,radius*radius-half_flat*half_flat))+margin
        if keep_radius<=0 or flat_y>=keep_radius:raise ValueError(f"A {margin:g} µm margin leaves no usable area inside a {radius*2.0:g} µm wafer outline.")
        angle=math.radians(-orientation);cos_a,sin_a=math.cos(angle),math.sin(angle)
        def contains(px,py):
            dx,dy=px-cx,py-cy;lx,ly=dx*cos_a-dy*sin_a,dx*sin_a+dy*cos_a
            return lx*lx+ly*ly<=keep_radius*keep_radius+1e-6 and ly>=flat_y-1e-9
        return contains,f"{radius*2.0/1000.0:g} mm wafer outline ({margin/1000.0:g} mm edge margin)"

    def remove_outside_chip_outline(self) -> None:
        if not self.components:
            return
        try:
            contains,description=self.boundary_containment_test()
            dropped=set();trimmed={}
            for component in self.components:
                if component.get("kind") in BOUNDARY_COMPONENT_KINDS:continue
                if component.get("kind") in {"RF test block","Photonic test block"}:
                    # A test block builds many devices; drop only the ones crossing the edge.
                    outside=self.test_block_devices_outside(component,contains)
                    if outside is None:continue
                    kept,excluded=outside
                    if not kept:dropped.add(int(component["uid"]))
                    elif excluded:trimmed[int(component["uid"])]=excluded
                    continue
                box=self.component_world_bbox(component)
                if box is None:continue
                if not all(contains(px,py) for px in (box[0][0],box[1][0]) for py in (box[0][1],box[1][1])):dropped.add(int(component["uid"]))
            if not dropped and not trimmed:
                self.statusBar().showMessage(f'Every component already fits inside the {description}.',8000);return
            snapshot=self.snapshot()
            self.components=[component for component in self.components if int(component["uid"]) not in dropped]
            device_total=0
            for component in self.components:
                if component.get("kind")=="E-beam multipass":
                    component["coverage_source_uids"]=[int(uid) for uid in component.get("coverage_source_uids",[]) if int(uid) not in dropped]
                excluded=trimmed.get(int(component["uid"]))
                if excluded:component["params"]["excluded_device_indices"]=sorted(excluded);device_total+=len(excluded)
            parts=[]
            if dropped:parts.append(f'{len(dropped)} component(s)')
            if device_total:parts.append(f'{device_total} test-block device(s)')
            self.commit_interaction_snapshot(snapshot);self.rebuild_scene();self.statusBar().showMessage(f'Removed {" and ".join(parts)} reaching outside the {description}.',8000)
        except Exception as exc:
            QMessageBox.critical(self,'Could not push within boundary',str(exc))

    def center_entire_layout_at_origin(self) -> None:
        if not self.components:
            return
        try:
            centered,initial_center,_=recenter_components_at_origin(self.components)
            snapshot=self.snapshot();self.components=centered
            self.commit_interaction_snapshot(snapshot);self.rebuild_scene();self.center_origin();self.statusBar().showMessage(f'Moved entire GDS by ({-initial_center[0]:.6g}, {-initial_center[1]:.6g}) µm; bounding-box center is now (0, 0).',8000)
        except Exception as exc:
            QMessageBox.critical(self,'Could not center complete GDS',str(exc))

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
            if component.get("kind") in {"Chip outline", "4-inch wafer outline"}:
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

    def start_fit_drawn_region(self) -> None:
        self.view.begin_fit_region()
        self.statusBar().showMessage("Drag a rectangle around the region to fit.",5000)

    def fit_drawn_region(self, bounds: QRectF) -> None:
        margin=max(bounds.width(),bounds.height())*0.02
        self.fit_rect(bounds,margin)
        self.statusBar().showMessage("Fitted the drawn region.",3000)

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
        self.synchronize_automatic_simulation_companions(component)
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

    def position_selected_ebeam_blocks(self) -> None:
        """Move only selected write-field components to an absolute center.

        Coverage-source UIDs are metadata used by the explicit Update / prune
        action.  They are deliberately not moved here, so the underlying GDS
        devices remain exactly where the user placed them.
        """
        blocks = [
            component for component in self.selected_components()
            if component.get("kind") == "E-beam multipass"
        ]
        if not blocks:
            QMessageBox.information(
                self,
                "Move Entire E-beam Block",
                "Select an E-beam write-field group first.",
            )
            return
        bounds = self.components_world_bounds(blocks)
        center_x = (bounds[0] + bounds[2]) / 2.0
        center_y = (bounds[1] + bounds[3]) / 2.0
        x, accepted = QInputDialog.getDouble(
            self,
            "Move Entire E-beam Block",
            "Target write-field center X (µm):",
            center_x,
            -1e9,
            1e9,
            6,
        )
        if not accepted:
            return
        y, accepted = QInputDialog.getDouble(
            self,
            "Move Entire E-beam Block",
            "Target write-field center Y (µm):",
            center_y,
            -1e9,
            1e9,
            6,
        )
        if not accepted:
            return
        snapshot = self.snapshot()
        dx, dy = x - center_x, y - center_y
        for component in blocks:
            component["x"] = float(component.get("x", 0.0)) + dx
            component["y"] = float(component.get("y", 0.0)) + dy
            component.setdefault("params", {})["manual_layout_locked"] = True
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(select_uids=[int(component["uid"]) for component in blocks])
        self.statusBar().showMessage(
            f"Moved {len(blocks)} E-beam field group(s) to ({x:g}, {y:g}) µm; source GDS stayed fixed.",
            8000,
        )

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
                "manual_layout_locked": False,
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
        component.setdefault("params",{})["manual_layout_locked"]=True
        moved=self.deoverlap_ebeam_fields(component,field_key);self.commit_interaction_snapshot(snapshot)
        QTimer.singleShot(0,lambda:self.rebuild_scene(select_uids=[uid]));self.statusBar().showMessage(f"Write field updated; {moved} overlap correction(s) applied. Manual layout is preserved.",7000)

    def finish_ebeam_group_move(self, uid: int, snapshot: str) -> None:
        component=self.component_by_uid(uid)
        if component is None:return
        component.setdefault("params",{})["manual_layout_locked"]=True
        self.commit_interaction_snapshot(snapshot)
        QTimer.singleShot(0,lambda:self.rebuild_scene(select_uids=[uid]));self.statusBar().showMessage("Moved the complete write-field set independently; source GDS stayed fixed.",7000)

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
            params = component["params"]
            params["manual_field_offsets"] = {}
            params["manual_field_order"] = {}
            params["removed_field_keys"] = []
            params["auto_pruned_field_keys"] = []
            params["manual_layout_locked"] = False
            source_components = [
                self.component_by_uid(int(uid))
                for uid in component.get("coverage_source_uids", [])
            ]
            source_components = [source for source in source_components if source is not None]
            if source_components:
                source_polygons = self.ebeam_source_polygons(source_components)
                if source_polygons:
                    bounds = self.polygon_collection_bounds(source_polygons)
                    component["x"] = (bounds[0] + bounds[2]) / 2.0
                    component["y"] = (bounds[1] + bounds[3]) / 2.0
                    params["target_width"] = bounds[2] - bounds[0]
                    params["target_height"] = bounds[3] - bounds[1]
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
            elif kind in {"Double-ring test block", "Grating test block", "Grating angle-taper test block", "MMI + Reference test block", "MMI split-combine test block", "Long MZI test block", "Vertical-GC MZI test block", "Vertical-GC MZI + CPW test block", "Vertical-GC MZI + segmented electrode test block", "Straight-GC MZI + segmented RF bends test block", "Straight-GC MZI + CPW RF bends test block"}:
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
    def rf_lumerical_target_component(
        self, clicked_component: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Resolve an RF geometry target from a device or manual RF plane."""
        if clicked_component is not None and str(clicked_component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS:
            return clicked_component
        if clicked_component is not None and str(clicked_component.get("kind", "")) in RF_SIMULATION_OBJECT_KINDS:
            parent_uid = clicked_component.get("simulation_parent_uid")
            if parent_uid is not None:
                parent = next(
                    (
                        component for component in self.components
                        if int(component.get("uid", -1)) == int(parent_uid)
                        and str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
                    ),
                    None,
                )
                if parent is not None:
                    return parent
            selected_targets = [
                component for component in self.selected_components()
                if str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
            ]
            if len(selected_targets) == 1:
                return selected_targets[0]
            candidates = [
                component for component in self.components
                if str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS
            ]
            if candidates:
                x = float(clicked_component.get("x", 0.0))
                y = float(clicked_component.get("y", 0.0))
                return min(
                    candidates,
                    key=lambda component: math.hypot(
                        float(component.get("x", 0.0)) - x,
                        float(component.get("y", 0.0)) - y,
                    ),
                )
        return None

    def rf_lumerical_scope_options(
        self, target_component: dict[str, Any]
    ) -> list[tuple[str, list[int]]]:
        """Offer RF-only geometry/plane scopes, preferring manual planes."""
        target_uid = int(target_component["uid"])
        options: list[tuple[str, list[int]]] = []
        seen: set[tuple[int, ...]] = set()

        def add(label: str, members: list[dict[str, Any]]) -> None:
            ordered: list[int] = []
            for member in members:
                kind = str(member.get("kind", ""))
                if kind not in RF_SIMULATABLE_COMPONENT_KINDS | RF_SIMULATION_OBJECT_KINDS:
                    continue
                uid = int(member["uid"])
                if uid not in ordered:
                    ordered.append(uid)
            if target_uid not in ordered:
                ordered.insert(0, target_uid)
            signature = tuple(ordered)
            if signature in seen:
                return
            seen.add(signature)
            options.append((label, ordered))

        selected = [
            component for component in self.selected_components()
            if str(component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS | RF_SIMULATION_OBJECT_KINDS
        ]
        selected_manual = [
            component for component in selected
            if str(component.get("kind", "")) in RF_SIMULATION_OBJECT_KINDS
        ]
        if selected_manual:
            add(
                f"Current RF selection — device + {len(selected_manual)} manual RF plane(s)",
                [target_component, *selected],
            )

        attached = [
            component for component in self.components
            if str(component.get("kind", "")) in RF_SIMULATION_OBJECT_KINDS
            and int(component.get("simulation_parent_uid", -1)) == target_uid
        ]
        if attached:
            add(
                f"Component + {len(attached)} attached RF port/monitor object(s)",
                [target_component, *attached],
            )

        for key, title in (
            ("group_id", "Complete RF group"),
            ("module_instance_id", "Complete RF module"),
        ):
            value = target_component.get(key)
            if value:
                members = [component for component in self.components if component.get(key) == value]
                add(f"{title} — {value}", members)

        add(
            f"Component only — {component_display_name(str(target_component.get('kind', 'CPW')))} "
            "(endpoint fallback if needed)",
            [target_component],
        )
        return options

    def export_lumerical_rf_notebook(
        self, clicked_component: dict[str, Any] | bool | None = None
    ) -> None:
        """Export a MODE/FDE or 3D FDTD RF notebook for a CPW structure."""
        if isinstance(clicked_component, bool):
            clicked_component = None
        if clicked_component is None:
            selected = self.selected_components()
            clicked_component = selected[0] if selected else None
        target_component = self.rf_lumerical_target_component(clicked_component)
        if target_component is None:
            QMessageBox.information(
                self,
                "Lumerical RF",
                "Select a CPW, CPW taper, CPW bend/open/short, segmented electrode, or one of its RF planes.",
            )
            return
        saved = copy.deepcopy(target_component.get("lumerical_rf_export_settings", {}))
        dialog = RFLumericalExportDialog(
            self.components,
            target_component,
            self.rf_lumerical_scope_options(target_component),
            saved=saved,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            configuration = dialog.configuration()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical RF settings", str(exc))
            return
        selected_uids = set(map(int, configuration.get("scope_uids", [])))
        selected_uids.add(int(target_component["uid"]))
        export_components = [
            safe_json_copy(component) for component in self.components
            if int(component.get("uid", -1)) in selected_uids
        ]
        base_kind = re.sub(
            r"[^A-Za-z0-9_-]+", "_", str(target_component.get("kind", "cpw"))
        ).strip("_").lower() or "cpw"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lumerical RF notebook",
            str(Path.home() / f"{base_kind}_rf.ipynb"),
            "Jupyter notebook (*.ipynb)",
        )
        if not path:
            return
        if not path.lower().endswith(".ipynb"):
            path += ".ipynb"
        try:
            warnings = write_lumerical_rf_notebook(path, export_components, configuration)
        except Exception as exc:
            QMessageBox.critical(self, "Lumerical RF export failed", str(exc))
            return
        snapshot = self.snapshot()
        target_component["lumerical_rf_export_settings"] = safe_json_copy(configuration)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)
        workflow = str(configuration.get("rf_workflow", "rf")).upper()
        message = f"Exported {workflow} Lumerical RF notebook: {path}"
        if warnings:
            message += f" ({len(warnings)} note(s) are recorded inside the notebook)"
        self.statusBar().showMessage(message, 10000)

    def lumerical_scope_options(self, clicked_component: dict[str, Any] | None) -> list[tuple[str, list[int]]]:
        """Geometry choices shown after a component-oriented simulation export."""
        options: list[tuple[str, list[int]]] = []
        seen: set[tuple[int, ...]] = set()

        def add(label: str, members: list[dict[str, Any]]) -> None:
            ordered_uids: list[int] = []
            member_seen: set[int] = set()
            for component in members:
                uid = int(component["uid"])
                if component.get("kind") == "E-beam multipass" or uid in member_seen:
                    continue
                member_seen.add(uid)
                ordered_uids.append(uid)
            uids = tuple(ordered_uids)
            if not uids or uids in seen:
                return
            seen.add(uids)
            options.append((label, list(uids)))

        selected = self.selected_components()
        if clicked_component is not None:
            clicked_group_id = clicked_component.get("group_id")
            if clicked_group_id:
                add(
                    f"Component with its automatic simulation setup — {clicked_group_id}",
                    [component for component in self.components if component.get("group_id") == clicked_group_id],
                )
            placed_simulation = [component for component in self.components if component.get("kind") in SIMULATION_COMPONENT_KINDS]
            physical_components = [
                component for component in self.components
                if component.get("kind") not in SIMULATION_COMPONENT_KINDS
                and component.get("kind") != "E-beam multipass"
            ]
            if clicked_component.get("kind") in SIMULATION_COMPONENT_KINDS and physical_components:
                selected_physical = [
                    component for component in selected
                    if component.get("kind") not in SIMULATION_COMPONENT_KINDS
                    and component.get("kind") != "E-beam multipass"
                ]
                if selected_physical:
                    add(
                        f"Selected device geometry + {len(placed_simulation)} placed port/monitor object(s)",
                        [*selected_physical, *placed_simulation],
                    )

                clicked_x = float(clicked_component.get("x", 0.0))
                clicked_scene_y = -float(clicked_component.get("y", 0.0))

                def device_distance(component: dict[str, Any]) -> float:
                    item = self.items_by_uid.get(int(component["uid"]))
                    if item is not None:
                        bounds = item.sceneBoundingRect()
                        dx = max(float(bounds.left()) - clicked_x, 0.0, clicked_x - float(bounds.right()))
                        dy = max(float(bounds.top()) - clicked_scene_y, 0.0, clicked_scene_y - float(bounds.bottom()))
                        return math.hypot(dx, dy)
                    return math.hypot(
                        float(component.get("x", 0.0)) - clicked_x,
                        float(component.get("y", 0.0)) - float(clicked_component.get("y", 0.0)),
                    )

                nearest = min(physical_components, key=device_distance)
                add(
                    f"Nearest device ({nearest.get('kind')}, UID {nearest.get('uid')}) + {len(placed_simulation)} placed port/monitor object(s)",
                    [nearest, *placed_simulation],
                )
            elif placed_simulation:
                add(
                    f"Clicked component + {len(placed_simulation)} placed port/monitor object(s)",
                    [clicked_component, *placed_simulation],
                )
            clicked_only_label = f"Clicked component only — {clicked_component.get('kind')} (UID {clicked_component.get('uid')})"
            if clicked_component.get("kind") in SIMULATION_COMPONENT_KINDS:
                clicked_only_label += " — no device geometry"
            add(clicked_only_label, [clicked_component])
        if selected:
            add(f"Current selection — {len(selected)} component(s)", selected)
        if clicked_component is not None:
            for key, title in (
                ("module_instance_id", "Complete module"),
                ("group_id", "Complete group"),
                ("array_group_id", "Complete array"),
            ):
                value = clicked_component.get(key)
                if value:
                    add(f"{title} — {value}", [component for component in self.components if component.get(key) == value])
        add(f"Entire layout — {len(self.components)} component(s)", self.components)
        return options

    def export_lumerical_notebook(self, clicked_component: dict[str, Any] | bool | None = None) -> None:
        if isinstance(clicked_component, bool):
            clicked_component = None
        if not self.components:
            QMessageBox.information(self, "Lumerical notebook", "Add at least one component before exporting.")
            return
        if clicked_component is None:
            selected = self.selected_components()
            clicked_component = selected[0] if selected else None
        rf_target = self.rf_lumerical_target_component(clicked_component)
        if rf_target is not None:
            self.export_lumerical_rf_notebook(rf_target)
            return
        if clicked_component is not None and str(clicked_component.get("kind", "")) in RF_COMPONENT_KINDS:
            QMessageBox.information(
                self,
                "Lumerical RF",
                "This RF container is not a direct solver target. Select one of its CPW, taper, bend, open/short, or segmented-electrode children.",
            )
            return
        settings_component = clicked_component
        if clicked_component and clicked_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            parent_uid = clicked_component.get("simulation_parent_uid")
            settings_component = next(
                (component for component in self.components if int(component.get("uid", -1)) == int(parent_uid)),
                clicked_component,
            ) if parent_uid is not None else clicked_component
        saved = settings_component.get("lumerical_export_settings", {}) if settings_component else {}
        if settings_component and settings_component.get("kind") == "1x2 MMI":
            saved = mmi_lumerical_export_settings(saved)
        if settings_component and settings_component.get("kind") == "Grating coupler":
            saved = grating_lumerical_export_settings(saved)
        if settings_component and settings_component.get("kind") == "GC-SOI" and saved:
            saved = copy.deepcopy(saved)
            # Migrate the first compact-domain preset, which cropped the
            # terminal arc and enabled half-width symmetry by default. Keep
            # later user-edited bounds untouched.
            if int(saved.get("gc_domain_version", 0)) < 3:
                saved.pop("domain_padding_um", None)
                saved["official_gc_domain"] = True
                saved["use_y_antisymmetry"] = False
                saved["gc_domain_version"] = 3
        if settings_component and settings_component.get("kind") == "GC-SOI" and not saved:
            saved = {
                "stack_preset": "SOI grating coupler (Ansys)",
                "material_stack": default_stack("SOI grating coupler (Ansys)"),
                "wavelength_start_um": 1.50,
                "wavelength_stop_um": 1.60,
                "frequency_points": 31,
                "resource_mode": "GPU",
                "dimension": "3D",
                "official_gc_domain": True,
                "use_y_antisymmetry": False,
                "gc_domain_version": 3,
            }
        dialog = LumericalExportDialog(
            self.components,
            self.lumerical_scope_options(clicked_component),
            saved=saved,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        configuration = dialog.configuration()
        selected_uids = set(map(int, configuration.get("scope_uids", [])))
        export_components = [safe_json_copy(component) for component in self.components if int(component["uid"]) in selected_uids]
        base_name = "lumerical_simulation"
        primary_geometry = next(
            (component for component in export_components if component.get("kind") not in SIMULATION_COMPONENT_KINDS),
            None,
        )
        naming_component = primary_geometry or clicked_component
        if naming_component is not None:
            base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(naming_component.get("kind", "component"))).strip("_").lower() or "component"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lumerical notebook",
            str(Path.home() / f"{base_name}.ipynb"),
            "Jupyter notebook (*.ipynb)",
        )
        if not path:
            return
        if not path.lower().endswith(".ipynb"):
            path += ".ipynb"
        try:
            warnings = write_lumerical_notebook(path, export_components, configuration)
        except Exception as exc:
            QMessageBox.critical(self, "Notebook export failed", str(exc))
            return

        snapshot = self.snapshot()
        if settings_component is not None:
            settings_component["lumerical_export_settings"] = safe_json_copy(configuration)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)
        message = f"Exported self-contained Lumerical notebook: {path}"
        if warnings:
            message += f" ({len(warnings)} note(s) are recorded inside the notebook)"
        self.statusBar().showMessage(message, 10000)

    def export_lumerical_sweep_notebook(
        self, clicked_component: dict[str, Any] | bool | None = None
    ) -> None:
        """Export a one-session Cartesian geometry sweep for one physical component."""
        if isinstance(clicked_component, bool):
            clicked_component = None
        if clicked_component is None:
            selected = self.selected_components()
            clicked_component = selected[0] if selected else None
        if clicked_component is None:
            QMessageBox.information(
                self, "Lumerical sweep", "Right-click the physical component whose parameters you want to sweep."
            )
            return
        target_component = clicked_component
        if target_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            parent_uid = target_component.get("simulation_parent_uid")
            target_component = next(
                (
                    component for component in self.components
                    if parent_uid is not None and int(component.get("uid", -1)) == int(parent_uid)
                ),
                None,
            )
        if target_component is None or target_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            QMessageBox.information(
                self,
                "Lumerical sweep",
                "A port or monitor cannot be the sweep target. Right-click its parent device instead.",
            )
            return
        if str(target_component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS:
            QMessageBox.information(
                self,
                "Lumerical RF sweep",
                "RF parameter sweeps require RF-specific S-parameter/impedance objectives. Export a Lumerical RF run first; the optical sweep exporter is intentionally not used for CPW devices.",
            )
            return
        if target_component.get("kind") in {"Grating coupler", "GC-SOI"}:
            # Ensure the sweep dialog and every generated case expose only the
            # canonical project-JSON name, including layouts made by older
            # versions of the editor.
            migrate_grating_fiber_offset_parameter(target_component)

        sweep_dialog = LumericalSweepDialog(
            target_component,
            saved=target_component.get("lumerical_sweep_settings", {}),
            parent=self,
        )
        if not sweep_dialog.parameters:
            QMessageBox.information(
                self,
                "Lumerical sweep",
                f"{target_component.get('kind')} has no supported scalar geometry parameters to sweep.",
            )
            return
        if sweep_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sweep_spec = sweep_dialog.sweep_spec()

        saved_export = copy.deepcopy(target_component.get("lumerical_export_settings", {}))
        if target_component.get("kind") == "1x2 MMI":
            saved_export = mmi_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "Grating coupler":
            saved_export = grating_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "GC-SOI" and saved_export:
            # Apply the same compact-domain migration as the one-run export.
            # Old saved bounds could crop the terminal arc during a sweep.
            if int(saved_export.get("gc_domain_version", 0)) < 3:
                saved_export.pop("domain_padding_um", None)
                saved_export["official_gc_domain"] = True
                saved_export["use_y_antisymmetry"] = False
                saved_export["gc_domain_version"] = 3
        if target_component.get("kind") == "GC-SOI" and not saved_export:
            saved_export = {
                "stack_preset": "SOI grating coupler (Ansys)",
                "material_stack": default_stack("SOI grating coupler (Ansys)"),
                "wavelength_start_um": 1.50,
                "wavelength_stop_um": 1.60,
                "frequency_points": 31,
                "resource_mode": "GPU",
                "dimension": "3D",
                "official_gc_domain": True,
                "use_y_antisymmetry": False,
                "gc_domain_version": 3,
                "run_after_build": True,
            }
        export_dialog = LumericalExportDialog(
            self.components,
            self.lumerical_scope_options(target_component),
            saved=saved_export,
            parent=self,
        )
        export_dialog.setWindowTitle("Lumerical sweep — stack, domain, and GPU settings")
        if export_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            configuration = export_dialog.configuration()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical sweep settings", str(exc))
            return
        selected_uids = set(map(int, configuration.get("scope_uids", [])))
        export_components = [
            safe_json_copy(component)
            for component in self.components
            if int(component["uid"]) in selected_uids
        ]
        target_uid = int(target_component["uid"])
        if not any(int(component.get("uid", -1)) == target_uid for component in export_components):
            QMessageBox.critical(
                self,
                "Invalid Lumerical sweep scope",
                "The selected export geometry does not contain the component being swept.",
            )
            return

        class _SweepCompanionContext:
            make_component = NativeLayoutWindow.make_component
            synchronize_automatic_simulation_companions = (
                NativeLayoutWindow.synchronize_automatic_simulation_companions
            )

        sweep_cases = []
        for values in expand_lumerical_sweep_points(sweep_spec):
            variant_components = copy.deepcopy(export_components)
            variant_target = next(
                component for component in variant_components
                if int(component.get("uid", -1)) == target_uid
            )
            apply_lumerical_sweep_values(variant_target, values)
            context = _SweepCompanionContext()
            context.components = variant_components
            context.next_uid = 1 + max(
                (int(component.get("uid", 0)) for component in variant_components),
                default=0,
            )
            context.synchronize_automatic_simulation_companions(variant_target)
            sweep_cases.append(
                {"values": dict(values), "components": context.components}
            )

        base_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(target_component.get("kind", "component")),
        ).strip("_").lower() or "component"
        default_path = Path.home() / f"{base_name}_sweep.ipynb"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save optimized Lumerical sweep notebook",
            str(default_path),
            "Jupyter notebook (*.ipynb)",
        )
        if not path:
            return
        if not path.lower().endswith(".ipynb"):
            path += ".ipynb"
        configuration["lumerical_sweep"] = sweep_spec
        configuration["run_after_build"] = True
        configuration["resource_mode"] = "GPU"
        configuration["project_file"] = f"{base_name}_sweep.fsp"
        try:
            warnings = write_lumerical_sweep_notebook(
                path,
                sweep_cases,
                configuration,
                sweep_spec,
                nominal_components=export_components,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Lumerical sweep export failed", str(exc))
            return

        snapshot = self.snapshot()
        target_component["lumerical_sweep_settings"] = safe_json_copy(sweep_spec)
        target_component["lumerical_export_settings"] = safe_json_copy(configuration)
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)
        message = (
            f"Exported {sweep_spec['point_count']}-point optimized Lumerical sweep: {path}. "
            "One live model; no per-point FSP saves."
        )
        if warnings:
            message += f" ({len(warnings)} export note(s))"
        self.statusBar().showMessage(message, 12000)

    def export_lumerical_multigpu_sweep_notebook(
        self, clicked_component: dict[str, Any] | bool | None = None
    ) -> None:
        """Export an isolated Cartesian sweep distributed across A100 workers."""
        title = "Lumerical sweep-multithread"
        if isinstance(clicked_component, bool):
            clicked_component = None
        if clicked_component is None:
            selected = self.selected_components()
            clicked_component = selected[0] if selected else None
        if clicked_component is None:
            QMessageBox.information(
                self,
                title,
                "Right-click the physical component whose parameters you want to sweep.",
            )
            return
        target_component = clicked_component
        if target_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            parent_uid = target_component.get("simulation_parent_uid")
            target_component = next(
                (
                    component
                    for component in self.components
                    if parent_uid is not None
                    and int(component.get("uid", -1)) == int(parent_uid)
                ),
                None,
            )
        if target_component is None or target_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            QMessageBox.information(
                self,
                title,
                "A port or monitor cannot be the sweep target. Right-click its parent device instead.",
            )
            return
        if str(target_component.get("kind", "")) in RF_SIMULATABLE_COMPONENT_KINDS:
            QMessageBox.information(
                self,
                "Lumerical RF sweep-multithread",
                "The optical multi-GPU sweep is intentionally disabled for CPW devices. Start with the dedicated Lumerical RF run so its MODE/FDTD ports, materials, and S-parameter normalization are used.",
            )
            return
        if target_component.get("kind") in {"Grating coupler", "GC-SOI"}:
            migrate_grating_fiber_offset_parameter(target_component)

        sweep_dialog = LumericalMultigpuSweepDialog(
            target_component,
            saved=target_component.get("lumerical_multigpu_sweep_settings", {}),
            parent=self,
        )
        if not sweep_dialog.parameters:
            QMessageBox.information(
                self,
                title,
                f"{target_component.get('kind')} has no supported scalar geometry parameters to sweep.",
            )
            return
        if sweep_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sweep_spec = sweep_dialog.sweep_spec()
        multigpu_configuration = sweep_dialog.parallel_configuration()

        saved_export = copy.deepcopy(
            target_component.get("lumerical_multigpu_export_settings", {})
        )
        if target_component.get("kind") == "1x2 MMI":
            saved_export = mmi_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "Grating coupler":
            saved_export = grating_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "GC-SOI" and saved_export:
            if int(saved_export.get("gc_domain_version", 0)) < 3:
                saved_export.pop("domain_padding_um", None)
                saved_export["official_gc_domain"] = True
                saved_export["use_y_antisymmetry"] = False
                saved_export["gc_domain_version"] = 3
        if target_component.get("kind") == "GC-SOI" and not saved_export:
            saved_export = {
                "stack_preset": "SOI grating coupler (Ansys)",
                "material_stack": default_stack("SOI grating coupler (Ansys)"),
                "wavelength_start_um": 1.50,
                "wavelength_stop_um": 1.60,
                "frequency_points": 31,
                "resource_mode": "GPU",
                "dimension": "3D",
                "official_gc_domain": True,
                "use_y_antisymmetry": False,
                "gc_domain_version": 3,
                "run_after_build": True,
            }
        export_dialog = LumericalExportDialog(
            self.components,
            self.lumerical_scope_options(target_component),
            saved=saved_export,
            parent=self,
        )
        export_dialog.setWindowTitle(
            "Lumerical sweep-multithread — stack parameters, domain, and GPU settings"
        )
        if export_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            configuration = export_dialog.configuration()
        except Exception as exc:
            QMessageBox.critical(self, f"Invalid {title} settings", str(exc))
            return
        selected_uids = set(map(int, configuration.get("scope_uids", [])))
        export_components = [
            safe_json_copy(component)
            for component in self.components
            if int(component["uid"]) in selected_uids
        ]
        target_uid = int(target_component["uid"])
        if not any(
            int(component.get("uid", -1)) == target_uid
            for component in export_components
        ):
            QMessageBox.critical(
                self,
                f"Invalid {title} scope",
                "The selected export geometry does not contain the component being swept.",
            )
            return

        class _MultigpuSweepCompanionContext:
            make_component = NativeLayoutWindow.make_component
            synchronize_automatic_simulation_companions = (
                NativeLayoutWindow.synchronize_automatic_simulation_companions
            )

        sweep_cases = []
        for values in expand_lumerical_sweep_points(sweep_spec):
            variant_components = copy.deepcopy(export_components)
            variant_target = next(
                component
                for component in variant_components
                if int(component.get("uid", -1)) == target_uid
            )
            apply_lumerical_sweep_values(variant_target, values)
            context = _MultigpuSweepCompanionContext()
            context.components = variant_components
            context.next_uid = 1 + max(
                (int(component.get("uid", 0)) for component in variant_components),
                default=0,
            )
            context.synchronize_automatic_simulation_companions(variant_target)
            sweep_cases.append({"values": dict(values), "components": context.components})

        base_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(target_component.get("kind", "component")),
        ).strip("_").lower() or "component"
        default_path = Path.home() / f"{base_name}_sweep_multithread.ipynb"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save multi-GPU Lumerical sweep notebook",
            str(default_path),
            "Jupyter notebook (*.ipynb)",
        )
        if not path:
            return
        if not path.lower().endswith(".ipynb"):
            path += ".ipynb"
        configuration["lumerical_sweep"] = sweep_spec
        configuration["lumerical_multigpu"] = multigpu_configuration
        configuration["run_after_build"] = True
        configuration["resource_mode"] = "GPU"
        configuration["project_file"] = f"{base_name}_sweep_multithread.fsp"
        try:
            warnings = write_lumerical_multigpu_sweep_notebook(
                path,
                sweep_cases,
                configuration,
                sweep_spec,
                nominal_components=export_components,
            )
        except Exception as exc:
            QMessageBox.critical(self, f"{title} export failed", str(exc))
            return

        saved_sweep = safe_json_copy(sweep_spec)
        saved_sweep["parallel"] = safe_json_copy(multigpu_configuration)
        snapshot = self.snapshot()
        target_component["lumerical_multigpu_sweep_settings"] = saved_sweep
        target_component["lumerical_multigpu_export_settings"] = safe_json_copy(
            configuration
        )
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)
        slots = int(multigpu_configuration["max_parallel_simulations"])
        message = (
            f"Exported {sweep_spec['point_count']}-point {title}: {path}. "
            f"Up to {slots} independent A100 worker(s); the sequential sweep export is unchanged."
        )
        if warnings:
            message += f" ({len(warnings)} export note(s))"
        self.statusBar().showMessage(message, 14000)

    def export_lumerical_optimization_notebook(
        self, clicked_component: dict[str, Any] | bool | None = None
    ) -> None:
        """Export a two-page 3D LumOpt shape-adjoint optimization notebook."""
        title = "Lumerical adjoint optimization"
        supported_kinds = {"Grating coupler", "GC-SOI", "1x2 MMI"}
        if isinstance(clicked_component, bool):
            clicked_component = None
        if clicked_component is None:
            selected = self.selected_components()
            clicked_component = selected[0] if selected else None
        if clicked_component is None:
            QMessageBox.information(
                self,
                title,
                "Right-click a grating coupler or 1x2 MMI whose geometry you want to optimize.",
            )
            return

        target_component = clicked_component
        if target_component.get("kind") in SIMULATION_COMPONENT_KINDS:
            parent_uid = target_component.get("simulation_parent_uid")
            target_component = next(
                (
                    component
                    for component in self.components
                    if parent_uid is not None
                    and int(component.get("uid", -1)) == int(parent_uid)
                ),
                None,
            )
        if target_component is None or str(target_component.get("kind", "")) not in supported_kinds:
            QMessageBox.information(
                self,
                title,
                "Shape-adjoint export currently supports Grating coupler, GC-SOI, and symmetric 1x2 MMI components only.",
            )
            return
        if target_component.get("kind") in {"Grating coupler", "GC-SOI"}:
            migrate_grating_fiber_offset_parameter(target_component)

        saved_spec = copy.deepcopy(
            target_component.get("lumerical_optimization_settings", {})
        )
        if not saved_spec:
            previous_export = dict(
                target_component.get("lumerical_optimization_export_settings", {})
                or target_component.get("lumerical_export_settings", {})
            )
            default_center_um = (
                1.55 if target_component.get("kind") == "GC-SOI" else 1.30
            )
            wavelength_start_um = float(
                previous_export.get("wavelength_start_um", default_center_um - 0.05)
            )
            wavelength_stop_um = float(
                previous_export.get("wavelength_stop_um", default_center_um + 0.05)
            )
            if wavelength_stop_um < wavelength_start_um:
                wavelength_start_um, wavelength_stop_um = (
                    wavelength_stop_um,
                    wavelength_start_um,
                )
            saved_spec = {
                "objective": {
                    "center_wavelength_um": 0.5
                    * (wavelength_start_um + wavelength_stop_um),
                    "bandwidth_nm": 1000.0
                    * (wavelength_stop_um - wavelength_start_um),
                    "wavelength_points": max(
                        1, int(previous_export.get("frequency_points", 7))
                    ),
                },
                "optimizer": {"max_iterations": 30},
            }

        optimization_dialog = LumericalOptimizationDialog(
            target_component,
            saved=saved_spec,
            parent=self,
        )
        if not optimization_dialog.parameters:
            QMessageBox.information(
                self,
                title,
                f"{target_component.get('kind')} has no supported continuous shape-adjoint parameters.",
            )
            return
        if optimization_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            optimization_spec = optimization_dialog.optimization_spec()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical optimization", str(exc))
            return

        objective = dict(optimization_spec.get("objective", {}))
        if target_component.get("kind") == "1x2 MMI":
            objective.setdefault("identifier", "mmi_top_output_over_input")
            objective.setdefault(
                "description",
                "Top/upper output branch power divided by input power",
            )
        else:
            objective.setdefault("identifier", "grating_coupling_efficiency")
            objective.setdefault(
                "description",
                "Fiber-to-waveguide coupling efficiency",
            )
        optimization_spec["objective"] = objective
        center_wavelength_um = float(
            objective.get(
                "center_wavelength_um",
                saved_spec.get("objective", {}).get("center_wavelength_um", 1.30),
            )
        )
        bandwidth_nm = max(
            0.0,
            float(
                objective.get(
                    "bandwidth_nm",
                    saved_spec.get("objective", {}).get("bandwidth_nm", 0.0),
                )
            ),
        )
        wavelength_start_um = float(
            objective.get(
                "wavelength_start_um",
                center_wavelength_um - 0.0005 * bandwidth_nm,
            )
        )
        wavelength_stop_um = float(
            objective.get(
                "wavelength_stop_um",
                center_wavelength_um + 0.0005 * bandwidth_nm,
            )
        )
        wavelength_points = max(
            1,
            int(
                objective.get(
                    "wavelength_points",
                    saved_spec.get("objective", {}).get("wavelength_points", 7),
                )
            ),
        )

        saved_export = copy.deepcopy(
            target_component.get("lumerical_optimization_export_settings", {})
            or target_component.get("lumerical_export_settings", {})
        )
        if target_component.get("kind") == "1x2 MMI":
            saved_export = mmi_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "Grating coupler":
            saved_export = grating_lumerical_export_settings(saved_export)
        if target_component.get("kind") == "GC-SOI" and saved_export:
            if int(saved_export.get("gc_domain_version", 0)) < 3:
                saved_export.pop("domain_padding_um", None)
                saved_export["official_gc_domain"] = True
                saved_export["use_y_antisymmetry"] = False
                saved_export["gc_domain_version"] = 3
        if target_component.get("kind") == "GC-SOI" and not saved_export:
            saved_export = {
                "stack_preset": "SOI grating coupler (Ansys)",
                "material_stack": default_stack("SOI grating coupler (Ansys)"),
                "official_gc_domain": True,
                "use_y_antisymmetry": False,
                "gc_domain_version": 3,
            }
        saved_export.update(
            {
                "wavelength_start_um": wavelength_start_um,
                "wavelength_stop_um": wavelength_stop_um,
                "frequency_points": wavelength_points,
                "resource_mode": "GPU",
                "dimension": "3D",
                "run_after_build": True,
            }
        )

        export_dialog = LumericalExportDialog(
            self.components,
            self.lumerical_scope_options(target_component),
            saved=saved_export,
            parent=self,
        )
        export_dialog.setWindowTitle(
            "Lumerical adjoint optimization — stack, domain, and GPU settings"
        )
        # Page one owns the objective spectrum. Keep page two visually explicit
        # without allowing a conflicting wavelength or resource selection.
        export_dialog.wavelength_start.setValue(wavelength_start_um)
        export_dialog.wavelength_stop.setValue(wavelength_stop_um)
        export_dialog.frequency_points.setValue(wavelength_points)
        for widget in (
            export_dialog.wavelength_start,
            export_dialog.wavelength_stop,
            export_dialog.frequency_points,
        ):
            widget.setEnabled(False)
            widget.setToolTip(
                "Controlled by the center wavelength and bandwidth on the optimization page."
            )
        export_dialog.resource_mode.setCurrentText("GPU")
        export_dialog.resource_mode.setEnabled(False)
        export_dialog.resource_mode.setToolTip(
            "Forward and adjoint 3D electromagnetic solves run on the GPU."
        )
        export_dialog.run_after_build.setChecked(True)
        export_dialog.run_after_build.setEnabled(False)
        if export_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            configuration = export_dialog.configuration()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Lumerical optimization settings", str(exc))
            return
        configuration.update(
            {
                "wavelength_start_um": wavelength_start_um,
                "wavelength_stop_um": wavelength_stop_um,
                "frequency_points": wavelength_points,
                "resource_mode": "GPU",
                "dimension": "3D",
                "run_after_build": True,
                "lumerical_optimization": safe_json_copy(optimization_spec),
            }
        )

        selected_uids = set(map(int, configuration.get("scope_uids", [])))
        export_components = [
            safe_json_copy(component)
            for component in self.components
            if int(component["uid"]) in selected_uids
        ]
        target_uid = int(target_component["uid"])
        if not any(
            int(component.get("uid", -1)) == target_uid
            for component in export_components
        ):
            QMessageBox.critical(
                self,
                "Invalid Lumerical optimization scope",
                "The selected export geometry does not contain the component being optimized.",
            )
            return

        base_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(target_component.get("kind", "component")),
        ).strip("_").lower() or "component"
        configuration["project_file"] = f"{base_name}_optimized.fsp"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lumerical adjoint optimization notebook",
            str(Path.home() / f"{base_name}_optimization.ipynb"),
            "Jupyter notebook (*.ipynb)",
        )
        if not path:
            return
        if not path.lower().endswith(".ipynb"):
            path += ".ipynb"
        try:
            warnings = write_lumerical_adjoint_notebook(
                path,
                export_components,
                configuration,
                optimization_spec,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Lumerical optimization export failed", str(exc))
            return

        snapshot = self.snapshot()
        target_component["lumerical_optimization_settings"] = safe_json_copy(
            optimization_spec
        )
        target_component["lumerical_optimization_export_settings"] = safe_json_copy(
            configuration
        )
        self.commit_interaction_snapshot(snapshot)
        self.rebuild_scene(preserve_selection=True)
        message = f"Exported Lumerical shape-adjoint optimization notebook: {path}"
        if warnings:
            message += f" ({len(warnings)} export note(s))"
        self.statusBar().showMessage(message, 14000)

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

    def run_llm_api_test(self) -> None:
        """Test cloud authentication/model access without modifying the project."""
        if self.llm_process is not None:
            self.append_llm_chat("error", "An LLM request is already running.")
            return
        request_fd, request_name = tempfile.mkstemp(prefix="photonic_llm_test_", suffix=".json")
        response_fd, response_name = tempfile.mkstemp(prefix="photonic_llm_test_response_", suffix=".json")
        os.close(request_fd);os.close(response_fd)
        self.llm_request_file=Path(request_name);self.llm_response_file=Path(response_name);self.llm_response_file.write_text("")
        self.llm_request_file.write_text(json.dumps({"task":"test","model":self.llm_model.text().strip() or "gpt-5.6","api_key":self.llm_api_key.text().strip()}))
        process=QProcess(self);self.llm_process=process;self.llm_send_button.setEnabled(False);self.llm_test_button.setEnabled(False)
        process.setProgram(sys.executable);process.setArguments([launcher_path(),"--worker-llm",str(self.llm_request_file),str(self.llm_response_file)])
        process.finished.connect(self.llm_request_finished);process.start();self.append_llm_chat("assistant","Testing OpenAI API connection…");self.statusBar().showMessage("Testing OpenAI API…")

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
            if response.get("test"):
                self.append_llm_chat("assistant", str(response.get("message") or "OpenAI API connection successful."))
            elif "plan" in response:
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
            self.llm_test_button.setEnabled(True)
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
            "pitch", "fill_factor", "duty_cycle", "fill_factors", "tooth_shape", "N", "target_length",
            "alpha_t", "taper_L", "L_extra", "wg_width", "wg_length",
            "gc_pitch", "gc_fill_factor", "gc_fill_factors", "gc_N", "gc_alpha_t", "gc_taper_L", "gc_wg_length",
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

        if kind in {"Grating coupler", "GC-SOI"}:
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
            self.properties_form.addRow("GC straight lead length (µm)" if key=="gc_wg_length" else "CPW taper model" if key=="cpw_profile" else key.replace("_", " "), widget)
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
        component = self.component_by_uid(component_item.uid)
        if component is None:
            return
        # A generated scan is one selectable parent containing many devices.
        # Its normal selection panel already exposes the block-level controls;
        # resolving a child polygon here is both ambiguous and unsafe while a
        # very large QGraphicsPath has just handled the mouse release.
        if component.get("kind") in {"RF test block", "Photonic test block"}:
            self.show_component_properties(component)
            return
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
            form.addRow("GC straight lead length (µm)" if key=="gc_wg_length" else "CPW taper model" if key=="cpw_profile" else key.replace("_", " "), widget)
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
        scan_action = None
        scan_component = None
        clicked_component = None
        export_lumerical_action = None
        sweep_lumerical_action = None
        multigpu_sweep_lumerical_action = None
        optimize_lumerical_action = None
        if component_item is not None and component_item.data(10) is not None:
            component = self.component_by_uid(int(component_item.data(10)))
            if component is not None:
                clicked_component = component
                component_item.setSelected(True)
                if component.get("kind") in {"RF test block", "Photonic test block"}:
                    scan_component = component
                    scan_action = menu.addAction("Edit scan ranges and defaults…")
                    scan_font = scan_action.font();scan_font.setBold(True);scan_action.setFont(scan_font)
                    menu.addSeparator()
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
                rf_target = self.rf_lumerical_target_component(component)
                export_lumerical_action = menu.addAction("Lumerical run…")
                if rf_target is not None:
                    export_lumerical_action.setText("Lumerical RF run…")
                export_font = export_lumerical_action.font(); export_font.setBold(True); export_lumerical_action.setFont(export_font)
                if rf_target is None:
                    sweep_lumerical_action = menu.addAction("Lumerical sweep…")
                    multigpu_sweep_lumerical_action = menu.addAction("Lumerical sweep-multithread…")
                    optimize_lumerical_action = menu.addAction("Lumerical optimization…")
                menu.addSeparator()
        save_module_action = menu.addAction("Add selection to User modules…")
        save_module_action.setEnabled(bool([component for component in self.selected_components() if component.get("kind") != "E-beam multipass"]))
        lattice_action=menu.addAction("Create photonic-crystal lattice…");lattice_action.setEnabled(save_module_action.isEnabled())
        boolean_menu=menu.addMenu("Boolean operation");boolean_actions={boolean_menu.addAction(label):op for label,op in (("Union","union"),("Difference (first minus rest)","difference"),("Intersection","intersection"),("XOR","xor"))};boolean_menu.setEnabled(len(self.selected_components())>=2)
        delete_action = menu.addAction("Delete Selected")
        delete_action.setEnabled(bool(self.scene.selectedItems()) or bool(self.active_field))
        menu.addSeparator()
        menu.addAction("Fit Selection", self.fit_selection)
        menu.addAction("Fit Drawn Region…", self.start_fit_drawn_region)
        menu.addAction("Fit Design", self.fit_design)
        chosen = menu.exec(self.view.viewport().mapToGlobal(viewport_position))
        if scan_action is not None and chosen is scan_action:
            self.edit_test_block_scan(scan_component)
        elif chosen in section_actions:
            self.edit_component_section(*section_actions[chosen])
        elif chosen is export_lumerical_action and clicked_component is not None:
            self.export_lumerical_notebook(clicked_component)
        elif chosen is sweep_lumerical_action and clicked_component is not None:
            self.export_lumerical_sweep_notebook(clicked_component)
        elif chosen is multigpu_sweep_lumerical_action and clicked_component is not None:
            self.export_lumerical_multigpu_sweep_notebook(clicked_component)
        elif chosen is optimize_lumerical_action and clicked_component is not None:
            self.export_lumerical_optimization_notebook(clicked_component)
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
