"""Array, e-beam, and module-variable dialogs."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QScrollArea, QSpinBox, QVBoxLayout, QWidget

from ..constants import COMPONENT_SPECS


class ArrayDialog(QDialog):
    def __init__(self, param_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create editable array")
        form = QFormLayout(self)
        self.nx = QSpinBox()
        self.nx.setRange(1, 1000)
        self.nx.setValue(2)
        self.ny = QSpinBox()
        self.ny.setRange(1, 1000)
        self.ny.setValue(1)
        self.dx = QLineEdit("1000")
        self.dy = QLineEdit("1000")
        sequence_help = (
            "Accepted: one value; comma, space, or semicolon lists; "
            "[a, b, c]; (start, step, stop); start:step:stop; "
            "or linspace(start, stop, count)."
        )
        self.dx.setPlaceholderText("1000 or [1000, 1200, 1400]")
        self.dy.setPlaceholderText("1000 or [1000, 1200, 1400]")
        self.dx.setToolTip(sequence_help)
        self.dy.setToolTip(sequence_help)
        self.x_param = QComboBox()
        self.y_param = QComboBox()
        self.x_param.addItem("")
        self.y_param.addItem("")
        self.x_param.addItems(param_names)
        self.y_param.addItems(param_names)
        self.x_values = QLineEdit("")
        self.y_values = QLineEdit("")
        self.x_values.setPlaceholderText("Example: [1, 2, 3] or linspace(1, 3, 3)")
        self.y_values.setPlaceholderText("Example: [1, 2, 3] or linspace(1, 3, 3)")
        self.x_values.setToolTip(sequence_help)
        self.y_values.setToolTip(sequence_help)
        self.auto_label = QCheckBox("Add sweep label at top-left of each array cell")
        self.label_prefix = QLineEdit("A")
        self.label_offset_x = QDoubleSpinBox()
        self.label_offset_x.setRange(-1e9, 1e9)
        self.label_offset_x.setValue(20.0)
        self.label_offset_y = QDoubleSpinBox()
        self.label_offset_y.setRange(-1e9, 1e9)
        self.label_offset_y.setValue(20.0)
        form.addRow("Nx", self.nx)
        form.addRow("Ny", self.ny)
        form.addRow("Δx or spacing list", self.dx)
        form.addRow("Δy or spacing list", self.dy)
        form.addRow("X sweep parameter", self.x_param)
        form.addRow("X values", self.x_values)
        form.addRow("Y sweep parameter", self.y_param)
        form.addRow("Y values", self.y_values)
        form.addRow(self.auto_label)
        form.addRow("Label prefix", self.label_prefix)
        form.addRow("Label X offset", self.label_offset_x)
        form.addRow("Label Y offset", self.label_offset_y)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class EbeamDialog(QDialog):
    def __init__(self, defaults: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("E-beam write-field coverage")
        form = QFormLayout(self)
        self.field_size = QDoubleSpinBox()
        self.field_size.setRange(0.001, 1e9)
        self.field_size.setDecimals(6)
        self.field_size.setValue(float(defaults.get("field_size", 520.0)))
        self.clearance = QDoubleSpinBox()
        self.clearance.setRange(0, 1e9)
        self.clearance.setDecimals(6)
        self.clearance.setValue(float(defaults.get("edge_clearance", 10.0)))
        self.overlap_x = QDoubleSpinBox()
        self.overlap_x.setRange(0, 99.999)
        self.overlap_x.setValue(float(defaults.get("overlap_x_percent", 0.0)))
        self.overlap_y = QDoubleSpinBox()
        self.overlap_y.setRange(0, 99.999)
        self.overlap_y.setValue(float(defaults.get("overlap_y_percent", 0.0)))
        self.enable_x = QCheckBox("Enable X overlap")
        self.enable_y = QCheckBox("Enable Y overlap")
        self.start_corner = QComboBox()
        self.start_corner.addItems(["top-left", "top-right", "bottom-left", "bottom-right"])
        self.primary_axis = QComboBox()
        self.primary_axis.addItems(["x", "y"])
        self.serpentine = QCheckBox("Serpentine order")
        self.serpentine.setChecked(True)
        self.wg_dose = QDoubleSpinBox()
        self.gc_dose = QDoubleSpinBox()
        self.marker_dose = QDoubleSpinBox()
        for widget, key in (
            (self.wg_dose, "beamer_wg_dose"),
            (self.gc_dose, "beamer_gc_dose"),
            (self.marker_dose, "beamer_marker_dose"),
        ):
            widget.setRange(0, 1e9)
            widget.setDecimals(6)
            widget.setValue(float(defaults.get(key, 1.8)))
        form.addRow("Field size A (µm)", self.field_size)
        form.addRow("Edge clearance (µm)", self.clearance)
        form.addRow(self.enable_x)
        form.addRow("X overlap (%)", self.overlap_x)
        form.addRow(self.enable_y)
        form.addRow("Y overlap (%)", self.overlap_y)
        form.addRow("Start corner", self.start_corner)
        form.addRow("Primary axis", self.primary_axis)
        form.addRow(self.serpentine)
        form.addRow("WG dose", self.wg_dose)
        form.addRow("GC dose", self.gc_dose)
        form.addRow("Marker dose", self.marker_dose)
        note = QLabel(
            "Defaults: 520 µm fields, 10 µm edge clearance, layer 6 Ebeam. "
            "Empty fields are removed by exact Qt shape intersection."
        )
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class ModuleVariablesDialog(QDialog):
    def __init__(self, main_window: "NativeLayoutWindow", members: list[dict[str, Any]]) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.members = members
        self.inputs: list[tuple[dict[str, Any], str, QWidget, str]] = []
        self.setWindowTitle(f"Module variables — {members[0].get('module_name', 'Module')}")
        self.resize(650, 700)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        for index, component in enumerate(members, start=1):
            alias = component.get("module_component_alias") or f"{component.get('kind','component')}_{index}"
            params = component.get("params", {})
            specs = COMPONENT_SPECS.get(component.get("kind"), {})
            for key, value in params.items():
                spec = specs.get(key, ["string" if isinstance(value, str) else "float", value])
                widget = main_window.make_parameter_widget(key, spec, value)
                form.addRow(f"{alias}.{key}", widget)
                self.inputs.append((component, key, widget, spec[0]))
        scroll.setWidget(content)
        outer.addWidget(scroll)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_values)
        buttons.accepted.connect(self.accept_values)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def apply_values(self) -> None:
        snapshot = self.main_window.snapshot()
        try:
            for component, key, widget, spec_type in self.inputs:
                component["params"][key] = self.main_window.read_parameter_widget(widget, spec_type)
            self.main_window.commit_interaction_snapshot(snapshot)
            self.main_window.rebuild_scene(preserve_selection=True)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid module parameter", str(exc))

    def accept_values(self) -> None:
        self.apply_values()
        self.accept()
