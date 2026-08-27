"""Array, e-beam, and module-variable dialogs."""

from __future__ import annotations

from typing import Any
import json

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QScrollArea, QSpinBox, QVBoxLayout, QWidget

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


class ArrayEditDialog(QDialog):
    """Edit one persisted array recipe and every component template it contains."""

    _INTERNAL_PARAMS = {
        "manual_field_offsets",
        "manual_field_order",
        "removed_field_keys",
        "auto_pruned_field_keys",
        "geometry_field_adjacency",
        "explicit_fields",
        "polygons",
    }

    def __init__(
        self,
        main_window: "NativeLayoutWindow",
        array_id: str,
        spec: dict[str, Any],
        templates: list[tuple[str, dict[str, Any]]],
    ) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.array_id = str(array_id)
        self.template_inputs: dict[str, dict[str, tuple[QWidget, str]]] = {}
        self.template_param_widgets: dict[str, list[QWidget]] = {}
        self.setWindowTitle(f"Edit entire array {self.array_id}")
        self.resize(760, 820)

        outer = QVBoxLayout(self)
        note = QLabel(
            "Array layout and sweep values apply to every cell. Component values below "
            "are shared by the matching component in every cell; row/column sweep "
            "parameters remain controlled by the array recipe."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        array_group = QGroupBox("Array layout and sweeps")
        array_form = QFormLayout(array_group)
        self.nx = QSpinBox()
        self.nx.setRange(1, 1000)
        self.nx.setValue(max(1, int(spec.get("nx", 1))))
        self.ny = QSpinBox()
        self.ny.setRange(1, 1000)
        self.ny.setValue(max(1, int(spec.get("ny", 1))))
        self.dx = QLineEdit(self._sequence_text(spec.get("x_spacings", []), "1000"))
        self.dy = QLineEdit(self._sequence_text(spec.get("y_spacings", []), "1000"))

        numeric_names = sorted(
            {
                key
                for _template_id, component in templates
                for key, value in component.get("params", {}).items()
                if key not in self._INTERNAL_PARAMS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        self.x_param = QComboBox()
        self.y_param = QComboBox()
        for widget in (self.x_param, self.y_param):
            widget.addItem("")
            widget.addItems(numeric_names)
        self.x_param.setCurrentText(str(spec.get("x_parameter", "")))
        self.y_param.setCurrentText(str(spec.get("y_parameter", "")))
        self.x_values = QLineEdit(self._sequence_text(spec.get("x_values", []), ""))
        self.y_values = QLineEdit(self._sequence_text(spec.get("y_values", []), ""))
        self.auto_label = QCheckBox("Add a sweep label to every array cell")
        self.auto_label.setChecked(bool(spec.get("auto_label", False)))
        self.label_prefix = QLineEdit(str(spec.get("label_prefix", "A")))
        self.label_offset_x = QDoubleSpinBox()
        self.label_offset_y = QDoubleSpinBox()
        for widget, key in (
            (self.label_offset_x, "label_offset_x"),
            (self.label_offset_y, "label_offset_y"),
        ):
            widget.setRange(-1e9, 1e9)
            widget.setDecimals(6)
            widget.setValue(float(spec.get(key, 20.0)))
        array_form.addRow("Columns (Nx)", self.nx)
        array_form.addRow("Rows (Ny)", self.ny)
        array_form.addRow("Δx or spacing list (µm)", self.dx)
        array_form.addRow("Δy or spacing list (µm)", self.dy)
        array_form.addRow("Column sweep parameter", self.x_param)
        array_form.addRow("Column values", self.x_values)
        array_form.addRow("Row sweep parameter", self.y_param)
        array_form.addRow("Row values", self.y_values)
        array_form.addRow(self.auto_label)
        array_form.addRow("Label prefix", self.label_prefix)
        array_form.addRow("Label X offset (µm)", self.label_offset_x)
        array_form.addRow("Label Y offset (µm)", self.label_offset_y)
        content_layout.addWidget(array_group)

        for template_id, component in templates:
            group = QGroupBox(
                f"{component.get('kind', 'Component')} template · {template_id}"
            )
            form = QFormLayout(group)
            inputs: dict[str, tuple[QWidget, str]] = {}
            for key, value in (
                ("x", component.get("x", 0.0)),
                ("y", component.get("y", 0.0)),
                ("orientation_deg", component.get("orientation_deg", 0.0)),
            ):
                widget = main_window.make_parameter_widget(key, ["float", value], value)
                form.addRow("Base-cell " + key, widget)
                inputs[key] = (widget, "float")
            mirrored = main_window.make_parameter_widget(
                "mirrored", ["bool", component.get("mirrored", False)], component.get("mirrored", False)
            )
            form.addRow("mirrored", mirrored)
            inputs["mirrored"] = (mirrored, "bool")

            specs = COMPONENT_SPECS.get(component.get("kind"), {})
            for key, value in component.get("params", {}).items():
                if key in self._INTERNAL_PARAMS:
                    continue
                spec_row = specs.get(key)
                if spec_row is None:
                    if isinstance(value, bool):
                        spec_row = ["bool", value]
                    elif isinstance(value, int):
                        spec_row = ["int", value]
                    elif isinstance(value, float):
                        spec_row = ["float", value]
                    elif isinstance(value, str):
                        spec_row = ["string", value]
                    else:
                        spec_row = ["json", value]
                if spec_row[0] == "json":
                    widget = QLineEdit(json.dumps(value, separators=(",", ":")))
                else:
                    widget = main_window.make_parameter_widget(key, spec_row, value)
                form.addRow(key, widget)
                inputs[f"params.{key}"] = (widget, str(spec_row[0]))
                self.template_param_widgets.setdefault(key, []).append(widget)
            self.template_inputs[str(template_id)] = inputs
            content_layout.addWidget(group)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.x_param.currentTextChanged.connect(self._refresh_swept_widgets)
        self.y_param.currentTextChanged.connect(self._refresh_swept_widgets)
        self._refresh_swept_widgets()

    @staticmethod
    def _sequence_text(values: Any, fallback: str) -> str:
        if not isinstance(values, (list, tuple)) or not values:
            return fallback
        return ", ".join(f"{float(value):g}" for value in values)

    def raw_specification(self) -> dict[str, Any]:
        return {
            "nx": self.nx.value(),
            "ny": self.ny.value(),
            "x_spacings_text": self.dx.text(),
            "y_spacings_text": self.dy.text(),
            "x_parameter": self.x_param.currentText().strip(),
            "x_values_text": self.x_values.text(),
            "y_parameter": self.y_param.currentText().strip(),
            "y_values_text": self.y_values.text(),
            "auto_label": self.auto_label.isChecked(),
            "label_prefix": self.label_prefix.text().strip(),
            "label_offset_x": self.label_offset_x.value(),
            "label_offset_y": self.label_offset_y.value(),
        }

    def _refresh_swept_widgets(self) -> None:
        swept = {self.x_param.currentText().strip(), self.y_param.currentText().strip()} - {""}
        for key, widgets in self.template_param_widgets.items():
            for widget in widgets:
                is_swept = key in swept
                widget.setEnabled(not is_swept)
                widget.setToolTip(
                    "This value is controlled independently for each array row/column above."
                    if is_swept
                    else ""
                )

    def template_updates(self) -> dict[str, dict[str, Any]]:
        updates: dict[str, dict[str, Any]] = {}
        for template_id, inputs in self.template_inputs.items():
            row: dict[str, Any] = {"params": {}}
            for key, (widget, spec_type) in inputs.items():
                if not widget.isEnabled():
                    continue
                if spec_type == "json":
                    value = json.loads(str(widget.text()))
                else:
                    value = self.main_window.read_parameter_widget(widget, spec_type)
                if key.startswith("params."):
                    row["params"][key.split(".", 1)[1]] = value
                else:
                    row[key] = value
            updates[template_id] = row
        return updates


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
