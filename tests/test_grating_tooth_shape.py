from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest

import numpy as np

from max_layout.constants import (
    CHOICE_PARAMETERS,
    COMPONENT_SPECS,
    DEFAULT_COMPONENT_VALUES,
)
from max_layout.gds.build import component_geometry_arrays
from max_layout.lumerical import (
    generate_lumerical_notebook,
    sweepable_component_parameters,
)
from max_layout.lumerical_optimization import (
    adjoint_optimizable_component_parameters,
)
from max_layout.ui.window import (
    NativeLayoutWindow,
    grating_fiber_center_local_um,
    grating_first_flare_local_x_um,
)


def component(kind: str, uid: int = 1) -> dict:
    return {
        "uid": uid,
        "kind": kind,
        "x": 0.0,
        "y": 0.0,
        "orientation_deg": 0.0,
        "mirrored": False,
        "params": deepcopy(DEFAULT_COMPONENT_VALUES[kind]),
        "attachment": None,
    }


def assignment_value(notebook: dict, name: str):
    import ast

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"No notebook assignment named {name!r}")


def polygon_bounds(polygons: list[tuple[np.ndarray, int, int]], layer: int):
    result = []
    for points, polygon_layer, datatype in polygons:
        if polygon_layer != layer:
            continue
        points = np.asarray(points, dtype=float)
        result.append((points.min(axis=0), points.max(axis=0), points, datatype))
    return result


class GratingToothShapeTests(unittest.TestCase):
    def test_curved_is_the_visible_default_for_both_grating_components(self) -> None:
        self.assertEqual(
            CHOICE_PARAMETERS["tooth_shape"], ["curved", "rectangular"]
        )
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    DEFAULT_COMPONENT_VALUES[kind]["tooth_shape"], "curved"
                )
                self.assertEqual(
                    COMPONENT_SPECS[kind]["tooth_shape"],
                    ["choice", "curved", ["curved", "rectangular"]],
                )

    def test_curved_default_is_identical_for_legacy_json_without_the_choice(self) -> None:
        expected_snapshot = {
            "Grating coupler": {
                "polygon_count": 32,
                "minimum": np.asarray([0.0, -12.121]),
                "maximum": np.asarray([58.294, 12.121]),
            },
            "GC-SOI": {
                "polygon_count": 47,
                "minimum": np.asarray([0.0, -18.27381511]),
                "maximum": np.asarray([70.91271704, 18.27381511]),
            },
        }
        for kind, snapshot in expected_snapshot.items():
            with self.subTest(kind=kind):
                explicit = component(kind)
                legacy = deepcopy(explicit)
                legacy["params"].pop("tooth_shape")
                explicit_polygons, _ = component_geometry_arrays(explicit)
                legacy_polygons, _ = component_geometry_arrays(legacy)

                self.assertEqual(len(explicit_polygons), snapshot["polygon_count"])
                self.assertEqual(len(explicit_polygons), len(legacy_polygons))
                for explicit_polygon, legacy_polygon in zip(
                    explicit_polygons, legacy_polygons
                ):
                    self.assertEqual(explicit_polygon[1:], legacy_polygon[1:])
                    np.testing.assert_allclose(
                        explicit_polygon[0], legacy_polygon[0], rtol=0.0, atol=1e-12
                    )

                all_points = np.vstack(
                    [points for points, _layer, _datatype in explicit_polygons]
                )
                np.testing.assert_allclose(
                    all_points.min(axis=0), snapshot["minimum"], rtol=0.0, atol=1e-8
                )
                np.testing.assert_allclose(
                    all_points.max(axis=0), snapshot["maximum"], rtol=0.0, atol=1e-8
                )

    def test_generic_rectangular_teeth_preserve_pitch_count_and_apodization(self) -> None:
        grating = component("Grating coupler")
        grating["params"].update(
            {
                "tooth_shape": "rectangular",
                "pitch": 0.8,
                "fill_factor": 0.9,  # The apodization expression overrides this.
                "fill_factors": "linspace(0.25, 0.55)",
                "N": 4,
                "L_extra": 0.0,
            }
        )
        polygons, _ = component_geometry_arrays(grating)

        params = grating["params"]
        half_angle = math.radians(float(params["alpha_t"])) / 2.0
        focus_offset = (float(params["wg_width"]) / 2.0) / math.tan(half_angle)
        flare_x = (
            float(params["wg_length"])
            - focus_offset
            + float(params["taper_L"])
        )
        flare_half_width = float(params["taper_L"]) * math.sin(half_angle)
        fills = np.linspace(0.25, 0.55, 4)
        # ``add_parent_focusing_gc`` uses gdstk boolean clipping at 1 nm
        # layout precision, so its output vertices are quantized to 0.001 um.
        geometry_tolerance = 6e-4

        layer_bounds = polygon_bounds(polygons, layer=2)
        teeth = sorted(
            (entry for entry in layer_bounds if entry[0][0] > flare_x + 1e-9),
            key=lambda entry: float(entry[0][0]),
        )
        self.assertEqual(len(teeth), 4)
        for index, (low, high, points, datatype) in enumerate(teeth):
            fill = float(fills[index])
            expected_low_x = flare_x + index * 0.8 + 0.8 * (1.0 - fill)
            expected_high_x = flare_x + (index + 1) * 0.8
            self.assertEqual(datatype, 0)
            self.assertEqual(len(points), 4)
            np.testing.assert_allclose(
                low,
                [expected_low_x, -flare_half_width],
                rtol=0.0,
                atol=geometry_tolerance,
            )
            np.testing.assert_allclose(
                high,
                [expected_high_x, flare_half_width],
                rtol=0.0,
                atol=geometry_tolerance,
            )

        taper_candidates = [
            entry
            for entry in layer_bounds
            if entry[0][0] <= float(params["wg_length"]) + 1e-9
            and abs(float(entry[1][0]) - flare_x) <= geometry_tolerance
        ]
        self.assertEqual(len(taper_candidates), 1)
        taper_points = taper_candidates[0][2]
        endpoint = taper_points[
            np.isclose(taper_points[:, 0], flare_x, atol=geometry_tolerance)
        ]
        self.assertEqual(len(endpoint), 2)
        np.testing.assert_allclose(
            np.sort(endpoint[:, 1]),
            [-flare_half_width, flare_half_width],
            rtol=0.0,
            atol=geometry_tolerance,
        )

    def test_soi_rectangular_teeth_use_the_taper_end_width(self) -> None:
        grating = component("GC-SOI")
        grating["params"].update(
            {
                "tooth_shape": "rectangular",
                "pitch": 0.8,
                "duty_cycle": 0.9,  # The apodization expression overrides this.
                "fill_factors": "linspace(0.25, 0.55)",
                "target_length": 2.6,  # ceil(2.6 / 0.8) = four teeth.
                "L_extra": 0.0,
            }
        )
        polygons, _ = component_geometry_arrays(grating)

        params = grating["params"]
        # Preserve the existing geometry-exact first-flare/reference plane:
        # the curved SOI input sector reaches its centerline radius at
        # ``wg_length + radius``.  The rectangular option replaces the taper
        # and input sector with one straight taper ending on that same plane.
        flare_x = float(params["wg_length"]) + float(params["radius"])
        flare_half_width = float(params["y_span"]) / 2.0
        fills = np.linspace(0.25, 0.55, 4)

        etched_bounds = polygon_bounds(polygons, layer=2)
        teeth = sorted(
            (entry for entry in etched_bounds if entry[0][0] > flare_x + 1e-9),
            key=lambda entry: float(entry[0][0]),
        )
        self.assertEqual(len(teeth), 4)
        for index, (low, high, points, datatype) in enumerate(teeth):
            fill = float(fills[index])
            expected_low_x = flare_x + index * 0.8 + 0.8 * (1.0 - fill)
            expected_high_x = flare_x + (index + 1) * 0.8
            self.assertEqual(datatype, 0)
            self.assertEqual(len(points), 4)
            np.testing.assert_allclose(
                low,
                [expected_low_x, -flare_half_width],
                rtol=0.0,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                high,
                [expected_high_x, flare_half_width],
                rtol=0.0,
                atol=1e-9,
            )

        taper_candidates = [
            entry
            for entry in etched_bounds
            if abs(float(entry[0][0]) - float(params["wg_length"])) <= 1e-9
            and abs(float(entry[1][0]) - flare_x) <= 1e-9
        ]
        self.assertEqual(len(taper_candidates), 1)
        taper_points = taper_candidates[0][2]
        endpoint = taper_points[np.isclose(taper_points[:, 0], flare_x, atol=1e-9)]
        self.assertEqual(len(endpoint), 2)
        np.testing.assert_allclose(
            np.sort(endpoint[:, 1]),
            [-flare_half_width, flare_half_width],
            rtol=0.0,
            atol=1e-9,
        )

    def test_rectangular_choice_survives_project_json_and_notebook_export(self) -> None:
        class Factory:
            make_component = NativeLayoutWindow.make_component
            project_payload = NativeLayoutWindow.project_payload

            def __init__(self) -> None:
                self.components = []
                self.next_uid = 1
                self.next_group_id = 1
                self.next_array_id = 1
                self.next_module_instance_id = 1

        factory = Factory()
        for kind in ("Grating coupler", "GC-SOI"):
            placed = factory.make_component(kind, 0.0, 0.0)
            placed["params"]["tooth_shape"] = "rectangular"
            placed["params"]["fill_factors"] = "linspace(0.31, 0.49)"
            factory.components.append(placed)

        saved = json.loads(json.dumps(factory.project_payload()))
        self.assertEqual(
            [row["params"]["tooth_shape"] for row in saved["components"]],
            ["rectangular", "rectangular"],
        )

        notebook, _warnings = generate_lumerical_notebook(
            saved["components"], {"included_layers": [[1, 0], [2, 0]]}
        )
        exported = assignment_value(notebook, "SOURCE_COMPONENTS_JSON")
        self.assertEqual(
            [row["params"]["tooth_shape"] for row in exported],
            ["rectangular", "rectangular"],
        )
        self.assertEqual(
            [row["params"]["fill_factors"] for row in exported],
            ["linspace(0.31, 0.49)", "linspace(0.31, 0.49)"],
        )

    def test_notebook_embeds_the_exact_rectangular_editor_polygons(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                grating = component(kind)
                grating["params"]["tooth_shape"] = "rectangular"
                editor_polygons, _ = component_geometry_arrays(grating)
                notebook, _warnings = generate_lumerical_notebook(
                    [grating], {"included_layers": [[1, 0], [2, 0]]}
                )
                embedded = assignment_value(notebook, "GEOMETRY")
                self.assertEqual(len(embedded), len(editor_polygons))
                # Notebook geometry is translated into its simulation-local
                # frame.  The rigid translation may differ, but every vertex
                # and polygon must otherwise be identical to the editor/GDS.
                translation = (
                    np.asarray(embedded[0]["vertices_um"], dtype=float)[0]
                    - np.asarray(editor_polygons[0][0], dtype=float)[0]
                )
                for exported_polygon, (points, layer, datatype) in zip(
                    embedded, editor_polygons
                ):
                    self.assertEqual(exported_polygon["layer"], layer)
                    self.assertEqual(exported_polygon["datatype"], datatype)
                    np.testing.assert_allclose(
                        exported_polygon["vertices_um"],
                        np.asarray(points, dtype=float) + translation,
                        rtol=0.0,
                        atol=1e-12,
                    )

    def test_shape_choice_is_fixed_topology_not_a_numeric_optimizer_axis(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                grating = component(kind)
                self.assertNotIn(
                    "tooth_shape",
                    {
                        row["parameter"]
                        for row in sweepable_component_parameters(grating)
                    },
                )
                self.assertNotIn(
                    "tooth_shape",
                    {
                        row["parameter"]
                        for row in adjoint_optimizable_component_parameters(grating)
                    },
                )

    def test_shape_toggle_preserves_fiber_and_first_flare_anchors(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                curved = component(kind)
                rectangular = deepcopy(curved)
                rectangular["params"]["tooth_shape"] = "rectangular"
                self.assertEqual(
                    grating_first_flare_local_x_um(curved),
                    grating_first_flare_local_x_um(rectangular),
                )
                self.assertEqual(
                    grating_fiber_center_local_um(curved),
                    grating_fiber_center_local_um(rectangular),
                )

    def test_unknown_tooth_shape_is_rejected(self) -> None:
        for kind in ("Grating coupler", "GC-SOI"):
            with self.subTest(kind=kind):
                invalid = component(kind)
                invalid["params"]["tooth_shape"] = "triangular"
                with self.assertRaisesRegex(ValueError, "tooth_shape"):
                    component_geometry_arrays(invalid)


if __name__ == "__main__":
    unittest.main()
