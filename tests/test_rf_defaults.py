from __future__ import annotations

import unittest

from max_layout.constants import DEFAULT_COMPONENT_VALUES, SIMULATION_COMPONENT_KINDS
from max_layout.gds.build import component_geometry_arrays
from max_layout.rf_defaults import (
    RF_FDE_COMPONENT_KINDS,
    RF_FDTD_COMPONENT_KINDS,
    default_rf_configuration,
    normalize_rf_configuration,
    rf_workflow_for_component,
)


class RFDefaultsTests(unittest.TestCase):
    def test_solver_split_matches_official_examples(self) -> None:
        self.assertEqual(RF_FDE_COMPONENT_KINDS, {"CPW"})
        self.assertEqual(rf_workflow_for_component("CPW"), "fde")
        for kind in RF_FDTD_COMPONENT_KINDS:
            self.assertEqual(rf_workflow_for_component(kind), "fdtd")

    def test_default_resources_are_cpu_for_fde_and_gpu_for_fdtd(self) -> None:
        fde = default_rf_configuration("CPW")
        fdtd = default_rf_configuration("Tapered CPW")
        self.assertEqual(fde["resource_mode"], "CPU")
        self.assertEqual(fdtd["resource_mode"], "GPU")
        self.assertEqual(fdtd["rf_port_strategy"], "manual_only")
        self.assertFalse(fdtd["use_endpoint_reference_planes"])

    def test_tfln_rf_stack_has_real_microwave_and_metal_properties(self) -> None:
        configuration = normalize_rf_configuration("CPW")
        stack = configuration["material_stack"]
        tfln = next(row for row in stack if row["name"] == "X-cut TFLN")
        metal = next(row for row in stack if row["role"] == "metal")
        self.assertEqual(
            (
                tfln["relative_permittivity_x"],
                tfln["relative_permittivity_y"],
                tfln["relative_permittivity_z"],
            ),
            (27.9, 44.3, 44.3),
        )
        self.assertGreater(metal["thickness_um"], 0.0)
        self.assertGreater(metal["conductivity_s_per_m"], 0.0)

    def test_wrong_solver_or_resource_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the FDE"):
            normalize_rf_configuration("CPW", {"rf_workflow": "fdtd"})
        with self.assertRaisesRegex(ValueError, "uses GPU"):
            normalize_rf_configuration("Tapered CPW", {"resource_mode": "CPU"})
        with self.assertRaisesRegex(ValueError, "needs manual RF planes"):
            normalize_rf_configuration(
                "Tapered CPW", {"rf_port_strategy": "cross_section_only"}
            )

    def test_rf_ports_are_explicit_simulation_only_objects(self) -> None:
        for kind in ("RF mode port", "RF power monitor"):
            self.assertIn(kind, SIMULATION_COMPONENT_KINDS)
            self.assertIn(kind, DEFAULT_COMPONENT_VALUES)
            component = {
                "uid": 100,
                "kind": kind,
                "x": 0.0,
                "y": 0.0,
                "orientation_deg": 0.0,
                "mirrored": False,
                "params": dict(DEFAULT_COMPONENT_VALUES[kind]),
            }
            self.assertEqual(component_geometry_arrays(component), ([], []))


if __name__ == "__main__":
    unittest.main()
