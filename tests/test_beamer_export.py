from __future__ import annotations

import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from max_layout.beamer import (
    beamer_alignment_marks,
    beamer_cjob_template,
    beamer_edge_coupler_patterns,
    beamer_frame_size_um,
    beamer_global_alignment_marks,
    beamer_gpf_name,
    beamer_local_alignment_marks,
    beamer_mask_size_mm,
)
from max_layout.constants import (
    DEFAULT_COMPONENT_VALUES,
    EXPONENTIAL_EDGE_COUPLER_FEED_KIND,
    GLOBAL_MARKER_LAYER,
    edge_coupler_local_marker_layer,
    edge_coupler_slab_layer,
)
from max_layout.ui.window import NativeLayoutWindow


class BeamerCjobExportTests(unittest.TestCase):
    def test_chip_outline_is_authoritative_frame(self) -> None:
        components = [
            {
                "kind": "Chip outline",
                "params": {"width": 15000.0, "height": 13000.0},
            }
        ]
        frame = beamer_frame_size_um(
            components,
            [{"xmin": 0.0, "ymin": 0.0, "xmax": 4000.0, "ymax": 3000.0}],
        )
        self.assertEqual(frame, (15000.0, 13000.0))
        self.assertEqual(beamer_mask_size_mm(frame), (13.0, 11.0))

    def test_cjob_matches_supplied_single_pattern_structure(self) -> None:
        text = beamer_cjob_template(
            "O_Band_4_MZI_1950.gpf",
            (15000.0, 13000.0),
            exposure_name="oband",
        )
        root = ElementTree.fromstring(text)
        self.assertEqual(root.tag, "cjob")
        self.assertEqual(root.attrib, {"type": "ebpg5200", "version": "v02_23"})
        self.assertEqual(root.find("color").attrib["pattern"], "O_Band_4_MZI_1950.gpf")
        self.assertEqual(root.find("./substrate/mask").attrib["size"], "13mmx11mm")
        exposure = root.find("./substrate/exposure")
        self.assertEqual(exposure.attrib["ht"], "100kV")
        self.assertEqual(exposure.attrib["workinglevel"], "high")
        self.assertEqual(exposure.find("checks").attrib["enabled"], "false")
        pattern = exposure.find("pattern")
        self.assertEqual(pattern.attrib["name"], "O_Band_4_MZI_1950.gpf")
        self.assertEqual(
            pattern.find("beam").attrib,
            {"dose": "1000", "defocus": "#0", "name": "10na_300.beam_100"},
        )

    def test_gpf_name_is_taken_from_the_ftxt_export_node(self) -> None:
        flow = """FILE_NAME = .%5Cignored.gds
FILE_NAME = .%5CO_Band_4_MZI_1950.gpf
"""
        self.assertEqual(beamer_gpf_name(flow), "O_Band_4_MZI_1950.gpf")

    def test_local_edge_coupler_marks_are_embedded_with_global_coordinates(self) -> None:
        component = {
            "uid": 7,
            "kind": "Exponential edge coupler",
            "x": 1000.0,
            "y": 2000.0,
            "orientation_deg": 90.0,
            "mirrored": False,
            "params": dict(DEFAULT_COMPONENT_VALUES["Exponential edge coupler"]),
        }
        marks = beamer_local_alignment_marks([component])
        self.assertEqual(len(marks), 2)
        self.assertEqual(
            {(mark["side"], mark["x"], mark["y"]) for mark in marks},
            {("top", 900.0, 2110.0), ("bottom", 1100.0, 2110.0)},
        )
        text = beamer_cjob_template(
            "edge.gpf",
            (15000.0, 13000.0),
            local_alignment_marks=marks,
        )
        ElementTree.fromstring(text)
        self.assertIn(
            'LOCAL_ALIGNMENT ident="L1" uid="7" side="top" coord_um="900,2110" size_um="50" layer="11" name="L-marker 1" coupler="1" end="single" slab_layer="12" gpf="edge.gpf"',
            text,
        )
        self.assertIn(
            'LOCAL_ALIGNMENT ident="L1" uid="7" side="bottom" coord_um="1100,2110" size_um="50" layer="11" name="L-marker 1" coupler="1" end="single" slab_layer="12" gpf="edge.gpf"',
            text,
        )

    def test_each_edge_coupler_has_its_own_gpf_and_local_marker_pair(self) -> None:
        components = []
        for uid, x in ((30, 3000.0), (10, 1000.0), (20, 2000.0)):
            components.append(
                {
                    "uid": uid,
                    "kind": "Exponential edge coupler",
                    "x": x,
                    "y": 0.0,
                    "orientation_deg": 0.0,
                    "mirrored": False,
                    "params": dict(
                        DEFAULT_COMPONENT_VALUES["Exponential edge coupler"]
                    ),
                }
            )
        patterns = beamer_edge_coupler_patterns(components, "mask.gpf")
        self.assertEqual(
            [pattern["name"] for pattern in patterns],
            [
                "mask_slab_001.gpf",
                "mask_slab_002.gpf",
                "mask_slab_003.gpf",
            ],
        )
        self.assertEqual(
            [
                (pattern["slab_layer"], pattern["local_marker_layer"])
                for pattern in patterns
            ],
            [
                (edge_coupler_slab_layer(1), edge_coupler_local_marker_layer(1)),
                (edge_coupler_slab_layer(2), edge_coupler_local_marker_layer(2)),
                (edge_coupler_slab_layer(3), edge_coupler_local_marker_layer(3)),
            ],
        )
        marks = beamer_local_alignment_marks(components)
        text = beamer_cjob_template(
            "mask.gpf",
            (14000.0, 12000.0),
            local_alignment_marks=marks,
            edge_coupler_patterns=patterns,
        )
        root = ElementTree.fromstring(text)
        self.assertEqual(
            [node.attrib["name"] for node in root.findall("./substrate/exposure/pattern")],
            ["mask.gpf", "mask_slab_001.gpf", "mask_slab_002.gpf", "mask_slab_003.gpf"],
        )
        self.assertIn(
            'LOCAL_ALIGNMENT ident="L2" uid="20" side="top" coord_um="2110,100" size_um="50" layer="11" name="L-marker 2" coupler="2" end="single" slab_layer="13" gpf="mask_slab_002.gpf"',
            text,
        )

    def test_pair_feed_and_modulator_export_one_pattern_per_physical_end(self) -> None:
        pair = {
            "uid": 10,
            "kind": EXPONENTIAL_EDGE_COUPLER_FEED_KIND,
            "x": 0.0,
            "y": 0.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "params": dict(
                DEFAULT_COMPONENT_VALUES[EXPONENTIAL_EDGE_COUPLER_FEED_KIND]
            ),
        }
        modulator_parameters = dict(
            DEFAULT_COMPONENT_VALUES["MZI + CPW modulator"]
        )
        modulator_parameters["add_edge_coupler_tapers"] = True
        modulator = {
            "uid": 20,
            "kind": "MZI + CPW modulator",
            "x": 0.0,
            "y": 1000.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "params": modulator_parameters,
        }
        patterns = beamer_edge_coupler_patterns([modulator, pair], "mask.gpf")
        self.assertEqual(len(patterns), 4)
        self.assertEqual(
            [(item["coupler_number"], item["uid"], item["end"]) for item in patterns],
            [(1, 10, "left"), (2, 10, "right"), (3, 20, "left"), (4, 20, "right")],
        )
        marks = beamer_local_alignment_marks([modulator, pair])
        self.assertEqual(len(marks), 8)
        self.assertEqual(
            {(item["coupler_number"], item["uid"], item["end"]) for item in marks},
            {
                (1, 10, "left"),
                (2, 10, "right"),
                (3, 20, "left"),
                (4, 20, "right"),
            },
        )
        text = beamer_cjob_template(
            "mask.gpf",
            (14000.0, 12000.0),
            local_alignment_marks=marks,
            edge_coupler_patterns=patterns,
        )
        ElementTree.fromstring(text)
        self.assertIn('uid="20" end="left" slab_layer="14"', text)
        self.assertIn('uid="20" end="right" slab_layer="15"', text)

    def test_chip_frame_global_marks_are_embedded_with_global_coordinates(self) -> None:
        component = {
            "uid": 8,
            "kind": "Chip outline",
            "x": 1000.0,
            "y": 2000.0,
            "orientation_deg": 0.0,
            "mirrored": False,
            "params": dict(DEFAULT_COMPONENT_VALUES["Chip outline"]),
        }
        marks = beamer_global_alignment_marks([component])
        self.assertEqual(len(marks), 12)
        self.assertEqual(
            {(mark["side"], mark["x"], mark["y"]) for mark in marks},
            {
                ("top_1", 800.0, 7965.0),
                ("top_2", 1000.0, 7965.0),
                ("top_3", 1200.0, 7965.0),
                ("bottom_1", 800.0, -3965.0),
                ("bottom_2", 1000.0, -3965.0),
                ("bottom_3", 1200.0, -3965.0),
                ("left_1", -5965.0, 1800.0),
                ("left_2", -5965.0, 2000.0),
                ("left_3", -5965.0, 2200.0),
                ("right_1", 7965.0, 1800.0),
                ("right_2", 7965.0, 2000.0),
                ("right_3", 7965.0, 2200.0),
            },
        )
        text = beamer_cjob_template(
            "frame.gpf",
            (14000.0, 12000.0),
            local_alignment_marks=marks,
        )
        ElementTree.fromstring(text)
        self.assertIn(
            f'GLOBAL_ALIGNMENT ident="GLOBAL" uid="8" side="top_2" coord_um="1000,7965" size_um="50" layer="{GLOBAL_MARKER_LAYER}" name="Ebeam global marker"',
            text,
        )

    def test_large_legacy_write_field_extent_is_a_frame_fallback(self) -> None:
        frame = beamer_frame_size_um(
            [],
            [
                {"xmin": -2500.0, "ymin": -2000.0, "xmax": 0.0, "ymax": 0.0},
                {"xmin": 0.0, "ymin": 0.0, "xmax": 2500.0, "ymax": 2000.0},
            ],
        )
        self.assertEqual(frame, (5000.0, 4000.0))
        self.assertEqual(beamer_mask_size_mm(frame), (3.0, 2.0))

    def test_too_small_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must exceed"):
            beamer_mask_size_mm((1800.0, 4000.0))

    def test_ftxt_export_writes_matching_sibling_cjob(self) -> None:
        records = [
            {
                "xmin": -260.0,
                "ymin": -260.0,
                "xmax": 260.0,
                "ymax": 260.0,
                "region_name": "R1",
            }
        ]
        messages = []
        fake_window = SimpleNamespace(
            components=[
                {
                    "kind": "E-beam multipass",
                    "params": {
                        "beamer_wg_dose": 1.8,
                        "beamer_gc_dose": 1.8,
                        "beamer_marker_dose": 1.8,
                    },
                },
                {
                    "uid": 8,
                    "kind": "Chip outline",
                    "x": 0.0,
                    "y": 0.0,
                    "orientation_deg": 0.0,
                    "mirrored": False,
                    "params": {
                        **DEFAULT_COMPONENT_VALUES["Chip outline"],
                        "width": 15000.0,
                        "height": 13000.0,
                    },
                },
                {
                    "uid": 9,
                    "kind": "Linear edge coupler",
                    "x": 0.0,
                    "y": 0.0,
                    "orientation_deg": 0.0,
                    "mirrored": False,
                    "params": dict(DEFAULT_COMPONENT_VALUES["Linear edge coupler"]),
                },
            ],
            collect_field_records=lambda: records,
            beamer_flow_template=lambda regions, wg, gc, marker, **kwargs: (
                NativeLayoutWindow.beamer_flow_template(
                    None, regions, wg, gc, marker, **kwargs
                )
            ),
            statusBar=lambda: SimpleNamespace(showMessage=messages.append),
        )
        with TemporaryDirectory() as directory:
            ftxt_path = Path(directory) / "O_Band_4_MZI.ftxt"
            with patch(
                "max_layout.ui.window.QFileDialog.getSaveFileName",
                return_value=(str(ftxt_path), "BEAMER FTEXT (*.ftxt)"),
            ):
                NativeLayoutWindow.export_beamer_ftext(fake_window)

            cjob_path = ftxt_path.with_suffix(".cjob")
            edge_ftxt_path = Path(directory) / "O_Band_4_MZI_slab_001.ftxt"
            self.assertTrue(ftxt_path.is_file())
            self.assertTrue(cjob_path.is_file())
            self.assertTrue(edge_ftxt_path.is_file())
            flow = ftxt_path.read_text(encoding="utf-8")
            self.assertIn(
                "LAYERSET = 1(0)%2C%207(0)%2C%2010(0)%2C%2011(0)", flow
            )
            self.assertIn("LAYER_MAPPING = 1(0)%20%3A%20WG", flow)
            self.assertIn("LAYER_MAPPING = 7(0)%20%3A%20WG_TIP", flow)
            self.assertIn("LAYER_MAPPING = 10(0)%20%3A%20G_MARKERS", flow)
            self.assertIn("LAYER_MAPPING = 11(0)%20%3A%20L_MARKERS", flow)
            self.assertNotIn("LAYER_MAPPING = 8(0)", flow)
            cjob_text = cjob_path.read_text(encoding="utf-8")
            cjob = ElementTree.fromstring(cjob_text)
            expected_gpf = beamer_gpf_name(flow)
            self.assertEqual(
                cjob.find("./substrate/exposure/pattern").attrib["name"],
                expected_gpf,
            )
            self.assertEqual(cjob.find("./substrate/mask").attrib["size"], "13mmx11mm")
            self.assertIn('LOCAL_ALIGNMENT ident="L1" uid="9" side="top"', cjob_text)
            self.assertIn('coord_um="110,100" size_um="50" layer="11"', cjob_text)
            self.assertIn('slab_layer="12" gpf="O_Band_4_MZI_slab_001.gpf"', cjob_text)
            self.assertIn(
                'GLOBAL_ALIGNMENT ident="GLOBAL" uid="8" side="top_2" coord_um="0,6465" size_um="50" layer="10" name="Ebeam global marker"',
                cjob_text,
            )
            patterns = cjob.findall("./substrate/exposure/pattern")
            self.assertEqual(
                [pattern.attrib["name"] for pattern in patterns],
                ["O_Band_4_MZI.gpf", "O_Band_4_MZI_slab_001.gpf"],
            )
            edge_flow = edge_ftxt_path.read_text(encoding="utf-8")
            self.assertIn("LAYER_MAPPING = 12(0)%20%3A%20SLAB_001", edge_flow)
            self.assertIn("LAYER_MAPPING = 11(0)%20%3A%20L_MARKERS", edge_flow)
            self.assertIn("LAYER_MAPPING = 10(0)%20%3A%20G_MARKERS", edge_flow)
            self.assertIn(
                "LAYERSET = 12(0)%2C%2011(0)%2C%2010(0)", edge_flow
            )
            self.assertIn("MULTIPASS_LAYER = SLAB_001", edge_flow)
            self.assertIn("FILE_NAME = .%5CO_Band_4_MZI_slab_001.gpf", edge_flow)
            self.assertEqual(len(beamer_alignment_marks(fake_window.components)), 14)
            self.assertIn(str(cjob_path), messages[-1])


if __name__ == "__main__":
    unittest.main()
