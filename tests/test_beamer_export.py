from __future__ import annotations

import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from max_layout.beamer import (
    beamer_cjob_template,
    beamer_frame_size_um,
    beamer_gpf_name,
    beamer_mask_size_mm,
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
                    "kind": "Chip outline",
                    "params": {"width": 15000.0, "height": 13000.0},
                },
            ],
            collect_field_records=lambda: records,
            beamer_flow_template=lambda regions, wg, gc, marker: (
                NativeLayoutWindow.beamer_flow_template(
                    None, regions, wg, gc, marker
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
            self.assertTrue(ftxt_path.is_file())
            self.assertTrue(cjob_path.is_file())
            flow = ftxt_path.read_text(encoding="utf-8")
            cjob = ElementTree.fromstring(cjob_path.read_text(encoding="utf-8"))
            expected_gpf = beamer_gpf_name(flow)
            self.assertEqual(
                cjob.find("./substrate/exposure/pattern").attrib["name"],
                expected_gpf,
            )
            self.assertEqual(cjob.find("./substrate/mask").attrib["size"], "13mmx11mm")
            self.assertIn(str(cjob_path), messages[-1])


if __name__ == "__main__":
    unittest.main()
