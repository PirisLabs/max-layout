#!/usr/bin/env python3
"""Create the extract-and-double-click Windows distribution bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "Max Layout Windows.zip"
BUNDLE_FILES = (
    (ROOT / "LICENSE", "LICENSE"),
    (ROOT / "Max Layout.pyz", "Max Layout.pyz"),
    (ROOT / "Max Layout Windows.cmd", "Max Layout Windows.cmd"),
    (
        ROOT / "Start Piris 3D Simulations Windows.cmd",
        "Start Piris 3D Simulations Windows.cmd",
    ),
    (
        ROOT / "windows" / "Install-And-Launch-MaxLayout.ps1",
        "windows/Install-And-Launch-MaxLayout.ps1",
    ),
    (
        ROOT / "windows" / "requirements-windows.txt",
        "windows/requirements-windows.txt",
    ),
    (
        ROOT / "windows" / "Start-Piris3DSimulations.ps1",
        "windows/Start-Piris3DSimulations.ps1",
    ),
    (
        ROOT / "windows" / "requirements-3d-launcher.txt",
        "windows/requirements-3d-launcher.txt",
    ),
    (
        ROOT / "windows" / "run_piris_3d_windows.py",
        "windows/run_piris_3d_windows.py",
    ),
    (ROOT / "windows" / "README.txt", "windows/README.txt"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [str(source) for source, _name in BUNDLE_FILES if not source.is_file()]
    if missing:
        parser.error("missing required bundle file(s): " + ", ".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, archive_name in BUNDLE_FILES:
            archive.write(source, archive_name)

    size_kb = args.output.stat().st_size / 1024.0
    print(f'wrote "{args.output}" ({size_kb:.0f} KB)')
    print('Windows: extract the ZIP, then double-click "Max Layout Windows.cmd"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
