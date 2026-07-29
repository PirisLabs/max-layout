#!/usr/bin/env python3
"""Build "Max Layout.pyz" from the source package in src/.

    python3 build.py

The archive is a standard zipapp: src/__main__.py becomes the entry point and
the max_layout package rides along beside it.  Run the result with

    python3 "Max Layout.pyz"
"""

from __future__ import annotations

import argparse
import compileall
import shutil
import sys
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
DEFAULT_OUTPUT = ROOT / "Max Layout.pyz"


def clean_caches(root: Path) -> None:
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip byte-compiling the package before building",
    )
    args = parser.parse_args()

    if not (SOURCE / "__main__.py").exists():
        print(f"error: {SOURCE / '__main__.py'} is missing", file=sys.stderr)
        return 1

    clean_caches(SOURCE)

    if not args.skip_check:
        if not compileall.compile_dir(str(SOURCE), quiet=1, force=True):
            print("error: source does not compile", file=sys.stderr)
            return 1
        clean_caches(SOURCE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    zipapp.create_archive(SOURCE, args.output, compressed=True)
    size_kb = args.output.stat().st_size / 1024
    modules = sum(1 for _ in SOURCE.rglob("*.py"))
    print(f"wrote {args.output.name}  ({size_kb:.0f} KB, {modules} modules)")
    print(f'run it with:  python3 "{args.output}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
