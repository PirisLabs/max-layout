"""Locating the running application for subprocess re-invocation and export.

Before the package split every one of these questions had the same answer:
``combined_editor.py``.  A single file was the launcher, the thing a worker
subprocess re-ran, and the blob embedded into exported standalone scripts.
Now the application is a package that may be running either from a zipapp
archive or from a source checkout, so those three uses are answered here.
"""

from __future__ import annotations

import base64
import io
import sys
import zipapp
import zipfile
from pathlib import Path


def source_root() -> Path:
    """Directory holding ``__main__.py`` (the parent of this package)."""
    return Path(__file__).resolve().parent.parent


def launcher_path() -> str:
    """A path that ``python3 <path> ...`` runs as this application.

    Running from a zipapp gives the ``.pyz`` itself; running from a source
    checkout gives the directory containing ``__main__.py``.  Python executes
    both forms directly, so either is a valid ``sys.executable`` argument.
    """
    argv0 = str(sys.argv[0])
    if argv0.lower().endswith((".pyz", ".pyzw")):
        return str(Path(argv0).resolve())
    return str(source_root())


def package_archive_bytes() -> bytes:
    """A self-contained zipapp of the whole application.

    Used by the standalone-Python export, which needs to embed the entire
    program rather than a single module.
    """
    target = Path(launcher_path())
    if target.is_file():
        return target.read_bytes()
    buffer = io.BytesIO()
    zipapp.create_archive(target, buffer, compressed=True)
    return buffer.getvalue()


def package_archive_b64() -> str:
    return base64.b64encode(package_archive_bytes()).decode("ascii")


def iter_package_sources() -> list[tuple[str, str]]:
    """Every Python source file in the application as ``(name, text)``."""
    target = Path(launcher_path())
    if target.is_file():
        with zipfile.ZipFile(target) as archive:
            return [
                (name, archive.read(name).decode("utf-8"))
                for name in sorted(archive.namelist())
                if name.endswith(".py")
            ]
    root = Path(target)
    return [
        (str(path.relative_to(root)), path.read_text())
        for path in sorted(root.rglob("*.py"))
    ]
