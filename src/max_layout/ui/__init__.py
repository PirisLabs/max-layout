"""Qt user interface layer.

The GUI dependency check runs on import, before any submodule imports PySide6,
mirroring the ordering of the original single-file build.
"""

from __future__ import annotations

from ..bootstrap import _ensure_native_gui_dependencies

_ensure_native_gui_dependencies()
