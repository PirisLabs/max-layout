"""Max Layout — photonic and RF layout editor.

Built by Ali Khalatpour — Piris Labs.

Importing this package installs any missing runtime dependencies before the
submodules pull in gdstk and numpy, matching the ordering the original
single-file build relied on.
"""

from __future__ import annotations

from .bootstrap import _ensure_runtime_dependencies

_ensure_runtime_dependencies()

__all__ = ["__version__"]
__version__ = "V1"
