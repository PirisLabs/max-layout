"""Reusable gdstk photonic component library.

Extracted from the base64 blob that previously lived in combined_editor.py
and was exec()d into a synthetic module at import time.
"""

from . import photonic_components_gdstk
from . import photonic_components_gdstk as backend

__all__ = ["backend", "photonic_components_gdstk"]
