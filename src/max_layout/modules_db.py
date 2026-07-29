"""Persistent user-defined component modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

NATIVE_MODULE_DB = Path.home() / ".photonic_layout_native_modules.json"


def load_native_modules() -> dict[str, Any]:
    try:
        data = json.loads(NATIVE_MODULE_DB.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_native_modules(modules: dict[str, Any]) -> None:
    NATIVE_MODULE_DB.write_text(json.dumps(modules, indent=2))
