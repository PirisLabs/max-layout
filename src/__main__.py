#!/usr/bin/env python3
"""Launcher for Max Layout.

Runs the desktop editor, or one of the headless worker modes the editor
re-invokes as a subprocess.
"""

import sys

from max_layout.acceleration import configure

configure()

WORKERS = {
    "--worker-export-gds": ("max_layout.gds.export", "_worker_export_gds", "PROJECT_JSON OUTPUT_GDS"),
    "--worker-export-python": ("max_layout.gds.export", "_worker_export_python", "PROJECT_JSON OUTPUT_PY"),
    "--worker-llm": ("max_layout.llm", "_worker_llm_assistant", "REQUEST_JSON RESPONSE_JSON"),
}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode in WORKERS:
        module_name, function_name, usage = WORKERS[mode]
        if len(sys.argv) != 4:
            raise SystemExit(f"Usage: app.pyz {mode} {usage}")
        module = __import__(module_name, fromlist=[function_name])
        getattr(module, function_name)(sys.argv[2], sys.argv[3])
        return
    from max_layout.ui.app import native_main

    native_main()


if __name__ == "__main__":
    main()
