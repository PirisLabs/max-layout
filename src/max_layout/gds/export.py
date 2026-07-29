"""GDS and standalone-Python export."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pprint

from .. import __version__
from ..gds.build import library_bbox_and_center, resolve_and_build
from ..runtime import package_archive_b64


def generate_python_export(components: list[dict[str, Any]]) -> str:
    library = resolve_and_build(components)
    bbox, center = library_bbox_and_center(library)
    version = __version__
    component_literal = pprint.pformat(components, width=120, sort_dicts=False)
    # The exported script must be self-contained.  When the editor was one
    # file that meant embedding that file; now it means embedding a zipapp of
    # the whole package, which Python imports directly off sys.path.
    embedded_archive_b64 = package_archive_b64()
    return f'''from __future__ import annotations

import base64
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

OUTPUT_GDS_FILE = Path(__file__).with_name("generated_layout.gds")
CENTER_LAYOUT_AT_ORIGIN = False
CURRENT_GEOMETRY_CENTER_UM = {center!r}
CURRENT_BOUNDING_BOX_UM = {bbox!r}
LAYER_NAME_MAP = {{1: "WG", 2: "GC", 3: "Marker", 4: "RF", 5: "Probe", 6: "Ebeam"}}
COMPONENTS = {component_literal}

EMBEDDED_EDITOR_PYZ_B64 = {embedded_archive_b64!r}
_archive = Path(tempfile.gettempdir()) / "max_layout_embedded_{version}.pyz"
if not _archive.exists():
    _archive.write_bytes(base64.b64decode(EMBEDDED_EDITOR_PYZ_B64))
if str(_archive) not in sys.path:
    sys.path.insert(0, str(_archive))

from max_layout.gds.build import library_bbox_and_center, resolve_and_build


def build_layout():
    components = deepcopy(COMPONENTS)
    initial_library = resolve_and_build(components)
    _, initial_center = library_bbox_and_center(initial_library)

    if CENTER_LAYOUT_AT_ORIGIN:
        for component in components:
            component["x"] = float(component["x"]) - initial_center[0]
            component["y"] = float(component["y"]) - initial_center[1]

    library = resolve_and_build(components)
    bounding_box, geometry_center = library_bbox_and_center(library)
    return library, components, bounding_box, geometry_center


if __name__ == "__main__":
    library, resolved_components, bounding_box, geometry_center = build_layout()
    library.write_gds(str(OUTPUT_GDS_FILE))
    print(f"Geometry center (µm): {{geometry_center}}")
    print(f"Bounding box (µm): {{bounding_box}}")
    print(f"Wrote: {{OUTPUT_GDS_FILE.resolve()}}")
'''


def _worker_export_gds(project_file: str, output_file: str) -> None:
    payload = json.loads(Path(project_file).read_text())
    components = payload.get("components", payload)
    if not isinstance(components, list):
        raise ValueError("Project does not contain a component list.")
    library = resolve_and_build(components)
    library.write_gds(output_file)


def _worker_export_python(project_file: str, output_file: str) -> None:
    payload = json.loads(Path(project_file).read_text())
    components = payload.get("components", payload)
    if not isinstance(components, list):
        raise ValueError("Project does not contain a component list.")
    Path(output_file).write_text(generate_python_export(components))
