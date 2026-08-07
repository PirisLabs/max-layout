# Max Layout

Photonic, RF, and mixed TFLN layout editor with parameterized components,
automated test blocks, E-beam write-field planning, and GDS export.

Version 1 (V1) · **Built by Ali Khalatpour — Piris Labs** · MIT licensed

---

## Quick start

**1. Download** [`Max Layout.pyz`](https://github.com/PirisLabs/max-layout/raw/main/Max%20Layout.pyz)
— that one file is the whole application.

**2. Run it** from the folder where you saved it:

```bash
python3 "Max Layout.pyz"        # macOS / Linux
py "Max Layout.pyz"             # Windows
```

That is all. On first launch Max Layout installs the Python packages it needs
(`PySide6`, `numpy`, `gdstk`) automatically, so the first start takes longer
than later ones.

Only Python 3.9 or newer is required up front. If you do not have it, install
from [python.org](https://www.python.org/downloads/) — on Windows, check
**Add Python to PATH** during installation.

More detail: [macOS](#macos-step-by-step) · [Windows](#windows-step-by-step) ·
[Troubleshooting](#troubleshooting) · [Building from source](#building-from-source)

---

## What file do I run?

```text
Max Layout.pyz
```

The `.pyz` contains every Max Layout program file: component generators, menus,
defaults, Boolean tools, lattice tools, and GDS/project export code. You do not
need to download anything else to run it.

The readable source for that archive lives in `src/`. See
[Building from source](#building-from-source) if you want to modify the program.

`PySide6`, `numpy`, and `gdstk` contain operating-system-specific binary code,
so they are not embedded in the cross-platform `.pyz`. Max Layout installs them
for you on first run; the sections below give the manual command in case that
automatic step cannot reach the network or lacks permission.

## macOS step by step

1. Open **Terminal**.
2. Confirm Python is available:

```bash
python3 --version
```

3. Run Max Layout from the folder containing the file:

```bash
python3 "Max Layout.pyz"
```

Or give the full path, for example:

```bash
python3 "$HOME/Downloads/Max Layout.pyz"
```

If the file is stored elsewhere, drag `Max Layout.pyz` from Finder into the
Terminal after typing `python3 `; macOS will insert the correct path.

4. Only if the automatic dependency install fails, install the packages
   yourself and run again:

```bash
python3 -m pip install --user PySide6 numpy gdstk
```

## Windows step by step

1. Install Python 3.9 or newer from
   [python.org](https://www.python.org/downloads/). During installation, check
   **Add Python to PATH**.
2. Open **PowerShell** in the folder containing `Max Layout.pyz`.
3. Run Max Layout:

```powershell
py "Max Layout.pyz"
```

You can also use the complete path:

```powershell
py "C:\Users\YOUR_NAME\Desktop\Max Layout\Max Layout.pyz"
```

4. Only if the automatic dependency install fails:

```powershell
py -m pip install PySide6 numpy gdstk
```

## Troubleshooting

**`py` or `python3` is not found**

- macOS: install a current Python release from
  [python.org](https://www.python.org/downloads/macos/).
- Windows: reinstall Python and enable **Add Python to PATH**, or use `python`
  instead of `py` in the commands above.

**The window does not appear, or PySide6 fails to import**

Install the packages manually with the `pip` command for your platform above,
then run Max Layout again.

**Large layouts feel slow**

Raise the thread count at **Layout → Performance / CPU Threads…**, or set
`PHOTONIC_CPU_THREADS` before launching. See
[Environment variables](#environment-variables).

## Main capabilities

- Photonic, RF, and mixed TFLN layout components.
- MMI, MZI, grating coupler, ring, racetrack, CPW, and segmented-electrode tools.
- One consolidated Photonic crystal component with selectable rectangle,
  ellipse, or hexagon slab; square or triangular hole lattice; circular or
  elliptical holes; crystal size; bulk or line-defect-waveguide mode; and
  positive hole-mask or negative Boolean-subtracted slab output.
- Boolean Union, Difference, Intersection, and XOR.
- Image-to-GDS vectorization for PNG, JPEG, BMP, and TIFF photographs or
  drawings, with physical output width, threshold, inversion, detail, and
  polygon-merging controls. The result is a normal editable GDS component.
  For ordinary photographs, start with **Photo edges — recommended**, detail
  192, edge strength 38, polygon merging enabled, and a 500 µm output width.
  Camera/EXIF rotation is applied automatically.
- Generic RF and photonic three-step test-block wizards: choose a library component and label settings, choose common or full parameters, then enter uniform Start/Stop/Step ranges or explicit nonuniform lists such as `500, 1000, 2000`. Each generated device receives visible parameter text such as `CP,Gap=3,Length=500`, anchored consistently from its true top-left corner using editable text height and X/Y offsets (20 µm default height). Length-like values define aligned columns, while every combination of gap, width, and other values defines aligned rows. Per-column widths, per-row heights, and label geometry preserve the requested edge spacing without a manual column-count setting.
- The initial CPW test-block scan uses signal widths `10, 11, 12, 13, 14, 15 µm` as six rows and lengths `500, 1000, 2000 µm` as three columns.
- Tapered CPW and symmetric CPW taper scans use `500, 1000, 2000 µm` as their initial taper-length columns.
- RF taper test blocks ask whether the center is plain CPW or a T electrode. Their generated sequence is probe CPW → input taper → center section → output taper → probe CPW. Probe CPW leads default to 100 µm; extra input/output transitions default to 0 µm. A taper-contained T electrode defaults to `inner_flat_length = 0 µm`, `end_flat_length = 0 µm`, and `transition_length = 0 µm`.
- A standalone T electrode defaults to `inner_flat_length = 0 µm`, `end_flat_length = 100 µm`, and `transition_length = 0 µm`. T-electrode GUI previews use an exact preview-only metal union instead of dropping dense polygons, so the canvas matches the exported GDS while remaining lighter to render.
- Three-step RF test-block wizard: choose an RF library component, select common
  or full physical parameters, enter numeric ranges or taper-profile values,
  and place the resulting scan with configurable edge-to-edge spacing.
- 4-inch (100 mm) wafer outline with the standard 32.5 mm primary flat.
- Test-block sweep tables let users select the major parameters and enter an
  inclusive start, stop, and step for each selected sweep before generation.
- Smart Sketch removes hand shake and fits each stroke to a clean standard
  component: Straight, S-bend, Euler bend, circular ring, elliptical ring, or
  triangle-to-grating-coupler. Multi-stroke drawings create multiple editable
  standard components rather than exporting noisy freehand paths.
- Persistent User modules created from right-clicked geometry.
- Port snapping, arrays, grouping, rotation, mirroring, ruler, layers, and labels.
- Right-click Lumerical notebook export for a component, the current selection,
  a complete group/module/array, or the entire layout. Exported notebooks embed
  polygon geometry directly, so no GDS sidecar is required.
- A dedicated **Ports & monitors** section in the left component library with
  movable standard FDTD ports, a separate Ansys tilted fiber core/cladding geometry group, power monitors, mode-expansion
  monitors, and field-profile monitors. Port data uses the
  Lumerical compact-model names `name`, `dir`, `loc`, `pos`, and `order`; port
  footprints are visible in the editor but are never written to GDS. Common
  waveguides, tapers, bends, and MMIs receive movable endpoint starter ports.
  A grating coupler also receives its waveguide port, separate fiber geometry,
  and fiber-axis FDTD port; every starter object remains editable/removable.
- Editable bottom-to-top TFLN, SOI, Al2O3, SiO2, silicon, and metal material
  stacks with a named exported cross-section, etch depth, and waveguide
  sidewall angle. Top cladding rows can be conformal so SiO2 fills etched gaps
  before covering the device. A zero thickness disables that material.
  Generated 3D FDTD notebooks use GPU execution by default.
- E-beam write-field creation and GDS, project JSON, FTEXT, and field export.
- CPU thread control and optional OpenGL canvas acceleration.
- Cached component geometry and layer-batched rendering for fast generation of
  repeated arrays and large photonic-crystal structures.

## Lumerical simulation notebooks

Open **Ports & monitors** in the left component library, then double-click or
use **Add selected component** to place a waveguide FDTD port, a separate fiber
core/cladding geometry group, a standard Z-axis FDTD port through that fiber,
power monitor, mode-expansion monitor, or field-profile monitor. These objects
move and rotate like normal layout geometry and expose their position, span,
offset distance, mode, and fiber-port settings in the Properties panel.
X-normal objects appear as vertical lines in the top view, Y-normal objects as
horizontal lines, and Z-normal objects as top-view areas. Surface monitors use
explicit x/y/z spans with the normal-axis dimension set to zero.
They are saved in the project but omitted from every GDS export.

Right-click a component on the canvas or in the project tree and open
**Lumerical simulation** to export it. The first geometry choice includes the
clicked component together with the placed ports and monitors; selection,
group/module/array, component-only, and entire-layout choices remain available.
When a standalone port or monitor is right-clicked, the first choice instead
uses the nearest physical device plus all placed simulation objects. A
port/monitor-only choice remains available for setup inspection, but is clearly
marked as containing no device geometry.

Choose **Export simulation notebook** to select the geometry scope, GDS layers,
material stack, film thicknesses, etch depths, wavelengths, mesh, padding, and compute resource. The `.ipynb`
contains the geometry, the exact compact-model `PORTS_JSON` mapping, the lumapi
loader, 3D FDTD construction, verified `.fsp` saving, GPU discovery/system checks, and an
optional run cell (enabled by default for new exports). Every simulation export is
strictly 3D; there is no 2D solver option. Before solving, every generated notebook—including
waveguides, tapers, MMIs, coplanar waveguides, gratings, groups, and complete layouts—displays
an `XY` top view, `XZ` side view, and `YZ` end view built from the exact embedded polygons,
films, etch depths, and sidewall angles sent to Lumerical. The combined verification image is
also saved with the results.

The default TFLN stack is 2 µm silicon, 5 µm SiO2 BOX, 400 nm TFLN with a
200 nm etch, 1 µm conformal SiO2 cladding, and 1 µm top Air. The export window
includes Air in the material list, a live XZ/YZ process-stack cross-section,
corner/edge dragging, and exact numeric X-min/X-max/Y-min/Y-max/Z-min/Z-max
FDTD boundaries. The cross-section camera stays locked while those boundaries
move, so the fixed stack/device geometry no longer appears to resize; **Fit
preview** explicitly reframes it when requested. Dragging inside the red box
translates its center in XZ/YZ without resizing; signed boundary offsets allow
any boundary to sit inside Air or another layer. Domain edits never modify the physical polygon dimensions.
A **Show me a 3D version of the file I have built** button opens
an interactive pre-export view of the exact polygons, stack, ports, separate
fiber geometry, material-colored layer faces, a material-name legend, and the
FDTD wireframe. It supports orbit, pan, wheel zoom, and independent visibility
checkboxes for device geometry, ports, fiber, the FDTD box, and every individual
active material-stack layer. Materials are rendered as colored volumes between
their lower and upper planes, including slab faces and patterned sidewalls,
rather than as colored top planes only. Very thick first/last background layers are
cropped by the chosen domain and continue through the PML.

Every generated notebook follows the Piris Lambda licence
lifecycle: seed Shared Web, roam three HPC Packs, build/run, save and fetch all
results, close FDTD, check the packs back in, then close the remote session.
Remote build, solve, analysis, and save stages are checked explicitly so a hidden remote
traceback cannot turn into misleading missing-file messages at the end.
Grating-coupler exports use only standard waveguide and Z-axis FDTD ports,
with the Z-axis port passing through a separate 7° fiber geometry group. The
starter fiber axis is placed on the grating side after the complete taper using
the editable **Fiber offset after taper toward grating (µm)** parameter (5 µm by default).
The standalone grating straight lead defaults to 5 µm, and its waveguide FDTD
port defaults to 3 µm inward from the external lead end using the editable
**FDTD port offset from waveguide end (µm)** parameter.
The export centers the upward exit monitor near that port,
record fiber coupling efficiency in dB versus wavelength, and save the natural
3D far-field radiation pattern in polar coordinates with its peak exit angles.
There is no synthetic “fiber port” object: the notebook creates the core/cladding
structure group and the standard FDTD port independently, matching the Ansys example. The FDTD
region keeps at least a quarter-wavelength clearance from ordinary device
features, while substrate/top-cladding films and ported waveguide continuations
extend through their PML boundaries as in the official Ansys example. New
component exports default to 2 µm transverse clearance, placing endpoint ports
2 µm from the matching PML boundary. Ported waveguides and the outer material
stack continue 1 µm beyond the FDTD boundaries by default on X, Y, top, and
bottom; **Geometry overlap beyond FDTD boundary** makes this editable. Exported cross-section rows use
Lumerical Layer Builder so
their sidewall angles are physical, rather than just notebook metadata.
New simulations default to a 1.25–1.35 µm wavelength sweep. Automatic starter
ports can be dragged independently in the top-view editor; doing so detaches
their automatic position without removing them from the component export group.
Material rows are ordered bottom-to-top; rows with a thickness of `0 µm` are
intentionally skipped. TFLN is created as a dispersive anisotropic sampled
material from the Zelmon/Moretti model with editable crystal cut and temperature.

## Optional AI assistant

Max Layout includes an optional assistant panel that can edit a layout or
propose source changes. It is off unless you supply an OpenAI API key, either in
the panel's password-masked field or through the `OPENAI_API_KEY` environment
variable. No key is stored in the program.

Be aware of what leaves your machine when you use it: layout mode sends your
current component list to the OpenAI API, and source mode sends the application
source. Source mode never overwrites your installation — it validates the
proposed edits, compiles them, and writes the changed files to a new timestamped
folder for you to review.

## Environment variables

| Variable | Effect |
| --- | --- |
| `PHOTONIC_CPU_THREADS` | Thread count for NumPy/BLAS geometry work and export workers. Defaults to 8, capped by CPU count. |
| `PHOTONIC_USE_GPU` | Set to `0`, `false`, `no`, or `off` to skip GPU backend detection. |
| `PHOTONIC_LAYOUT_USE_OPENGL` | Set to `1` to enable the OpenGL viewport when Qt supports it. |
| `OPENAI_API_KEY` | Key for the optional AI assistant. |
| `OPENAI_MODEL` | Overrides the assistant's default model. |

## Building from source

`Max Layout.pyz` is built from the package in `src/`. Rebuild it with:

```bash
python3 build.py
```

That byte-compiles every module, then writes `Max Layout.pyz` as a zipapp with
`src/__main__.py` as the entry point. To run straight from source without
building:

```bash
python3 src
```

```text
src/
├── __main__.py                  launcher and headless worker modes
└── max_layout/
    ├── acceleration.py          CPU thread and GPU backend selection
    ├── bootstrap.py             runtime dependency installation
    ├── constants.py             layers, component kinds, default parameters
    ├── runtime.py               locating the running app for subprocesses
    ├── utils.py                 sweep and parsing helpers
    ├── modules_db.py            persistent user modules
    ├── params.py                parameter resizing rules
    ├── ports.py                 port positions and attachment solving
    ├── llm.py                   optional OpenAI assistant
    ├── backend/                 gdstk component library (GC, MMI, bends)
    ├── geometry/                transforms, shapes, Euler bends, RF tapers,
    │                            landmark point sets
    ├── gds/                     primitives, couplers, e-beam fields,
    │                            geometry construction, export
    └── ui/                      theme, graphics items, dialogs, main window
```

Modules are layered so imports only ever point downward — `constants` and
`backend` at the bottom, then `geometry`, `ports`, `gds`, and `ui` on top. There
are no import cycles.

## License

Max Layout is released under the MIT License; see the `LICENSE` file for the
exact terms. Use, modification, redistribution, and commercial use are all
permitted provided the copyright notice and permission notice are retained:

> Copyright (c) 2026 Ali Khalatpour — Piris Labs

The software is provided without warranty of any kind.

## Notes for contributors

Do not commit passwords, API keys, or fabrication credentials. Keep generated
fabrication output such as `.gds` mask files out of version control; the
repository's `.gitignore` already excludes them.
