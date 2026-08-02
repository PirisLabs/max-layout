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
- E-beam write-field creation and GDS, project JSON, FTEXT, and field export.
- CPU thread control and optional OpenGL canvas acceleration.
- Cached component geometry and layer-batched rendering for fast generation of
  repeated arrays and large photonic-crystal structures.

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
