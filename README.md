# Max Layout

Version 1 (V1)

**Built by Ali Khalatpour — Piris Labs**

Max Layout is provided free of charge to use and share with attribution.
Copyright remains with Ali Khalatpour / Piris Labs. The software is provided
without warranty.

## What file do I run?

Run this single application file:

```text
Max Layout.pyz
```

The `.pyz` contains all Max Layout program source files, component generators,
menus, defaults, Boolean tools, lattice tools, and GDS/project export code. You
do not need to download separate Max Layout Python source files.

The readable source for that archive lives in `src/`; see
[Source layout](#source-layout) below if you want to modify the program.

The computer still needs Python and three platform-installed packages:
`PySide6`, `numpy`, and `gdstk`. These include operating-system-specific binary
code and therefore are not embedded in the same cross-platform `.pyz`.

## macOS instructions

1. Open **Terminal**.
2. Confirm Python is available:

```bash
python3 --version
```

3. Install the required packages once:

```bash
python3 -m pip install --user PySide6 numpy gdstk
```

4. Run Max Layout:

```bash
python3 "/Users/alimac/Desktop/Latest Codes/Max Layout.pyz"
```

If the file is stored elsewhere, drag `Max Layout.pyz` from Finder into the
Terminal after typing `python3 `; macOS will insert the correct path.

## Windows instructions

1. Install Python 3.9 or newer from [python.org](https://www.python.org/downloads/).
   During installation, check **Add Python to PATH**.
2. Open **PowerShell** in the folder containing `Max Layout.pyz`.
3. Install the required packages once:

```powershell
py -m pip install PySide6 numpy gdstk
```

4. Run Max Layout:

```powershell
py "Max Layout.pyz"
```

You can also use the complete path:

```powershell
py "C:\Users\YOUR_NAME\Desktop\Max Layout\Max Layout.pyz"
```

## If `py` or `python3` is not found

- macOS: install a current Python release from
  [python.org](https://www.python.org/downloads/macos/).
- Windows: reinstall Python and enable **Add Python to PATH**, or use `python`
  instead of `py` in the commands above.

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
- Parameterized test blocks and selectable parameter sweeps.
- Test-block sweep tables let users select the major parameters and enter an
  inclusive start, stop, and step for each selected sweep before generation.
- Smart Sketch converts rough multi-stroke drawings into editable standard
  components. Lines favor straight waveguides, curved strokes favor Euler
  bends, circular loops favor rings, triangular loops favor grating couplers,
  and branched multi-stroke paths favor an MZI. Generated sections remain
  clickable so recognition results can be corrected numerically.
- Persistent User modules created from right-clicked geometry.
- Port snapping, arrays, grouping, rotation, mirroring, ruler, layers, and labels.
- E-beam write-field creation and GDS, project JSON, FTEXT, and field export.
- CPU thread control and optional OpenGL canvas acceleration.
- Cached component geometry and layer-batched rendering for fast generation of
  repeated arrays and large photonic-crystal structures.

## Source layout

`Max Layout.pyz` is built from the package in `src/`. Rebuild it with:

```bash
python3 build.py
```

That byte-compiles every module, then writes `Max Layout.pyz` as a zipapp with
`src/__main__.py` as the entry point. To run straight from source without
building, use `python3 src`.

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
exact terms. Use, modification, redistribution, and commercial use are
permitted provided the copyright notice and permission notice are retained:

> Copyright (c) 2026 Ali Khalatpour — Piris Labs

The software is provided without warranty of any kind.

## Important

Do not publish passwords, API keys, or fabrication credentials in a GitHub
repository. Keep generated fabrication output such as `.gds` mask files out of
version control.
