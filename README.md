# Max Layout

Photonic, RF, and mixed TFLN layout editor with parameterized components,
automated test blocks, E-beam write-field planning, and GDS export.

Version 1 (V1) · **Built by Ali Khalatpour — Piris Labs** · MIT licensed

---

## Quick start

### Windows — recommended one-click package

1. Download [`Max Layout Windows.zip`](https://github.com/PirisLabs/max-layout/raw/main/Max%20Layout%20Windows.zip).
2. Choose **Extract All** in File Explorer. Do not run it from inside the ZIP.
3. Double-click **Max Layout Windows.cmd** in the extracted folder.

The first launch finds a compatible 64-bit Python or installs Python 3.12,
creates a private Max Layout environment, installs `PySide6`, `numpy`, and
`gdstk`, and opens the editor. Later launches reuse that environment and
normally open immediately. No command line or separate dependency installation
is required.

### macOS / Linux

Download [`Max Layout.pyz`](https://github.com/PirisLabs/max-layout/raw/main/Max%20Layout.pyz),
then run it from the folder where you saved it:

```bash
python3 "Max Layout.pyz"
```

The cross-platform archive installs its three runtime packages automatically on
first launch. Python 3.9 or newer is required up front for this manual route.

More detail: [macOS](#macos-step-by-step) · [Windows](#windows-step-by-step) ·
[Troubleshooting](#troubleshooting) · [Building from source](#building-from-source)

---

## What file do I run?

On Windows, extract `Max Layout Windows.zip` and double-click
`Max Layout Windows.cmd`. On macOS or Linux, run `Max Layout.pyz` with Python.

The `.pyz` inside the Windows package is the same complete application archive:
component generators, menus, defaults, Boolean tools, lattice tools, and
GDS/project export code. The other small Windows files provide the first-run
installer and double-click entry points.

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

1. Download [`Max Layout Windows.zip`](https://github.com/PirisLabs/max-layout/raw/main/Max%20Layout%20Windows.zip).
2. Right-click the downloaded ZIP and choose **Extract All**.
3. Open the extracted folder and double-click `Max Layout Windows.cmd`.
4. Keep the setup window open during the first launch. It closes the setup work
   after the editor has started; any error remains visible and points to its log.

Max Layout keeps its private Windows environment and setup logs under:

```text
%LOCALAPPDATA%\PirisLabs\MaxLayout
```

The launcher installs Python for the current Windows user if no compatible
64-bit version exists. Administrator rights are not normally required. If you
prefer the manual route, download `Max Layout.pyz` and run:

```powershell
py "Max Layout.pyz"
```

## Troubleshooting

**`py` or `python3` is not found**

- macOS: install a current Python release from
  [python.org](https://www.python.org/downloads/macos/).
- Windows: reinstall Python and enable **Add Python to PATH**, or use `python`
  instead of `py` in the commands above.

**The window does not appear, or PySide6 fails to import**

For a manual Windows `.pyz` launch, run
`py -m pip install PySide6 numpy gdstk`. On macOS, use the `python3` command
shown above. Then run Max Layout again.

**The Windows first-run setup stops**

Read the final message in the setup window, then open the newest log under
`%LOCALAPPDATA%\PirisLabs\MaxLayout\logs`. Make sure the complete ZIP was
extracted before double-clicking the launcher.

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

The component right-click menu also provides two independent parameter-sweep
exports. **Lumerical sweep…** keeps one persistent FDTD session on one GPU and
hot-swaps geometry sequentially. **Lumerical sweep-multithread…** builds one
nominal FSP, loads it into persistent workers on the selected pre-provisioned
A100 nodes, and assigns sweep points concurrently. The parallel notebook uses
unique worker folders, atomic shared checkpoints for resume, one simulation per
GPU, per-worker licence cleanup, and CPU-only result aggregation. The node count
remains selectable and defaults to eight A100 nodes, but production exports
intentionally allow only one independent Lumerical process per GPU. Multiple
processes on one GPU remain unavailable until model-memory and licence preflight
is implemented. Multi-node notebooks require a compatible private remote
execution environment; that confidential launcher and its credentials are not
distributed in this repository or the public Windows package.

**Lumerical optimization…** is a separate 3D alignment + shape-adjoint export. Its first
page selects continuous geometry and grating-alignment parameters by their exact project-JSON names,
hard minimum/maximum bounds, a center wavelength, and an optimization
bandwidth; its second page reuses the material-stack, FDTD-domain, mesh, and GPU
controls. The notebook uses the official GPU-aware `lumopt2` API when the
installed Lumerical release provides it, with the bundled legacy LumOpt
shape-adjoint API as a genuine-adjoint compatibility fallback. Selected
`angle_theta` and `fiber_offset` values are optimized first with bounded GPU
forward solves because they move the excitation/measurement basis; the fiber
core/cladding, source port, and passive fiber-power plane remain concentric and
change together. Their best alignment is then frozen for the true shape-adjoint
device-geometry stage. During the blocking GPU run, the notebook tails a
lightweight remote JSONL stream and redraws every completed alignment and
shape-adjoint iteration with its linear objective and complete selected JSON
parameter vector. Reporting never launches an extra solve. Grating-coupler
optimization targets fiber-to-waveguide coupling efficiency. Symmetric 1×2 MMI
optimization uses upper-branch fundamental-TE power divided by incident
input-mode power, with the physically balanced target of 0.5, then reports both
branches, total transmission, and imbalance for validation. Integer tooth
counts, `port_sep`, waveguide receivers, meshes, material identifiers, and other
topology controls remain fixed. The notebook saves
only the best FSP plus compact convergence, parameter-patch, and response
artifacts; it does not save an FSP for every iteration.

For a **1×2 MMI**, the simulation setup uses three access-waveguide FDTD ports
(input, upper output, and lower output) with one editable effective-index target
and the same local fundamental-TE selection rule. Before solving, the notebook
shows Ex and Ey for all three ports and reports their selected effective indices.
An input power monitor placed before the input taper provides the normalization.
The primary single-run and sweep graph is therefore upper-output/input,
lower-output/input, and total-output/input in linear power. A separate secondary
panel divides each branch by total transmitted output only to diagnose whether
the device is balanced around 50/50; it is not the transmission result. MMI sweep files use `MMI-...`
names, and the primary sweep objective is upper-branch power divided by measured
input power. A single-run MMI notebook additionally records the longitudinal
field plane and plots separate solved `|Ex|` and `|Ey|` maps on one common
magnitude scale; sequential and multi-GPU sweeps deliberately omit that large
field monitor and retain only their compact power spectra. The same
stack-layer mesh factors and mesh orders used by ordinary exports are preserved
by both MMI sweep exporters. New 1×2 MMI exports default to 3 µm of bottom SiO2,
the patterned TFLN cross-section, conformal top oxide, and Air, with no silicon
substrate row; existing custom stacks remain untouched.

For non-MMI TFLN devices, the default stack is 2 µm silicon, 5 µm SiO2 BOX, 400 nm TFLN with a
200 nm etch, 1 µm conformal SiO2 cladding, and 1 µm top Air. The export window
includes Air in the material list and two synchronized FDTD views shown side by
side. The left XZ view is the process-stack cross-section and controls X/Z; the
right XY view is an equal-scale top/GDS view of the selected polygons and
controls X/Y. Their common X-min/X-max update together when either view is
dragged. The same domain can be entered numerically as six exact bounds or as
X/Y/Z center and size. Corner/edge dragging resizes it and dragging inside the
red box moves its center. The cross-section cameras stay locked while those
boundaries move, so the fixed stack/device geometry no longer appears to resize;
**Fit both previews** explicitly reframes them when requested. Signed boundary
offsets allow any boundary to sit inside Air or another layer. Domain edits never
modify the physical polygon dimensions.
A **Show me a 3D version of the file I have built** button opens
an interactive pre-export view of the exact polygons, stack, ports, separate
fiber geometry, material-colored layer faces, a material-name legend, and the
FDTD wireframe. It supports orbit, pan, wheel zoom, and independent visibility
checkboxes for device geometry, ports, fiber, the FDTD box, and every individual
active material-stack layer. Materials are rendered as colored volumes between
their lower and upper planes, including slab faces and patterned sidewalls,
rather than as colored top planes only. Very thick first/last background layers are
cropped by the chosen domain and continue through the PML.

Every generated Lumerical notebook manages the complete remote licence
lifecycle: acquire the required solver and HPC licences, build and run, save
and fetch results, close FDTD, release the HPC packs, and close the remote
session. Remote build, solve, analysis, and save stages are checked explicitly
so a hidden remote traceback cannot become a misleading missing-file message.

Every completed single run, sequential sweep, multi-GPU sweep, and adjoint
optimization also writes and fetches `summary.txt`. It records the exact source
JSON parameters while also presenting the major device values one per line.
Each report is divided into named Parameters, Material Stack and Mesh,
Simulation Settings, Sources/Ports/Monitors, and Results Summary sections.
Sweep summaries additionally include every axis/range, completed and failed
counts, separate peak-best and target-wavelength-best cases with their complete
parameters, and explicit FSP provenance. Adjoint summaries separate optimizer
FOM from the final best-design forward validation and record the optimizer,
bounds, derived editor patch, and retained artifact paths.
Grating-coupler exports use only standard waveguide and Z-axis FDTD ports,
with the Z-axis port passing through a separate 7° fiber geometry group. The
standard grating and GC-SOI property panels include **Grating tooth geometry**:
**curved** preserves the focusing arcs, while **rectangular** replaces the
curved face with a straight taper and makes every tooth span exactly the
taper-end width. Pitch, tooth count, uniform or apodized fill factors, terminal
extension, fiber alignment, and simulation ports remain unchanged when this
geometry choice is switched.
With either choice, the starter fiber axis is placed on the grating side using the editable **Fiber
offset (µm)** parameter (project JSON key `fiber_offset`). It is a signed
distance on the component's local X axis from the geometry-exact first flare
boundary to the fiber bottom center; local Y remains zero. The standard grating
default is 5 µm. **Angle theta (degrees)** (project JSON key `angle_theta`)
controls the complete fiber tilt; changing it updates the fiber core/cladding,
source port, passive fiber-power plane, rotation offset, and concentric plane
positions together. Both parameters are available in sequential and multi-GPU
sweeps (`TH` and `FO`) and in the optimization exporter.
The standalone grating straight lead defaults to 5 µm, and its waveguide FDTD
port defaults to 3 µm inward from the external lead end using the editable
**FDTD port offset from waveguide end (µm)** parameter.
The export centers the upward exit monitor near that port and records linear
fiber-to-waveguide coupling efficiency versus wavelength.
There is no synthetic “fiber port” object: the notebook creates the core/cladding
structure group and the standard FDTD port independently, matching the Ansys example. The
quarter-wavelength button is a convenient starting point, but manually typed or
dragged X/Y/Z bounds are honored exactly; export reports a clear validation error
if a source, port, or monitor would touch the chosen domain. Substrate/top-cladding films
and ported waveguide continuations
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
`src/__main__.py` as the entry point.

After rebuilding the `.pyz`, create the extract-and-double-click Windows
distribution with:

```bash
python3 build_windows_bundle.py
```

That writes `Max Layout Windows.zip` from an explicit public-file allowlist.
The bundle contains Max Layout and its Windows bootstrap files only.

To run straight from source without building:

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
