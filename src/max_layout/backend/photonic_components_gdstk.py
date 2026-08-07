"""
Reusable gdstk photonic component library.

Included components
-------------------
1. Focusing grating coupler
2. 1 x 2 MMI
3. Smooth S-bend
4. Circular / partial-Euler / full-Euler bend

Coordinate convention
---------------------
0 degrees   -> +x
90 degrees  -> +y
180 degrees -> -x
270 degrees -> -y

All dimensions are in micrometers.

Endpoint-return convention
--------------------------
Single-output components return:
    end_center
    end_orientation_deg

The 1 x 2 MMI returns:
    upper_end_center
    lower_end_center
    end_orientation_deg
"""

import numpy as np
import gdstk


__all__ = [
    "make_focusing_gc_gds",
    "make_1x2_mmi_gdstk",
    "make_s_bend_gdstk",
    "make_euler_bend_gdstk",
]


def make_focusing_gc_gds(
    pitch,
    fill_factor,
    N,
    alpha_t,
    taper_L,
    wg_width=1.0,
    wg_length=20.0,
    wg_end_center=(0.0, 0.0),
    orientation_deg=0.0,
    layer=1,
    datatype=0,
    cell_name="FOCUSING_GC",
    gds_file="focusing_gc.gds",
    tolerance=0.0005,
    L_extra=0.0,
):
    """
    Generate a focusing grating coupler.

    Coordinate convention
    ---------------------
    wg_end_center:
        Center coordinate of the external end of the straight waveguide.

    orientation_deg:
        Direction from the external waveguide end toward the grating.

        0 deg   -> grating extends toward +x
        90 deg  -> grating extends toward +y
        180 deg -> grating extends toward -x
        270 deg -> grating extends toward -y

    fill_factor:
        LN tooth width divided by pitch. May be one scalar or an N-element
        array for an apodized grating.

        tooth_width = pitch * fill_factor
        gap_width   = pitch * (1 - fill_factor)

    All dimensions are in micrometers.

    Returns
    -------
    lib : gdstk.Library
        Library containing the grating-coupler cell.
    cell : gdstk.Cell
        Generated focusing grating-coupler cell.
    end_center : tuple
        Global centerline coordinate at the final grating edge.
    end_orientation_deg : float
        Direction from the external waveguide end toward the grating.
    meta : dict
        Geometry parameters and useful global coordinates.

    Notes
    -----
    ``end_center`` is the geometric center of the final grating edge.
    The physical waveguide port remains ``wg_end_center``.
    """

    N = int(N)

    if N <= 0:
        raise ValueError("N must be greater than zero.")

    if wg_length <= 0:
        raise ValueError("wg_length must be positive.")

    if wg_width <= 0:
        raise ValueError("wg_width must be positive.")

    if L_extra < 0:
        raise ValueError("L_extra cannot be negative.")

    alpha_rad = np.deg2rad(alpha_t)
    orientation_rad = np.deg2rad(orientation_deg)

    if not 0 < alpha_t < 180:
        raise ValueError("alpha_t must be between 0 and 180 degrees.")

    wg_end_center = np.asarray(wg_end_center, dtype=float)

    if wg_end_center.shape != (2,):
        raise ValueError("wg_end_center must be an (x, y) coordinate.")

    # ── Convert scalar or array pitch to an N-element array ──────────────────
    pitch_array = np.asarray(pitch, dtype=float)

    if pitch_array.ndim == 0:
        pitch_array = np.full(N, float(pitch_array))
    elif pitch_array.size != N:
        raise ValueError("When pitch is an array, its length must equal N.")

    # ── Convert scalar or array fill factor to an N-element array ────────────
    fill_array = np.asarray(fill_factor, dtype=float)

    if fill_array.ndim == 0:
        fill_array = np.full(N, float(fill_array))
    elif fill_array.size != N:
        raise ValueError(
            "When fill_factor is an array, its length must equal N."
        )

    if np.any(pitch_array <= 0):
        raise ValueError("All pitch values must be positive.")

    if np.any((fill_array <= 0) | (fill_array >= 1)):
        raise ValueError("All fill factors must be between 0 and 1.")

    tooth_widths = pitch_array * fill_array
    gap_widths = pitch_array * (1.0 - fill_array)

    # ── Local-coordinate geometry ────────────────────────────────────────────
    #
    # External waveguide end:       (0, 0)
    # Taper/waveguide connection:   (wg_length, 0)
    # Grating extends toward:       +x
    #
    wg_connection_local = np.array([wg_length, 0.0])

    focus_offset = (wg_width / 2) / np.tan(alpha_rad / 2)

    focus_local = np.array(
        [
            wg_length - focus_offset,
            0.0,
        ]
    )

    radius_at_wg = (wg_width / 2) / np.sin(alpha_rad / 2)

    if taper_L <= radius_at_wg:
        raise ValueError(
            f"taper_L must be larger than {radius_at_wg:.4f} µm "
            "for this waveguide width and aperture angle."
        )

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)

    # Straight waveguide from external end to taper.
    waveguide = gdstk.rectangle(
        (0.0, -wg_width / 2),
        (wg_length, wg_width / 2),
        layer=layer,
        datatype=datatype,
    )

    # Focusing taper.
    taper = gdstk.ellipse(
        center=tuple(focus_local),
        radius=taper_L,
        initial_angle=-alpha_rad / 2,
        final_angle=alpha_rad / 2,
        layer=layer,
        datatype=datatype,
        tolerance=tolerance,
    )

    input_geometry = gdstk.boolean(
        [waveguide, taper],
        [],
        "or",
        layer=layer,
        datatype=datatype,
    )

    cell.add(*input_geometry)

    # ── Focusing grating teeth ───────────────────────────────────────────────
    radius = float(taper_L)

    tooth_inner_radii = []
    tooth_outer_radii = []

    for i in range(N):

        gap = gap_widths[i]
        tooth = tooth_widths[i]

        # Each period begins with an etched gap.
        radius += gap

        inner_radius = radius
        outer_radius = inner_radius + tooth

        tooth_polygon = gdstk.ellipse(
            center=tuple(focus_local),
            radius=outer_radius,
            inner_radius=inner_radius,
            initial_angle=-alpha_rad / 2,
            final_angle=alpha_rad / 2,
            layer=layer,
            datatype=datatype,
            tolerance=tolerance,
        )

        cell.add(tooth_polygon)

        tooth_inner_radii.append(inner_radius)
        tooth_outer_radii.append(outer_radius)

        radius = outer_radius

    # Optional solid terminal sector, matching the thick output arc in the
    # official Ansys SOI focusing-grating geometry.
    if L_extra > 0.0:
        output_sector = gdstk.ellipse(
            center=tuple(focus_local),
            radius=radius + float(L_extra),
            inner_radius=radius,
            initial_angle=-alpha_rad / 2,
            final_angle=alpha_rad / 2,
            layer=layer,
            datatype=datatype,
            tolerance=tolerance,
        )
        cell.add(output_sector)

    # ── Rotate and translate complete grating coupler ────────────────────────
    #
    # Rotation occurs around the local external waveguide end at (0, 0).
    # Translation then places this point at wg_end_center.
    #
    for polygon in cell.polygons:
        polygon.rotate(orientation_rad, center=(0.0, 0.0))
        polygon.translate(
            float(wg_end_center[0]),
            float(wg_end_center[1]),
        )

    # Coordinate transformation helper.
    rotation_matrix = np.array(
        [
            [np.cos(orientation_rad), -np.sin(orientation_rad)],
            [np.sin(orientation_rad),  np.cos(orientation_rad)],
        ]
    )

    def local_to_global(point):
        point = np.asarray(point, dtype=float)
        return wg_end_center + rotation_matrix @ point

    wg_connection_global = local_to_global(wg_connection_local)
    focus_global = local_to_global(focus_local)

    first_tooth_center_local = focus_local + np.array(
        [
            tooth_inner_radii[0]
            + 0.5 * tooth_widths[0],
            0.0,
        ]
    )

    final_grating_edge_local = focus_local + np.array(
        [
            tooth_outer_radii[-1],
            0.0,
        ]
    )

    first_tooth_center_global = local_to_global(
        first_tooth_center_local
    )

    final_grating_edge_global = local_to_global(
        final_grating_edge_local
    )

    # Explicit values returned for direct component chaining.
    # For a grating coupler, this is the geometric endpoint at the
    # final grating edge, not an additional waveguide port.
    end_center = tuple(float(v) for v in final_grating_edge_global)
    end_orientation_deg = float(orientation_deg % 360)

    if gds_file is not None:
        lib.write_gds(gds_file)

    meta = {
        "wg_end_center": tuple(float(v) for v in wg_end_center),
        "orientation_deg": float(orientation_deg % 360),

        "end_center": end_center,
        "end_orientation_deg": end_orientation_deg,

        "wg_connection_center": tuple(float(v) for v in wg_connection_global),
        "focus_center": tuple(float(v) for v in focus_global),
        "first_tooth_center": tuple(float(v) for v in first_tooth_center_global),
        "final_grating_edge_center": end_center,

        "pitch": pitch_array,
        "fill_factor": fill_array,
        "gap_widths": gap_widths,
        "tooth_widths": tooth_widths,

        "grating_length": float(np.sum(pitch_array)),
        "L_extra": float(L_extra),
        "taper_radius": float(taper_L),
        "first_tooth_inner_radius": float(tooth_inner_radii[0]),
        "last_tooth_outer_radius": float(tooth_outer_radii[-1]),
    }

    return lib, cell, end_center, end_orientation_deg, meta


def make_1x2_mmi_gdstk(
    mmi_width=6.0,
    mmi_length=29.0,
    wg_width=1.2,
    taper_width=2.7,
    input_taper_length=10.0,
    output_taper_length=10.0,
    input_length=6.0,
    output_length=6.0,
    port_sep=3.25,
    taper_power=1.0,
    taper_points=41,
    input_center=(0.0, 0.0),
    orientation_deg=0.0,
    layer=1,
    datatype=0,
    cell_name="MMI_1X2",
    gds_file=None,
    lib=None,
):
    """
    Generate a 1x2 MMI using gdstk.

    Coordinate convention
    ---------------------
    input_center:
        Global coordinate of the center of the external input-waveguide end.

    orientation_deg:
        Direction from the input port toward the MMI.

          0 deg -> +x
         90 deg -> +y
        180 deg -> -x
        270 deg -> -y

    taper_power:
        Taper profile exponent.

        taper_power = 1.0 gives a linear taper.

    Returns
    -------
    lib:
        gdstk.Library

    cell:
        gdstk.Cell containing the MMI.

    upper_end_center:
        Global center coordinate of the upper output port.

    lower_end_center:
        Global center coordinate of the lower output port.

    end_orientation_deg:
        Propagation direction at both output ports.

    ports:
        Dictionary containing global input and output-port information.

    meta:
        Dictionary containing device coordinates and dimensions.
    """

    # ── Parameter checks ─────────────────────────────────────────────────────
    if mmi_width <= 0 or mmi_length <= 0:
        raise ValueError("mmi_width and mmi_length must be positive.")

    if wg_width <= 0 or taper_width <= 0:
        raise ValueError("wg_width and taper_width must be positive.")

    if input_length < 0 or output_length < 0:
        raise ValueError(
            "input_length and output_length cannot be negative."
        )

    if input_taper_length <= 0 or output_taper_length <= 0:
        raise ValueError("Taper lengths must be positive.")

    if port_sep <= 0:
        raise ValueError("port_sep must be positive.")

    if taper_power <= 0:
        raise ValueError("taper_power must be positive.")

    taper_points = int(taper_points)

    if taper_points < 2:
        raise ValueError("taper_points must be at least 2.")

    # Check that the two output taper entrances fit inside the MMI.
    if port_sep / 2 + taper_width / 2 > mmi_width / 2:
        raise ValueError(
            "Output tapers do not fit inside the MMI width. "
            "Increase mmi_width or reduce port_sep or taper_width."
        )

    input_center = np.asarray(input_center, dtype=float)

    if input_center.shape != (2,):
        raise ValueError("input_center must be an (x, y) coordinate.")

    theta = np.deg2rad(orientation_deg)

    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ]
    )

    # ── Local-to-global coordinate conversion ────────────────────────────────
    def local_to_global(points):
        points = np.asarray(points, dtype=float)

        if points.ndim == 1:
            return input_center + rotation_matrix @ points

        return input_center + points @ rotation_matrix.T

    # ── Geometry helpers ─────────────────────────────────────────────────────
    def rectangle_points(x0, x1, y_center, width):
        return np.array(
            [
                [x0, y_center - width / 2],
                [x1, y_center - width / 2],
                [x1, y_center + width / 2],
                [x0, y_center + width / 2],
            ],
            dtype=float,
        )

    def taper_points_array(
        x0,
        x1,
        width_start,
        width_end,
        y_center,
    ):
        x = np.linspace(x0, x1, taper_points)
        s = np.linspace(0.0, 1.0, taper_points)

        width = (
            width_start
            + (width_end - width_start) * s**taper_power
        )

        upper = np.column_stack(
            (
                x,
                y_center + width / 2,
            )
        )

        lower = np.column_stack(
            (
                x[::-1],
                y_center - width[::-1] / 2,
            )
        )

        return np.vstack((upper, lower))

    def add_local_polygon(points):
        polygon = gdstk.Polygon(
            local_to_global(points),
            layer=layer,
            datatype=datatype,
        )

        cell.add(polygon)
        return polygon

    # ── Longitudinal positions in local coordinates ─────────────────────────
    #
    # External input port center:
    #     (0, 0)
    #
    # Device propagation:
    #     local +x
    #
    x_input_start = 0.0
    x_input_taper_start = input_length

    x_mmi_start = (
        input_length
        + input_taper_length
    )

    x_mmi_end = (
        x_mmi_start
        + mmi_length
    )

    x_output_taper_end = (
        x_mmi_end
        + output_taper_length
    )

    x_output_end = (
        x_output_taper_end
        + output_length
    )

    y_upper = port_sep / 2
    y_lower = -port_sep / 2

    # ── Create GDS library and cell ──────────────────────────────────────────
    if lib is None:
        lib = gdstk.Library(
            unit=1e-6,
            precision=1e-9,
        )

    cell = lib.new_cell(cell_name)

    # ── Input straight waveguide ─────────────────────────────────────────────
    add_local_polygon(
        rectangle_points(
            x0=x_input_start,
            x1=x_input_taper_start,
            y_center=0.0,
            width=wg_width,
        )
    )

    # ── Input taper ──────────────────────────────────────────────────────────
    add_local_polygon(
        taper_points_array(
            x0=x_input_taper_start,
            x1=x_mmi_start,
            width_start=wg_width,
            width_end=taper_width,
            y_center=0.0,
        )
    )

    # ── MMI multimode section ────────────────────────────────────────────────
    add_local_polygon(
        rectangle_points(
            x0=x_mmi_start,
            x1=x_mmi_end,
            y_center=0.0,
            width=mmi_width,
        )
    )

    # ── Upper output taper ───────────────────────────────────────────────────
    add_local_polygon(
        taper_points_array(
            x0=x_mmi_end,
            x1=x_output_taper_end,
            width_start=taper_width,
            width_end=wg_width,
            y_center=y_upper,
        )
    )

    # ── Lower output taper ───────────────────────────────────────────────────
    add_local_polygon(
        taper_points_array(
            x0=x_mmi_end,
            x1=x_output_taper_end,
            width_start=taper_width,
            width_end=wg_width,
            y_center=y_lower,
        )
    )

    # ── Upper output straight ────────────────────────────────────────────────
    add_local_polygon(
        rectangle_points(
            x0=x_output_taper_end,
            x1=x_output_end,
            y_center=y_upper,
            width=wg_width,
        )
    )

    # ── Lower output straight ────────────────────────────────────────────────
    add_local_polygon(
        rectangle_points(
            x0=x_output_taper_end,
            x1=x_output_end,
            y_center=y_lower,
            width=wg_width,
        )
    )

    # ── Important global coordinates ─────────────────────────────────────────
    input_port_center = local_to_global(
        [0.0, 0.0]
    )

    input_taper_start_center = local_to_global(
        [x_input_taper_start, 0.0]
    )

    mmi_input_center = local_to_global(
        [x_mmi_start, 0.0]
    )

    mmi_output_center = local_to_global(
        [x_mmi_end, 0.0]
    )

    upper_output_center = local_to_global(
        [x_output_end, y_upper]
    )

    lower_output_center = local_to_global(
        [x_output_end, y_lower]
    )

    # Explicit endpoint values for direct chaining to another component.
    upper_end_center = tuple(
        float(value) for value in upper_output_center
    )

    lower_end_center = tuple(
        float(value) for value in lower_output_center
    )

    end_orientation_deg = float(
        orientation_deg % 360
    )

    # Unit propagation and transverse vectors.
    propagation_vector = np.array(
        [
            np.cos(theta),
            np.sin(theta),
        ]
    )

    transverse_vector = np.array(
        [
            -np.sin(theta),
            np.cos(theta),
        ]
    )

    ports = {
        "input": {
            "center": tuple(input_port_center),
            "width": float(wg_width),
            "orientation_deg": float(
                (orientation_deg + 180) % 360
            ),
        },
        "output_upper": {
            "center": upper_end_center,
            "width": float(wg_width),
            "orientation_deg": end_orientation_deg,
        },
        "output_lower": {
            "center": lower_end_center,
            "width": float(wg_width),
            "orientation_deg": end_orientation_deg,
        },
    }

    meta = {
        "input_center": tuple(input_port_center),
        "input_taper_start_center": tuple(
            input_taper_start_center
        ),
        "mmi_input_center": tuple(mmi_input_center),
        "mmi_output_center": tuple(mmi_output_center),
        "upper_output_center": upper_end_center,
        "lower_output_center": lower_end_center,
        "upper_end_center": upper_end_center,
        "lower_end_center": lower_end_center,
        "end_orientation_deg": end_orientation_deg,
        "orientation_deg": float(orientation_deg),
        "propagation_vector": tuple(propagation_vector),
        "transverse_vector": tuple(transverse_vector),
        "total_length": float(x_output_end),
        "mmi_width": float(mmi_width),
        "mmi_length": float(mmi_length),
        "wg_width": float(wg_width),
        "taper_width": float(taper_width),
        "port_sep": float(port_sep),
    }

    if gds_file is not None:
        lib.write_gds(gds_file)

    return (
        lib,
        cell,
        upper_end_center,
        lower_end_center,
        end_orientation_deg,
        ports,
        meta,
    )


def make_s_bend_gdstk(
    length,
    offset,
    width,
    start_center=(0.0, 0.0),
    orientation_deg=0.0,
    layer=1,
    datatype=0,
    tolerance=0.001,
):
    """
    Create a smooth S-bend using gdstk.

    Parameters
    ----------
    length : float
        Longitudinal length of the S-bend in µm.

    offset : float
        Signed transverse displacement in µm.

        offset > 0 : bends toward local +y
        offset < 0 : bends toward local -y

    width : float
        Waveguide width in µm.

    start_center : tuple
        Global (x, y) coordinate of the input-port center.

    orientation_deg : float
        Input propagation direction:

          0°   -> +x
          90°  -> +y
          180° -> -x
          270° -> -y

    Returns
    -------
    path : gdstk.RobustPath
        S-bend geometry.

    end_center : tuple
        Global coordinate of the output-port center.

    end_orientation_deg : float
        Propagation orientation at the output port.

    ports : dict
        Input and output-port coordinates and orientations.

    meta : dict
        Additional geometry information.
    """

    if length <= 0:
        raise ValueError("length must be positive.")

    if width <= 0:
        raise ValueError("width must be positive.")

    start_center = np.asarray(start_center, dtype=float)

    if start_center.shape != (2,):
        raise ValueError("start_center must be an (x, y) coordinate.")

    orientation_rad = np.deg2rad(orientation_deg)

    rotation_matrix = np.array(
        [
            [np.cos(orientation_rad), -np.sin(orientation_rad)],
            [np.sin(orientation_rad),  np.cos(orientation_rad)],
        ],
        dtype=float,
    )

    # Fifth-order smoothstep:
    # zero slope and zero curvature at both ends.
    def centerline(u):
        smooth = (
            10.0 * u**3
            - 15.0 * u**4
            + 6.0 * u**5
        )

        return (
            length * u,
            offset * smooth,
        )

    def centerline_gradient(u):
        ds_du = (
            30.0 * u**2
            - 60.0 * u**3
            + 30.0 * u**4
        )

        return (
            length,
            offset * ds_du,
        )

    # Construct locally with the input centered at (0, 0).
    path = gdstk.RobustPath(
        initial_point=(0.0, 0.0),
        width=width,
        ends="flush",
        tolerance=tolerance,
        layer=layer,
        datatype=datatype,
    )

    path.parametric(
        centerline,
        path_gradient=centerline_gradient,
        relative=True,
    )

    # Rotate around the local input port.
    path.rotate(
        orientation_rad,
        center=(0.0, 0.0),
    )

    # Translate the local input port to start_center.
    path.translate(
        float(start_center[0]),
        float(start_center[1]),
    )

    # Calculate the global output coordinate.
    output_local = np.array(
        [length, offset],
        dtype=float,
    )

    output_center = (
        start_center
        + rotation_matrix @ output_local
    )

    # Explicit endpoint values for direct chaining.
    end_center = tuple(
        float(value) for value in output_center
    )

    end_orientation_deg = float(
        orientation_deg % 360
    )

    input_orientation_deg = float(
        orientation_deg % 360
    )

    ports = {
        "input": {
            "center": tuple(
                float(value) for value in start_center
            ),
            "width": float(width),
            "propagation_orientation_deg": input_orientation_deg,
            "outward_orientation_deg": float(
                (input_orientation_deg + 180) % 360
            ),
        },
        "output": {
            "center": end_center,
            "width": float(width),
            "propagation_orientation_deg": end_orientation_deg,
            "outward_orientation_deg": end_orientation_deg,
        },
    }

    meta = {
        "start_center": tuple(
            float(value) for value in start_center
        ),
        "output_center": end_center,
        "end_center": end_center,
        "orientation_deg": float(orientation_deg),
        "end_orientation_deg": end_orientation_deg,
        "length": float(length),
        "offset": float(offset),
        "width": float(width),
    }

    return (
        path,
        end_center,
        end_orientation_deg,
        ports,
        meta,
    )


def make_euler_bend_gdstk(
    radius,
    bend_angle_deg,
    width,
    start_center=(0.0, 0.0),
    orientation_deg=0.0,
    euler_fraction=1.0,
    layer=1,
    datatype=0,
    tolerance=0.001,
    max_evals=2000,
    integration_order=24,
):
    """
    Create a circular, partial-Euler, or full-Euler bend using gdstk.

    Parameters
    ----------
    radius : float
        Minimum centerline bend radius in µm.

        For euler_fraction < 1, this is also the radius of the
        constant-curvature circular section.

    bend_angle_deg : float
        Signed total bend angle in degrees.

        Positive -> counterclockwise / left turn
        Negative -> clockwise / right turn

    width : float
        Waveguide width in µm.

    start_center : tuple
        Global (x, y) coordinate of the input-port center.

    orientation_deg : float
        Input propagation direction.

          0°   -> +x
          90°  -> +y
          180° -> -x
          270° -> -y

    euler_fraction : float
        Fraction of the total bend angle occupied by the two
        Euler transition sections.

        0.0 -> circular bend
        0.5 -> partial Euler bend
        1.0 -> full Euler bend

    tolerance : float
        Geometric approximation tolerance used by gdstk.

    max_evals : int
        Maximum number of path evaluations used by gdstk.

    integration_order : int
        Gauss-Legendre quadrature order used to calculate the
        Euler centerline coordinates.

    Returns
    -------
    path : gdstk.RobustPath
        Euler-bend geometry.

    end_center : tuple
        Global coordinate of the output-port center.

    end_orientation_deg : float
        Propagation orientation at the output port.

    ports : dict
        Input and output port coordinates and orientations.

    meta : dict
        Bend geometry information.
    """

    # ── Validate inputs ──────────────────────────────────────────────────────
    if radius <= 0:
        raise ValueError("radius must be positive.")

    if width <= 0:
        raise ValueError("width must be positive.")

    if bend_angle_deg == 0:
        raise ValueError("bend_angle_deg cannot be zero.")

    if abs(bend_angle_deg) > 180:
        raise ValueError(
            "bend_angle_deg must be between -180 and +180 degrees."
        )

    if not 0.0 <= euler_fraction <= 1.0:
        raise ValueError(
            "euler_fraction must be between 0 and 1."
        )

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_evals < 100:
        raise ValueError("max_evals should be at least 100.")

    integration_order = int(integration_order)

    if integration_order < 8:
        raise ValueError(
            "integration_order should be at least 8."
        )

    start_center = np.asarray(
        start_center,
        dtype=float,
    )

    if start_center.shape != (2,):
        raise ValueError(
            "start_center must be an (x, y) coordinate."
        )

    # ── Bend parameters ──────────────────────────────────────────────────────
    input_angle_rad = np.deg2rad(
        orientation_deg
    )

    bend_angle_rad = np.deg2rad(
        bend_angle_deg
    )

    bend_sign = np.sign(
        bend_angle_rad
    )

    theta = abs(
        bend_angle_rad
    )

    p = float(
        euler_fraction
    )

    # Length of each Euler transition section.
    euler_length = radius * p * theta

    # Length of the constant-radius circular section.
    circular_length = radius * (1.0 - p) * theta

    # Total centerline length.
    total_length = (
        2.0 * euler_length
        + circular_length
    )

    s_euler_1_end = euler_length

    s_circular_end = (
        euler_length
        + circular_length
    )

    # ── Tangent angle along the bend ─────────────────────────────────────────
    def tangent_angle_from_s(s):
        """
        Tangent angle relative to the input direction as a function
        of centerline arc length s.
        """

        s = np.asarray(
            s,
            dtype=float,
        )

        # Pure circular bend.
        if p == 0.0:
            return bend_sign * s / radius

        # First Euler section:
        # curvature increases linearly from 0 to 1 / radius.
        phi_first = (
            s**2
            / (2.0 * radius * euler_length)
        )

        # Circular section.
        phi_circular = (
            p * theta / 2.0
            + (s - s_euler_1_end) / radius
        )

        # Second Euler section:
        # curvature decreases linearly from 1 / radius to 0.
        q = s - s_circular_end

        phi_second = (
            theta * (1.0 - p / 2.0)
            + q / radius
            - q**2
            / (2.0 * radius * euler_length)
        )

        phi = np.where(
            s <= s_euler_1_end,
            phi_first,
            np.where(
                s <= s_circular_end,
                phi_circular,
                phi_second,
            ),
        )

        return bend_sign * phi

    # ── Numerical integration setup ──────────────────────────────────────────
    nodes, weights = np.polynomial.legendre.leggauss(
        integration_order
    )

    def integrate_interval(s_start, s_end):
        """
        Integrate dx/ds and dy/ds over one smooth arc-length interval.
        """

        if s_end <= s_start:
            return np.array(
                [0.0, 0.0],
                dtype=float,
            )

        s_values = (
            0.5 * (s_end - s_start) * nodes
            + 0.5 * (s_start + s_end)
        )

        scaled_weights = (
            0.5
            * (s_end - s_start)
            * weights
        )

        phi = tangent_angle_from_s(
            s_values
        )

        dx = np.sum(
            scaled_weights * np.cos(phi)
        )

        dy = np.sum(
            scaled_weights * np.sin(phi)
        )

        return np.array(
            [dx, dy],
            dtype=float,
        )

    def integrate_centerline(s_end):
        """
        Integrate from s = 0 to s = s_end while splitting at the
        Euler/circular boundaries.
        """

        s_end = float(
            np.clip(
                s_end,
                0.0,
                total_length,
            )
        )

        result = np.array(
            [0.0, 0.0],
            dtype=float,
        )

        section_boundaries = [
            0.0,
            s_euler_1_end,
            s_circular_end,
            total_length,
        ]

        for section_start, section_end in zip(
            section_boundaries[:-1],
            section_boundaries[1:],
        ):
            if s_end <= section_start:
                break

            active_end = min(
                s_end,
                section_end,
            )

            if active_end > section_start:
                result += integrate_interval(
                    section_start,
                    active_end,
                )

        return result

    # ── Parametric path functions ────────────────────────────────────────────
    def centerline(u):
        """
        Centerline coordinate for normalized parameter u from 0 to 1.
        """

        u = float(
            np.clip(
                u,
                0.0,
                1.0,
            )
        )

        s = u * total_length

        return tuple(
            integrate_centerline(s)
        )

    def centerline_gradient(u):
        """
        Derivative of centerline position with respect to u.
        """

        u = float(
            np.clip(
                u,
                0.0,
                1.0,
            )
        )

        s = u * total_length

        phi = float(
            tangent_angle_from_s(s)
        )

        return (
            total_length * np.cos(phi),
            total_length * np.sin(phi),
        )

    # ── Build path in local coordinates ──────────────────────────────────────
    path = gdstk.RobustPath(
        initial_point=(0.0, 0.0),
        width=width,
        ends="flush",
        tolerance=tolerance,
        max_evals=max_evals,
        layer=layer,
        datatype=datatype,
    )

    path.parametric(
        path_function=centerline,
        path_gradient=centerline_gradient,
        relative=True,
    )

    # Output position before global rotation and translation.
    output_local = np.asarray(
        centerline(1.0),
        dtype=float,
    )

    # ── Rotate to requested input orientation ────────────────────────────────
    path.rotate(
        input_angle_rad,
        center=(0.0, 0.0),
    )

    # ── Translate input port to start_center ─────────────────────────────────
    path.translate(
        float(start_center[0]),
        float(start_center[1]),
    )

    rotation_matrix = np.array(
        [
            [
                np.cos(input_angle_rad),
                -np.sin(input_angle_rad),
            ],
            [
                np.sin(input_angle_rad),
                np.cos(input_angle_rad),
            ],
        ],
        dtype=float,
    )

    output_center = (
        start_center
        + rotation_matrix @ output_local
    )

    input_orientation_deg = (
        orientation_deg % 360
    )

    output_orientation_deg = (
        orientation_deg
        + bend_angle_deg
    ) % 360

    # Explicit endpoint values for direct component chaining.
    end_center = tuple(
        float(value) for value in output_center
    )

    end_orientation_deg = float(
        output_orientation_deg
    )

    # ── Port information ────────────────────────────────────────────────────
    ports = {
        "input": {
            "center": tuple(start_center),
            "width": float(width),

            "propagation_orientation_deg": float(
                input_orientation_deg
            ),

            "outward_orientation_deg": float(
                (input_orientation_deg + 180) % 360
            ),
        },

        "output": {
            "center": end_center,
            "width": float(width),

            "propagation_orientation_deg": end_orientation_deg,

            "outward_orientation_deg": end_orientation_deg,
        },
    }

    # ── Metadata ─────────────────────────────────────────────────────────────
    if p == 0.0:
        bend_type = "circular"
    elif p == 1.0:
        bend_type = "full_euler"
    else:
        bend_type = "partial_euler"

    meta = {
        "bend_type": bend_type,

        "start_center": tuple(
            start_center
        ),

        "output_center": end_center,

        "end_center": end_center,

        "output_local": tuple(
            output_local
        ),

        "input_orientation_deg": float(
            input_orientation_deg
        ),

        "output_orientation_deg": end_orientation_deg,

        "end_orientation_deg": end_orientation_deg,

        "bend_angle_deg": float(
            bend_angle_deg
        ),

        "radius": float(
            radius
        ),

        "euler_fraction": float(
            p
        ),

        "euler_section_length": float(
            euler_length
        ),

        "circular_section_length": float(
            circular_length
        ),

        "centerline_length": float(
            total_length
        ),

        "width": float(
            width
        ),
    }

    return (
        path,
        end_center,
        end_orientation_deg,
        ports,
        meta,
    )
