"""Grating-coupler attachment builders."""

from __future__ import annotations

from typing import Any

import gdstk
import numpy as np

from ..backend import backend
from ..constants import DEFAULT_DATATYPE, GC_LAYER, PHOTONIC_LAYER
from ..gds.primitives import add_rect
from ..geometry.euler import gc_euler_output_local
from ..geometry.shapes import mmi_total_length
from ..geometry.transforms import rot, transform_points


def add_parent_focusing_gc(
    top: gdstk.Cell,
    p: dict[str, Any],
    waveguide_center: tuple[float, float],
    orientation_deg: float,
    width: float,
    layer: int,
    datatype: int,
    cell_name: str,
) -> None:
    """Add a GC with a layer-1 connector and layer-2 grating geometry.

    The immutable GC block is generated exactly as supplied.  Assembly then
    removes its initial rectangular waveguide from the GC layer and recreates
    that connector once on the photonic layer.  The connector and taper meet
    at one boundary, so no rectangular overlap remains.
    """
    photonic_layer = PHOTONIC_LAYER
    photonic_datatype = DEFAULT_DATATYPE
    gc_layer = int(p.get("gc_layer", GC_LAYER))
    gc_datatype = int(p.get("gc_datatype", DEFAULT_DATATYPE))
    wg_length = float(p.get("gc_wg_length", p.get("wg_length", 20.0)))

    _, gc_cell, *_ = backend.make_focusing_gc_gds(
        pitch=float(p.get("gc_pitch", p.get("pitch", 0.75))),
        fill_factor=float(p.get("gc_fill_factor", p.get("fill_factor", 0.57))),
        N=int(p.get("gc_N", p.get("N", 30))),
        alpha_t=float(p.get("gc_alpha_t", p.get("alpha_t", 25.0))),
        taper_L=float(p.get("gc_taper_L", p.get("taper_L", 22.0))),
        wg_width=float(width),
        wg_length=wg_length,
        wg_end_center=waveguide_center,
        orientation_deg=float(orientation_deg) % 360.0,
        layer=gc_layer,
        datatype=gc_datatype,
        cell_name=cell_name,
        gds_file=None,
        tolerance=float(p.get("gc_tolerance", p.get("tolerance", 0.0005))),
        L_extra=float(p.get("gc_L_extra", p.get("L_extra", 0.0))),
    )

    connector_local = np.array(
        [
            [0.0, -float(width) / 2.0],
            [wg_length, -float(width) / 2.0],
            [wg_length, float(width) / 2.0],
            [0.0, float(width) / 2.0],
        ],
        dtype=float,
    )
    connector_points = transform_points(
        connector_local,
        waveguide_center,
        float(orientation_deg) % 360.0,
    )

    # One non-overlapping photonic connector on layer 1.
    top.add(
        gdstk.Polygon(
            connector_points,
            layer=photonic_layer,
            datatype=photonic_datatype,
        )
    )

    # Remove the same rectangle from the GC layer.  The remaining taper and
    # teeth are all on layer 2 and abut the connector at x = wg_length.
    gc_polygons = [polygon.copy() for polygon in gc_cell.polygons]
    cutter = gdstk.Polygon(
        connector_points,
        layer=gc_layer,
        datatype=gc_datatype,
    )
    remaining = gdstk.boolean(
        gc_polygons,
        [cutter],
        "not",
        layer=gc_layer,
        datatype=gc_datatype,
    )
    if remaining:
        top.add(*remaining)


def add_soi_grating_coupler(
    top: gdstk.Cell,
    p: dict[str, Any],
    waveguide_center: tuple[float, float],
    orientation_deg: float,
) -> None:
    """Draw the official Ansys 3D SOI focusing-grating geometry.

    Layer 1 is the 120 nm residual silicon footprint and layer 2 is the
    additional 100 nm silicon.  With the matching stack preset this reproduces
    the example's 220 nm device layer and 100 nm partial etch.
    """
    pitch = float(p.get("pitch", 0.6713))
    target_length = float(p.get("target_length", 25.0))
    duty_cycle = float(p.get("duty_cycle", 0.3992))
    radius = float(p.get("radius", 25.0))
    y_span = float(p.get("y_span", 15.0))
    extra_length = float(p.get("L_extra", 10.0))
    wg_width = float(p.get("wg_width", 0.5))
    wg_length = float(p.get("wg_length", 10.0))
    taper_exponent = float(p.get("taper_exponent", 1.15))
    slab_layer = int(p.get("slab_layer", PHOTONIC_LAYER))
    slab_datatype = int(p.get("slab_datatype", DEFAULT_DATATYPE))
    etched_layer = int(p.get("etched_layer", GC_LAYER))
    etched_datatype = int(p.get("etched_datatype", DEFAULT_DATATYPE))
    tolerance = max(1e-5, float(p.get("tolerance", 0.0005)))
    if min(pitch, target_length, radius, y_span, wg_width, wg_length) <= 0:
        raise ValueError("GC-SOI dimensions and pitch must be positive.")
    if not 0.0 < duty_cycle < 1.0:
        raise ValueError("GC-SOI duty cycle must be between 0 and 1.")
    if y_span >= 2.0 * radius:
        raise ValueError("GC-SOI y span must be smaller than twice its radius.")

    period_count = int(np.ceil(target_length / pitch))
    tooth_width = duty_cycle * pitch
    gap_width = pitch - tooth_width
    grating_length = period_count * pitch + gap_width
    half_angle = float(np.arcsin(0.5 * y_span / radius))
    taper_length = radius * float(np.cos(half_angle))
    focus = np.array([wg_length, 0.0], dtype=float)

    def transformed(points: np.ndarray, layer: int, datatype: int) -> gdstk.Polygon:
        return gdstk.Polygon(
            transform_points(points, waveguide_center, orientation_deg),
            layer=layer,
            datatype=datatype,
        )

    def annular(inner: float, outer: float, layer: int, datatype: int) -> gdstk.Polygon:
        arc_length = max(outer, 1.0) * 2.0 * half_angle
        samples = max(32, min(2048, int(np.ceil(arc_length / max(0.02, np.sqrt(tolerance))))))
        angles = np.linspace(-half_angle, half_angle, samples)
        outer_points = focus + np.column_stack((outer * np.cos(angles), outer * np.sin(angles)))
        inner_points = focus + np.column_stack((inner * np.cos(angles[::-1]), inner * np.sin(angles[::-1])))
        return transformed(np.vstack((outer_points, inner_points)), layer, datatype)

    waveguide = np.array(
        [[0.0, -wg_width / 2.0], [wg_length, -wg_width / 2.0],
         [wg_length, wg_width / 2.0], [0.0, wg_width / 2.0]],
        dtype=float,
    )
    taper_samples = max(81, int(np.ceil(taper_length / 0.1)) + 1)
    taper_x = np.linspace(0.0, taper_length, taper_samples)
    half_width = y_span / 2.0 - (y_span / 2.0 - wg_width / 2.0) * (
        (taper_length - taper_x) / taper_length
    ) ** taper_exponent
    taper = np.vstack(
        (
            np.column_stack((wg_length + taper_x, -half_width)),
            np.column_stack((wg_length + taper_x[::-1], half_width[::-1])),
        )
    )

    # Both vertical silicon portions exist under the unetched waveguide,
    # nonlinear taper, input sector, and final output sector.
    for layer, datatype in ((slab_layer, slab_datatype), (etched_layer, etched_datatype)):
        top.add(transformed(waveguide, layer, datatype))
        top.add(transformed(taper, layer, datatype))
        top.add(annular(taper_length, radius, layer, datatype))
        top.add(annular(radius + grating_length, radius + grating_length + extra_length, layer, datatype))

    # The residual slab is continuous beneath every etched grating period.
    top.add(annular(radius, radius + grating_length, slab_layer, slab_datatype))
    for index in range(period_count):
        inner = radius + index * pitch + gap_width
        outer = radius + (index + 1) * pitch
        top.add(annular(inner, outer, etched_layer, etched_datatype))


def add_routed_parent_gc(
    top: gdstk.Cell,
    p: dict[str, Any],
    start_center: tuple[float, float],
    start_orientation_deg: float,
    bend_angle_deg: float,
    width: float,
    layer: int,
    datatype: int,
    cell_name: str,
) -> tuple[tuple[float, float], float]:
    endpoint = (float(start_center[0]), float(start_center[1]))
    endpoint_orientation = float(start_orientation_deg) % 360.0
    if abs(float(bend_angle_deg)) >= 1e-12:
        path, endpoint, endpoint_orientation, *_ = backend.make_euler_bend_gdstk(
            radius=float(p.get("gc_euler_radius", 100.0)),
            bend_angle_deg=float(bend_angle_deg),
            width=float(width),
            start_center=endpoint,
            orientation_deg=endpoint_orientation,
            euler_fraction=float(p.get("gc_euler_fraction", 1.0)),
            layer=layer,
            datatype=datatype,
            tolerance=float(p.get("gc_euler_tolerance", 0.001)),
        )
        top.add(path)
    add_parent_focusing_gc(
        top, p, endpoint, endpoint_orientation, width, layer, datatype, cell_name
    )
    return endpoint, endpoint_orientation


def add_three_euler_inward_gc(top: gdstk.Cell, p: dict[str,Any], start_center: tuple[float,float], start_orientation_deg: float, left_end: bool, width: float, layer: int, datatype: int, cell_name: str) -> tuple[tuple[float,float],float]:
    """Draw three Euler bends with the last mirrored so the two GCs point outward."""
    side=1.0 if str(p.get("gc_vertical_side","up")).lower()=="up" else -1.0
    bend_angles=(-side*90.0,-side*90.0,side*90.0);point=(float(start_center[0]),float(start_center[1]));heading=float(start_orientation_deg)%360;prebend=float(p.get("gc_prebend_straight",10.0))
    if prebend<0:raise ValueError("The straight section before the Euler bends cannot be negative.")
    if prebend>0:add_rect(top,np.array([[0,-width/2],[prebend,-width/2],[prebend,width/2],[0,width/2]],float),point,heading,layer,datatype);point=tuple(np.asarray(point)+rot((prebend,0),heading))
    for bend_index,bend_angle in enumerate(bend_angles):
        path,point,heading,*_=backend.make_euler_bend_gdstk(radius=float(p.get("gc_euler_radius",100.0)),bend_angle_deg=bend_angle,width=float(width),start_center=point,orientation_deg=heading,euler_fraction=float(p.get("gc_euler_fraction",1.0)),layer=layer,datatype=datatype,tolerance=float(p.get("gc_euler_tolerance",.001)));top.add(path)
        if bend_index<2:
            fraction=float(p.get("gc_inward_run_fraction",.45));total=2*mmi_total_length(p)+2*float(p.get("s_bend_length",80))+float(p.get("arm_length",9718))
            if fraction<0 or fraction>=.5:raise ValueError("GC inward-run fraction must be at least 0 and below 0.5.")
            run=float(p.get("gc_vertical_run",100.0) if bend_index==0 else (fraction*total if fraction>0 else p.get("gc_inward_run",300.0)))
            if bend_index==1 and bool(p.get("gc_align_gc_to_mzi_center",True)):
                target=np.asarray(start_center,float)+rot((total/2,0),(float(start_orientation_deg)+180)%360);last_delta=rot(gc_euler_output_local(p,bend_angles[2]),heading);direction=rot((1,0),heading);run=float(np.dot(target-np.asarray(point)-last_delta,direction))
            if run<0:raise ValueError("Vertical-GC route straight lengths cannot be negative.")
            add_rect(top,np.array([[0,-width/2],[run,-width/2],[run,width/2],[0,width/2]],float),point,heading,layer,datatype);point=tuple(np.asarray(point)+rot((run,0),heading))
    add_parent_focusing_gc(top,p,point,heading,width,layer,datatype,cell_name)
    return point,heading
