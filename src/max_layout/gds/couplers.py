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
