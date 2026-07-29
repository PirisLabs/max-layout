"""Component geometry construction and layout assembly."""

from __future__ import annotations

from typing import Any
import json
import math

import gdstk
import numpy as np

from ..backend import backend
from ..constants import DEFAULT_COMPONENT_VALUES, DEFAULT_DATATYPE, EBEAM_LAYER, GC_COMPOSITE_KINDS, GC_LAYER, MARKER_COMPONENT_KINDS, MARKER_LAYER, PHOTONIC_LAYER, RF_COMPONENT_KINDS, RF_LAYER
from ..gds.couplers import add_parent_focusing_gc, add_routed_parent_gc, add_three_euler_inward_gc
from ..gds.ebeam import add_ebeam_field_outline, add_ebeam_parameter_text, add_write_field_number, multipass_field_layout
from ..gds.primitives import add_rect, copy_cell_polygons_to_top
from ..geometry.euler import grating_route_bend_angle, mmi_gc_fanout_local_points
from ..geometry.landmarks import cpw_bend_landmarks, feedline_landmarks, loopback_landmarks, resonator_x_positions, ring_two_feedline_landmarks, segmented_electrode_landmarks
from ..geometry.rf_taper import _gap_transition_values, gap_profile, symmetric_cpw_taper_profile
from ..geometry.shapes import _capsule_polygons, cpw_annular_sector_points, mmi_total_length
from ..geometry.transforms import _transform_polygon, add_local_polygon, rot, transform_points
from ..ports import component_global_ports, component_local_ports, normalize_port_name, solve_attachment
from ..utils import compact_parameter_range, inclusive_sweep, numeric_list, safe_json_copy


def add_component_centered_in_field(top: gdstk.Cell, kind: str, params: dict[str,Any], center: tuple[float,float], orientation: float, mirrored: bool, uid: int, field_size: float, clearance: float, draw_field: bool = True) -> None:
    """Flatten a subdevice, center its true geometry in a write field, and verify clearance."""
    lib=gdstk.Library(unit=1e-6,precision=1e-9);cell=lib.new_cell(f"FIELD_TMP_{uid}");_add_component_geometry_to_cell({"uid":uid,"kind":kind,"x":0.0,"y":0.0,"orientation_deg":0.0,"mirrored":mirrored,"params":safe_json_copy(params)},cell);bbox=cell.bounding_box()
    if bbox is None:return
    minimum=np.asarray(bbox[0],float);maximum=np.asarray(bbox[1],float);size=maximum-minimum;usable=float(field_size)-2*float(clearance)
    if size[0]>usable+1e-6 or size[1]>usable+1e-6:raise ValueError(f"{kind} geometry {size[0]:.3f} × {size[1]:.3f} µm does not fit the {field_size:g} µm field with {clearance:g} µm edge clearance.")
    geometry_center=(minimum+maximum)/2
    for polygon in cell.get_polygons(apply_repetitions=True,include_paths=True):
        points=np.asarray(polygon.points,float)-geometry_center;top.add(gdstk.Polygon(transform_points(points,center,orientation),layer=int(polygon.layer),datatype=int(polygon.datatype)))
    for label in cell.labels:
        point=tuple(np.asarray(center)+rot(np.asarray(label.origin,float)-geometry_center,orientation));top.add(gdstk.Label(label.text,point,rotation=float(label.rotation or 0)+math.radians(orientation),layer=int(label.layer),texttype=int(label.texttype)))
    if draw_field:add_ebeam_field_outline(top,center,orientation,field_size)


def add_ring_to_gds(top: gdstk.Cell, ring_center_local: tuple[float, float], radius: float, width: float, points: int, component_center: tuple[float, float], orientation: float, layer: int, datatype: int) -> None:
    """Add a closed annular ring with no physical coupling seam or air gap."""
    if radius <= 0 or width <= 0 or radius <= width / 2.0:
        raise ValueError("Ring radius must be larger than half the waveguide width.")
    tolerance = max(1e-5, 2.0 * np.pi * (radius + width / 2.0) / max(64, int(points)) ** 2)
    ring = gdstk.ellipse(
        center=tuple(float(v) for v in ring_center_local),
        radius=radius + width / 2.0,
        inner_radius=radius - width / 2.0,
        tolerance=tolerance,
        layer=layer,
        datatype=datatype,
    )
    top.add(_transform_polygon(ring, component_center, orientation))


def add_racetrack_to_gds(top: gdstk.Cell, race_center_local: tuple[float, float], radius: float, coupling_length: float, width: float, points: int, component_center: tuple[float, float], orientation: float, layer: int, datatype: int) -> None:
    """Add a continuous closed racetrack using outer-minus-inner capsules."""
    if radius <= 0 or coupling_length < 0 or width <= 0 or radius <= width / 2.0:
        raise ValueError("Racetrack radius must exceed half the width; coupling_length cannot be negative.")
    cx, cy = (float(race_center_local[0]), float(race_center_local[1]))
    tolerance = max(1e-5, 2.0 * np.pi * (radius + width / 2.0) / max(64, int(points)) ** 2)
    outer = _capsule_polygons(cx, cy, radius + width / 2.0, coupling_length, layer, datatype, tolerance)
    inner = _capsule_polygons(cx, cy, radius - width / 2.0, coupling_length, layer, datatype, tolerance)
    track = gdstk.boolean(outer, inner, "not", layer=layer, datatype=datatype)
    if not track:
        raise ValueError("Could not generate a closed racetrack polygon.")
    for polygon in track:
        top.add(_transform_polygon(polygon, component_center, orientation))


def add_feedline_to_gds(component: dict[str, Any], top: gdstk.Cell) -> dict[str, tuple[float, float]]:
    p = dict(component.get("params", {}))
    start = (float(component["x"]), float(component["y"]))
    orientation = float(component.get("orientation_deg", 0.0)) % 360.0
    layer = int(p.get("layer", 1))
    datatype = int(p.get("datatype", 0))
    uid = int(component.get("uid", 0))
    points = feedline_landmarks(p, bool(component.get("mirrored", False)))
    width = float(p["wg_width"])

    if bool(p.get("add_grating_couplers", True)):
        add_parent_focusing_gc(
            top, p, start, (orientation + 180.0) % 360.0, width,
            PHOTONIC_LAYER, DEFAULT_DATATYPE, f"FEEDLINE_INPUT_GC_{uid}",
        )

    add_rect(top, np.array([[0.0, -width/2.0], [float(p["input_straight_length"]), -width/2.0], [float(p["input_straight_length"]), width/2.0], [0.0, width/2.0]], dtype=float), start, orientation, layer, datatype)

    first_start = tuple(np.asarray(start) + rot(points["input_straight_end"], orientation))
    first_offset = points["first_s_bend_end"][1]
    first_s, first_end, first_orientation, *_ = backend.make_s_bend_gdstk(
        length=float(p["s_bend_length"]), offset=first_offset, width=width,
        start_center=first_start, orientation_deg=orientation, layer=layer, datatype=datatype,
        tolerance=float(p.get("tolerance", 0.001)),
    )
    top.add(first_s)

    add_rect(top, np.array([[0.0, -width/2.0], [float(p["Lc"]), -width/2.0], [float(p["Lc"]), width/2.0], [0.0, width/2.0]], dtype=float), first_end, first_orientation, layer, datatype)

    second_start = tuple(np.asarray(first_end) + rot((float(p["Lc"]), 0.0), first_orientation))
    second_offset = points["second_s_bend_end"][1] - points["lc_end"][1]
    second_s, second_end, second_orientation, *_ = backend.make_s_bend_gdstk(
        length=float(p["s_bend_length"]), offset=second_offset, width=width,
        start_center=second_start, orientation_deg=first_orientation, layer=layer, datatype=datatype,
        tolerance=float(p.get("tolerance", 0.001)),
    )
    top.add(second_s)

    add_rect(top, np.array([[0.0, -width/2.0], [float(p["output_straight_length"]), -width/2.0], [float(p["output_straight_length"]), width/2.0], [0.0, width/2.0]], dtype=float), second_end, second_orientation, layer, datatype)

    output_global = tuple(np.asarray(second_end) + rot((float(p["output_straight_length"]), 0.0), second_orientation))
    if bool(p.get("add_grating_couplers", True)):
        add_parent_focusing_gc(
            top, p, output_global, orientation, width,
            PHOTONIC_LAYER, DEFAULT_DATATYPE, f"FEEDLINE_OUTPUT_GC_{uid}",
        )
    return points


def _add_component_geometry_to_cell(component: dict[str, Any], top: gdstk.Cell) -> None:
    kind = str(component["kind"]); p = dict(component.get("params", {}))
    start = (float(component["x"]), float(component["y"])); orientation = float(component.get("orientation_deg", 0.0)) % 360.0
    mirrored = bool(component.get("mirrored", False)); ms = -1.0 if mirrored else 1.0
    if kind == "E-beam multipass":
        # EPBG write fields always export on the dedicated Ebeam layer.
        layer = EBEAM_LAYER
        datatype = DEFAULT_DATATYPE
    elif kind in RF_COMPONENT_KINDS:
        layer = RF_LAYER
        datatype = DEFAULT_DATATYPE
    elif kind in MARKER_COMPONENT_KINDS:
        layer = MARKER_LAYER
        datatype = DEFAULT_DATATYPE
    elif kind == "Grating coupler":
        layer = GC_LAYER
        datatype = DEFAULT_DATATYPE
    elif kind == "Chip outline":
        layer = int(p.get("layer", 100))
        datatype = int(p.get("datatype", 0))
    else:
        layer = PHOTONIC_LAYER
        datatype = DEFAULT_DATATYPE
    uid = int(component.get("uid", 0))

    if kind == "Straight":
        length = float(p["length"]); width = float(p["width"])
        add_rect(top, np.array([[0,-width/2],[length,-width/2],[length,width/2],[0,width/2]], float), start, orientation, layer, datatype); return
    if kind == "Taper":
        length = float(p["length"]); w0 = float(p["width_start"]); w1 = float(p["width_end"])
        add_rect(top, np.array([[0,-w0/2],[length,-w1/2],[length,w1/2],[0,w0/2]], float), start, orientation, layer, datatype); return
    if kind == "S-bend":
        path, *_ = backend.make_s_bend_gdstk(length=float(p["length"]), offset=ms*float(p["offset"]), width=float(p["width"]), start_center=start, orientation_deg=orientation, layer=layer, datatype=datatype, tolerance=float(p["tolerance"])); top.add(path); return
    if kind == "Euler bend":
        path, *_ = backend.make_euler_bend_gdstk(radius=float(p["radius"]), bend_angle_deg=ms*float(p["bend_angle_deg"]), width=float(p["width"]), start_center=start, orientation_deg=orientation, euler_fraction=float(p["euler_fraction"]), layer=layer, datatype=datatype, tolerance=float(p["tolerance"])); top.add(path); return
    if kind == "Grating coupler":
        standalone_gc_params = dict(p)
        standalone_gc_params["gc_layer"] = GC_LAYER
        standalone_gc_params["gc_datatype"] = DEFAULT_DATATYPE
        standalone_gc_params["gc_wg_length"] = float(p["wg_length"])
        standalone_gc_params["gc_pitch"] = float(p["pitch"])
        standalone_gc_params["gc_fill_factor"] = float(p["fill_factor"])
        standalone_gc_params["gc_N"] = int(p["N"])
        standalone_gc_params["gc_alpha_t"] = float(p["alpha_t"])
        standalone_gc_params["gc_taper_L"] = float(p["taper_L"])
        standalone_gc_params["gc_tolerance"] = float(p["tolerance"])
        add_parent_focusing_gc(
            top, standalone_gc_params, start, orientation, float(p["wg_width"]),
            PHOTONIC_LAYER, DEFAULT_DATATYPE, f"GC_{uid}",
        )
        return
    if kind == "MMI split-combine cascade":
        count=max(1,int(p.get("cascade_count",3)));d=float(p.get("interconnect_length",5.0));width=float(p["wg_width"]);m=mmi_total_length(p)
        if d<0:raise ValueError("MMI split-combine interconnect length cannot be negative.")
        current=np.asarray(start,float)
        final_combiner_output=tuple(current)
        gc_params=dict(p);gc_params["gc_wg_length"]=float(p.get("gc_wg_length",50.0))
        if bool(p.get("add_input_grating_coupler",True)):
            add_parent_focusing_gc(top,gc_params,tuple(current),(orientation+180)%360,width,layer,datatype,f"SPLIT_COMBINE_{uid}_INPUT_GC")
        for index in range(count):
            _,splitter,upper,lower,end_orientation,*_=backend.make_1x2_mmi_gdstk(mmi_width=float(p["mmi_width"]),mmi_length=float(p["mmi_length"]),wg_width=width,taper_width=float(p["taper_width"]),input_taper_length=float(p["input_taper_length"]),output_taper_length=float(p["output_taper_length"]),input_length=float(p["input_length"]),output_length=float(p["output_length"]),port_sep=float(p["port_sep"]),taper_power=float(p["taper_power"]),taper_points=int(p["taper_points"]),input_center=tuple(current),orientation_deg=orientation,layer=layer,datatype=datatype,cell_name=f"SPLIT_COMBINE_{uid}_{index+1}_SPLITTER",gds_file=None,lib=None)
            copy_cell_polygons_to_top(splitter,top)
            for branch in (upper,lower):
                add_rect(top,np.array([[0,-width/2],[d,-width/2],[d,width/2],[0,width/2]],float),branch,end_orientation,layer,datatype)
            combiner_single=tuple(current+rot((2*m+d,0),orientation))
            final_combiner_output=combiner_single
            _,combiner,*_=backend.make_1x2_mmi_gdstk(mmi_width=float(p["mmi_width"]),mmi_length=float(p["mmi_length"]),wg_width=width,taper_width=float(p["taper_width"]),input_taper_length=float(p["input_taper_length"]),output_taper_length=float(p["output_taper_length"]),input_length=float(p["input_length"]),output_length=float(p["output_length"]),port_sep=float(p["port_sep"]),taper_power=float(p["taper_power"]),taper_points=int(p["taper_points"]),input_center=combiner_single,orientation_deg=(orientation+180)%360,layer=layer,datatype=datatype,cell_name=f"SPLIT_COMBINE_{uid}_{index+1}_COMBINER",gds_file=None,lib=None)
            copy_cell_polygons_to_top(combiner,top)
            if index<count-1:
                add_rect(top,np.array([[0,-width/2],[d,-width/2],[d,width/2],[0,width/2]],float),combiner_single,orientation,layer,datatype)
                current=np.asarray(combiner_single)+rot((d,0),orientation)
        opposed=bool(p.get("add_opposed_output_s_bends",True));sb_length=float(p.get("output_s_bend_length",200.0));sb_offset=float(p.get("output_s_bend_offset",100.0))
        if opposed and (sb_length<=0 or sb_offset<0):raise ValueError("Output S-bend length must be positive and offset cannot be negative.")
        if bool(p.get("add_output_grating_coupler",True)):
            dut_gc_point=final_combiner_output;dut_gc_orientation=orientation
            if opposed:
                dut_bend,dut_gc_point,dut_gc_orientation,*_=backend.make_s_bend_gdstk(length=sb_length,offset=-ms*sb_offset,width=width,start_center=final_combiner_output,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("gc_euler_tolerance",0.001)));top.add(dut_bend)
            add_parent_focusing_gc(top,gc_params,dut_gc_point,dut_gc_orientation,width,layer,datatype,f"SPLIT_COMBINE_{uid}_OUTPUT_GC")
        if bool(p.get("add_reference_waveguide",True)):
            reference_offset=ms*float(p.get("reference_vertical_offset",150.0))
            if abs(reference_offset)<width:raise ValueError("Reference vertical offset must separate the reference from the cascade.")
            reference_start=tuple(np.asarray(start)+rot((0,reference_offset),orientation));reference_length=count*(2*m+d)+max(0,count-1)*d
            add_rect(top,np.array([[0,-width/2],[reference_length,-width/2],[reference_length,width/2],[0,width/2]],float),reference_start,orientation,layer,datatype)
            reference_end=tuple(np.asarray(reference_start)+rot((reference_length,0),orientation))
            if bool(p.get("add_input_grating_coupler",True)):add_parent_focusing_gc(top,gc_params,reference_start,(orientation+180)%360,width,layer,datatype,f"SPLIT_COMBINE_{uid}_REFERENCE_INPUT_GC")
            if bool(p.get("add_output_grating_coupler",True)):
                reference_gc_point=reference_end;reference_gc_orientation=orientation
                if opposed:
                    reference_bend,reference_gc_point,reference_gc_orientation,*_=backend.make_s_bend_gdstk(length=sb_length,offset=ms*sb_offset,width=width,start_center=reference_end,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("gc_euler_tolerance",0.001)));top.add(reference_bend)
                add_parent_focusing_gc(top,gc_params,reference_gc_point,reference_gc_orientation,width,layer,datatype,f"SPLIT_COMBINE_{uid}_REFERENCE_OUTPUT_GC")
        return
    if kind == "Cascaded MMI":
        levels=int(p.get("N_levels",p.get("N_mmi",1)))
        if levels<1 or levels>8:raise ValueError("Cascaded MMI requires N_levels between 1 and 8 (total MMIs = 2^N_levels - 1).")
        sb_min=float(p.get("s_bend_length",300));min_radius=float(p.get("minimum_s_bend_radius",200));width=float(p["wg_width"])
        if sb_min<=0 or min_radius<=0:raise ValueError("Cascaded MMI s_bend_length and minimum_s_bend_radius must be positive.")
        if bool(p.get("add_input_grating_coupler",False)):
            bend=grating_route_bend_angle(p,mirrored,"gc_route",180.0);add_routed_parent_gc(top,p,start,(orientation+180.0)%360.0,bend,width,layer,datatype,f"CASCADE_{uid}_INPUT_GC")
        stage_starts=[start];terminal=[];mmi_index=0
        gc_spacing=float(p.get("output_gc_spacing",150))
        if gc_spacing<=float(p["port_sep"]):raise ValueError("output_gc_spacing must be larger than the MMI port separation.")
        for level in range(levels):
            next_starts=[];spread=gc_spacing*(2**max(0,levels-level-2));sep=float(p["port_sep"])/2.0;sb_length=max(sb_min,math.sqrt(6.0*min_radius*max(0.0,spread-sep)))
            for stage_start in stage_starts:
                mmi_index+=1;_,cell,upper_end,lower_end,end_orientation,*_=backend.make_1x2_mmi_gdstk(mmi_width=float(p["mmi_width"]),mmi_length=float(p["mmi_length"]),wg_width=width,taper_width=float(p["taper_width"]),input_taper_length=float(p["input_taper_length"]),output_taper_length=float(p["output_taper_length"]),input_length=float(p["input_length"]),output_length=float(p["output_length"]),port_sep=float(p["port_sep"]),taper_power=float(p["taper_power"]),taper_points=int(p["taper_points"]),input_center=stage_start,orientation_deg=orientation,layer=layer,datatype=datatype,cell_name=f"CASCADE_{uid}_L{level+1}_MMI_{mmi_index}",gds_file=None,lib=None)
                copy_cell_polygons_to_top(cell,top)
                if level==levels-1:terminal.extend((upper_end,lower_end));continue
                sep=float(p["port_sep"])/2.0
                for branch_end,sign in ((upper_end,1.0),(lower_end,-1.0)):
                    offset=sign*(spread-sep)
                    if mirrored:offset=-offset
                    path,next_start,next_orientation,*_=backend.make_s_bend_gdstk(length=sb_length,offset=offset,width=width,start_center=branch_end,orientation_deg=end_orientation,layer=layer,datatype=datatype,tolerance=float(p.get("gc_euler_tolerance",0.001)))
                    top.add(path);next_starts.append(next_start)
            stage_starts=next_starts
        if bool(p.get("add_output_grating_coupler",True)):
            output_sb=float(p.get("output_s_bend_length",300));gc_spacing=float(p.get("output_gc_spacing",150))
            if output_sb<=0 or gc_spacing<=0:raise ValueError("Final grating fan-out requires positive output_s_bend_length and output_gc_spacing.")
            bend=grating_route_bend_angle(p,mirrored,"gc_route",0.0);sep=float(p["port_sep"])/2.0
            # terminal is stored as repeated (upper, lower) pairs. Every pair
            # receives the exact same S-bend and its mirrored counterpart.
            for leaf_index,leaf in enumerate(terminal,1):
                sign=1.0 if (leaf_index-1)%2==0 else -1.0;offset=sign*(gc_spacing/2.0-sep)
                if mirrored:offset=-offset
                path,gc_start,gc_orientation,*_=backend.make_s_bend_gdstk(length=output_sb,offset=offset,width=width,start_center=leaf,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("gc_euler_tolerance",0.001)));top.add(path);add_routed_parent_gc(top,p,gc_start,gc_orientation,bend,width,layer,datatype,f"CASCADE_{uid}_LEAF_{leaf_index}_GC")
        return
    if kind == "1x2 MMI":
        _, cell, upper_end, lower_end, end_orientation, *_ = backend.make_1x2_mmi_gdstk(
            mmi_width=float(p["mmi_width"]), mmi_length=float(p["mmi_length"]),
            wg_width=float(p["wg_width"]), taper_width=float(p["taper_width"]),
            input_taper_length=float(p["input_taper_length"]), output_taper_length=float(p["output_taper_length"]),
            input_length=float(p["input_length"]), output_length=float(p["output_length"]),
            port_sep=float(p["port_sep"]), taper_power=float(p["taper_power"]),
            taper_points=int(p["taper_points"]), input_center=start, orientation_deg=orientation,
            layer=layer, datatype=datatype, cell_name=f"MMI_{uid}", gds_file=None, lib=None,
        )
        copy_cell_polygons_to_top(cell, top)
        if bool(p.get("add_grating_couplers", False)):
            width = float(p["wg_width"])
            input_bend = grating_route_bend_angle(p, mirrored, "gc_input_route", 180.0)
            add_routed_parent_gc(
                top, p, start, (orientation + 180.0) % 360.0, input_bend,
                width, layer, datatype, f"MMI_{uid}_INPUT_GC",
            )

            fanout = mmi_gc_fanout_local_points(p, mirrored)
            fanout_length = float(p.get("gc_s_bend_length", 80.0))
            upper_offset = fanout["upper_fanout_end"][1] - fanout["upper_start"][1]
            lower_offset = fanout["lower_fanout_end"][1] - fanout["lower_start"][1]
            upper_s, upper_fanout_end, upper_fanout_orientation, *_ = backend.make_s_bend_gdstk(
                length=fanout_length, offset=upper_offset, width=width,
                start_center=upper_end, orientation_deg=end_orientation,
                layer=layer, datatype=datatype, tolerance=float(p.get("gc_euler_tolerance", 0.001)),
            )
            lower_s, lower_fanout_end, lower_fanout_orientation, *_ = backend.make_s_bend_gdstk(
                length=fanout_length, offset=lower_offset, width=width,
                start_center=lower_end, orientation_deg=end_orientation,
                layer=layer, datatype=datatype, tolerance=float(p.get("gc_euler_tolerance", 0.001)),
            )
            top.add(upper_s, lower_s)

            upper_bend = grating_route_bend_angle(p, mirrored, "gc_upper_output_route", 0.0)
            lower_bend = grating_route_bend_angle(p, mirrored, "gc_lower_output_route", 0.0)
            add_routed_parent_gc(
                top, p, upper_fanout_end, upper_fanout_orientation, upper_bend,
                width, layer, datatype, f"MMI_{uid}_UPPER_GC",
            )
            add_routed_parent_gc(
                top, p, lower_fanout_end, lower_fanout_orientation, lower_bend,
                width, layer, datatype, f"MMI_{uid}_LOWER_GC",
            )
        return
    if kind == "MMI + Reference":
        _, mmi_cell, upper_end, lower_end, end_orientation, *_ = backend.make_1x2_mmi_gdstk(
            mmi_width=float(p["mmi_width"]),
            mmi_length=float(p["mmi_length"]),
            wg_width=float(p["wg_width"]),
            taper_width=float(p["taper_width"]),
            input_taper_length=float(p["input_taper_length"]),
            output_taper_length=float(p["output_taper_length"]),
            input_length=float(p["input_length"]),
            output_length=float(p["output_length"]),
            port_sep=float(p["port_sep"]),
            taper_power=float(p["taper_power"]),
            taper_points=int(p["taper_points"]),
            input_center=start,
            orientation_deg=orientation,
            layer=layer,
            datatype=datatype,
            cell_name=f"MMI_REFERENCE_MMI_{uid}",
            gds_file=None,
            lib=None,
        )
        copy_cell_polygons_to_top(mmi_cell, top)

        width = float(p["wg_width"])
        total_mmi_length = mmi_total_length(p)
        reference_y = ms * float(p.get("reference_dy", 250.0))
        selected_upper = str(p.get("reference_branch", "upper")).lower() != "lower"
        branch_sign = 1.0 if selected_upper else -1.0
        target_half = max(
            abs(float(p["port_sep"])) / 2.0,
            abs(float(p.get("gc_output_separation", p["port_sep"]))) / 2.0,
        )
        branch_fanout_y = ms * branch_sign * target_half
        fanout_length = float(p.get("gc_s_bend_length", 80.0))

        add_rect(
            top,
            np.array(
                [
                    [0.0, reference_y - width / 2.0],
                    [total_mmi_length, reference_y - width / 2.0],
                    [total_mmi_length, reference_y + width / 2.0],
                    [0.0, reference_y + width / 2.0],
                ],
                dtype=float,
            ),
            start,
            orientation,
            layer,
            datatype,
        )

        reference_bend_start = tuple(
            np.asarray(start)
            + rot((total_mmi_length, reference_y), orientation)
        )
        reference_s_bend, reference_fanout_end, reference_orientation, *_ = (
            backend.make_s_bend_gdstk(
                length=fanout_length,
                offset=branch_fanout_y,
                width=width,
                start_center=reference_bend_start,
                orientation_deg=orientation,
                layer=layer,
                datatype=datatype,
                tolerance=float(p.get("gc_euler_tolerance", 0.001)),
            )
        )
        top.add(reference_s_bend)

        if bool(p.get("add_grating_couplers", False)):
            input_bend = grating_route_bend_angle(
                p, mirrored, "gc_input_route", 180.0
            )
            upper_bend = grating_route_bend_angle(
                p, mirrored, "gc_upper_output_route", 0.0
            )
            lower_bend = grating_route_bend_angle(
                p, mirrored, "gc_lower_output_route", 0.0
            )
            branch_bend = upper_bend if selected_upper else lower_bend

            add_routed_parent_gc(
                top,
                p,
                start,
                (orientation + 180.0) % 360.0,
                input_bend,
                width,
                layer,
                datatype,
                f"MMI_REFERENCE_{uid}_MMI_INPUT_GC",
            )

            fanout = mmi_gc_fanout_local_points(p, mirrored)
            upper_offset = (
                fanout["upper_fanout_end"][1]
                - fanout["upper_start"][1]
            )
            lower_offset = (
                fanout["lower_fanout_end"][1]
                - fanout["lower_start"][1]
            )
            upper_s, upper_fanout_end, upper_orientation, *_ = (
                backend.make_s_bend_gdstk(
                    length=fanout_length,
                    offset=upper_offset,
                    width=width,
                    start_center=upper_end,
                    orientation_deg=end_orientation,
                    layer=layer,
                    datatype=datatype,
                    tolerance=float(p.get("gc_euler_tolerance", 0.001)),
                )
            )
            lower_s, lower_fanout_end, lower_orientation, *_ = (
                backend.make_s_bend_gdstk(
                    length=fanout_length,
                    offset=lower_offset,
                    width=width,
                    start_center=lower_end,
                    orientation_deg=end_orientation,
                    layer=layer,
                    datatype=datatype,
                    tolerance=float(p.get("gc_euler_tolerance", 0.001)),
                )
            )
            top.add(upper_s, lower_s)
            add_routed_parent_gc(
                top,
                p,
                upper_fanout_end,
                upper_orientation,
                upper_bend,
                width,
                layer,
                datatype,
                f"MMI_REFERENCE_{uid}_MMI_UPPER_GC",
            )
            add_routed_parent_gc(
                top,
                p,
                lower_fanout_end,
                lower_orientation,
                lower_bend,
                width,
                layer,
                datatype,
                f"MMI_REFERENCE_{uid}_MMI_LOWER_GC",
            )

            reference_input = tuple(
                np.asarray(start) + rot((0.0, reference_y), orientation)
            )
            add_routed_parent_gc(
                top,
                p,
                reference_input,
                (orientation + 180.0) % 360.0,
                input_bend,
                width,
                layer,
                datatype,
                f"MMI_REFERENCE_{uid}_REFERENCE_INPUT_GC",
            )
            add_routed_parent_gc(
                top,
                p,
                reference_fanout_end,
                reference_orientation,
                branch_bend,
                width,
                layer,
                datatype,
                f"MMI_REFERENCE_{uid}_REFERENCE_OUTPUT_GC",
            )
        return
    if kind in {"Vertical-GC MZI test block","Vertical-GC MZI + CPW test block","Vertical-GC MZI + segmented electrode test block"}:
        active=set(p.get("sweep_parameters",["mmi_length"]));count=max(1,int(p.get("mzi_count",5))) if "mmi_length" in active else 1;row_spacing=float(p.get("vertical_spacing",1800));field=float(p.get("ebeam_field_size",520));clearance=float(p.get("ebeam_edge_clearance",10));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["MZI vertical GC"])
        for key,value in p.items():
            if key in sub:sub[key]=value
        # Never allow the requested pitch to be smaller than the actual optical
        # device height.  This is especially important for vertical GC routes.
        probe_total=float(p.get("mzi_total_length",10000));sub["mmi_length"]=float(p.get("mmi_length_start",27));sub["arm_length"]=probe_total-2*mmi_total_length(sub)-2*float(sub["s_bend_length"])
        probe_library=gdstk.Library(unit=1e-6,precision=1e-9);probe_cell=probe_library.new_cell(f"VERTICAL_MZI_SPACING_{uid}");_add_component_geometry_to_cell({"uid":uid*1000,"kind":"MZI vertical GC","x":0.0,"y":0.0,"orientation_deg":0.0,"mirrored":mirrored,"params":sub},probe_cell);probe_bbox=probe_cell.bounding_box()
        if probe_bbox is not None:row_spacing=max(row_spacing,float(probe_bbox[1][1]-probe_bbox[0][1])+20.0)
        field_order=0;block_min=np.array([np.inf,np.inf]);block_max=np.array([-np.inf,-np.inf])
        for index in range(count):
            total=float(p.get("mzi_total_length",10000));sub["mmi_length"]=(float(p.get("mmi_length_start",27))+index*float(p.get("mmi_length_step",1))) if "mmi_length" in active else float(p.get("mmi_length",29));sub["arm_length"]=total-2*mmi_total_length(sub)-2*float(sub["s_bend_length"])
            if sub["arm_length"]<=0:raise ValueError("MMI sweep leaves no positive MZI arm length.")
            local_y=(index-(count-1)/2)*row_spacing;row_start=tuple(np.asarray(start)+rot((0,local_y),orientation));library=gdstk.Library(unit=1e-6,precision=1e-9);cell=library.new_cell(f"VERTICAL_MZI_TMP_{uid}_{index}");_add_component_geometry_to_cell({"uid":uid*1000+index+1,"kind":"MZI vertical GC","x":0.0,"y":0.0,"orientation_deg":0.0,"mirrored":mirrored,"params":sub},cell);device_polygons=cell.get_polygons(apply_repetitions=True,include_paths=True);bbox=cell.bounding_box()
            if bbox is None:continue
            block_min=np.minimum(block_min,np.asarray(bbox[0],float)+np.array([0,local_y]));block_max=np.maximum(block_max,np.asarray(bbox[1],float)+np.array([0,local_y]))
            for polygon in device_polygons:top.add(gdstk.Polygon(transform_points(np.asarray(polygon.points,float),row_start,orientation),layer=int(polygon.layer),datatype=int(polygon.datatype)))
            if kind=="Vertical-GC MZI + CPW test block" or bool(p.get("include_symmetric_cpw",False)):
                taper_length=float(p.get("cpw_taper_length",500));end_length=.05*total;clear=10.0;cpw_length=float(sub["arm_length"])-2*clear;middle_length=cpw_length-2*taper_length-2*end_length
                if min(cpw_length,middle_length,taper_length,end_length)<=0:raise ValueError("CPW middle-flat, taper, and outer-flat lengths must be positive.")
                cpw_params=safe_json_copy(DEFAULT_COMPONENT_VALUES["Symmetric CPW taper"]);cpw_params.update({"signal_width":float(p.get("cpw_signal_width",130)),"ground_width":float(p.get("cpw_ground_width",130)),"initial_gap":float(p.get("cpw_end_gap",14.5)),"middle_gap":float(p.get("cpw_middle_gap",3)),"end_straight_length":end_length,"taper_length":taper_length,"middle_straight_length":middle_length,"profile":str(p.get("cpw_profile","klopfenstein")),"target_s11_db":float(p.get("cpw_target_s11_db",20)),"exponential_factor":float(p.get("cpw_exponential_factor",1)),"points":int(p.get("cpw_points",161)),"layer":RF_LAYER,"datatype":0});cpw_start=tuple(np.asarray(row_start)+rot((mmi_total_length(sub)+float(sub["s_bend_length"])+clear,0),orientation));_add_component_geometry_to_cell({"uid":uid*100000+index,"kind":"Symmetric CPW taper","x":cpw_start[0],"y":cpw_start[1],"orientation_deg":orientation,"mirrored":False,"params":cpw_params},top)
            if kind=="Vertical-GC MZI + segmented electrode test block" or bool(p.get("include_segmented_electrode",False)):
                clear=float(p.get("seg_s_bend_clearance",10));transition=float(p.get("seg_taper_length",500));end_flat=float(p.get("seg_end_flat_length",50));inner_flat=float(p.get("seg_inner_flat_length",50));top_length=float(p.get("seg_t_top_length",45));segment_spacing=float(p.get("seg_segment_spacing",3));target_length=float(sub["arm_length"])-2*clear
                if min(target_length,transition,end_flat,inner_flat,top_length)<=0 or segment_spacing<0:raise ValueError("Segmented-electrode length, tapers, flats, finger length, and spacing must fit between the MZI S-bends.")
                if bool(p.get("seg_auto_segment_count",True)):
                    period=top_length+segment_spacing;available=target_length-2*(transition+end_flat+inner_flat);segments=int(math.floor(available/period))
                    if segments<1:raise ValueError("The MZI arm is too short for two 500 µm tapers and a segmented-electrode region.")
                    inner_flat+=(target_length-(2*(transition+end_flat+inner_flat)+segments*period))/2
                else:segments=max(1,int(p.get("seg_segment_count",80)))
                seg_params=safe_json_copy(DEFAULT_COMPONENT_VALUES["Segmented electrode"]);seg_params.update({"signal_width":float(p.get("seg_signal_width",130)),"end_gap":float(p.get("seg_end_gap",14.5)),"gap":float(p.get("seg_gap",3)),"ground_width":float(p.get("seg_ground_width",130)),"transition_length":transition,"end_flat_length":end_flat,"inner_flat_length":inner_flat,"t_top_width":float(p.get("seg_t_top_width",2)),"t_top_length":top_length,"t_neck_width":float(p.get("seg_t_neck_width",4)),"t_neck_length":float(p.get("seg_t_neck_length",18)),"segment_spacing":segment_spacing,"segment_count":segments,"include_oxide_masks":bool(p.get("seg_include_oxide_masks",False)),"layer":RF_LAYER,"datatype":0});seg_start=tuple(np.asarray(row_start)+rot((mmi_total_length(sub)+float(sub["s_bend_length"])+clear,0),orientation));_add_component_geometry_to_cell({"uid":uid*200000+index,"kind":"Segmented electrode","x":seg_start[0],"y":seg_start[1],"orientation_deg":orientation,"mirrored":False,"params":seg_params},top)
            if not bool(p.get("include_ebeam_fields",True)):continue
            minimum=np.asarray(bbox[0],float);maximum=np.asarray(bbox[1],float);extent=maximum-minimum;grid_center=(minimum+maximum)/2;nx=max(1,math.ceil((extent[0]+2*clearance)/field));ny=max(1,math.ceil((extent[1]+2*clearance)/field))
            for iy in range(ny):
                for ix in range(nx):
                    center_local=grid_center+np.array([(ix-(nx-1)/2)*field,(iy-(ny-1)/2)*field]);rect=gdstk.rectangle(tuple(center_local-field/2),tuple(center_local+field/2));intersection=gdstk.boolean([rect],device_polygons,"and",precision=1e-6)
                    if not intersection:continue
                    field_order+=1;field_center=tuple(np.asarray(row_start)+rot(center_local,orientation));add_ebeam_field_outline(top,field_center,orientation,field);add_write_field_number(top,field_order,field_center,orientation,field)
        if np.all(np.isfinite(block_min)):
            first_mmi=float(p.get("mmi_length_start",27));last_mmi=first_mmi+(count-1)*float(p.get("mmi_length_step",1));label=f"VERTICAL MZI BLOCK  MMI={first_mmi:g}-{last_mmi:g}  L={float(p.get('mzi_total_length',10000)):g}"
            if kind=="Vertical-GC MZI + CPW test block":label+=f"  CPW TAP={float(p.get('cpw_taper_length',500)):g} CLR=10"
            if kind=="Vertical-GC MZI + segmented electrode test block":label+=f"  T-SEG N={'AUTO' if bool(p.get('seg_auto_segment_count',True)) else int(p.get('seg_segment_count',80))} TAP={float(p.get('seg_taper_length',500)):g} CLR={float(p.get('seg_s_bend_clearance',10)):g}"
            label_local=((block_min[0]+block_max[0])/2,block_max[1]+20);label_point=tuple(np.asarray(start)+rot(label_local,orientation));add_ebeam_parameter_text(top,label,label_point,orientation,0,float(p.get("parameter_text_height",10)))
        return
    if kind == "Long MZI test block":
        total=float(p.get("mzi_total_length",10000));count=max(1,int(p.get("mzi_count",5)));vertical_spacing=float(p.get("vertical_spacing",1000));gc_straight=float(p.get("gc_straight_length",200));width=float(p.get("wg_width",1.2));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["MZI"]);sub.update({key:p[key] for key in ("mmi_width","mmi_length","wg_width","taper_width","input_taper_length","output_taper_length","input_length","output_length","port_sep","arm_separation","s_bend_length","taper_power","taper_points") if key in p});mlen=mmi_total_length(sub);arm=total-2*mlen-2*float(sub["s_bend_length"])
        if arm<=0 or gc_straight<=0:raise ValueError("Long MZI total length must exceed both MMIs and S-bends; gc_straight_length must be positive.")
        sub["arm_length"]=arm;sub["add_grating_couplers"]=False;gc_params=dict(p);left_bend=grating_route_bend_angle(gc_params,mirrored,"gc_input_route",180);right_bend=grating_route_bend_angle(gc_params,mirrored,"gc_output_route",0)
        for index in range(count):
            local_y=(index-(count-1)/2)*vertical_spacing;row_start=tuple(np.asarray(start)+rot((0,local_y),orientation));_add_component_geometry_to_cell({"uid":uid*1000+index+1,"kind":"MZI","x":row_start[0],"y":row_start[1],"orientation_deg":orientation,"mirrored":mirrored,"params":sub},top);add_rect(top,np.array([[-gc_straight,-width/2],[0,-width/2],[0,width/2],[-gc_straight,width/2]],float),row_start,orientation,layer,datatype);right=tuple(np.asarray(row_start)+rot((total,0),orientation));add_rect(top,np.array([[0,-width/2],[gc_straight,-width/2],[gc_straight,width/2],[0,width/2]],float),right,orientation,layer,datatype);left_gc=tuple(np.asarray(row_start)+rot((-gc_straight,0),orientation));right_gc=tuple(np.asarray(row_start)+rot((total+gc_straight,0),orientation));add_routed_parent_gc(top,gc_params,left_gc,(orientation+180)%360,left_bend,width,layer,datatype,f"LONG_MZI_{uid}_{index+1}_LEFT_GC");add_routed_parent_gc(top,gc_params,right_gc,orientation,right_bend,width,layer,datatype,f"LONG_MZI_{uid}_{index+1}_RIGHT_GC")
            if bool(p.get("include_ebeam_fields",True)):
                field=float(p.get("ebeam_field_size",520));clearance=float(p.get("ebeam_edge_clearance",10));gc_extent=float(p.get("gc_taper_L",22))+float(p.get("gc_wg_length",20))+int(p.get("gc_N",30))*float(p.get("gc_pitch",.75));device_span=total+2*gc_straight+2*gc_extent;field_count=max(1,math.ceil((device_span+2*clearance)/field));coverage_width=field_count*field;coverage_center=total/2
                for field_index in range(field_count):
                    local_x=coverage_center+(field_index-(field_count-1)/2)*field;field_center=tuple(np.asarray(row_start)+rot((local_x,0),orientation));add_ebeam_field_outline(top,field_center,orientation,field);add_write_field_number(top,index*field_count+field_index+1,field_center,orientation,field);add_ebeam_parameter_text(top,f"MZI{index+1} L={total:g}",field_center,orientation,field,float(p.get("parameter_text_height",10)))
        return
    if kind in {"MZI", "MZI vertical GC"}:
        common = dict(
            mmi_width=float(p["mmi_width"]), mmi_length=float(p["mmi_length"]),
            wg_width=float(p["wg_width"]), taper_width=float(p["taper_width"]),
            input_taper_length=float(p["input_taper_length"]), output_taper_length=float(p["output_taper_length"]),
            input_length=float(p["input_length"]), output_length=float(p["output_length"]),
            port_sep=float(p["port_sep"]), taper_power=float(p["taper_power"]),
            taper_points=int(p["taper_points"]), layer=layer, datatype=datatype, gds_file=None, lib=None,
        )
        _, first, upper, lower, arm_orientation, *_ = backend.make_1x2_mmi_gdstk(
            input_center=start, orientation_deg=orientation, cell_name=f"MZI_A_{uid}", **common
        )
        copy_cell_polygons_to_top(first, top)
        delta = max(0.0, (float(p["arm_separation"]) - float(p["port_sep"])) / 2.0)
        sb_len = float(p["s_bend_length"]); arm_len = float(p["arm_length"]); width = float(p["wg_width"])
        up1, up_end, up_o, *_ = backend.make_s_bend_gdstk(length=sb_len, offset=delta, width=width, start_center=upper, orientation_deg=arm_orientation, layer=layer, datatype=datatype)
        lo1, lo_end, lo_o, *_ = backend.make_s_bend_gdstk(length=sb_len, offset=-delta, width=width, start_center=lower, orientation_deg=arm_orientation, layer=layer, datatype=datatype)
        top.add(up1, lo1)
        for c0, o0 in ((up_end, up_o), (lo_end, lo_o)):
            local = np.array([[0,-width/2],[arm_len,-width/2],[arm_len,width/2],[0,width/2]], float)
            add_rect(top, local, c0, o0, layer, datatype)
        up_mid = tuple(np.asarray(up_end)+rot((arm_len,0),up_o)); lo_mid = tuple(np.asarray(lo_end)+rot((arm_len,0),lo_o))
        up2, up_final, _, *_ = backend.make_s_bend_gdstk(length=sb_len, offset=-delta, width=width, start_center=up_mid, orientation_deg=up_o, layer=layer, datatype=datatype)
        lo2, lo_final, _, *_ = backend.make_s_bend_gdstk(length=sb_len, offset=delta, width=width, start_center=lo_mid, orientation_deg=lo_o, layer=layer, datatype=datatype)
        top.add(up2, lo2)
        mlen = mmi_total_length(p)
        output_center = tuple((np.asarray(up_final)+np.asarray(lo_final))/2.0 + rot((mlen,0), orientation))
        _, second, *_ = backend.make_1x2_mmi_gdstk(input_center=output_center, orientation_deg=(orientation+180)%360, cell_name=f"MZI_B_{uid}", **common)
        copy_cell_polygons_to_top(second, top)
        if bool(p.get("add_grating_couplers", False)):
            total_length = 2.0 * mlen + 2.0 * sb_len + arm_len
            right_external = tuple(np.asarray(start) + rot((total_length, 0.0), orientation))
            if kind=="MZI vertical GC" or bool(p.get("gc_three_euler_inward",False)):
                add_three_euler_inward_gc(top,p,start,(orientation+180)%360,True,width,layer,datatype,f"MZI_{uid}_LEFT_VERTICAL_GC");add_three_euler_inward_gc(top,p,right_external,orientation,False,width,layer,datatype,f"MZI_{uid}_RIGHT_VERTICAL_GC")
            else:
                left_bend = grating_route_bend_angle(p, mirrored, "gc_input_route", 180.0);right_bend = grating_route_bend_angle(p, mirrored, "gc_output_route", 0.0);add_routed_parent_gc(top, p, start, (orientation + 180.0) % 360.0, left_bend, width, layer, datatype, f"MZI_{uid}_LEFT_GC");add_routed_parent_gc(top, p, right_external, orientation, right_bend, width, layer, datatype, f"MZI_{uid}_RIGHT_GC")
        return
    if kind == "MZI + CPW module":
        mlen = mmi_total_length(p)
        sb_len = float(p["s_bend_length"]); arm_len = float(p["arm_length"])
        active_len = 2.0 * sb_len + arm_len
        tin = float(p["rf_input_taper_length"]); tout = float(p["rf_output_taper_length"])
        total_rf = tin + active_len + tout
        ws = float(p["signal_width"]); wg = float(p["ground_width"])
        g_ext = float(p["external_gap"]); g_int = float(p["interaction_gap"])
        if min(tin, tout, active_len, ws, wg, g_ext, g_int) <= 0:
            raise ValueError("Integrated MZI + CPW module dimensions must be positive.")
        rf_layer = RF_LAYER; rf_dt = DEFAULT_DATATYPE
        add_rect(top, np.array([[0,-ws/2],[total_rf,-ws/2],[total_rf,ws/2],[0,ws/2]],float), start, orientation, rf_layer, rf_dt)
        count = max(16, int(p.get("points", 161)))
        xs = np.linspace(0.0, total_rf, count)
        gaps = np.empty_like(xs)
        def taper_value(u: float, a: float, b: float, profile: str) -> float:
            uu, values = _gap_transition_values(
                a, b, max(65, count), profile, ws,
                float(p.get("target_s11_db", 20.0)),
                float(p.get("exponential_factor", 1.0)),
            )
            return float(np.interp(min(1.0, max(0.0, float(u))), uu, values))
        for i, x in enumerate(xs):
            if x <= tin:
                gaps[i] = taper_value(x/tin, g_ext, g_int, str(p.get("rf_input_taper_profile", "linear")).lower())
            elif x <= tin + active_len:
                gaps[i] = g_int
            else:
                gaps[i] = taper_value((x-tin-active_len)/tout, g_int, g_ext, str(p.get("rf_output_taper_profile", "linear")).lower())
        upper_inner = np.column_stack((xs, ws/2.0 + gaps))
        upper_outer = np.column_stack((xs[::-1], (ws/2.0 + gaps + wg)[::-1]))
        lower_outer = np.column_stack((xs, -ws/2.0 - gaps - wg))
        lower_inner = np.column_stack((xs[::-1], (-ws/2.0 - gaps)[::-1]))
        add_local_polygon(top, np.vstack((upper_inner, upper_outer)), start, orientation, rf_layer, rf_dt)
        add_local_polygon(top, np.vstack((lower_outer, lower_inner)), start, orientation, rf_layer, rf_dt)

        common = dict(
            mmi_width=float(p["mmi_width"]), mmi_length=float(p["mmi_length"]),
            wg_width=float(p["wg_width"]), taper_width=float(p["taper_width"]),
            input_taper_length=float(p["input_taper_length"]), output_taper_length=float(p["output_taper_length"]),
            input_length=float(p["input_length"]), output_length=float(p["output_length"]),
            port_sep=float(p["port_sep"]), taper_power=float(p["taper_power"]),
            taper_points=int(p["taper_points"]), layer=layer, datatype=datatype, gds_file=None, lib=None,
        )
        mzi_start = tuple(np.asarray(start) + rot((tin-mlen, 0.0), orientation))
        _, first, upper, lower, arm_orientation, *_ = backend.make_1x2_mmi_gdstk(input_center=mzi_start, orientation_deg=orientation, cell_name=f"MZI_CPW_A_{uid}", **common)
        copy_cell_polygons_to_top(first, top)
        delta=max(0.0,(float(p["arm_separation"])-float(p["port_sep"]))/2.0); width=float(p["wg_width"])
        up1,up_end,up_o,*_=backend.make_s_bend_gdstk(length=sb_len,offset=delta,width=width,start_center=upper,orientation_deg=arm_orientation,layer=layer,datatype=datatype)
        lo1,lo_end,lo_o,*_=backend.make_s_bend_gdstk(length=sb_len,offset=-delta,width=width,start_center=lower,orientation_deg=arm_orientation,layer=layer,datatype=datatype)
        top.add(up1,lo1)
        for c0,o0 in ((up_end,up_o),(lo_end,lo_o)):
            add_rect(top,np.array([[0,-width/2],[arm_len,-width/2],[arm_len,width/2],[0,width/2]],float),c0,o0,layer,datatype)
        up_mid=tuple(np.asarray(up_end)+rot((arm_len,0),up_o)); lo_mid=tuple(np.asarray(lo_end)+rot((arm_len,0),lo_o))
        up2,up_final,*_=backend.make_s_bend_gdstk(length=sb_len,offset=-delta,width=width,start_center=up_mid,orientation_deg=up_o,layer=layer,datatype=datatype)
        lo2,lo_final,*_=backend.make_s_bend_gdstk(length=sb_len,offset=delta,width=width,start_center=lo_mid,orientation_deg=lo_o,layer=layer,datatype=datatype)
        top.add(up2,lo2)
        output_center=tuple((np.asarray(up_final)+np.asarray(lo_final))/2.0+rot((mlen,0),orientation))
        _,second,*_=backend.make_1x2_mmi_gdstk(input_center=output_center,orientation_deg=(orientation+180)%360,cell_name=f"MZI_CPW_B_{uid}",**common)
        copy_cell_polygons_to_top(second,top)
        return
    if kind == "Ring":
        add_ring_to_gds(top, (0.0, 0.0), float(p["radius"]), float(p["width"]), int(p.get("points", 256)), start, orientation, layer, datatype); return
    if kind in {"Photonic crystal","Photonic crystal slab"}:
        length=float(p.get("length",50.0));width=float(p.get("width",20.0));columns=max(1,int(p.get("columns",120)));rows=max(1,int(p.get("rows",24)));px=float(p.get("pitch_x",.42));py=float(p.get("pitch_y",px*math.sqrt(3)/2));rx=float(p.get("hole_radius_x",.12));ry=float(p.get("hole_radius_y",rx));points=max(12,int(p.get("points",48)));hole_layer=int(p.get("hole_layer",3));hole_datatype=int(p.get("hole_datatype",0));triangular=str(p.get("lattice","triangular")).lower()=="triangular";elliptical=str(p.get("hole_shape","circular")).lower()=="elliptical";defect=str(p.get("device_type","bulk crystal"))=="line-defect waveguide" or bool(p.get("include_line_defect",False));defect_rows=max(1,int(p.get("defect_rows",1)));shape=str(p.get("slab_shape","rectangle"));negative=str(p.get("mask_tone","negative: slab minus holes")).startswith("negative")
        if min(length,width,px,py,rx,ry)<=0:raise ValueError("Photonic-crystal dimensions, pitches, and hole radii must be positive.")
        if 2*rx>=px or 2*ry>=py:raise ValueError("Photonic-crystal holes must be smaller than their lattice pitch.")
        if shape=="ellipse":slab_local=gdstk.ellipse((0,0),(length/2,width/2),tolerance=.001).points
        elif shape=="hexagon":slab_local=np.column_stack(((length/2)*np.cos(np.linspace(0,2*math.pi,6,endpoint=False)),(width/2)*np.sin(np.linspace(0,2*math.pi,6,endpoint=False))))
        else:slab_local=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]],float)
        hole_polys=[]
        defect_start=(rows-defect_rows)//2
        for row_index in range(rows):
            signed_row=row_index-(rows-1)/2
            if defect and defect_start<=row_index<defect_start+defect_rows:continue
            y=signed_row*py;offset=px/2 if triangular and row_index%2 else 0.0
            for column_index in range(columns):
                x=(column_index-(columns-1)/2)*px+offset
                if abs(x)+rx>length/2 or abs(y)+ry>width/2:continue
                hole_polys.append(gdstk.ellipse((x,y),(rx,ry if elliptical else rx),tolerance=max(.0001,min(rx,ry)/points)))
        local_result=gdstk.boolean([gdstk.Polygon(slab_local)],hole_polys,"not",layer=layer,datatype=datatype) if negative else hole_polys
        output_layer=layer if negative else hole_layer;output_datatype=datatype if negative else hole_datatype
        for poly in local_result:
            pts=np.asarray(poly.points,float)
            if mirrored:pts[:,1]*=-1
            world=np.asarray(start)+np.asarray([rot(point,orientation) for point in pts]);top.add(gdstk.Polygon(world,layer=output_layer,datatype=output_datatype))
        return
    if kind == "Boolean geometry":
        for points in p.get("polygons",[]):
            array=np.asarray(points,float)
            if array.ndim!=2 or len(array)<3:continue
            if mirrored:array[:,1]*=-1
            world=np.asarray(start)+np.asarray([rot(point,orientation) for point in array]);top.add(gdstk.Polygon(world,layer=layer,datatype=datatype))
        return
    if kind == "Elliptical ring":
        rx=float(p.get("radius_x",200));ry=float(p.get("radius_y",100));width=float(p.get("width",1.2))
        if min(rx,ry)<=width/2 or width<=0:raise ValueError("Elliptical-ring radii must exceed half the waveguide width.")
        outer=gdstk.ellipse((0,0),(rx+width/2,ry+width/2),tolerance=.001);inner=gdstk.ellipse((0,0),(rx-width/2,ry-width/2),tolerance=.001);rings=gdstk.boolean([outer],[inner],"not",layer=layer,datatype=datatype)
        for ring in rings:
            points=np.asarray(ring.points,float)
            if mirrored:points[:,1]*=-1
            top.add(gdstk.Polygon(np.asarray(start)+np.asarray([rot(point,orientation) for point in points]),layer=layer,datatype=datatype))
        return
    if kind == "Racetrack":
        add_racetrack_to_gds(top, (0.0, 0.0), float(p["radius"]), float(p["coupling_length"]), float(p["width"]), int(p.get("points", 128)), start, orientation, layer, datatype); return
    if kind == "Double-ring test block":
        active=set(p.get("sweep_parameters",["coupling_gap","ring_radius"]));gaps=numeric_list(p.get("gap_values","0.5,0.6,0.7,0.8,0.9,1.0")) if "coupling_gap" in active else [float(p.get("nominal_gap",.7))];radii=numeric_list(p.get("radius_values","20,50,100,200")) if "ring_radius" in active else [float(p.get("nominal_radius",200))]
        if not gaps or not radii or any(v<=0 for v in gaps+radii):raise ValueError("Ring test-block gaps and radii must be positive.")
        dx=float(p.get("column_spacing",900));dy=float(p.get("row_spacing",700))
        for row,gap_value in enumerate(gaps):
            for col,radius_value in enumerate(radii):
                local=((col-(len(radii)-1)/2)*dx,(row-(len(gaps)-1)/2)*dy);point=tuple(np.asarray(start)+rot(local,orientation));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["Ring + two feedlines"]);sub.update({"ring_radius":radius_value,"coupling_gap":gap_value,"feedline_length":float(p.get("grating_end_to_end_distance",p.get("feedline_length",365))),"feedline_width":float(p.get("feedline_width",1.2)),"ring_width":float(p.get("ring_width",1.2)),"s_bend_length":float(p.get("s_bend_length",80)),"s_bend_offset":float(p.get("s_bend_offset",20)),"input_s_bend_direction":p.get("input_s_bend_direction","up"),"output_s_bend_direction":p.get("output_s_bend_direction","up"),"pitch":float(p.get("pitch",.75)),"fill_factor":float(p.get("fill_factor",.57)),"N":int(p.get("N",30)),"alpha_t":float(p.get("alpha_t",25)),"taper_L":float(p.get("taper_L",22)),"gc_wg_length":float(p.get("gc_wg_length",20))});add_component_centered_in_field(top,"Ring + two feedlines",sub,point,orientation,mirrored,uid*1000+row*len(radii)+col+1,float(p.get("ebeam_field_size",520)),float(p.get("ebeam_edge_clearance",10)))
                add_ebeam_parameter_text(top,f"G={gap_value:g} R={radius_value:g}",point,orientation,float(p.get("ebeam_field_size",520)),float(p.get("parameter_text_height",10)))
        return
    if kind == "Grating test block":
        active=set(p.get("sweep_parameters",["pitch","fill_factor"]));nominal_pitch=float(p.get("nominal_pitch",.75));nominal_fill=float(p.get("nominal_fill",.57));pitches=inclusive_sweep(p.get("pitch_start",.73),p.get("pitch_stop",.77),p.get("pitch_step",.005)) if "pitch" in active else [nominal_pitch];fills=inclusive_sweep(p.get("fill_start",.47),p.get("fill_stop",.67),p.get("fill_step",.05)) if "fill_factor" in active else [nominal_fill]
        if not any(abs(v-nominal_pitch)<1e-9 for v in pitches):pitches.append(nominal_pitch);pitches.sort()
        if not any(abs(v-nominal_fill)<1e-9 for v in fills):fills.append(nominal_fill);fills.sort()
        dx=float(p.get("device_x_spacing",600));packing=float(p.get("packing_pitch",100));field=float(p.get("write_field_size",520));per_field=max(1,int(p.get("devices_per_field",4)));total=float(p.get("grating_end_to_end_distance",p.get("feedline_end_to_end",365)));sb=float(p.get("s_bend_length",80));straight=50.;lc=total-2*sb-2*straight
        if lc<=0:raise ValueError("feedline_end_to_end must exceed 2*s_bend_length + 100 µm.")
        devices=[(pitch_value,fill_value) for fill_value in fills for pitch_value in pitches];max_devices=max(1,int(p.get("max_devices",36)))
        if len(devices)>max_devices:
            devices=sorted(devices,key=lambda item:((item[0]-nominal_pitch)/max(abs(pitches[-1]-pitches[0]),1e-12))**2+((item[1]-nominal_fill)/max(abs(fills[-1]-fills[0]),1e-12))**2)[:max_devices];devices.sort(key=lambda item:(item[1],item[0]))
        groups=math.ceil(len(devices)/per_field)
        for index,(pitch_value,fill_value) in enumerate(devices):
                group=index//per_field;slot=index%per_field;group_size=min(per_field,len(devices)-group*per_field);group_local=((group-(groups-1)/2)*dx,0.0);local=(group_local[0],(slot-(group_size-1)/2)*packing);point=tuple(np.asarray(start)+rot(local,orientation));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"]);sub.update({"pitch":pitch_value,"fill_factor":fill_value,"N":int(p.get("N",30)),"alpha_t":float(p.get("alpha_t",25)),"taper_L":float(p.get("taper_L",22)),"gc_wg_length":float(p.get("gc_wg_length",20)),"wg_width":float(p.get("wg_width",1.2)),"input_straight_length":straight,"output_straight_length":straight,"s_bend_length":sb,"offset":float(p.get("endpoint_offset",100))/2,"Lc":lc,"input_s_bend_direction":"up","output_s_bend_direction":"up"});add_component_centered_in_field(top,"Feedline",sub,point,orientation,mirrored,uid*1000+index+1,field,float(p.get("ebeam_edge_clearance",10)),False)
                if slot==0:
                    field_center=tuple(np.asarray(start)+rot(group_local,orientation));add_ebeam_field_outline(top,field_center,orientation,field);group_values=devices[group*per_field:min((group+1)*per_field,len(devices))];add_ebeam_parameter_text(top,f"P={compact_parameter_range([v[0] for v in group_values])} F={compact_parameter_range([v[1] for v in group_values],2)}",field_center,orientation,field,float(p.get("parameter_text_height",10)))
        return
    if kind == "Grating angle-taper test block":
        active=set(p.get("sweep_parameters",["alpha_t","taper_L"]));nominal_angle=float(p.get("nominal_angle_deg",25));nominal_taper=float(p.get("nominal_taper_length",22));angles=inclusive_sweep(p.get("angle_start_deg",22),p.get("angle_stop_deg",28),p.get("angle_step_deg",1)) if "alpha_t" in active else [nominal_angle];tapers=inclusive_sweep(p.get("taper_length_start",20),p.get("taper_length_stop",24),p.get("taper_length_step",1)) if "taper_L" in active else [nominal_taper];devices=[(angle,taper) for taper in tapers for angle in angles];max_devices=max(1,int(p.get("max_devices",36)))
        if len(devices)>max_devices:
            devices=sorted(devices,key=lambda item:((item[0]-nominal_angle)/max(abs(angles[-1]-angles[0]),1e-12))**2+((item[1]-nominal_taper)/max(abs(tapers[-1]-tapers[0]),1e-12))**2)[:max_devices];devices.sort(key=lambda item:(item[1],item[0]))
        packing=float(p.get("packing_pitch",100));field=float(p.get("write_field_size",520));per_field=max(1,int(p.get("devices_per_field",4)));dx=float(p.get("device_x_spacing",600));groups=math.ceil(len(devices)/per_field);total=float(p.get("grating_end_to_end_distance",p.get("feedline_end_to_end",365)));sb=float(p.get("s_bend_length",80));straight=50.;lc=total-2*sb-2*straight
        if lc<=0:raise ValueError("feedline_end_to_end must exceed 2*s_bend_length + 100 µm.")
        for index,(angle,taper) in enumerate(devices):
            group=index//per_field;slot=index%per_field;group_size=min(per_field,len(devices)-group*per_field);group_local=((group-(groups-1)/2)*dx,0.0);local=(group_local[0],(slot-(group_size-1)/2)*packing);point=tuple(np.asarray(start)+rot(local,orientation));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["Feedline"]);sub.update({"pitch":float(p.get("pitch",.75)),"fill_factor":float(p.get("fill_factor",.57)),"N":int(p.get("N",30)),"alpha_t":angle,"taper_L":taper,"gc_wg_length":float(p.get("gc_wg_length",20)),"wg_width":float(p.get("wg_width",1.2)),"input_straight_length":straight,"output_straight_length":straight,"s_bend_length":sb,"offset":float(p.get("endpoint_offset",100))/2,"Lc":lc,"input_s_bend_direction":"up","output_s_bend_direction":"up"});add_component_centered_in_field(top,"Feedline",sub,point,orientation,mirrored,uid*1000+index+1,field,float(p.get("ebeam_edge_clearance",10)),False)
            if slot==0:
                field_center=tuple(np.asarray(start)+rot(group_local,orientation));add_ebeam_field_outline(top,field_center,orientation,field);group_values=devices[group*per_field:min((group+1)*per_field,len(devices))];add_ebeam_parameter_text(top,f"A={compact_parameter_range([v[0] for v in group_values],1)} TL={compact_parameter_range([v[1] for v in group_values],1)}",field_center,orientation,field,float(p.get("parameter_text_height",10)))
        return
    if kind == "MMI + Reference test block":
        active=set(p.get("sweep_parameters",["mmi_length","taper_width"]));nominal=float(p.get("nominal_taper_width",2.7));lengths=inclusive_sweep(p.get("mmi_length_start",26),p.get("mmi_length_stop",33),p.get("mmi_length_step",1)) if "mmi_length" in active else [29.0];widths=inclusive_sweep(p.get("taper_width_start",2.5),p.get("taper_width_stop",3.1),p.get("taper_width_step",.1)) if "taper_width" in active else [nominal];dx=float(p.get("device_x_spacing",600));dy=float(p.get("device_y_spacing",600))
        if not any(abs(v-nominal)<1e-9 for v in widths):widths.append(nominal);widths.sort()
        for row,taper_width in enumerate(widths):
            for col,mmi_length in enumerate(lengths):
                local=((col-(len(lengths)-1)/2)*dx,(row-(len(widths)-1)/2)*dy);point=tuple(np.asarray(start)+rot(local,orientation));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["MMI + Reference"]);sub.update({"mmi_length":mmi_length,"taper_width":taper_width,"mmi_width":float(p.get("mmi_width",7)),"wg_width":float(p.get("wg_width",1.2)),"input_taper_length":float(p.get("input_taper_length",10)),"output_taper_length":float(p.get("output_taper_length",10)),"input_length":float(p.get("input_length",6)),"output_length":float(p.get("output_length",6)),"port_sep":float(p.get("port_sep",3.25)),"reference_dy":float(p.get("reference_dy",150)),"reference_branch":p.get("reference_branch","upper"),"gc_pitch":float(p.get("pitch",.75)),"gc_fill_factor":float(p.get("fill_factor",.57)),"gc_N":int(p.get("N",30)),"gc_alpha_t":float(p.get("alpha_t",25)),"gc_taper_L":float(p.get("taper_L",22)),"gc_wg_length":float(p.get("gc_wg_length",20)),"add_grating_couplers":True});add_component_centered_in_field(top,"MMI + Reference",sub,point,orientation,mirrored,uid*1000+row*len(lengths)+col+1,float(p.get("ebeam_field_size",520)),float(p.get("ebeam_edge_clearance",10)))
                add_ebeam_parameter_text(top,f"L={mmi_length:g} W={taper_width:g}",point,orientation,float(p.get("ebeam_field_size",520)),float(p.get("parameter_text_height",10)))
        return
    if kind == "MMI split-combine test block":
        active=set(p.get("sweep_parameters",["taper_length","taper_width"]));nominal=float(p.get("nominal_taper_width",2.7));taper_lengths=inclusive_sweep(p.get("taper_length_start",8),p.get("taper_length_stop",12),p.get("taper_length_step",1)) if "taper_length" in active else [10.0];widths=inclusive_sweep(p.get("taper_width_start",2.5),p.get("taper_width_stop",3.1),p.get("taper_width_step",.1)) if "taper_width" in active else [nominal];dx=float(p.get("device_x_spacing",950));dy=float(p.get("device_y_spacing",950));field=float(p.get("ebeam_field_size",850));clearance=float(p.get("ebeam_edge_clearance",10))
        if not any(abs(v-nominal)<1e-9 for v in widths):widths.append(nominal);widths.sort()
        for row,taper_width in enumerate(widths):
            for col,taper_length in enumerate(taper_lengths):
                local=((col-(len(taper_lengths)-1)/2)*dx,(row-(len(widths)-1)/2)*dy);point=tuple(np.asarray(start)+rot(local,orientation));sub=safe_json_copy(DEFAULT_COMPONENT_VALUES["MMI split-combine cascade"]);sub.update({key:safe_json_copy(value) for key,value in p.items() if key not in {"taper_length_start","taper_length_stop","taper_length_step","taper_width_start","taper_width_stop","taper_width_step","nominal_taper_width","device_x_spacing","device_y_spacing","ebeam_field_size","ebeam_edge_clearance","include_ebeam_fields","parameter_text_height"}});sub.update({"input_taper_length":taper_length,"output_taper_length":taper_length,"taper_width":taper_width})
                add_component_centered_in_field(top,"MMI split-combine cascade",sub,point,orientation,mirrored,uid*1000+row*len(taper_lengths)+col+1,field,clearance,bool(p.get("include_ebeam_fields",True)))
                add_ebeam_parameter_text(top,f"SC TL={taper_length:g} W={taper_width:g}",point,orientation,field,float(p.get("parameter_text_height",10)))
        return
    if kind == "Ring + two feedlines":
        radius=float(p["ring_radius"]); ring_width=float(p["ring_width"]); feed_width=float(p["feedline_width"]); gap=float(p["coupling_gap"]); sb=float(p["s_bend_length"])
        if radius<=ring_width/2 or feed_width<=0 or gap<0 or sb<=0: raise ValueError("Invalid ring + two feedlines dimensions.")
        q=ring_two_feedline_landmarks(p)
        add_ring_to_gds(top,(0,0),radius,ring_width,int(p.get("points",256)),start,orientation,layer,datatype)
        # Each symmetric bus transitions between the ring-coupling separation
        # and the independently specified grating-coupler separation.
        for prefix in ("upper", "lower"):
            left_gc=q[f"{prefix}_left_gc"]; left_bus=q[f"{prefix}_left_bus"]; right_bus=q[f"{prefix}_right_bus"]; right_gc=q[f"{prefix}_right_gc"]
            left_global=tuple(np.asarray(start)+rot(left_gc,orientation))
            left_path,*_=backend.make_s_bend_gdstk(length=sb,offset=float(left_bus[1]-left_gc[1]),width=feed_width,start_center=left_global,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("tolerance",0.001)))
            top.add(left_path)
            add_rect(top,np.array([[left_bus[0],left_bus[1]-feed_width/2],[right_bus[0],right_bus[1]-feed_width/2],[right_bus[0],right_bus[1]+feed_width/2],[left_bus[0],left_bus[1]+feed_width/2]],float),start,orientation,layer,datatype)
            right_global=tuple(np.asarray(start)+rot(right_bus,orientation))
            right_path,*_=backend.make_s_bend_gdstk(length=sb,offset=float(right_gc[1]-right_bus[1]),width=feed_width,start_center=right_global,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("tolerance",0.001)))
            top.add(right_path)
        # Four focusing grating couplers, one at each externally separated bus endpoint.
        for label,key,dir_deg in (("UL","upper_left_gc",orientation+180),("UR","upper_right_gc",orientation),("LL","lower_left_gc",orientation+180),("LR","lower_right_gc",orientation)):
            global_point=tuple(np.asarray(start)+rot(q[key],orientation))
            add_parent_focusing_gc(
                top, p, global_point, dir_deg % 360.0, feed_width,
                PHOTONIC_LAYER, DEFAULT_DATATYPE,
                f"RING2_{component.get('uid','X')}_{label}",
            )
        return
    if kind == "Text / Number":
        text=str(p.get("text","")); height=float(p["height"])
        if height <= 0:
            raise ValueError("Text height must be positive.")
        polygons=list(gdstk.text(text,height,(0.0,0.0),layer=layer,datatype=datatype))
        if polygons:
            mins=[]; maxs=[]
            for poly in polygons:
                bb=poly.bounding_box()
                if bb is not None:
                    mins.append(np.asarray(bb[0],float)); maxs.append(np.asarray(bb[1],float))
            if mins:
                center_local=0.5*(np.min(np.vstack(mins),axis=0)+np.max(np.vstack(maxs),axis=0))
                for poly in polygons:
                    poly.translate(-float(center_local[0]),-float(center_local[1]))
                    top.add(_transform_polygon(poly,start,orientation))
        return
    if kind == "Edge coupler":
        tip=float(p["tip_width"]); width=float(p["wg_width"]); taper=float(p["taper_length"]); output=float(p["wg_straight_length"])
        if tip <= 0 or width <= 0 or taper <= 0 or output < 0:
            raise ValueError("Edge coupler widths and taper_length must be positive; wg_straight_length cannot be negative.")
        # One true trapezoidal inverse taper followed by an optional straight.
        add_rect(top,np.array([[0,-tip/2],[taper,-width/2],[taper,width/2],[0,tip/2]],float),start,orientation,layer,datatype)
        if output>0: add_rect(top,np.array([[taper,-width/2],[taper+output,-width/2],[taper+output,width/2],[taper,width/2]],float),start,orientation,layer,datatype)
        return
    if kind == "Loopback mirror":
        q=loopback_landmarks(p,mirrored); width=float(p["width"]); lc=float(p["Lc"]); sb=float(p["s_bend_length"]); radius=float(p["arc_radius"]); offset=float(q["offset"][0])
        if width<=0 or lc<0 or sb<=0 or radius<=0 or float(p["gap"])<0:
            raise ValueError("Loopback requires positive width, S-bend length and arc radius; Lc and gap cannot be negative.")
        # Two parallel Lc sections.
        for y0 in (q["left_upper"][1], q["left_lower"][1]):
            add_rect(top,np.array([[0,y0-width/2],[lc,y0-width/2],[lc,y0+width/2],[0,y0+width/2]],float),start,orientation,layer,datatype)
        # Smooth symmetric S-bends to the arc endpoints.
        upper_start=tuple(np.asarray(start)+rot(q["upper_straight_end"],orientation)); lower_start=tuple(np.asarray(start)+rot(q["lower_straight_end"],orientation))
        upper_offset=(q["upper_s_bend_end"][1]-q["upper_straight_end"][1]); lower_offset=(q["lower_s_bend_end"][1]-q["lower_straight_end"][1])
        upper_path,*_=backend.make_s_bend_gdstk(length=sb,offset=upper_offset,width=width,start_center=upper_start,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("tolerance",0.001)))
        lower_path,*_=backend.make_s_bend_gdstk(length=sb,offset=lower_offset,width=width,start_center=lower_start,orientation_deg=orientation,layer=layer,datatype=datatype,tolerance=float(p.get("tolerance",0.001)))
        top.add(upper_path,lower_path)
        # Right-facing semicircular arc connects top to bottom continuously.
        arc=gdstk.ellipse(center=q["arc_center"],radius=radius+width/2,inner_radius=radius-width/2,initial_angle=-math.pi/2,final_angle=math.pi/2,layer=layer,datatype=datatype,tolerance=float(p.get("tolerance",0.001)))
        top.add(_transform_polygon(arc,start,orientation))
        return
    if kind in {"Feedline", "Ring + feedline", "Racetrack + feedline"}:
        points = add_feedline_to_gds(component, top)
        if kind == "Ring + feedline":
            side = 1.0 if str(p.get("resonator_side", "upper")).lower() == "upper" else -1.0
            side *= ms
            bus_y = points["first_s_bend_end"][1]
            radius = float(p["ring_radius"]); ring_width = float(p["ring_width"])
            y = bus_y + side * (float(p["wg_width"])/2.0 + float(p["coupling_gap"]) + ring_width/2.0 + radius)
            for x in resonator_x_positions(p, points["first_s_bend_end"][0], points["lc_end"][0]):
                add_ring_to_gds(top, (x, y), radius, ring_width, 256, start, orientation, int(p.get("resonator_layer",layer)), int(p.get("resonator_datatype",datatype)))
        elif kind == "Racetrack + feedline":
            side = 1.0 if str(p.get("resonator_side", "upper")).lower() == "upper" else -1.0
            side *= ms
            bus_y = points["first_s_bend_end"][1]
            radius = float(p["racetrack_radius"]); race_width = float(p["racetrack_width"])
            y = bus_y + side * (float(p["wg_width"])/2.0 + float(p["coupling_gap"]) + race_width/2.0 + radius)
            for x in resonator_x_positions(p, points["first_s_bend_end"][0], points["lc_end"][0]):
                add_racetrack_to_gds(top, (x, y), radius, float(p["racetrack_coupling_length"]), race_width, 128, start, orientation, int(p.get("resonator_layer",layer)), int(p.get("resonator_datatype",datatype)))
        return
    if kind == "CPW":
        length=float(p["length"]); ws=float(p["signal_width"]); gap=float(p["gap"]); wg=float(p["ground_width"])
        add_rect(top,np.array([[0,-ws/2],[length,-ws/2],[length,ws/2],[0,ws/2]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,ws/2+gap],[length,ws/2+gap],[length,ws/2+gap+wg],[0,ws/2+gap+wg]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,-ws/2-gap-wg],[length,-ws/2-gap-wg],[length,-ws/2-gap],[0,-ws/2-gap]],float),start,orientation,layer,datatype); return
    if kind == "CPW open":
        length=float(p["length"]); ws=float(p["signal_width"]); gap=float(p["gap"]); wg=float(p["ground_width"]); recess=float(p.get("signal_recess",20.0))
        if length <= 0 or ws <= 0 or gap < 0 or wg <= 0 or recess < 0 or recess >= length:
            raise ValueError("CPW open requires positive length, signal/ground widths, nonnegative gap, and 0 <= signal_recess < length.")
        signal_end=length-recess
        add_rect(top,np.array([[0,-ws/2],[signal_end,-ws/2],[signal_end,ws/2],[0,ws/2]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,ws/2+gap],[length,ws/2+gap],[length,ws/2+gap+wg],[0,ws/2+gap+wg]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,-ws/2-gap-wg],[length,-ws/2-gap-wg],[length,-ws/2-gap],[0,-ws/2-gap]],float),start,orientation,layer,datatype)
        return
    if kind == "CPW short":
        length=float(p["length"]); ws=float(p["signal_width"]); gap=float(p["gap"]); wg=float(p["ground_width"]); bridge=float(p.get("bridge_length",20.0))
        if length <= 0 or ws <= 0 or gap < 0 or wg <= 0 or bridge <= 0 or bridge > length:
            raise ValueError("CPW short requires positive dimensions, nonnegative gap, and 0 < bridge_length <= length.")
        add_rect(top,np.array([[0,-ws/2],[length,-ws/2],[length,ws/2],[0,ws/2]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,ws/2+gap],[length,ws/2+gap],[length,ws/2+gap+wg],[0,ws/2+gap+wg]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[0,-ws/2-gap-wg],[length,-ws/2-gap-wg],[length,-ws/2-gap],[0,-ws/2-gap]],float),start,orientation,layer,datatype)
        bridge_poly=np.array([[length-bridge,-ws/2-gap-wg],[length,-ws/2-gap-wg],[length,ws/2+gap+wg],[length-bridge,ws/2+gap+wg]],float)
        add_rect(top,bridge_poly,start,orientation,layer,datatype)
        return
    if kind == "Tapered CPW":
        length=float(p["length"]); ws=float(p["signal_width"]); wg=float(p["ground_width"]); u,gaps=gap_profile(p); xs=length*u
        add_rect(top,np.array([[0,-ws/2],[length,-ws/2],[length,ws/2],[0,ws/2]],float),start,orientation,layer,datatype)
        upper_inner=np.column_stack((xs,ws/2+gaps)); upper_outer=np.column_stack((xs[::-1],(ws/2+gaps+wg)[::-1])); lower_outer=np.column_stack((xs,-ws/2-gaps-wg)); lower_inner=np.column_stack((xs[::-1],(-ws/2-gaps)[::-1]))
        for vertices in (np.vstack((upper_inner,upper_outer)),np.vstack((lower_outer,lower_inner))):
            polygon=gdstk.Polygon(transform_points(vertices,start,orientation),layer=layer,datatype=datatype)
            top.add(*polygon.fracture(max_points=4000,precision=0.001))
        return
    if kind == "Symmetric CPW taper":
        xs,gaps,length,_=symmetric_cpw_taper_profile(p); ws=float(p["signal_width"]); wg=float(p["ground_width"])
        if ws <= 0.0 or wg <= 0.0:
            raise ValueError("Symmetric CPW taper signal_width and ground_width must be positive.")
        add_rect(top,np.array([[0,-ws/2],[length,-ws/2],[length,ws/2],[0,ws/2]],float),start,orientation,layer,datatype)
        upper_inner=np.column_stack((xs,ws/2+gaps)); upper_outer=np.column_stack((xs[::-1],(ws/2+gaps+wg)[::-1])); lower_outer=np.column_stack((xs,-ws/2-gaps-wg)); lower_inner=np.column_stack((xs[::-1],(-ws/2-gaps)[::-1]))
        for vertices in (np.vstack((upper_inner,upper_outer)),np.vstack((lower_outer,lower_inner))):
            polygon=gdstk.Polygon(transform_points(vertices,start,orientation),layer=layer,datatype=datatype)
            top.add(*polygon.fracture(max_points=4000,precision=0.001))
        return
    if kind == "CPW bend":
        q = cpw_bend_landmarks(p, mirrored)
        center = q["curvature_center"]
        for inner_radius, outer_radius in q["radial_ranges"].values():
            points = cpw_annular_sector_points(
                center, inner_radius, outer_radius,
                q["start_angle_rad"], q["end_angle_rad"], int(p.get("points", 161)),
            )
            top.add(gdstk.Polygon(transform_points(points, start, orientation), layer=layer, datatype=datatype))
        return
    if kind == "Segmented electrode":
        q = segmented_electrode_landmarks(p)
        x0 = 0.0
        x_flat_left = q["end_flat_length"]
        xa = q["segment_start"]
        xb = q["segment_end"]
        x3 = q["total_length"]
        x_flat_right = x3 - q["end_flat_length"]
        x_taper_left_end = xa - q["inner_flat_length"]
        x_taper_right_start = xb + q["inner_flat_length"]

        def local_rect(x_left: float, x_right: float, y0: float, y1: float, lyr: int, dt: int) -> None:
            if x_right <= x_left or y1 <= y0:
                return
            add_rect(top, np.array([[x_left,y0],[x_right,y0],[x_right,y1],[x_left,y1]], float), start, orientation, lyr, dt)

        # Plain 50-µm CPW flats at both external ends.
        for left, right in ((x0, x_flat_left), (x_flat_right, x3)):
            local_rect(left,right,q["plain_signal_lower"],q["plain_signal_upper"],layer,datatype)
            local_rect(left,right,q["plain_upper_ground_inner"],q["plain_upper_ground_outer"],layer,datatype)
            local_rect(left,right,q["plain_lower_ground_outer"],q["plain_lower_ground_inner"],layer,datatype)

        # Symmetric linear tapers from the 14.5-µm external gap to a plain CPW
        # whose gap equals the requested finger-tip separation.
        def tapered_strip(xl, xr, left_low, left_high, right_low, right_high):
            if xr <= xl:return
            points=np.array([[xl,left_low],[xr,right_low],[xr,right_high],[xl,left_high]],float)
            top.add(gdstk.Polygon(transform_points(points,start,orientation),layer=layer,datatype=datatype))
        for xl,xr,reverse in ((x_flat_left,x_taper_left_end,False),(x_taper_right_start,x_flat_right,True)):
            a,b=(q["plain_signal_lower"],q["inner_signal_lower"]);c,d=(q["plain_signal_upper"],q["inner_signal_upper"])
            if reverse:a,b=b,a;c,d=d,c
            tapered_strip(xl,xr,a,c,b,d)
            a,b=(q["plain_upper_ground_inner"],q["inner_upper_ground_inner"]);c,d=(q["plain_upper_ground_outer"],q["inner_upper_ground_outer"])
            if reverse:a,b=b,a;c,d=d,c
            tapered_strip(xl,xr,a,c,b,d)
            a,b=(q["plain_lower_ground_outer"],q["inner_lower_ground_outer"]);c,d=(q["plain_lower_ground_inner"],q["inner_lower_ground_inner"])
            if reverse:a,b=b,a;c,d=d,c
            tapered_strip(xl,xr,a,c,b,d)

        # Required 3-µm-gap flat CPW extension immediately before and after
        # the T-finger array.
        for left,right in ((x_taper_left_end,xa),(xb,x_taper_right_start)):
            local_rect(left,right,q["inner_signal_lower"],q["inner_signal_upper"],layer,datatype)
            local_rect(left,right,q["inner_upper_ground_inner"],q["inner_upper_ground_outer"],layer,datatype)
            local_rect(left,right,q["inner_lower_ground_outer"],q["inner_lower_ground_inner"],layer,datatype)

        # Wide-gap segmented CPW section.
        local_rect(xa,xb,q["signal_lower"],q["signal_upper"],layer,datatype)
        local_rect(xa,xb,q["upper_ground_inner"],q["upper_ground_outer"],layer,datatype)
        local_rect(xa,xb,q["lower_ground_outer"],q["lower_ground_inner"],layer,datatype)

        # Four T electrodes per row, one from each metal edge into both gaps.
        # Row centers remain exactly one full period apart.  The first and last
        # rows are clipped at the patterned boundaries, creating contained
        # half-fingers with no metal protrusion into the plain CPW leads.
        s=q["s"]; r=q["r"]; h=q["h"]; t=q["t"]
        su=q["signal_upper"]; sl=q["signal_lower"]
        ugi=q["upper_ground_inner"]; lgi=q["lower_ground_inner"]
        for xc in q["finger_centers"]:
            neck_left = max(xa, xc - t / 2.0)
            neck_right = min(xb, xc + t / 2.0)
            bar_left = max(xa, xc - r / 2.0)
            bar_right = min(xb, xc + r / 2.0)

            # Upper gap: signal-side T and ground-side T.
            local_rect(neck_left,neck_right,su,su+h,layer,datatype)
            local_rect(bar_left,bar_right,su+h,su+h+s,layer,datatype)
            local_rect(neck_left,neck_right,ugi-h,ugi,layer,datatype)
            local_rect(bar_left,bar_right,ugi-h-s,ugi-h,layer,datatype)
            # Lower gap: signal-side T and ground-side T.
            local_rect(neck_left,neck_right,sl-h,sl,layer,datatype)
            local_rect(bar_left,bar_right,sl-h-s,sl-h,layer,datatype)
            local_rect(neck_left,neck_right,lgi,lgi+h,layer,datatype)
            local_rect(bar_left,bar_right,lgi+h,lgi+h+s,layer,datatype)

        if bool(p.get("include_oxide_masks", False)):
            ox_layer=int(p.get("oxide_layer",3)); ox_dt=int(p.get("oxide_datatype",0))
            for left, right in ((x0, xa), (xb, x3)):
                local_rect(left,right,q["plain_signal_upper"],q["plain_upper_ground_inner"],ox_layer,ox_dt)
                local_rect(left,right,q["plain_lower_ground_inner"],q["plain_signal_lower"],ox_layer,ox_dt)
            local_rect(xa,xb,q["signal_upper"],q["upper_ground_inner"],ox_layer,ox_dt)
            local_rect(xa,xb,q["lower_ground_inner"],q["signal_lower"],ox_layer,ox_dt)
        return
    if kind == "E-beam multipass":
        # Export each active EPBG write field as one simple filled A x A
        # rectangle on layer 6 (Ebeam).  The GUI may display outlines and order
        # numbers, but the GDS contains exactly one rectangle per active field.
        layout = multipass_field_layout(p)
        field_size = float(layout["field_size"])
        half = field_size / 2.0
        for field in layout["fields"]:
            raw_rect = field.get("rect")
            if isinstance(raw_rect, (list, tuple)) and len(raw_rect) == 4:
                x0, y0, x1, y1 = (float(value) for value in raw_rect)
            else:
                cx, cy = field["center"]
                x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
            add_rect(
                top,
                np.array(
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                    ],
                    dtype=float,
                ),
                start,
                orientation,
                EBEAM_LAYER,
                DEFAULT_DATATYPE,
            )
            if bool(p.get("show_order",True)):
                field_center=tuple(np.asarray(start)+rot(((x0+x1)/2,(y0+y1)/2),orientation));add_write_field_number(top,int(field.get("order",1)),field_center,orientation,max(x1-x0,y1-y0))
        return
    if kind == "Chip outline":
        width=float(p["width"]); height=float(p["height"]); line_width=float(p["line_width"])
        if width <= 0 or height <= 0 or line_width <= 0 or 2.0 * line_width >= min(width, height):
            raise ValueError("Chip outline requires positive width/height and line_width smaller than half the smallest side.")
        x0=-width/2.0; x1=width/2.0; y0=-height/2.0; y1=height/2.0
        add_rect(top,np.array([[x0,y0],[x1,y0],[x1,y0+line_width],[x0,y0+line_width]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[x0,y1-line_width],[x1,y1-line_width],[x1,y1],[x0,y1]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[x0,y0+line_width],[x0+line_width,y0+line_width],[x0+line_width,y1-line_width],[x0,y1-line_width]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[x1-line_width,y0+line_width],[x1,y0+line_width],[x1,y1-line_width],[x1-line_width,y1-line_width]],float),start,orientation,layer,datatype)
        if int(p.get("show_dimensions", 1)):
            dim_layer=int(p.get("dimension_layer", layer)); scale_label=max(0.01,float(p.get("dimension_text_scale",1.0)))
            bottom=tuple(np.asarray(start)+rot((0.0,y0-float(p.get("dimension_offset",150.0))),orientation))
            left=tuple(np.asarray(start)+rot((x0-float(p.get("dimension_offset",150.0)),0.0),orientation))
            top.add(gdstk.Label(f"W = {width:g} um",bottom,anchor="n",rotation=math.radians(orientation),magnification=scale_label,layer=dim_layer,texttype=datatype))
            top.add(gdstk.Label(f"H = {height:g} um",left,anchor="n",rotation=math.radians(orientation+90.0),magnification=scale_label,layer=dim_layer,texttype=datatype))
        return
    if kind == "Chip marker block":
        chip_w=float(p.get("chip_width",14000));chip_h=float(p.get("chip_height",12000));size=float(p.get("corner_square_size",50));clear=float(p.get("edge_clearance",0));half=size/2;inset=clear+half
        if min(chip_w,chip_h,size)<=0 or clear<0 or size+2*clear>min(chip_w,chip_h):raise ValueError("Invalid chip marker dimensions or edge clearance.")
        corners=(("BL",-chip_w/2+inset,-chip_h/2+inset),("BR",chip_w/2-inset,-chip_h/2+inset),("TL",-chip_w/2+inset,chip_h/2-inset),("TR",chip_w/2-inset,chip_h/2-inset))
        label_height=float(p.get("corner_label_height",14.0))
        if label_height<=0 or label_height>size*0.6:raise ValueError("Corner label height must be positive and no more than 60% of the marker size.")
        for label,cx,cy in corners:
            line=2.0;add_rect(top,np.array([[cx-half,cy-half],[cx+half,cy-half],[cx+half,cy-half+line],[cx-half,cy-half+line]],float),start,orientation,MARKER_LAYER,datatype);add_rect(top,np.array([[cx-half,cy+half-line],[cx+half,cy+half-line],[cx+half,cy+half],[cx-half,cy+half]],float),start,orientation,MARKER_LAYER,datatype);add_rect(top,np.array([[cx-half,cy-half],[cx-half+line,cy-half],[cx-half+line,cy+half],[cx-half,cy+half]],float),start,orientation,MARKER_LAYER,datatype);add_rect(top,np.array([[cx+half-line,cy-half],[cx+half,cy-half],[cx+half,cy+half],[cx+half-line,cy+half]],float),start,orientation,MARKER_LAYER,datatype)
            text_polys=list(gdstk.text(label,label_height,(0,0),layer=MARKER_LAYER,datatype=datatype))
            text_points=np.vstack([np.asarray(poly.points,dtype=float) for poly in text_polys]);text_center=(text_points.min(axis=0)+text_points.max(axis=0))/2.0
            for poly in text_polys:
                local=np.asarray(poly.points,dtype=float)-text_center+np.array([cx,cy],dtype=float)
                top.add(gdstk.Polygon(transform_points(local,start,orientation),layer=MARKER_LAYER,datatype=datatype))
        side_count=max(1,int(p.get("side_mark_count",3)));side_size=float(p.get("side_mark_size",100));side_inset=clear+side_size/2
        side_params={"size":side_size,"bar_width":float(p.get("side_mark_bar_width",8)),"square_size":float(p.get("side_mark_square_size",18)),"square_gap":float(p.get("side_mark_square_gap",5)),"layer":MARKER_LAYER,"datatype":datatype}
        if side_size<=0 or 2*side_inset>min(chip_w,chip_h):raise ValueError("Side alignment marks do not fit inside the chip dimensions.")
        side_points=[]
        for index in range(side_count):
            fraction=(index+1)/(side_count+1);x=-chip_w/2+fraction*chip_w;y=-chip_h/2+fraction*chip_h
            side_points.extend((((x,-chip_h/2+side_inset),(1,0)),((x,chip_h/2-side_inset),(1,0)),((-chip_w/2+side_inset,y),(0,1)),((chip_w/2-side_inset,y),(0,1))))
        solid_count=max(0,int(p.get("side_solid_square_count",2)));solid_size=float(p.get("side_solid_square_size",50));solid_interval=float(p.get("side_solid_square_interval",50))
        if solid_size<=0 or solid_interval<0:raise ValueError("Solid side-square size must be positive and its interval cannot be negative.")
        cluster_span=side_size if solid_count==0 else side_size/2+solid_interval+solid_count*solid_size+(solid_count-1)*solid_interval
        if side_count>1 and cluster_span>=min(chip_w,chip_h)/(side_count+1)-1e-9:raise ValueError("Marker groups overlap. Reduce marker count/size/spacing or increase the chip dimensions.")
        for index,(local_point,tangent) in enumerate(side_points):
            point=tuple(np.asarray(start)+rot(local_point,orientation));_add_component_geometry_to_cell({"uid":uid*1000+100+index,"kind":"Cross + squares mark","x":point[0],"y":point[1],"orientation_deg":orientation,"mirrored":False,"params":side_params},top)
            for square_index in range(solid_count):
                offset=side_size/2+solid_interval+solid_size/2+square_index*(solid_size+solid_interval);square_local=np.asarray(local_point,float)+np.asarray(tangent,float)*offset;square_point=tuple(np.asarray(start)+rot(square_local,orientation));square_params={"size":solid_size,"line_width":1.0,"filled":1,"layer":MARKER_LAYER,"datatype":datatype};_add_component_geometry_to_cell({"uid":uid*100000+index*100+square_index,"kind":"Square mark","x":square_point[0],"y":square_point[1],"orientation_deg":orientation,"mirrored":False,"params":square_params},top)
        if bool(p.get("include_center_vernier",True)):
            fl=float(p.get("vernier_finger_length",40));base=float(p.get("vernier_base_thickness",8));gap=float(p.get("vernier_row_gap",8));vernier_y=-chip_h/2+clear+gap/2+fl+base
            sub={"finger_count":int(p.get("vernier_finger_count",11)),"finger_width":float(p.get("vernier_finger_width",2)),"finger_length":fl,"pitch":float(p.get("vernier_pitch",10)),"pitch_delta":float(p.get("vernier_pitch_delta",.2)),"row_gap":gap,"base_thickness":base,"layer_a":MARKER_LAYER,"layer_b":MARKER_LAYER,"datatype":datatype};vernier_point=tuple(np.asarray(start)+rot((0,vernier_y),orientation));_add_component_geometry_to_cell({"uid":uid*1000+1,"kind":"Vernier mark","x":vernier_point[0],"y":vernier_point[1],"orientation_deg":orientation,"mirrored":False,"params":sub},top)
            count=max(2,int(sub["finger_count"]));span=max((count-1)*sub["pitch"]+sub["finger_width"],(count-1)*(sub["pitch"]+sub["pitch_delta"])+sub["finger_width"])+2*sub["pitch"];label_height=float(p.get("bottom_vernier_label_height",18));label_polys=list(gdstk.text("B",label_height,(0,0),layer=MARKER_LAYER,datatype=datatype));label_points=np.vstack([np.asarray(poly.points,dtype=float) for poly in label_polys]);label_center=(label_points.min(axis=0)+label_points.max(axis=0))/2;label_position=np.array([span/2+label_height,vernier_y],dtype=float)
            for poly in label_polys:
                local=np.asarray(poly.points,dtype=float)-label_center+label_position;top.add(gdstk.Polygon(transform_points(local,start,orientation),layer=MARKER_LAYER,datatype=datatype))
        return
    if kind == "Square mark":
        size=float(p["size"]); line_width=float(p["line_width"]); filled=int(p.get("filled",0))
        if size <= 0 or line_width <= 0:
            raise ValueError("Square mark size and line_width must be positive.")
        h=size/2.0
        if filled:
            add_rect(top,np.array([[-h,-h],[h,-h],[h,h],[-h,h]],float),start,orientation,layer,datatype)
        else:
            if 2.0*line_width >= size:
                raise ValueError("Square mark line_width must be smaller than half the size for an outline mark.")
            add_rect(top,np.array([[-h,-h],[h,-h],[h,-h+line_width],[-h,-h+line_width]],float),start,orientation,layer,datatype)
            add_rect(top,np.array([[-h,h-line_width],[h,h-line_width],[h,h],[-h,h]],float),start,orientation,layer,datatype)
            add_rect(top,np.array([[-h,-h+line_width],[-h+line_width,-h+line_width],[-h+line_width,h-line_width],[-h,h-line_width]],float),start,orientation,layer,datatype)
            add_rect(top,np.array([[h-line_width,-h+line_width],[h,-h+line_width],[h,h-line_width],[h-line_width,h-line_width]],float),start,orientation,layer,datatype)
        return
    if kind == "Cross mark":
        size=float(p["size"]); line_width=float(p["line_width"])
        if size <= 0 or line_width <= 0 or line_width > size:
            raise ValueError("Cross mark requires positive size and line_width no larger than size.")
        half=size/2.0; half_width=line_width/2.0
        add_rect(top,np.array([[-half,-half_width],[half,-half_width],[half,half_width],[-half,half_width]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[-half_width,-half], [half_width,-half], [half_width,half], [-half_width,half]],float),start,orientation,layer,datatype)
        return
    if kind == "Pointy cross mark":
        size=float(p["size"]); line_width=float(p["line_width"]); tip_length=float(p.get("tip_length",18.0))
        half=size/2.0; outer=half-line_width/2.0; inner=tip_length
        if size<=0 or line_width<=0 or line_width>=size/2.0 or tip_length<=0 or inner+line_width>=outer:
            raise ValueError("Pointy cross requires positive dimensions and enough room for four inward corner chevrons.")

        # Four isolated corner chevrons.  Each chevron lies on a diagonal and
        # points inward toward the open center, matching the reference mark.
        def add_pointed_arm(base, apex):
            base=np.asarray(base,dtype=float); apex=np.asarray(apex,dtype=float)
            vector=apex-base; length=float(np.linalg.norm(vector))
            if length<=0:
                raise ValueError("Invalid pointy-cross arm geometry.")
            direction=vector/length; normal=np.array([-direction[1],direction[0]],dtype=float)
            point_depth=min(1.5*line_width,0.45*length)
            body_end=apex-direction*point_depth
            polygon=np.array([
                base+normal*line_width/2.0,
                base-normal*line_width/2.0,
                body_end-normal*line_width/2.0,
                apex,
                body_end+normal*line_width/2.0,
            ],dtype=float)
            top.add(gdstk.Polygon(transform_points(polygon,start,orientation),layer=layer,datatype=datatype))

        for sx,sy in ((-1.0,1.0),(1.0,1.0),(-1.0,-1.0),(1.0,-1.0)):
            apex=np.array([sx*inner,sy*inner],dtype=float)
            horizontal_base=np.array([sx*(outer-tip_length),sy*outer],dtype=float)
            vertical_base=np.array([sx*outer,sy*(outer-tip_length)],dtype=float)
            add_pointed_arm(horizontal_base,apex)
            add_pointed_arm(vertical_base,apex)
        return
    if kind == "Cross + squares mark":
        size=float(p["size"]); bar_width=float(p["bar_width"]); square_size=float(p["square_size"]); square_gap=float(p["square_gap"])
        if size <= 0 or bar_width <= 0 or square_size <= 0 or square_gap < 0 or bar_width > size:
            raise ValueError("Cross + squares mark requires positive dimensions and nonnegative square_gap.")
        half=size/2.0; half_bar=bar_width/2.0
        square_center_offset=half_bar+square_gap+square_size/2.0
        if square_center_offset + square_size/2.0 > half:
            raise ValueError("The four squares do not fit inside the selected mark size. Reduce square_size/square_gap or increase size.")
        add_rect(top,np.array([[-half,-half_bar],[half,-half_bar],[half,half_bar],[-half,half_bar]],float),start,orientation,layer,datatype)
        add_rect(top,np.array([[-half_bar,-half],[half_bar,-half],[half_bar,half],[-half_bar,half]],float),start,orientation,layer,datatype)
        hs=square_size/2.0
        for sx,sy in ((-1.0,1.0),(1.0,1.0),(-1.0,-1.0),(1.0,-1.0)):
            cx=sx*square_center_offset; cy=sy*square_center_offset
            add_rect(top,np.array([[cx-hs,cy-hs],[cx+hs,cy-hs],[cx+hs,cy+hs],[cx-hs,cy+hs]],float),start,orientation,layer,datatype)
        return
    if kind == "Vernier mark":
        count=max(2,int(p["finger_count"])); fw=float(p["finger_width"]); fl=float(p["finger_length"]); pitch=float(p["pitch"]); pitch_delta=float(p["pitch_delta"]); gap=float(p["row_gap"]); base=float(p["base_thickness"]); layer_a=int(p["layer_a"]); layer_b=int(p["layer_b"])
        if fw <= 0 or fl <= 0 or pitch <= 0 or pitch + pitch_delta <= 0 or gap < 0 or base <= 0:
            raise ValueError("Vernier dimensions and pitches must be positive; row_gap cannot be negative.")
        span_a=(count-1)*pitch+fw; span_b=(count-1)*(pitch+pitch_delta)+fw; span=max(span_a,span_b)+2.0*pitch
        add_rect(top,np.array([[-span/2,-gap/2-fl-base],[span/2,-gap/2-fl-base],[span/2,-gap/2-fl],[-span/2,-gap/2-fl]],float),start,orientation,layer_a,datatype)
        add_rect(top,np.array([[-span/2,gap/2+fl],[span/2,gap/2+fl],[span/2,gap/2+fl+base],[-span/2,gap/2+fl+base]],float),start,orientation,layer_b,datatype)
        for i in range(count):
            xa=(i-(count-1)/2.0)*pitch
            xb=(i-(count-1)/2.0)*(pitch+pitch_delta)
            add_rect(top,np.array([[xa-fw/2,-gap/2-fl],[xa+fw/2,-gap/2-fl],[xa+fw/2,-gap/2],[xa-fw/2,-gap/2]],float),start,orientation,layer_a,datatype)
            add_rect(top,np.array([[xb-fw/2,gap/2],[xb+fw/2,gap/2],[xb+fw/2,gap/2+fl],[xb-fw/2,gap/2+fl]],float),start,orientation,layer_b,datatype)
        return
    raise ValueError(f"Unsupported component type: {kind}")


def add_component_to_gds(component: dict[str, Any], top: gdstk.Cell, library: gdstk.Library) -> None:
    """Add one fully resolved component directly to the single TOP cell."""
    direct_component = json.loads(json.dumps(component))
    _add_component_geometry_to_cell(direct_component, top)


def _canonicalize_component_layers(component: dict[str, Any]) -> None:
    kind = str(component.get("kind", ""))
    params = component.setdefault("params", {})
    if kind == "E-beam multipass":
        params["field_layer"] = EBEAM_LAYER
        params["field_datatype"] = DEFAULT_DATATYPE
    elif kind in RF_COMPONENT_KINDS:
        params["layer"] = RF_LAYER
        params["datatype"] = DEFAULT_DATATYPE
    elif kind in MARKER_COMPONENT_KINDS:
        params["layer"] = MARKER_LAYER
        params["datatype"] = DEFAULT_DATATYPE
        if kind == "Vernier mark":
            params["layer_a"] = MARKER_LAYER
            params["layer_b"] = MARKER_LAYER
    elif kind == "Grating coupler":
        params["layer"] = GC_LAYER
        params["datatype"] = DEFAULT_DATATYPE
        params["waveguide_layer"] = PHOTONIC_LAYER
        params["waveguide_datatype"] = DEFAULT_DATATYPE
    elif kind in GC_COMPOSITE_KINDS:
        params["layer"] = PHOTONIC_LAYER
        params["datatype"] = DEFAULT_DATATYPE
        params["gc_layer"] = GC_LAYER
        params["gc_datatype"] = DEFAULT_DATATYPE
    elif kind == "MZI + CPW module":
        params["layer"] = PHOTONIC_LAYER
        params["datatype"] = DEFAULT_DATATYPE
        params["rf_layer"] = RF_LAYER
        params["rf_datatype"] = DEFAULT_DATATYPE
    elif kind not in {
        "Chip outline", "E-beam multipass",
    }:
        params["layer"] = PHOTONIC_LAYER
        params["datatype"] = DEFAULT_DATATYPE


def resolve_and_build(components: list[dict[str, Any]]) -> gdstk.Library:
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    top = library.new_cell("TOP")
    by_uid = {int(c["uid"]): json.loads(json.dumps(c)) for c in components}
    for component in by_uid.values():
        _canonicalize_component_layers(component)

    global_ports: dict[int, dict[str, dict[str, Any]]] = {}
    remaining = set(by_uid)
    resolved_order: list[int] = []

    while remaining:
        progressed = False
        for uid in list(remaining):
            c = by_uid[uid]
            att = c.get("attachment")
            if att:
                target_uid = int(att.get("target_uid", -1))
                if target_uid in remaining:
                    continue
                target_kind = str(by_uid[target_uid].get("kind", "")) if target_uid in by_uid else ""
                target_name = normalize_port_name(target_kind, str(att.get("target_port", "")))
                own_name = normalize_port_name(str(c.get("kind", "")), str(att.get("own_port", "")))
                att["target_port"] = target_name
                att["own_port"] = own_name
                target = global_ports.get(target_uid, {}).get(target_name)
                if target is None:
                    raise ValueError(f"Component {uid} references a missing target connection point.")
                own = component_local_ports(c).get(own_name)
                if own is None or own["domain"] != target["domain"]:
                    raise ValueError(f"Component {uid} has an incompatible attachment.")
                solve_attachment(c, target)
            global_ports[uid] = component_global_ports(c)
            resolved_order.append(uid)
            remaining.remove(uid)
            progressed = True
        if not progressed:
            raise ValueError("The project contains a circular attachment.")

    # Every component and group is written directly into TOP.  No references
    # or child cells are created, so the exported GDS is always one flat cell.
    for uid in resolved_order:
        add_component_to_gds(by_uid[uid], top, library)

    try:
        top.set_property("LAYER_NAME_1", "WG")
        top.set_property("LAYER_NAME_2", "GC")
        top.set_property("LAYER_NAME_3", "Marker")
        top.set_property("LAYER_NAME_4", "RF")
        top.set_property("LAYER_NAME_5", "Probe")
        top.set_property("LAYER_NAME_6", "Ebeam")
    except Exception:
        pass

    if len(library.cells) != 1 or library.cells[0].name != "TOP":
        raise RuntimeError("Flat-export verification failed: expected exactly one TOP cell.")
    return library


def library_bbox_and_center(library: gdstk.Library) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]]:
    if not library.cells:
        raise ValueError("The layout has no cells.")
    bbox = library.cells[0].bounding_box()
    if bbox is None:
        raise ValueError("The layout has no geometry.")
    minimum = np.asarray(bbox[0], dtype=float)
    maximum = np.asarray(bbox[1], dtype=float)
    center = 0.5 * (minimum + maximum)
    return (
        (
            (float(minimum[0]), float(minimum[1])),
            (float(maximum[0]), float(maximum[1])),
        ),
        (float(center[0]), float(center[1])),
    )


def recenter_components_at_origin(components: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    copied = json.loads(json.dumps(components))
    initial_library = resolve_and_build(copied)
    initial_bbox, initial_center = library_bbox_and_center(initial_library)
    for component in copied:
        component["x"] = float(component["x"]) - initial_center[0]
        component["y"] = float(component["y"]) - initial_center[1]
    centered_library = resolve_and_build(copied)
    _, centered_center = library_bbox_and_center(centered_library)
    if abs(centered_center[0]) > 1e-6 or abs(centered_center[1]) > 1e-6:
        raise ValueError(f"Centering verification failed: final center is {centered_center} µm.")
    return copied, initial_center, initial_bbox


def rotate_components_layout(components: list[dict[str, Any]], angle_deg: float, pivot_mode: str = "center") -> tuple[list[dict[str, Any]], tuple[float, float]]:
    copied = json.loads(json.dumps(components))
    if not copied:
        raise ValueError("The layout is empty.")
    angle_deg = float(angle_deg)
    if not math.isfinite(angle_deg):
        raise ValueError("Global rotation angle must be finite.")
    if str(pivot_mode).lower() == "origin":
        pivot = (0.0, 0.0)
    else:
        library = resolve_and_build(copied)
        _, pivot = library_bbox_and_center(library)
    pivot_array = np.asarray(pivot, dtype=float)
    for component in copied:
        point = np.asarray([float(component["x"]), float(component["y"])], dtype=float)
        rotated = pivot_array + rot(point - pivot_array, angle_deg)
        component["x"] = float(rotated[0])
        component["y"] = float(rotated[1])
        component["orientation_deg"] = (float(component.get("orientation_deg", 0.0)) + angle_deg) % 360.0
    # Build once to verify all grouped geometry and attachment relationships remain valid.
    resolve_and_build(copied)
    return copied, (float(pivot[0]), float(pivot[1]))
