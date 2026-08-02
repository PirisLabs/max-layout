"""Component port positions and attachment solving."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .geometry.euler import euler_output_local, grating_route_bend_angle, mmi_gc_fanout_local_points, routed_gc_local_endpoint, three_euler_inward_gc_endpoint
from .geometry.landmarks import cpw_bend_landmarks, feedline_landmarks, loopback_landmarks, resonator_x_positions, ring_two_feedline_landmarks, segmented_electrode_landmarks
from .geometry.rf_taper import gap_profile, symmetric_cpw_taper_profile
from .geometry.shapes import mmi_total_length
from .geometry.transforms import rot

PORT_ALIASES: dict[str, dict[str, str]] = {
    "Straight": {"input": "left", "output": "right"},
    "Taper": {"input": "left", "output": "right"},
    "S-bend": {"input": "left", "output": "right"},
    "Euler bend": {"input": "start", "output": "end"},
    "Grating coupler": {"waveguide": "waveguide_point"},
    "1x2 MMI": {"input": "left_external", "upper": "upper_right", "lower": "lower_right"},
    "Cascaded MMI": {"input": "input", "output": "output"},
    "MMI split-combine cascade": {"input": "input_gc_waveguide", "output": "output_gc_waveguide", "reference_input": "reference_input_gc_waveguide", "reference_output": "reference_output_gc_waveguide"},
    "MMI + Reference": {"input": "left_external", "upper": "upper_right", "lower": "lower_right", "reference_input": "reference_left_external", "reference_output": "reference_right"},
    "MZI": {"input": "left_external", "output": "right_external", "splitter_upper_output": "splitter_upper_end", "splitter_lower_output": "splitter_lower_end", "combiner_upper_input": "combiner_upper_start", "combiner_lower_input": "combiner_lower_start"},
    "MZI vertical GC": {"input": "left_external", "output": "right_external", "left_gc": "left_gc_waveguide", "right_gc": "right_gc_waveguide"},
    "MZI + CPW module": {"optical_left": "left_external", "optical_right": "right_external", "rf_signal_left": "signal_left", "rf_signal_right": "signal_right", "upper_gap_left": "upper_gap_left", "lower_gap_left": "lower_gap_left", "upper_gap_right": "upper_gap_right", "lower_gap_right": "lower_gap_right"},
    "CPW": {"signal_input": "signal_left", "upper_gap_input": "upper_gap_left", "lower_gap_input": "lower_gap_left", "signal_output": "signal_right", "upper_gap_output": "upper_gap_right", "lower_gap_output": "lower_gap_right", "input": "signal_left", "output": "signal_right"},
    "CPW open": {"signal_connection": "signal_left", "upper_gap_connection": "upper_gap_left", "lower_gap_connection": "lower_gap_left", "input": "signal_left", "reference": "reference_plane"},
    "CPW short": {"signal_connection": "signal_left", "upper_gap_connection": "upper_gap_left", "lower_gap_connection": "lower_gap_left", "input": "signal_left", "reference": "reference_plane"},
    "Tapered CPW": {"signal_input": "signal_left", "upper_gap_input": "upper_gap_left", "lower_gap_input": "lower_gap_left", "signal_output": "signal_right", "upper_gap_output": "upper_gap_right", "lower_gap_output": "lower_gap_right", "input": "signal_left", "output": "signal_right"},
    "Symmetric CPW taper": {"signal_input": "signal_left", "upper_gap_input": "upper_gap_left", "lower_gap_input": "lower_gap_left", "signal_output": "signal_right", "upper_gap_output": "upper_gap_right", "lower_gap_output": "lower_gap_right", "input": "signal_left", "output": "signal_right"},
    "CPW bend": {"signal_start": "signal_start", "upper_gap_start": "upper_gap_start", "lower_gap_start": "lower_gap_start", "signal_end": "signal_end", "upper_gap_end": "upper_gap_end", "lower_gap_end": "lower_gap_end"},
    "Segmented electrode": {"input": "signal_left", "output": "signal_right", "signal_input": "signal_left", "signal_output": "signal_right", "upper_gap_input": "upper_gap_left", "lower_gap_input": "lower_gap_left", "upper_gap_output": "upper_gap_right", "lower_gap_output": "lower_gap_right"},
    "Edge coupler": {"facet": "tip", "waveguide": "waveguide_end"},
    "Loopback mirror": {"input": "left_upper", "output": "left_lower", "bend_start": "upper_straight_end", "bend_end": "lower_s_bend_end"},
    "Feedline": {"input_gc_waveguide": "left_gc_point", "output_gc_waveguide": "right_gc_point"},
    "Ring + feedline": {"input_gc_waveguide": "left_gc_point", "output_gc_waveguide": "right_gc_point"},
    "Racetrack + feedline": {"input_gc_waveguide": "left_gc_point", "output_gc_waveguide": "right_gc_point"},
    "Ring + two feedlines": {"upper_input": "upper_left_gc", "upper_output": "upper_right_gc", "lower_input": "lower_left_gc", "lower_output": "lower_right_gc"},
}


def normalize_port_name(kind: str, name: str) -> str:
    return PORT_ALIASES.get(kind, {}).get(name, name)


def component_local_ports(component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kind = str(component["kind"])
    p = dict(component.get("params", {}))
    mirrored = bool(component.get("mirrored", False))
    ms = -1.0 if mirrored else 1.0
    ports: dict[str, dict[str, Any]] = {}

    def add(name: str, center: tuple[float, float], outward: float, domain: str = "optical") -> None:
        ports[name] = {"center": center, "outward_orientation_deg": outward % 360.0, "domain": domain}

    if kind in {"Straight", "Taper"}:
        add("left", (0.0, 0.0), 180.0); add("right", (float(p["length"]), 0.0), 0.0)
    elif kind == "S-bend":
        add("left", (0.0, 0.0), 180.0); add("right", (float(p["length"]), ms * float(p["offset"])), 0.0)
    elif kind == "Euler bend":
        endpoint = euler_output_local(p, mirrored)
        add("start", (0.0, 0.0), 180.0); add("end", endpoint, ms * float(p["bend_angle_deg"]))
    elif kind == "Grating coupler":
        add("waveguide_point", (0.0, 0.0), 180.0)
    elif kind == "1x2 MMI":
        x1=float(p["input_length"]); x2=x1+float(p["input_taper_length"]); x3=x2+float(p["mmi_length"]); L=mmi_total_length(p); sep=ms*float(p["port_sep"])/2.0
        add("left_external", (0.0,0.0),180.0); add("left_straight_end",(x1,0.0),0.0); add("mmi_left_edge",(x2,0.0),0.0)
        add("mmi_upper_right_edge",(x3,sep),0.0); add("mmi_lower_right_edge",(x3,-sep),0.0); add("upper_right",(L,sep),0.0); add("lower_right",(L,-sep),0.0)
        if bool(p.get("add_grating_couplers", False)):
            q = mmi_gc_fanout_local_points(p, mirrored)
            add("upper_gc_fanout_end", q["upper_fanout_end"], 0.0)
            add("lower_gc_fanout_end", q["lower_fanout_end"], 0.0)
            input_bend = grating_route_bend_angle(p, mirrored, "gc_input_route", 180.0)
            upper_bend = grating_route_bend_angle(p, mirrored, "gc_upper_output_route", 0.0)
            lower_bend = grating_route_bend_angle(p, mirrored, "gc_lower_output_route", 0.0)
            input_gc, input_o = routed_gc_local_endpoint((0.0,0.0),180.0,input_bend,p)
            upper_gc, upper_o = routed_gc_local_endpoint(q["upper_fanout_end"],0.0,upper_bend,p)
            lower_gc, lower_o = routed_gc_local_endpoint(q["lower_fanout_end"],0.0,lower_bend,p)
            add("input_gc_waveguide",input_gc,input_o)
            add("upper_gc_waveguide",upper_gc,upper_o)
            add("lower_gc_waveguide",lower_gc,lower_o)
    elif kind == "MMI split-combine cascade":
        count=max(1,int(p.get("cascade_count",1)));d=max(0.0,float(p.get("interconnect_length",5.0)));m=mmi_total_length(p)
        total=count*(2*m+d)+max(0,count-1)*d
        add("input",(0.0,0.0),180.0);add("output",(total,0.0),0.0)
        if bool(p.get("add_input_grating_coupler",True)):add("input_gc_waveguide",(0.0,0.0),180.0)
        opposed=bool(p.get("add_opposed_output_s_bends",True));sb_length=float(p.get("output_s_bend_length",200.0));sb_offset=float(p.get("output_s_bend_offset",100.0))
        dut_delta=(sb_length,-ms*sb_offset) if opposed else (0.0,0.0)
        if bool(p.get("add_output_grating_coupler",True)):add("output_gc_waveguide",(total+dut_delta[0],dut_delta[1]),0.0)
        if bool(p.get("add_reference_waveguide",True)):
            ref_y=ms*float(p.get("reference_vertical_offset",150.0));ref_delta=(sb_length,ms*sb_offset) if opposed else (0.0,0.0);add("reference_input_gc_waveguide",(0.0,ref_y),180.0);add("reference_output_gc_waveguide",(total+ref_delta[0],ref_y+ref_delta[1]),0.0)
        for index in range(count):
            stage_x=index*(2*m+2*d)
            add(f"stage_{index+1}_split",(stage_x,0.0),180.0)
            add(f"stage_{index+1}_combine",(stage_x+2*m+d,0.0),0.0)
    elif kind == "Cascaded MMI":
        levels=max(1,int(p.get("N_levels",p.get("N_mmi",1))));length=mmi_total_length(p);sb_min=max(0.0,float(p.get("s_bend_length",300)));min_radius=max(0.0,float(p.get("minimum_s_bend_radius",200)));gc_spacing=max(0.0,float(p.get("output_gc_spacing",150)));sep=float(p["port_sep"])/2.0;nodes=[(0.0,0.0)];add("input",(0.0,0.0),180.0)
        for level in range(levels):
            outputs=[]
            for node_index,(x,y) in enumerate(nodes):
                upper=(x+length,y+ms*sep);lower=(x+length,y-ms*sep)
                if level==levels-1:outputs.extend((upper,lower))
                else:
                    spread=gc_spacing*(2**(levels-level-2));offset=max(0.0,spread-sep);sb=max(sb_min,math.sqrt(6.0*min_radius*offset));outputs.extend(((x+length+sb,y+ms*spread),(x+length+sb,y-ms*spread)))
            nodes=outputs
        # Every terminal branch has a mandatory final S-bend so the focusing
        # gratings are distributed on a safe, user-controlled pitch.
        output_sb=max(0.0,float(p.get("output_s_bend_length",300)));fanout=[]
        for index,point in enumerate(nodes):
            sign=1.0 if index%2==0 else -1.0;fanout.append((point[0]+output_sb,point[1]+ms*sign*(gc_spacing/2-sep)))
        fanout.sort(key=lambda point:point[1])
        for index,point in enumerate(fanout,1):add(f"leaf_{index}",point,0.0)
        add("output",fanout[0],0.0)
    elif kind == "MMI + Reference":
        x1 = float(p["input_length"])
        x2 = x1 + float(p["input_taper_length"])
        x3 = x2 + float(p["mmi_length"])
        total_length = mmi_total_length(p)
        port_y = ms * float(p["port_sep"]) / 2.0
        reference_y = ms * float(p.get("reference_dy", 250.0))
        selected_upper = str(p.get("reference_branch", "upper")).lower() != "lower"
        branch_sign = 1.0 if selected_upper else -1.0
        target_half = max(
            abs(float(p["port_sep"])) / 2.0,
            abs(float(p.get("gc_output_separation", p["port_sep"]))) / 2.0,
        )
        branch_fanout_y = ms * branch_sign * target_half
        fanout_length = float(p.get("gc_s_bend_length", 80.0))
        reference_fanout_end = (
            total_length + fanout_length,
            reference_y + branch_fanout_y,
        )

        add("left_external", (0.0, 0.0), 180.0)
        add("left_straight_end", (x1, 0.0), 0.0)
        add("mmi_left_edge", (x2, 0.0), 0.0)
        add("mmi_upper_right_edge", (x3, port_y), 0.0)
        add("mmi_lower_right_edge", (x3, -port_y), 0.0)
        add("upper_right", (total_length, port_y), 0.0)
        add("lower_right", (total_length, -port_y), 0.0)

        add("reference_left_external", (0.0, reference_y), 180.0)
        add("reference_straight_end", (total_length, reference_y), 0.0)
        add("reference_fanout_end", reference_fanout_end, 0.0)
        add("reference_right", reference_fanout_end, 0.0)

        if bool(p.get("add_grating_couplers", False)):
            fanout = mmi_gc_fanout_local_points(p, mirrored)
            add("upper_gc_fanout_end", fanout["upper_fanout_end"], 0.0)
            add("lower_gc_fanout_end", fanout["lower_fanout_end"], 0.0)

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

            input_gc, input_o = routed_gc_local_endpoint(
                (0.0, 0.0), 180.0, input_bend, p
            )
            upper_gc, upper_o = routed_gc_local_endpoint(
                fanout["upper_fanout_end"], 0.0, upper_bend, p
            )
            lower_gc, lower_o = routed_gc_local_endpoint(
                fanout["lower_fanout_end"], 0.0, lower_bend, p
            )
            reference_input_gc, reference_input_o = routed_gc_local_endpoint(
                (0.0, reference_y), 180.0, input_bend, p
            )
            reference_output_gc, reference_output_o = routed_gc_local_endpoint(
                reference_fanout_end, 0.0, branch_bend, p
            )

            add("input_gc_waveguide", input_gc, input_o)
            add("upper_gc_waveguide", upper_gc, upper_o)
            add("lower_gc_waveguide", lower_gc, lower_o)
            add(
                "reference_input_gc_waveguide",
                reference_input_gc,
                reference_input_o,
            )
            add(
                "reference_output_gc_waveguide",
                reference_output_gc,
                reference_output_o,
            )
    elif kind == "Long MZI test block":
        total=float(p.get("mzi_total_length",10000));straight=float(p.get("gc_straight_length",200));count=max(1,int(p.get("mzi_count",5)));spacing=float(p.get("vertical_spacing",1000))
        for index in range(count):
            y=(index-(count-1)/2)*spacing;add(f"left_{index+1}",(0,y),180);add(f"right_{index+1}",(total,y),0);add(f"left_gc_{index+1}",(-straight,y),180);add(f"right_gc_{index+1}",(total+straight,y),0)
        add("center",(total/2,0),0,"alignment")
    elif kind == "Chip marker block":
        width=float(p.get("chip_width",14000));height=float(p.get("chip_height",12000));size=float(p.get("corner_square_size",50));inset=float(p.get("edge_clearance",0))+size/2
        for name,x,y in (("lower_left",-width/2+inset,-height/2+inset),("lower_right",width/2-inset,-height/2+inset),("upper_left",-width/2+inset,height/2-inset),("upper_right",width/2-inset,height/2-inset)):add(name,(x,y),0,"alignment")
        count=max(1,int(p.get("side_mark_count",3)));side_inset=float(p.get("edge_clearance",0))+float(p.get("side_mark_size",100))/2
        for index in range(count):
            fraction=(index+1)/(count+1);x=-width/2+fraction*width;y=-height/2+fraction*height
            add(f"bottom_side_{index+1}",(x,-height/2+side_inset),0,"alignment");add(f"top_side_{index+1}",(x,height/2-side_inset),0,"alignment")
            add(f"left_side_{index+1}",(-width/2+side_inset,y),0,"alignment");add(f"right_side_{index+1}",(width/2-side_inset,y),0,"alignment")
        fl=float(p.get("vernier_finger_length",40));base=float(p.get("vernier_base_thickness",8));gap=float(p.get("vernier_row_gap",8));vernier_y=-height/2+float(p.get("edge_clearance",0))+gap/2+fl+base
        add("bottom_vernier",(0,vernier_y),0,"alignment")
    elif kind in {"MZI", "MZI vertical GC"}:
        M=mmi_total_length(p); x1=float(p["input_length"]); x2=x1+float(p["input_taper_length"]); x3=x2+float(p["mmi_length"]); sb=float(p["s_bend_length"]); arm=float(p["arm_length"]); ps=ms*float(p["port_sep"])/2.0; sep=ms*float(p["arm_separation"])/2.0
        xa=M+sb; xb=xa+arm; xc=xb+sb; L=2*M+2*sb+arm; comb_edge=L-x3; out_start=L-x1
        add("left_external",(0,0),180); add("left_straight_end",(x1,0),0); add("splitter_mmi_left_edge",(x2,0),0); add("splitter_upper_edge",(x3,ps),0); add("splitter_lower_edge",(x3,-ps),0)
        add("splitter_upper_end",(M,ps),0); add("splitter_lower_end",(M,-ps),0); add("upper_arm_left",(xa,sep),0); add("lower_arm_left",(xa,-sep),0); add("upper_arm_right",(xb,sep),0); add("lower_arm_right",(xb,-sep),0)
        add("center",((xa+xb)/2,0),0,"alignment"); add("optical_gap_center",((xa+xb)/2,0),0,"alignment"); add("combiner_upper_start",(xc,ps),0); add("combiner_lower_start",(xc,-ps),0); add("combiner_upper_edge",(comb_edge,ps),0); add("combiner_lower_edge",(comb_edge,-ps),0); add("right_straight_start",(out_start,0),0); add("right_external",(L,0),0)
        if bool(p.get("add_grating_couplers", False)):
            if kind=="MZI vertical GC" or bool(p.get("gc_three_euler_inward",False)):
                left_gc,left_o=three_euler_inward_gc_endpoint((0.0,0.0),180.0,p,True);right_gc,right_o=three_euler_inward_gc_endpoint((L,0.0),0.0,p,False)
            else:
                left_bend = grating_route_bend_angle(p, mirrored, "gc_input_route", 180.0);right_bend = grating_route_bend_angle(p, mirrored, "gc_output_route", 0.0);left_gc, left_o = routed_gc_local_endpoint((0.0,0.0),180.0,left_bend,p);right_gc, right_o = routed_gc_local_endpoint((L,0.0),0.0,right_bend,p)
            add("left_gc_waveguide",left_gc,left_o)
            add("right_gc_waveguide",right_gc,right_o)
    elif kind == "MZI + CPW module":
        mlen = mmi_total_length(p)
        sb = float(p["s_bend_length"]); arm = float(p["arm_length"])
        active = 2.0 * sb + arm
        tin = float(p["rf_input_taper_length"]); tout = float(p["rf_output_taper_length"])
        total = tin + active + tout
        ws = float(p["signal_width"]); ge = float(p["external_gap"]); gi = float(p["interaction_gap"])
        wg = float(p["ground_width"])
        mzi_start = tin - mlen
        optical_total = 2.0 * mlen + active
        add("left_external", (mzi_start, 0.0), 180.0)
        add("right_external", (mzi_start + optical_total, 0.0), 0.0)
        add("signal_left", (0.0, 0.0), 180.0, "rf")
        add("ws_center_left", (0.0, 0.0), 180.0, "rf")
        add("upper_gap_left", (0.0, ws/2.0 + ge/2.0), 180.0, "alignment")
        add("lower_gap_left", (0.0, -ws/2.0 - ge/2.0), 180.0, "alignment")
        add("upper_ground_left", (0.0, ws/2.0 + ge + wg/2.0), 180.0, "rf")
        add("lower_ground_left", (0.0, -ws/2.0 - ge - wg/2.0), 180.0, "rf")
        add("active_start", (tin, 0.0), 0.0, "alignment")
        add("center", (tin + active/2.0, 0.0), 0.0, "alignment")
        add("cpw_center", (tin + active/2.0, 0.0), 0.0, "alignment")
        add("upper_optical_arm_center", (tin + active/2.0, ms*float(p["arm_separation"])/2.0), 0.0, "alignment")
        add("lower_optical_arm_center", (tin + active/2.0, -ms*float(p["arm_separation"])/2.0), 0.0, "alignment")
        add("active_end", (tin + active, 0.0), 0.0, "alignment")
        add("signal_right", (total, 0.0), 0.0, "rf")
        add("ws_center_right", (total, 0.0), 0.0, "rf")
        add("upper_gap_right", (total, ws/2.0 + ge/2.0), 0.0, "alignment")
        add("lower_gap_right", (total, -ws/2.0 - ge/2.0), 0.0, "alignment")
        add("upper_ground_right", (total, ws/2.0 + ge + wg/2.0), 0.0, "rf")
        add("lower_ground_right", (total, -ws/2.0 - ge - wg/2.0), 0.0, "rf")
    elif kind in {"CPW", "Tapered CPW"}:
        L=float(p["length"]); ws=float(p["signal_width"]); wg=float(p["ground_width"])
        if kind=="CPW": gi=go=float(p["gap"])
        else:
            _, gv=gap_profile(p); gi=float(gv[0]); go=float(gv[-1])
        add("signal_left",(0,0),180,"rf"); add("upper_gap_left",(0,ws/2+gi/2),180,"alignment"); add("lower_gap_left",(0,-ws/2-gi/2),180,"alignment")
        add("upper_ground_left",(0,ws/2+gi+wg/2),180,"rf"); add("lower_ground_left",(0,-ws/2-gi-wg/2),180,"rf")
        add("center",(L/2.0,0.0),0,"alignment")
        add("signal_right",(L,0),0,"rf"); add("upper_gap_right",(L,ws/2+go/2),0,"alignment"); add("lower_gap_right",(L,-ws/2-go/2),0,"alignment")
        add("upper_ground_right",(L,ws/2+go+wg/2),0,"rf"); add("lower_ground_right",(L,-ws/2-go-wg/2),0,"rf")
    elif kind == "Symmetric CPW taper":
        _, _, L, q = symmetric_cpw_taper_profile(p)
        ws=float(p["signal_width"]); gi=float(p["initial_gap"]); gm=float(p["middle_gap"]); wg=float(p["ground_width"])
        add("signal_left",(0,0),180,"rf"); add("upper_gap_left",(0,ws/2+gi/2),180,"alignment"); add("lower_gap_left",(0,-ws/2-gi/2),180,"alignment")
        add("upper_ground_left",(0,ws/2+gi+wg/2),180,"rf"); add("lower_ground_left",(0,-ws/2-gi-wg/2),180,"rf")
        add("input_straight_end",(q["input_straight_end"],0.0),0,"alignment")
        add("upper_middle_start",(q["middle_start"],ws/2+gm/2),0,"alignment"); add("lower_middle_start",(q["middle_start"],-ws/2-gm/2),0,"alignment")
        add("center",(L/2.0,0.0),0,"alignment")
        add("upper_middle_end",(q["middle_end"],ws/2+gm/2),0,"alignment"); add("lower_middle_end",(q["middle_end"],-ws/2-gm/2),0,"alignment")
        add("output_straight_start",(q["output_straight_start"],0.0),0,"alignment")
        add("signal_right",(L,0),0,"rf"); add("upper_gap_right",(L,ws/2+gi/2),0,"alignment"); add("lower_gap_right",(L,-ws/2-gi/2),0,"alignment")
        add("upper_ground_right",(L,ws/2+gi+wg/2),0,"rf"); add("lower_ground_right",(L,-ws/2-gi-wg/2),0,"rf")
    elif kind in {"CPW open", "CPW short"}:
        L=float(p["length"]); ws=float(p["signal_width"]); gap=float(p["gap"]); wg=float(p["ground_width"])
        add("signal_left",(0.0,0.0),180.0,"rf")
        add("upper_gap_left",(0.0,ws/2.0+gap/2.0),180.0,"alignment")
        add("lower_gap_left",(0.0,-ws/2.0-gap/2.0),180.0,"alignment")
        add("upper_ground_left",(0.0,ws/2.0+gap+wg/2.0),180.0,"rf")
        add("lower_ground_left",(0.0,-ws/2.0-gap-wg/2.0),180.0,"rf")
        add("center",(L/2.0,0.0),0.0,"alignment")
        add("reference_plane",(L,0.0),0.0,"alignment")
        add("upper_gap_right",(L,ws/2.0+gap/2.0),0.0,"alignment")
        add("lower_gap_right",(L,-ws/2.0-gap/2.0),0.0,"alignment")
        add("upper_ground_right",(L,ws/2.0+gap+wg/2.0),0.0,"rf")
        add("lower_ground_right",(L,-ws/2.0-gap-wg/2.0),0.0,"rf")
        if kind == "CPW open":
            recess=max(0.0,min(float(p.get("signal_recess",20.0)),L))
            add("signal_tip",(L-recess,0.0),0.0,"rf")
        else:
            bridge=max(0.001,min(float(p.get("bridge_length",20.0)),L))
            add("short_center",(L-bridge/2.0,0.0),0.0,"alignment")
    elif kind == "CPW bend":
        q = cpw_bend_landmarks(p, mirrored)
        add("signal_start", q["signal_start"], 180.0, "rf")
        add("upper_gap_start", q["upper_gap_start"], 180.0, "alignment")
        add("lower_gap_start", q["lower_gap_start"], 180.0, "alignment")
        add("upper_ground_start", q["upper_ground_start"], 180.0, "rf")
        add("lower_ground_start", q["lower_ground_start"], 180.0, "rf")
        add("signal_end", q["signal_end"], q["angle_deg"], "rf")
        add("upper_gap_end", q["upper_gap_end"], q["angle_deg"], "alignment")
        add("lower_gap_end", q["lower_gap_end"], q["angle_deg"], "alignment")
        add("upper_ground_end", q["upper_ground_end"], q["angle_deg"], "rf")
        add("lower_ground_end", q["lower_ground_end"], q["angle_deg"], "rf")
        add("center", q["center"], q["angle_deg"] / 2.0, "alignment")
    elif kind == "Segmented electrode":
        q = segmented_electrode_landmarks(p)
        L = q["total_length"]
        ws=float(p["signal_width"]); gap=float(p.get("end_gap",15.0)); wg=float(p["ground_width"])
        add("signal_left", (0.0, 0.0), 180.0, "rf")
        add("upper_gap_left", (0.0, q["upper_gap_center"]), 180.0, "alignment")
        add("lower_gap_left", (0.0, q["lower_gap_center"]), 180.0, "alignment")
        add("upper_ground_left", (0.0, ws/2.0+gap+wg/2.0), 180.0, "rf")
        add("lower_ground_left", (0.0, -ws/2.0-gap-wg/2.0), 180.0, "rf")
        add("center", (L / 2.0, 0.0), 0.0, "alignment")
        add("signal_right", (L, 0.0), 0.0, "rf")
        add("upper_gap_right", (L, q["upper_gap_center"]), 0.0, "alignment")
        add("lower_gap_right", (L, q["lower_gap_center"]), 0.0, "alignment")
        add("upper_ground_right", (L, ws/2.0+gap+wg/2.0), 0.0, "rf")
        add("lower_ground_right", (L, -ws/2.0-gap-wg/2.0), 0.0, "rf")
    elif kind == "Edge coupler":
        taper=float(p["taper_length"]); total=taper+float(p["wg_straight_length"])
        add("tip",(0,0),180); add("taper_end",(taper,0),0); add("waveguide_end",(total,0),0)
    elif kind == "Loopback mirror":
        q=loopback_landmarks(p, mirrored)
        add("left_upper",q["left_upper"],180); add("left_lower",q["left_lower"],180); add("upper_straight_end",q["upper_straight_end"],0); add("lower_straight_end",q["lower_straight_end"],0); add("upper_s_bend_end",q["upper_s_bend_end"],0); add("lower_s_bend_end",q["lower_s_bend_end"],180); add("arc_center",q["arc_center"],0,"alignment")
    elif kind in {"Feedline", "Ring + feedline", "Racetrack + feedline"}:
        q=feedline_landmarks(p, mirrored)
        add("left_gc_point",q["input"],180); add("left_straight_end",q["input_straight_end"],0); add("first_s_bend_end",q["first_s_bend_end"],0); add("Lc_end",q["lc_end"],0); add("second_s_bend_end",q["second_s_bend_end"],0); add("right_gc_point",q["output"],0)
        if kind=="Ring + feedline":
            side=(1 if str(p.get("resonator_side","upper")).lower()=="upper" else -1)*ms; y=q["first_s_bend_end"][1]+side*(float(p["wg_width"])/2+float(p["coupling_gap"])+float(p["ring_width"])/2+float(p["ring_radius"]))
            for i,x in enumerate(resonator_x_positions(p,q["first_s_bend_end"][0],q["lc_end"][0]),1): add(f"ring_center_{i}",(float(x),float(y)),0,"alignment")
        elif kind=="Racetrack + feedline":
            side=(1 if str(p.get("resonator_side","upper")).lower()=="upper" else -1)*ms; y=q["first_s_bend_end"][1]+side*(float(p["wg_width"])/2+float(p["coupling_gap"])+float(p["racetrack_width"])/2+float(p["racetrack_radius"]))
            for i,x in enumerate(resonator_x_positions(p,q["first_s_bend_end"][0],q["lc_end"][0]),1): add(f"racetrack_center_{i}",(float(x),float(y)),0,"alignment")
    elif kind == "Ring + two feedlines":
        q=ring_two_feedline_landmarks(p)
        add("upper_left_gc",q["upper_left_gc"],180); add("upper_left_bus",q["upper_left_bus"],0); add("upper_right_bus",q["upper_right_bus"],0); add("upper_right_gc",q["upper_right_gc"],0)
        add("lower_left_gc",q["lower_left_gc"],180); add("lower_left_bus",q["lower_left_bus"],0); add("lower_right_bus",q["lower_right_bus"],0); add("lower_right_gc",q["lower_right_gc"],0)
        add("center",q["center"],0,"alignment"); add("upper_bus_center",q["upper_bus_center"],0,"alignment"); add("lower_bus_center",q["lower_bus_center"],0,"alignment")
    elif kind in {"Double-ring test block", "Grating test block", "Grating angle-taper test block", "MMI + Reference test block", "MMI split-combine test block", "Vertical-GC MZI test block", "Vertical-GC MZI + CPW test block", "Vertical-GC MZI + segmented electrode test block", "Straight-GC MZI + segmented RF bends test block", "Straight-GC MZI + CPW RF bends test block", "RF test block", "Photonic test block", "4-inch wafer outline"}:
        # Compound test arrays expose one stable parent alignment port.  The
        # child devices are intentionally not connectable at this level.
        add("center",(0.0,0.0),0.0,"alignment")
    elif kind in {"Photonic crystal","Photonic crystal slab"}:
        length=float(p.get("length",50.0));add("left",(-length/2,0.0),180.0);add("right",(length/2,0.0),0.0)
    elif kind.startswith("Photonic crystal ") or kind == "Boolean geometry":
        add("center",(0.0,0.0),0.0,"alignment")
    elif kind in {"Ring", "Elliptical ring", "Racetrack", "Text / Number", "Square mark", "Cross mark", "Pointy cross mark", "Cross + squares mark", "Vernier mark"}:
        add("center",(0,0),0,"alignment")
    elif kind in {"Chip outline", "E-beam multipass"}:
        add("center",(0.0,0.0),0.0,"alignment")
    else:
        raise ValueError(f"Unsupported component type: {kind}")

    # Every element exposes a neutral alignment center, even when its
    # original immutable component library only defines edge ports.
    if "center" not in ports:
        if ports:
            xs = [float(port["center"][0]) for port in ports.values()]
            ys = [float(port["center"][1]) for port in ports.values()]
            add("center",((min(xs)+max(xs))/2.0,(min(ys)+max(ys))/2.0),0.0,"alignment")
        else:
            add("center",(0.0,0.0),0.0,"alignment")
    return ports


def component_global_ports(component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    center = np.array([float(component["x"]), float(component["y"])], dtype=float)
    orientation = float(component.get("orientation_deg", 0.0))
    result: dict[str, dict[str, Any]] = {}
    for name, port in component_local_ports(component).items():
        result[name] = {
            "center": tuple(center + rot(port["center"], orientation)),
            "outward_orientation_deg": (orientation + float(port["outward_orientation_deg"])) % 360.0,
            "domain": port["domain"],
            "hidden": bool(port.get("hidden", False)),
        }
    return result


def solve_attachment(component: dict[str, Any], target_port: dict[str, Any]) -> None:
    attachment = component.get("attachment")
    if not attachment:
        return
    own_name = normalize_port_name(str(component.get("kind", "")), str(attachment.get("own_port", "left")))
    attachment["own_port"] = own_name
    own = component_local_ports(component).get(own_name)
    if own is None:
        raise ValueError(f"Missing own port {own_name!r} on component {component.get('uid')}")
    orientation = float(component.get("orientation_deg", 0.0))
    own_domain = str(own.get("domain", "alignment"))
    target_domain = str(target_port.get("domain", "alignment"))
    # Alignment points (including centers and CPW gap centers) translate the
    # component without unexpectedly rotating it.  True optical/RF ports still
    # rotate face-to-face before their centers are coincident.
    if own_domain != "alignment" and target_domain != "alignment":
        orientation = (float(target_port["outward_orientation_deg"]) + 180.0 - float(own["outward_orientation_deg"])) % 360.0
    start = np.asarray(target_port["center"], dtype=float) - rot(own["center"], orientation)
    component["x"], component["y"] = float(start[0]), float(start[1])
    component["orientation_deg"] = orientation
