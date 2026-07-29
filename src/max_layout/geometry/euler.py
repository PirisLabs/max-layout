"""Euler bend endpoints and grating-coupler routing geometry."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from ..geometry.shapes import mmi_total_length
from ..geometry.transforms import rot


def euler_output_local(p: dict[str, Any], mirrored: bool) -> tuple[float, float]:
    radius = float(p["radius"])
    angle = math.radians((-1.0 if mirrored else 1.0) * float(p["bend_angle_deg"]))
    sign = 1.0 if angle > 0 else -1.0
    theta = abs(angle)
    frac = float(p["euler_fraction"])
    le = radius * frac * theta
    lc = radius * (1.0 - frac) * theta
    total = 2.0 * le + lc
    if total == 0:
        return (0.0, 0.0)

    def phi(s: float) -> float:
        if frac == 0.0:
            return sign * s / radius
        if s <= le:
            value = s * s / (2.0 * radius * le)
        elif s <= le + lc:
            value = frac * theta / 2.0 + (s - le) / radius
        else:
            q = s - (le + lc)
            value = theta * (1.0 - frac / 2.0) + q / radius - q * q / (2.0 * radius * le)
        return sign * value

    n = 1000
    ds = total / n
    x = y = 0.0
    for i in range(n):
        a = phi((i + 0.5) * ds)
        x += math.cos(a) * ds
        y += math.sin(a) * ds
    return (x, y)


def grating_route_bend_angle(
    p: dict[str, Any],
    mirrored: bool = False,
    parameter_name: str = "grating_route",
    outward_orientation_local_deg: float = 0.0,
) -> float:
    """Return a 90-degree Euler routing angle for one specific external port.

    ``up`` and ``down`` are interpreted in the component's local transverse
    direction.  A left-facing port therefore uses the opposite signed bend
    angle from a right-facing port to reach the same physical side.
    """
    fallback = p.get("grating_route", "straight")
    route = str(p.get(parameter_name, fallback)).strip().lower()
    transverse_sign = 1.0 if route == "up" else -1.0 if route == "down" else 0.0
    if transverse_sign == 0.0:
        return 0.0
    heading = float(outward_orientation_local_deg) % 360.0
    longitudinal_sign = 1.0 if math.cos(math.radians(heading)) >= 0.0 else -1.0
    mirror_sign = -1.0 if mirrored else 1.0
    return mirror_sign * longitudinal_sign * 90.0 * transverse_sign


def gc_euler_output_local(p: dict[str, Any], bend_angle_deg: float) -> tuple[float, float]:
    if abs(float(bend_angle_deg)) < 1e-12:
        return (0.0, 0.0)
    return euler_output_local(
        {
            "radius": float(p.get("gc_euler_radius", 100.0)),
            "bend_angle_deg": float(bend_angle_deg),
            "euler_fraction": float(p.get("gc_euler_fraction", 1.0)),
        },
        False,
    )


def routed_gc_local_endpoint(
    start_local: tuple[float, float],
    start_orientation_local_deg: float,
    bend_angle_deg: float,
    p: dict[str, Any],
) -> tuple[tuple[float, float], float]:
    if abs(float(bend_angle_deg)) < 1e-12:
        return (float(start_local[0]), float(start_local[1])), float(start_orientation_local_deg) % 360.0
    delta = rot(gc_euler_output_local(p, bend_angle_deg), start_orientation_local_deg)
    endpoint = np.asarray(start_local, dtype=float) + delta
    return (float(endpoint[0]), float(endpoint[1])), (float(start_orientation_local_deg) + float(bend_angle_deg)) % 360.0


def mmi_gc_fanout_local_points(
    p: dict[str, Any],
    mirrored: bool = False,
) -> dict[str, tuple[float, float]]:
    """Return the MMI output positions before and after GC fan-out S-bends."""
    ms = -1.0 if mirrored else 1.0
    length = mmi_total_length(p)
    port_half = abs(float(p["port_sep"])) / 2.0
    requested_half = abs(float(p.get("gc_output_separation", 133.0))) / 2.0
    target_half = max(port_half, requested_half)
    s_length = float(p.get("gc_s_bend_length", 80.0))
    if s_length <= 0:
        raise ValueError("gc_s_bend_length must be positive.")
    return {
        "upper_start": (length, ms * port_half),
        "lower_start": (length, -ms * port_half),
        "upper_fanout_end": (length + s_length, ms * target_half),
        "lower_fanout_end": (length + s_length, -ms * target_half),
    }


def three_euler_inward_gc_endpoint(start_center: tuple[float,float], start_orientation_deg: float, p: dict[str,Any], left_end: bool) -> tuple[tuple[float,float],float]:
    """Endpoint for an inward route whose mirrored last bend points away from the MZI."""
    side=1.0 if str(p.get("gc_vertical_side","up")).lower()=="up" else -1.0
    bend_angles=(-side*90.0,-side*90.0,side*90.0);point=np.asarray(start_center,float);heading=float(start_orientation_deg)%360;prebend=float(p.get("gc_prebend_straight",10.0))
    if prebend<0:raise ValueError("The straight section before the Euler bends cannot be negative.")
    point+=rot((prebend,0.0),heading)
    for bend_index,bend_angle in enumerate(bend_angles):
        point+=rot(gc_euler_output_local(p,bend_angle),heading);heading=(heading+bend_angle)%360
        if bend_index<2:
            fraction=float(p.get("gc_inward_run_fraction",.45));total=2*mmi_total_length(p)+2*float(p.get("s_bend_length",80))+float(p.get("arm_length",9718))
            if fraction<0 or fraction>=.5:raise ValueError("GC inward-run fraction must be at least 0 and below 0.5.")
            run=float(p.get("gc_vertical_run",100.0) if bend_index==0 else (fraction*total if fraction>0 else p.get("gc_inward_run",300.0)))
            if bend_index==1 and bool(p.get("gc_align_gc_to_mzi_center",True)):
                target=np.asarray(start_center,float)+rot((total/2,0),(float(start_orientation_deg)+180)%360);last_delta=rot(gc_euler_output_local(p,bend_angles[2]),heading);direction=rot((1,0),heading);run=float(np.dot(target-point-last_delta,direction))
            if run<0:raise ValueError("Vertical-GC route straight lengths cannot be negative.")
            point+=rot((run,0.0),heading)
    return (float(point[0]),float(point[1])),heading
