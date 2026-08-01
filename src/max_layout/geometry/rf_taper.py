"""CPW impedance maths and Klopfenstein taper profiles."""

from __future__ import annotations

from typing import Any
from functools import lru_cache
import math

import numpy as np


def bessel_i1(value: float) -> float:
    term = value / 2.0
    total = term
    for k in range(1, 60):
        term *= (value * value / 4.0) / (k * (k + 1.0))
        total += term
        if abs(term) < 1e-14 * max(1.0, abs(total)):
            break
    return total


def klopfenstein_shape(u_values: np.ndarray, a: float) -> np.ndarray:
    a = max(1e-6, float(a))
    grid = np.linspace(-1.0, 1.0, 2001)
    z = a * np.sqrt(np.maximum(0.0, 1.0 - grid * grid))
    integrand = np.empty_like(z)
    small = np.abs(z) < 1e-10
    integrand[small] = 0.5
    integrand[~small] = np.array([bessel_i1(float(v)) / float(v) for v in z[~small]])
    cumulative = np.zeros_like(grid)
    cumulative[1:] = np.cumsum((integrand[:-1] + integrand[1:]) * 0.5 * np.diff(grid))
    cumulative /= cumulative[-1]
    return np.interp(2.0 * np.asarray(u_values) - 1.0, grid, cumulative)


def elliptic_k_parameter(m: float) -> float:
    """Complete elliptic integral K(m), evaluated by the AGM method."""
    m = min(1.0 - 1e-15, max(0.0, float(m)))
    a = 1.0
    b = math.sqrt(1.0 - m)
    for _ in range(80):
        next_a = 0.5 * (a + b)
        next_b = math.sqrt(a * b)
        if abs(next_a - next_b) <= 1e-15 * max(1.0, next_a):
            a = next_a
            break
        a, b = next_a, next_b
    return math.pi / (2.0 * a)


def cpw_impedance_factor(signal_width: float, gap: float) -> float:
    """Quasi-static CPW impedance factor K(k') / K(k).

    The common permittivity multiplier cancels when only the endpoint
    impedance ratio is needed for Klopfenstein synthesis.
    """
    signal_width = float(signal_width)
    gap = float(gap)
    if signal_width <= 0.0 or gap <= 0.0:
        raise ValueError("CPW signal width and taper gaps must be positive.")
    k = signal_width / (signal_width + 2.0 * gap)
    m = k * k
    return elliptic_k_parameter(1.0 - m) / elliptic_k_parameter(m)


def _elliptic_k_parameter_array(m_values: np.ndarray) -> np.ndarray:
    """Vectorized AGM evaluation matching :func:`elliptic_k_parameter`."""
    m = np.clip(np.asarray(m_values, dtype=float), 0.0, 1.0 - 1e-15)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    for _ in range(80):
        next_a = 0.5 * (a + b)
        next_b = np.sqrt(a * b)
        converged = np.abs(next_a - next_b) <= 1e-15 * np.maximum(1.0, next_a)
        a = next_a
        b = next_b
        if bool(np.all(converged)):
            break
    return np.pi / (2.0 * a)


def _cpw_impedance_factor_array(signal_width: float, gaps: np.ndarray) -> np.ndarray:
    gaps = np.asarray(gaps, dtype=float)
    k = float(signal_width) / (float(signal_width) + 2.0 * gaps)
    m = k * k
    return _elliptic_k_parameter_array(1.0 - m) / _elliptic_k_parameter_array(m)


def klopfenstein_auto_a(
    signal_width: float,
    start_gap: float,
    end_gap: float,
    target_s11_db: float,
) -> tuple[float, float, float]:
    """Return (A, Gamma0, Gamma_m) from geometry and desired return loss."""
    z0 = cpw_impedance_factor(signal_width, start_gap)
    z1 = cpw_impedance_factor(signal_width, end_gap)
    gamma0 = 0.5 * abs(math.log(z1 / z0))
    gamma_m = 10.0 ** (-abs(float(target_s11_db)) / 20.0)
    if gamma0 <= gamma_m or gamma0 <= 1e-15:
        return 0.0, gamma0, gamma_m
    return math.acosh(gamma0 / gamma_m), gamma0, gamma_m


def _invert_cpw_impedance_factor(
    signal_width: float,
    target_factor: float,
    gap_a: float,
    gap_b: float,
) -> float:
    lo = min(float(gap_a), float(gap_b))
    hi = max(float(gap_a), float(gap_b))
    f_lo = cpw_impedance_factor(signal_width, lo)
    increasing = cpw_impedance_factor(signal_width, hi) >= f_lo
    for _ in range(72):
        mid = 0.5 * (lo + hi)
        value = cpw_impedance_factor(signal_width, mid)
        if (value < target_factor) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _invert_cpw_impedance_factors(
    signal_width: float,
    target_factors: np.ndarray,
    gap_a: float,
    gap_b: float,
) -> np.ndarray:
    """Vectorized form of the exact 72-step scalar bisection."""
    targets = np.asarray(target_factors, dtype=float)
    lo_value = min(float(gap_a), float(gap_b))
    hi_value = max(float(gap_a), float(gap_b))
    lo = np.full(targets.shape, lo_value, dtype=float)
    hi = np.full(targets.shape, hi_value, dtype=float)
    f_lo = cpw_impedance_factor(signal_width, lo_value)
    increasing = cpw_impedance_factor(signal_width, hi_value) >= f_lo
    for _ in range(72):
        mid = 0.5 * (lo + hi)
        values = _cpw_impedance_factor_array(signal_width, mid)
        move_lo = (values < targets) == increasing
        lo = np.where(move_lo, mid, lo)
        hi = np.where(move_lo, hi, mid)
    return 0.5 * (lo + hi)


def _gap_transition_values(
    start_gap: float,
    end_gap: float,
    count: int,
    profile: str,
    signal_width: float,
    target_s11_db: float = 20.0,
    exponential_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    u, gaps = _cached_gap_transition_values(
        float(start_gap), float(end_gap), max(2, int(count)),
        str(profile or "linear").lower(), float(signal_width),
        float(target_s11_db), float(exponential_factor),
    )
    # Callers may modify endpoint arrays while assembling symmetric profiles.
    return u.copy(), gaps.copy()


@lru_cache(maxsize=256)
def _cached_gap_transition_values(
    start_gap: float,
    end_gap: float,
    count: int,
    profile: str,
    signal_width: float,
    target_s11_db: float,
    exponential_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache expensive full-resolution profiles shared by repeated tapers."""
    count = max(2, int(count))
    u = np.linspace(0.0, 1.0, count)
    start_gap = float(start_gap)
    end_gap = float(end_gap)
    profile = str(profile or "linear").lower()
    if start_gap <= 0.0 or end_gap <= 0.0:
        raise ValueError("CPW taper gaps must be positive.")
    if profile == "exponential":
        factor = float(exponential_factor)
        # A factor of zero is the neutral, smooth exponential interpolation.
        # Clamping it to a tiny positive power makes u**factor nearly one for
        # every u > 0, which creates a visible step immediately after the
        # taper input (and immediately before the symmetric output endpoint).
        exponent = 1.0 if abs(factor) <= 1e-12 else max(1e-6, factor)
        gaps = start_gap * (end_gap / start_gap) ** (u**exponent)
        gaps[0] = start_gap
        gaps[-1] = end_gap
        return u, gaps
    if profile == "klopfenstein":
        a, _, _ = klopfenstein_auto_a(signal_width, start_gap, end_gap, target_s11_db)
        shape = u if a <= 1e-12 else klopfenstein_shape(u, a)
        z0 = cpw_impedance_factor(signal_width, start_gap)
        z1 = cpw_impedance_factor(signal_width, end_gap)
        target_z = np.exp(math.log(z0) + (math.log(z1) - math.log(z0)) * shape)
        gaps = _invert_cpw_impedance_factors(signal_width, target_z, start_gap, end_gap)
        gaps[0] = start_gap
        gaps[-1] = end_gap
        return u, gaps
    return u, start_gap + (end_gap - start_gap) * u


def rf_taper_point_count(length_um: float) -> int:
    """One longitudinal point per 0.5 µm, including both taper endpoints."""
    length = float(length_um)
    if length < 0.0:
        raise ValueError("RF taper length cannot be negative.")
    return max(2, int(math.ceil(length / 0.5)) + 1)


def synchronize_rf_taper_points(component: dict[str, Any]) -> None:
    """Store the derived 0.5-µm spacing count in editable/project JSON fields."""
    kind = str(component.get("kind", ""))
    params = component.setdefault("params", {})
    if kind == "Tapered CPW":
        params["points"] = rf_taper_point_count(float(params.get("length", 0.0)))
    elif kind == "Symmetric CPW taper":
        params["points"] = rf_taper_point_count(float(params.get("taper_length", 0.0)))
    elif kind == "MZI + CPW module":
        params["points"] = max(
            rf_taper_point_count(float(params.get("rf_input_taper_length", 0.0))),
            rf_taper_point_count(float(params.get("rf_output_taper_length", 0.0))),
        )
    elif kind in {"Vertical-GC MZI + CPW test block", "Straight-GC MZI + CPW RF bends test block"}:
        params["cpw_points"] = rf_taper_point_count(float(params.get("cpw_taper_length", 0.0)))


def gap_profile(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    count = rf_taper_point_count(float(p["length"]))
    return _gap_transition_values(
        start_gap=float(p["initial_gap"]),
        end_gap=float(p["final_gap"]),
        count=count,
        profile=str(p.get("profile", "linear")),
        signal_width=float(p["signal_width"]),
        target_s11_db=float(p.get("target_s11_db", 20.0)),
        exponential_factor=float(p.get("exponential_factor", 1.0)),
    )


def symmetric_cpw_taper_profile(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    # A symmetric taper has one shared end-straight length, one shared taper
    # length, and one shared taper profile. Legacy project keys are accepted.
    end_straight_length = float(
        p.get("end_straight_length", p.get("input_straight_length", p.get("output_straight_length", 0.0)))
    )
    taper_length = float(
        p.get("taper_length", p.get("input_taper_length", p.get("output_taper_length", 0.0)))
    )
    l1 = l2 = end_straight_length
    lt1 = lt2 = taper_length
    lm = float(p["middle_straight_length"])
    lengths = (l1, lt1, lm, lt2, l2)
    if any(value < 0.0 for value in lengths):
        raise ValueError("Symmetric CPW taper section lengths cannot be negative.")
    total = sum(lengths)
    if total <= 0.0:
        raise ValueError("Symmetric CPW taper total length must be positive.")
    initial_gap = float(p["initial_gap"])
    middle_gap = float(p["middle_gap"])
    if initial_gap < 0.0 or middle_gap < 0.0:
        raise ValueError("Symmetric CPW taper gaps cannot be negative.")
    points = rf_taper_point_count(taper_length)
    profile = str(p.get("profile", p.get("input_profile", p.get("output_profile", "linear"))))
    signal_width = float(p["signal_width"])
    target_s11_db = float(p.get("target_s11_db", 20.0))
    exponential_factor = float(p.get("exponential_factor", 1.0))

    xs: list[float] = [0.0]
    gaps: list[float] = [initial_gap]
    x = l1
    xs.append(x); gaps.append(initial_gap)

    if lt1 > 0.0:
        u, input_values = _gap_transition_values(
            initial_gap, middle_gap, points, profile, signal_width,
            target_s11_db, exponential_factor,
        )
        xs.extend((x + lt1 * u[1:]).tolist()); gaps.extend(input_values[1:].tolist())
        x += lt1
    else:
        xs.append(x); gaps.append(middle_gap)

    xs.append(x + lm); gaps.append(middle_gap)
    x += lm

    if lt2 > 0.0:
        # Use the exact spatial reverse of the input taper so the complete
        # component remains geometrically symmetric for every profile.
        if "input_values" not in locals():
            u, input_values = _gap_transition_values(
                initial_gap, middle_gap, points, profile, signal_width,
                target_s11_db, exponential_factor,
            )
        output_values = input_values[::-1]
        xs.extend((x + lt2 * u[1:]).tolist()); gaps.extend(output_values[1:].tolist())
        x += lt2
    else:
        xs.append(x); gaps.append(initial_gap)

    xs.append(x + l2); gaps.append(initial_gap)
    boundaries = {
        "input_straight_end": l1,
        "middle_start": l1 + lt1,
        "middle_end": l1 + lt1 + lm,
        "output_straight_start": l1 + lt1 + lm + lt2,
        "total_length": total,
    }
    return np.asarray(xs, dtype=float), np.asarray(gaps, dtype=float), total, boundaries
