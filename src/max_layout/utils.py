"""Small parsing and sweep helpers."""

from __future__ import annotations

from typing import Any
import json
import re

import numpy as np


def numeric_list(value: Any) -> list[float]:
    if isinstance(value,(list,tuple,np.ndarray)):return [float(v) for v in value]
    return [float(token.strip()) for token in re.split(r"[,;\s]+",str(value).strip()) if token.strip()]


def inclusive_sweep(start: float, stop: float, step: float) -> list[float]:
    start=float(start);stop=float(stop);step=float(step)
    if step<=0 or stop<start:raise ValueError("Sweep stop must be >= start and step must be positive.")
    count=int(round((stop-start)/step));values=[start+i*step for i in range(count+1)]
    if not any(abs(v-stop)<1e-9 for v in values):values.append(stop)
    return values


def compact_parameter_range(values: list[float], decimals: int = 3) -> str:
    """Format one value or a compact min-max range without leading zeroes."""
    def one(value: float) -> str:
        result=f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        if result.startswith("0."):result=result[1:]
        if result.startswith("-0."):result="-."+result[3:]
        return result
    low=min(values);high=max(values)
    return one(low) if abs(high-low)<1e-12 else f"{one(low)}-{one(high)}"


def safe_json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def parse_sequence(text: str, count: int) -> list[float]:
    """Parse scalar, list, linspace, or start/step/stop array values.

    Accepted examples:
        1000
        1000, 1200, 1400
        [1000, 1200, 1400]
        1000 1200 1400
        1000; 1200; 1400
        (1000, 200, 1400)
        1000:200:1400
        linspace(1000, 1400, 3)

    A trailing micrometre unit (um, µm, or μm) is accepted on each number.
    """
    if count <= 0:
        return []

    source = str(text).strip()
    if not source:
        return [0.0] * count

    # Normalize characters commonly introduced by copy/paste from documents.
    source = (
        source.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("，", ",")
        .replace("；", ";")
    )

    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    def parse_number(token: str) -> float:
        cleaned = token.strip()
        cleaned = re.sub(r"\s*(?:u[mM]|µ[mM]|μ[mM])\s*$", "", cleaned)
        if not cleaned:
            raise ValueError("An empty value was found in the array entry.")
        if not re.fullmatch(number_pattern, cleaned):
            raise ValueError(
                f"'{token.strip()}' is not a valid number. "
                "Use entries such as 10, 10.5, -2, or 1e-3."
            )
        try:
            return float(cleaned)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not convert '{token.strip()}' to a number.") from exc

    def split_values(value_text: str) -> list[float]:
        body = value_text.strip()
        if not body:
            return []
        # Commas and semicolons are explicit separators. When neither is
        # present, whitespace may be used to paste a column from a spreadsheet.
        if "," in body or ";" in body:
            raw_tokens = re.split(r"[;,]", body)
        else:
            raw_tokens = re.split(r"\s+", body)
        tokens = [token for token in raw_tokens if token.strip()]
        return [parse_number(token) for token in tokens]

    linspace_match = re.fullmatch(
        rf"linspace\s*\(\s*({number_pattern})\s*[,;]\s*({number_pattern})\s*[,;]\s*(\d+)\s*\)",
        source,
        re.I,
    )
    if linspace_match:
        start = parse_number(linspace_match.group(1))
        stop = parse_number(linspace_match.group(2))
        n = int(linspace_match.group(3))
        if n <= 0:
            raise ValueError("linspace count must be greater than zero.")
        values = np.linspace(start, stop, n).tolist()
    elif source.startswith("[") and source.endswith("]"):
        values = split_values(source[1:-1])
    elif source.startswith("(") and source.endswith(")"):
        parts = split_values(source[1:-1])
        if len(parts) != 3:
            raise ValueError("Parenthesized sweep must be (start, step, stop).")
        start, step, stop = parts
        if step == 0:
            raise ValueError("Sweep step cannot be zero.")
        if (stop - start) * step < 0:
            raise ValueError("Sweep step points away from the stop value.")
        values = []
        value = start
        comparator = (
            (lambda current: current <= stop + 1e-12)
            if step > 0
            else (lambda current: current >= stop - 1e-12)
        )
        # Safety limit prevents accidental infinite or enormous sweeps.
        while comparator(value):
            values.append(float(value))
            if len(values) > 100000:
                raise ValueError("Sweep generates more than 100,000 values.")
            value += step
    elif source.count(":") == 2:
        parts = [parse_number(token) for token in source.split(":")]
        start, step, stop = parts
        if step == 0:
            raise ValueError("Sweep step cannot be zero.")
        if (stop - start) * step < 0:
            raise ValueError("Sweep step points away from the stop value.")
        values = []
        value = start
        comparator = (
            (lambda current: current <= stop + 1e-12)
            if step > 0
            else (lambda current: current >= stop - 1e-12)
        )
        while comparator(value):
            values.append(float(value))
            if len(values) > 100000:
                raise ValueError("Sweep generates more than 100,000 values.")
            value += step
    else:
        values = split_values(source)

    if not values:
        raise ValueError("No numeric array values were found.")
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"Expected {count} values, but received {len(values)}. "
            "Use one value to repeat it, or provide exactly one value per array position."
        )
    return [float(value) for value in values]
