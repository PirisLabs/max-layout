"""Optional OpenAI assistant for layout and source edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import re
import tempfile
import urllib.request

from .constants import DEFAULT_COMPONENT_VALUES
from .runtime import iter_package_sources


def _extract_openai_native_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            value = content.get("text")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()


def _openai_native_request(
    api_key: str,
    model: str,
    instructions: str,
    timeout: int = 180,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "input": instructions,
            "reasoning": {"effort": "medium"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI API error {exc.code}: {detail[:1200]}") from exc
    output = _extract_openai_native_text(data)
    if not output:
        raise ValueError("The OpenAI response contained no text.")
    return re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        output.strip(),
        flags=re.I | re.S,
    )


def _call_openai_native_layout(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = str(payload.get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OpenAI cloud mode needs an API key.")
    model = str(payload.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-5.6").strip()
    prompt = str(payload.get("prompt") or "").strip()
    layout = payload.get("layout", [])
    kinds = list(DEFAULT_COMPONENT_VALUES)
    instructions = f"""You control a native photonic/RF graphical layout editor.
Return ONLY valid JSON with keys message and actions.
Allowed component kinds: {kinds}.
Allowed actions:
add(kind,x,y,orientation_deg,mirrored,params), select_all,
select_kind(kind), move_selected(x,y), move_selected_by(dx,dy),
rotate_selected(angle), mirror_selected, connect_nearest,
duplicate_selected(dx,dy), delete_selected, set_params(params),
center_layout, fit.
Coordinates and dimensions are micrometers. Use exact component names and
existing parameters only. connect_nearest attaches the selected component
input port to the closest compatible optical, RF, or neutral alignment point.
Current layout:
{json.dumps(layout, separators=(',', ':'))}
User instruction:
{prompt}
"""
    output = _openai_native_request(api_key, model, instructions, timeout=120)
    try:
        plan = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The layout assistant returned invalid JSON: {output[:1200]}") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("actions", []), list):
        raise ValueError("The layout assistant response is missing an actions list.")
    return plan


def _call_openai_native_source(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = str(payload.get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Source-update mode needs an OpenAI API key.")
    model = str(payload.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-5.6").strip()
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Enter a source-code update request.")
    sources = dict(iter_package_sources())
    if not sources:
        raise ValueError("Could not read the application source.")
    listing = "\n".join(
        f"=== FILE: {name} ===\n{text}" for name, text in sorted(sources.items())
    )
    instructions = f"""You are updating a multi-file native PySide6 photonic/RF layout editor.
The source is supplied as several files, each introduced by a line of the form
=== FILE: relative/path.py ===
Return ONLY valid JSON with keys message and replacements.
replacements is a list of objects with exact strings old and new.
Each old string must occur exactly once across all supplied files combined.
Make the smallest safe changes. Preserve all existing functionality.
Do not modify or reproduce the immutable photonic component function definitions.
The result must remain valid runnable Python.
User request:
{prompt}

CURRENT SOURCE:
{listing}
"""
    output = _openai_native_request(api_key, model, instructions, timeout=240)
    try:
        plan = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The source assistant returned invalid JSON: {output[:1200]}") from exc
    replacements = plan.get("replacements", [])
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("The source assistant returned no replacements.")
    if len(replacements) > 50:
        raise ValueError("The source assistant returned too many replacements.")
    updated = dict(sources)
    for index, replacement in enumerate(replacements, start=1):
        if not isinstance(replacement, dict):
            raise ValueError(f"Replacement {index} is invalid.")
        old = replacement.get("old")
        new = replacement.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError(f"Replacement {index} requires old and new strings.")
        # The exactly-once rule now spans the whole package, not one file.
        hits = [name for name, text in updated.items() if old in text]
        count = sum(updated[name].count(old) for name in hits)
        if count != 1:
            raise ValueError(
                f"Replacement {index} old text occurs {count} times; exactly one is required."
            )
        target = hits[0]
        updated[target] = updated[target].replace(old, new, 1)
    changed = {name: text for name, text in updated.items() if text != sources[name]}
    for name, text in changed.items():
        compile(text, name, "exec")
    timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    output_dir = Path(tempfile.gettempdir()) / f"max_layout_ai_update_{timestamp}"
    for name, text in changed.items():
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    return {
        "message": str(plan.get("message") or "Created a validated source update."),
        "output_file": str(output_dir),
        "changed_files": sorted(changed),
        "replacement_count": len(replacements),
    }


def _worker_llm_assistant(request_file: str, response_file: str) -> None:
    payload = json.loads(Path(request_file).read_text())
    task = str(payload.get("task") or "layout")
    if task == "source":
        result = _call_openai_native_source(payload)
    else:
        result = {"plan": _call_openai_native_layout(payload)}
    Path(response_file).write_text(json.dumps(result))
