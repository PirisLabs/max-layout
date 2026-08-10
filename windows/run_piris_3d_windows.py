#!/usr/bin/env python3
"""Windows compatibility entry point for the private Piris 3D launcher.

This file deliberately contains no Lambda, Google Drive, SSH, or Lumerical
credentials.  It loads the user's existing private ``launch_3d_simulations.py``
at runtime and replaces only the few local-desktop operations that differ on
Windows.  The private launcher remains the authority for provisioning, syncing,
licensing, and simulation behavior.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import getpass
import importlib.util
import os
import subprocess
import sys
import urllib.parse
import webbrowser
from types import ModuleType
from typing import Sequence


MINIMUM_PYTHON = (3, 10)


def _unquote_dropped_path(value: str, empty_message: str) -> str:
    """Strip one Windows drag/drop quote pair without interpreting backslashes."""
    raw = str(value).strip()
    if not raw:
        raise ValueError(empty_message)
    if raw[0] in ("'", '"'):
        if len(raw) < 2 or raw[-1] != raw[0]:
            raise ValueError("the selected path has an unmatched quote")
        raw = raw[1:-1]
    elif raw[-1] in ("'", '"'):
        raise ValueError("the selected path has an unmatched quote")
    if not raw:
        raise ValueError(empty_message)
    return raw


def windows_dragged_notebook_path(value: str) -> str:
    """Return one dropped notebook path without POSIX ``shlex`` processing.

    Windows Explorer commonly surrounds a dropped path containing spaces with
    quotes.  Backslashes must remain literal: POSIX ``shlex.split`` would treat
    them as escape characters and corrupt paths such as ``C:\\Users\\...``.
    """
    raw = _unquote_dropped_path(
        value, "drag exactly one .ipynb notebook into the window"
    )
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(raw)))
    if not os.path.isfile(path) or not path.casefold().endswith(".ipynb"):
        raise ValueError("the selected file must be an existing .ipynb notebook")
    return path


def discover_ali_shared_ssh_keys(
    launcher_script: str | None = None,
    environment: dict[str, str] | None = None,
) -> list[str]:
    """Find, but never read or copy, the private ``piris_alik`` access key.

    Candidate priority is explicit configuration, the current Windows account,
    then fixed locations inside the user-selected confidential launcher folder.
    The public repository and Windows ZIP never contain this private file.
    """
    environment = environment if environment is not None else dict(os.environ)
    candidates = []
    configured = environment.get("PIRIS_SHARED_SSH_PRIVATE_KEY")
    if configured:
        candidates.append(configured)
    user_profile = environment.get("USERPROFILE")
    if user_profile:
        candidates.append(os.path.join(user_profile, ".ssh", "piris_alik"))
    if launcher_script:
        script = os.path.abspath(os.path.expanduser(launcher_script))
        scripts_directory = os.path.dirname(script)
        requirements_directory = os.path.dirname(scripts_directory)
        launcher_directory = os.path.dirname(requirements_directory)
        for directory in (
            launcher_directory,
            requirements_directory,
            scripts_directory,
            os.path.join(requirements_directory, "SSH"),
            os.path.join(requirements_directory, "Keys"),
            os.path.join(scripts_directory, "helpers"),
        ):
            candidates.append(os.path.join(directory, "piris_alik"))

    discovered = []
    seen = set()
    for candidate in candidates:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(candidate))))
        token = os.path.normcase(path)
        if token in seen:
            continue
        seen.add(token)
        if os.path.isfile(path) and not path.casefold().endswith(".pub"):
            discovered.append(path)
    return discovered


def _prompt_for_private_key_path() -> str:
    print("Ali's shared access key was not found automatically.")
    print("Drag the private piris_alik key file into this window and press Return.")
    while True:
        try:
            raw = input("Ali shared access key: ")
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("cancelled; no SSH private key was selected") from None
        try:
            value = _unquote_dropped_path(raw, "the SSH key path cannot be blank")
        except ValueError as exc:
            print("  " + str(exc))
            continue
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(value)))
        if os.path.isfile(path) and not path.casefold().endswith(".pub"):
            return path
        print("  Select the existing private key file, not its .pub file.")


def _select_windows_user_ssh_key(module: ModuleType) -> str:
    """Offer the shared Ali key or a newly registered per-computer key."""
    configured = os.environ.get("PIRIS_SSH_PRIVATE_KEY")
    if configured:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
        if not os.path.isfile(path):
            raise RuntimeError("PIRIS_SSH_PRIVATE_KEY does not name an existing file")
        return path
    shared_override = os.environ.get("PIRIS_SHARED_SSH_PRIVATE_KEY")
    if shared_override:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(
            shared_override
        )))
        if not os.path.isfile(path) or path.casefold().endswith(".pub"):
            raise RuntimeError(
                "PIRIS_SHARED_SSH_PRIVATE_KEY must name Ali's existing private key"
            )
        return path
    saved = module._read_user_config().get("private_key")
    if saved:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(saved))))
        if os.path.isfile(path):
            return path

    shared_keys = discover_ali_shared_ssh_keys(module.__file__)
    print("\nFirst-time SSH setup for this computer:")
    shared_label = "Use Ali shared access key"
    create_label = "Create a new Piris SSH key"
    options = [shared_label, create_label]
    shared_tokens = {os.path.normcase(path) for path in shared_keys}
    existing = [
        path for path in module._existing_private_keys()
        if os.path.normcase(os.path.abspath(path)) not in shared_tokens
    ]
    options.extend("Use existing key: " + path for path in existing)
    selected = module.choose_from_list("SSH key", options)
    if selected == shared_label:
        return shared_keys[0] if shared_keys else _prompt_for_private_key_path()
    if selected == create_label:
        # ensure_user_ssh immediately registers and then returns this exact key.
        return os.path.abspath(module._create_user_ssh_key())
    return os.path.abspath(selected.split(": ", 1)[1])


def _windows_state_root() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        local_app_data = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.abspath(os.path.join(
        local_app_data, "PirisLabs", "3DLauncher", "state"
    ))


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _load_psutil():
    try:
        import psutil  # type: ignore
    except ImportError:
        raise RuntimeError(
            "the Windows launcher runtime is missing psutil; rerun the clickable "
            "launcher so it can repair its requirements"
        ) from None
    return psutil


def _cancel_windows_watchdog(module: ModuleType, controller_id: str,
                             directory: str | None = None) -> bool:
    """Cancel only a process whose command line proves it is our watchdog."""
    directory = os.path.abspath(directory or module.WATCHDOG_DIR)
    state_path, config_path, _log_path = module._watchdog_paths(
        controller_id, directory
    )
    try:
        state = module._read_json_file(state_path)
        pid = int(state["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False

    # Never trust a path from a stale/tampered state file as a deletion target.
    recorded_config = os.path.abspath(str(state.get("config_path", config_path)))
    config_is_expected = _same_path(recorded_config, config_path)
    psutil = _load_psutil()
    stopped = False
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
    except psutil.NoSuchProcess:
        command = []
    except (psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(
            "could not verify the prior idle watchdog: %s" % exc
        ) from None

    normalized_arguments = []
    for argument in command:
        try:
            normalized_arguments.append(os.path.normcase(os.path.abspath(argument)))
        except (OSError, TypeError, ValueError):
            continue
    verified = (
        config_is_expected
        and "--idle-watchdog" in command
        and os.path.normcase(config_path) in normalized_arguments
    )
    if verified:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stopped = True
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, psutil.TimeoutExpired, OSError) as exc:
            raise RuntimeError(
                "could not cancel the prior idle watchdog: %s" % exc
            ) from None

    for path in (state_path, config_path):
        try:
            os.unlink(path)
        except OSError:
            pass
    return stopped


def _watchdog_subprocess_command(module: ModuleType, config_path: str) -> list[str]:
    """Build a credential-free command that re-enters this compatibility file."""
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--launcher-script",
        os.path.abspath(module.__file__),
        "--idle-watchdog",
        "--watchdog-config",
        os.path.abspath(config_path),
    ]


def _schedule_windows_watchdog(module: ModuleType, lambda_key: str,
                               instances: list[dict], key_path: str,
                               directory: str | None = None) -> dict:
    """Start a detached Windows watchdog without placing credentials in argv."""
    if not instances:
        raise ValueError("cannot schedule an idle watchdog without instances")
    directory = os.path.abspath(directory or module.WATCHDOG_DIR)
    controller_id = str(instances[0]["id"])
    _cancel_windows_watchdog(module, controller_id, directory)
    state_path, config_path, log_path = module._watchdog_paths(
        controller_id, directory
    )
    config = {
        "version": 1,
        "controller_id": controller_id,
        "nodes": [
            {
                "id": str(item["id"]),
                "name": str(item.get("name", "")),
                "ip": str(item.get("ip", "")),
            }
            for item in instances
        ],
        "ssh_key_path": os.path.abspath(os.path.expanduser(key_path)),
        "idle_seconds": module.IDLE_TERMINATION_SECONDS,
        "poll_seconds": module.IDLE_WATCHDOG_POLL_SECONDS,
        "watchdog_directory": directory,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    module._write_private_json(config_path, config)
    command = _watchdog_subprocess_command(module, config_path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    creation_flags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    process = None
    try:
        with open(log_path, "a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creation_flags,
                close_fds=True,
            )
            if process.stdin is None:
                raise RuntimeError(
                    "could not create the private watchdog credential pipe"
                )
            try:
                process.stdin.write(lambda_key + "\n")
                process.stdin.flush()
            finally:
                process.stdin.close()
        if process.poll() is not None:
            raise RuntimeError(
                "the local idle watchdog exited before it was scheduled"
            )
        state = {
            "pid": process.pid,
            "controller_id": controller_id,
            "config_path": config_path,
            "log_path": log_path,
        }
        module._write_private_json(state_path, state)
        return state
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
        for path in (state_path, config_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def _run_windows_browser_session(
    module: ModuleType,
    ssh_base: list[str],
    ip: str,
    key_path: str,
    remote_port: int,
    project_root: str,
    notebook: str,
    token: str,
    control_python: str,
    config_path: str,
    session_name: str,
    open_browser: bool,
) -> None:
    """Preserve the private launcher's session lifecycle and use the Windows browser."""
    del project_root  # Kept in the signature for exact launcher compatibility.
    local_port = module.free_local_port()
    tunnel_command = [
        "ssh", "-i", key_path, "-N",
        "-L", "%d:127.0.0.1:%d" % (local_port, remote_port),
        "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=20",
        "ubuntu@" + ip,
    ]
    tunnel = subprocess.Popen(
        tunnel_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        module.wait_local_port(local_port, tunnel)
        encoded = urllib.parse.quote(notebook, safe="/")
        url = "http://127.0.0.1:%d/lab/tree/%s?token=%s" % (
            local_port,
            encoded,
            urllib.parse.quote(token, safe=""),
        )
        print("\nJupyterLab is ready:")
        print("  " + url)
        if open_browser and not webbrowser.open(url, new=2):
            print("The browser did not open automatically; copy the URL above.")
        print("\nResults are syncing automatically while this window stays open.")
        input("When the simulation is finished, save the notebook and press Enter here: ")
        module.final_sync(ssh_base, control_python, config_path, session_name)
        print("Executed notebook and results are saved in the numbered Results folder.")
        module.offer_notebook_publish(ssh_base, control_python, config_path)
    except KeyboardInterrupt:
        print("\nClosing the tunnel; attempting one final sync.")
        module.final_sync(ssh_base, control_python, config_path, session_name)
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()


def _harden_windows_private_key(path: str) -> None:
    """Restrict a private key to the current Windows account with ``icacls``."""
    identity = os.environ.get("USERNAME") or getpass.getuser()
    result = subprocess.run(
        [
            "icacls",
            os.path.abspath(path),
            "/inheritance:r",
            "/grant:r",
            "%s:(F)" % identity,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(
            "Windows could not secure the selected SSH private key with icacls: %s"
            % (detail or "unknown icacls error")
        )


def _lambda_remote_contract_failures(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        return ["lambda_remote.py could not be inspected (%s)" % exc]
    lambda_class = next(
        (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Lambda"
        ),
        None,
    )
    if lambda_class is None:
        return ["lambda_remote.py has no Lambda class"]
    methods = {
        node.name: node
        for node in lambda_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    failures = []
    for name in ("run_once", "stop_work_processes"):
        if name not in methods:
            failures.append("lambda_remote.Lambda.%s is missing" % name)
    initializer = methods.get("__init__")
    if initializer is None:
        failures.append("lambda_remote.Lambda.__init__ is missing")
    else:
        arguments = initializer.args
        accepted = {
            item.arg
            for item in list(arguments.posonlyargs)
            + list(arguments.args)
            + list(arguments.kwonlyargs)
        }
        accepts_arbitrary_keywords = arguments.kwarg is not None
        for name in ("host", "key"):
            if name not in accepted and not accepts_arbitrary_keywords:
                failures.append(
                    "lambda_remote.Lambda.__init__ does not accept %s=" % name
                )
    return failures


def validate_multigpu_helper_contract(module: ModuleType) -> None:
    """Reject outdated private helpers before any A100 nodes are provisioned."""
    failures = []
    node_session = module.local_file("node_session.py")
    lambda_remote = module.local_file("lambda_remote.py")
    if not node_session:
        failures.append("node_session.py was not found")
    else:
        try:
            with open(node_session, encoding="utf-8") as handle:
                node_text = handle.read()
        except OSError as exc:
            failures.append("node_session.py could not be inspected (%s)" % exc)
        else:
            if "--lumerical-inventory" not in node_text:
                failures.append("node_session.py lacks --lumerical-inventory")
            if "PIRIS_LUMERICAL_INVENTORY" not in node_text:
                failures.append("node_session.py does not export PIRIS_LUMERICAL_INVENTORY")
    if not lambda_remote:
        failures.append("lambda_remote.py was not found")
    else:
        failures.extend(_lambda_remote_contract_failures(lambda_remote))
    if failures:
        raise RuntimeError(
            "the private Piris 3D Launcher multi-GPU helpers are outdated: %s. "
            "Update the private Requirements helper files before retrying. "
            "No Lambda nodes were launched."
            % "; ".join(failures)
        )


def apply_windows_patches(module: ModuleType) -> None:
    """Install Windows-only behavior into one dynamically loaded launcher module."""
    state_root = _windows_state_root()
    module.USER_CONFIG = os.path.join(state_root, "user.json")
    module.WATCHDOG_DIR = os.path.join(state_root, "idle-watchdogs")
    module._dragged_notebook_path = windows_dragged_notebook_path
    module._select_user_ssh_key = lambda: _select_windows_user_ssh_key(module)

    module.cancel_idle_termination_watchdog = (
        lambda controller_id, directory=None: _cancel_windows_watchdog(
            module, controller_id, directory
        )
    )
    module.watchdog_subprocess_command = (
        lambda _script_path, config_path: _watchdog_subprocess_command(
            module, config_path
        )
    )
    module.schedule_idle_termination_watchdog = (
        lambda lambda_key, instances, key_path, directory=None:
        _schedule_windows_watchdog(
            module, lambda_key, instances, key_path, directory
        )
    )
    module.run_browser_session = (
        lambda ssh_base, ip, key_path, remote_port, project_root, notebook,
        token, control_python, config_path, session_name, open_browser:
        _run_windows_browser_session(
            module, ssh_base, ip, key_path, remote_port, project_root,
            notebook, token, control_python, config_path, session_name,
            open_browser,
        )
    )

    original_public_key_for = module._public_key_for
    original_ensure_user_ssh = module.ensure_user_ssh

    def secured_public_key_for(private_path: str) -> str:
        _harden_windows_private_key(private_path)
        return original_public_key_for(private_path)

    def secured_ensure_user_ssh(lambda_key: str,
                                allow_register: bool = True) -> tuple[str, str]:
        private_path, cloud_name = original_ensure_user_ssh(
            lambda_key, allow_register
        )
        _harden_windows_private_key(private_path)
        return private_path, cloud_name

    module._public_key_for = secured_public_key_for
    module.ensure_user_ssh = secured_ensure_user_ssh

    original_execution_plan = module.notebook_execution_plan

    def checked_execution_plan(portal, project: dict, notebook: str) -> dict:
        plan = original_execution_plan(portal, project, notebook)
        if bool(plan.get("multigpu")):
            validate_multigpu_helper_contract(module)
        return plan

    module.notebook_execution_plan = checked_execution_plan


def load_private_launcher(path: str) -> ModuleType:
    launcher_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(launcher_path):
        raise RuntimeError(
            "the private Piris launcher script was not found: %s" % launcher_path
        )
    specification = importlib.util.spec_from_file_location(
        "_piris_private_3d_launcher", launcher_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("the private Piris launcher could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(specification.name, None)
        raise
    return module


def main(argv: Sequence[str] | None = None) -> int:
    if sys.version_info < MINIMUM_PYTHON:
        raise RuntimeError("Piris 3D Launcher requires Python 3.10 or newer")
    parser = argparse.ArgumentParser(
        description="Windows entry point for the private Piris 3D Launcher",
        add_help=False,
    )
    parser.add_argument("--launcher-script", required=True)
    wrapper_args, launcher_args = parser.parse_known_args(argv)
    module = load_private_launcher(wrapper_args.launcher_script)
    apply_windows_patches(module)
    sys.argv = [os.path.abspath(module.__file__), *launcher_args]
    if "--idle-watchdog" in launcher_args:
        return int(module.run_hidden_idle_watchdog(launcher_args))
    return int(module.main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
