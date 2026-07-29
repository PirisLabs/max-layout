"""Runtime dependency bootstrapping (pip installs, user site-packages)."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import site
import subprocess
import sys


def _ensure_runtime_dependencies() -> None:
    def add_user_site_to_path() -> None:
        user_sites = site.getusersitepackages()
        if isinstance(user_sites, str):
            user_sites = [user_sites]
        for user_site in user_sites:
            if user_site:
                site.addsitedir(user_site)
                if user_site not in sys.path:
                    sys.path.insert(0, user_site)

    def dependencies_available() -> bool:
        try:
            import gdstk  # noqa: F401
            import numpy  # noqa: F401
            return True
        except (ModuleNotFoundError, ImportError):
            sys.modules.pop("gdstk", None)
            sys.modules.pop("numpy", None)
            return False

    # Apple Command Line Tools Python can install into the user site without
    # automatically adding that directory to sys.path. Add it explicitly.
    add_user_site_to_path()
    importlib.invalidate_caches()
    if dependencies_available():
        return

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

    # First use the normal user installation. This works for standard macOS
    # and Windows Python installations and reuses packages already downloaded.
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "--upgrade",
                "gdstk>=0.9.60",
                "numpy>=1.24",
            ]
        )
    except subprocess.CalledProcessError:
        pass

    add_user_site_to_path()
    importlib.invalidate_caches()
    if dependencies_available():
        return

    # Fallback: install into an editor-owned runtime folder and put it first
    # on sys.path. This avoids PATH and site-packages permission problems.
    runtime_dir = (
        Path.home()
        / ".photonic_layout_editor_runtime"
        / f"python_{sys.version_info.major}_{sys.version_info.minor}"
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = str(runtime_dir)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            runtime_path,
            "gdstk>=0.9.60",
            "numpy>=1.24",
        ]
    )

    importlib.invalidate_caches()
    sys.modules.pop("gdstk", None)
    sys.modules.pop("numpy", None)
    if not dependencies_available():
        raise RuntimeError(
            "Dependencies were installed but could not be imported. "
            f"Runtime folder: {runtime_path}"
        )


def _ensure_native_gui_dependencies() -> None:
    def add_user_site() -> None:
        paths = site.getusersitepackages()
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if path:
                site.addsitedir(path)
                if path not in sys.path:
                    sys.path.insert(0, path)

    add_user_site()
    importlib.invalidate_caches()
    try:
        import PySide6  # noqa: F401
        return
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "--upgrade",
                "PySide6>=6.6",
            ]
        )
    except subprocess.CalledProcessError:
        runtime_dir = (
            Path.home()
            / ".photonic_layout_editor_runtime"
            / f"python_{sys.version_info.major}_{sys.version_info.minor}"
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(runtime_dir),
                "PySide6>=6.6",
            ]
        )
        site.addsitedir(str(runtime_dir))
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))

    add_user_site()
    importlib.invalidate_caches()
    import PySide6  # noqa: F401
