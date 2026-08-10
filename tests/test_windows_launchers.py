from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest
import zipfile

from build_windows_bundle import bundle_payload


ROOT = Path(__file__).resolve().parents[1]
MAX_CMD = ROOT / "Max Layout Windows.cmd"
LAMBDA_CMD = ROOT / "Start Piris 3D Simulations Windows.cmd"
MAX_PS = ROOT / "windows" / "Install-And-Launch-MaxLayout.ps1"
LAMBDA_PS = ROOT / "windows" / "Start-Piris3DSimulations.ps1"
COMPATIBILITY_WRAPPER = ROOT / "windows" / "run_piris_3d_windows.py"
GUI_REQUIREMENTS = ROOT / "windows" / "requirements-windows.txt"
LAMBDA_REQUIREMENTS = ROOT / "windows" / "requirements-3d-launcher.txt"
BUNDLE_BUILDER = ROOT / "build_windows_bundle.py"
TRACKED_BUNDLE = ROOT / "Max Layout Windows.zip"

BUNDLE_FILES = {
    "LICENSE": ROOT / "LICENSE",
    "Max Layout.pyz": ROOT / "Max Layout.pyz",
    "Max Layout Windows.cmd": MAX_CMD,
    "Start Piris 3D Simulations Windows.cmd": LAMBDA_CMD,
    "windows/Install-And-Launch-MaxLayout.ps1": MAX_PS,
    "windows/requirements-windows.txt": GUI_REQUIREMENTS,
    "windows/Start-Piris3DSimulations.ps1": LAMBDA_PS,
    "windows/requirements-3d-launcher.txt": LAMBDA_REQUIREMENTS,
    "windows/run_piris_3d_windows.py": COMPATIBILITY_WRAPPER,
    "windows/README.txt": ROOT / "windows" / "README.txt",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in _text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line
        for operator in ("===", "==", "~=", ">=", "<=", "!=", ">", "<"):
            name = name.split(operator, 1)[0]
        names.add(name.strip().casefold().replace("_", "-"))
    return names


class WindowsCommandLauncherTests(unittest.TestCase):
    def test_max_layout_cmd_is_a_quoted_per_process_bootstrap(self) -> None:
        text = _text(MAX_CMD)
        lower = text.casefold()
        self.assertIn('set "app_root=%~dp0."', lower)
        self.assertIn('set "bootstrap=%~dp0windows\\install-and-launch-maxlayout.ps1"', lower)
        self.assertIn('-executionpolicy bypass', lower)
        self.assertIn('-file "%bootstrap%"', lower)
        self.assertIn('-approot "%app_root%"', lower)
        self.assertIn("pause", lower)
        self.assertNotIn("set-executionpolicy", lower)

    def test_lambda_cmd_forwards_all_launcher_arguments(self) -> None:
        text = _text(LAMBDA_CMD)
        lower = text.casefold()
        self.assertIn('set "app_root=%~dp0."', lower)
        self.assertIn('set "bootstrap=%~dp0windows\\start-piris3dsimulations.ps1"', lower)
        self.assertIn("powershell.exe -sta", lower)
        self.assertIn('-file "%bootstrap%"', lower)
        self.assertIn('-searchroot "%app_root%" %*', lower)
        self.assertNotIn("set-executionpolicy", lower)


class WindowsPowerShellBootstrapTests(unittest.TestCase):
    def test_max_layout_bootstrap_contract(self) -> None:
        text = _text(MAX_PS)
        lower = text.casefold()
        for required in (
            "$env:localappdata",
            "pirislabs\\maxlayout",
            "system.threading.mutex",
            "requirements.sha256",
            "get-filehash",
            "python.python.3.12",
            "get-authenticodesignature",
            "python software foundation",
            '"venv"',
            "--only-binary=:all:",
            "pythonw.exe",
            "start-process",
            "start-transcript",
            "replacing an incompatible or incomplete max layout environment",
            "remove-item -literalpath $venvroot -recurse -force",
            "platform.machine().lower()",
            "< (3, 13)",
            "[void](install-pythonwithwinget)",
        ):
            self.assertIn(required, lower)
        self.assertNotIn('"-3.13"', lower)
        self.assertNotIn("set-executionpolicy", lower)

    def test_lambda_bootstrap_contract(self) -> None:
        text = _text(LAMBDA_PS)
        lower = text.casefold()
        for required in (
            "valuefromremainingarguments",
            "$launcherarguments",
            "folderbrowserdialog",
            "confidential piris 3d launcher",
            "openssh.client",
            "ssh.exe",
            "scp.exe",
            "ssh-keygen.exe",
            "requirements.sha256",
            "get-filehash",
            "run_piris_3d_windows.py",
            "--launcher-script",
            "@launcherarguments",
            "(3, 10) <= sys.version_info",
            "python 3.10-3.13",
            "psutil",
            "function test-launcherenvironment",
            "replacing an incompatible or incomplete launcher environment",
            "remove-item -literalpath $venvroot -recurse -force",
            "short-lived jupyter url token",
            "$transcriptstarted = $false",
            "x64 python is still unavailable",
            "$missing = @(",
            "windows powershell 5.1 collapses",
        ):
            self.assertIn(required, lower)
        self.assertNotIn('"-3.9"', lower)
        self.assertNotIn("(3, 9) <= sys.version_info", lower)
        self.assertNotIn("set-executionpolicy", lower)


class WindowsCompatibilityTests(unittest.TestCase):
    def test_compatibility_wrapper_is_valid_python(self) -> None:
        source = _text(COMPATIBILITY_WRAPPER)
        ast.parse(source, filename=str(COMPATIBILITY_WRAPPER))

    def test_compatibility_wrapper_covers_windows_only_gaps(self) -> None:
        source = _text(COMPATIBILITY_WRAPPER)
        lower = source.casefold()
        for required in (
            "webbrowser.open",
            "psutil.process",
            "creationflags",
            "create_new_process_group",
            "detached_process",
            "create_no_window",
            "icacls",
            "--idle-watchdog",
            "--watchdog-config",
            "--lumerical-inventory",
            "piris_lumerical_inventory",
            "run_once",
            "stop_work_processes",
            "--launcher-script",
            "piris_shared_ssh_private_key",
            "piris_alik",
            "_create_user_ssh_key",
            "register",
        ):
            self.assertIn(required, lower)

    def test_public_wrapper_contains_no_private_credential_file(self) -> None:
        lower = _text(COMPATIBILITY_WRAPPER).casefold()
        self.assertNotIn("lumerical-504521-7c755c2c58a7.json", lower)
        self.assertNotIn("remote-token.json", lower)
        self.assertNotIn("begin openssh private key", lower)

    def test_explicit_shared_key_override_precedes_saved_key(self) -> None:
        source = _text(COMPATIBILITY_WRAPPER)
        self.assertLess(
            source.index('shared_override = os.environ.get("PIRIS_SHARED_SSH_PRIVATE_KEY")'),
            source.index('saved = module._read_user_config().get("private_key")'),
        )


class WindowsDependencyTests(unittest.TestCase):
    def test_gui_bootstrap_has_only_the_three_runtime_packages(self) -> None:
        self.assertEqual(
            _requirement_names(GUI_REQUIREMENTS),
            {"pyside6", "numpy", "gdstk"},
        )
        text = _text(GUI_REQUIREMENTS).casefold()
        self.assertNotIn("torch", text)
        self.assertNotIn("cupy", text)

    def test_lambda_bootstrap_has_only_google_and_watchdog_packages(self) -> None:
        self.assertEqual(
            _requirement_names(LAMBDA_REQUIREMENTS),
            {"google-auth", "google-api-python-client", "psutil"},
        )


class WindowsBundleTests(unittest.TestCase):
    def test_gitignore_excludes_known_private_launcher_credentials(self) -> None:
        text = _text(ROOT / ".gitignore")
        for pattern in (
            "**/LUMERICAL.txt",
            "**/lumerical-*.json",
            "**/remote-token.json",
            "**/.lambda-ssh/",
            "**/piris_alik",
            "**/piris_3d_*",
            "**/piris_windows_*",
        ):
            self.assertIn(pattern, text)

    def test_builder_uses_an_explicit_public_allowlist(self) -> None:
        source = _text(BUNDLE_BUILDER)
        ast.parse(source, filename=str(BUNDLE_BUILDER))
        self.assertIn("BUNDLE_FILES", source)
        self.assertNotIn("rglob(", source)
        self.assertNotIn("glob(", source)

    def test_built_zip_is_complete_and_has_no_private_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "Max Layout Windows.zip"
            completed = subprocess.run(
                [sys.executable, str(BUNDLE_BUILDER), "--output", str(output)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, set(BUNDLE_FILES))
                for archive_name, source_path in BUNDLE_FILES.items():
                    self.assertEqual(
                        archive.read(archive_name),
                        bundle_payload(source_path, archive_name),
                    )

                for archive_name in names:
                    path = PurePosixPath(archive_name)
                    basename = path.name.casefold()
                    self.assertNotEqual(basename, "lumerical.txt")
                    self.assertNotEqual(basename, "remote-token.json")
                    self.assertNotIn(basename, {"piris_alik", "piris_alik.pub"})
                    self.assertFalse(basename.startswith("piris_3d_"))
                    self.assertFalse(basename.startswith("piris_windows_"))
                    self.assertFalse(
                        basename.startswith("lumerical-") and basename.endswith(".json")
                    )
                    self.assertNotIn(".lambda-ssh", {part.casefold() for part in path.parts})

    def test_tracked_download_bundle_matches_current_sources(self) -> None:
        self.assertTrue(TRACKED_BUNDLE.is_file())
        with zipfile.ZipFile(TRACKED_BUNDLE) as archive:
            self.assertEqual(set(archive.namelist()), set(BUNDLE_FILES))
            for archive_name, source_path in BUNDLE_FILES.items():
                self.assertEqual(
                    archive.read(archive_name),
                    bundle_payload(source_path, archive_name),
                    "%s is stale in the tracked Windows ZIP" % archive_name,
                )


if __name__ == "__main__":
    unittest.main()
