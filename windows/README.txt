MAX LAYOUT FOR WINDOWS
======================

1. Extract the complete Max Layout Windows.zip file.
2. Double-click "Max Layout Windows.cmd".
3. Keep the setup window open during the first launch.

The first launch automatically:

- locates a compatible Python installation;
- installs Python 3.12 for the current user when Python is missing;
- creates a private environment under
  %LOCALAPPDATA%\PirisLabs\MaxLayout\runtime;
- installs PySide6, NumPy, and gdstk; and
- opens Max Layout.

Later launches verify the environment and normally open immediately. The
setup runs again only when its dependency list changes or the environment is
incomplete.

No administrator rights are normally required. Setup logs are stored under:

  %LOCALAPPDATA%\PirisLabs\MaxLayout\logs

If setup fails, leave the command window open and use the log path printed at
the bottom of the window.

STARTING A LAMBDA NODE / 3D SIMULATION
--------------------------------------

Double-click "Start Piris 3D Simulations Windows.cmd". On its first launch it
installs the Google Drive launcher packages and, when necessary, Windows
OpenSSH Client. It then runs the same project, notebook, Lambda-node, Jupyter,
result-sync, and one-hour idle-termination workflow as the Mac launcher.

The confidential Piris 3D Launcher folder is intentionally not included in
this public bundle because it contains private company credentials. The first
Windows launch searches Downloads and Desktop, then asks you to select that
private folder. The chosen location is remembered locally for later launches.

During first-time SSH setup, choose either:

- "Use Ali shared access key". Put the confidential private key at
  %USERPROFILE%\.ssh\piris_alik, or set PIRIS_SHARED_SSH_PRIVATE_KEY to its
  existing secure location; or
- "Create a new Piris SSH key". The launcher registers only its public half
  with Lambda and immediately uses the matching local private key.

Neither the shared key nor any newly created private key is copied into this
folder, the public ZIP, or GitHub. Lambda-launcher logs are stored under:

  %LOCALAPPDATA%\PirisLabs\3DLauncher\logs
