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
