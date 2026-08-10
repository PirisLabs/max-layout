@echo off
setlocal EnableExtensions
title Max Layout

set "APP_ROOT=%~dp0."
set "BOOTSTRAP=%~dp0windows\Install-And-Launch-MaxLayout.ps1"

if not exist "%BOOTSTRAP%" (
    echo.
    echo Max Layout cannot find its Windows setup file:
    echo   %BOOTSTRAP%
    echo.
    echo Extract the complete "Max Layout Windows.zip" before launching.
    pause
    exit /b 2
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo.
    echo Windows PowerShell is required but was not found.
    echo Install or enable Windows PowerShell, then double-click this file again.
    pause
    exit /b 3
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -AppRoot "%APP_ROOT%"
set "RESULT=%ERRORLEVEL%"

if not "%RESULT%"=="0" (
    echo.
    echo Max Layout setup did not finish. Review the message above and the launcher log.
    echo Log folder: %LOCALAPPDATA%\PirisLabs\MaxLayout\logs
    pause
)

exit /b %RESULT%
