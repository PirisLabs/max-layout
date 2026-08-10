@echo off
setlocal EnableExtensions
title Piris 3D Simulations

set "APP_ROOT=%~dp0."
set "BOOTSTRAP=%~dp0windows\Start-Piris3DSimulations.ps1"

if not exist "%BOOTSTRAP%" (
    echo.
    echo The Windows 3D launcher setup file is missing:
    echo   %BOOTSTRAP%
    echo.
    echo Extract the complete "Max Layout Windows.zip" before launching.
    pause
    exit /b 2
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo Windows PowerShell is required but was not found.
    pause
    exit /b 3
)

powershell.exe -STA -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP%" -SearchRoot "%APP_ROOT%" %*
set "RESULT=%ERRORLEVEL%"

echo.
if "%RESULT%"=="0" (
    echo Piris 3D Simulations launcher finished.
) else (
    echo The launcher stopped with an error. Review the message above and its log.
    echo Log folder: %LOCALAPPDATA%\PirisLabs\3DLauncher\logs
)
pause
exit /b %RESULT%
