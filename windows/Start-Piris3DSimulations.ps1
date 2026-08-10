[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SearchRoot,
    [string]$LauncherRoot = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$SearchRoot = (Resolve-Path -LiteralPath $SearchRoot).Path
$StateRoot = Join-Path $env:LOCALAPPDATA "PirisLabs\3DLauncher"
$VenvRoot = Join-Path $StateRoot "runtime"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$ReadyMarker = Join-Path $StateRoot "requirements.sha256"
$SavedLauncherRoot = Join-Path $StateRoot "launcher-root.txt"
$LogRoot = Join-Path $StateRoot "logs"
$LogFile = Join-Path $LogRoot ("launcher-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$RequirementsFile = Join-Path $SearchRoot "windows\requirements-3d-launcher.txt"
$CompatibilityWrapper = Join-Path $SearchRoot "windows\run_piris_3d_windows.py"
$PythonVersion = "3.12.10"
$LauncherMutex = [System.Threading.Mutex]::new($false, "Local\PirisLabs.ThreeDLauncher.Bootstrap")
$MutexAcquired = $false
try {
    $MutexAcquired = $LauncherMutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $MutexAcquired = $true
}
if (-not $MutexAcquired) {
    Write-Host "The Piris 3D launcher is already running in another window." -ForegroundColor Yellow
    exit 4
}

New-Item -ItemType Directory -Force -Path $StateRoot, $LogRoot | Out-Null
$TranscriptStarted = $false
try {
    Start-Transcript -Path $LogFile -Append | Out-Null
    $TranscriptStarted = $true
} catch {
    Write-Warning "The launcher log could not be started: $($_.Exception.Message)"
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("[Piris 3D] {0}" -f $Message) -ForegroundColor Cyan
}

function Test-LauncherRoot {
    param([string]$Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $false
    }
    return Test-Path -LiteralPath (
        Join-Path $Candidate "Requirements\Scripts\launch_3d_simulations.py"
    )
}

function Select-LauncherRoot {
    param([string]$RequestedRoot)
    $Candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $Candidates.Add($RequestedRoot)
    }
    if (Test-Path -LiteralPath $SavedLauncherRoot) {
        $Candidates.Add((Get-Content -LiteralPath $SavedLauncherRoot -Raw).Trim())
    }
    $Candidates.Add($SearchRoot)
    $Candidates.Add((Join-Path $SearchRoot "Piris 3D Launcher"))
    $Candidates.Add((Join-Path (Split-Path -Parent $SearchRoot) "Piris 3D Launcher"))
    $Candidates.Add((Join-Path $env:USERPROFILE "Downloads\Piris 3D Launcher"))
    $Candidates.Add((Join-Path $env:USERPROFILE "Desktop\Piris 3D Launcher"))

    foreach ($Candidate in $Candidates) {
        if (Test-LauncherRoot $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }

    Add-Type -AssemblyName System.Windows.Forms
    $Dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $Dialog.Description = "Select the confidential Piris 3D Launcher folder containing Requirements."
    $Dialog.ShowNewFolderButton = $false
    if ($Dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK -and
        (Test-LauncherRoot $Dialog.SelectedPath)) {
        return (Resolve-Path -LiteralPath $Dialog.SelectedPath).Path
    }
    throw "The Piris 3D Launcher folder was not selected. Download and extract the private company launcher, then try again."
}

function Find-CompatiblePython {
    $Candidates = New-Object System.Collections.Generic.List[object]
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        foreach ($Selector in @("-3.13", "-3.12", "-3.11", "-3.10")) {
            $Candidates.Add([pscustomobject]@{
                Executable = $PyLauncher.Source
                Prefix = @($Selector)
                Label = "py $Selector"
            })
        }
    }
    foreach ($Name in @("python.exe", "python3.exe")) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($null -ne $Command) {
            if ($Command.Source -match "\\WindowsApps\\") {
                continue
            }
            $Candidates.Add([pscustomobject]@{
                Executable = $Command.Source
                Prefix = @()
                Label = $Command.Source
            })
        }
    }
    foreach ($ExplicitPath in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")
    )) {
        if (Test-Path -LiteralPath $ExplicitPath) {
            $Candidates.Add([pscustomobject]@{
                Executable = $ExplicitPath
                Prefix = @()
                Label = $ExplicitPath
            })
        }
    }
    foreach ($Candidate in $Candidates) {
        try {
            $Arguments = @($Candidate.Prefix) + @(
                "-c",
                "import platform,struct,sys; print('%d.%d' % sys.version_info[:2]); raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64','x86_64') else 9)"
            )
            $Version = & $Candidate.Executable @Arguments 2>$null | Select-Object -Last 1
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$Version)) {
                $Candidate | Add-Member -NotePropertyName Version -NotePropertyValue ([string]$Version) -Force
                return $Candidate
            }
        } catch { }
    }
    return $null
}

function Install-CompatiblePython {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -ne $Winget) {
        Write-Step "Python is missing. Installing Python 3.12 for the current user..."
        & $Winget.Source install --exact --id Python.Python.3.12 --source winget --scope user --architecture x64 --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            if ($null -ne (Find-CompatiblePython)) {
                return
            }
            Write-Warning "Windows Package Manager completed, but x64 Python is still unavailable to this process. Trying the signed installer."
        } else {
            Write-Warning "Windows Package Manager returned exit code $LASTEXITCODE."
        }
    }

    Write-Step "Downloading the signed Python installer from python.org..."
    # Use x64 Python so every pinned launcher dependency has a Windows wheel,
    # including when Windows-on-ARM provides x64 emulation.
    $InstallerSuffix = "amd64.exe"
    $InstallerName = "python-$PythonVersion-$InstallerSuffix"
    $InstallerPath = Join-Path $env:TEMP $InstallerName
    $InstallerUri = "https://www.python.org/ftp/python/$PythonVersion/$InstallerName"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $InstallerUri -OutFile $InstallerPath
        $Signature = Get-AuthenticodeSignature -LiteralPath $InstallerPath
        $Signer = if ($null -ne $Signature.SignerCertificate) {
            [string]$Signature.SignerCertificate.Subject
        } else { "" }
        $SignatureIsValid = (
            $Signature.Status -eq [System.Management.Automation.SignatureStatus]::Valid -and
            $Signer -match "Python Software Foundation"
        )
        if (-not $SignatureIsValid) {
            throw "The downloaded Python installer does not have a valid Python Software Foundation signature."
        }
        $Process = Start-Process -FilePath $InstallerPath -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1",
            "Include_test=0", "Shortcuts=0"
        ) -Wait -PassThru
        if ($Process.ExitCode -ne 0) {
            throw "The official Python installer returned exit code $($Process.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $InstallerPath -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-OpenSshClient {
    $OpenSshPath = Join-Path $env:WINDIR "System32\OpenSSH"
    if (Test-Path -LiteralPath $OpenSshPath) {
        $env:PATH = $OpenSshPath + ";" + $env:PATH
    }
    # Windows PowerShell 5.1 collapses a zero/one-item pipeline result to
    # $null/a scalar.  The outer @() guarantees Count is always available.
    $Missing = @(
        @("ssh.exe", "scp.exe", "ssh-keygen.exe") | Where-Object {
            $null -eq (Get-Command $_ -ErrorAction SilentlyContinue)
        }
    )
    if ($Missing.Count -eq 0) {
        return
    }

    Write-Step "Windows OpenSSH Client is missing. Approve the Windows administrator prompt to install it..."
    $CapabilityCommand = "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0 | Out-Null"
    $Process = Start-Process -FilePath powershell.exe -Verb RunAs -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $CapabilityCommand
    ) -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "Windows OpenSSH Client could not be installed (exit code $($Process.ExitCode))."
    }
    $env:PATH = $OpenSshPath + ";" + $env:PATH
    foreach ($Command in @("ssh.exe", "scp.exe", "ssh-keygen.exe")) {
        if ($null -eq (Get-Command $Command -ErrorAction SilentlyContinue)) {
            throw "$Command is still unavailable after installing Windows OpenSSH Client."
        }
    }
}

function Test-LauncherEnvironment {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        return $false
    }
    $HealthCode = "import google.oauth2, googleapiclient, platform, psutil, struct, sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64','x86_64') else 9)"
    & $VenvPython -c $HealthCode 2>$null
    return ($LASTEXITCODE -eq 0)
}

try {
    if (-not (Test-Path -LiteralPath $RequirementsFile)) {
        throw "The private-launcher requirements file is missing: $RequirementsFile"
    }
    if (-not (Test-Path -LiteralPath $CompatibilityWrapper)) {
        throw "The Windows compatibility file is missing: $CompatibilityWrapper"
    }
    $LauncherRoot = Select-LauncherRoot $LauncherRoot
    Set-Content -LiteralPath $SavedLauncherRoot -Value $LauncherRoot -Encoding UTF8
    $LauncherScript = Join-Path $LauncherRoot "Requirements\Scripts\launch_3d_simulations.py"
    $ScriptsRoot = Split-Path -Parent $LauncherScript
    Write-Host ("Private launcher: {0}" -f $LauncherRoot)

    Ensure-OpenSshClient
    $Python = Find-CompatiblePython
    if ($null -eq $Python) {
        Install-CompatiblePython
        $Python = Find-CompatiblePython
    }
    if ($null -eq $Python) {
        throw "A compatible 64-bit Python 3.10-3.13 installation could not be found."
    }
    Write-Host ("Using Python {0}: {1}" -f $Python.Version, $Python.Label)

    $RequirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsFile).Hash
    $RecordedHash = if (Test-Path -LiteralPath $ReadyMarker) {
        (Get-Content -LiteralPath $ReadyMarker -Raw).Trim()
    } else { "" }
    $EnvironmentReady = $false
    if ($RecordedHash -eq $RequirementsHash) {
        $EnvironmentReady = Test-LauncherEnvironment
    }
    if (-not $EnvironmentReady) {
        Write-Step "Preparing the private 3D launcher environment..."
        if ((Test-Path -LiteralPath $VenvPython) -and -not (Test-LauncherEnvironment)) {
            Write-Host "Replacing an incompatible or incomplete launcher environment..."
            Remove-Item -LiteralPath $VenvRoot -Recurse -Force
        }
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            $CreateArguments = @($Python.Prefix) + @("-m", "venv", $VenvRoot)
            & $Python.Executable @CreateArguments
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
                throw "Python could not create the 3D launcher environment."
            }
        }
        & $VenvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw "pip could not be initialized." }
        & $VenvPython -m pip install --disable-pip-version-check --upgrade pip wheel
        if ($LASTEXITCODE -ne 0) { throw "pip could not update its installation tools." }
        & $VenvPython -m pip install --disable-pip-version-check --only-binary=:all: --upgrade --requirement $RequirementsFile
        if ($LASTEXITCODE -ne 0) { throw "The Google Drive launcher packages could not be installed." }
        if (-not (Test-LauncherEnvironment)) {
            throw "The Google Drive and Windows watchdog packages could not be imported by a compatible x64 Python."
        }
        Set-Content -LiteralPath $ReadyMarker -Value $RequirementsHash -Encoding ASCII
    } else {
        Write-Host "The private 3D launcher environment is already ready; setup was skipped." -ForegroundColor Green
    }

    Write-Step "Starting the same Lambda/Google Drive workflow used by the Mac launcher..."
    if ($TranscriptStarted) {
        # The private workflow prints a short-lived Jupyter URL token. Keep it
        # visible in this console but never persist it in a bootstrap log.
        try {
            Stop-Transcript | Out-Null
        } catch {
            throw "The secure setup log could not be closed before Jupyter startup. No node session was opened."
        }
        $TranscriptStarted = $false
    }
    $env:PYTHONUTF8 = "1"
    Push-Location $ScriptsRoot
    try {
        & $VenvPython $CompatibilityWrapper --launcher-script $LauncherScript @LauncherArguments
        $LauncherExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    exit $LauncherExitCode
} catch {
    Write-Host ""
    Write-Host "Piris 3D Simulations could not start." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ("Launcher log: {0}" -f $LogFile)
    exit 1
} finally {
    if ($TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
    }
    if ($MutexAcquired) {
        try { $LauncherMutex.ReleaseMutex() } catch { }
    }
    $LauncherMutex.Dispose()
}
