[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AppRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$AppRoot = (Resolve-Path -LiteralPath $AppRoot).Path
$AppArchive = Join-Path $AppRoot "Max Layout.pyz"
$RequirementsFile = Join-Path $AppRoot "windows\requirements-windows.txt"
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "PirisLabs\MaxLayout"
$VenvRoot = Join-Path $RuntimeRoot "runtime"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$VenvPythonw = Join-Path $VenvRoot "Scripts\pythonw.exe"
$ReadyMarker = Join-Path $RuntimeRoot "requirements.sha256"
$LogRoot = Join-Path $RuntimeRoot "logs"
$LogFile = Join-Path $LogRoot ("launcher-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$PythonVersion = "3.12.10"
$PythonWingetId = "Python.Python.3.12"
$BootstrapMutex = New-Object System.Threading.Mutex($false, "Local\PirisLabs.MaxLayout.Bootstrap")
$MutexAcquired = $false
try {
    $MutexAcquired = $BootstrapMutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $MutexAcquired = $true
}
if (-not $MutexAcquired) {
    Write-Host "Max Layout setup is already running in another window." -ForegroundColor Yellow
    exit 4
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $LogRoot | Out-Null
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
    Write-Host ("[Max Layout] {0}" -f $Message) -ForegroundColor Cyan
}

function Invoke-CandidatePython {
    param(
        [Parameter(Mandatory = $true)]$Candidate,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $AllArguments = @($Candidate.Prefix) + $Arguments
    & $Candidate.Executable @AllArguments
}

function Find-CompatiblePython {
    $Candidates = New-Object System.Collections.Generic.List[object]
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        foreach ($Selector in @("-3.12", "-3.11", "-3.10", "-3.9")) {
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
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe")
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
            $VersionCode = "import platform,struct,sys; print('%d.%d' % sys.version_info[:2]); raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64','x86_64') else 9)"
            $Version = Invoke-CandidatePython $Candidate @("-c", $VersionCode) 2>$null | Select-Object -Last 1
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$Version)) {
                $Candidate | Add-Member -NotePropertyName Version -NotePropertyValue ([string]$Version) -Force
                return $Candidate
            }
        } catch {
            # Try the next candidate. Windows Store aliases commonly fail here.
        }
    }
    return $null
}

function Install-PythonWithWinget {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -eq $Winget) {
        return $false
    }
    Write-Step "Python was not found. Installing Python 3.12 for the current Windows user..."
    & $Winget.Source install --exact --id $PythonWingetId --source winget --scope user --architecture x64 --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Windows Package Manager returned exit code $LASTEXITCODE."
        return $false
    }
    return $true
}

function Install-PythonFromOfficialInstaller {
    Write-Step "Windows Package Manager is unavailable. Downloading the signed Python installer from python.org..."
    # Use x64 wheels on both x64 Windows and Windows-on-ARM emulation. The
    # pinned scientific packages do not currently ship a complete ARM64 set.
    $InstallerSuffix = "amd64.exe"
    $InstallerName = "python-$PythonVersion-$InstallerSuffix"
    $InstallerUri = "https://www.python.org/ftp/python/$PythonVersion/$InstallerName"
    $InstallerPath = Join-Path $env:TEMP $InstallerName
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $InstallerUri -OutFile $InstallerPath
        $Signature = Get-AuthenticodeSignature -LiteralPath $InstallerPath
        $Signer = if ($null -ne $Signature.SignerCertificate) {
            [string]$Signature.SignerCertificate.Subject
        } else {
            ""
        }
        $SignatureIsValid = (
            $Signature.Status -eq [System.Management.Automation.SignatureStatus]::Valid -and
            $Signer -match "Python Software Foundation"
        )
        if (-not $SignatureIsValid) {
            throw "The downloaded Python installer does not have a valid Python Software Foundation signature."
        }
        $Process = Start-Process -FilePath $InstallerPath -ArgumentList @(
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_launcher=1",
            "Include_test=0",
            "Shortcuts=0"
        ) -Wait -PassThru
        if ($Process.ExitCode -ne 0) {
            throw "The official Python installer returned exit code $($Process.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $InstallerPath -Force -ErrorAction SilentlyContinue
    }
}

function Test-MaxLayoutEnvironment {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        return $false
    }
    $HealthCode = "import PySide6, gdstk, numpy, platform, struct, sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64','x86_64') else 9)"
    & $VenvPython -c $HealthCode 2>$null
    return ($LASTEXITCODE -eq 0)
}

try {
    if (-not (Test-Path -LiteralPath $AppArchive)) {
        throw "Max Layout.pyz is missing from $AppRoot. Extract the complete Windows bundle first."
    }
    if (-not (Test-Path -LiteralPath $RequirementsFile)) {
        throw "The Windows requirements file is missing: $RequirementsFile"
    }

    Write-Step "Checking the first-run environment..."
    $Python = Find-CompatiblePython
    if ($null -eq $Python) {
        [void](Install-PythonWithWinget)
        $Python = Find-CompatiblePython
        if ($null -eq $Python) {
            Install-PythonFromOfficialInstaller
            $Python = Find-CompatiblePython
        }
    }
    if ($null -eq $Python) {
        throw "A compatible x64 Python 3.9-3.12 could not be installed or located. See $LogFile"
    }
    Write-Host ("Using Python {0}: {1}" -f $Python.Version, $Python.Label)

    $RequirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsFile).Hash
    $RecordedHash = if (Test-Path -LiteralPath $ReadyMarker) {
        (Get-Content -LiteralPath $ReadyMarker -Raw).Trim()
    } else {
        ""
    }
    $EnvironmentReady = (
        $RecordedHash -eq $RequirementsHash -and
        (Test-MaxLayoutEnvironment)
    )

    if (-not $EnvironmentReady) {
        Write-Step "Preparing the private Max Layout environment. This is required only on first launch or after a dependency update..."
        if ((Test-Path -LiteralPath $VenvPython) -and -not (Test-MaxLayoutEnvironment)) {
            Write-Host "Replacing an incompatible or incomplete Max Layout environment..."
            Remove-Item -LiteralPath $VenvRoot -Recurse -Force
        }
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            Invoke-CandidatePython $Python @("-m", "venv", $VenvRoot)
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
                throw "Python could not create the Max Layout environment at $VenvRoot"
            }
        }

        & $VenvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Python could not initialize pip in the Max Layout environment."
        }
        & $VenvPython -m pip install --disable-pip-version-check --upgrade pip wheel
        if ($LASTEXITCODE -ne 0) {
            throw "pip could not update its installation tools."
        }
        & $VenvPython -m pip install --disable-pip-version-check --only-binary=:all: --upgrade --requirement $RequirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "The Max Layout Python packages could not be installed."
        }
        if (-not (Test-MaxLayoutEnvironment)) {
            throw "The packages were installed but PySide6, NumPy, or gdstk could not be imported."
        }
        Set-Content -LiteralPath $ReadyMarker -Value $RequirementsHash -Encoding ASCII
        Write-Host "First-run setup completed successfully." -ForegroundColor Green
    } else {
        Write-Host "The Max Layout environment is already ready; setup was skipped." -ForegroundColor Green
    }

    Write-Step "Opening Max Layout..."
    $LaunchExecutable = if (Test-Path -LiteralPath $VenvPythonw) { $VenvPythonw } else { $VenvPython }
    $QuotedArchive = '"' + $AppArchive.Replace('"', '\"') + '"'
    $Process = Start-Process -FilePath $LaunchExecutable -ArgumentList $QuotedArchive -WorkingDirectory $AppRoot -PassThru
    Write-Host ("Max Layout started successfully (process {0})." -f $Process.Id) -ForegroundColor Green
    Write-Host ("Launcher log: {0}" -f $LogFile)
    exit 0
} catch {
    Write-Host ""
    Write-Host "Max Layout could not start." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ("Launcher log: {0}" -f $LogFile)
    exit 1
} finally {
    if ($TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
    }
    if ($MutexAcquired) {
        try { $BootstrapMutex.ReleaseMutex() } catch { }
    }
    $BootstrapMutex.Dispose()
}
