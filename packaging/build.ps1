[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Version = "",
    [switch]$Offline,
    [switch]$SkipBuildRequirements,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path (Join-Path $RepoRoot $Python)).Path
if (-not $Version) {
    $match = Select-String -LiteralPath (Join-Path $RepoRoot "pyproject.toml") -Pattern '^version = "([0-9]+\.[0-9]+\.[0-9]+)"$' | Select-Object -First 1
    if (-not $match) { throw "Unable to read project version from pyproject.toml" }
    $Version = $match.Matches[0].Groups[1].Value
}
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "Invalid release version: $Version" }
& (Join-Path $PSScriptRoot "fetch-dependencies.ps1")
if (-not $SkipBuildRequirements) {
    uv pip install --python $PythonPath -r (Join-Path $PSScriptRoot "build-requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Unable to install packaging tools" }
} else {
    & $PythonPath -c "import PyInstaller"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller is missing from the selected Python" }
}

Push-Location $RepoRoot
try {
    & $PythonPath -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "songvoice.spec")
    if ($LASTEXITCODE -ne 0) { throw "SongVoice PyInstaller build failed" }
    & $PythonPath (Join-Path $PSScriptRoot "bundle-third-party-licenses.py") --distribution (Join-Path $RepoRoot "dist\SongVoice")
    if ($LASTEXITCODE -ne 0) { throw "Third-party license bundle failed" }
    & $PythonPath -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "launcher.spec")
    if ($LASTEXITCODE -ne 0) { throw "Launcher PyInstaller build failed" }
} finally {
    Pop-Location
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "dist\SongVoiceLauncher.exe") -Destination (Join-Path $RepoRoot "dist\SongVoice\SongVoiceLauncher.exe") -Force

& (Join-Path $PSScriptRoot "validate-dist.ps1") -Distribution (Join-Path $RepoRoot "dist\SongVoice")
if ($SkipInstaller) { exit 0 }

$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $candidate = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    if (Test-Path -LiteralPath $candidate) { $iscc = $candidate }
}
if (-not $iscc) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\InnoSetup6\ISCC.exe"
    if (Test-Path -LiteralPath $candidate) { $iscc = $candidate }
}
if (-not $iscc) { throw "Inno Setup 6 is required to build the installer" }
if ($Offline) {
    & $iscc "/DMyAppVersion=$Version" "/DOfflineBuild=1" (Join-Path $PSScriptRoot "SongVoice.iss")
} else {
    & $iscc "/DMyAppVersion=$Version" (Join-Path $PSScriptRoot "SongVoice.iss")
}
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
