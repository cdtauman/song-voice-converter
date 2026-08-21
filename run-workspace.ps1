[CmdletBinding()]
param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$ffmpegBin = Join-Path $repoRoot "packaging\build\dependencies\ffmpeg\bin"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Workspace is not configured. Run .\setup-workspace.ps1 first."
}
& $venvPython -c "import PySide6, torch, audio_separator, svc_app, svc_engine"
if ($LASTEXITCODE -ne 0) {
    throw "Workspace engine dependencies are incomplete. Run .\setup-workspace.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $ffmpegBin "ffmpeg.exe") -PathType Leaf)) {
    throw "Verified FFmpeg is missing. Run .\setup-workspace.ps1 first."
}

$env:PATH = "$ffmpegBin$([IO.Path]::PathSeparator)$env:PATH"
$env:SONGVOICE_ENGINE_PYTHON = $venvPython
Push-Location $repoRoot
try {
    & $venvPython -c "from svc_app.engine_client import EngineClient; c = EngineClient(); c.ping('workspace-launch'); c.stop()"
    if ($LASTEXITCODE -ne 0) {
        throw "The workspace engine failed its startup check. Run .\setup-workspace.ps1 again."
    }
    $appArgs = @("-m", "svc_app.main")
    if ($SmokeTest) {
        $appArgs += "--smoke-test"
    }
    & $venvPython @appArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
