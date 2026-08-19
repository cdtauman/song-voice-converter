[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Installer,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$Installer = (Resolve-Path -LiteralPath $Installer).Path
$expectedName = if ($Offline) { "Offline-Setup" } else { "-Setup" }
if ([IO.Path]::GetFileNameWithoutExtension($Installer) -notlike "*$expectedName*") {
    throw "Installer kind does not match requested clean-machine flow"
}
$install = Join-Path $env:LOCALAPPDATA "Programs\SongVoice"
$data = Join-Path $env:LOCALAPPDATA "SongVoice"
$log = Join-Path $env:TEMP "songvoice-clean-install.log"

$proc = Start-Process -FilePath $Installer -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=$log" -Wait -PassThru -WindowStyle Hidden
if ($proc.ExitCode -ne 0) { throw "Installer failed: $($proc.ExitCode). See $log" }
foreach ($file in "SongVoice.exe", "SongVoiceLauncher.exe", "_internal\runtime\ffmpeg\bin\ffmpeg.exe") {
    if (-not (Test-Path -LiteralPath (Join-Path $install $file) -PathType Leaf)) {
        throw "Clean install is missing $file"
    }
}
if ($Offline -and -not (Test-Path -LiteralPath (Join-Path $data "models\rmvpe.pt"))) {
    throw "Offline installer did not install its verified model payload"
}
$env:SONGVOICE_HOME = Join-Path $env:TEMP ("songvoice-clean-home-" + [guid]::NewGuid().ToString("N"))
try {
    $app = Start-Process -FilePath (Join-Path $install "SongVoice.exe") -ArgumentList "--smoke-test" -Wait -PassThru -WindowStyle Hidden
    if ($app.ExitCode -ne 0) { throw "Installed GUI smoke test failed: $($app.ExitCode)" }
} finally {
    $smoke = $env:SONGVOICE_HOME
    Remove-Item Env:\SONGVOICE_HOME -ErrorAction SilentlyContinue
    $tempBase = [IO.Path]::GetFullPath($env:TEMP) + [IO.Path]::DirectorySeparatorChar
    $resolved = [IO.Path]::GetFullPath($smoke)
    if ($resolved.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
    }
}
$uninstaller = Join-Path $install "unins000.exe"
$uninstall = Start-Process -FilePath $uninstaller -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru -WindowStyle Hidden
if ($uninstall.ExitCode -ne 0) { throw "Uninstaller failed: $($uninstall.ExitCode)" }
if (Test-Path -LiteralPath $install) { throw "Uninstaller left the application directory behind" }
if (Test-Path -LiteralPath $data) { throw "Uninstaller left SongVoice user data behind" }
Write-Host "Clean-machine install, smoke, and uninstall flow passed."
