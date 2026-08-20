[CmdletBinding()]
param([Parameter(Mandatory)][string]$Distribution)

$ErrorActionPreference = "Stop"
$Distribution = (Resolve-Path -LiteralPath $Distribution).Path
$app = Join-Path $Distribution "SongVoice.exe"
$launcher = Join-Path $Distribution "SongVoiceLauncher.exe"
$ffmpeg = Join-Path $Distribution "_internal\runtime\ffmpeg\bin\ffmpeg.exe"
$licenseBundle = Join-Path $Distribution "_internal\licenses\torch-third-party-licenses.zip"
foreach ($required in $app, $launcher, $ffmpeg, $licenseBundle) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing: $required" }
}
$bad = Get-ChildItem -LiteralPath $Distribution -Recurse | Where-Object {
    $_.FullName -match "env-bench|benchmark[\\/]runtimes|seed-vc|ddsp-svc"
}
if ($bad) { throw "Benchmark runtime leaked into the distribution: $($bad[0].FullName)" }
$version = & $ffmpeg -version 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $version -match "--enable-gpl" -or $version -match "--enable-nonfree") { throw "Distributed ffmpeg failed LGPL gate" }

$smokeRoot = Join-Path $env:TEMP ("songvoice-smoke-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $smokeRoot | Out-Null
try {
    $env:SONGVOICE_HOME = $smokeRoot
    $process = Start-Process -FilePath $app -ArgumentList "--smoke-test" -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(30000)) { $process.Kill(); throw "Packaged GUI smoke test timed out" }
    if ($process.ExitCode -ne 0) { throw "Packaged GUI smoke test failed: $($process.ExitCode)" }
} finally {
    Remove-Item Env:\SONGVOICE_HOME -ErrorAction SilentlyContinue
    $resolvedTemp = [IO.Path]::GetFullPath($smokeRoot)
    $resolvedBase = [IO.Path]::GetFullPath($env:TEMP) + [IO.Path]::DirectorySeparatorChar
    if ($resolvedTemp.StartsWith($resolvedBase, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
$checksums = Get-ChildItem -LiteralPath $Distribution -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relative = [IO.Path]::GetRelativePath($Distribution, $_.FullName).Replace("\", "/")
    "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
}
$checksums | Set-Content -LiteralPath (Join-Path $Distribution "SHA256SUMS.txt") -Encoding ascii
Write-Host "Distribution validated: $Distribution"
