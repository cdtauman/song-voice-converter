[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $PSScriptRoot "build"
$DownloadRoot = Join-Path $BuildRoot "downloads"
$DependencyRoot = Join-Path $BuildRoot "dependencies"

function Assert-UnderBuild([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetFullPath($BuildRoot) + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing operation outside packaging/build: $full"
    }
}

New-Item -ItemType Directory -Force -Path $DownloadRoot, $DependencyRoot | Out-Null
$manifest = Get-Content (Join-Path $PSScriptRoot "dependencies.json") -Raw | ConvertFrom-Json
$ffmpeg = $manifest.ffmpeg
$archive = Join-Path $DownloadRoot "ffmpeg-win64-lgpl-shared.zip"
if ($Force -or -not (Test-Path -LiteralPath $archive)) {
    $temporary = "$archive.part"
    Assert-UnderBuild $temporary
    Invoke-WebRequest -Uri $ffmpeg.url -OutFile $temporary -Headers @{
        Accept = "application/octet-stream"
        "User-Agent" = "SongVoice-Packager"
    }
    Move-Item -LiteralPath $temporary -Destination $archive -Force
}
$actualSize = (Get-Item -LiteralPath $archive).Length
$actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSize -ne [long]$ffmpeg.size_bytes -or $actualHash -ne $ffmpeg.sha256) {
    throw "ffmpeg archive checksum/size mismatch"
}

$extract = Join-Path $DependencyRoot "ffmpeg-extract"
$target = Join-Path $DependencyRoot "ffmpeg"
Assert-UnderBuild $extract
Assert-UnderBuild $target
if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
New-Item -ItemType Directory -Path $extract, $target | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $extract
$binary = Get-ChildItem -LiteralPath $extract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
if (-not $binary) { throw "ffmpeg.exe missing from verified archive" }
$distribution = Split-Path (Split-Path $binary.FullName -Parent) -Parent
Get-ChildItem -LiteralPath $distribution | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
}

$version = & (Join-Path $target "bin\ffmpeg.exe") -version 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $version -match "--enable-gpl" -or $version -match "--enable-nonfree") {
    throw "The selected ffmpeg build is not an LGPL-only build"
}
$filters = & (Join-Path $target "bin\ffmpeg.exe") -hide_banner -filters 2>&1 | Out-String
foreach ($required in $ffmpeg.required_filters) {
    if ($filters -notmatch [regex]::Escape([string]$required)) {
        throw "The LGPL ffmpeg build is missing filter: $required"
    }
}
$evidence = [ordered]@{
    schema = 1
    source = $ffmpeg.url
    archive_sha256 = $actualHash
    archive_size_bytes = $actualSize
    variant = $ffmpeg.variant
    gpl_enabled = $false
    verified_at_utc = [DateTime]::UtcNow.ToString("o")
}
$evidence | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target "songvoice-verification.json") -Encoding utf8
Write-Host "Verified LGPL ffmpeg: $actualHash"
