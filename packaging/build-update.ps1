[CmdletBinding()]
param(
    [string]$Version = "1.0.0",
    [string]$BaseUrl = "https://github.com/cdtauman/song-voice-converter/releases/download"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Distribution = (Resolve-Path (Join-Path $RepoRoot "dist\SongVoice")).Path
$Output = Join-Path $PSScriptRoot "output"
$Stage = Join-Path $PSScriptRoot "build\update-payload"
$resolvedBuild = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "build")) + [IO.Path]::DirectorySeparatorChar
$resolvedStage = [IO.Path]::GetFullPath($Stage)
if (-not $resolvedStage.StartsWith($resolvedBuild, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing update staging outside packaging/build"
}
if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage, $Output -Force | Out-Null

# SongVoiceLauncher.exe is the stable transaction host and is updated only by
# the installer. Everything else can be replaced before SongVoice.exe starts.
Get-ChildItem -LiteralPath $Distribution | Where-Object Name -NE "SongVoiceLauncher.exe" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $Stage -Recurse -Force
}
$archiveName = "SongVoice-$Version-Update.zip"
$archive = Join-Path $Output $archiveName
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item -LiteralPath $archive).Length
$manifest = [ordered]@{
    version = $Version
    url = "$BaseUrl/v$Version/$archiveName"
    sha256 = $hash
    size_bytes = $size
    notes_he = "עדכון SongVoice $Version"
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "update.json") -Encoding utf8
Write-Host "Update payload: $archiveName ($hash)"
