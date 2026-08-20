[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$FirstPart,
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$first = Get-Item -LiteralPath $FirstPart -ErrorAction Stop
if ($first.Name -notmatch '^(?<name>.+)\.part001$') {
    throw "FirstPart must be the .part001 file produced by split-release-asset.ps1"
}

$basePath = Join-Path $first.DirectoryName $matches.name
if (-not $Destination) { $Destination = $basePath }
$destinationPath = [IO.Path]::GetFullPath($Destination)
$directoryPath = [IO.Path]::GetFullPath($first.DirectoryName)
if (-not $destinationPath.StartsWith($directoryPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must remain next to the downloaded parts"
}

$parts = Get-ChildItem -LiteralPath $first.DirectoryName -File -Filter "$($matches.name).part*" |
    Sort-Object Name
if ($parts.Count -lt 2) { throw "At least two sequential parts are required" }
for ($index = 0; $index -lt $parts.Count; $index++) {
    $expected = "{0}.part{1:D3}" -f $matches.name, ($index + 1)
    if ($parts[$index].Name -ne $expected) { throw "Missing or misnamed part: $expected" }
}

$output = [IO.File]::Create($destinationPath)
try {
    foreach ($part in $parts) {
        $input = [IO.File]::OpenRead($part.FullName)
        try { $input.CopyTo($output) }
        finally { $input.Dispose() }
    }
}
finally {
    $output.Dispose()
}
Write-Host "Reassembled $($parts.Count) parts into $destinationPath"
