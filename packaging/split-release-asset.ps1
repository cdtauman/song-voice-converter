[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Path,
    [Int64]$MaximumPartBytes = 1900MB
)

$ErrorActionPreference = "Stop"

$source = Get-Item -LiteralPath $Path -ErrorAction Stop
if ($source.Length -le $MaximumPartBytes) {
    Write-Host "Release asset is within GitHub's per-file limit: $($source.Name)"
    exit 0
}

# GitHub Release assets must be smaller than 2 GiB.  Keep a margin so the
# workflow remains valid if GitHub changes how it measures the upload size.
$buffer = New-Object byte[] 1048576
$input = [IO.File]::OpenRead($source.FullName)
try {
    $partNumber = 1
    while ($input.Position -lt $input.Length) {
        $partPath = "{0}.part{1:D3}" -f $source.FullName, $partNumber
        if (Test-Path -LiteralPath $partPath) { Remove-Item -LiteralPath $partPath -Force }
        $output = [IO.File]::Create($partPath)
        try {
            [Int64]$written = 0
            while ($written -lt $MaximumPartBytes -and $input.Position -lt $input.Length) {
                $toRead = [Math]::Min([Int64]$buffer.Length, $MaximumPartBytes - $written)
                $read = $input.Read($buffer, 0, [int]$toRead)
                if ($read -le 0) { throw "Unexpected end of $($source.Name)" }
                $output.Write($buffer, 0, $read)
                $written += $read
            }
        }
        finally {
            $output.Dispose()
        }
        Write-Host "Created $(Split-Path -Leaf $partPath) ($written bytes)"
        $partNumber++
    }
}
finally {
    $input.Dispose()
}

Remove-Item -LiteralPath $source.FullName -Force
Write-Host "Replaced $($source.Name) with $($partNumber - 1) GitHub-compatible parts."
