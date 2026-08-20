[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
if (-not $env:WINDOWS_CERT_BASE64 -or -not $env:WINDOWS_CERT_PASSWORD) {
    throw "WINDOWS_CERT_BASE64 and WINDOWS_CERT_PASSWORD are required"
}
$certificate = Join-Path $env:RUNNER_TEMP "songvoice-signing.pfx"
[IO.File]::WriteAllBytes($certificate, [Convert]::FromBase64String($env:WINDOWS_CERT_BASE64))
try {
    $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe |
        Where-Object FullName -Match "\\x64\\" | Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $signtool) { throw "signtool.exe was not found" }
    Get-ChildItem (Join-Path $PSScriptRoot "output\*.exe") | ForEach-Object {
        & $signtool.FullName sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com /f $certificate /p $env:WINDOWS_CERT_PASSWORD $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "Signing failed: $($_.Name)" }
        & $signtool.FullName verify /pa $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "Signature verification failed: $($_.Name)" }
    }
} finally {
    Remove-Item -LiteralPath $certificate -Force -ErrorAction SilentlyContinue
}
