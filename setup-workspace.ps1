[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
# Keep every workspace process in UTF-8 mode; the editable .pth itself is made
# code-page-safe by Set-SongVoiceEditablePathFile below.
$env:PYTHONUTF8 = "1"

function Set-SongVoiceEditablePathFile {
    param([switch]$Required)

    $editablePth = Join-Path $repoRoot ".venv\Lib\site-packages\_editable_impl_songvoice.pth"
    if (-not (Test-Path -LiteralPath $editablePth -PathType Leaf)) {
        if ($Required) { throw "SongVoice editable-install path file is missing" }
        return
    }
    $sourcePath = (Join-Path $repoRoot "src").Replace("\", "/")
    $escapedChars = foreach ($character in $sourcePath.ToCharArray()) {
        $codePoint = [int][char]$character
        if ($codePoint -gt 127) { "\u{0:x4}" -f $codePoint }
        elseif ($character -eq '"') { '\"' }
        else { [string]$character }
    }
    $asciiSource = $escapedChars -join ""
    $pthLine = "import sys; sys.path.insert(0, `"$asciiSource`")`n"
    [IO.File]::WriteAllText($editablePth, $pthLine, [Text.UTF8Encoding]::new($false))
}

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        uv venv --python 3.11 .venv
        if ($LASTEXITCODE -ne 0) { throw "Unable to create the Python 3.11 workspace" }
    }
    # -S bypasses site-packages so setup can repair an older non-ASCII .pth.
    $pythonVersion = & $venvPython -S -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pythonVersion -ne "3.11") {
        throw "SongVoice requires Python 3.11; .venv uses $pythonVersion"
    }
    # Repair a previous editable install before uv inspects the interpreter.
    Set-SongVoiceEditablePathFile
    uv pip install --python $venvPython --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -c constraints.txt -r rvc-requirements.lock -e ".[dev,gui,rvc,training]"
    if ($LASTEXITCODE -ne 0) { throw "Unable to install the locked SongVoice runtime" }
    # Python 3.11 decodes .pth files with the Windows ANSI code page.  Hatch's
    # editable file is UTF-8 and therefore breaks before Python can start when
    # the repository path contains Hebrew.  Replace it with an ASCII-only
    # import line whose Unicode path is represented with \u escapes.
    Set-SongVoiceEditablePathFile -Required
    uv pip check --python $venvPython
    if ($LASTEXITCODE -ne 0) { throw "The SongVoice workspace has incompatible packages" }
    & (Join-Path $repoRoot "packaging\fetch-dependencies.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Unable to provision the verified FFmpeg runtime" }
    Write-Host "SongVoice workspace is ready. Run .\run-workspace.ps1"
}
finally {
    Pop-Location
}
