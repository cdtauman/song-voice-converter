param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("seed", "ddsp")]
    [string]$Engine
)

$ErrorActionPreference = "Stop"
$BenchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $BenchRoot
$Sources = Get-Content -LiteralPath (Join-Path $BenchRoot "sources.json") -Raw | ConvertFrom-Json
$Spec = $Sources.engines.$Engine
$Runtime = Join-Path $BenchRoot "runtimes\$Engine"
$Checkout = Join-Path $Runtime "source"
$Venv = Join-Path $Runtime ".venv"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to create an isolated benchmark runtime"
}
if ((Resolve-Path -LiteralPath $BenchRoot).Path -eq (Resolve-Path -LiteralPath $RepoRoot).Path) {
    throw "env-bench boundary resolution failed"
}
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
if (-not (Test-Path -LiteralPath $Checkout)) {
    git clone --filter=blob:none --no-checkout $Spec.repository $Checkout
}
git -C $Checkout fetch --depth 1 origin $Spec.commit
git -C $Checkout checkout --detach $Spec.commit
$Actual = (git -C $Checkout rev-parse HEAD).Trim()
if ($Actual -ne $Spec.commit) {
    throw "source pin mismatch for $Engine"
}
if (-not (Test-Path -LiteralPath $Venv)) {
    uv venv --python 3.10 $Venv
}
$Python = Join-Path $Venv "Scripts\python.exe"
uv pip install --python $Python -r (Join-Path $Checkout "requirements.txt")
uv pip check --python $Python

$Receipt = @{
    engine = $Engine
    repository = $Spec.repository
    commit = $Actual
    license = $Spec.license
    python = $Python
    core_constraints_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $RepoRoot "constraints.txt")).Hash.ToLower()
}
$Receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Runtime "receipt.json") -Encoding utf8
