param(
    [Parameter(Mandatory = $true)]
    [string]$SourceEngineRoot,

    [Parameter(Mandatory = $true)]
    [string]$SourceDataRoot,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot
)

$ErrorActionPreference = "Stop"
$SourceEngineRoot = (Resolve-Path $SourceEngineRoot).Path
$SourceDataRoot = (Resolve-Path $SourceDataRoot).Path
New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

$items = @("dailystonks", "config", "requirements.txt", "pyproject.toml")
foreach ($item in $items) {
    $src = Join-Path $SourceEngineRoot $item
    $dst = Join-Path $DestinationRoot $item
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse -Force $src $dst
}

$dstData = Join-Path $DestinationRoot "data"
if (Test-Path $dstData) { Remove-Item -Recurse -Force $dstData }
Copy-Item -Recurse -Force $SourceDataRoot $dstData

New-Item -ItemType Directory -Force -Path (Join-Path $DestinationRoot "scripts") | Out-Null
foreach ($script in @("run_live_terminal.py", "run_live_terminal.ps1", "run_live_terminal.sh", "run_report.py", "smoke_test_offline.py", "run_black_live.ps1", "run_black_offline.ps1")) {
    Copy-Item -Force (Join-Path $SourceEngineRoot "scripts\$script") (Join-Path $DestinationRoot "scripts\$script")
}

Write-Host "Split complete: $DestinationRoot" -ForegroundColor Green
Write-Host "Install there with: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe -m pip install -e ."
