param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsForDailyStonks
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Local venv not found; creating/installing first..." -ForegroundColor Yellow
    & "$Root\install_windows.ps1"
}

& $Py -m dailystonks.live_terminal @ArgsForDailyStonks
