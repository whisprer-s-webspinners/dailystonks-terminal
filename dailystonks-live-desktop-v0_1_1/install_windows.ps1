$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.10+ first."
}

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .

Write-Host ""
Write-Host "DailyStonks Live Desktop installed." -ForegroundColor Green
Write-Host "Run it with: .\run_windows.ps1 --tier black --tickers SPY,QQQ,AAPL,MSFT --interval 1d"
