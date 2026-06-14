$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo\..

$env:PYTHONPATH = (Get-Location)

python -m dailystonks `
  --tier black `
  --out out\report_black.html `
  --start 2024-01-01 `
  --max-universe 60 `
  --tickers SPY,QQQ,AAPL,MSFT,NVDA