# DailyStonks Live Desktop

Standalone local DailyStonks market viewer.

This package is intentionally separated from the original DailyStonks email subscription/server project. It contains only the parts required to generate and view local reports/charts:

- card registry and all card/chart modules
- tier/slot config
- local market-universe CSV
- universe refresh/import tool
- yfinance-backed market data provider
- HTML renderer and plotting utilities
- live terminal/browser dashboard
- simple offline smoke-test mode

It does **not** contain the subscription database, email delivery runner, payment hooks, Docker web service, admin server, or server-side subscriber machinery.

## Windows install

From PowerShell inside this folder:

```powershell
.\install_windows.ps1
```

Then start the live terminal:

```powershell
.\run_windows.ps1 --tier black --tickers SPY,QQQ,AAPL,MSFT --interval 1d
```

Or, after install:

```powershell
.\.venv\Scripts\Activate.ps1
stonks-live --tier black --tickers SPY,QQQ,AAPL,MSFT --interval 1d
```

## Linux/macOS install

```bash
chmod +x install_linux.sh run_linux.sh scripts/run_live_terminal.sh
./install_linux.sh
./run_linux.sh --tier black --tickers SPY,QQQ,AAPL,MSFT --interval 1d
```

## Refresh or import the local universe CSV

The live app reads:

```text
data/sp500_constituents.csv
```

You can rebuild that file from a public S&P 500 table:

```powershell
.\.venv\Scripts\Activate.ps1
dailystonks-refresh-universe --source sp500
```

Preview first without writing:

```powershell
dailystonks-refresh-universe --source sp500 --dry-run
```

Import your original big DailyStonks CSV instead:

```powershell
dailystonks-refresh-universe --source custom --input "D:\path\to\your\big_universe.csv"
```

Build a broader US-listed universe from NasdaqTrader symbol directory files:

```powershell
dailystonks-refresh-universe --source us-listed
```

Keep ETFs when using NasdaqTrader sources:

```powershell
dailystonks-refresh-universe --source us-listed --include-etfs
```

Every write makes a timestamped backup of the previous CSV unless you pass:

```powershell
dailystonks-refresh-universe --source sp500 --no-backup
```

The canonical CSV columns are:

```text
Symbol,Name,Sector,Industry,Exchange,ETF,Source,DateAdded,CIK
```

Custom imports accept common alternatives such as `Ticker`, `Company`, `Security`, `GICS Sector`, and `GICS Sub-Industry`.

## Live terminal commands

At the `stonks>` prompt:

```text
open report
open slot S06
open slot S06 reversal.magic_full_chart
open card price.candles_enhanced
slots
cards risk
cards heavy
set tickers SPY,QQQ,NVDA,TSLA
set interval 5m
set refresh auto
views
refresh all
stop all
exit
```

## Offline smoke test

This checks that the local code works without yfinance/network access:

```powershell
.\run_windows.ps1 --offline-synth --no-browser --once report
```

or:

```bash
./run_linux.sh --offline-synth --no-browser --once report
```

The output HTML appears under `out/live_terminal/`.

## What to copy to another device

Copy this whole folder:

```text
DailyStonks Live Desktop/
  dailystonks/
  config/
  data/
  scripts/
  requirements.txt
  pyproject.toml
  install_windows.ps1
  run_windows.ps1
  install_linux.sh
  run_linux.sh
  README.md
```

Do not copy the original `dailystonks-delivery/` folder unless you intentionally want the online email/subscription service.

## Notes

- This is a local dashboard. It uses your browser as the display surface and a terminal as the control surface.
- The browser pages auto-refresh on the cadence implied by `--interval`, or by `--refresh-seconds` if supplied.
- yfinance is not a tick-level live feed. It updates according to Yahoo/yfinance candle availability and rate limits.
- Keep this in its own virtual environment so it cannot collide with the original DailyStonks server repo.
- The public S&P 500 refresh uses Wikipedia's constituents table. That is practical and free, but not an official S&P Global licensed constituent feed.
- The broad US-listed refresh uses NasdaqTrader symbol directory files. Sector is set to `Unknown` for those rows unless you import/enrich your own data.
