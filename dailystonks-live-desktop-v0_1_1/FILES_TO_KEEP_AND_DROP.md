# DailyStonks standalone split map

## Keep for the local live desktop app

```text
dailystonks/            # Python package: cards, registry, market data, renderer, live terminal
config/                 # tiers.yaml and slots.yaml
data/                   # sp500_constituents.csv
dailystonks/data/universe_refresh.py  # refresh/import local universe CSV
scripts/run_live_terminal.*
scripts/run_report.py
scripts/smoke_test_offline.py
requirements.txt
pyproject.toml
install_windows.ps1
run_windows.ps1
install_linux.sh
run_linux.sh
README.md
```

## Drop from the portable viewer

```text
dailystonks-delivery/   # email/subscription/server/payment layer
app/db.py
app/models.py
app/delivery/
app/payments/
app/main.py
docker-compose.yml
server/admin/subscriber tests
```

The live desktop app does not need any subscriber DB, payment credentials, SMTP credentials, Docker container, API server, or online web app.

## Universe handling

The portable viewer only needs one active universe file: `data/sp500_constituents.csv`. Rebuild it with `dailystonks-refresh-universe --source sp500`, build a broad NasdaqTrader US-listed universe with `--source us-listed`, or import the original DailyStonks CSV with `--source custom --input <csv>`.
