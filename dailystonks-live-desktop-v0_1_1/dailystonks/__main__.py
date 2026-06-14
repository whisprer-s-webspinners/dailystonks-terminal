import argparse
from pathlib import Path
from .core.runner import run_report

def main():
    ap = argparse.ArgumentParser(description="DailyStonks modular report generator")
    ap.add_argument("--tier", default="black", choices=["free","basic","pro","black"])
    ap.add_argument("--out", default="out/report.html")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--interval", default="1d", help="yfinance interval (1d, 1h, 15m, ...)")
    ap.add_argument("--universe", default="sp500", choices=["sp500"])
    ap.add_argument("--max-universe", type=int, default=80, help="cap universe size for screeners")
    ap.add_argument("--tickers", default="SPY,QQQ", help="comma-separated tickers for spotlight charts")
    ap.add_argument("--seed", type=int, default=0, help="0 => deterministic by date")
    ap.add_argument("--overrides", default=None, help="JSON dict of slot->card_key overrides")
    ap.add_argument("--offline-synth", action="store_true", help="Use synthetic OHLCV (offline smoke test)")
    args = ap.parse_args()

    overrides = {}
    if args.overrides:
        import json
        overrides = json.loads(args.overrides)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_report(
        tier=args.tier,
        out_path=str(out_path),
        start=args.start,
        end=args.end,
        interval=args.interval,
        universe=args.universe,
        max_universe=args.max_universe,
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        seed=args.seed,
        overrides=overrides,
        offline_synth=args.offline_synth,
    )

if __name__ == "__main__":
    main()
