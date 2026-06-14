import argparse
from pathlib import Path
from dailystonks.core.runner import run_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="black")
    ap.add_argument("--out", default="out/report.html")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--max-universe", type=int, default=60)
    ap.add_argument("--tickers", default="SPY,QQQ,AAPL,MSFT")
    ap.add_argument("--offline-synth", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_report(
        tier=args.tier,
        out_path=str(out),
        start=args.start,
        end=args.end,
        interval=args.interval,
        universe=args.universe,
        max_universe=args.max_universe,
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        seed=0,
        overrides={},
        offline_synth=args.offline_synth,
    )

if __name__ == "__main__":
    main()
