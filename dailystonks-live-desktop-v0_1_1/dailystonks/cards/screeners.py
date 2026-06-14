from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, rsi

def _universe_tickers(ctx: CardContext):
    return ctx.sp500.tickers(max_n=ctx.max_universe)

def _compute_row_from_df(tk: str, df: pd.DataFrame):
    df = df.iloc[-260:].copy()
    close = df["Close"]
    last = float(close.iloc[-1])
    r = float(rsi(close).iloc[-1])
    hi52 = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
    lo52 = float(close.rolling(252).min().iloc[-1]) if len(close) >= 252 else float(close.min())
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
    dist_hi = (last - hi52) / hi52 if hi52 else np.nan
    dist_ma200 = (last - ma200) / ma200 if ma200 else np.nan
    return {
        "Symbol": tk,
        "Close": last,
        "RSI": r,
        "%from52WHigh": dist_hi * 100,
        "%fromMA200": dist_ma200 * 100,
        "52WLow": lo52,
        "52WHigh": hi52,
    }

@register_card("screen.sp500_core_table", "S&P500 Core Screener Table", "screen", min_tier="free", cost=8, heavy=False, slots=("S01","S04","S05"))
def sp500_table(ctx: CardContext) -> CardResult:
    tickers = _universe_tickers(ctx)
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")

    rows = []
    failed = 0
    for raw in tickers:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            failed += 1
            continue
        try:
            rows.append(_compute_row_from_df(tk, df))
        except Exception:
            failed += 1

    if not rows:
        return CardResult(
            key="screen.sp500_core_table",
            title="S&P500 Screener Table",
            summary="No rows computed (data fetch failures).",
            warnings=[f"Failed symbols: {failed}/{len(tickers)}"]
        )

    df = pd.DataFrame(rows)
    df["RankScore"] = -df["%from52WHigh"].abs() - 0.5 * df["%fromMA200"].abs()
    df = df.sort_values("RankScore", ascending=False).head(25)

    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(f"S&P500 Screener (top 25 of first {len(tickers)} symbols)")
    show = df[["Symbol", "Close", "RSI", "%from52WHigh", "%fromMA200"]].round(2)
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
    table.scale(1, 1.4)
    png = fig_to_png_bytes(fig)

    bullets = [
        "Batched download used (more reliable than per-ticker calls).",
        "RankScore is heuristic; replace with your production scoring engine."
    ]
    warnings = []
    if failed:
        warnings.append(f"Skipped {failed} symbols due to missing/failed data.")

    return CardResult(
        key="screen.sp500_core_table",
        title="S&P500 Screener Table",
        summary="Top candidates from a capped S&P500 universe.",
        bullets=bullets,
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="sp500_table.png", payload=png)]
    )

# Leave your other screener cards as-is in your repo if present.
# If you need the batched versions for dip_finder / ML rank too, tell me and I’ll patch those next.