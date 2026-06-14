
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

TRADING_DAYS = 252

def _spy_close(ctx: CardContext) -> pd.Series:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("SPY OHLCV missing/empty")
    return df["Close"].astype(float).dropna()

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

@register_card("stress.tail_risk_snapshot", "Stress: Tail Risk Snapshot (SPY)", "stress", min_tier="black", cost=7, heavy=False, slots=("S11",))
def tail_risk(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-9000:]
    r = close.pct_change().dropna()
    if len(r) < 400:
        return CardResult(
            key="stress.tail_risk_snapshot",
            title="Stress: Tail Risk Snapshot (SPY)",
            summary="Not enough data (need ~400+ daily returns)."
        )

    # Historical VaR/ES at 95/99
    q95 = float(np.quantile(r.values, 0.05))
    q99 = float(np.quantile(r.values, 0.01))
    es95 = float(r[r <= q95].mean())
    es99 = float(r[r <= q99].mean())

    worst = r.sort_values().head(12)
    tab = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in worst.index],
        "Return%": [f"{x*100:+.2f}" for x in worst.values],
    })
    png_tbl = _table_png("Worst Daily Returns (SPY)", tab)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.hist((r.values*100), bins=80)
    ax.set_title("SPY daily return distribution (%)")
    ax.grid(True, alpha=0.25, axis="y")
    png_hist = fig_to_png_bytes(fig)

    metrics = {
        "VaR95%": round(q95*100, 2),
        "ES95%": round(es95*100, 2),
        "VaR99%": round(q99*100, 2),
        "ES99%": round(es99*100, 2),
        "Obs": int(len(r)),
    }

    bullets = [
        "VaR/ES are historical (non-parametric) — descriptive, not predictive.",
        "Use alongside vol regime + event risk for context.",
    ]

    return CardResult(
        key="stress.tail_risk_snapshot",
        title="Stress: Tail Risk Snapshot (SPY)",
        summary="Historical tail loss stats + worst days list.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="tail_hist.png", payload=png_hist),
            Artifact(kind="image/png", name="worst_days.png", payload=png_tbl),
        ],
    )

@register_card("stress.drawdown_distribution", "Stress: Drawdown Distribution (SPY)", "stress", min_tier="pro", cost=6, heavy=False, slots=("S11",))
def drawdown_dist(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-9000:]
    if len(close) < 400:
        return CardResult(
            key="stress.drawdown_distribution",
            title="Stress: Drawdown Distribution (SPY)",
            summary="Not enough data (need ~400+ bars)."
        )

    eq = close / close.iloc[0]
    dd = (eq / eq.cummax()) - 1.0
    dd_depth = (-dd).values  # positive depth

    # Histogram of drawdown depths
    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.hist(dd_depth*100, bins=60)
    ax.set_title("Drawdown depth distribution (%, SPY)")
    ax.grid(True, alpha=0.25, axis="y")
    png_hist = fig_to_png_bytes(fig)

    # Time under water: proportion of days in drawdown and quantiles
    in_dd = float((dd < 0).mean()*100)
    maxdd = float(dd.min()*100)
    q50 = float(np.quantile(dd_depth, 0.50)*100)
    q90 = float(np.quantile(dd_depth, 0.90)*100)
    q99 = float(np.quantile(dd_depth, 0.99)*100)

    # Longest drawdown run (consecutive days dd<0)
    is_dd = (dd < 0).astype(int).values
    longest = cur = 0
    for v in is_dd:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)

    metrics = {
        "%DaysInDD": round(in_dd, 1),
        "MaxDD%": round(maxdd, 2),
        "DDdepth p50%": round(q50, 2),
        "DDdepth p90%": round(q90, 2),
        "DDdepth p99%": round(q99, 2),
        "LongestDDRun(days)": int(longest),
    }

    return CardResult(
        key="stress.drawdown_distribution",
        title="Stress: Drawdown Distribution (SPY)",
        summary="Drawdown depth histogram + time-under-water style metrics.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="drawdown_hist.png", payload=png_hist)],
    )

@register_card("stress.streaks_best_worst", "Stress: Best/Worst Streaks (SPY)", "stress", min_tier="black", cost=7, heavy=False, slots=("S11",))
def streaks(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-9000:]
    r = close.pct_change().dropna()
    if len(r) < 800:
        return CardResult(
            key="stress.streaks_best_worst",
            title="Stress: Best/Worst Streaks (SPY)",
            summary="Not enough data (need ~800+ daily returns)."
        )

    # rolling compounded returns over 5D and 10D
    roll5 = (1+r).rolling(5).apply(np.prod, raw=True) - 1
    roll10 = (1+r).rolling(10).apply(np.prod, raw=True) - 1

    best5 = roll5.sort_values(ascending=False).head(6)
    worst5 = roll5.sort_values().head(6)
    best10 = roll10.sort_values(ascending=False).head(6)
    worst10 = roll10.sort_values().head(6)

    def make_rows(series, label):
        rows=[]
        for d, v in series.items():
            rows.append((label, d.strftime("%Y-%m-%d"), f"{v*100:+.2f}"))
        return rows

    rows = []
    rows += make_rows(best5, "Best 5D")
    rows += make_rows(worst5, "Worst 5D")
    rows += make_rows(best10, "Best 10D")
    rows += make_rows(worst10, "Worst 10D")
    tab = pd.DataFrame(rows, columns=["Bucket","EndDate","Return%"])
    png_tbl = _table_png("Best/Worst Streaks (rolling compounded returns)", tab)

    # Quick chart of rolling 10D returns (last 600)
    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.plot((roll10.iloc[-600:]*100).values)
    ax.axhline(0, linewidth=1)
    ax.set_title("Rolling 10D compounded return (%) — last ~600 days")
    ax.grid(True, alpha=0.25)
    png_line = fig_to_png_bytes(fig)

    metrics = {
        "Worst5D%": round(float(worst5.iloc[0]*100), 2),
        "Worst10D%": round(float(worst10.iloc[0]*100), 2),
        "Best5D%": round(float(best5.iloc[0]*100), 2),
        "Best10D%": round(float(best10.iloc[0]*100), 2),
    }

    return CardResult(
        key="stress.streaks_best_worst",
        title="Stress: Best/Worst Streaks (SPY)",
        summary="Worst and best multi-day streaks + rolling 10D line.",
        metrics=metrics,
        artifacts=[
            Artifact(kind="image/png", name="streaks_table.png", payload=png_tbl),
            Artifact(kind="image/png", name="roll10_line.png", payload=png_line),
        ],
    )
