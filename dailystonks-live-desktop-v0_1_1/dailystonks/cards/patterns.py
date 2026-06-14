from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes
from ..render.plotting import plot_candles

# -----------------------------
# Robust, lightweight pattern detectors (no TA-Lib dependency)
# -----------------------------
def _body(o, c): 
    return (c - o).abs()

def _range(h, l):
    r = (h - l).abs()
    return r.replace(0, np.nan)

def is_doji(df: pd.DataFrame, tol: float = 0.1) -> pd.Series:
    o,c,h,l = df["Open"], df["Close"], df["High"], df["Low"]
    return (_body(o,c) <= tol * (h-l)).fillna(False)

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po,pc = o.shift(1), c.shift(1)
    prev_red = pc < po
    curr_green = c > o
    engulfs = (o <= pc) & (c >= po)
    return (prev_red & curr_green & engulfs).fillna(False)

def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o,c = df["Open"], df["Close"]
    po,pc = o.shift(1), c.shift(1)
    prev_green = pc > po
    curr_red = c < o
    engulfs = (o >= pc) & (c <= po)
    return (prev_green & curr_red & engulfs).fillna(False)

def hammer(df: pd.DataFrame) -> pd.Series:
    o,c,h,l = df["Open"], df["Close"], df["High"], df["Low"]
    body = _body(o,c)
    rng = _range(h,l)
    lower = np.minimum(o,c) - l
    upper = h - np.maximum(o,c)
    cond = (lower >= 2.0*body) & (upper <= 0.3*body) & (body <= 0.3*rng)
    return cond.fillna(False)

def shooting_star(df: pd.DataFrame) -> pd.Series:
    o,c,h,l = df["Open"], df["Close"], df["High"], df["Low"]
    body = _body(o,c)
    rng = _range(h,l)
    upper = h - np.maximum(o,c)
    lower = np.minimum(o,c) - l
    cond = (upper >= 2.0*body) & (lower <= 0.3*body) & (body <= 0.3*rng)
    return cond.fillna(False)

PATTERNS = {
    "Doji": is_doji,
    "Bull Engulf": bullish_engulfing,
    "Bear Engulf": bearish_engulfing,
    "Hammer": hammer,
    "Shooting Star": shooting_star,
}

def _latest_hit(df: pd.DataFrame):
    hits = []
    for name, fn in PATTERNS.items():
        m = fn(df)
        idx = np.where(m.values)[0]
        if len(idx):
            hits.append((idx[-1], name))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[0]

@register_card("patterns.pattern_spotlight", "Pattern Spotlight", "patterns", min_tier="pro", cost=8, heavy=False, slots=("S11","S06"))
def pattern_spotlight(ctx: CardContext) -> CardResult:
    cand = ctx.tickers[:] if ctx.tickers else []
    if len(cand) < 3:
        cand += ctx.sp500.tickers(max_n=min(ctx.max_universe, 50))
    cand = list(dict.fromkeys([t.replace('.', '-') for t in cand]))[:70]

    data = ctx.market.get_ohlcv_many(cand, start=ctx.start, end=ctx.end, interval="1d")

    best = None
    for t in cand:
        df = data.get(t)
        if df is None or df.empty:
            continue
        df = df.iloc[-180:].copy()
        if set(["Open","High","Low","Close"]).issubset(df.columns) is False:
            continue
        hit = _latest_hit(df)
        if hit is None:
            continue
        ix, name = hit
        if best is None or ix > best[2]:
            best = (t, name, ix, df)

    if best is None:
        return CardResult(
            key="patterns.pattern_spotlight",
            title="Pattern Spotlight",
            summary="No pattern hits found in the scanned window.",
            warnings=[f"Scanned {len(cand)} tickers (capped)."]
        )

    t, name, ix, df = best
    fig = plt.figure(figsize=(10,6.3))
    ax1 = fig.add_subplot(2,1,1)
    d = plot_candles(ax1, df, title=f"{t} — {name}", max_bars=None)
    ax1.scatter([ix], [float(df['Low'].iloc[ix])], marker="^", s=90, label="pattern hit")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    fwd5 = df["Close"].pct_change(5).shift(-5)
    ax2.plot(fwd5.values, label="Forward 5D return (shifted)")
    ax2.axvline(ix, linewidth=1)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)

    png = fig_to_png_bytes(fig)
    return CardResult(
        key="patterns.pattern_spotlight",
        title=f"Pattern Spotlight: {t} — {name}",
        summary="Most recent detected hit (simple candle rules).",
        artifacts=[Artifact(kind="image/png", name="pattern_spotlight.png", payload=png)],
        metrics={"Ticker": t, "Pattern": name, "HitIndex": int(ix)},
    )

@register_card("patterns.weekly_top10_table_thumbs", "Weekly Pattern Hits (Top 10)", "patterns", min_tier="black", cost=10, heavy=True, slots=("S11",))
def weekly_top10(ctx: CardContext) -> CardResult:
    uni = ctx.sp500.tickers(max_n=min(ctx.max_universe, 90))
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows = []
    for raw in uni:
        t = raw.replace(".", "-")
        df = data.get(t)
        if df is None or df.empty:
            continue
        df = df.iloc[-30:].copy()
        if set(["Open","High","Low","Close"]).issubset(df.columns) is False:
            continue
        for pname, fn in PATTERNS.items():
            m = fn(df)
            hit_idx = np.where(m.values)[0]
            for ix in hit_idx:
                if ix >= len(df) - 5:
                    rows.append((t, pname, str(df.index[ix].date())))

    if not rows:
        return CardResult(
            key="patterns.weekly_top10_table_thumbs",
            title="Weekly Pattern Hits (Top 10)",
            summary="No recent hits found in the capped universe window."
        )

    tab = pd.DataFrame(rows, columns=["Symbol","Pattern","Date"])
    tab["Count"] = 1
    top = tab.groupby(["Symbol","Pattern"]).agg({"Count":"sum","Date":"max"}).reset_index()
    top = top.sort_values(["Count","Date"], ascending=[False,False]).head(10)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title("Weekly Pattern Hits (Top 10)")
    show = top[["Symbol","Pattern","Count","Date"]]
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
    table.scale(1, 1.5)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="patterns.weekly_top10_table_thumbs",
        title="Weekly Pattern Hits (Top 10)",
        summary="Top pattern occurrences over the last ~5 trading days (capped universe).",
        artifacts=[Artifact(kind="image/png", name="pattern_top10.png", payload=png)]
    )

@register_card("patterns.pattern_accuracy_leaderboard", "Pattern Accuracy Leaderboard", "patterns", min_tier="black", cost=10, heavy=True, slots=("S11",))
def pattern_accuracy(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-1600:].copy()
    fwd5 = df["Close"].pct_change(5).shift(-5)

    rows=[]
    for pname, fn in PATTERNS.items():
        m = fn(df).astype(bool)
        vals = fwd5[m].dropna()
        if len(vals) < 10:
            continue
        rows.append((pname, float(vals.mean()), float((vals>0).mean()), int(len(vals))))

    if not rows:
        return CardResult(
            key="patterns.pattern_accuracy_leaderboard",
            title="Pattern Accuracy Leaderboard",
            summary="Not enough pattern samples to score."
        )

    tab = pd.DataFrame(rows, columns=["Pattern","MeanFwd5","WinRate","N"]).sort_values("MeanFwd5", ascending=False)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title("Pattern Accuracy (SPY, forward 5D)")
    show = tab.round({"MeanFwd5":4,"WinRate":3})
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
    table.scale(1, 1.4)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="patterns.pattern_accuracy_leaderboard",
        title="Pattern Accuracy Leaderboard (SPY)",
        summary="Mean forward 5D return and win-rate after pattern hits (baseline).",
        artifacts=[Artifact(kind="image/png", name="pattern_accuracy.png", payload=png)]
    )