from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema

# ---------------- helpers ----------------
def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=max(30, win//3)).mean()
    sd = s.rolling(win, min_periods=max(30, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _table_png(title: str, df: pd.DataFrame, scale_y: float = 1.4) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, scale_y)
    return fig_to_png_bytes(fig)

def _ewo(close: pd.Series, fast=5, slow=35) -> pd.Series:
    return ema(close, fast) - ema(close, slow)

def _true_range(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    pc = c.shift(1)
    return pd.concat([
        (df["High"].astype(float) - df["Low"].astype(float)).abs(),
        (df["High"].astype(float) - pc).abs(),
        (df["Low"].astype(float) - pc).abs(),
    ], axis=1).max(axis=1)

# ---------------- 1) Wofl-ish reversal scan ----------------
@register_card("alerts.reversal_watchlist", "Reversal Watchlist (Wofl scan)", "alerts", min_tier="black", cost=11, heavy=True, slots=("S06",))
def reversal_watchlist(ctx: CardContext) -> CardResult:
    spdf = ctx.sp500.df()
    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 200)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    fails=0

    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1; continue
        df = df.iloc[-520:].copy()
        if len(df) < 120:
            fails += 1; continue

        close = df["Close"].astype(float)
        ma20 = close.rolling(20, min_periods=20).mean()

        dist = (close.iloc[-1] - ma20.iloc[-1]) / (ma20.iloc[-1] + 1e-12)
        abs_dist = abs(float(dist))

        e = _ewo(close)
        ez = _zscore(e, 120)
        es = e.diff()

        ez_last = float(ez.iloc[-1]) if np.isfinite(ez.iloc[-1]) else float("nan")
        es_last = float(es.iloc[-1]) if np.isfinite(es.iloc[-1]) else float("nan")

        near_ma = abs_dist <= 0.03
        below_ma = dist < 0
        above_ma = dist > 0
        low_ext = ez_last <= -1.0
        high_ext = ez_last >= +1.0
        rising = es_last > 0
        falling = es_last < 0

        bottom_pass = sum([near_ma, below_ma, low_ext, rising])
        top_pass = sum([near_ma, above_ma, high_ext, falling])

        if max(bottom_pass, top_pass) < 3:
            continue

        if bottom_pass > top_pass:
            verdict = "BOTTOM"
            rule_strength = bottom_pass / 4.0
        elif top_pass > bottom_pass:
            verdict = "TOP"
            rule_strength = top_pass / 4.0
        else:
            verdict = "AMBIG"
            rule_strength = bottom_pass / 4.0

        proximity = max(0.0, 1.0 - (abs_dist / 0.03))
        extremity = min(1.0, abs(ez_last) / 2.0) if np.isfinite(ez_last) else 0.0
        score = 100.0 * (0.60*rule_strength + 0.25*proximity + 0.15*extremity)
        score = float(max(0.0, min(100.0, score)))

        sector = ""
        try:
            sector = str(spdf.loc[spdf["Symbol"]==raw, "Sector"].iloc[0])
        except Exception:
            sector = ""

        note = f"distMA20 {abs_dist*100:.2f}% | EWOz {ez_last:.2f}"
        rows.append((tk, sector, verdict, score, note))

    if not rows:
        return CardResult(
            key="alerts.reversal_watchlist",
            title="Reversal Watchlist (Wofl scan)",
            summary="No strong candidates found in capped universe window.",
            warnings=[f"fails={fails}"],
        )

    tab = pd.DataFrame(rows, columns=["Symbol","Sector","Verdict","Score","Notes"]).sort_values("Score", ascending=False).head(18)
    png = _table_png("Reversal Watchlist (Wofl scan) — Top Candidates", tab)

    metrics = {"Candidates": int(len(rows)), "Shown": int(len(tab)), "Fails": int(fails)}
    bullets = [
        "Heuristic scan: MA20 proximity + EWO z-extreme + EWO slope direction.",
        "Use as a watchlist; confirm with your preferred entry/exit rules.",
    ]

    return CardResult(
        key="alerts.reversal_watchlist",
        title="Reversal Watchlist (Wofl scan)",
        summary="Top reversal candidates from capped S&P universe.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="reversal_watchlist.png", payload=png)],
    )

# ---------------- 2) MA200 cross + 52w high scan ----------------
@register_card("alerts.ma200_52w_scan", "MA200 Cross + 52W High Scan", "alerts", min_tier="pro", cost=10, heavy=True, slots=("S08",))
def ma200_52w_scan(ctx: CardContext) -> CardResult:
    spdf = ctx.sp500.df()
    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 220)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    fails=0

    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1; continue
        c = df["Close"].astype(float).dropna()
        if len(c) < 260:
            fails += 1; continue

        c = c.iloc[-520:]
        ma200 = c.rolling(200).mean()
        hi52 = c.rolling(252).max()

        if not np.isfinite(ma200.iloc[-2]) or not np.isfinite(ma200.iloc[-1]) or not np.isfinite(hi52.iloc[-1]):
            fails += 1; continue

        bull = (c.iloc[-2] <= ma200.iloc[-2]) and (c.iloc[-1] > ma200.iloc[-1])
        bear = (c.iloc[-2] >= ma200.iloc[-2]) and (c.iloc[-1] < ma200.iloc[-1])

        dist_hi = (c.iloc[-1] - hi52.iloc[-1]) / (hi52.iloc[-1] + 1e-12)
        near_hi = dist_hi >= -0.02  # within 2% of 52w high

        if not (bull or bear or near_hi):
            continue

        sector = ""
        try:
            sector = str(spdf.loc[spdf["Symbol"]==raw, "Sector"].iloc[0])
        except Exception:
            sector = ""

        sig = "MA200↑" if bull else ("MA200↓" if bear else "")
        note = f"52W {dist_hi*100:+.2f}%"
        score = (2 if bull else 0) + (2 if bear else 0) + (1 if near_hi else 0) + (1.0 - min(abs(dist_hi)/0.02, 1.0))
        rows.append((tk, sector, sig, f"{dist_hi*100:+.2f}%", round(score,3), note))

    if not rows:
        return CardResult(
            key="alerts.ma200_52w_scan",
            title="MA200 Cross + 52W High Scan",
            summary="No hits found in capped universe window.",
            warnings=[f"fails={fails}"]
        )

    tab = pd.DataFrame(rows, columns=["Symbol","Sector","Signal","%From52WHigh","Rank","Notes"]).sort_values("Rank", ascending=False).head(20)
    png = _table_png("MA200 Cross + Near 52W High — Watchlist", tab)

    metrics = {"Candidates": int(len(rows)), "Shown": int(len(tab)), "Fails": int(fails)}
    bullets = [
        "Hits include MA200 cross events and symbols within ~2% of 52W high.",
        "Rank is a simple heuristic for prioritization.",
    ]

    return CardResult(
        key="alerts.ma200_52w_scan",
        title="MA200 Cross + 52W High Scan",
        summary="Trend/strength watchlist from capped S&P universe.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="ma200_52w_watchlist.png", payload=png)],
    )

# ---------------- 3) Volatility breakout watchlist ----------------
@register_card("alerts.volatility_breakout_watchlist", "Volatility Breakout Watchlist", "alerts", min_tier="pro", cost=10, heavy=True, slots=("S09",))
def vol_breakouts(ctx: CardContext) -> CardResult:
    spdf = ctx.sp500.df()
    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 220)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    fails=0

    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or not set(["Open","High","Low","Close"]).issubset(df.columns):
            fails += 1; continue

        df = df.iloc[-520:].copy()
        if len(df) < 80:
            fails += 1; continue

        close = df["Close"].astype(float)
        ret = close.pct_change()
        tr = _true_range(df)
        atr14 = tr.rolling(14).mean()
        tr_z = _zscore(tr, 120)

        if not np.isfinite(tr_z.iloc[-1]) or not np.isfinite(atr14.iloc[-1]) or close.iloc[-1] == 0:
            fails += 1; continue

        atrp = float(atr14.iloc[-1] / close.iloc[-1])
        sigma20 = float(ret.rolling(20).std(ddof=1).iloc[-1]) if np.isfinite(ret.rolling(20).std(ddof=1).iloc[-1]) else float("nan")
        move_z = abs(float(ret.iloc[-1])) / (sigma20 + 1e-12) if np.isfinite(sigma20) else float("nan")

        score = float(max(tr_z.iloc[-1], move_z if np.isfinite(move_z) else 0.0))
        if score < 2.0:
            continue

        sector = ""
        try:
            sector = str(spdf.loc[spdf["Symbol"]==raw, "Sector"].iloc[0])
        except Exception:
            sector = ""

        note = f"TRz {float(tr_z.iloc[-1]):.2f} | |r|/σ {move_z:.2f}" if np.isfinite(move_z) else f"TRz {float(tr_z.iloc[-1]):.2f}"
        rows.append((tk, sector, round(score,2), f"{atrp*100:.2f}%", note))

    if not rows:
        return CardResult(
            key="alerts.volatility_breakout_watchlist",
            title="Volatility Breakout Watchlist",
            summary="No breakouts found in capped universe window.",
            warnings=[f"fails={fails}"]
        )

    tab = pd.DataFrame(rows, columns=["Symbol","Sector","Score","ATR%","Notes"]).sort_values("Score", ascending=False).head(20)
    png = _table_png("Volatility Breakout Watchlist — Top Candidates", tab)

    metrics = {"Candidates": int(len(rows)), "Shown": int(len(tab)), "Fails": int(fails)}
    bullets = [
        "Score uses max(TR z-score, |return|/sigma20).",
        "Use as an alert list for news/earnings/vol regime changes.",
    ]

    return CardResult(
        key="alerts.volatility_breakout_watchlist",
        title="Volatility Breakout Watchlist",
        summary="High-volatility candidates from capped S&P universe.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="vol_breakout_watchlist.png", payload=png)],
    )