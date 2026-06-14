
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema

TRADING_DAYS = 252

def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=max(30, win//3)).mean()
    sd = s.rolling(win, min_periods=max(30, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _true_range(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    pc = c.shift(1)
    return pd.concat([
        (df["High"].astype(float) - df["Low"].astype(float)).abs(),
        (df["High"].astype(float) - pc).abs(),
        (df["Low"].astype(float) - pc).abs(),
    ], axis=1).max(axis=1)

def _regimes(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    vol20 = ret.rolling(20).std(ddof=1) * math.sqrt(TRADING_DAYS)
    vol_ref = vol20.rolling(252, min_periods=120).median()
    vol_reg = np.where(vol20 > vol_ref, "HighVol", "LowVol")

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    slope50 = ma50.diff()

    up = (close > ma200) & (slope50 > 0)
    down = (close < ma200) & (slope50 < 0)
    trend = np.where(up, "TrendUp", np.where(down, "TrendDown", "Range"))

    return pd.DataFrame({"vol": vol_reg, "trend": trend, "vol20": vol20, "vol_ref": vol_ref}, index=close.index)

# ---- candle patterns (simple + robust) ----
def _body(o,c): return (c-o).abs()
def _range(h,l):
    r = (h-l).abs()
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
    "BullEngulf": bullish_engulfing,
    "BearEngulf": bearish_engulfing,
    "Hammer": hammer,
    "ShootingStar": shooting_star,
}

def _best_bucket(mask: pd.Series, reg: pd.DataFrame, f5: pd.Series, f20: pd.Series, min_n: int = 15):
    m = mask.fillna(False)
    df = pd.DataFrame({
        "m": m,
        "reg": (reg["vol"].astype(str) + "/" + reg["trend"].astype(str)),
        "f5": f5,
        "f20": f20,
    }).dropna(subset=["reg"])
    df = df[df["m"]]
    if df.empty:
        return None

    g = df.groupby("reg").agg(
        N=("f5","count"),
        MeanFwd5=("f5","mean"),
        WinFwd5=("f5", lambda x: float((x>0).mean()) if len(x) else np.nan),
        MeanFwd20=("f20","mean"),
        WinFwd20=("f20", lambda x: float((x>0).mean()) if len(x) else np.nan),
    )
    g = g[g["N"] >= min_n]
    if g.empty:
        # fallback: take max N bucket
        g = df.groupby("reg").agg(
            N=("f5","count"),
            MeanFwd5=("f5","mean"),
            WinFwd5=("f5", lambda x: float((x>0).mean()) if len(x) else np.nan),
            MeanFwd20=("f20","mean"),
            WinFwd20=("f20", lambda x: float((x>0).mean()) if len(x) else np.nan),
        ).sort_values("N", ascending=False).head(1)
    else:
        g = g.sort_values("MeanFwd5", ascending=False).head(1)

    row = g.iloc[0]
    return g.index[0], int(row["N"]), float(row["MeanFwd5"]), float(row["WinFwd5"]), float(row["MeanFwd20"]), float(row["WinFwd20"])

@register_card("backtest.patterns_by_regime", "Backtest: Patterns by Regime (SPY)", "backtest", min_tier="black", cost=8, heavy=False, slots=("S11",))
def patterns_by_regime(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-2600:].copy()
    if df.empty or len(df) < 600:
        return CardResult(
            key="backtest.patterns_by_regime",
            title="Backtest: Patterns by Regime (SPY)",
            summary="Not enough history (need ~600+ bars)."
        )

    close = df["Close"].astype(float)
    reg = _regimes(close)
    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    rows=[]
    for name, fn in PATTERNS.items():
        mask = fn(df)
        total_n = int(mask.sum())
        if total_n < 25:
            continue
        best = _best_bucket(mask, reg, f5, f20, min_n=15)
        if not best:
            continue
        regname, n, m5, w5, m20, w20 = best
        rows.append([name, total_n, regname, n, f"{m5*100:+.2f}", f"{w5*100:.1f}", f"{m20*100:+.2f}", f"{w20*100:.1f}"])

    if not rows:
        return CardResult(
            key="backtest.patterns_by_regime",
            title="Backtest: Patterns by Regime (SPY)",
            summary="No patterns had enough samples to score."
        )

    out = pd.DataFrame(rows, columns=["Pattern","TotalN","BestRegime","N@Best","MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"])
    png = _table_png("Patterns by Regime (SPY) — best bucket by MeanFwd5", out)

    return CardResult(
        key="backtest.patterns_by_regime",
        title="Backtest: Patterns by Regime (SPY)",
        summary="Which regimes each pattern likes (best bucket by MeanFwd5).",
        artifacts=[Artifact(kind="image/png", name="patterns_by_regime.png", payload=png)],
        bullets=[
            "Regimes: vol=High/Low via vol20 vs rolling median; trend=TrendUp/Down/Range via MA200 and MA50 slope.",
            "BestRegime chosen by highest MeanFwd5 with N>=15 (fallback to biggest N bucket).",
        ]
    )

@register_card("backtest.vol_breakout_by_regime", "Backtest: Vol Breakouts by Regime (SPY)", "backtest", min_tier="black", cost=8, heavy=False, slots=("S11",))
def vol_breakout_by_regime(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-2600:].copy()
    if df.empty or len(df) < 600:
        return CardResult(
            key="backtest.vol_breakout_by_regime",
            title="Backtest: Vol Breakouts by Regime (SPY)",
            summary="Not enough history (need ~600+ bars)."
        )

    close = df["Close"].astype(float)
    ret = close.pct_change()
    tr = _true_range(df)
    tr_z = _zscore(tr, 120)

    sigma20 = ret.rolling(20).std(ddof=1)
    move_z = (ret.abs() / (sigma20 + 1e-12))

    # breakout event: TR z-score >= 2 OR |ret| >= 2 sigma (whichever hits)
    breakout = (tr_z >= 2.0) | (move_z >= 2.0)

    reg = _regimes(close)
    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    best = _best_bucket(breakout, reg, f5, f20, min_n=20)
    if not best:
        return CardResult(
            key="backtest.vol_breakout_by_regime",
            title="Backtest: Vol Breakouts by Regime (SPY)",
            summary="No breakout events (unexpected)."
        )

    # full table by regime
    df_evt = pd.DataFrame({
        "m": breakout.fillna(False),
        "reg": (reg["vol"].astype(str) + "/" + reg["trend"].astype(str)),
        "f5": f5,
        "f20": f20,
    })
    df_evt = df_evt[df_evt["m"]].dropna(subset=["reg"])
    g = df_evt.groupby("reg").agg(
        N=("f5","count"),
        MeanFwd5=("f5","mean"),
        WinFwd5=("f5", lambda x: float((x>0).mean()) if len(x) else np.nan),
        MeanFwd20=("f20","mean"),
        WinFwd20=("f20", lambda x: float((x>0).mean()) if len(x) else np.nan),
    ).sort_values("N", ascending=False)

    out = g.reset_index()
    out["MeanFwd5%"] = out["MeanFwd5"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:+.2f}")
    out["WinFwd5%"]  = out["WinFwd5"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:.1f}")
    out["MeanFwd20%"] = out["MeanFwd20"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:+.2f}")
    out["WinFwd20%"]  = out["WinFwd20"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:.1f}")
    out = out[["reg","N","MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"]].head(12)
    png = _table_png("Vol Breakouts by Regime (SPY) — TRz>=2 OR |ret|>=2σ", out)

    return CardResult(
        key="backtest.vol_breakout_by_regime",
        title="Backtest: Vol Breakouts by Regime (SPY)",
        summary="How vol-breakout days behave by regime (forward returns).",
        artifacts=[Artifact(kind="image/png", name="vol_breakout_by_regime.png", payload=png)],
        bullets=[
            "Breakout event: TR z-score >= 2 OR |return| >= 2*sigma20.",
            "Forward stats are close-to-close; no fees/slippage.",
        ]
    )

@register_card("risk.regime_nowcast", "Regime Nowcast (SPY)", "risk", min_tier="pro", cost=5, heavy=False, slots=("S09","S11"))
def regime_nowcast(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-800:].copy()
    close = df["Close"].astype(float)
    reg = _regimes(close)

    vol20 = float(reg["vol20"].iloc[-1]) if np.isfinite(reg["vol20"].iloc[-1]) else float("nan")
    volref = float(reg["vol_ref"].iloc[-1]) if np.isfinite(reg["vol_ref"].iloc[-1]) else float("nan")
    vol_lbl = str(reg["vol"].iloc[-1])
    trend_lbl = str(reg["trend"].iloc[-1])

    ma200 = close.rolling(200).mean()
    ma50 = close.rolling(50).mean()

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="SPY Close")
    ax1.plot(ma50.values, label="MA50")
    ax1.plot(ma200.values, label="MA200")
    ax1.set_title(f"Regime Nowcast — {vol_lbl} / {trend_lbl}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    ax2.plot((reg["vol20"]*100).values, label="vol20% (ann.)")
    ax2.plot((reg["vol_ref"]*100).values, label="vol median% (rolling)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "VolReg": vol_lbl,
        "TrendReg": trend_lbl,
        "Vol20%": round(vol20*100, 2) if np.isfinite(vol20) else None,
        "VolRef%": round(volref*100, 2) if np.isfinite(volref) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="risk.regime_nowcast",
        title="Regime Nowcast (SPY)",
        summary="Current vol/trend regime + context chart.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="regime_nowcast.png", payload=png)],
    )
