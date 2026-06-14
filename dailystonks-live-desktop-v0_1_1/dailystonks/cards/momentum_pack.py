
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema

TRADING_DAYS = 252

def _spy_df(ctx: CardContext) -> pd.DataFrame:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d")
    if df is None or df.empty:
        raise RuntimeError("SPY OHLCV missing/empty")
    need = {"Open","High","Low","Close"}
    if not need.issubset(df.columns):
        raise RuntimeError("SPY OHLCV missing required columns")
    return df.copy()

def _true_range(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    pc = c.shift(1)
    return pd.concat([
        (df["High"].astype(float) - df["Low"].astype(float)).abs(),
        (df["High"].astype(float) - pc).abs(),
        (df["Low"].astype(float) - pc).abs(),
    ], axis=1).max(axis=1)

def _roll_slope_r2(y: pd.Series, win: int = 50) -> tuple[pd.Series, pd.Series]:
    # slope + R^2 of y ~ a*x+b over rolling windows
    # y is assumed numeric; uses closed-form least squares per window.
    x = np.arange(win, dtype=float)
    sx = x.sum()
    sxx = (x*x).sum()
    denom = (win*sxx - sx*sx) + 1e-12

    def slope(a: np.ndarray) -> float:
        ay = a
        sy = ay.sum()
        sxy = (x*ay).sum()
        b1 = (win*sxy - sx*sy) / denom
        return float(b1)

    def r2(a: np.ndarray) -> float:
        ay = a
        sy = ay.sum()
        sxy = (x*ay).sum()
        b1 = (win*sxy - sx*sy) / denom
        b0 = (sy - b1*sx) / win
        yhat = b0 + b1*x
        ss_res = float(np.sum((ay - yhat)**2))
        ss_tot = float(np.sum((ay - ay.mean())**2)) + 1e-12
        return float(1.0 - ss_res/ss_tot)

    s = y.rolling(win, min_periods=win).apply(slope, raw=True)
    r = y.rolling(win, min_periods=win).apply(r2, raw=True)
    return s, r

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@register_card("momentum.trend_taxonomy", "Momentum: Trend Taxonomy (SPY)", "momentum", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def trend_taxonomy(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    logp = np.log(close.replace(0, np.nan)).dropna()

    if len(logp) < 300:
        return CardResult(
            key="momentum.trend_taxonomy",
            title="Momentum: Trend Taxonomy (SPY)",
            summary="Not enough data (need ~300+ bars)."
        )

    slope50, r2_50 = _roll_slope_r2(logp, 50)
    slope120, r2_120 = _roll_slope_r2(logp, 120)

    # annualize slope in log-space
    ann50 = np.exp(slope50 * TRADING_DAYS) - 1.0
    ann120 = np.exp(slope120 * TRADING_DAYS) - 1.0

    last_ann50 = float(ann50.dropna().iloc[-1])
    last_r2_50 = float(r2_50.dropna().iloc[-1])
    last_ann120 = float(ann120.dropna().iloc[-1])
    last_r2_120 = float(r2_120.dropna().iloc[-1])

    # regime label (simple + explainable)
    thr = 0.06  # ~6% annual drift threshold
    qual = 0.25 # minimum R^2 for "trend"
    if last_r2_50 >= qual and last_ann50 > thr:
        reg = "TrendUp"
    elif last_r2_50 >= qual and last_ann50 < -thr:
        reg = "TrendDown"
    else:
        reg = "Range/Chop"

    trend_strength = abs(last_ann50) * max(0.0, last_r2_50)  # 0..~1 scale-ish

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.set_title(f"Trend Taxonomy — {reg}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot((ann50*100).values, label="Ann drift (50D) %")
    ax2.plot((r2_50*100).values, label="R² (50D) x100")
    ax2.axhline(thr*100, linewidth=1)
    ax2.axhline(-thr*100, linewidth=1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "Regime": reg,
        "AnnDrift50%": round(last_ann50*100, 2),
        "R2_50": round(last_r2_50, 3),
        "AnnDrift120%": round(last_ann120*100, 2),
        "R2_120": round(last_r2_120, 3),
        "TrendStrength": round(float(trend_strength), 4),
    }

    bullets = [
        "AnnDrift uses log-price slope annualized (exp(slope*252)-1).",
        "R² measures ‘trend quality’ (higher = cleaner trend).",
        "Regime uses 50D drift + R² threshold for a simple label.",
    ]

    return CardResult(
        key="momentum.trend_taxonomy",
        title="Momentum: Trend Taxonomy (SPY)",
        summary="Drift + R² based trend label and strength readout.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="trend_taxonomy.png", payload=png)],
    )

@register_card("momentum.ma_ribbon_compression", "Momentum: MA Ribbon Compression (SPY)", "momentum", min_tier="pro", cost=6, heavy=False, slots=("S11",))
def ma_ribbon_compression(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)

    spans = [8, 13, 21, 34, 55, 89, 144]
    emas = {f"EMA{n}": ema(close, n) for n in spans}
    E = pd.DataFrame(emas)

    # ribbon width as % of price
    width = (E.max(axis=1) - E.min(axis=1)) / (close + 1e-12) * 100.0
    w = width.dropna()
    if len(w) < 400:
        return CardResult(
            key="momentum.ma_ribbon_compression",
            title="Momentum: MA Ribbon Compression (SPY)",
            summary="Not enough data for ribbon width stats."
        )

    last = float(w.iloc[-1])
    pctl = float((w.rank(pct=True).iloc[-1]) * 100.0)  # 0..100

    # show compression threshold at p20 and expansion at p80
    p20 = float(np.quantile(w.values, 0.20))
    p80 = float(np.quantile(w.values, 0.80))

    fig = plt.figure(figsize=(10,6.6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="Close")
    for n in spans:
        ax1.plot(E[f"EMA{n}"].values, linewidth=1, alpha=0.9, label=f"EMA{n}")
    ax1.set_title("EMA Ribbon (SPY)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=4, fontsize=7)

    ax2.plot(width.values, label="Ribbon width %")
    ax2.axhline(p20, linewidth=1, label="p20 (compression)")
    ax2.axhline(p80, linewidth=1, label="p80 (expansion)")
    ax2.set_title(f"Ribbon width now: {last:.3f}% (percentile {pctl:.1f})")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=3, fontsize=8)

    png = fig_to_png_bytes(fig)

    # label
    if last <= p20:
        lbl = "Compressed"
    elif last >= p80:
        lbl = "Expanded"
    else:
        lbl = "Neutral"

    metrics = {
        "WidthNow%": round(last, 4),
        "Percentile": round(pctl, 1),
        "p20%": round(p20, 4),
        "p80%": round(p80, 4),
        "State": lbl,
    }

    bullets = [
        "Compression can precede breakouts; expansion can signal trend continuation or exhaustion.",
        "Use with vol regime + momentum to avoid false ‘squeeze’ reads.",
    ]

    return CardResult(
        key="momentum.ma_ribbon_compression",
        title="Momentum: MA Ribbon Compression (SPY)",
        summary="EMA ribbon + compression/expansion width metric.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="ma_ribbon_compression.png", payload=png)],
    )

@register_card("momentum.breakout_probability_heuristic", "Momentum: Breakout Probability (heuristic, SPY)", "momentum", min_tier="black", cost=6, heavy=False, slots=("S11",))
def breakout_probability(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    ret = close.pct_change()

    # features
    # 1) ribbon compression percent rank (lower width => higher breakout odds)
    spans = [8, 13, 21, 34, 55, 89]
    E = pd.DataFrame({f"EMA{n}": ema(close, n) for n in spans})
    width = (E.max(axis=1) - E.min(axis=1)) / (close + 1e-12)
    w = width.rolling(500, min_periods=200).apply(lambda a: float(pd.Series(a).rank(pct=True).iloc[-1]), raw=False)  # 0..1
    comp = 1.0 - w  # higher = more compressed

    # 2) trend strength = |ann drift 50| * R²
    logp = np.log(close.replace(0, np.nan)).dropna()
    slope50, r2_50 = _roll_slope_r2(logp, 50)
    ann50 = np.exp(slope50 * TRADING_DAYS) - 1.0
    trend_strength = (ann50.abs() * r2_50).reindex(close.index)

    # 3) vol regime: vol20 percentile (higher vol => more likely big move)
    vol20 = ret.rolling(20).std(ddof=1) * math.sqrt(TRADING_DAYS)
    v = vol20.rolling(500, min_periods=200).apply(lambda a: float(pd.Series(a).rank(pct=True).iloc[-1]), raw=False)  # 0..1

    # combine (pure heuristic)
    # comp high => breakout odds up
    # trend_strength high => breakout odds up
    # vol percentile high => breakout odds up (bigger moves)
    X = (
        2.2*(comp.fillna(0.0)) +
        1.6*(trend_strength.fillna(0.0)) +
        1.2*(v.fillna(0.0)) -
        1.8
    )
    p = pd.Series(_sigmoid(X.values), index=close.index)

    last_p = float(p.dropna().iloc[-1]) if p.dropna().shape[0] else float("nan")
    last_comp = float(comp.dropna().iloc[-1]) if comp.dropna().shape[0] else float("nan")
    last_tr = float(trend_strength.dropna().iloc[-1]) if trend_strength.dropna().shape[0] else float("nan")
    last_v = float(v.dropna().iloc[-1]) if v.dropna().shape[0] else float("nan")

    fig = plt.figure(figsize=(10,6.0))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.set_title("SPY with Breakout Probability (heuristic)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot((p*100).values, label="BreakoutProb%")
    ax2.axhline(50, linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    label = "Low"
    if np.isfinite(last_p):
        if last_p >= 0.70: label = "High"
        elif last_p >= 0.55: label = "Medium"

    metrics = {
        "BreakoutProb%": round(last_p*100, 1) if np.isfinite(last_p) else None,
        "Level": label,
        "Compression(0-1)": round(last_comp, 3) if np.isfinite(last_comp) else None,
        "TrendStrength": round(last_tr, 4) if np.isfinite(last_tr) else None,
        "VolPct(0-1)": round(last_v, 3) if np.isfinite(last_v) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "Heuristic score combines: compression (ribbon tightness), trend strength (drift×R²), and vol percentile.",
        "Use as a dial for ‘breakout risk/odds’, not a prediction guarantee.",
    ]

    return CardResult(
        key="momentum.breakout_probability_heuristic",
        title="Momentum: Breakout Probability (heuristic, SPY)",
        summary="A simple, explainable breakout-odds dial using compression + trend + vol.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="breakout_prob.png", payload=png)],
    )
