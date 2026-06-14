from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema
from ..render.plotting import plot_candles

def ewo(close: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    return ema(close, fast) - ema(close, slow)

def detect_magic_signals(df: pd.DataFrame, *, ma_win: int = 20, tol: float = 0.03):
    # Approximation from your docs:
    # Bottom: EWO low rising, price below MA20 but within 3%
    # Top: EWO high falling, price above MA20 but within 3%
    close = df["Close"]
    ma = close.rolling(ma_win).mean()
    e = ewo(close)
    # slope proxy
    e_slope = e.diff()

    # "low/high" based on rolling quantiles
    low_thr = e.rolling(60, min_periods=30).quantile(0.2)
    high_thr = e.rolling(60, min_periods=30).quantile(0.8)

    near_ma = (close - ma).abs() / ma
    below = close < ma
    above = close > ma

    bottom = (near_ma < tol) & below & (e < low_thr) & (e_slope > 0)
    top    = (near_ma < tol) & above & (e > high_thr) & (e_slope < 0)
    return bottom.fillna(False), top.fillna(False), ma, e

@register_card("reversal.ewo_confirmation", "EWO Confirmation", "reversal", min_tier="basic", cost=4, heavy=False, slots=("S06","S07"))
def ewo_confirmation(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval).iloc[-260:]
    e = ewo(df["Close"])
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.bar(range(len(df)), e.values, alpha=0.6)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{t} Elliott Wave Oscillator (EMA5-EMA35)")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    last = float(e.iloc[-1])
    prev = float(e.iloc[-2]) if len(e) > 1 else float("nan")
    trend = "rising" if last > prev else "falling"
    return CardResult(
        key="reversal.ewo_confirmation",
        title=f"{t}: EWO Confirmation",
        summary="EWO rising can support bottom reversals; falling can support top reversals.",
        metrics={"EWO": round(last, 6), "EWO_trend": trend},
        artifacts=[Artifact(kind="image/png", name=f"{t}_ewo.png", payload=png)]
    )

@register_card("reversal.magic_full_chart", "Magic Reversal (Chart)", "reversal", min_tier="black", cost=8, heavy=False, slots=("S06","S07"))
def magic_full(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval)
    dfp = df.iloc[-260:].copy()

    bottom, top, ma, e = detect_magic_signals(dfp)

    fig = plt.figure(figsize=(10,7.6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    d = plot_candles(ax1, dfp, title=f"{t} Magic Reversal ({ctx.interval})", max_bars=None)
    x = np.arange(len(d))
    ma_v = ma.iloc[-len(d):].values
    ax1.plot(x, ma_v, linewidth=2, label="MA20 (blue line)")
    # markers
    b_idx = np.where(bottom.iloc[-len(d):].values)[0]
    t_idx = np.where(top.iloc[-len(d):].values)[0]
    if len(b_idx):
        ax1.scatter(b_idx, d["Low"].iloc[b_idx].values, marker="^", s=70, label="Bottom signal")
    if len(t_idx):
        ax1.scatter(t_idx, d["High"].iloc[t_idx].values, marker="v", s=70, label="Top signal")
    ax1.legend(loc="upper left")

    ax2.bar(x, e.iloc[-len(d):].values, alpha=0.6)
    ax2.axhline(0, linewidth=1)
    ax2.set_title("EWO (EMA5-EMA35)")
    ax2.grid(True, alpha=0.25)

    png = fig_to_png_bytes(fig)

    last_close = float(dfp["Close"].iloc[-1])
    last_ma = float(ma.iloc[-1])
    dist = abs(last_close - last_ma) / last_ma if last_ma != 0 else float("nan")
    return CardResult(
        key="reversal.magic_full_chart",
        title=f"{t}: Magic Reversal Chart",
        summary="Triangles show candidate reversal zones (approximation of your 'magic reversal' rules).",
        metrics={"Close": round(last_close,4), "MA20": round(last_ma,4), "dist_to_MA20%": round(dist*100,2)},
        artifacts=[Artifact(kind="image/png", name=f"{t}_magic.png", payload=png)]
    )

@register_card("reversal.magic_checklist_card", "Magic Reversal (Checklist)", "reversal", min_tier="black", cost=5, heavy=False, slots=("S05","S07"))
def magic_checklist(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval).copy()
    dfp = df.iloc[-260:]
    bottom, top, ma, e = detect_magic_signals(dfp)
    last = dfp.iloc[-1]
    ma20 = float(ma.iloc[-1])
    close = float(last["Close"])
    dist = abs(close - ma20)/ma20 if ma20 else float("nan")

    is_bottom = bool(bottom.iloc[-1])
    is_top = bool(top.iloc[-1])

    bullets = [
        f"Price vs MA20 distance: {dist*100:.2f}% (target < 3%).",
        f"EWO value: {float(e.iloc[-1]):.6f}.",
        "Bottom setup requires: below MA20, near MA20, EWO low & rising.",
        "Top setup requires: above MA20, near MA20, EWO high & falling.",
    ]
    verdict = "BOTTOM candidate" if is_bottom else ("TOP candidate" if is_top else "No candidate today")
    return CardResult(
        key="reversal.magic_checklist_card",
        title=f"{t}: Magic Reversal Checklist",
        summary=f"Verdict: {verdict}",
        metrics={"Verdict": verdict, "dist_to_MA20%": round(dist*100,2)},
        bullets=bullets
    )

@register_card("reversal.risk_box_rr_stops", "Risk Box (Stops + R:R)", "reversal", min_tier="pro", cost=3, heavy=False, slots=("S07",))
def risk_box(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval).iloc[-80:]
    close = float(df["Close"].iloc[-1])
    swing_low = float(df["Low"].rolling(10).min().iloc[-1])
    swing_high = float(df["High"].rolling(10).max().iloc[-1])

    # Simple default: long idea uses swing_low stop; short uses swing_high stop
    long_stop = swing_low
    long_risk = close - long_stop
    long_tp = close + 1.5 * long_risk

    short_stop = swing_high
    short_risk = short_stop - close
    short_tp = close - 1.5 * short_risk

    bullets = [
        "Heuristic risk box based on recent 10-bar swing extremes.",
        "Use as a starting point; align with setup context (trend/reversal)."
    ]
    return CardResult(
        key="reversal.risk_box_rr_stops",
        title=f"{t}: Risk Box (R:R 1:1.5)",
        summary="Stop/target suggestion from local swing extremes.",
        metrics={
            "Close": round(close,4),
            "Long_Stop": round(long_stop,4),
            "Long_TP": round(long_tp,4),
            "Short_Stop": round(short_stop,4),
            "Short_TP": round(short_tp,4),
        },
        bullets=bullets
    )
