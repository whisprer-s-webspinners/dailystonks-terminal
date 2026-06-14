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

def _ewo(close: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    return ema(close, fast) - ema(close, slow)

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

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

    return pd.DataFrame({"vol": vol_reg, "trend": trend}, index=close.index)

def _bucket_stats(mask: pd.Series, fwd5: pd.Series, fwd20: pd.Series, reg: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for v in ["LowVol","HighVol"]:
        for t in ["TrendUp","Range","TrendDown"]:
            m = (mask.fillna(False)) & (reg["vol"]==v) & (reg["trend"]==t)
            a5 = fwd5[m].dropna()
            a20 = fwd20[m].dropna()
            rows.append([
                v, t,
                int(m.sum()),
                "" if len(a5)==0 else f"{a5.mean()*100:+.2f}",
                "" if len(a5)==0 else f"{(a5>0).mean()*100:.1f}",
                "" if len(a20)==0 else f"{a20.mean()*100:+.2f}",
                "" if len(a20)==0 else f"{(a20>0).mean()*100:.1f}",
            ])
    return pd.DataFrame(rows, columns=["VolReg","TrendReg","N","MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"])

@register_card("backtest.wofl_magic_by_regime", "Backtest: Wofl Magic by Regime (SPY)", "backtest", min_tier="black", cost=10, heavy=False, slots=("S11",))
def wofl_by_regime(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-2500:].copy()
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    dist = (close - ma20) / (ma20 + 1e-12)

    e = _ewo(close)
    ez = _zscore(e, 120)
    es = e.diff()

    near_ma = dist.abs() <= 0.03
    bottom = near_ma & (dist < 0) & (ez <= -1.0) & (es > 0)
    top    = near_ma & (dist > 0) & (ez >= +1.0) & (es < 0)

    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    reg = _regimes(close)

    tab_b = _bucket_stats(bottom, f5, f20, reg)
    tab_t = _bucket_stats(top, f5, f20, reg)

    png_b = _table_png("Wofl BOTTOM → Forward stats by regime (SPY)", tab_b)
    png_t = _table_png("Wofl TOP → Forward stats by regime (SPY)", tab_t)

    return CardResult(
        key="backtest.wofl_magic_by_regime",
        title="Backtest: Wofl Magic by Regime (SPY)",
        summary="Same Wofl rules, split by volatility + trend regime.",
        artifacts=[
            Artifact(kind="image/png", name="wofl_bottom_by_regime.png", payload=png_b),
            Artifact(kind="image/png", name="wofl_top_by_regime.png", payload=png_t),
        ],
        bullets=[
            "VolReg: HighVol if vol20 > rolling median(vol20).",
            "TrendReg: TrendUp/TrendDown if price vs MA200 aligns with MA50 slope; else Range.",
        ]
    )

@register_card("backtest.meanrev_by_regime", "Backtest: Mean-Reversion by Regime (SPY)", "backtest", min_tier="black", cost=10, heavy=False, slots=("S11",))
def meanrev_by_regime(ctx: CardContext) -> CardResult:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-2500:].copy()
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    dist = (close - ma20) / (ma20 + 1e-12)
    z = _zscore(dist, 120)

    # entry signals (event-based): z <= -2
    entry = (z <= -2.0)

    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    reg = _regimes(close)
    tab = _bucket_stats(entry, f5, f20, reg)

    png = _table_png("MeanRev Entry (z<=-2) → Forward stats by regime (SPY)", tab)

    return CardResult(
        key="backtest.meanrev_by_regime",
        title="Backtest: Mean-Reversion by Regime (SPY)",
        summary="Mean-reversion entry stats split by volatility + trend regime.",
        artifacts=[Artifact(kind="image/png", name="meanrev_by_regime.png", payload=png)],
        bullets=[
            "Entry signal is event-based (z<=-2), not a full position backtest.",
            "Use to understand which regimes reward the signal."
        ]
    )