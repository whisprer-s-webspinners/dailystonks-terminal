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

def _perf_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 60:
        return {}
    ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(TRADING_DAYS) * ret.mean() / (ret.std(ddof=1) + 1e-12))
    dd = (eq / eq.cummax()) - 1.0
    maxdd = float(dd.min())
    # CAGR approx from daily length
    years = len(eq) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / max(years, 1e-9)) - 1.0)
    return {"CAGR%": round(cagr*100,2), "Sharpe": round(sharpe,2), "MaxDD%": round(maxdd*100,2), "Bars": int(len(eq))}

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _ewo(close: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    return ema(close, fast) - ema(close, slow)

# ----------------------------
# 1) Wofl magic -> forward stats
# ----------------------------
@register_card("backtest.wofl_magic_forward_stats", "Backtest-lite: Wofl Magic Forward Stats", "backtest", min_tier="black", cost=9, heavy=False, slots=("S11",))
def wofl_forward(ctx: CardContext) -> CardResult:
    t = "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-2000:].copy()
    if df.empty or len(df) < 400:
        return CardResult(
            key="backtest.wofl_magic_forward_stats",
            title="Backtest-lite: Wofl Magic Forward Stats",
            summary="Not enough history for stats (need ~400+ bars)."
        )

    close = df["Close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()
    dist = (close - ma20) / (ma20 + 1e-12)
    abs_dist = dist.abs()

    e = _ewo(close)
    ez = _zscore(e, 120)
    es = e.diff()

    near_ma = abs_dist <= 0.03
    below_ma = dist < 0
    above_ma = dist > 0
    low_ext = ez <= -1.0
    high_ext = ez >= +1.0
    rising = es > 0
    falling = es < 0

    bottom = near_ma & below_ma & low_ext & rising
    top    = near_ma & above_ma & high_ext & falling

    # forward returns
    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    def stats(mask: pd.Series):
        m = mask.fillna(False)
        a5 = f5[m].dropna()
        a20 = f20[m].dropna()
        return {
            "N": int(m.sum()),
            "MeanFwd5%": float(a5.mean()*100) if len(a5) else float("nan"),
            "WinFwd5%": float((a5>0).mean()*100) if len(a5) else float("nan"),
            "MeanFwd20%": float(a20.mean()*100) if len(a20) else float("nan"),
            "WinFwd20%": float((a20>0).mean()*100) if len(a20) else float("nan"),
        }

    sb = stats(bottom)
    st = stats(top)

    tab = pd.DataFrame([
        ["BOTTOM"] + [sb["N"], sb["MeanFwd5%"], sb["WinFwd5%"], sb["MeanFwd20%"], sb["WinFwd20%"]],
        ["TOP"]    + [st["N"], st["MeanFwd5%"], st["WinFwd5%"], st["MeanFwd20%"], st["WinFwd20%"]],
    ], columns=["Signal","N","MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"])

    # format
    out = tab.copy()
    for c in ["MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"]:
        out[c] = out[c].map(lambda x: "" if not np.isfinite(x) else f"{x:.2f}")

    png = _table_png("Wofl Magic Rules → Forward Return Stats (SPY)", out)

    bullets = [
        "Signal definition matches the explainer: near MA20 + EWO extreme + EWO slope.",
        "Forward returns are simple close-to-close; no fees/slippage.",
    ]

    return CardResult(
        key="backtest.wofl_magic_forward_stats",
        title="Backtest-lite: Wofl Magic Forward Stats",
        summary="How the Wofl rules behaved historically on SPY (forward returns).",
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="wofl_forward_stats.png", payload=png)]
    )

# ----------------------------
# 2) Pattern hits -> forward stats (SPY)
# ----------------------------
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

@register_card("backtest.pattern_forward_stats", "Backtest-lite: Pattern Forward Stats (SPY)", "backtest", min_tier="pro", cost=9, heavy=False, slots=("S11",))
def pattern_forward(ctx: CardContext) -> CardResult:
    t="SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-2000:].copy()
    if df.empty or len(df) < 400:
        return CardResult(
            key="backtest.pattern_forward_stats",
            title="Backtest-lite: Pattern Forward Stats (SPY)",
            summary="Not enough history for pattern stats."
        )

    close = df["Close"].astype(float)
    f5  = close.pct_change(5).shift(-5)
    f20 = close.pct_change(20).shift(-20)

    rows=[]
    for name, fn in PATTERNS.items():
        m = fn(df).fillna(False)
        a5 = f5[m].dropna()
        a20 = f20[m].dropna()
        if int(m.sum()) < 10:
            continue
        rows.append([
            name,
            int(m.sum()),
            float(a5.mean()*100) if len(a5) else float("nan"),
            float((a5>0).mean()*100) if len(a5) else float("nan"),
            float(a20.mean()*100) if len(a20) else float("nan"),
            float((a20>0).mean()*100) if len(a20) else float("nan"),
        ])

    if not rows:
        return CardResult(
            key="backtest.pattern_forward_stats",
            title="Backtest-lite: Pattern Forward Stats (SPY)",
            summary="No patterns had enough samples to score."
        )

    tab = pd.DataFrame(rows, columns=["Pattern","N","MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"]).sort_values("MeanFwd5%", ascending=False)

    out = tab.copy()
    for c in ["MeanFwd5%","WinFwd5%","MeanFwd20%","WinFwd20%"]:
        out[c] = out[c].map(lambda x: "" if not np.isfinite(x) else f"{x:.2f}")

    png = _table_png("Pattern Hits → Forward Return Stats (SPY)", out)

    bullets = [
        "Patterns are simple candle rules (no TA-Lib).",
        "Useful for ranking, not proof of causality; regime matters.",
    ]

    return CardResult(
        key="backtest.pattern_forward_stats",
        title="Backtest-lite: Pattern Forward Stats (SPY)",
        summary="Forward return stats for candle patterns on SPY.",
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="pattern_forward_stats.png", payload=png)]
    )

# ----------------------------
# 3) Mean-reversion equity curve (SPY)
# ----------------------------
@register_card("backtest.meanrev_equity_curve", "Backtest-lite: Mean-Reversion Equity Curve (SPY)", "backtest", min_tier="pro", cost=10, heavy=False, slots=("S10","S11"))
def meanrev_equity(ctx: CardContext) -> CardResult:
    t="SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-2500:].copy()
    close = df["Close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()
    dist = (close - ma20) / (ma20 + 1e-12)
    z = _zscore(dist, 120)

    # Strategy: long when z <= -2, exit when z >= 0
    pos = pd.Series(0.0, index=close.index)
    inpos = False
    for i in range(len(pos)):
        zi = z.iloc[i]
        if not np.isfinite(zi):
            pos.iloc[i] = 1.0 if inpos else 0.0
            continue
        if (not inpos) and (zi <= -2.0):
            inpos = True
        elif inpos and (zi >= 0.0):
            inpos = False
        pos.iloc[i] = 1.0 if inpos else 0.0

    ret = close.pct_change().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret
    eq = (1.0 + strat).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    fig = plt.figure(figsize=(10,6.6))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    ax1.plot(eq.values, label="Equity")
    ax1.set_title("Mean-Reversion Strategy (SPY): enter z<=-2, exit z>=0")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(dd.values, label="Drawdown")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left")

    ax3.plot(z.values, label="z(dist to MA20)")
    ax3.axhline(-2, linewidth=1)
    ax3.axhline(0, linewidth=1)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    stats = _perf_stats(eq)
    stats["Trades~"] = int(((pos.diff().abs() > 0).sum())/2)

    bullets = [
        "Toy backtest: no fees/slippage; signals use close prices.",
        "Meant as a baseline; integrate with your production entry/exit + risk rules."
    ]

    return CardResult(
        key="backtest.meanrev_equity_curve",
        title="Backtest-lite: Mean-Reversion Equity Curve (SPY)",
        summary="Baseline equity curve from a simple z-score mean-reversion rule.",
        metrics=stats,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="meanrev_equity.png", payload=png)]
    )