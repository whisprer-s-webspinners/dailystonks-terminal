from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@register_card("risk.volatility_regime_panel", "Volatility Regime Panel", "risk", min_tier="pro", cost=6, heavy=False, slots=("S09",))
def vol_regime(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-800:].copy()
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0)

    vol20 = ret.rolling(20).std(ddof=1) * math.sqrt(252)
    vol60 = ret.rolling(60).std(ddof=1) * math.sqrt(252)

    prev_close = close.shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atrp = atr14 / close

    # vol-of-vol proxy: std of daily sigma over 20d (annualized-ish)
    sigma_d = ret.rolling(20).std(ddof=1)
    vov = sigma_d.rolling(20).std(ddof=1)

    fig = plt.figure(figsize=(10,6.6))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.set_title(f"{t} Volatility Regime Panel")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(vol20.values, label="Realized vol 20D (ann.)")
    ax2.plot(vol60.values, label="Realized vol 60D (ann.)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    ax3.plot((atrp*100).values, label="ATR% 14D")
    ax3.plot((vov*100).values, label="Vol-of-vol proxy (x100)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "Vol20%": round(float(vol20.iloc[-1]*100), 2) if np.isfinite(vol20.iloc[-1]) else None,
        "Vol60%": round(float(vol60.iloc[-1]*100), 2) if np.isfinite(vol60.iloc[-1]) else None,
        "ATR14%": round(float(atrp.iloc[-1]*100), 2) if np.isfinite(atrp.iloc[-1]) else None,
        "VoV": round(float(vov.iloc[-1]*100), 3) if np.isfinite(vov.iloc[-1]) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="risk.volatility_regime_panel",
        title=f"{t}: Volatility Regime Panel",
        summary="Realized vol + ATR% + vol-of-vol proxy to characterize risk regime.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name=f"{t}_vol_regime.png", payload=png)]
    )

@register_card("risk.big_move_probability", "Big Move Probability (next day/week)", "risk", min_tier="pro", cost=4, heavy=False, slots=("S09",))
def big_move_prob(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-800:].copy()
    close = df["Close"].astype(float)
    ret = close.pct_change().dropna()

    sigma_d = float(ret.rolling(20).std(ddof=1).iloc[-1]) if len(ret) >= 25 else float("nan")
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        return CardResult(
            key="risk.big_move_probability",
            title=f"{t}: Big Move Probability",
            summary="Not enough data to estimate sigma.",
            warnings=["Need >= ~25 daily returns."]
        )

    def p_big(thresh: float, horizon_days: int):
        # assume iid normal, scale sigma by sqrt(h)
        sig = sigma_d * math.sqrt(horizon_days)
        z = thresh / sig
        return 2.0 * (1.0 - _norm_cdf(z))

    p2_1 = p_big(0.02, 1)
    p3_1 = p_big(0.03, 1)
    p2_5 = p_big(0.02, 5)
    p5_5 = p_big(0.05, 5)

    bullets = [
        "Heuristic using normal assumption + recent 20D sigma.",
        "Useful as a risk dial, not a prediction guarantee."
    ]
    metrics = {
        "sigma_d% (20D)": round(sigma_d*100, 2),
        "P(|1D|>2%)": round(p2_1*100, 1),
        "P(|1D|>3%)": round(p3_1*100, 1),
        "P(|1W|>2%)": round(p2_5*100, 1),
        "P(|1W|>5%)": round(p5_5*100, 1),
    }
    return CardResult(
        key="risk.big_move_probability",
        title=f"{t}: Big Move Probability",
        summary="Probability of exceeding move thresholds from recent volatility.",
        metrics=metrics,
        bullets=bullets
    )