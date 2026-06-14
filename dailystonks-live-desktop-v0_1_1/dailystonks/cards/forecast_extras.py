from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("forecast.fan_chart_drift_vol", "Fan Chart (drift+vol baseline)", "forecast", min_tier="pro", cost=7, heavy=False, slots=("S10",))
def fan_chart(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-1000:].copy()
    close = df["Close"].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < 120:
        return CardResult(
            key="forecast.fan_chart_drift_vol",
            title=f"{t}: Fan Chart",
            summary="Not enough history for drift/vol estimate.",
            warnings=["Need >= ~120 daily returns."]
        )

    mu = float(ret.tail(252).mean())           # daily drift
    sig = float(ret.tail(252).std(ddof=1))     # daily vol
    S0 = float(close.iloc[-1])

    horizon = 30  # trading days
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]

    # lognormal approximation: log S ~ N(log S0 + (mu - 0.5 sig^2) h, sig^2 h)
    hs = np.arange(0, horizon+1)
    med = np.zeros_like(hs, dtype=float)
    bands = {q: np.zeros_like(hs, dtype=float) for q in qs}

    for i,h in enumerate(hs):
        m = math.log(S0) + (mu - 0.5*sig*sig)*h
        v = (sig*sig)*h
        sd = math.sqrt(v) if v > 0 else 0.0
        # quantiles in log-space
        for q in qs:
            z = {0.1:-1.2816, 0.25:-0.6745, 0.5:0.0, 0.75:0.6745, 0.9:1.2816}[q]
            bands[q][i] = math.exp(m + z*sd)
        med[i] = bands[0.5][i]

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(hs, med, label="median")
    ax.fill_between(hs, bands[0.25], bands[0.75], alpha=0.25, label="50% band")
    ax.fill_between(hs, bands[0.10], bands[0.90], alpha=0.15, label="80% band")
    ax.set_title(f"{t} Fan Chart (30D) — drift+vol baseline")
    ax.set_xlabel("Days ahead")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    png = fig_to_png_bytes(fig)

    metrics = {
        "S0": round(S0, 4),
        "mu_d%": round(mu*100, 3),
        "sig_d%": round(sig*100, 3),
        "median_30D": round(float(med[-1]), 4),
        "p10_30D": round(float(bands[0.1][-1]), 4),
        "p90_30D": round(float(bands[0.9][-1]), 4),
    }

    return CardResult(
        key="forecast.fan_chart_drift_vol",
        title=f"{t}: Fan Chart (baseline)",
        summary="Non-AI baseline fan chart from historical drift+vol (lognormal approx).",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name=f"{t}_fan.png", payload=png)]
    )