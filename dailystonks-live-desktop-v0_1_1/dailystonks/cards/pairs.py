from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("pairs.cointegration_spread_zscore", "Cointegration Spread + Z-score", "pairs", min_tier="black", cost=10, heavy=True, slots=("S11",))
def cointegration_card(ctx: CardContext) -> CardResult:
    tks = (ctx.tickers + ["SPY","QQQ"])[:2]
    a,b = tks[0], tks[1]
    dfa = ctx.market.get_ohlcv(a, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    dfb = ctx.market.get_ohlcv(b, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    pa = dfa["Close"].rename("A")
    pb = dfb["Close"].rename("B")
    df = pd.concat([pa,pb], axis=1).dropna()
    score,pval,_ = coint(df["A"].values, df["B"].values)
    # simple hedge ratio via OLS B ~ beta*A
    beta = np.polyfit(df["A"].values, df["B"].values, 1)[0]
    spread = df["B"] - beta*df["A"]
    z = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std(ddof=1) + 1e-12)

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(spread.values, label="Spread (B - beta*A)")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.25)
    ax2.plot(z.values, label="Z(60)")
    ax2.axhline(2, linewidth=1); ax2.axhline(-2, linewidth=1)
    ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.25)
    ax2.set_title("Z-score (mean reversion bands)")
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="pairs.cointegration_spread_zscore",
        title=f"Pairs: {a} vs {b} Cointegration",
        summary="Cointegration test + spread and z-score for mean reversion.",
        metrics={"p-value": round(float(pval),4), "beta": round(float(beta),4), "z_last": round(float(z.iloc[-1]),3)},
        artifacts=[Artifact(kind="image/png", name="coint_spread.png", payload=png)]
    )
