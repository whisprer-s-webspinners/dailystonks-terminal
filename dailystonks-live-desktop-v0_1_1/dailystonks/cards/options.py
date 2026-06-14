from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("options.binary_payoff_plots", "Binary Payoff Plots", "options", min_tier="pro", cost=4, heavy=False, slots=("S12",))
def binary_payoff(ctx: CardContext) -> CardResult:
    S = np.linspace(50, 150, 200)
    K = 100
    call = (S > K).astype(float)
    put = (S < K).astype(float)

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(S, call, label="Binary Call payoff")
    ax.plot(S, put, label="Binary Put payoff")
    ax.set_title("Binary Option Payoffs")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="options.binary_payoff_plots",
        title="Options: Binary Payoff Shapes",
        summary="Simple payoff diagrams (educational + quick intuition).",
        artifacts=[Artifact(kind="image/png", name="binary_payoff.png", payload=png)]
    )

def binary_call_price(S, K, r, sigma, T):
    # Binary call: price = e^{-rT} * N(d2)
    d2 = (log(S/K) + (r - 0.5*sigma*sigma)*T) / (sigma*sqrt(T) + 1e-12)
    return exp(-r*T)*norm.cdf(d2)

@register_card("options.binary_pricing_panel", "Binary Pricing Panel", "options", min_tier="black", cost=4, heavy=False, slots=("S12",))
def binary_pricing(ctx: CardContext) -> CardResult:
    # Example panel (can be tied to live underlying later)
    S = 100
    K = 100
    r = 0.05
    sigma = 0.25
    T = 30/365
    price = binary_call_price(S,K,r,sigma,T)

    bullets = [
        "Binary call price = e^{-rT} * N(d2).",
        "Panel uses example parameters; wire S from live underlying to operationalize."
    ]
    return CardResult(
        key="options.binary_pricing_panel",
        title="Options: Binary Pricing Panel",
        summary="Closed-form binary call fair value estimate.",
        metrics={"S": S, "K": K, "r": r, "sigma": sigma, "T_years": round(T,4), "Price": round(price,4)},
        bullets=bullets
    )

@register_card("options.chain_snapshot_iv", "Option Chain Snapshot (yfinance)", "options", min_tier="pro", cost=8, heavy=True, slots=("S12",))
def chain_snapshot(ctx: CardContext) -> CardResult:
    # Requires network + yfinance option chain
    import pandas as pd
    import yfinance as yf

    t = ctx.tickers[0] if ctx.tickers else "SPY"
    tk = yf.Ticker(t)
    expiries = tk.options
    if not expiries:
        raise RuntimeError(f"No option expiries for {t}.")
    exp0 = expiries[0]
    chain = tk.option_chain(exp0)
    calls = chain.calls.head(10)[["strike","lastPrice","bid","ask","impliedVolatility","volume","openInterest"]]
    puts  = chain.puts.head(10)[["strike","lastPrice","bid","ask","impliedVolatility","volume","openInterest"]]

    fig = plt.figure(figsize=(10,7.0))
    ax1 = fig.add_subplot(2,1,1); ax1.axis("off")
    ax2 = fig.add_subplot(2,1,2); ax2.axis("off")
    ax1.set_title(f"{t} Calls (top 10) exp {exp0}")
    ax2.set_title(f"{t} Puts (top 10) exp {exp0}")
    t1 = ax1.table(cellText=calls.round(4).values, colLabels=calls.columns, loc="center")
    t2 = ax2.table(cellText=puts.round(4).values, colLabels=puts.columns, loc="center")
    t1.scale(1, 1.2); t2.scale(1, 1.2)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="options.chain_snapshot_iv",
        title=f"{t}: Option Chain Snapshot",
        summary="Quick view of near-term option chain (top rows).",
        artifacts=[Artifact(kind="image/png", name=f"{t}_chain.png", payload=png)]
    )
