from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def laguerre_filter(x: np.ndarray, gamma: float) -> np.ndarray:
    l0 = l1 = l2 = l3 = 0.0
    out = np.zeros_like(x, dtype=float)
    for i, price in enumerate(x):
        l0_new = (1 - gamma) * price + gamma * l0
        l1_new = -gamma * l0_new + l0 + gamma * l1
        l2_new = -gamma * l1_new + l1 + gamma * l2
        l3_new = -gamma * l2_new + l2 + gamma * l3
        l0, l1, l2, l3 = l0_new, l1_new, l2_new, l3_new
        out[i] = (l0 + 2*l1 + 2*l2 + l3) / 6.0
    return out

def kalman_1d(y: np.ndarray, q: float = 1e-5, r: float = 1e-2) -> np.ndarray:
    # Simple 1D Kalman smoother
    x = y[0]
    p = 1.0
    out = np.zeros_like(y, dtype=float)
    for i, z in enumerate(y):
        # predict
        p = p + q
        # update
        k = p / (p + r)
        x = x + k*(z - x)
        p = (1 - k)*p
        out[i] = x
    return out

@register_card("filter.laguerre_raw_vs_smooth", "Laguerre Smooth vs Raw", "filter", min_tier="pro", cost=6, heavy=False, slots=("S06","S08"))
def laguerre_vs_raw(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    y = df["Close"].values.astype(float)
    sm = laguerre_filter(y, gamma=0.7)
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, label="Close")
    ax.plot(sm, label="Laguerre(g=0.7)")
    ax.set_title(f"{t} Laguerre Smoothing")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="filter.laguerre_raw_vs_smooth",
        title=f"{t}: Laguerre Smooth vs Raw",
        summary="Laguerre filter reduces noise while tracking price.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_laguerre.png", payload=png)]
    )

@register_card("filter.laguerre_crossover_signals", "Laguerre Crossovers", "filter", min_tier="pro", cost=7, heavy=False, slots=("S08",))
def laguerre_cross(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    y = df["Close"].values.astype(float)
    fast = laguerre_filter(y, gamma=0.5)
    slow = laguerre_filter(y, gamma=0.8)
    cross_up = np.where((fast[1:] >= slow[1:]) & (fast[:-1] < slow[:-1]))[0] + 1
    cross_dn = np.where((fast[1:] <= slow[1:]) & (fast[:-1] > slow[:-1]))[0] + 1

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, label="Close", alpha=0.6)
    ax.plot(fast, label="Laguerre g=0.5")
    ax.plot(slow, label="Laguerre g=0.8")
    ax.scatter(cross_up, y[cross_up], marker="^", s=60, label="Cross Up")
    ax.scatter(cross_dn, y[cross_dn], marker="v", s=60, label="Cross Down")
    ax.set_title(f"{t} Laguerre Crossover Signals")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="filter.laguerre_crossover_signals",
        title=f"{t}: Laguerre Crossovers",
        summary="Crossovers of fast/slow Laguerre filters as signal proxy.",
        metrics={"cross_up": int(len(cross_up)), "cross_down": int(len(cross_dn))},
        artifacts=[Artifact(kind="image/png", name=f"{t}_lag_cross.png", payload=png)]
    )

@register_card("filter.adaptive_gamma_panel", "Adaptive Gamma Panel", "filter", min_tier="black", cost=9, heavy=False, slots=("S08",))
def adaptive_gamma(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    y = df["Close"].values.astype(float)
    # gamma based on rolling ATR% proxy
    tr = np.maximum(df["High"] - df["Low"], np.maximum((df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()))
    atr = tr.rolling(14).mean().fillna(method="bfill")
    atrp = (atr / df["Close"]).clip(0, 0.1).values
    gamma = 0.9 - 0.6*(atrp/0.1)  # high vol -> lower gamma (less smoothing)
    gamma = np.clip(gamma, 0.2, 0.9)
    sm = np.zeros_like(y)
    l0=l1=l2=l3=0.0
    for i, price in enumerate(y):
        g = float(gamma[i])
        l0n = (1-g)*price + g*l0
        l1n = -g*l0n + l0 + g*l1
        l2n = -g*l1n + l1 + g*l2
        l3n = -g*l2n + l2 + g*l3
        l0,l1,l2,l3 = l0n,l1n,l2n,l3n
        sm[i] = (l0+2*l1+2*l2+l3)/6.0

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(y, label="Close", alpha=0.6)
    ax1.plot(sm, label="Adaptive Laguerre")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.25)
    ax2.plot(gamma, label="gamma")
    ax2.set_ylim(0,1); ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.25)
    ax2.set_title("Gamma adapts inversely to ATR%")
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="filter.adaptive_gamma_panel",
        title=f"{t}: Adaptive Gamma Smoothing",
        summary="Adaptive Laguerre smoothing using ATR%-driven gamma.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_adaptive_gamma.png", payload=png)]
    )

@register_card("filter.kalman_smooth_overlay", "Kalman Smooth Overlay", "filter", min_tier="black", cost=7, heavy=False, slots=("S06","S08"))
def kalman_overlay(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    y = df["Close"].values.astype(float)
    sm = kalman_1d(y, q=1e-5, r=1e-2)
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, label="Close", alpha=0.6)
    ax.plot(sm, label="Kalman smooth")
    ax.set_title(f"{t} Kalman Smoothing Overlay")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="filter.kalman_smooth_overlay",
        title=f"{t}: Kalman Smooth Overlay",
        summary="Kalman smoothing can suppress noise and fakeouts.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_kalman.png", payload=png)]
    )
