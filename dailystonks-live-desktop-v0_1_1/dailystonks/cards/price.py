from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema, rsi, macd
from ..render.plotting import plot_candles

@register_card("price.candles_basic", "Candles (Basic)", "price", min_tier="free", cost=2, heavy=False, slots=("S01","S02","S06"))
def candles_basic(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval)
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    plot_candles(ax, df, title=f"{t} Candles ({ctx.interval})")
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="price.candles_basic",
        title=f"{t}: Candlestick Chart",
        summary="Baseline price action view.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_candles.png", payload=png)]
    )

@register_card("price.ma20_blue_line", "Candles + MA20 (Blue Line)", "price", min_tier="basic", cost=3, heavy=False, slots=("S02","S06"))
def ma20_blue(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval)
    ma20 = df["Close"].rolling(20).mean()
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    d = plot_candles(ax, df, title=f"{t} Candles + MA20")
    ax.plot(range(len(d)), ma20.iloc[-len(d):].values, linewidth=2, label="MA20")
    ax.legend(loc="upper left")
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="price.ma20_blue_line",
        title=f"{t}: MA20 ‘Blue Line’ Overlay",
        summary="MA20 overlay used by reversal logic.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_ma20.png", payload=png)]
    )

@register_card("price.support_resistance", "Support/Resistance (Rolling)", "price", min_tier="pro", cost=4, heavy=False, slots=("S02","S06"))
def support_resistance(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval)
    win = 20
    sup = df["Low"].rolling(win).min()
    res = df["High"].rolling(win).max()
    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    d = plot_candles(ax, df, title=f"{t} Rolling S/R ({win})")
    ax.plot(range(len(d)), sup.iloc[-len(d):].values, linewidth=1.8, label="Support")
    ax.plot(range(len(d)), res.iloc[-len(d):].values, linewidth=1.8, label="Resistance")
    ax.legend(loc="upper left")
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="price.support_resistance",
        title=f"{t}: Rolling Support/Resistance",
        summary=f"Support=min(Low,{win}), Resistance=max(High,{win}).",
        artifacts=[Artifact(kind="image/png", name=f"{t}_sr.png", payload=png)]
    )

@register_card("price.candles_enhanced", "Candles (Enhanced: RSI/MACD/Volume)", "price", min_tier="basic", cost=6, heavy=False, slots=("S02","S06","S08"))
def candles_enhanced(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=ctx.interval)
    # Trim to last ~240 bars for readability
    dfp = df.iloc[-240:].copy()

    r = rsi(dfp["Close"])
    macd_line, macd_sig, macd_hist = macd(dfp["Close"])

    fig = plt.figure(figsize=(10,8.2))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    ax4 = fig.add_subplot(4,1,4, sharex=ax1)

    d = plot_candles(ax1, dfp, title=f"{t} Enhanced ({ctx.interval})", max_bars=None)
    x = range(len(d))
    ax1.plot(x, ema(d["Close"], 20).values, linewidth=1.5, label="EMA20")
    ax1.plot(x, ema(d["Close"], 50).values, linewidth=1.2, label="EMA50")
    ax1.legend(loc="upper left")

    ax2.plot(x, r.values, label="RSI(14)")
    ax2.axhline(70, linewidth=1)
    ax2.axhline(30, linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)

    ax3.plot(x, macd_line.values, label="MACD")
    ax3.plot(x, macd_sig.values, label="Signal")
    ax3.bar(list(x), macd_hist.values, alpha=0.4)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.25)

    ax4.bar(list(x), d["Volume"].values, alpha=0.6)
    ax4.set_title("Volume")
    ax4.grid(True, alpha=0.25)

    png = fig_to_png_bytes(fig)
    last_close = float(dfp["Close"].iloc[-1])
    return CardResult(
        key="price.candles_enhanced",
        title=f"{t}: Enhanced Candles + RSI/MACD",
        summary="Price + EMA stack + RSI + MACD + Volume.",
        metrics={"Close": round(last_close, 4)},
        artifacts=[Artifact(kind="image/png", name=f"{t}_enhanced.png", payload=png)]
    )
