from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, rsi

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

@register_card("bt.backtrader_equity_curve_analyzers", "Equity Curve + Stats (SMA cross)", "backtest", min_tier="pro", cost=9, heavy=False, slots=("S09",))
def equity_curve(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-1200:].copy()
    close = df["Close"]
    fast = sma(close, 20)
    slow = sma(close, 100)
    pos = (fast > slow).astype(int).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strat = pos * ret
    eq = (1 + strat).cumprod()

    dd = eq / eq.cummax() - 1.0
    sharpe = float(np.sqrt(252) * strat.mean() / (strat.std(ddof=1) + 1e-12))
    cagr = float(eq.iloc[-1] ** (252/len(eq)) - 1)

    fig = plt.figure(figsize=(10,5.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(eq.values, label="Equity")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.25)
    ax2.plot(dd.values, label="Drawdown")
    ax2.legend(loc="lower left"); ax2.grid(True, alpha=0.25)
    ax2.set_title("Drawdown")
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="bt.backtrader_equity_curve_analyzers",
        title=f"{t}: SMA Cross Backtest (light)",
        summary="Lightweight backtest proxy (replaces Backtrader plot for now).",
        metrics={"CAGR": round(cagr,4), "Sharpe": round(sharpe,3), "MaxDD": round(float(dd.min()),4)},
        artifacts=[Artifact(kind="image/png", name="equity.png", payload=png)]
    )

@register_card("bt.multi_asset_backtest_pack", "Multi-Asset Equity (equal weight)", "backtest", min_tier="black", cost=12, heavy=True, slots=("S11",))
def multi_asset(ctx: CardContext) -> CardResult:
    # Equal-weight monthly rebalance on given tickers (or defaults).
    tks = (ctx.tickers + ["SPY","QQQ","IWM","TLT"])[:6]
    frames=[]
    used=[]
    for tk in tks:
        try:
            df = ctx.market.get_ohlcv(tk, start=ctx.start, end=ctx.end, interval="1d").iloc[-1200:]
            frames.append(df["Close"].rename(tk))
            used.append(tk)
        except Exception:
            continue
    if len(frames) < 2:
        raise RuntimeError("Need >=2 tickers for multi-asset equity.")
    prices = pd.concat(frames, axis=1).dropna()
    rets = prices.pct_change().fillna(0)

    # monthly rebalance weights
    dates = prices.index
    month = dates.to_period("M")
    rebalance = month != month.shift(1)
    w = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    cur = np.ones(len(prices.columns)) / len(prices.columns)
    for i, d in enumerate(dates):
        if rebalance.iloc[i]:
            cur = np.ones(len(prices.columns)) / len(prices.columns)
        w.iloc[i] = cur
    port = (w.shift(1).fillna(method="bfill") * rets).sum(axis=1)
    eq = (1 + port).cumprod()

    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(eq.values, label="Portfolio equity")
    ax.set_title(f"Equal-Weight Portfolio (monthly rebalance): {', '.join(used)}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    cagr = float(eq.iloc[-1] ** (252/len(eq)) - 1)
    dd = float((eq/eq.cummax() - 1).min())
    return CardResult(
        key="bt.multi_asset_backtest_pack",
        title="Multi-Asset Equity Pack",
        summary="Equal-weight portfolio with monthly rebalance (fast baseline).",
        metrics={"CAGR": round(cagr,4), "MaxDD": round(dd,4)},
        artifacts=[Artifact(kind="image/png", name="multi_asset.png", payload=png)]
    )

@register_card("bt.vectorbt_param_sweep_heatmap", "Param Sweep Heatmap (RSI)", "backtest", min_tier="black", cost=14, heavy=True, slots=("S11",))
def param_sweep(ctx: CardContext) -> CardResult:
    # Sweep RSI thresholds to create a heatmap of total return.
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-1200:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)

    lows = list(range(20, 41, 5))
    highs = list(range(60, 81, 5))
    mat = np.full((len(lows), len(highs)), np.nan)

    r = rsi(close).fillna(50)
    for i, lo in enumerate(lows):
        for j, hi in enumerate(highs):
            buy = r < lo
            sell = r > hi
            pos = np.zeros(len(r), dtype=int)
            holding = 0
            for k in range(len(r)):
                if holding == 0 and buy.iloc[k]:
                    holding = 1
                elif holding == 1 and sell.iloc[k]:
                    holding = 0
                pos[k] = holding
            strat = pd.Series(pos, index=ret.index).shift(1).fillna(0) * ret
            mat[i,j] = float((1+strat).prod() - 1)

    fig = plt.figure(figsize=(9.2,6.2))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_title(f"{t} RSI Param Sweep (Total Return)")
    ax.set_xlabel("Sell threshold (hi)")
    ax.set_ylabel("Buy threshold (lo)")
    ax.set_xticks(range(len(highs))); ax.set_xticklabels(highs)
    ax.set_yticks(range(len(lows))); ax.set_yticklabels(lows)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = fig_to_png_bytes(fig)

    best = np.nanmax(mat)
    return CardResult(
        key="bt.vectorbt_param_sweep_heatmap",
        title=f"{t}: RSI Param Sweep Heatmap",
        summary="Heatmap of total return across buy/sell RSI thresholds.",
        metrics={"best_total_return": round(float(best),4)},
        artifacts=[Artifact(kind="image/png", name="rsi_sweep.png", payload=png)]
    )
