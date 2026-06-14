from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

TRADING_DAYS = 252

def _perf_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 40:
        return {}
    ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(TRADING_DAYS) * ret.mean() / (ret.std(ddof=1) + 1e-12))
    cagr = float(eq.iloc[-1] ** (TRADING_DAYS / max(len(eq),1)) - 1.0)
    dd = (eq / eq.cummax()) - 1.0
    maxdd = float(dd.min())
    return {
        "CAGR%": round(cagr*100, 2),
        "Sharpe": round(sharpe, 2),
        "MaxDD%": round(maxdd*100, 2),
        "LastEq": round(float(eq.iloc[-1]), 4),
    }

def _monthly_rebalance_index(idx: pd.DatetimeIndex) -> pd.Series:
    m = idx.to_period("M")
    return (m != m.shift(1)).fillna(True)

def _basket_series(ctx: CardContext, tickers: list[str], bars: int = 1200) -> pd.DataFrame:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    frames = []
    used = []
    for raw in tickers:
        t = raw.replace(".", "-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].astype(float).rename(t).iloc[-bars:]
        frames.append(s)
        used.append(t)
    if len(frames) < 2:
        raise RuntimeError("Not enough series for basket.")
    prices = pd.concat(frames, axis=1).dropna(how="any")
    return prices

@register_card("port.equal_weight_basket", "Portfolio: Equal-Weight Basket", "portfolio", min_tier="pro", cost=9, heavy=False, slots=("S11",))
def equal_weight_basket(ctx: CardContext) -> CardResult:
    # If user provides >=2 tickers, use those; else default multi-asset basket
    tks = (ctx.tickers[:] if len(ctx.tickers) >= 2 else ["SPY","QQQ","IWM","TLT","GLD","BTC-USD"])[:8]
    tks = list(dict.fromkeys([t.replace(".","-").lstrip("$").upper() for t in tks]))

    prices = _basket_series(ctx, tks, bars=1400)
    rets = prices.pct_change().fillna(0)

    # monthly rebalance equal weight
    reb = _monthly_rebalance_index(prices.index)
    n = prices.shape[1]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    cur = np.ones(n) / n
    for i, d in enumerate(prices.index):
        if reb.iloc[i]:
            cur = np.ones(n) / n
        w.iloc[i] = cur

    port = (w.shift(1).fillna(method="bfill") * rets).sum(axis=1)
    eq = (1 + port).cumprod()

    dd = (eq / eq.cummax()) - 1.0

    fig = plt.figure(figsize=(10,5.8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(eq.values, label="Equity")
    ax1.set_title("Equal-Weight Basket (monthly rebalance)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")
    ax2.plot(dd.values, label="Drawdown")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left")
    png = fig_to_png_bytes(fig)

    stats = _perf_stats(eq)
    stats["Assets"] = ", ".join(prices.columns.tolist())

    return CardResult(
        key="port.equal_weight_basket",
        title="Portfolio: Equal-Weight Basket",
        summary="Fast baseline portfolio with monthly rebalance (equal weight).",
        metrics=stats,
        artifacts=[Artifact(kind="image/png", name="equal_weight_basket.png", payload=png)]
    )

@register_card("port.risk_parity_lite_weights", "Portfolio: Risk-Parity Lite (weights)", "portfolio", min_tier="black", cost=9, heavy=False, slots=("S11",))
def risk_parity_weights(ctx: CardContext) -> CardResult:
    tks = (ctx.tickers[:] if len(ctx.tickers) >= 2 else ["SPY","QQQ","IWM","TLT","GLD","BTC-USD"])[:10]
    tks = list(dict.fromkeys([t.replace(".","-").lstrip("$").upper() for t in tks]))

    prices = _basket_series(ctx, tks, bars=520)
    rets = prices.pct_change().dropna()

    # inverse vol weights (60D), normalized
    vol = rets.rolling(60).std(ddof=1).iloc[-1]
    inv = 1.0 / (vol + 1e-12)
    w = inv / inv.sum()

    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.bar(w.index.tolist(), w.values*100)
    ax.set_title("Risk-Parity Lite Weights (inverse 60D vol)")
    ax.grid(True, alpha=0.25, axis="y")
    png = fig_to_png_bytes(fig)

    show = pd.DataFrame({
        "Asset": w.index,
        "Weight%": (w.values*100).round(2),
        "Vol60%": (vol.values*np.sqrt(TRADING_DAYS)*100).round(1),
    }).sort_values("Weight%", ascending=False)

    metrics = {f"W{i+1}": f"{r.Asset}:{r.Weight_pct:.2f}%" for i, r in enumerate(show.rename(columns={"Weight%":"Weight_pct"}).itertuples()) if i < 6}

    bullets = [
        "Weights are purely inverse-vol (60D) — no correlations, no constraints.",
        "Good default for ‘stable basket’ customization; add caps if desired."
    ]

    return CardResult(
        key="port.risk_parity_lite_weights",
        title="Portfolio: Risk-Parity Lite (weights)",
        summary="Inverse-vol weights from recent returns (best-effort baseline).",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="risk_parity_weights.png", payload=png),
        ]
    )

@register_card("port.sector_tilt_rotation", "Portfolio: Sector Tilt (20D momentum)", "portfolio", min_tier="black", cost=11, heavy=True, slots=("S11",))
def sector_tilt(ctx: CardContext) -> CardResult:
    # Sector ETFs (US). XLRE sometimes spotty; keep it included but tolerate failures.
    sectors = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU","XLRE"]
    prices = _basket_series(ctx, sectors, bars=260)

    mom20 = prices.pct_change(20).iloc[-1]
    mom20 = mom20.sort_values(ascending=False)

    topn = 3
    top = mom20.head(topn)
    chosen = top.index.tolist()

    # Equal weight chosen sectors
    rets = prices[chosen].pct_change().fillna(0)
    eq = (1 + rets.mean(axis=1)).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.bar(mom20.index.tolist(), mom20.values*100)
    ax1.set_title("Sector Momentum (20D %)")
    ax1.grid(True, alpha=0.25, axis="y")
    ax2.plot(eq.values, label=f"Equal-weight Top{topn}: {', '.join(chosen)}")
    ax2.plot(dd.values, label="Drawdown")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)
    png = fig_to_png_bytes(fig)

    show = pd.DataFrame({"SectorETF": mom20.index, "Mom20%": (mom20.values*100).round(2)})
    table_png = fig_to_png_bytes(_table_fig("Sector Momentum Table (20D)", show.head(12)))

    stats = _perf_stats(eq)
    stats["Chosen"] = ", ".join(chosen)

    return CardResult(
        key="port.sector_tilt_rotation",
        title="Portfolio: Sector Tilt (20D momentum)",
        summary="Top-sector momentum tilt (equal-weight top sectors).",
        metrics=stats,
        artifacts=[
            Artifact(kind="image/png", name="sector_tilt.png", payload=png),
        ]
    )

def _table_fig(title: str, df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig