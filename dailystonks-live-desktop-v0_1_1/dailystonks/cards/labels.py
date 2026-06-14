from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes
from ..render.plotting import plot_candles

def triple_barrier_labels(close: pd.Series, pt: float, sl: float, horizon: int):
    # Label +1 if PT hit first, -1 if SL hit first, 0 otherwise by horizon.
    c = close.values.astype(float)
    n = len(c)
    lab = np.zeros(n, dtype=int)
    for i in range(n - horizon - 1):
        entry = c[i]
        future = c[i+1:i+1+horizon]
        up = (future - entry) / entry
        dn = (entry - future) / entry
        up_hit = np.where(up >= pt)[0]
        dn_hit = np.where(dn >= sl)[0]
        t_up = up_hit[0] if len(up_hit) else 10**9
        t_dn = dn_hit[0] if len(dn_hit) else 10**9
        if t_up < t_dn:
            lab[i] = 1
        elif t_dn < t_up:
            lab[i] = -1
        else:
            lab[i] = 0
    return pd.Series(lab, index=close.index)

@register_card("labels.triple_barrier_visual", "Triple Barrier Visual", "labels", min_tier="black", cost=9, heavy=False, slots=("S06",))
def barrier_visual(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-200:].copy()
    close = df["Close"]
    pt = 0.05; sl = 0.03; horizon = 20
    entry_i = len(close) - horizon - 2
    entry = float(close.iloc[entry_i])
    upper = entry * (1+pt)
    lower = entry * (1-sl)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    d = plot_candles(ax, df, title=f"{t} Triple Barrier Example", max_bars=None)
    x = np.arange(len(d))
    ax.axhline(upper, linewidth=1.5, label=f"PT +{pt:.0%}")
    ax.axhline(lower, linewidth=1.5, label=f"SL -{sl:.0%}")
    ax.axvline(entry_i, linewidth=1, label="Entry")
    ax.axvline(entry_i+horizon, linewidth=1, label="Horizon")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="labels.triple_barrier_visual",
        title=f"{t}: Triple Barrier Visual",
        summary="One entry point with profit-take, stop-loss, and time barrier.",
        metrics={"PT": pt, "SL": sl, "Horizon": horizon},
        artifacts=[Artifact(kind="image/png", name="triple_barrier.png", payload=png)]
    )

@register_card("labels.label_strip_timeline", "Label Strip Timeline", "labels", min_tier="black", cost=9, heavy=False, slots=("S07",))
def label_strip(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    lab = triple_barrier_labels(close, pt=0.04, sl=0.03, horizon=15).iloc[-240:]

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(lab.values.reshape(1,-1), aspect="auto")
    ax.set_title(f"{t} Label Strip (+1 / 0 / -1)")
    ax.set_yticks([])
    ax.set_xticks([])
    png = fig_to_png_bytes(fig)

    counts = {"+1": int((lab==1).sum()), "0": int((lab==0).sum()), "-1": int((lab==-1).sum())}
    return CardResult(
        key="labels.label_strip_timeline",
        title=f"{t}: Triple-Barrier Label Strip",
        summary="Visual sanity-check of labels used for ML training.",
        metrics=counts,
        artifacts=[Artifact(kind="image/png", name="label_strip.png", payload=png)]
    )
