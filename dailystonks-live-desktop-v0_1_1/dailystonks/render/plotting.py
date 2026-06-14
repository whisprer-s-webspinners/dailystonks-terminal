from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_candles(ax: plt.Axes, ohlcv: pd.DataFrame, *, title: str = "", max_bars: Optional[int] = 180):
    df = ohlcv.copy()
    if max_bars is not None and len(df) > max_bars:
        df = df.iloc[-max_bars:]
    x = np.arange(len(df))
    opens = df["Open"].values
    highs = df["High"].values
    lows  = df["Low"].values
    closes= df["Close"].values

    up = closes >= opens
    down = ~up

    # Wicks
    ax.vlines(x, lows, highs, linewidth=1)
    # Bodies
    body_low = np.minimum(opens, closes)
    body_h = np.abs(closes - opens)
    ax.bar(x[up], body_h[up], bottom=body_low[up], width=0.6)
    ax.bar(x[down], body_h[down], bottom=body_low[down], width=0.6, alpha=0.6)

    ax.set_title(title)
    ax.set_xlim(-1, len(df))
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    return df

def plot_line(ax: plt.Axes, x, y, *, label: Optional[str]=None):
    ax.plot(x, y, label=label)
    ax.grid(True, alpha=0.25)
    if label:
        ax.legend(loc="upper left")
