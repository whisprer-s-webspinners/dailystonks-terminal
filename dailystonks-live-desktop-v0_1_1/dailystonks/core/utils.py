from __future__ import annotations
import base64
from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def b64_png(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def safe_pct(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return 100.0 * (a - b) / b
