from __future__ import annotations
import math
import numpy as np
import pandas as pd

from ..core.registry import register_card
from ..core.models import CardContext, CardResult

def _safe(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return round(float(x), nd)
    except Exception:
        return None

def _pct(x, nd=2):
    v = _safe(x * 100.0, nd)
    return v

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@register_card(
    "narrative.summary",
    "Narrative Summary: what changed / what matters / what to watch",
    "narrative",
    min_tier="free",
    cost=3,
    heavy=False,
    slots=("S01",),
)
def narrative_summary(ctx: CardContext) -> CardResult:
    warnings = []

    # SPY baseline
    spy = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = spy["Close"].astype(float).dropna()
    ret1 = close.pct_change()

    r1  = float(ret1.iloc[-1]) if len(ret1) >= 2 else float("nan")
    r5  = float(close.pct_change(5).iloc[-1]) if len(close) >= 7 else float("nan")
    r20 = float(close.pct_change(20).iloc[-1]) if len(close) >= 25 else float("nan")
    vol20 = float(ret1.rolling(20).std(ddof=1).iloc[-1] * math.sqrt(252)) if len(ret1) >= 25 else float("nan")

    prev_close = close.shift(1)
    tr = pd.concat([
        (spy["High"] - spy["Low"]).abs(),
        (spy["High"] - prev_close).abs(),
        (spy["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 20 else float("nan")
    atrp14 = float(atr14 / close.iloc[-1]) if np.isfinite(atr14) and close.iloc[-1] != 0 else float("nan")

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    slope20 = float(ema20.diff().iloc[-1]) if len(ema20) >= 3 else float("nan")
    slope50 = float(ema50.diff().iloc[-1]) if len(ema50) >= 3 else float("nan")
    trend = "Uptrend" if (np.isfinite(slope20) and np.isfinite(slope50) and slope20 > 0 and slope50 > 0) else \
            "Downtrend" if (np.isfinite(slope20) and np.isfinite(slope50) and slope20 < 0 and slope50 < 0) else \
            "Mixed/Range"

    # RSI14
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi14 = float(100 - (100 / (1 + rs)).iloc[-1]) if len(rs) >= 20 else float("nan")

    sigma_d = (vol20 / math.sqrt(252)) if np.isfinite(vol20) else float("nan")
    thresh = 0.02
    p_big = 2.0 * (1.0 - _norm_cdf(thresh / sigma_d)) if np.isfinite(sigma_d) and sigma_d > 0 else float("nan")

    # Breadth best-effort
    adv = dec = total = 0
    try:
        uni = ctx.sp500.tickers(max_n=min(ctx.max_universe, 120))
        data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")
        for raw in uni:
            tk = raw.replace(".", "-")
            df = data.get(tk)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            c = df["Close"].astype(float).dropna()
            if len(c) < 60:
                continue
            rr = float(c.pct_change().iloc[-1])
            adv += 1 if rr > 0 else 0
            dec += 1 if rr < 0 else 0
            total += 1
    except Exception as e:
        warnings.append(f"Breadth proxy failed: {e!r}")

    what_changed = [
        f"SPY daily return: {_pct(r1)}%",
        f"5D: {_pct(r5)}% · 20D: {_pct(r20)}%",
        f"Realized vol (20D ann.): {_pct(vol20,1)}% · ATR% (14): {_pct(atrp14,2)}%",
    ]
    if total >= 20:
        what_changed.append(f"Breadth: adv/dec {adv}/{dec} (N={total})")

    what_matters = [f"Trend bias: {trend}."]
    if np.isfinite(rsi14): what_matters.append(f"RSI14: {_safe(rsi14,1)}")
    if np.isfinite(p_big): what_matters.append(f"P(|1D|>2%): {_pct(p_big,1)}% (rough)")

    what_to_watch = [
        "Breadth falling while index rises can signal fragile rallies.",
        "Rising realized vol favors smaller size / defined-risk setups.",
        "In range regimes, MA-touch + reversal setups often beat breakouts.",
    ]

    bullets = (
        ["What changed"] + [f"- {x}" for x in what_changed] +
        ["What matters"] + [f"- {x}" for x in what_matters] +
        ["What to watch"] + [f"- {x}" for x in what_to_watch]
    )

    metrics = {
        "SPY 1D %": _pct(r1),
        "SPY 5D %": _pct(r5),
        "SPY 20D %": _pct(r20),
        "Vol20 ann %": _pct(vol20, 1),
        "ATR14 %": _pct(atrp14, 2),
        "Trend": trend,
        "RSI14": _safe(rsi14, 1),
        "P(|1D|>2%) %": _pct(p_big, 1),
    }

    return CardResult(
        key="narrative.summary",
        title="Narrative Summary",
        summary="High-signal synopsis from SPY + breadth proxy (best-effort).",
        metrics={k:v for k,v in metrics.items() if v is not None},
        bullets=bullets,
        warnings=warnings,
    )