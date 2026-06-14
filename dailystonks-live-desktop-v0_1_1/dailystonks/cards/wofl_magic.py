from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema
from ..render.plotting import plot_candles

def ewo(close: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    return ema(close, fast) - ema(close, slow)

def zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=max(30, win//3)).mean()
    sd = s.rolling(win, min_periods=max(30, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _fmt_ok(name: str, ok: bool, extra: str = "") -> str:
    mark = "✅" if ok else "❌"
    return f"{mark} {name}{(' — ' + extra) if extra else ''}"

@register_card(
    "reversal.wofl_magic_explainer",
    "Wofl Magic Reversal — score + rule audit",
    "reversal",
    min_tier="black",
    cost=6,
    heavy=False,
    slots=("S07","S06"),
)
def wofl_magic_explainer(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    if df.empty or len(df) < 80:
        return CardResult(
            key="reversal.wofl_magic_explainer",
            title=f"{t}: Wofl Magic Reversal",
            summary="Not enough daily history to score (need ~80+ bars).",
            warnings=[f"bars={len(df)}"]
        )

    close = df["Close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()

    e = ewo(close)
    ez = zscore(e, win=120)
    eslope = e.diff()

    last_close = float(close.iloc[-1])
    last_ma20  = float(ma20.iloc[-1]) if np.isfinite(ma20.iloc[-1]) else float("nan")
    dist = (last_close - last_ma20) / last_ma20 if np.isfinite(last_ma20) and last_ma20 != 0 else float("nan")
    abs_dist = abs(dist) if np.isfinite(dist) else float("nan")

    ez_last = float(ez.iloc[-1]) if np.isfinite(ez.iloc[-1]) else float("nan")
    es_last = float(eslope.iloc[-1]) if np.isfinite(eslope.iloc[-1]) else float("nan")

    # ---- Rule checks (heuristic but consistent + explainable) ----
    near_ma = np.isfinite(abs_dist) and (abs_dist <= 0.03)  # within 3%
    below_ma = np.isfinite(dist) and (dist < 0)
    above_ma = np.isfinite(dist) and (dist > 0)

    ewo_low_extreme  = np.isfinite(ez_last) and (ez_last <= -1.0)
    ewo_high_extreme = np.isfinite(ez_last) and (ez_last >= +1.0)

    ewo_rising  = np.isfinite(es_last) and (es_last > 0)
    ewo_falling = np.isfinite(es_last) and (es_last < 0)

    bottom_checks = {
        "Near MA20 (≤3%)": near_ma,
        "Below MA20": below_ma,
        "EWO extreme low (z≤-1)": ewo_low_extreme,
        "EWO rising": ewo_rising,
    }
    top_checks = {
        "Near MA20 (≤3%)": near_ma,
        "Above MA20": above_ma,
        "EWO extreme high (z≥+1)": ewo_high_extreme,
        "EWO falling": ewo_falling,
    }

    bottom_pass = sum(1 for v in bottom_checks.values() if v)
    top_pass    = sum(1 for v in top_checks.values() if v)

    # Decide verdict
    if bottom_pass >= 3 and bottom_pass > top_pass:
        verdict = "BOTTOM candidate"
        direction = "bottom"
    elif top_pass >= 3 and top_pass > bottom_pass:
        verdict = "TOP candidate"
        direction = "top"
    elif bottom_pass == top_pass and bottom_pass >= 3:
        verdict = "AMBIGUOUS (both score high)"
        direction = "ambiguous"
    else:
        verdict = "No candidate"
        direction = "none"

    # ---- Composite score (0..100) ----
    proximity = 1.0 - _clip01((abs_dist / 0.03)) if np.isfinite(abs_dist) else 0.0
    extremity = _clip01(abs(ez_last) / 2.0) if np.isfinite(ez_last) else 0.0

    # slope score scaled by recent typical slope
    slope_ref = float(eslope.abs().rolling(60, min_periods=20).quantile(0.8).iloc[-1])
    if np.isfinite(es_last) and np.isfinite(slope_ref) and slope_ref > 0:
        slope_score = _clip01(abs(es_last) / slope_ref)
    else:
        slope_score = 0.0

    rule_strength = max(bottom_pass, top_pass) / 4.0
    score = 100.0 * (0.55 * rule_strength + 0.25 * proximity + 0.15 * extremity + 0.05 * slope_score)
    score = float(max(0.0, min(100.0, score)))

    # ---- Bullets / explanation ----
    bullets = []
    bullets.append("Rule audit (Bottom candidate)")
    for k,v in bottom_checks.items():
        bullets.append(_fmt_ok(k, v))
    bullets.append("Rule audit (Top candidate)")
    for k,v in top_checks.items():
        bullets.append(_fmt_ok(k, v))

    bullets.append("Interpretation")
    bullets.append(f"- Verdict: **{verdict}** (passes: bottom {bottom_pass}/4, top {top_pass}/4)")
    bullets.append(f"- Proximity to MA20: {abs_dist*100:.2f}% (target ≤3%)" if np.isfinite(abs_dist) else "- Proximity to MA20: n/a")
    bullets.append(f"- EWO z-score: {ez_last:.2f} (extreme if |z|≥1)" if np.isfinite(ez_last) else "- EWO z-score: n/a")
    bullets.append(f"- EWO slope: {es_last:.6f} (rising supports bottoms; falling supports tops)" if np.isfinite(es_last) else "- EWO slope: n/a")

    # ---- Compact chart (last ~180 bars) ----
    dfp = df.iloc[-180:].copy()
    ma20p = dfp["Close"].rolling(20, min_periods=20).mean()
    ep = ewo(dfp["Close"])

    fig = plt.figure(figsize=(10,6.6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    d = plot_candles(ax1, dfp, title=f"{t} — Wofl Magic Reversal", max_bars=None)
    x = np.arange(len(d))
    ax1.plot(x, ma20p.values, linewidth=2, label="MA20")
    ax1.scatter([len(d)-1], [float(dfp['Close'].iloc[-1])], s=40, label="Last")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    ax2.bar(x, ep.values, alpha=0.6)
    ax2.axhline(0, linewidth=1)
    ax2.set_title("EWO (EMA5-EMA35)")
    ax2.grid(True, alpha=0.25)

    png = fig_to_png_bytes(fig)

    metrics = {
        "Verdict": verdict,
        "Score": round(score, 1),
        "Close": round(last_close, 4),
        "MA20": round(last_ma20, 4) if np.isfinite(last_ma20) else None,
        "dist_to_MA20%": round(abs_dist*100, 2) if np.isfinite(abs_dist) else None,
        "EWO_z": round(ez_last, 2) if np.isfinite(ez_last) else None,
        "EWO_slope": round(es_last, 6) if np.isfinite(es_last) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="reversal.wofl_magic_explainer",
        title=f"{t}: Wofl Magic Reversal — Explainer",
        summary="Composite score + explicit rule audit for bottom/top reversal candidates.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name=f"{t}_wofl_magic.png", payload=png)]
    )