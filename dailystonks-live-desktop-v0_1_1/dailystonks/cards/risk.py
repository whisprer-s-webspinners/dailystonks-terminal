from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema

def monte_carlo_paths(returns: np.ndarray, *, n_paths: int = 1000, horizon: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed if seed != 0 else None)
    boot = rng.choice(returns, size=(n_paths, horizon), replace=True)
    paths = np.cumprod(1.0 + boot, axis=1)
    return paths

def max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / peak
    return float(dd.min())

@register_card("risk.reversal_risk_heatmap", "Reversal Risk Heatmap", "risk", min_tier="black", cost=8, heavy=False, slots=("S04",))
def reversal_heat(ctx: CardContext) -> CardResult:
    # Batch download to avoid 60+ separate Yahoo hits
    tickers = ctx.sp500.tickers(max_n=ctx.max_universe)
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")

    scores = []
    names = []
    fails = 0

    for raw in tickers:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1
            continue
        try:
            d = df.iloc[-120:].copy()
            close = d["Close"]
            ma20 = close.rolling(20).mean().iloc[-1]
            ewo_now = float(ema(close, 5).iloc[-1] - ema(close, 35).iloc[-1])
            ewo_prev = float(ema(close, 5).iloc[-2] - ema(close, 35).iloc[-2])
            e_slope = ewo_now - ewo_prev
            dist = abs(float(close.iloc[-1]) - float(ma20)) / float(ma20) if float(ma20) else np.nan
            score = (1.0 - min(dist / 0.06, 1.0)) * (1.0 + np.tanh(abs(ewo_now) * 5)) * (1.0 + np.tanh(abs(e_slope) * 20))
            scores.append(score)
            names.append(tk)
        except Exception:
            fails += 1
            continue

    warnings = []
    if fails:
        warnings.append(f"Skipped {fails} symbols due to missing/failed data.")

    if len(scores) < 10:
        return CardResult(
            key="risk.reversal_risk_heatmap",
            title="Universe: Reversal Potential Heatmap",
            summary=f"Not enough symbols computed for a heatmap (got {len(scores)}).",
            warnings=warnings
        )

    n = len(scores)
    cols = 10
    rows = int(np.ceil(n / cols))
    grid = np.full((rows, cols), np.nan)
    labels = [["" for _ in range(cols)] for __ in range(rows)]

    order = np.argsort(scores)[::-1]
    for idx, oi in enumerate(order):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        grid[r, c] = scores[oi]
        labels[r][c] = names[oi]

    fig = plt.figure(figsize=(10, 1.0 + 0.6 * rows))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(grid, aspect="auto", origin="upper")
    ax.set_title("Reversal Potential Heatmap (Top scores)")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(rows):
        for c in range(cols):
            if labels[r][c]:
                ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="risk.reversal_risk_heatmap",
        title="Universe: Reversal Potential Heatmap",
        summary="Heuristic score based on distance-to-MA20 and EWO dynamics (batched fetch).",
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="reversal_heatmap.png", payload=png)]
    )

# Keep your other risk cards in their existing file if you have them.
# If you already have a bigger risk.py, DO NOT overwrite it with this file.
# (If you do, you lose other risk cards.)