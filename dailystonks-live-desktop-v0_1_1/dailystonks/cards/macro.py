from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("macro.cross_asset_overlay", "Cross-Asset Overlay (normalized)", "macro", min_tier="pro", cost=9, heavy=False, slots=("S03",))
def cross_asset(ctx: CardContext) -> CardResult:
    # Use robust proxies:
    # - UUP as DXY proxy (more reliable than DX-Y.NYB)
    basket = ["SPY", "BTC-USD", "UUP", "TLT", "GLD"]

    data = ctx.market.get_ohlcv_many(basket, start=ctx.start, end=ctx.end, interval="1d")
    frames = []
    missing = []
    for t in basket:
        mt = t.replace(".", "-")  # mapping mirror
        if mt in data and not data[mt].empty and "Close" in data[mt].columns:
            frames.append(data[mt]["Close"].rename(mt))
        else:
            missing.append(t)

    warnings = []
    if missing:
        warnings.append(f"Missing series (skipped): {', '.join(missing)}")

    if len(frames) < 2:
        return CardResult(
            key="macro.cross_asset_overlay",
            title="Macro & Cross-Asset Overlay",
            summary="Not enough series downloaded to build overlay (need >=2).",
            warnings=warnings
        )

    prices = pd.concat(frames, axis=1).dropna(how="any")
    if prices.empty or len(prices) < 10:
        return CardResult(
            key="macro.cross_asset_overlay",
            title="Macro & Cross-Asset Overlay",
            summary="Downloaded series did not overlap enough to plot.",
            warnings=warnings
        )

    normed = prices / prices.iloc[0]

    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    for c in normed.columns:
        ax.plot(normed.index, normed[c].values, label=c)
    ax.set_title("Macro Overlay (normalized to 1.0 at start)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    png = fig_to_png_bytes(fig)

    # rolling corr vs SPY if available
    metrics = {}
    if "SPY" in prices.columns:
        rets = prices.pct_change().dropna()
        for col in prices.columns:
            if col == "SPY":
                continue
            try:
                metrics[f"Corr60(SPY,{col})"] = round(float(rets["SPY"].rolling(60).corr(rets[col]).iloc[-1]), 3)
            except Exception:
                pass

    return CardResult(
        key="macro.cross_asset_overlay",
        title="Macro & Cross-Asset Overlay",
        summary="Normalized price overlay + rolling correlations vs SPY (when available).",
        metrics=metrics,
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="macro_overlay.png", payload=png)]
    )