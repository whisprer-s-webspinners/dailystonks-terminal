
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _spy_df(ctx: CardContext) -> pd.DataFrame:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d")
    if df is None or df.empty:
        raise RuntimeError("SPY OHLCV missing/empty")
    need = {"Open","High","Low","Close","Volume"}
    if not need.issubset(df.columns):
        raise RuntimeError("SPY OHLCV missing required columns")
    return df.copy()

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _zscore(s: pd.Series, win: int = 60) -> pd.Series:
    m = s.rolling(win, min_periods=max(20, win//2)).mean()
    sd = s.rolling(win, min_periods=max(20, win//2)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

@register_card("volume.volume_regime_bands", "Volume: Regime Bands (SPY)", "volume", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def volume_regime_bands(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float).replace(0, np.nan).dropna()
    if len(vol) < 260:
        return CardResult(
            key="volume.volume_regime_bands",
            title="Volume: Regime Bands (SPY)",
            summary="Not enough data (need ~260+ bars)."
        )

    # rolling bands
    v = vol.reindex(df.index).astype(float)
    med = v.rolling(252, min_periods=120).median()
    p10 = v.rolling(252, min_periods=120).quantile(0.10)
    p90 = v.rolling(252, min_periods=120).quantile(0.90)
    z = _zscore(v, 60)

    last_v = float(v.dropna().iloc[-1])
    last_med = float(med.dropna().iloc[-1])
    last_p10 = float(p10.dropna().iloc[-1])
    last_p90 = float(p90.dropna().iloc[-1])
    last_z = float(z.dropna().iloc[-1])

    # regime label
    if last_v <= last_p10:
        reg = "VeryLow"
    elif last_v <= last_med:
        reg = "Low"
    elif last_v <= last_p90:
        reg = "High"
    else:
        reg = "VeryHigh"

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.set_title(f"SPY Volume Regime — {reg}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(v.values, label="Volume")
    ax2.plot(med.values, label="Median(252)")
    ax2.plot(p10.values, label="p10(252)")
    ax2.plot(p90.values, label="p90(252)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=4, fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "Regime": reg,
        "Vol_now": int(last_v),
        "Vol_z60": round(last_z, 2),
        "p10": int(last_p10),
        "median": int(last_med),
        "p90": int(last_p90),
    }

    return CardResult(
        key="volume.volume_regime_bands",
        title="Volume: Regime Bands (SPY)",
        summary="Rolling volume bands + current regime classification.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="volume_regime.png", payload=png)]
    )

@register_card("volume.volume_surprise_zscore", "Volume: Surprise Z-Score + Spikes (SPY)", "volume", min_tier="black", cost=6, heavy=False, slots=("S11",))
def volume_surprise(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float).replace(0, np.nan)
    r = close.pct_change()

    z = _zscore(vol, 60)
    x = pd.DataFrame({"vol": vol, "z": z, "ret": r}).dropna()
    if len(x) < 260:
        return CardResult(
            key="volume.volume_surprise_zscore",
            title="Volume: Surprise Z-Score + Spikes (SPY)",
            summary="Not enough data for z-score (need ~260+)."
        )

    last_z = float(x["z"].iloc[-1])

    # spikes
    spikes = x.sort_values("z", ascending=False).head(12)
    tab = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in spikes.index],
        "Vol_z60": [f"{v:.2f}" for v in spikes["z"].values],
        "Ret%": [f"{v*100:+.2f}" for v in spikes["ret"].values],
    })
    png_tbl = _table_png("Top Volume Surprise Days (SPY)", tab)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x["z"].iloc[-600:].values, label="Vol z-score (60D)")
    ax.axhline(2, linewidth=1, label="+2")
    ax.axhline(0, linewidth=1)
    ax.axhline(-2, linewidth=1, label="-2")
    ax.set_title(f"Volume Surprise Z-Score (60D) — now {last_z:.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    png_line = fig_to_png_bytes(fig)

    # percentile (rank of last within last 252)
    recent = x["z"].iloc[-252:]
    pctl = float((recent.rank(pct=True).iloc[-1]) * 100.0)

    metrics = {
        "Vol_z60 now": round(last_z, 2),
        "z percentile(252)": round(pctl, 1),
        "Obs": int(len(x)),
    }

    bullets = [
        "High z-score often corresponds to news/earnings/macro shock days.",
        "Useful as a ‘why did it move’ context tool.",
    ]

    return CardResult(
        key="volume.volume_surprise_zscore",
        title="Volume: Surprise Z-Score + Spikes (SPY)",
        summary="Volume z-score series + top spike days list.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="volume_z_line.png", payload=png_line),
            Artifact(kind="image/png", name="volume_spikes.png", payload=png_tbl),
        ]
    )

@register_card("liquidity.amihud_illiquidity", "Liquidity: Amihud Illiquidity Proxy (SPY)", "liquidity", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def amihud(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float).replace(0, np.nan)
    r = close.pct_change().abs()

    dollar_vol = (close * vol).replace(0, np.nan)
    illiq = (r / (dollar_vol + 1e-12)) * 1e6  # scaled

    x = illiq.dropna()
    if len(x) < 400:
        return CardResult(
            key="liquidity.amihud_illiquidity",
            title="Liquidity: Amihud Illiquidity Proxy (SPY)",
            summary="Not enough data for illiquidity proxy."
        )

    med = x.rolling(252, min_periods=120).median()
    p10 = x.rolling(252, min_periods=120).quantile(0.10)
    p90 = x.rolling(252, min_periods=120).quantile(0.90)

    last = float(x.iloc[-1])
    last_p10 = float(p10.dropna().iloc[-1])
    last_med = float(med.dropna().iloc[-1])
    last_p90 = float(p90.dropna().iloc[-1])

    if last <= last_p10:
        reg = "VeryLiquid"
    elif last <= last_med:
        reg = "Liquid"
    elif last <= last_p90:
        reg = "Illiquid"
    else:
        reg = "VeryIlliquid"

    fig = plt.figure(figsize=(10,6.0))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.set_title(f"Amihud Illiquidity Proxy — {reg}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(illiq.values, label="Illiq (scaled)")
    ax2.plot(med.values, label="Median(252)")
    ax2.plot(p10.values, label="p10(252)")
    ax2.plot(p90.values, label="p90(252)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=4, fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "Regime": reg,
        "Illiq now": round(last, 4),
        "p10": round(last_p10, 4),
        "median": round(last_med, 4),
        "p90": round(last_p90, 4),
    }

    bullets = [
        "Amihud proxy ~ |return| / $volume (scaled). Higher = worse liquidity / higher price impact.",
        "Good for context when vol spikes feel ‘thin’ vs ‘thick’.",
    ]

    return CardResult(
        key="liquidity.amihud_illiquidity",
        title="Liquidity: Amihud Illiquidity Proxy (SPY)",
        summary="Liquidity regime proxy from |ret| divided by dollar volume.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="amihud_illiquidity.png", payload=png)]
    )
