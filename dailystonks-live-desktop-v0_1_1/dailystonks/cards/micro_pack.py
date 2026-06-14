
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
    need = {"Open","High","Low","Close"}
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

def _true_range(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    pc = c.shift(1)
    return pd.concat([
        (df["High"].astype(float) - df["Low"].astype(float)).abs(),
        (df["High"].astype(float) - pc).abs(),
        (df["Low"].astype(float) - pc).abs(),
    ], axis=1).max(axis=1)

@register_card("micro.gap_vs_intraday_decomp", "Micro: Gap vs Intraday Return Decomposition (SPY)", "micro", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def gap_vs_intraday(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-9000:].copy()
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)

    gap = (o - pc) / (pc + 1e-12)           # close->open
    intra = (c - o) / (o + 1e-12)           # open->close
    total = (c - pc) / (pc + 1e-12)         # close->close

    x = pd.DataFrame({"gap": gap, "intra": intra, "total": total}).dropna()
    if len(x) < 500:
        return CardResult(
            key="micro.gap_vs_intraday_decomp",
            title="Micro: Gap vs Intraday Return Decomposition (SPY)",
            summary="Not enough rows (need ~500+)."
        )

    # contribution via absolute return share
    contrib_gap = float((x["gap"].abs().sum()) / (x["total"].abs().sum() + 1e-12))
    contrib_intra = float((x["intra"].abs().sum()) / (x["total"].abs().sum() + 1e-12))

    # rolling means last 250
    rg = x["gap"].rolling(250).mean()
    ri = x["intra"].rolling(250).mean()
    rt = x["total"].rolling(250).mean()

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot((rg*100).values, label="Gap mean (250D)")
    ax1.plot((ri*100).values, label="Intraday mean (250D)")
    ax1.plot((rt*100).values, label="Total mean (250D)")
    ax1.set_title("Rolling mean returns (250D) — SPY")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    # cumulative contributions (sum)
    cg = (x["gap"].fillna(0)).cumsum()
    ci = (x["intra"].fillna(0)).cumsum()
    ct = (x["total"].fillna(0)).cumsum()
    ax2.plot((cg*100).values, label="Cum gap (%)")
    ax2.plot((ci*100).values, label="Cum intraday (%)")
    ax2.plot((ct*100).values, label="Cum total (%)")
    ax2.set_title("Cumulative return components (sum of daily %) — SPY")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "AbsContrib Gap": round(contrib_gap, 3),
        "AbsContrib Intraday": round(contrib_intra, 3),
        "MeanGap%": round(float(x["gap"].mean()*100), 3),
        "MeanIntra%": round(float(x["intra"].mean()*100), 3),
        "Obs": int(len(x)),
    }

    return CardResult(
        key="micro.gap_vs_intraday_decomp",
        title="Micro: Gap vs Intraday Return Decomposition (SPY)",
        summary="Decompose daily return into gap (C→O) and intraday (O→C).",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="gap_intraday_decomp.png", payload=png)]
    )

@register_card("micro.open_close_edge", "Micro: Open→Close vs Close→Open Edge (SPY)", "micro", min_tier="black", cost=6, heavy=False, slots=("S11",))
def open_close_edge(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-9000:].copy()
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)

    gap = (o - pc) / (pc + 1e-12)       # C->O
    intra = (c - o) / (o + 1e-12)       # O->C

    x = pd.DataFrame({"gap": gap, "intra": intra}).dropna()
    if len(x) < 800:
        return CardResult(
            key="micro.open_close_edge",
            title="Micro: Open→Close vs Close→Open Edge (SPY)",
            summary="Not enough rows (need ~800+)."
        )

    def stats(s: pd.Series):
        return {
            "Mean%": float(s.mean()*100),
            "Win%": float((s>0).mean()*100),
            "P05%": float(np.quantile(s.values, 0.05)*100),
            "P95%": float(np.quantile(s.values, 0.95)*100),
        }

    sg = stats(x["gap"])
    si = stats(x["intra"])

    tab = pd.DataFrame([
        ["Close→Open (gap)", f"{sg['Mean%']:+.3f}", f"{sg['Win%']:.1f}", f"{sg['P05%']:+.2f}", f"{sg['P95%']:+.2f}"],
        ["Open→Close (intra)", f"{si['Mean%']:+.3f}", f"{si['Win%']:.1f}", f"{si['P05%']:+.2f}", f"{si['P95%']:+.2f}"],
    ], columns=["Component","Mean%","Win%","P05%","P95%"])

    png_tbl = _table_png("Open→Close vs Close→Open stats (SPY)", tab)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.hist(x["gap"].values*100, bins=60, alpha=0.6, label="Gap (C→O)")
    ax.hist(x["intra"].values*100, bins=60, alpha=0.6, label="Intraday (O→C)")
    ax.set_title("Distribution of components (%) — SPY")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left")
    png_hist = fig_to_png_bytes(fig)

    metrics = {
        "Mean Gap%": round(sg["Mean%"], 3),
        "Mean Intra%": round(si["Mean%"], 3),
        "Win Gap%": round(sg["Win%"], 1),
        "Win Intra%": round(si["Win%"], 1),
        "Obs": int(len(x)),
    }

    return CardResult(
        key="micro.open_close_edge",
        title="Micro: Open→Close vs Close→Open Edge (SPY)",
        summary="Compare edge/shape of overnight gap vs intraday session returns.",
        metrics=metrics,
        artifacts=[
            Artifact(kind="image/png", name="oc_stats_table.png", payload=png_tbl),
            Artifact(kind="image/png", name="oc_hist.png", payload=png_hist),
        ]
    )

@register_card("micro.atr_percent_bands", "Micro: ATR% Regime Bands (SPY)", "micro", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def atr_percent_bands(ctx: CardContext) -> CardResult:
    df = _spy_df(ctx).iloc[-2600:].copy()
    close = df["Close"].astype(float)
    tr = _true_range(df)
    atr14 = tr.rolling(14).mean()
    atrp = (atr14 / (close + 1e-12)) * 100.0

    if atrp.dropna().shape[0] < 200:
        return CardResult(
            key="micro.atr_percent_bands",
            title="Micro: ATR% Regime Bands (SPY)",
            summary="Not enough ATR% observations."
        )

    q10 = float(np.quantile(atrp.dropna().values, 0.10))
    q50 = float(np.quantile(atrp.dropna().values, 0.50))
    q90 = float(np.quantile(atrp.dropna().values, 0.90))
    last = float(atrp.dropna().iloc[-1])

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(atrp.values, label="ATR% (14D)")
    ax.axhline(q10, linewidth=1, label="p10")
    ax.axhline(q50, linewidth=1, label="p50")
    ax.axhline(q90, linewidth=1, label="p90")
    ax.set_title("ATR% Regime Bands (SPY)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    png = fig_to_png_bytes(fig)

    # regime label
    if last <= q10:
        reg = "VeryLow"
    elif last <= q50:
        reg = "Low"
    elif last <= q90:
        reg = "High"
    else:
        reg = "VeryHigh"

    metrics = {
        "ATR% now": round(last, 3),
        "ATR% p10": round(q10, 3),
        "ATR% p50": round(q50, 3),
        "ATR% p90": round(q90, 3),
        "Regime": reg,
    }

    return CardResult(
        key="micro.atr_percent_bands",
        title="Micro: ATR% Regime Bands (SPY)",
        summary="ATR% bands for quick volatility regime context.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="atr_percent_bands.png", payload=png)]
    )
