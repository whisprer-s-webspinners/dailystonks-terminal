
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _spy_close(ctx: CardContext) -> pd.Series:
    df = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").copy()
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("SPY OHLCV missing/empty")
    c = df["Close"].astype(float).dropna()
    return c

def _ret1(close: pd.Series) -> pd.Series:
    return close.pct_change()

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(8, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

@register_card("seasonality.dow_profile", "Seasonality: Day-of-Week Profile (SPY)", "seasonality", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def dow_profile(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-6000:]
    r = _ret1(close).dropna()
    if len(r) < 250:
        return CardResult(
            key="seasonality.dow_profile",
            title="Seasonality: Day-of-Week Profile (SPY)",
            summary="Not enough data (need ~250+ daily returns)."
        )

    # Monday=0 ... Friday=4
    dow = r.index.weekday
    g = pd.DataFrame({"r": r.values, "dow": dow}).groupby("dow")["r"].agg(["count", "mean", lambda x: (x>0).mean()])
    g.columns = ["N", "Mean", "WinRate"]
    names = ["Mon","Tue","Wed","Thu","Fri"]
    g.index = [names[i] for i in g.index]

    fig = plt.figure(figsize=(10,5.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.bar(g.index.tolist(), (g["Mean"]*100).values)
    ax1.set_title("SPY mean daily return by weekday (%)")
    ax1.grid(True, alpha=0.25, axis="y")

    ax2.bar(g.index.tolist(), (g["WinRate"]*100).values)
    ax2.set_title("SPY win-rate by weekday (%)")
    ax2.grid(True, alpha=0.25, axis="y")
    png = fig_to_png_bytes(fig)

    best_mean = g["Mean"].idxmax()
    worst_mean = g["Mean"].idxmin()
    metrics = {
        "BestMeanDay": best_mean,
        "WorstMeanDay": worst_mean,
        "BestMean%": round(float(g.loc[best_mean,"Mean"]*100), 3),
        "WorstMean%": round(float(g.loc[worst_mean,"Mean"]*100), 3),
    }

    tab = g.copy()
    out = pd.DataFrame({
        "Day": tab.index,
        "N": tab["N"].astype(int).values,
        "Mean%": (tab["Mean"]*100).map(lambda x: f"{x:+.3f}").values,
        "Win%": (tab["WinRate"]*100).map(lambda x: f"{x:.1f}").values,
    })
    png_tbl = _table_png("Day-of-Week Table (SPY)", out)

    return CardResult(
        key="seasonality.dow_profile",
        title="Seasonality: Day-of-Week Profile (SPY)",
        summary="Weekday mean returns and win-rate for SPY.",
        metrics=metrics,
        artifacts=[
            Artifact(kind="image/png", name="dow_profile.png", payload=png),
            Artifact(kind="image/png", name="dow_table.png", payload=png_tbl),
        ],
    )

@register_card("seasonality.month_profile", "Seasonality: Month-of-Year Profile (SPY)", "seasonality", min_tier="pro", cost=5, heavy=False, slots=("S11",))
def month_profile(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-8000:]
    r = _ret1(close).dropna()
    if len(r) < 500:
        return CardResult(
            key="seasonality.month_profile",
            title="Seasonality: Month-of-Year Profile (SPY)",
            summary="Not enough data (need ~500+ daily returns)."
        )

    m = r.index.month
    g = pd.DataFrame({"r": r.values, "m": m}).groupby("m")["r"].agg(["count", "mean", lambda x: (x>0).mean()])
    g.columns = ["N", "Mean", "WinRate"]
    g.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = plt.figure(figsize=(10,5.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.bar(g.index.tolist(), (g["Mean"]*100).values)
    ax1.set_title("SPY mean daily return by month (%)")
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.set_xticklabels(g.index.tolist(), rotation=30, ha="right")

    ax2.bar(g.index.tolist(), (g["WinRate"]*100).values)
    ax2.set_title("SPY win-rate by month (%)")
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.set_xticklabels(g.index.tolist(), rotation=30, ha="right")

    png = fig_to_png_bytes(fig)

    best = g["Mean"].idxmax()
    worst = g["Mean"].idxmin()
    metrics = {
        "BestMeanMonth": best,
        "WorstMeanMonth": worst,
        "BestMean%": round(float(g.loc[best,"Mean"]*100), 3),
        "WorstMean%": round(float(g.loc[worst,"Mean"]*100), 3),
    }

    out = pd.DataFrame({
        "Month": g.index,
        "N": g["N"].astype(int).values,
        "Mean%": (g["Mean"]*100).map(lambda x: f"{x:+.3f}").values,
        "Win%": (g["WinRate"]*100).map(lambda x: f"{x:.1f}").values,
    })
    png_tbl = _table_png("Month-of-Year Table (SPY)", out)

    return CardResult(
        key="seasonality.month_profile",
        title="Seasonality: Month-of-Year Profile (SPY)",
        summary="Month-of-year mean returns and win-rate for SPY.",
        metrics=metrics,
        artifacts=[
            Artifact(kind="image/png", name="month_profile.png", payload=png),
            Artifact(kind="image/png", name="month_table.png", payload=png_tbl),
        ],
    )

@register_card("seasonality.turnaround_effects", "Seasonality: Turnaround & Calendar Effects (SPY)", "seasonality", min_tier="black", cost=6, heavy=False, slots=("S11",))
def turnaround_effects(ctx: CardContext) -> CardResult:
    close = _spy_close(ctx).iloc[-9000:]
    r = _ret1(close).dropna()
    if len(r) < 800:
        return CardResult(
            key="seasonality.turnaround_effects",
            title="Seasonality: Turnaround & Calendar Effects (SPY)",
            summary="Not enough data (need ~800+ daily returns)."
        )

    df = pd.DataFrame({"r": r})
    df["dow"] = df.index.weekday  # 0..4
    df["mon_r"] = df["r"].where(df["dow"]==0)
    df["tue_r"] = df["r"].where(df["dow"]==1)
    df["wed_r"] = df["r"].where(df["dow"]==2)
    df["thu_r"] = df["r"].where(df["dow"]==3)
    df["fri_r"] = df["r"].where(df["dow"]==4)

    # Conditions
    mon_down = (df["mon_r"] < 0)
    thu_down = (df["thu_r"] < 0)

    # Tue after Mon down (classic “turnaround Tuesday” proxy)
    tue_after_mon_down = df["tue_r"][mon_down.shift(1).fillna(False)].dropna()

    # Fri after Thu down (bounce into weekend)
    fri_after_thu_down = df["fri_r"][thu_down.shift(1).fillna(False)].dropna()

    # First/last trading day of month effects
    pm = df.index.to_period("M")
    is_first = pm != pm.shift(1)
    is_last = pm != pm.shift(-1)
    first_r = df["r"][is_first].dropna()
    last_r = df["r"][is_last].dropna()

    def stat(series: pd.Series):
        if series is None or len(series)==0:
            return (0, np.nan, np.nan)
        return (int(len(series)), float(series.mean()*100), float((series>0).mean()*100))

    rows=[]
    rows.append(("Tue after Mon down",) + stat(tue_after_mon_down))
    rows.append(("Fri after Thu down",) + stat(fri_after_thu_down))
    rows.append(("All Mondays",) + stat(df["mon_r"].dropna()))
    rows.append(("All Fridays",) + stat(df["fri_r"].dropna()))
    rows.append(("First trading day (month)",) + stat(first_r))
    rows.append(("Last trading day (month)",) + stat(last_r))

    out = pd.DataFrame(rows, columns=["Effect","N","Mean%","Win%"])
    out2 = out.copy()
    out2["Mean%"] = out2["Mean%"].map(lambda x: "" if not np.isfinite(x) else f"{x:+.3f}")
    out2["Win%"] = out2["Win%"].map(lambda x: "" if not np.isfinite(x) else f"{x:.1f}")

    png_tbl = _table_png("Turnaround & Calendar Effects (SPY)", out2)

    # quick bar chart of Mean%
    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    vals = [0 if (not np.isfinite(x)) else x for x in out["Mean%"].values]
    ax.bar(out["Effect"].values, vals)
    ax.set_title("Mean return (%) by effect (SPY)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_xticklabels(out["Effect"].values, rotation=25, ha="right")
    png_bar = fig_to_png_bytes(fig)

    metrics = {}
    if np.isfinite(out.loc[out["Effect"]=="Tue after Mon down","Mean%"].iloc[0]):
        metrics["TueAfterMonDown Mean%"] = round(float(out.loc[out["Effect"]=="Tue after Mon down","Mean%"].iloc[0]), 3)
    if np.isfinite(out.loc[out["Effect"]=="First trading day (month)","Mean%"].iloc[0]):
        metrics["FirstDay Mean%"] = round(float(out.loc[out["Effect"]=="First trading day (month)","Mean%"].iloc[0]), 3)

    return CardResult(
        key="seasonality.turnaround_effects",
        title="Seasonality: Turnaround & Calendar Effects (SPY)",
        summary="Conditional day effects + first/last trading day of month.",
        metrics=metrics,
        artifacts=[
            Artifact(kind="image/png", name="seasonality_effects_table.png", payload=png_tbl),
            Artifact(kind="image/png", name="seasonality_effects_bar.png", payload=png_bar),
        ],
        bullets=[
            "These are descriptive stats (not a guarantee).",
            "Use with regime filters (vol/trend) if you want tighter behavior.",
        ]
    )
