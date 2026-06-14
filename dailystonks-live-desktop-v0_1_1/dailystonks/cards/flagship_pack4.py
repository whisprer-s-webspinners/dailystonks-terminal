
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

TRADING_DAYS = 252

def _zscore(s: pd.Series, win: int = 252) -> pd.Series:
    m = s.rolling(win, min_periods=max(60, win//3)).mean()
    sd = s.rolling(win, min_periods=max(60, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _table_png(title: str, df: pd.DataFrame, scale_y: float = 1.4) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, scale_y)
    return fig_to_png_bytes(fig)

def _true_range(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    pc = c.shift(1)
    return pd.concat([
        (df["High"].astype(float) - df["Low"].astype(float)).abs(),
        (df["High"].astype(float) - pc).abs(),
        (df["Low"].astype(float) - pc).abs(),
    ], axis=1).max(axis=1)

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1600) -> pd.DataFrame:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    frames = []
    for raw in tickers:
        t = str(raw).upper().strip().lstrip("$").replace(".", "-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames.append(df["Close"].astype(float).rename(t).iloc[-bars:])
    if len(frames) < 2:
        raise RuntimeError("Not enough series returned.")
    return pd.concat(frames, axis=1).dropna(how="any")

# ---------------------------------------------------------
# 1) Earnings season pressure gauge (no earnings dates)
# ---------------------------------------------------------
@register_card("macro.earnings_pressure_gauge", "Macro: Earnings Season Pressure Gauge (S&P cap)", "macro",
               min_tier="black", cost=10, heavy=True, slots=("S03",))
def earnings_pressure(ctx: CardContext) -> CardResult:
    spdf = ctx.sp500.df()
    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 350)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    fails=0
    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty:
            fails += 1; continue
        need = {"Open","High","Low","Close"}
        if not need.issubset(df.columns):
            fails += 1; continue

        df = df.iloc[-520:].copy()
        if len(df) < 120:
            fails += 1; continue

        o = df["Open"].astype(float)
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        pc = c.shift(1)

        gap = (o - pc) / (pc + 1e-12)              # overnight gap C->O
        ret = c.pct_change()                        # close->close
        tr = _true_range(df)
        atr14 = tr.rolling(14).mean()
        atrp = (atr14 / (c + 1e-12))                # ATR%

        g = gap.dropna()
        r = ret.dropna()
        if len(g) < 60 or len(r) < 60:
            fails += 1; continue

        p95_gap = float(g.abs().quantile(0.95))
        sig20 = float(r.rolling(20).std(ddof=1).iloc[-1]) if np.isfinite(r.rolling(20).std(ddof=1).iloc[-1]) else float("nan")
        gap_score = float(p95_gap / (sig20 + 1e-12)) if np.isfinite(sig20) else float("nan")

        atrp_last = float(atrp.dropna().iloc[-1]) if atrp.dropna().shape[0] else float("nan")
        if not (np.isfinite(gap_score) and np.isfinite(atrp_last)):
            fails += 1; continue

        # simple combined pressure score (ranked later)
        rows.append((tk, gap_score, p95_gap, atrp_last))

    if not rows:
        return CardResult(
            key="macro.earnings_pressure_gauge",
            title="Macro: Earnings Season Pressure Gauge (S&P cap)",
            summary="No valid symbols (data coverage too low).",
            warnings=[f"fails={fails}"]
        )

    df = pd.DataFrame(rows, columns=["Symbol","GapScore","P95Gap","ATR%"]).sort_values("GapScore", ascending=False)
    ok = len(df)

    # rank-normalize for a composite
    df["GapScorePct"] = df["GapScore"].rank(pct=True)
    df["ATRpPct"] = df["ATR%"].rank(pct=True)
    df["Pressure"] = 0.6*df["GapScorePct"] + 0.4*df["ATRpPct"]

    # buckets (desk-y headline)
    # High event pressure: Pressure >= 0.80
    # Medium: 0.60-0.80
    # Low: < 0.60
    hi = int((df["Pressure"] >= 0.80).sum())
    med = int(((df["Pressure"] >= 0.60) & (df["Pressure"] < 0.80)).sum())
    lo = int((df["Pressure"] < 0.60).sum())

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.bar(["High pressure","Medium","Low"], [hi, med, lo])
    ax.set_title(f"Earnings/Event Pressure Proxy — universe ok={ok}, fails={fails}")
    ax.grid(True, alpha=0.25, axis="y")
    png_bar = fig_to_png_bytes(fig)

    top = df.sort_values("Pressure", ascending=False).head(20).copy()
    top_disp = pd.DataFrame({
        "Symbol": top["Symbol"].values,
        "Pressure(0-1)": top["Pressure"].map(lambda x: f"{x:.3f}").values,
        "GapScore": top["GapScore"].map(lambda x: f"{x:.2f}").values,
        "P95 |gap| %": top["P95Gap"].map(lambda x: f"{x*100:.2f}").values,
        "ATR%": top["ATR%"].map(lambda x: f"{x*100:.2f}").values,
    })
    png_tbl = _table_png("Top 20 ‘event pressure’ names (gap + ATR% proxy)", top_disp)

    # headline
    pressure_index = float(df["Pressure"].mean())
    if pressure_index >= 0.75:
        label = "HIGH"
    elif pressure_index >= 0.60:
        label = "ELEVATED"
    else:
        label = "NORMAL"

    metrics = {
        "PressureIndex": round(pressure_index, 3),
        "Label": label,
        "HighPressureCount": hi,
        "UniverseOK": ok,
        "Fails": fails,
    }

    bullets = [
        "This is an earnings/event *pressure proxy* using gap behavior + ATR% (no earnings calendar needed).",
        "GapScore = P95(|gap|) / sigma20 (bigger = more ‘gap-prone’ vs typical volatility).",
    ]

    return CardResult(
        key="macro.earnings_pressure_gauge",
        title="Macro: Earnings Season Pressure Gauge (S&P cap)",
        summary="Index-wide ‘event pressure’ dial + top risk names list.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="earnings_pressure_bar.png", payload=png_bar),
            Artifact(kind="image/png", name="earnings_pressure_top20.png", payload=png_tbl),
        ]
    )

# ---------------------------------------------------------
# 2) Macro grid correlation heatmap (cross-asset)
# ---------------------------------------------------------
@register_card("macro.macro_grid_corr_heatmap", "Macro: Cross-Asset Macro Grid (Corr + Returns)", "macro",
               min_tier="black", cost=7, heavy=False, slots=("S03",))
def macro_grid(ctx: CardContext) -> CardResult:
    # ETF/proxy basket (avoid fragile indices):
    # VIXY used as vol proxy (ETF), BTC-USD for crypto (often available)
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","HYG","UUP","VIXY","BTC-USD"]
    px = _get_closes(ctx, tickers, bars=1200)
    rets = px.pct_change().dropna()
    if len(rets) < 90:
        return CardResult(
            key="macro.macro_grid_corr_heatmap",
            title="Macro: Cross-Asset Macro Grid (Corr + Returns)",
            summary="Not enough overlapping returns for macro grid.",
            warnings=[f"rows={len(rets)} cols={px.columns.tolist()}"]
        )

    # Corr over last 60
    corr = rets.iloc[-60:].corr()

    fig = plt.figure(figsize=(10,7.2))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(corr.values, aspect="auto", origin="upper")
    ax.set_title("Cross-Asset Correlation Grid (last 60 trading days)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index.tolist())

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png_heat = fig_to_png_bytes(fig)

    # Returns/vol table
    last20 = px.pct_change(20).iloc[-1]
    vol20 = rets.rolling(20).std(ddof=1).iloc[-1] * math.sqrt(TRADING_DAYS)
    tab = pd.DataFrame({
        "Asset": px.columns,
        "20D%": [f"{last20[a]*100:+.2f}" if np.isfinite(last20[a]) else "" for a in px.columns],
        "Vol20%": [f"{vol20[a]*100:.1f}" if np.isfinite(vol20[a]) else "" for a in px.columns],
        "Corr60 vs SPY": [f"{corr.loc['SPY',a]:+.2f}" if ('SPY' in corr.index and a in corr.columns) else "" for a in px.columns],
    })
    png_tbl = _table_png("Macro Grid Quick Table (20D return, vol, corr vs SPY)", tab, scale_y=1.35)

    # simple headline tags
    tags=[]
    if "VIXY" in px.columns and "SPY" in corr.columns:
        c = float(corr.loc["SPY","VIXY"])
        tags.append(f"SPY~VIXY corr60 {c:+.2f}")
    if "HYG" in px.columns and "TLT" in px.columns:
        tags.append("Credit + duration in grid")
    if "BTC-USD" in px.columns:
        tags.append("BTC included")

    bullets = [
        "Macro grid = cross-asset corr matrix (60D) plus a quick returns/vol table.",
        "Use it to detect regime shifts: correlations tightening, diversifiers failing, vol proxy coupling, etc.",
    ] + [f"Tag: {t}" for t in tags]

    return CardResult(
        key="macro.macro_grid_corr_heatmap",
        title="Macro: Cross-Asset Macro Grid (Corr + Returns)",
        summary="Cross-asset correlation grid + quick returns/vol table.",
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="macro_grid_corr.png", payload=png_heat),
            Artifact(kind="image/png", name="macro_grid_table.png", payload=png_tbl),
        ]
    )
