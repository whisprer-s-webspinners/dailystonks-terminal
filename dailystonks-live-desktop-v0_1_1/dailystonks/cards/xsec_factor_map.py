
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, ema

TRADING_DAYS = 252

def _robust_z(x: pd.Series) -> pd.Series:
    # median/MAD z-score, clipped
    m = float(np.nanmedian(x.values))
    mad = float(np.nanmedian(np.abs(x.values - m))) + 1e-12
    z = (x - m) / (1.4826 * mad)
    return z.clip(-3, 3)

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _sector_map(ctx: CardContext) -> dict[str,str]:
    sp = ctx.sp500.df()
    if "GICS Sector" in sp.columns:
        sec = dict(zip(sp["Symbol"].astype(str), sp["GICS Sector"].astype(str)))
    elif "Sector" in sp.columns:
        sec = dict(zip(sp["Symbol"].astype(str), sp["Sector"].astype(str)))
    else:
        sec = {}
    # normalize tickers
    out={}
    for k,v in sec.items():
        out[k.replace(".","-").upper().strip()] = v
    return out

def _universe(ctx: CardContext, cap: int) -> list[str]:
    sp = ctx.sp500.df()
    syms = sp["Symbol"].astype(str).tolist()
    out=[]
    for s in syms[:cap]:
        t = s.replace(".","-").upper().strip()
        if t and t not in out:
            out.append(t)
    return out

def _trend_r2(logp: pd.Series, win: int = 90) -> float:
    y = logp.dropna().iloc[-win:]
    if len(y) < win:
        return float("nan")
    x = np.arange(win, dtype=float)
    sx = x.sum(); sxx = (x*x).sum()
    sy = float(y.sum()); sxy = float((x*y.values).sum())
    denom = (win*sxx - sx*sx) + 1e-12
    b1 = (win*sxy - sx*sy) / denom
    b0 = (sy - b1*sx) / win
    yhat = b0 + b1*x
    ss_res = float(np.sum((y.values - yhat)**2))
    ss_tot = float(np.sum((y.values - y.values.mean())**2)) + 1e-12
    return float(1.0 - ss_res/ss_tot)

@register_card("xsec.tech_factor_map", "Cross-Section: Technical Factor Map (S&P cap)", "xsec",
               min_tier="black", cost=12, heavy=False, slots=("S03","S11"))
def tech_factor_map(ctx: CardContext) -> CardResult:
    cap = min(int(getattr(ctx, "max_universe", 200) or 200), 220)
    uni = _universe(ctx, cap)
    secmap = _sector_map(ctx)

    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    ok=0
    fails=0
    for tk in uni:
        df = data.get(tk)
        if df is None or df.empty:
            fails += 1; continue
        need = {"Close","Volume"}
        if not need.issubset(df.columns):
            fails += 1; continue

        df = df.iloc[-700:].copy()
        c = df["Close"].astype(float).dropna()
        if len(c) < 300:
            fails += 1; continue

        v = df["Volume"].astype(float).replace(0, np.nan)

        # Factors (direction normalized so "higher = stronger"):
        # Momentum: 12-1 (exclude last ~1 month)
        if len(c) < 260:
            fails += 1; continue
        mom = (c.iloc[-21] / (c.iloc[-252] + 1e-12)) - 1.0

        # Low vol: negative realized vol (60D)
        r = c.pct_change().dropna()
        vol60 = float(r.iloc[-60:].std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) >= 60 else float("nan")
        lowvol = -vol60

        # Trend quality: R^2 of log price fit (90D)
        logp = np.log(c.replace(0, np.nan))
        trq = _trend_r2(logp, 90)

        # Mean reversion: oversold score = negative z(dist to MA20) over 120D
        ma20 = c.rolling(20, min_periods=20).mean()
        dist = (c - ma20) / (ma20 + 1e-12)
        dz = (dist - dist.rolling(120, min_periods=60).mean()) / (dist.rolling(120, min_periods=60).std(ddof=1) + 1e-12)
        meanrev = -float(dz.dropna().iloc[-1]) if dz.dropna().shape[0] else float("nan")

        # Liquidity: log dollar volume (20D mean)
        dv = (c * v).rolling(20, min_periods=10).mean()
        liq = float(np.log(dv.dropna().iloc[-1] + 1e-12)) if dv.dropna().shape[0] else float("nan")

        if not (np.isfinite(mom) and np.isfinite(lowvol) and np.isfinite(trq) and np.isfinite(meanrev) and np.isfinite(liq)):
            fails += 1; continue

        ok += 1
        rows.append((tk, secmap.get(tk, "Unknown"), mom, lowvol, trq, meanrev, liq))

    if ok < 60:
        return CardResult(
            key="xsec.tech_factor_map",
            title="Cross-Section: Technical Factor Map (S&P cap)",
            summary="Not enough usable tickers for cross-sectional factor map.",
            warnings=[f"ok={ok} fails={fails} cap={cap}"]
        )

    df = pd.DataFrame(rows, columns=["Ticker","Sector","Momentum","LowVol","TrendQ","MeanRev","Liquidity"])

    # Cross-sectional robust z-scores
    for col in ["Momentum","LowVol","TrendQ","MeanRev","Liquidity"]:
        df[col+"_Z"] = _robust_z(df[col])

    df["CompositeZ"] = df[[c+"_Z" for c in ["Momentum","LowVol","TrendQ","MeanRev","Liquidity"]]].mean(axis=1)

    # Sector means (heatmap)
    g = df.groupby("Sector")[[c+"_Z" for c in ["Momentum","LowVol","TrendQ","MeanRev","Liquidity","CompositeZ".replace("Z","")]]]
    # (fix: CompositeZ not in above list)
    sector = df.groupby("Sector")[["Momentum_Z","LowVol_Z","TrendQ_Z","MeanRev_Z","Liquidity_Z","CompositeZ"]].mean().sort_values("CompositeZ", ascending=False)

    # heatmap
    mat = sector[["Momentum_Z","LowVol_Z","TrendQ_Z","MeanRev_Z","Liquidity_Z","CompositeZ"]].values
    cols = ["Mom","LowVol","TrendQ","MeanRev","Liq","Comp"]
    idx = sector.index.tolist()

    fig = plt.figure(figsize=(10,6.8))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(mat, aspect="auto", origin="upper")
    ax.set_title("Sector Factor Map (mean robust z-scores)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png_heat = fig_to_png_bytes(fig)

    # Top composite table
    top = df.sort_values("CompositeZ", ascending=False).head(20).copy()
    out = pd.DataFrame({
        "Ticker": top["Ticker"].values,
        "Sector": top["Sector"].values,
        "CompZ": top["CompositeZ"].map(lambda x: f"{x:+.2f}").values,
        "MomZ": top["Momentum_Z"].map(lambda x: f"{x:+.2f}").values,
        "LowVolZ": top["LowVol_Z"].map(lambda x: f"{x:+.2f}").values,
        "TrendQZ": top["TrendQ_Z"].map(lambda x: f"{x:+.2f}").values,
        "MeanRevZ": top["MeanRev_Z"].map(lambda x: f"{x:+.2f}").values,
        "LiqZ": top["Liquidity_Z"].map(lambda x: f"{x:+.2f}").values,
    })
    png_top = _table_png("Top 20 Composite (tech factor mix)", out)

    # Leader picks per factor (top 8)
    leaders = {}
    for col in ["Momentum_Z","LowVol_Z","TrendQ_Z","MeanRev_Z","Liquidity_Z"]:
        leaders[col] = df.sort_values(col, ascending=False).head(8)["Ticker"].tolist()
    lead_tab = pd.DataFrame({
        "Momentum": leaders["Momentum_Z"],
        "LowVol": leaders["LowVol_Z"],
        "TrendQ": leaders["TrendQ_Z"],
        "MeanRev": leaders["MeanRev_Z"],
        "Liquidity": leaders["Liquidity_Z"],
    })
    png_lead = _table_png("Factor Leaders (top 8 tickers by z-score)", lead_tab)

    metrics = {
        "UniverseOK": int(ok),
        "Fails": int(fails),
        "TopComposite": str(top["Ticker"].iloc[0]),
        "TopSector": str(sector.index[0]) if len(sector) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "Terminal-style cross-section using OHLCV only (robust): Momentum (12-1), LowVol (-vol60), TrendQ (R²), MeanRev (oversold), Liquidity (log $vol).",
        "Scores are cross-sectional robust z-scores (median/MAD), clipped to ±3.",
    ]

    return CardResult(
        key="xsec.tech_factor_map",
        title="Cross-Section: Technical Factor Map (S&P cap)",
        summary="Sector factor heatmap + top composite names + factor leaders.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="sector_factor_map.png", payload=png_heat),
            Artifact(kind="image/png", name="top_composite.png", payload=png_top),
            Artifact(kind="image/png", name="factor_leaders.png", payload=png_lead),
        ]
    )
