
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=max(30, win//3)).mean()
    sd = s.rolling(win, min_periods=max(30, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1200) -> pd.DataFrame:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    frames = []
    for raw in tickers:
        t = raw.replace(".", "-").upper().lstrip("$")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames.append(df["Close"].astype(float).rename(t).iloc[-bars:])
    if len(frames) < 2:
        raise RuntimeError("Not enough series returned for rates pack.")
    px = pd.concat(frames, axis=1).dropna(how="any")
    return px

@register_card("rates.term_proxy_dashboard", "Rates: Term / Curve Proxy Dashboard", "rates", min_tier="pro", cost=6, heavy=False, slots=("S03","S11"))
def term_proxy(ctx: CardContext) -> CardResult:
    # Treasury ETFs: SHY (1-3y), IEF (7-10y), TLT (20y+)
    px = _get_closes(ctx, ["SHY","IEF","TLT"], bars=1600)
    rets = px.pct_change().dropna()

    # Proxies:
    # 1) Long/short ratio (price proxy): TLT / SHY (up => long yields down relative to short)
    ratio_ls = (px["TLT"] / (px["SHY"] + 1e-12))
    # 2) Term spread proxy (returns): TLT - SHY daily return (risk-on duration vs cash-like)
    spread_ret = rets["TLT"] - rets["SHY"]
    # 3) Intermediate/short ratio: IEF / SHY
    ratio_is = (px["IEF"] / (px["SHY"] + 1e-12))

    fig = plt.figure(figsize=(10,6.8))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    norm = px / px.iloc[0]
    for c in norm.columns:
        ax1.plot(norm[c].values, label=c)
    ax1.set_title("Treasury ETFs (normalized): SHY / IEF / TLT")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    ax2.plot((ratio_ls / ratio_ls.iloc[0]).values, label="TLT/SHY (norm)")
    ax2.plot((ratio_is / ratio_is.iloc[0]).values, label="IEF/SHY (norm)")
    ax2.set_title("Curve proxies (normalized ratios)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    ax3.plot((spread_ret.rolling(20).mean()*100).values, label="(TLT-SHY) 20D mean %")
    ax3.axhline(0, linewidth=1)
    ax3.set_title("Term spread proxy (duration vs short)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "TLT/SHY now": round(float(ratio_ls.iloc[-1]), 4),
        "IEF/SHY now": round(float(ratio_is.iloc[-1]), 4),
        "20D mean(TLT-SHY)%": round(float(spread_ret.rolling(20).mean().iloc[-1]*100), 3),
    }

    bullets = [
        "Price ratios are proxies (not yields).",
        "Rising TLT/SHY often aligns with easing long-end conditions; falling suggests tightening long-end relative to short.",
    ]

    return CardResult(
        key="rates.term_proxy_dashboard",
        title="Rates: Term / Curve Proxy Dashboard",
        summary="SHY/IEF/TLT view + curve proxies via ratios and spread returns.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="rates_term_proxy.png", payload=png)]
    )

@register_card("rates.stock_bond_corr_regime", "Rates: Stock/Bond Correlation Regime (SPY↔TLT)", "rates", min_tier="black", cost=6, heavy=False, slots=("S03","S11"))
def stock_bond_corr(ctx: CardContext) -> CardResult:
    px = _get_closes(ctx, ["SPY","TLT"], bars=1600)
    rets = px.pct_change().dropna()
    if len(rets) < 120:
        return CardResult(
            key="rates.stock_bond_corr_regime",
            title="Rates: Stock/Bond Correlation Regime (SPY↔TLT)",
            summary="Not enough overlap (need ~120+ return rows)."
        )

    corr60 = rets["SPY"].rolling(60).corr(rets["TLT"])
    last = float(corr60.dropna().iloc[-1])

    if last <= -0.25:
        reg = "Diversifying (neg corr)"
    elif last >= 0.25:
        reg = "Same-direction (pos corr)"
    else:
        reg = "Mixed/neutral"

    fig = plt.figure(figsize=(10,5.4))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot((px["SPY"]/px["SPY"].iloc[0]).values, label="SPY (norm)")
    ax1.plot((px["TLT"]/px["TLT"].iloc[0]).values, label="TLT (norm)")
    ax1.set_title("SPY vs TLT (normalized)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    ax2.plot(corr60.values, label="Corr60(SPY,TLT)")
    ax2.axhline(0, linewidth=1)
    ax2.axhline(0.25, linewidth=1)
    ax2.axhline(-0.25, linewidth=1)
    ax2.set_title(f"Rolling correlation regime: {reg} (now {last:+.2f})")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {"Corr60 now": round(last, 3), "Regime": reg}

    bullets = [
        "Negative corr helps diversification; positive corr often appears in inflation/tightening stress regimes.",
    ]

    return CardResult(
        key="rates.stock_bond_corr_regime",
        title="Rates: Stock/Bond Correlation Regime (SPY↔TLT)",
        summary="Rolling 60D correlation + regime label.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="stock_bond_corr.png", payload=png)]
    )

@register_card("rates.credit_spread_proxy", "Rates: Credit Stress Proxy (HYG/LQD)", "rates", min_tier="black", cost=6, heavy=False, slots=("S03","S11"))
def credit_proxy(ctx: CardContext) -> CardResult:
    # HYG (HY credit) vs LQD (IG credit). Ratio down => credit stress.
    px = _get_closes(ctx, ["HYG","LQD"], bars=1600)
    ratio = px["HYG"] / (px["LQD"] + 1e-12)
    rz = _zscore(ratio, 120)

    last = float(ratio.iloc[-1])
    lastz = float(rz.dropna().iloc[-1]) if rz.dropna().shape[0] else float("nan")

    if np.isfinite(lastz) and lastz <= -1.5:
        reg = "Credit stress (risk-off)"
    elif np.isfinite(lastz) and lastz >= 1.5:
        reg = "Credit easy (risk-on)"
    else:
        reg = "Neutral"

    fig = plt.figure(figsize=(10,5.6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot((ratio/ratio.iloc[0]).values, label="HYG/LQD (norm)")
    ax1.set_title("Credit proxy ratio (HYG/LQD)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(rz.values, label="Zscore120(HYG/LQD)")
    ax2.axhline(0, linewidth=1)
    ax2.axhline(1.5, linewidth=1)
    ax2.axhline(-1.5, linewidth=1)
    ax2.set_title(f"Credit proxy z-score: {reg} (now {lastz:+.2f})")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "HYG/LQD now": round(last, 5),
        "Z120 now": round(lastz, 2) if np.isfinite(lastz) else None,
        "Regime": reg
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "HYG/LQD is a simple proxy (not a true spread).",
        "Falling ratio often aligns with widening credit spreads / risk-off conditions.",
    ]

    return CardResult(
        key="rates.credit_spread_proxy",
        title="Rates: Credit Stress Proxy (HYG/LQD)",
        summary="Credit risk proxy using HYG relative to LQD + z-score regime.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="credit_proxy.png", payload=png)]
    )
