
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

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1600) -> pd.DataFrame:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    frames = []
    used = []
    for raw in tickers:
        t = str(raw).upper().strip().lstrip("$").replace(".", "-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames.append(df["Close"].astype(float).rename(t).iloc[-bars:])
        used.append(t)
    if len(frames) < 2:
        raise RuntimeError("Not enough series returned.")
    px = pd.concat(frames, axis=1).dropna(how="any")
    return px

# ---------------------------------------------------------
# 1) Commodities complex dashboard
# ---------------------------------------------------------
@register_card("macro.commodities_complex_dashboard", "Macro: Commodities Complex Dashboard", "macro", min_tier="black", cost=7, heavy=False, slots=("S03",))
def commodities_complex(ctx: CardContext) -> CardResult:
    # Broad: DBC (broad commodities), DBA (ag), USO (oil), GLD (gold), SLV (silver) + SPY for context
    tickers = ["SPY","DBC","DBA","USO","GLD","SLV"]
    px = _get_closes(ctx, tickers, bars=1800)

    needed = ["SPY","DBC","DBA","USO","GLD"]
    miss = [t for t in needed if t not in px.columns]
    if miss:
        return CardResult(
            key="macro.commodities_complex_dashboard",
            title="Macro: Commodities Complex Dashboard",
            summary="Missing required series for the commodities dashboard.",
            warnings=[f"missing: {', '.join(miss)}"]
        )

    norm = px / px.iloc[0]
    rets = px.pct_change().dropna()

    # Ratios (useful desk reads)
    gold_oil = px["GLD"] / (px["USO"] + 1e-12) if "USO" in px.columns else None
    broad_vs_spy = px["DBC"] / (px["SPY"] + 1e-12)

    # Rolling corr with SPY (60D)
    corr = {}
    for c in px.columns:
        if c == "SPY":
            continue
        corr[c] = rets["SPY"].rolling(60).corr(rets[c])
    corr_df = pd.DataFrame(corr)

    fig = plt.figure(figsize=(10,7.4))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    for c in norm.columns:
        ax1.plot(norm[c].values, label=c)
    ax1.set_title("Commodities complex (normalized)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    ax2.plot((broad_vs_spy / broad_vs_spy.iloc[0]).values, label="DBC/SPY (norm)")
    if gold_oil is not None:
        ax2.plot((gold_oil / gold_oil.iloc[0]).values, label="GLD/USO (norm)")
    ax2.axhline(1.0, linewidth=1)
    ax2.set_title("Key ratios (normalized)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    for c in corr_df.columns:
        ax3.plot(corr_df[c].values, label=f"Corr60 SPY~{c}")
    ax3.axhline(0.0, linewidth=1)
    ax3.set_title("Rolling correlation to SPY (60D)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", ncol=3, fontsize=7)

    png = fig_to_png_bytes(fig)

    # Metrics
    metrics = {}
    for t in ["DBC","DBA","USO","GLD","SLV"]:
        if t in rets.columns:
            metrics[f"{t} 20D%"] = round(float(px[t].pct_change(20).iloc[-1]*100), 2)
            metrics[f"Corr60 SPY~{t}"] = round(float(corr_df[t].dropna().iloc[-1]), 2) if t in corr_df.columns and corr_df[t].dropna().shape[0] else None
    metrics["DBC/SPY now"] = round(float(broad_vs_spy.iloc[-1]), 4)
    if gold_oil is not None:
        metrics["GLD/USO now"] = round(float(gold_oil.iloc[-1]), 4)
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "Desk-style view: broad commodities (DBC), agriculture (DBA), oil (USO), gold/silver.",
        "Ratios help frame inflation/real-asset tilt and commodity beta vs equities.",
    ]

    return CardResult(
        key="macro.commodities_complex_dashboard",
        title="Macro: Commodities Complex Dashboard",
        summary="Commodities complex + ratios + rolling equity correlation.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="commodities_complex.png", payload=png)],
    )

# ---------------------------------------------------------
# 2) FX risk dashboard
# ---------------------------------------------------------
@register_card("macro.fx_risk_dashboard", "Macro: FX Risk Dashboard (USD / EUR / JPY + EM)", "macro", min_tier="black", cost=7, heavy=False, slots=("S03",))
def fx_risk(ctx: CardContext) -> CardResult:
    # UUP (USD), FXE (EUR), FXY (JPY), EEM (EM eq proxy), SPY
    tickers = ["SPY","UUP","FXE","FXY","EEM"]
    px = _get_closes(ctx, tickers, bars=1800)
    needed = ["SPY","UUP","FXE","FXY"]
    miss = [t for t in needed if t not in px.columns]
    if miss:
        return CardResult(
            key="macro.fx_risk_dashboard",
            title="Macro: FX Risk Dashboard",
            summary="Missing required FX proxy ETFs for the dashboard.",
            warnings=[f"missing: {', '.join(miss)}"]
        )

    norm = px / px.iloc[0]
    rets = px.pct_change().dropna()

    # USD z-score regime (252)
    usd = px["UUP"]
    usd_z = _zscore(np.log(usd.replace(0, np.nan)).dropna(), 252).reindex(px.index)

    # Rolling corr vs SPY (60D)
    corr_uup = rets["SPY"].rolling(60).corr(rets["UUP"])
    corr_fxe = rets["SPY"].rolling(60).corr(rets["FXE"])
    corr_fxy = rets["SPY"].rolling(60).corr(rets["FXY"])

    fig = plt.figure(figsize=(10,7.2))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    for c in ["SPY","UUP","FXE","FXY"]:
        if c in norm.columns:
            ax1.plot(norm[c].values, label=c)
    if "EEM" in norm.columns:
        ax1.plot(norm["EEM"].values, label="EEM", linewidth=1.0)
    ax1.set_title("FX risk proxies (normalized)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    ax2.plot(usd_z.values, label="USD z-score (UUP, 252)")
    ax2.axhline(0.0, linewidth=1)
    ax2.axhline(1.5, linewidth=1)
    ax2.axhline(-1.5, linewidth=1)
    ax2.set_title("USD strength regime (z-score)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    ax3.plot(corr_uup.values, label="Corr60 SPY~UUP")
    ax3.plot(corr_fxe.values, label="Corr60 SPY~FXE")
    ax3.plot(corr_fxy.values, label="Corr60 SPY~FXY")
    ax3.axhline(0.0, linewidth=1)
    ax3.set_title("Rolling correlation to SPY (60D)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", ncol=3, fontsize=8)

    png = fig_to_png_bytes(fig)

    last_usd_z = float(usd_z.dropna().iloc[-1]) if usd_z.dropna().shape[0] else float("nan")
    if np.isfinite(last_usd_z) and last_usd_z >= 1.5:
        usd_reg = "USD very strong (often risk-off)"
    elif np.isfinite(last_usd_z) and last_usd_z <= -1.5:
        usd_reg = "USD weak (often risk-on)"
    else:
        usd_reg = "USD neutral"

    metrics = {
        "USD z252": round(last_usd_z, 2) if np.isfinite(last_usd_z) else None,
        "USD regime": usd_reg,
        "Corr60 SPY~UUP": round(float(corr_uup.dropna().iloc[-1]), 2) if corr_uup.dropna().shape[0] else None,
        "Corr60 SPY~FXE": round(float(corr_fxe.dropna().iloc[-1]), 2) if corr_fxe.dropna().shape[0] else None,
        "Corr60 SPY~FXY": round(float(corr_fxy.dropna().iloc[-1]), 2) if corr_fxy.dropna().shape[0] else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "UUP/FXE/FXY are liquid proxies (not spot FX).",
        "USD strength often coincides with tightening/risk-off; correlations help interpret equity sensitivity.",
    ]

    return CardResult(
        key="macro.fx_risk_dashboard",
        title="Macro: FX Risk Dashboard (USD / EUR / JPY + EM)",
        summary="FX proxy panel + USD regime + rolling equity correlations.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="fx_risk_dashboard.png", payload=png)],
    )

# ---------------------------------------------------------
# 3) Systemic risk composite (headline)
# ---------------------------------------------------------
@register_card("risk.systemic_risk_composite", "Risk: Systemic Risk Composite (headline)", "risk", min_tier="black", cost=9, heavy=False, slots=("S09","S03"))
def systemic_risk(ctx: CardContext) -> CardResult:
    # Components (desk-ish):
    #  - VIX term proxy (VIX/VXV) z (if available)
    #  - Credit proxy HYG/LQD z (stress = -z)
    #  - Stock/Bond corr (SPY~TLT) (stress increases with +corr)
    #  - Risk-on/off composite (stress = -risk_on_z)
    px = _get_closes(ctx, ["SPY","TLT","HYG","LQD","IWM","XLY","XLP","GLD","UUP"], bars=1800)
    need = ["SPY","TLT","HYG","LQD","IWM","XLY","XLP","GLD","UUP"]
    miss = [t for t in need if t not in px.columns]
    if miss:
        return CardResult(
            key="risk.systemic_risk_composite",
            title="Risk: Systemic Risk Composite (headline)",
            summary="Missing required series for systemic composite.",
            warnings=[f"missing: {', '.join(miss)}"]
        )

    rets = px.pct_change().dropna()
    # Credit (risk-off when HYG/LQD is low)
    credit = px["HYG"] / (px["LQD"] + 1e-12)
    credit_z = _zscore(np.log(credit.replace(0, np.nan)).dropna(), 252).reindex(px.index)

    # Stock/bond corr
    sb_corr = rets["SPY"].rolling(60).corr(rets["TLT"]).reindex(px.index)

    # Risk-on/off composite (small set)
    ratios = {
        "IWM/SPY": px["IWM"] / (px["SPY"] + 1e-12),
        "XLY/XLP": px["XLY"] / (px["XLP"] + 1e-12),
        "HYG/LQD": credit,
        "TLT/SPY": px["TLT"] / (px["SPY"] + 1e-12),
        "GLD/SPY": px["GLD"] / (px["SPY"] + 1e-12),
        "UUP": px["UUP"],
    }
    Z = {}
    for k, s in ratios.items():
        Z[k] = _zscore(np.log(s.replace(0, np.nan)).dropna(), 252)
    Z = pd.concat(Z, axis=1).dropna(how="any")

    sign = pd.Series(1.0, index=Z.columns)
    for k in Z.columns:
        if k.startswith("TLT/") or k.startswith("GLD/") or k == "UUP":
            sign[k] = -1.0
    riskon = (Z * sign).mean(axis=1)
    riskon = riskon.reindex(px.index)

    # VIX term (optional)
    vix_z = None
    vix_ratio = None
    vix_note = None
    try:
        import yfinance as yf
        def fetch(sym: str) -> pd.Series | None:
            h = yf.download(sym, start=str(ctx.start), progress=False, auto_adjust=False)
            if h is None or h.empty:
                return None
            c = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
            return c.astype(float).dropna().iloc[-1800:]
        vix = fetch("^VIX")
        vxv = fetch("^VXV")
        if vix is not None and vxv is not None and len(vix) > 200 and len(vxv) > 200:
            vv = pd.concat([vix.rename("VIX"), vxv.rename("VXV")], axis=1).dropna(how="any")
            vix_ratio = (vv["VIX"] / (vv["VXV"] + 1e-12))
            vix_z = _zscore(vix_ratio, 252).reindex(px.index)
        else:
            vix_note = "VIX indices unavailable (skipped)."
    except Exception:
        vix_note = "yfinance unavailable (skipped VIX term)."

    # Stress components:
    #  credit_stress = -credit_z
    #  sb_stress = sb_corr (positive corr is worse diversification)
    #  riskon_stress = -riskon
    credit_stress = (-credit_z).clip(-3, 3)
    sb_stress = sb_corr.clip(-1, 1)
    riskon_stress = (-riskon).clip(-3, 3)

    # vix term stress if available
    if vix_z is not None:
        vix_stress = vix_z.clip(-3, 3)
        stress = 0.35*vix_stress + 0.25*credit_stress + 0.20*sb_stress + 0.20*riskon_stress
    else:
        stress = 0.35*credit_stress + 0.30*sb_stress + 0.35*riskon_stress

    stress = stress.dropna()
    if stress.empty:
        return CardResult(
            key="risk.systemic_risk_composite",
            title="Risk: Systemic Risk Composite (headline)",
            summary="Stress composite had insufficient overlap to compute."
        )

    last = float(stress.iloc[-1])
    if last >= 0.75:
        label = "HIGH STRESS"
    elif last <= -0.75:
        label = "LOW STRESS"
    else:
        label = "MODERATE"

    fig = plt.figure(figsize=(10,7.6))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    ax4 = fig.add_subplot(4,1,4, sharex=ax1)

    ax1.plot(stress.values, label="SystemicStress")
    ax1.axhline(0, linewidth=1)
    ax1.axhline(0.75, linewidth=1)
    ax1.axhline(-0.75, linewidth=1)
    ax1.set_title(f"Systemic Risk Composite — {label} (now {last:+.2f})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(credit_stress.reindex(stress.index).values, label="Credit stress (-z HYG/LQD)")
    ax2.axhline(0, linewidth=1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    ax3.plot(sb_stress.reindex(stress.index).values, label="Stock/Bond corr (SPY~TLT)")
    ax3.axhline(0, linewidth=1)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)

    ax4.plot(riskon_stress.reindex(stress.index).values, label="Risk-on stress (-riskon composite)")
    ax4.axhline(0, linewidth=1)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper left", fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {
        "SystemicStress": round(last, 3),
        "Label": label,
        "CreditStress": round(float(credit_stress.dropna().iloc[-1]), 2) if credit_stress.dropna().shape[0] else None,
        "StockBondCorr": round(float(sb_corr.dropna().iloc[-1]), 2) if sb_corr.dropna().shape[0] else None,
        "RiskOnStress": round(float(riskon_stress.dropna().iloc[-1]), 2) if riskon_stress.dropna().shape[0] else None,
    }
    if vix_z is not None and vix_z.dropna().shape[0]:
        metrics["VIXtermZ"] = round(float(vix_z.dropna().iloc[-1]), 2)
        metrics["VIX/VXV"] = round(float(vix_ratio.dropna().iloc[-1]), 3) if vix_ratio is not None and vix_ratio.dropna().shape[0] else None
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "Composite blends: VIX term (if available), credit risk (HYG/LQD), stock/bond corr, and risk-on/off ratios.",
        "Positive = more stress/risk-off; negative = calmer/risk-on.",
    ]
    if vix_note:
        bullets.append(f"Note: {vix_note}")

    return CardResult(
        key="risk.systemic_risk_composite",
        title="Risk: Systemic Risk Composite (headline)",
        summary="Single headline risk dial built from classic desk proxies.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="systemic_risk_composite.png", payload=png)]
    )
