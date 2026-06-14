
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

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1800) -> pd.DataFrame:
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

def _beta_corr(rets: pd.DataFrame, y: str, x: str, win: int = 252):
    m = rets[[y, x]].dropna().iloc[-win:]
    if len(m) < max(80, win//3):
        return float("nan"), float("nan")
    vx = float(m[x].var(ddof=1))
    if not np.isfinite(vx) or vx <= 0:
        return float("nan"), float("nan")
    beta = float(m[y].cov(m[x]) / (vx + 1e-12))
    corr = float(m[y].corr(m[x]))
    return beta, corr

def _fetch_vix_term(ctx: CardContext) -> tuple[pd.Series | None, pd.Series | None, str | None]:
    # Optional: uses Yahoo indices (^VIX, ^VXV). If it fails, dashboard still works.
    note = None
    try:
        import yfinance as yf
    except Exception:
        return None, None, "yfinance unavailable (skipped VIX term)."

    def fetch(sym: str) -> pd.Series | None:
        try:
            h = yf.download(sym, start=str(ctx.start), progress=False, auto_adjust=False)
            if h is None or h.empty:
                return None
            c = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
            return c.astype(float).dropna().iloc[-1800:]
        except Exception:
            return None

    vix = fetch("^VIX")
    vxv = fetch("^VXV")
    if vix is None or vxv is None or len(vix) < 200 or len(vxv) < 200:
        return None, None, "Could not fetch ^VIX/^VXV reliably (skipped)."

    vv = pd.concat([vix.rename("VIX"), vxv.rename("VXV")], axis=1).dropna(how="any")
    ratio = vv["VIX"] / (vv["VXV"] + 1e-12)
    z = _zscore(ratio, 252)
    return ratio, z, note

@register_card("risk.headline_risk_dashboard", "Risk: Headline Risk Dashboard (one page)", "risk",
               min_tier="black", cost=10, heavy=False, slots=("S09","S03"))
def headline_risk(ctx: CardContext) -> CardResult:
    # Core desk proxies (ETF-based, robust)
    tickers = ["SPY","TLT","HYG","LQD","IWM","XLY","XLP","GLD","UUP","VIXY"]
    px = _get_closes(ctx, tickers, bars=1800)
    rets = px.pct_change().dropna()
    if "SPY" not in rets.columns or len(rets) < 260:
        return CardResult(
            key="risk.headline_risk_dashboard",
            title="Risk: Headline Risk Dashboard (one page)",
            summary="Need SPY + enough overlapping returns (~260+ rows).",
            warnings=[f"cols={px.columns.tolist()} rows={len(rets)}"]
        )

    # --- Credit stress (risk-off when HYG/LQD low) ---
    if ("HYG" in px.columns) and ("LQD" in px.columns):
        credit = px["HYG"] / (px["LQD"] + 1e-12)
        credit_z = _zscore(np.log(credit.replace(0, np.nan)).dropna(), 252).reindex(px.index)
        credit_stress = (-credit_z).clip(-3, 3)
    else:
        credit_z = None
        credit_stress = pd.Series(index=px.index, dtype=float)

    # --- Stock/Bond corr regime (SPY~TLT) ---
    if "TLT" in rets.columns:
        sb_corr = rets["SPY"].rolling(60).corr(rets["TLT"]).clip(-1, 1).reindex(px.index)
    else:
        sb_corr = pd.Series(index=px.index, dtype=float)

    # --- Risk-on/off composite (small set, signed) ---
    ratios = {}
    if "IWM" in px.columns: ratios["IWM/SPY"] = px["IWM"] / (px["SPY"] + 1e-12)
    if "XLY" in px.columns and "XLP" in px.columns: ratios["XLY/XLP"] = px["XLY"] / (px["XLP"] + 1e-12)
    if "HYG" in px.columns and "LQD" in px.columns: ratios["HYG/LQD"] = px["HYG"] / (px["LQD"] + 1e-12)
    if "TLT" in px.columns: ratios["TLT/SPY"] = px["TLT"] / (px["SPY"] + 1e-12)
    if "GLD" in px.columns: ratios["GLD/SPY"] = px["GLD"] / (px["SPY"] + 1e-12)
    if "UUP" in px.columns: ratios["UUP"] = px["UUP"]

    Z = {}
    for k, s in ratios.items():
        Z[k] = _zscore(np.log(s.replace(0, np.nan)).dropna(), 252)
    Z = pd.concat(Z, axis=1).dropna(how="any") if Z else pd.DataFrame(index=px.index)

    sign = pd.Series(1.0, index=Z.columns)
    for k in Z.columns:
        if k.startswith("TLT/") or k.startswith("GLD/") or k == "UUP":
            sign[k] = -1.0
    riskon = (Z * sign).mean(axis=1).reindex(px.index) if not Z.empty else pd.Series(index=px.index, dtype=float)
    riskon_stress = (-riskon).clip(-3, 3)

    # --- SPY vol regime ---
    vol20 = rets["SPY"].rolling(20).std(ddof=1) * math.sqrt(TRADING_DAYS)
    vol_med = vol20.rolling(252, min_periods=120).median()
    vol_z = _zscore(vol20, 252)
    vol_stress = vol_z.clip(-3, 3)

    # --- Optional VIX term ---
    vix_ratio, vix_z, vix_note = _fetch_vix_term(ctx)
    if vix_z is not None:
        vix_stress = vix_z.clip(-3, 3).reindex(px.index)
        stress = (0.30*vix_stress + 0.25*credit_stress + 0.20*sb_corr + 0.25*riskon_stress)
    else:
        stress = (0.35*credit_stress + 0.25*sb_corr + 0.40*riskon_stress + 0.20*vol_stress)

    stress = stress.dropna()
    if stress.empty:
        return CardResult(
            key="risk.headline_risk_dashboard",
            title="Risk: Headline Risk Dashboard (one page)",
            summary="Insufficient overlap to compute stress composite."
        )

    last = float(stress.iloc[-1])
    if last >= 0.9:
        label = "HIGH STRESS"
    elif last <= -0.9:
        label = "LOW STRESS"
    else:
        label = "MODERATE"

    # --- Build one screenshot-ready figure (4 panels) ---
    fig = plt.figure(figsize=(10,8.2))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    ax4 = fig.add_subplot(4,1,4, sharex=ax1)

    ax1.plot(stress.values, label="SystemicStress")
    ax1.axhline(0, linewidth=1)
    ax1.axhline(0.9, linewidth=1)
    ax1.axhline(-0.9, linewidth=1)
    ax1.set_title(f"Headline Risk Dashboard — {label} (now {last:+.2f})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    if vix_z is not None and vix_z.dropna().shape[0]:
        vz = vix_z.reindex(stress.index).dropna()
        if len(vz):
            ax1.plot(vz.values, label="VIX term z", linewidth=1)
            ax1.legend(loc="upper left", ncol=2, fontsize=8)

    ax2.plot(credit_stress.reindex(stress.index).values, label="CreditStress (-z HYG/LQD)")
    ax2.axhline(0, linewidth=1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    ax3.plot(sb_corr.reindex(stress.index).values, label="Stock/Bond Corr60 (SPY~TLT)")
    ax3.axhline(0, linewidth=1)
    ax3.axhline(0.25, linewidth=1)
    ax3.axhline(-0.25, linewidth=1)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)

    ax4.plot((vol20.reindex(stress.index)*100).values, label="SPY Vol20% (ann.)")
    ax4.plot((vol_med.reindex(stress.index)*100).values, label="Vol median% (rolling)")
    ax4.axhline(0, linewidth=1)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper left", ncol=2, fontsize=8)

    png = fig_to_png_bytes(fig)

    # headline metrics
    last_credit = float(credit_stress.dropna().iloc[-1]) if credit_stress.dropna().shape[0] else float("nan")
    last_corr = float(sb_corr.dropna().iloc[-1]) if sb_corr.dropna().shape[0] else float("nan")
    last_vol = float(vol20.dropna().iloc[-1]) if vol20.dropna().shape[0] else float("nan")
    last_riskon = float(riskon_stress.dropna().iloc[-1]) if riskon_stress.dropna().shape[0] else float("nan")

    metrics = {
        "SystemicStress": round(last, 3),
        "Label": label,
        "CreditStress": round(last_credit, 2) if np.isfinite(last_credit) else None,
        "StockBondCorr60": round(last_corr, 2) if np.isfinite(last_corr) else None,
        "SPY Vol20%": round(last_vol*100, 1) if np.isfinite(last_vol) else None,
        "RiskOnStress": round(last_riskon, 2) if np.isfinite(last_riskon) else None,
    }
    if vix_ratio is not None and vix_ratio.dropna().shape[0]:
        metrics["VIX/VXV"] = round(float(vix_ratio.dropna().iloc[-1]), 3)
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "One-page screenshot: stress composite + key drivers (credit, stock/bond corr, SPY vol regime).",
        "Composite blends classic desk proxies; positive = more stress/risk-off.",
    ]
    if vix_note:
        bullets.append(f"Note: {vix_note}")

    return CardResult(
        key="risk.headline_risk_dashboard",
        title="Risk: Headline Risk Dashboard (one page)",
        summary="Screenshot-ready risk page (systemic stress + key drivers).",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="headline_risk_dashboard.png", payload=png)],
    )
