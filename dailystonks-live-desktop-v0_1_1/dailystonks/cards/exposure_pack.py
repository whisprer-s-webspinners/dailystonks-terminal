from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _basket(ctx: CardContext, max_n: int = 10) -> list[str]:
    if ctx.tickers and len(ctx.tickers) >= 3:
        base = ctx.tickers[:]
    else:
        base = ["SPY","QQQ","IWM","TLT","GLD","BTC-USD","UUP","AAPL","MSFT","NVDA"]
    # normalize
    out=[]
    for t in base:
        t = str(t).upper().strip().lstrip("$").replace(".","-")
        if t and t not in out:
            out.append(t)
    return out[:max_n]

def _close_matrix(ctx: CardContext, tickers: list[str], bars: int = 520) -> pd.DataFrame:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    frames=[]
    for raw in tickers:
        t = raw.replace(".","-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames.append(df["Close"].astype(float).rename(t).iloc[-bars:])
    if len(frames) < 2:
        raise RuntimeError("Not enough series.")
    px = pd.concat(frames, axis=1).dropna(how="any")
    return px

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

@register_card("risk.corr_matrix_heatmap", "Correlation Matrix Heatmap (60D)", "risk", min_tier="pro", cost=9, heavy=False, slots=("S11",))
def corr_matrix(ctx: CardContext) -> CardResult:
    tickers = _basket(ctx, max_n=10)
    px = _close_matrix(ctx, tickers, bars=520)
    rets = px.pct_change().dropna()
    if len(rets) < 80:
        return CardResult(
            key="risk.corr_matrix_heatmap",
            title="Correlation Matrix Heatmap (60D)",
            summary="Not enough overlapping returns for correlation (need ~80+ rows).",
            warnings=[f"rows={len(rets)}"]
        )

    corr = rets.iloc[-60:].corr()

    fig = plt.figure(figsize=(10,7.0))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(corr.values, aspect="auto", origin="upper")
    ax.set_title("Correlation Matrix (last 60 trading days)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index.tolist())

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = fig_to_png_bytes(fig)

    # simple “top corr pairs” metric
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    stacked = c.stack().sort_values(ascending=False)
    top_pair = stacked.index[0] if len(stacked) else ("","")
    top_val = float(stacked.iloc[0]) if len(stacked) else float("nan")

    metrics = {"Assets": len(corr.columns), "TopCorrPair": f"{top_pair[0]}~{top_pair[1]}", "TopCorr": round(top_val, 3) if np.isfinite(top_val) else None}
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="risk.corr_matrix_heatmap",
        title="Correlation Matrix Heatmap (60D)",
        summary="Basket correlations using last 60 trading days of returns.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="corr_matrix.png", payload=png)]
    )

@register_card("risk.rolling_beta_panel", "Rolling Beta vs SPY (60D)", "risk", min_tier="pro", cost=9, heavy=False, slots=("S08",))
def rolling_beta(ctx: CardContext) -> CardResult:
    tickers = _basket(ctx, max_n=7)
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers
    tickers = list(dict.fromkeys(tickers))[:7]

    px = _close_matrix(ctx, tickers, bars=700)
    rets = px.pct_change().dropna()
    if "SPY" not in rets.columns or len(rets) < 120:
        return CardResult(
            key="risk.rolling_beta_panel",
            title="Rolling Beta vs SPY (60D)",
            summary="Need SPY and enough history for rolling beta.",
            warnings=[f"cols={rets.columns.tolist()} rows={len(rets)}"]
        )

    rspy = rets["SPY"]
    var = rspy.rolling(60).var(ddof=1)

    fig = plt.figure(figsize=(10,5.6))
    ax = fig.add_subplot(1,1,1)

    metrics={}
    for col in rets.columns:
        if col == "SPY":
            continue
        cov = rets[col].rolling(60).cov(rspy)
        beta = cov / (var + 1e-12)
        ax.plot(beta.values, label=f"{col}")
        if np.isfinite(beta.iloc[-1]):
            metrics[f"β60 {col}"] = round(float(beta.iloc[-1]), 2)

    ax.axhline(1.0, linewidth=1)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Rolling Beta vs SPY (60D)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="risk.rolling_beta_panel",
        title="Rolling Beta vs SPY (60D)",
        summary="Rolling beta estimates for basket constituents vs SPY.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="rolling_beta.png", payload=png)]
    )

@register_card("risk.factor_proxy_exposure", "Factor Proxy Exposures (OLS)", "risk", min_tier="black", cost=10, heavy=True, slots=("S03","S11"))
def factor_proxy(ctx: CardContext) -> CardResult:
    # Factor set: SPY (market), TLT (rates), UUP (USD), GLD (gold)
    factors = ["SPY","TLT","UUP","GLD"]
    assets = _basket(ctx, max_n=6)
    # ensure factors included for data fetch
    fetch = list(dict.fromkeys(factors + assets))

    px = _close_matrix(ctx, fetch, bars=900)
    rets = px.pct_change().dropna()
    if any(f not in rets.columns for f in factors):
        return CardResult(
            key="risk.factor_proxy_exposure",
            title="Factor Proxy Exposures (OLS)",
            summary="Missing one or more factor series (SPY/TLT/UUP/GLD).",
            warnings=[f"have={rets.columns.tolist()}"]
        )

    X = rets[factors].iloc[-252:].copy()
    X = X.dropna()
    if len(X) < 120:
        return CardResult(
            key="risk.factor_proxy_exposure",
            title="Factor Proxy Exposures (OLS)",
            summary="Not enough overlapping factor returns for regression.",
            warnings=[f"rows={len(X)}"]
        )

    # Add intercept
    Xmat = np.column_stack([np.ones(len(X)), X.values])
    cols = ["alpha"] + factors

    rows=[]
    for a in assets:
        if a not in rets.columns or a in factors:
            continue
        y = rets[a].reindex(X.index).dropna()
        common = X.index.intersection(y.index)
        if len(common) < 120:
            continue
        Xc = Xmat[[X.index.get_loc(i) for i in common], :]
        yc = y.loc[common].values

        # OLS
        b, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        yhat = Xc @ b
        resid = yc - yhat
        ssr = float(np.sum(resid**2))
        sst = float(np.sum((yc - yc.mean())**2)) + 1e-12
        r2 = 1.0 - ssr/sst
        resid_vol = float(np.std(resid, ddof=1) * np.sqrt(252))

        row = [a] + [float(v) for v in b] + [r2, resid_vol]
        rows.append(row)

    if not rows:
        return CardResult(
            key="risk.factor_proxy_exposure",
            title="Factor Proxy Exposures (OLS)",
            summary="No assets had enough overlap for regression."
        )

    df = pd.DataFrame(rows, columns=["Asset"] + cols + ["R2","ResidVol%"])
    # format
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda x: f"{x:+.3f}")
    out["R2"] = out["R2"].map(lambda x: f"{x:.2f}")
    out["ResidVol%"] = out["ResidVol%"].map(lambda x: f"{x*100:.1f}%")

    png = _table_png("Factor Proxy Exposures (OLS on SPY/TLT/UUP/GLD, last ~252D)", out)

    bullets = [
        "OLS is a simple proxy: exposures are not stable and can shift by regime.",
        "ResidVol is the annualized volatility not explained by the factors (higher = more idiosyncratic)."
    ]

    return CardResult(
        key="risk.factor_proxy_exposure",
        title="Factor Proxy Exposures (OLS)",
        summary="Simple factor exposure table using SPY/TLT/UUP/GLD (best-effort).",
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="factor_proxy.png", payload=png)]
    )