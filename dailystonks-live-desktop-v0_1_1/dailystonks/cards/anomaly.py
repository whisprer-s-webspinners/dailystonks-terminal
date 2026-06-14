from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("anomaly.sigma_intraday_alerts", "Sigma Intraday Alerts", "anomaly", min_tier="free", cost=3, heavy=False, slots=("S09",))
def sigma_intraday(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    # use hourly by default for speed if user is on 1d
    interval = ctx.interval if ctx.interval != "1d" else "1h"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval=interval).iloc[-400:]
    ret = np.log(df["Close"]).diff().dropna()
    z = (ret - ret.mean()) / (ret.std(ddof=1) + 1e-12)
    thr = 3.0
    hits = z[np.abs(z) >= thr]
    bullets = [f"Threshold: |z| >= {thr:.1f} on log-returns ({interval})."]
    if len(hits):
        bullets.append(f"Hits: {len(hits)} (latest z={float(hits.iloc[-1]):.2f})")
    else:
        bullets.append("Hits: 0")

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(z.values, label="z(log-return)")
    ax.axhline(thr, linewidth=1); ax.axhline(-thr, linewidth=1)
    ax.set_title(f"{t} Intraday Return z-score ({interval})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="anomaly.sigma_intraday_alerts",
        title=f"{t}: Sigma Alerts ({interval})",
        summary="Quick outlier scan for unusually large moves.",
        metrics={"hits": int(len(hits))},
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name=f"{t}_sigma.png", payload=png)]
    )

@register_card("anomaly.isolationforest_overlay", "IsolationForest Anomaly Overlay", "anomaly", min_tier="pro", cost=8, heavy=False, slots=("S09",))
def iso_overlay(ctx: CardContext) -> CardResult:
    from sklearn.ensemble import IsolationForest
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    X = np.column_stack([ret.values, vol.values])
    clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=0)
    pred = clf.fit_predict(X)  # -1 anomalies
    anom_idx = np.where(pred == -1)[0]

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(close.values, label="Close")
    if len(anom_idx):
        ax.scatter(anom_idx, close.values[anom_idx], s=25, label="Anomaly")
    ax.set_title(f"{t} IsolationForest Anomaly Overlay")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="anomaly.isolationforest_overlay",
        title=f"{t}: Anomaly Overlay",
        summary="IsolationForest on (return, rolling vol) features.",
        metrics={"anomaly_points": int(len(anom_idx))},
        artifacts=[Artifact(kind="image/png", name=f"{t}_iso.png", payload=png)]
    )

@register_card("anomaly.cleaning_report_before_after", "Cleaning Report (Before/After)", "anomaly", min_tier="black", cost=10, heavy=True, slots=("S11",))
def cleaning_report(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    ret = df["Close"].pct_change().dropna()
    z = (ret - ret.mean()) / (ret.std(ddof=1) + 1e-12)
    cleaned = ret.copy()
    cleaned[np.abs(z) > 4.0] = np.sign(cleaned[np.abs(z) > 4.0]) * 4.0 * ret.std(ddof=1)

    fig = plt.figure(figsize=(10,4.8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(ret.values, label="raw returns"); ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.25)
    ax2.plot(cleaned.values, label="cleaned returns"); ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.25)
    ax2.set_title("Clipped |z|>4 events")
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="anomaly.cleaning_report_before_after",
        title=f"{t}: Outlier Cleaning (Before/After)",
        summary="Simple z-score clipping shows impact on returns series.",
        metrics={"clipped_points": int((np.abs(z) > 4.0).sum())},
        artifacts=[Artifact(kind="image/png", name=f"{t}_clean.png", payload=png)]
    )

@register_card("anomaly.autoencoder_flags", "Autoencoder Flags (PCA Reconstruction)", "anomaly", min_tier="black", cost=12, heavy=True, slots=("S11",))
def ae_flags(ctx: CardContext) -> CardResult:
    from sklearn.decomposition import PCA
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    X = np.column_stack([ret.values, vol.values])
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=1, random_state=0)
    Z = pca.fit_transform(Xs)
    Xr = pca.inverse_transform(Z)
    err = np.mean((Xs - Xr)**2, axis=1)
    thr = float(np.quantile(err, 0.98))
    idx = np.where(err >= thr)[0]

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(err, label="recon error")
    ax.axhline(thr, linewidth=1, label="98% threshold")
    ax.scatter(idx, err[idx], s=25, label="flag")
    ax.set_title(f"{t} PCA-Reconstruction Flags (AE proxy)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="anomaly.autoencoder_flags",
        title=f"{t}: Reconstruction-Error Flags",
        summary="PCA reconstruction error used as lightweight autoencoder proxy.",
        metrics={"flags": int(len(idx))},
        artifacts=[Artifact(kind="image/png", name=f"{t}_ae.png", payload=png)]
    )
