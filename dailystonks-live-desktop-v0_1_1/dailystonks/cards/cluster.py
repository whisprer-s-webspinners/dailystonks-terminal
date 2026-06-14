from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("cluster.return_vol_kmeans_map", "Return/Vol Cluster Map", "cluster", min_tier="pro", cost=10, heavy=True, slots=("S04",))
def return_vol_map(ctx: CardContext) -> CardResult:
    from sklearn.cluster import KMeans
    tickers = ctx.sp500.tickers(max_n=ctx.max_universe)
    rows=[]
    for tk in tickers:
        try:
            df = ctx.market.get_ohlcv(tk, start=ctx.start, end=ctx.end, interval="1d").iloc[-260:]
            rets = df["Close"].pct_change().dropna()
            ann_ret = float((1+rets.mean())**252 - 1)
            ann_vol = float(rets.std(ddof=1) * np.sqrt(252))
            rows.append((tk, ann_ret, ann_vol))
        except Exception:
            continue
    if len(rows) < 20:
        raise RuntimeError("Not enough points for cluster map.")
    dfv = pd.DataFrame(rows, columns=["Symbol","AnnRet","AnnVol"])
    X = dfv[["AnnRet","AnnVol"]].values
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    lab = km.labels_

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(dfv["AnnVol"].values, dfv["AnnRet"].values, s=25)
    ax.set_xlabel("Annualized Vol")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Universe Return/Vol Map (KMeans ready)")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    # quick top by return/vol ratio
    dfv["SharpeLike"] = dfv["AnnRet"] / (dfv["AnnVol"] + 1e-12)
    top = dfv.sort_values("SharpeLike", ascending=False).head(8)
    metrics = {f"Top{i+1}": f"{r.Symbol}({r.SharpeLike:.2f})" for i,r in enumerate(top.itertuples())}

    return CardResult(
        key="cluster.return_vol_kmeans_map",
        title="Return vs Vol Map",
        summary="Scatter of annualized return vs vol for capped universe.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="return_vol_map.png", payload=png)]
    )

def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n+1, m+1), np.inf)
    dp[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[n,m])

@register_card("cluster.dtw_dendrogram", "DTW Dendrogram (Similarity)", "cluster", min_tier="black", cost=14, heavy=True, slots=("S11",))
def dtw_dendro(ctx: CardContext) -> CardResult:
    # Use a small subset for runtime.
    tickers = ctx.sp500.tickers(max_n=min(ctx.max_universe, 25))
    series=[]
    used=[]
    for tk in tickers:
        try:
            df = ctx.market.get_ohlcv(tk, start=ctx.start, end=ctx.end, interval="1d").iloc[-120:]
            rets = df["Close"].pct_change().fillna(0).values
            series.append(rets)
            used.append(tk)
        except Exception:
            continue
    if len(series) < 8:
        raise RuntimeError("Not enough series for DTW dendrogram.")
    # compute condensed distance matrix
    dists=[]
    for i in range(len(series)):
        for j in range(i+1, len(series)):
            dists.append(_dtw(series[i], series[j]))
    Z = linkage(dists, method="average")

    fig = plt.figure(figsize=(10,5.6))
    ax = fig.add_subplot(1,1,1)
    dendrogram(Z, labels=used, leaf_rotation=90, ax=ax)
    ax.set_title("DTW Similarity Dendrogram (returns)")
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="cluster.dtw_dendrogram",
        title="DTW Dendrogram",
        summary="Hierarchical clustering by DTW distance of recent return paths.",
        artifacts=[Artifact(kind="image/png", name="dtw_dendrogram.png", payload=png)]
    )

@register_card("cluster.xplot_3d_returns", "3D X-Plot (returns)", "cluster", min_tier="black", cost=10, heavy=True, slots=("S11",))
def xplot(ctx: CardContext) -> CardResult:
    # 3 tickers from ctx.tickers
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    tks = (ctx.tickers + ["SPY","QQQ","IWM"])[:3]
    rets=[]
    for tk in tks:
        df = ctx.market.get_ohlcv(tk, start=ctx.start, end=ctx.end, interval="1d").iloc[-120:]
        rets.append(df["Close"].pct_change().fillna(0).values)
    r1,r2,r3 = rets
    n = min(len(r1),len(r2),len(r3))
    r1,r2,r3 = r1[-n:], r2[-n:], r3[-n:]

    fig = plt.figure(figsize=(10,5.2))
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(r1, r2, r3, s=10)
    ax.set_title(f"3D Return Cloud: {tks[0]}, {tks[1]}, {tks[2]}")
    ax.set_xlabel(tks[0]); ax.set_ylabel(tks[1]); ax.set_zlabel(tks[2])
    png = fig_to_png_bytes(fig)
    return CardResult(
        key="cluster.xplot_3d_returns",
        title="3D Return X-Plot",
        summary="Return vector cloud for three assets (relationship shape).",
        artifacts=[Artifact(kind="image/png", name="xplot_3d.png", payload=png)]
    )
