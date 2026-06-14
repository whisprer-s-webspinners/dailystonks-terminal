from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, rsi, macd

@register_card("regime.pca_kmeans_colored_price", "Regime Map (PCA+KMeans)", "regime", min_tier="black", cost=12, heavy=True, slots=("S11",))
def pca_kmeans(ctx: CardContext) -> CardResult:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    r = rsi(close).fillna(50)
    mline, msig, mh = macd(close)
    feat = np.column_stack([ret.values, vol.values, r.values/100.0, mh.fillna(0).values])
    # standardize
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-12)

    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(feat)
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    lab = km.fit_predict(z)

    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(close.values, linewidth=1.2, label="Close")
    # background shading by regime label changes
    for k in range(3):
        idx = np.where(lab == k)[0]
        ax.scatter(idx, close.values[idx], s=10, label=f"Regime {k}")
    ax.set_title(f"{t} Regimes (PCA+KMeans)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    counts = {f"Regime{k}": int((lab==k).sum()) for k in range(3)}
    return CardResult(
        key="regime.pca_kmeans_colored_price",
        title=f"{t}: Regime Map (3 clusters)",
        summary="Unsupervised regimes from returns/vol/RSI/MACD-hist features.",
        metrics=counts,
        artifacts=[Artifact(kind="image/png", name=f"{t}_regimes.png", payload=png)]
    )

@register_card("regime.transition_matrix_badge", "Regime Transition Matrix", "regime", min_tier="black", cost=10, heavy=True, slots=("S11",))
def transition_matrix(ctx: CardContext) -> CardResult:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    r = rsi(close).fillna(50)
    mline, msig, mh = macd(close)
    feat = np.column_stack([ret.values, vol.values, r.values/100.0, mh.fillna(0).values])
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-12)
    z = PCA(n_components=2, random_state=0).fit_transform(feat)
    lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(z)

    # transition counts
    M = np.zeros((3,3), dtype=int)
    for i in range(1, len(lab)):
        M[lab[i-1], lab[i]] += 1
    P = M / (M.sum(axis=1, keepdims=True) + 1e-12)

    fig = plt.figure(figsize=(6.6,5.2))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(P, aspect="auto", origin="upper")
    ax.set_title(f"{t} Regime Transition Probabilities")
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="regime.transition_matrix_badge",
        title=f"{t}: Regime Transition Matrix",
        summary="Probabilities of switching between 3 unsupervised regimes.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_regime_transitions.png", payload=png)]
    )

@register_card("regime.performance_by_regime", "Performance by Regime", "regime", min_tier="black", cost=12, heavy=True, slots=("S11",))
def perf_by_regime(ctx: CardContext) -> CardResult:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    r = rsi(close).fillna(50)
    mline, msig, mh = macd(close)
    feat = np.column_stack([ret.values, vol.values, r.values/100.0, mh.fillna(0).values])
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-12)
    z = PCA(n_components=2, random_state=0).fit_transform(feat)
    lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(z)

    # future returns by regime
    fwd = 5
    fwd_ret = close.pct_change(fwd).shift(-fwd).fillna(0)
    rows = []
    for k in range(3):
        vals = fwd_ret[lab==k].values
        rows.append([k, float(vals.mean()), float(np.quantile(vals,0.1)), float(np.quantile(vals,0.9)), int(len(vals))])
    tab = pd.DataFrame(rows, columns=["Regime","Mean fwd5","P10","P90","N"])

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(f"{t} Forward 5D Returns by Regime")
    table = ax.table(cellText=tab.round(4).values, colLabels=tab.columns, loc="center")
    table.scale(1, 1.6)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="regime.performance_by_regime",
        title=f"{t}: Forward Returns by Regime",
        summary="Forward 5D return distribution by regime label.",
        artifacts=[Artifact(kind="image/png", name=f"{t}_regime_perf.png", payload=png)]
    )

@register_card("regime.hmm_state_timeline", "HMM-like State Timeline (MarkovRegression)", "regime", min_tier="black", cost=12, heavy=True, slots=("S03",))
def hmm_timeline(ctx: CardContext) -> CardResult:
    # statsmodels MarkovRegression on returns as a lightweight HMM proxy.
    import statsmodels.api as sm
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-800:].copy()
    ret = df["Close"].pct_change().dropna()
    if len(ret) < 200:
        raise RuntimeError("Not enough history for MarkovRegression.")
    mod = sm.tsa.MarkovRegression(ret.values, k_regimes=2, trend='c', switching_variance=True)
    res = mod.fit(disp=False)
    probs = res.smoothed_marginal_probabilities[1]  # regime 1 prob

    fig = plt.figure(figsize=(10,4.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(df["Close"].iloc[-len(probs):].values, label="Close")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.25)
    ax2.plot(probs.values, label="Prob(Regime1)")
    ax2.set_ylim(0,1)
    ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.25)
    ax2.set_title("Smoothed regime probability (2-state)")
    png = fig_to_png_bytes(fig)

    lastp = float(probs.iloc[-1])
    return CardResult(
        key="regime.hmm_state_timeline",
        title=f"{t}: 2-State Regime Probability",
        summary="MarkovRegression on returns as an HMM-like regime proxy.",
        metrics={"Prob(Regime1)": round(lastp,3)},
        artifacts=[Artifact(kind="image/png", name=f"{t}_hmm.png", payload=png)]
    )
