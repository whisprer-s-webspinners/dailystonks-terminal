from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, rsi

@register_card("tuning.successive_halving_report", "Successive Halving (toy tuning)", "tuning", min_tier="black", cost=12, heavy=True, slots=("S11",))
def halving_report(ctx: CardContext) -> CardResult:
    # Use HalvingGridSearchCV on a simple classifier for next-day direction.
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-800:].copy()
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(0)
    r = rsi(close).fillna(50)/100.0
    X = np.column_stack([ret.values, vol.values, r.values])
    y = (ret.shift(-1).fillna(0).values > 0).astype(int)

    X = X[:-1]
    y = y[:-1]
    # cap size for runtime
    if len(X) > 600:
        X = X[-600:]; y = y[-600:]

    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 5, 15],
    }
    search = HalvingGridSearchCV(clf, param_grid, factor=3, resource="n_estimators", max_resources=400, random_state=0, scoring="accuracy")
    search.fit(X, y)

    res = pd.DataFrame(search.cv_results_)
    # plot mean_test_score vs iter for top candidates
    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(res["iter"].values, res["mean_test_score"].values, marker="o")
    ax.set_title(f"{t} Successive Halving: mean CV score by iter")
    ax.set_xlabel("iter")
    ax.set_ylabel("mean_test_score")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    best = search.best_params_
    return CardResult(
        key="tuning.successive_halving_report",
        title=f"{t}: Successive Halving Tuning",
        summary="Toy tuning report for a RF classifier (next-day direction).",
        metrics={"best_score": round(float(search.best_score_),4), **{f"best_{k}": v for k,v in best.items()}},
        artifacts=[Artifact(kind="image/png", name="halving.png", payload=png)]
    )
