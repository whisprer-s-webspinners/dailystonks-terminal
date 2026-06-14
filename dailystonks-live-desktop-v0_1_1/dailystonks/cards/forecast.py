from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

@register_card("forecast.ridge_overlay_5d_slope", "Ridge Forecast Overlay (5D)", "forecast", min_tier="pro", cost=7, heavy=False, slots=("S10",))
def ridge_overlay(ctx: CardContext) -> CardResult:
    from sklearn.linear_model import Ridge
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-260:]
    y = df["Close"].values.astype(float)
    x = np.arange(len(y)).reshape(-1,1)
    model = Ridge(alpha=10.0)
    model.fit(x, y)
    # forecast 5 trading days ahead
    x_future = np.arange(len(y)+5).reshape(-1,1)
    yhat = model.predict(x_future)
    resid = y - model.predict(x)
    sigma = float(np.std(resid, ddof=1))

    fig = plt.figure(figsize=(10,4.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, label="Close")
    ax.plot(yhat, label="Ridge trend+5D")
    ax.fill_between(np.arange(len(yhat)), yhat-1.5*sigma, yhat+1.5*sigma, alpha=0.2, label="~1.5σ band")
    ax.set_title(f"{t} Ridge Forecast Overlay")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    slope = float(model.coef_[0])
    return CardResult(
        key="forecast.ridge_overlay_5d_slope",
        title=f"{t}: Ridge Forecast (5D overlay)",
        summary="Lightweight forecast: linear ridge trend with residual band.",
        metrics={"Slope (per bar)": round(slope,6), "Residual σ": round(sigma,4)},
        artifacts=[Artifact(kind="image/png", name=f"{t}_ridge_forecast.png", payload=png)]
    )

@register_card("forecast.bayesian_fan_credible", "Bayesian Fan (Credible Interval)", "forecast", min_tier="black", cost=12, heavy=True, slots=("S10",))
def bayesian_fan(ctx: CardContext) -> CardResult:
    # Conjugate Bayesian linear regression on time -> close with NIG prior.
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:]
    y = df["Close"].values.astype(float)
    n = len(y)
    X = np.column_stack([np.ones(n), np.arange(n)])
    # Prior
    beta0 = np.zeros(2)
    V0 = np.eye(2) * 1e6
    a0 = 3.0
    b0 = 1.0

    # Posterior
    XtX = X.T @ X
    V0_inv = np.linalg.inv(V0)
    Vn = np.linalg.inv(V0_inv + XtX)
    betan = Vn @ (V0_inv @ beta0 + X.T @ y)

    # residuals with posterior mean
    yhat = X @ betan
    resid = y - yhat
    an = a0 + n/2.0
    bn = b0 + 0.5*(resid @ resid)

    # Predictive for future points: Student-t
    horizon = 20
    Xf = np.column_stack([np.ones(n+horizon), np.arange(n+horizon)])
    mean_f = Xf @ betan
    # predictive variance factor
    s2 = bn / an
    # Var = s2 * (1 + x Vn x^T)
    quad = np.sum(Xf * (Xf @ Vn), axis=1)
    var_f = s2 * (1.0 + quad)
    std_f = np.sqrt(var_f)

    # approximate quantiles using normal (t close when df large); to keep fast.
    q10 = mean_f - 1.2816*std_f
    q50 = mean_f
    q90 = mean_f + 1.2816*std_f

    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(y, label="Close")
    ax.plot(q50, label="Posterior mean")
    ax.fill_between(np.arange(len(q50)), q10, q90, alpha=0.2, label="80% band")
    ax.set_title(f"{t} Bayesian Fan (20D)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    png = fig_to_png_bytes(fig)

    return CardResult(
        key="forecast.bayesian_fan_credible",
        title=f"{t}: Bayesian Fan (80% band)",
        summary="Bayesian linear model on time with credible band (fast, deterministic).",
        metrics={"Horizon": horizon, "Posterior s²": round(float(s2),6)},
        artifacts=[Artifact(kind="image/png", name=f"{t}_bayes_fan.png", payload=png)]
    )
