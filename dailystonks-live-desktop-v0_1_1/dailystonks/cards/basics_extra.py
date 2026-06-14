from __future__ import annotations
import numpy as np
import pandas as pd

from ..core.registry import register_card
from ..core.models import CardContext, CardResult
from ..core.utils import safe_pct

@register_card("composite.score_leaderboard", "Composite Score Leaderboard (stub scoring)", "composite", min_tier="pro", cost=6, heavy=False, slots=("S01","S08"))
def composite_score(ctx: CardContext) -> CardResult:
    # Heuristic composite score on capped universe.
    import numpy as np
    tickers = ctx.sp500.tickers(max_n=ctx.max_universe)
    rows=[]
    for tk in tickers:
        try:
            df = ctx.market.get_ohlcv(tk, start=ctx.start, end=ctx.end, interval="1d").iloc[-260:]
            close = df["Close"]
            ret20 = float(close.pct_change(20).iloc[-1])
            vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
            score = (ret20/(vol20+1e-6))
            rows.append((tk, score, float(close.iloc[-1])))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:15]
    bullets = [
        "Stub composite = 20D return / 20D vol (Sharpe-like).",
        "Replace with your full Wofl composite engine when ready."
    ]
    metrics = {f"#{i+1}": f"{sym} ({score:.2f})" for i,(sym,score,_) in enumerate(top[:6])}
    return CardResult(
        key="composite.score_leaderboard",
        title="Composite Score Leaderboard",
        summary="Top symbols by a quick Sharpe-like heuristic.",
        metrics=metrics,
        bullets=bullets
    )

@register_card("bt.naive_vs_model_error", "Baseline vs Model Error (toy)", "backtest", min_tier="pro", cost=6, heavy=False, slots=("S01","S10"))
def naive_vs_model(ctx: CardContext) -> CardResult:
    # Compare naive forecast (y_t) vs linear trend forecast (ridge) on last 120 points.
    from sklearn.linear_model import Ridge
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-260:]
    y = df["Close"].values.astype(float)
    y_true = y[1:]
    naive = y[:-1]
    naive_mae = float(np.mean(np.abs(y_true - naive)))

    x = np.arange(len(y)).reshape(-1,1)
    model = Ridge(alpha=10.0).fit(x, y)
    yhat = model.predict(x)[1:]
    model_mae = float(np.mean(np.abs(y_true - yhat)))

    return CardResult(
        key="bt.naive_vs_model_error",
        title=f"{t}: Baseline vs Model Error",
        summary="Sanity check: model must beat naive persistence to be worth it.",
        metrics={"Naive MAE": round(naive_mae,4), "Model MAE": round(model_mae,4), "Δ%": round(safe_pct(model_mae, naive_mae),2)}
    )
