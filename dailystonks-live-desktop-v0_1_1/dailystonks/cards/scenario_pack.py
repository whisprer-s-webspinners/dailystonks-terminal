
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

TRADING_DAYS = 252

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1400) -> pd.DataFrame:
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

def _beta_and_corr(rets: pd.DataFrame, y: str, x: str, win: int = 252):
    # beta(y~x), corr(y,x) using last win rows
    m = rets[[y,x]].dropna().iloc[-win:]
    if len(m) < max(80, win//3):
        return float("nan"), float("nan")
    vx = float(m[x].var(ddof=1))
    if not np.isfinite(vx) or vx <= 0:
        return float("nan"), float("nan")
    cov = float(m[y].cov(m[x]))
    beta = cov / (vx + 1e-12)
    corr = float(m[y].corr(m[x]))
    return float(beta), float(corr)

def _ann_vol(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 40:
        return float("nan")
    return float(s.std(ddof=1) * math.sqrt(TRADING_DAYS))

# -------------------------------
# 1) Scenario shock table (flagship)
# -------------------------------
@register_card("risk.scenario_shock_table", "Risk: Scenario Shock Table (SPY ±σ)", "risk",
               min_tier="black", cost=9, heavy=False, slots=("S09","S03"))
def scenario_shock(ctx: CardContext) -> CardResult:
    # Macro grid basket (robust ETF/proxies; BTC optional)
    basket = ["SPY","QQQ","IWM","TLT","GLD","DBC","HYG","UUP","VIXY","BTC-USD"]
    px = _get_closes(ctx, basket, bars=1600)
    rets = px.pct_change().dropna()
    if "SPY" not in rets.columns or len(rets) < 120:
        return CardResult(
            key="risk.scenario_shock_table",
            title="Risk: Scenario Shock Table (SPY ±σ)",
            summary="Need SPY and sufficient overlapping returns.",
            warnings=[f"cols={px.columns.tolist()} rows={len(rets)}"]
        )

    # Use current realized daily sigma from last 20 days (fast + “current”)
    sig20 = float(rets["SPY"].rolling(20).std(ddof=1).iloc[-1])
    if not np.isfinite(sig20) or sig20 <= 0:
        sig20 = float(rets["SPY"].std(ddof=1))
    if not np.isfinite(sig20) or sig20 <= 0:
        return CardResult(
            key="risk.scenario_shock_table",
            title="Risk: Scenario Shock Table (SPY ±σ)",
            summary="Could not compute SPY volatility."
        )

    shocks = {
        "SPY -2σ": -2.0*sig20,
        "SPY -1σ": -1.0*sig20,
        "SPY +1σ": +1.0*sig20,
        "SPY +2σ": +2.0*sig20,
    }

    rows=[]
    spy = "SPY"
    for a in rets.columns:
        if a == spy:
            beta, corr = 1.0, 1.0
        else:
            beta, corr = _beta_and_corr(rets, a, spy, win=252)
        vol = _ann_vol(rets[a])

        # expected move = beta * shock
        exp = {name: (beta * sh) for name, sh in shocks.items()} if np.isfinite(beta) else {name: float("nan") for name in shocks}
        rows.append([
            a,
            "" if not np.isfinite(beta) else f"{beta:+.2f}",
            "" if not np.isfinite(corr) else f"{corr:+.2f}",
            "" if not np.isfinite(vol) else f"{vol*100:.1f}",
            "" if not np.isfinite(exp["SPY -2σ"]) else f"{exp['SPY -2σ']*100:+.2f}",
            "" if not np.isfinite(exp["SPY -1σ"]) else f"{exp['SPY -1σ']*100:+.2f}",
            "" if not np.isfinite(exp["SPY +1σ"]) else f"{exp['SPY +1σ']*100:+.2f}",
            "" if not np.isfinite(exp["SPY +2σ"]) else f"{exp['SPY +2σ']*100:+.2f}",
        ])

    out = pd.DataFrame(rows, columns=["Asset","Beta","Corr","VolAnn%","Exp(-2σ)%","Exp(-1σ)%","Exp(+1σ)%","Exp(+2σ)%"])

    # Sort by Beta descending but keep SPY top
    out["_b"] = pd.to_numeric(out["Beta"].str.replace("+","", regex=False), errors="coerce")
    out = pd.concat([out[out["Asset"]=="SPY"], out[out["Asset"]!="SPY"].sort_values("_b", ascending=False)], axis=0)
    out = out.drop(columns=["_b"])

    png_tbl = _table_png(f"Scenario Shock Table — SPY σ20={sig20*100:.2f}% (daily)", out)

    # Chart: expected move under SPY -2σ for top betas
    chart_df = out[out["Asset"]!="SPY"].copy()
    chart_df["Exp(-2σ)%"] = pd.to_numeric(chart_df["Exp(-2σ)%"].str.replace("+","", regex=False), errors="coerce")
    chart_df = chart_df.dropna(subset=["Exp(-2σ)%"]).head(8)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    ax.bar(chart_df["Asset"].values, chart_df["Exp(-2σ)%"].values)
    ax.set_title("Expected move under SPY -2σ (beta-based)")
    ax.grid(True, alpha=0.25, axis="y")
    png_bar = fig_to_png_bytes(fig)

    metrics = {
        "SPY sigma20 daily %": round(sig20*100, 2),
        "AssetsUsed": int(len(rets.columns)),
    }

    bullets = [
        "Expected moves are beta-based (OLS via cov/var over ~252D).",
        "This is a scenario sheet, not a forecast: it answers ‘if SPY moves X, what likely moves with it?’",
    ]

    return CardResult(
        key="risk.scenario_shock_table",
        title="Risk: Scenario Shock Table (SPY ±σ)",
        summary="Beta-based expected moves for macro-grid assets under SPY shocks.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="scenario_shock_table.png", payload=png_tbl),
            Artifact(kind="image/png", name="scenario_shock_bar.png", payload=png_bar),
        ],
    )

# -------------------------------
# 2) Cross-asset VaR proxy (macro grid)
# -------------------------------
@register_card("risk.cross_asset_var_proxy", "Risk: Cross-Asset VaR Proxy (macro grid)", "risk",
               min_tier="black", cost=8, heavy=False, slots=("S09",))
def cross_asset_var(ctx: CardContext) -> CardResult:
    basket = ["SPY","QQQ","IWM","TLT","GLD","DBC","HYG","UUP","VIXY","BTC-USD"]
    px = _get_closes(ctx, basket, bars=1600)
    rets = px.pct_change().dropna()
    if len(rets) < 200:
        return CardResult(
            key="risk.cross_asset_var_proxy",
            title="Risk: Cross-Asset VaR Proxy (macro grid)",
            summary="Not enough overlapping returns (need ~200+)."
        )

    # Equal-weight portfolio VaR proxy from historical returns distribution
    cols = rets.columns.tolist()
    w = np.ones(len(cols)) / len(cols)
    port = rets[cols].values @ w

    var95 = float(np.quantile(port, 0.05))
    es95 = float(port[port <= var95].mean())
    var99 = float(np.quantile(port, 0.01))
    es99 = float(port[port <= var99].mean())

    # Risk contributions (approx) via covariance with portfolio return
    pr = pd.Series(port, index=rets.index)
    contrib=[]
    for a in cols:
        cov = float(rets[a].cov(pr))
        contrib.append((a, cov))
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)

    tab = pd.DataFrame({
        "Asset": [a for a,_ in contrib[:12]],
        "Cov(asset,port)": [f"{c:+.6f}" for _,c in contrib[:12]],
    })
    png_tbl = _table_png("Top risk contributors (by |cov with portfolio|)", tab)

    metrics = {
        "PortVaR95%": round(var95*100, 2),
        "PortES95%": round(es95*100, 2),
        "PortVaR99%": round(var99*100, 2),
        "PortES99%": round(es99*100, 2),
        "Assets": len(cols),
    }

    bullets = [
        "Equal-weight macro grid portfolio (historical daily VaR/ES proxy).",
        "Contrib uses covariance with portfolio return (quick risk ‘drivers’ list).",
    ]

    return CardResult(
        key="risk.cross_asset_var_proxy",
        title="Risk: Cross-Asset VaR Proxy (macro grid)",
        summary="Portfolio tail-risk proxy for the macro grid + top contributors.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="cross_asset_var_contrib.png", payload=png_tbl)],
    )

# -------------------------------
# 3) Diversifier rankings
# -------------------------------
@register_card("risk.diversifier_rankings", "Risk: Diversifier Rankings (vs SPY)", "risk",
               min_tier="black", cost=7, heavy=False, slots=("S09",))
def diversifier_rank(ctx: CardContext) -> CardResult:
    basket = ["SPY","TLT","GLD","UUP","DBC","HYG","VIXY","BTC-USD"]
    px = _get_closes(ctx, basket, bars=1600)
    rets = px.pct_change().dropna()
    if "SPY" not in rets.columns or len(rets) < 200:
        return CardResult(
            key="risk.diversifier_rankings",
            title="Risk: Diversifier Rankings (vs SPY)",
            summary="Need SPY + enough overlapping returns."
        )

    spy = "SPY"
    rows=[]
    for a in rets.columns:
        if a == spy:
            continue
        beta, corr = _beta_and_corr(rets, a, spy, win=252)

        # Hedge score: prefer negative corr, negative beta, and positive performance during SPY down days
        down = rets[spy] < 0
        hedge_ret = float(rets.loc[down, a].mean()) if down.sum() > 30 else float("nan")
        score = 0.0
        if np.isfinite(corr): score += (-corr)
        if np.isfinite(beta): score += (-0.5*beta)
        if np.isfinite(hedge_ret): score += (5.0*hedge_ret)

        rows.append((a, beta, corr, hedge_ret, score))

    df = pd.DataFrame(rows, columns=["Asset","Beta","Corr","MeanRet_on_SPYdown","HedgeScore"]).sort_values("HedgeScore", ascending=False)
    out = pd.DataFrame({
        "Asset": df["Asset"].values,
        "Beta": df["Beta"].map(lambda x: "" if not np.isfinite(x) else f"{x:+.2f}").values,
        "Corr": df["Corr"].map(lambda x: "" if not np.isfinite(x) else f"{x:+.2f}").values,
        "Mean on SPY↓ %": df["MeanRet_on_SPYdown"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:+.2f}").values,
        "HedgeScore": df["HedgeScore"].map(lambda x: "" if not np.isfinite(x) else f"{x:+.3f}").values,
    })
    png_tbl = _table_png("Diversifier rankings vs SPY (higher = better hedge)", out.head(12))

    best = df.iloc[0]["Asset"] if len(df) else ""
    metrics = {"TopDiversifier": str(best)}

    bullets = [
        "HedgeScore is a simple composite: prefers negative corr/beta and positive mean on SPY down days.",
        "Use as a quick ‘what hedges right now?’ desk read.",
    ]

    return CardResult(
        key="risk.diversifier_rankings",
        title="Risk: Diversifier Rankings (vs SPY)",
        summary="Rank assets by hedge behavior vs SPY (beta/corr + down-day behavior).",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="diversifier_rankings.png", payload=png_tbl)],
    )
