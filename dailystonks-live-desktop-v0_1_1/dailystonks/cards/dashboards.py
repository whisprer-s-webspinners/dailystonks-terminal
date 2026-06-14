from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes, rsi

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"].astype(float)
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0).abs()
    pmf = pos.rolling(period).sum()
    nmf = neg.rolling(period).sum()
    rr = pmf / (nmf + 1e-12)
    return 100.0 - (100.0 / (1.0 + rr))

def zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=max(30, win//3)).mean()
    sd = s.rolling(win, min_periods=max(30, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

@register_card("dash.trend_meanreversion", "Trend + Mean-Reversion Dashboard", "dash", min_tier="pro", cost=8, heavy=False, slots=("S08",))
def trend_meanrev(ctx: CardContext) -> CardResult:
    t = ctx.tickers[0] if ctx.tickers else "SPY"
    df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
    close = df["Close"].astype(float)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    r = rsi(close).fillna(method="bfill")
    mf = mfi(df).fillna(method="bfill")

    dist = (close - ma20) / (ma20 + 1e-12)
    zz = zscore(dist, 120)

    fig = plt.figure(figsize=(10,7.2))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    ax1.plot(close.values, label="Close")
    ax1.plot(ma20.values, label="MA20")
    ax1.plot(ma50.values, label="MA50")
    ax1.plot(ma200.values, label="MA200")
    ax1.set_title(f"{t} Trend + Mean-Reversion Dashboard")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=4, fontsize=8)

    ax2.plot(r.values, label="RSI14")
    ax2.plot(mf.values, label="MFI14")
    ax2.axhline(70, linewidth=1); ax2.axhline(30, linewidth=1)
    ax2.set_ylim(0,100)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    ax3.plot(zz.values, label="Z(dist to MA20)")
    ax3.axhline(0, linewidth=1)
    ax3.axhline(2, linewidth=1); ax3.axhline(-2, linewidth=1)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "RSI14": round(float(r.iloc[-1]), 1),
        "MFI14": round(float(mf.iloc[-1]), 1),
        "Z(dist->MA20)": round(float(zz.iloc[-1]), 2) if np.isfinite(zz.iloc[-1]) else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="dash.trend_meanreversion",
        title=f"{t}: Trend + Mean-Reversion Dashboard",
        summary="MA stack + RSI/MFI + z-score of distance to MA20.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name=f"{t}_trend_meanrev.png", payload=png)]
    )

@register_card("dash.breadth_participation", "Breadth + Participation Dashboard", "dash", min_tier="black", cost=10, heavy=True, slots=("S04",))
def breadth(ctx: CardContext) -> CardResult:
    uni = ctx.sp500.tickers(max_n=min(ctx.max_universe, 120))
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    adv = dec = flat = 0
    above50 = above200 = 0
    total = 0
    fails = 0

    for raw in uni:
        t = raw.replace(".", "-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1
            continue
        c = df["Close"].astype(float).dropna()
        if len(c) < 220:
            fails += 1
            continue
        r1 = float(c.pct_change().iloc[-1])
        if r1 > 0: adv += 1
        elif r1 < 0: dec += 1
        else: flat += 1
        ma50 = float(c.rolling(50).mean().iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1])
        last = float(c.iloc[-1])
        above50 += 1 if last > ma50 else 0
        above200 += 1 if last > ma200 else 0
        total += 1

    if total < 20:
        return CardResult(
            key="dash.breadth_participation",
            title="Breadth + Participation",
            summary="Not enough symbols to compute breadth (coverage too low).",
            warnings=[f"ok={total}, fails={fails}"]
        )

    adv_pct = adv/total
    dec_pct = dec/total
    net = (adv - dec) / total

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    cats = ["Adv%", "Dec%", "%>MA50", "%>MA200", "Net(Adv-Dec)"]
    vals = [adv_pct*100, dec_pct*100, (above50/total)*100, (above200/total)*100, net*100]
    ax.bar(cats, vals)
    ax.set_title("Breadth + Participation (capped S&P500)")
    ax.grid(True, alpha=0.25, axis="y")
    png = fig_to_png_bytes(fig)

    metrics = {
        "N": total,
        "Adv": adv,
        "Dec": dec,
        "%>MA50": round((above50/total)*100, 1),
        "%>MA200": round((above200/total)*100, 1),
        "Net%": round(net*100, 1),
        "Fails": fails,
    }

    return CardResult(
        key="dash.breadth_participation",
        title="Breadth + Participation Dashboard",
        summary="Advance/decline proxy + % above MA50/MA200 + net breadth.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="breadth_participation.png", payload=png)]
    )

@register_card("dash.macro_corr_panel", "Macro Overlay + Corr Table", "dash", min_tier="pro", cost=8, heavy=False, slots=("S03",))
def macro_corr(ctx: CardContext) -> CardResult:
    basket = ["SPY", "BTC-USD", "UUP", "TLT", "GLD"]
    data = ctx.market.get_ohlcv_many(basket, start=ctx.start, end=ctx.end, interval="1d")

    frames=[]
    missing=[]
    for raw in basket:
        t = raw.replace(".", "-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            missing.append(raw); continue
        frames.append(df["Close"].rename(t))

    if len(frames) < 2:
        return CardResult(
            key="dash.macro_corr_panel",
            title="Macro Overlay + Corr Table",
            summary="Not enough macro series downloaded to render.",
            warnings=[f"missing: {', '.join(missing)}"]
        )

    prices = pd.concat(frames, axis=1).dropna(how="any")
    normed = prices / prices.iloc[0]

    rets = prices.pct_change().dropna()
    corr60 = {}
    if "SPY" in rets.columns and len(rets) >= 70:
        for col in rets.columns:
            if col == "SPY": continue
            corr60[f"Corr60(SPY,{col})"] = round(float(rets["SPY"].rolling(60).corr(rets[col]).iloc[-1]), 3)

    fig = plt.figure(figsize=(10,5.6))
    ax = fig.add_subplot(1,1,1)
    for c in normed.columns:
        ax.plot(normed.index, normed[c].values, label=c)
    ax.set_title("Macro Overlay (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_xticks([])
    png = fig_to_png_bytes(fig)

    warnings=[]
    if missing: warnings.append(f"Missing series skipped: {', '.join(missing)}")

    return CardResult(
        key="dash.macro_corr_panel",
        title="Macro Overlay + Corr Table",
        summary="Cross-asset overlay + rolling correlation vs SPY.",
        metrics=corr60,
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="macro_corr.png", payload=png)]
    )