from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

@register_card("events.calendar_proxy", "Event Risk: Earnings Calendar (best-effort)", "events", min_tier="pro", cost=6, heavy=False, slots=("S02",))
def earnings_calendar(ctx: CardContext) -> CardResult:
    # Best-effort: yfinance calendar sometimes empty; we tolerate unknowns.
    try:
        import yfinance as yf
    except Exception as e:
        return CardResult(
            key="events.calendar_proxy",
            title="Event Risk: Earnings Calendar",
            summary="yfinance required for earnings calendar metadata.",
            warnings=[repr(e)]
        )

    tks = (ctx.tickers + ["SPY","QQQ","AAPL","MSFT","NVDA"])[:12]
    tks = list(dict.fromkeys([t.replace(".", "-") for t in tks]))

    rows=[]
    unknown=0
    for t in tks:
        date_str = "Unknown"
        days = ""
        try:
            cal = yf.Ticker(t).calendar
            # calendar is a DataFrame with index like 'Earnings Date'
            if cal is not None and not cal.empty:
                # Try to find an earnings date-like value
                # yfinance sometimes stores a Timestamp in first column for 'Earnings Date'
                if "Earnings Date" in cal.index:
                    v = cal.loc["Earnings Date"].values
                    # v could have 1-2 dates; take first non-null
                    cand = None
                    for x in v:
                        if pd.notna(x):
                            cand = x
                            break
                    if cand is not None:
                        d = pd.to_datetime(cand).date()
                        date_str = str(d)
                        days = (d - ctx.as_of).days
        except Exception:
            pass

        if date_str == "Unknown":
            unknown += 1
        rows.append((t, date_str, days))

    df = pd.DataFrame(rows, columns=["Symbol","Next earnings (est.)","Days"])
    png = _table_png("Event Risk (best-effort): Earnings Dates", df)

    warnings=[]
    if unknown:
        warnings.append(f"{unknown}/{len(tks)} symbols had no earnings date available via yfinance.")

    return CardResult(
        key="events.calendar_proxy",
        title="Event Risk: Earnings Calendar (best-effort)",
        summary="Calendar metadata via yfinance (often incomplete; treat as a hint).",
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="earnings_calendar.png", payload=png)]
    )

@register_card("risk.gap_risk_score", "Gap Risk Score (overnight gaps)", "risk", min_tier="black", cost=8, heavy=False, slots=("S09",))
def gap_risk(ctx: CardContext) -> CardResult:
    tks = (ctx.tickers + ["SPY","QQQ","AAPL","MSFT","NVDA"])[:10]
    tks = list(dict.fromkeys([t.replace(".", "-") for t in tks]))

    rows=[]
    for t in tks:
        try:
            df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-520:].copy()
            if df.empty or len(df) < 60:
                continue
            o = df["Open"].astype(float)
            c = df["Close"].astype(float)
            pc = c.shift(1)
            gap = (o - pc) / (pc + 1e-12)
            # gap magnitude stats
            g = gap.dropna()
            g_abs = g.abs()
            p95 = float(g_abs.quantile(0.95))
            mean = float(g_abs.mean())
            # combine with vol proxy from close returns
            r = c.pct_change().dropna()
            sig20 = float(r.rolling(20).std(ddof=1).iloc[-1]) if len(r) >= 25 else float("nan")
            score = (p95 / (sig20 + 1e-12)) if np.isfinite(sig20) else float("nan")
            rows.append((t, mean*100, p95*100, sig20*100 if np.isfinite(sig20) else np.nan, score))
        except Exception:
            continue

    if not rows:
        return CardResult(
            key="risk.gap_risk_score",
            title="Gap Risk Score",
            summary="No symbols had sufficient data for gap stats."
        )

    tab = pd.DataFrame(rows, columns=["Symbol","Mean |gap| %","P95 |gap| %","Sigma20 %","GapScore"])
    tab = tab.sort_values("GapScore", ascending=False)
    show = tab.copy()
    show["Mean |gap| %"] = show["Mean |gap| %"].map(lambda x: f"{x:.2f}")
    show["P95 |gap| %"] = show["P95 |gap| %"].map(lambda x: f"{x:.2f}")
    show["Sigma20 %"] = show["Sigma20 %"].map(lambda x: "" if not np.isfinite(x) else f"{x:.2f}")
    show["GapScore"] = show["GapScore"].map(lambda x: "" if not np.isfinite(x) else f"{x:.2f}")

    # small chart: top 5 by score
    top = tab.head(5).copy()
    fig = plt.figure(figsize=(10,4.6))
    ax = fig.add_subplot(1,1,1)
    ax.bar(top["Symbol"].values, top["GapScore"].values)
    ax.set_title("GapScore (P95 gap / sigma20) — Top 5")
    ax.grid(True, alpha=0.25, axis="y")
    png_chart = fig_to_png_bytes(fig)

    png_table = _table_png("Gap Risk Score (overnight gaps)", show.head(12))

    return CardResult(
        key="risk.gap_risk_score",
        title="Gap Risk Score",
        summary="Overnight gap magnitude vs recent volatility (higher = more gap risk).",
        artifacts=[
            Artifact(kind="image/png", name="gap_risk_table.png", payload=png_table),
            Artifact(kind="image/png", name="gap_risk_top5.png", payload=png_chart),
        ]
    )

@register_card("narrative.what_to_watch", "What to Watch (auto bullets)", "narrative", min_tier="pro", cost=4, heavy=False, slots=("S01",))
def what_to_watch(ctx: CardContext) -> CardResult:
    # Simple, robust heuristics that produce readable bullets every day.
    bullets=[]
    warnings=[]

    # SPY vol + momentum
    try:
        spy = ctx.market.get_ohlcv("SPY", start=ctx.start, end=ctx.end, interval="1d").iloc[-260:].copy()
        c = spy["Close"].astype(float)
        r = c.pct_change().dropna()
        vol20 = float(r.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(252))
        mom20 = float(c.pct_change(20).iloc[-1])
        bullets.append(f"- SPY 20D momentum: {mom20*100:+.2f}% · realized vol20: {vol20*100:.1f}%")
        if vol20 > (r.rolling(252).std(ddof=1).mean() * np.sqrt(252)):
            bullets.append("- Volatility is elevated vs typical: tighten size / prefer defined-risk setups.")
        if abs(mom20) < 0.02:
            bullets.append("- Momentum is muted: mean-reversion setups may have better expectancy than breakouts.")
    except Exception as e:
        warnings.append(f"SPY metrics failed: {e!r}")

    # Macro corr quick read (best effort)
    try:
        basket = ["SPY","BTC-USD","UUP","TLT","GLD"]
        data = ctx.market.get_ohlcv_many(basket, start=ctx.start, end=ctx.end, interval="1d")
        if "SPY" in data:
            spy = data["SPY"]["Close"].pct_change().dropna()
            for raw in ["BTC-USD","UUP","TLT","GLD"]:
                t = raw.replace(".","-")
                if t in data and "Close" in data[t]:
                    rr = data[t]["Close"].pct_change().dropna()
                    m = pd.concat([spy, rr], axis=1).dropna()
                    if len(m) >= 60:
                        corr = float(m.iloc[-60:,0].corr(m.iloc[-60:,1]))
                        bullets.append(f"- Corr60(SPY,{t}): {corr:+.2f}")
    except Exception:
        pass

    if not bullets:
        bullets = ["- No signals available (data coverage too low today)."]

    return CardResult(
        key="narrative.what_to_watch",
        title="What to Watch",
        summary="Auto-generated watch bullets (robust heuristics).",
        bullets=["What to watch"] + bullets,
        warnings=warnings
    )