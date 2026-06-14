
from __future__ import annotations
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

def _load_sp500_universe(ctx: CardContext, cap: int) -> tuple[list[str], dict[str,str]]:
    sp = ctx.sp500.df()
    syms = sp["Symbol"].astype(str).tolist()
    sec = {}
    if "GICS Sector" in sp.columns:
        sec = dict(zip(sp["Symbol"].astype(str), sp["GICS Sector"].astype(str)))
    elif "Sector" in sp.columns:
        sec = dict(zip(sp["Symbol"].astype(str), sp["Sector"].astype(str)))
    # normalize tickers for market router
    out=[]
    for s in syms[:cap]:
        t = s.replace(".", "-").upper().strip()
        if t and t not in out:
            out.append(t)
    # sector map needs normalized keys too
    sec2={}
    for k,v in sec.items():
        sec2[k.replace(".", "-").upper().strip()] = v
    return out, sec2

def _get_closes_many(ctx: CardContext, tickers: list[str], bars: int = 650) -> dict[str, pd.Series]:
    data = ctx.market.get_ohlcv_many(tickers, start=ctx.start, end=ctx.end, interval="1d")
    out={}
    for t in tickers:
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].astype(float).dropna().iloc[-bars:]
        if len(c) >= 260:
            out[t] = c
    return out

def _rets_df(closes: dict[str,pd.Series]) -> pd.DataFrame:
    frames=[]
    for k,c in closes.items():
        frames.append(c.pct_change().rename(k))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="any")

def _bool_df_from_series_map(m: dict[str,pd.Series]) -> pd.DataFrame:
    frames=[]
    for k,s in m.items():
        frames.append(s.rename(k))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="any")

def _pct_above_ma(closes: dict[str,pd.Series], ma_win: int) -> pd.Series:
    flags={}
    for k,c in closes.items():
        ma = c.rolling(ma_win, min_periods=ma_win).mean()
        flags[k] = (c > ma)
    df = _bool_df_from_series_map(flags)
    if df.empty:
        return pd.Series(dtype=float)
    return df.mean(axis=1) * 100.0

def _advance_decline(rets: pd.DataFrame) -> tuple[pd.Series,pd.Series,pd.Series,pd.Series]:
    adv = (rets > 0).sum(axis=1)
    dec = (rets < 0).sum(axis=1)
    net = adv - dec
    ad = net.cumsum()
    thrust = (adv / (adv + dec + 1e-12)).rolling(10).mean() * 100.0
    return adv, dec, net, ad, thrust

def _sector_participation_last(closes: dict[str,pd.Series], sector_map: dict[str,str], ma_win: int = 200) -> pd.DataFrame:
    rows=[]
    for k,c in closes.items():
        sec = sector_map.get(k, "Unknown")
        ma = c.rolling(ma_win, min_periods=ma_win).mean()
        if ma.dropna().empty:
            continue
        above = bool(c.iloc[-1] > ma.iloc[-1])
        rows.append((sec, k, above))
    if not rows:
        return pd.DataFrame(columns=["Sector","N","PctAboveMA200","TickersAbove"])
    df = pd.DataFrame(rows, columns=["Sector","Ticker","Above"])
    g = df.groupby("Sector").agg(
        N=("Ticker","count"),
        PctAboveMA200=("Above", lambda x: float(np.mean(x))*100.0),
        TickersAbove=("Above", lambda x: int(np.sum(x))),
    ).reset_index().sort_values("PctAboveMA200", ascending=False)
    g["PctAboveMA200"] = g["PctAboveMA200"].map(lambda x: f"{x:.1f}")
    return g

def _top_bottom_by_dist200(closes: dict[str,pd.Series], n: int = 10) -> tuple[pd.DataFrame,pd.DataFrame]:
    vals=[]
    for k,c in closes.items():
        ma200 = c.rolling(200, min_periods=200).mean()
        if ma200.dropna().empty:
            continue
        last = float(c.iloc[-1])
        m = float(ma200.iloc[-1])
        dist = (last - m) / (m + 1e-12) * 100.0
        vals.append((k, dist))
    if not vals:
        empty = pd.DataFrame([["(none)",""]], columns=["Ticker","DistToMA200%"])
        return empty, empty
    vals.sort(key=lambda x: x[1], reverse=True)
    top = pd.DataFrame([[k, f"{d:+.2f}"] for k,d in vals[:n]], columns=["Ticker","DistToMA200%"])
    bot = pd.DataFrame([[k, f"{d:+.2f}"] for k,d in vals[-n:]], columns=["Ticker","DistToMA200%"])
    bot = bot.iloc[::-1].reset_index(drop=True)
    return top, bot

# ---------------------------------------------------------
# 1) Big “screenshot page”
# ---------------------------------------------------------
@register_card("breadth.market_internals_dashboard", "Breadth: Market Internals Dashboard (S&P cap)", "breadth",
               min_tier="black", cost=12, heavy=True, slots=("S04","S11"))
def internals_dashboard(ctx: CardContext) -> CardResult:
    cap = min(getattr(ctx, "max_universe", 350) or 350, 350)
    uni, secmap = _load_sp500_universe(ctx, cap)
    closes = _get_closes_many(ctx, uni, bars=700)
    if len(closes) < 60:
        return CardResult(
            key="breadth.market_internals_dashboard",
            title="Breadth: Market Internals Dashboard (S&P cap)",
            summary="Not enough ticker data to compute breadth dashboard.",
            warnings=[f"ok={len(closes)} cap={cap}"]
        )

    rets = _rets_df(closes)
    if rets.empty or len(rets) < 120:
        return CardResult(
            key="breadth.market_internals_dashboard",
            title="Breadth: Market Internals Dashboard (S&P cap)",
            summary="Not enough overlapping returns for breadth series.",
            warnings=[f"rows={len(rets)}"]
        )

    adv, dec, net, ad, thrust = _advance_decline(rets)
    p20 = _pct_above_ma(closes, 20)
    p50 = _pct_above_ma(closes, 50)
    p200 = _pct_above_ma(closes, 200)

    sect = _sector_participation_last(closes, secmap, ma_win=200)
    top, bot = _top_bottom_by_dist200(closes, n=10)

    fig = plt.figure(figsize=(10,9.2))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    ax4 = fig.add_subplot(4,1,4)

    ax1.plot(ad.values, label="A/D line (cum net adv)")
    ax1.set_title("Market Internals — Advance/Decline + Breadth + Sector participation")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(thrust.values, label="Thrust (10D adv/(adv+dec)) %")
    ax2.axhline(50, linewidth=1)
    ax2.axhline(61.5, linewidth=1)
    ax2.axhline(40, linewidth=1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    ax3.plot(p20.reindex(rets.index).values, label="% > MA20")
    ax3.plot(p50.reindex(rets.index).values, label="% > MA50")
    ax3.plot(p200.reindex(rets.index).values, label="% > MA200")
    ax3.axhline(50, linewidth=1)
    ax3.set_title("Percent of universe above key moving averages")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", ncol=3, fontsize=8)

    # sector participation (bar)
    if not sect.empty:
        svals = sect.copy()
        svals["Pct"] = pd.to_numeric(svals["PctAboveMA200"], errors="coerce")
        svals = svals.dropna(subset=["Pct"]).head(11)
        ax4.bar(svals["Sector"].values, svals["Pct"].values)
        ax4.set_title("Sector participation: % above MA200 (top sectors)")
        ax4.grid(True, alpha=0.25, axis="y")
        ax4.set_xticklabels(svals["Sector"].values, rotation=25, ha="right")
    else:
        ax4.text(0.1,0.5,"No sector data", transform=ax4.transAxes)

    png = fig_to_png_bytes(fig)

    png_sect = _table_png("Sector participation (% above MA200)", sect.head(12) if not sect.empty else pd.DataFrame([["(none)",0,"",""]], columns=["Sector","N","PctAboveMA200","TickersAbove"]))
    png_top = _table_png("Top 10 vs MA200 (distance %)", top)
    png_bot = _table_png("Bottom 10 vs MA200 (distance %)", bot)

    metrics = {
        "UniverseOK": int(len(closes)),
        "ThrustNow%": round(float(thrust.dropna().iloc[-1]), 1) if thrust.dropna().shape[0] else None,
        "%AboveMA200Now": round(float(p200.dropna().iloc[-1]), 1) if p200.dropna().shape[0] else None,
        "NetAdvToday": int(net.iloc[-1]),
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "A/D line uses daily counts of advancers/decliners across the S&P cap universe.",
        "Thrust is 10D avg of adv/(adv+dec) — quick ‘breadth impulse’ read.",
        "% above MA tracks participation; sector bar shows where leadership breadth sits.",
    ]

    return CardResult(
        key="breadth.market_internals_dashboard",
        title="Breadth: Market Internals Dashboard (S&P cap)",
        summary="A/D line + breadth thrust + % above MA + sector participation (screenshot page).",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="market_internals_dashboard.png", payload=png),
            Artifact(kind="image/png", name="sector_participation_table.png", payload=png_sect),
            Artifact(kind="image/png", name="top_vs_ma200.png", payload=png_top),
            Artifact(kind="image/png", name="bottom_vs_ma200.png", payload=png_bot),
        ]
    )

# ---------------------------------------------------------
# 2) A/D proxy alone (lighter to include)
# ---------------------------------------------------------
@register_card("breadth.advance_decline_proxy", "Breadth: Advance/Decline Proxy (S&P cap)", "breadth",
               min_tier="pro", cost=9, heavy=True, slots=("S04","S11"))
def ad_proxy(ctx: CardContext) -> CardResult:
    cap = min(getattr(ctx, "max_universe", 350) or 350, 350)
    uni, _ = _load_sp500_universe(ctx, cap)
    closes = _get_closes_many(ctx, uni, bars=700)
    rets = _rets_df(closes)
    if rets.empty or len(rets) < 120:
        return CardResult(
            key="breadth.advance_decline_proxy",
            title="Breadth: Advance/Decline Proxy (S&P cap)",
            summary="Not enough overlap to compute A/D."
        )

    adv, dec, net, ad, thrust = _advance_decline(rets)

    fig = plt.figure(figsize=(10,5.8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.plot(ad.values, label="A/D line (cum net)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")
    ax1.set_title("Advance/Decline proxy (S&P cap)")

    ax2.plot(thrust.values, label="Thrust% (10D)")
    ax2.axhline(50, linewidth=1)
    ax2.axhline(61.5, linewidth=1)
    ax2.axhline(40, linewidth=1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "UniverseOK": int(rets.shape[1]),
        "NetAdvToday": int(net.iloc[-1]),
        "ThrustNow%": round(float(thrust.dropna().iloc[-1]), 1) if thrust.dropna().shape[0] else None,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="breadth.advance_decline_proxy",
        title="Breadth: Advance/Decline Proxy (S&P cap)",
        summary="A/D line + breadth thrust (adv/(adv+dec)).",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="advance_decline_proxy.png", payload=png)],
    )

# ---------------------------------------------------------
# 3) % Above MA stack alone (lighter to include)
# ---------------------------------------------------------
@register_card("breadth.percent_above_ma_stack", "Breadth: % Above MA (20/50/200) Stack (S&P cap)", "breadth",
               min_tier="pro", cost=9, heavy=True, slots=("S04","S11"))
def pct_above_ma(ctx: CardContext) -> CardResult:
    cap = min(getattr(ctx, "max_universe", 350) or 350, 350)
    uni, _ = _load_sp500_universe(ctx, cap)
    closes = _get_closes_many(ctx, uni, bars=700)
    if len(closes) < 60:
        return CardResult(
            key="breadth.percent_above_ma_stack",
            title="Breadth: % Above MA (20/50/200) Stack (S&P cap)",
            summary="Not enough ticker data to compute % above MA."
        )

    p20 = _pct_above_ma(closes, 20)
    p50 = _pct_above_ma(closes, 50)
    p200 = _pct_above_ma(closes, 200)
    if p200.dropna().shape[0] < 120:
        return CardResult(
            key="breadth.percent_above_ma_stack",
            title="Breadth: % Above MA (20/50/200) Stack (S&P cap)",
            summary="Not enough overlap to compute time series."
        )

    idx = p200.dropna().index
    fig = plt.figure(figsize=(10,5.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(p20.reindex(idx).values, label="% > MA20")
    ax.plot(p50.reindex(idx).values, label="% > MA50")
    ax.plot(p200.reindex(idx).values, label="% > MA200")
    ax.axhline(50, linewidth=1)
    ax.set_title("% of universe above MAs (breadth participation)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    png = fig_to_png_bytes(fig)

    metrics = {
        "%>MA20 now": round(float(p20.dropna().iloc[-1]), 1) if p20.dropna().shape[0] else None,
        "%>MA50 now": round(float(p50.dropna().iloc[-1]), 1) if p50.dropna().shape[0] else None,
        "%>MA200 now": round(float(p200.dropna().iloc[-1]), 1) if p200.dropna().shape[0] else None,
        "UniverseOK": int(len(closes)),
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    return CardResult(
        key="breadth.percent_above_ma_stack",
        title="Breadth: % Above MA (20/50/200) Stack (S&P cap)",
        summary="Breadth participation via % above MA20/50/200.",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="pct_above_ma_stack.png", payload=png)],
    )
