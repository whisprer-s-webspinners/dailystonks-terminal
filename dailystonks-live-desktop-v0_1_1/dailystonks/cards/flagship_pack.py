
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

TRADING_DAYS = 252

def _zscore(s: pd.Series, win: int = 252) -> pd.Series:
    m = s.rolling(win, min_periods=max(60, win//3)).mean()
    sd = s.rolling(win, min_periods=max(60, win//3)).std(ddof=1)
    return (s - m) / (sd + 1e-12)

def _table_png(title: str, df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.4)
    return fig_to_png_bytes(fig)

def _get_closes(ctx: CardContext, tickers: list[str], bars: int = 1600) -> pd.DataFrame:
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
    px = pd.concat(frames, axis=1).dropna(how="any")
    return px

# -------- 1) Risk-On/Risk-Off composite --------
@register_card("macro.risk_on_off_composite", "Macro: Risk-On / Risk-Off Composite", "macro", min_tier="black", cost=8, heavy=False, slots=("S03",))
def risk_on_off(ctx: CardContext) -> CardResult:
    # Classic desk ratios
    # (positive => risk-on): IWM/SPY, XLY/XLP, HYG/LQD, BTC/SPY
    # (negative => risk-off): TLT/SPY, GLD/SPY, UUP (USD strength often risk-off proxy)
    tickers = ["SPY","IWM","XLY","XLP","HYG","LQD","TLT","GLD","UUP","BTC-USD"]
    px = _get_closes(ctx, tickers, bars=1800)

    # ensure the minimum backbone exists
    must = ["SPY","IWM","XLY","XLP","HYG","LQD","TLT","GLD","UUP"]
    have = set(px.columns)
    missing = [m for m in must if m not in have]
    if missing:
        return CardResult(
            key="macro.risk_on_off_composite",
            title="Macro: Risk-On / Risk-Off Composite",
            summary="Missing required ETF series to compute composite.",
            warnings=[f"missing: {', '.join(missing)}"]
        )

    # Ratios
    spy = px["SPY"]
    ratios = {}
    ratios["Smallcaps IWM/SPY"] = (px["IWM"] / (spy + 1e-12))
    ratios["Cyc/Def XLY/XLP"]   = (px["XLY"] / (px["XLP"] + 1e-12))
    ratios["Credit HYG/LQD"]    = (px["HYG"] / (px["LQD"] + 1e-12))
    ratios["Duration TLT/SPY"]  = (px["TLT"] / (spy + 1e-12))
    ratios["Gold GLD/SPY"]      = (px["GLD"] / (spy + 1e-12))
    ratios["USD UUP"]           = (px["UUP"])
    if "BTC-USD" in px.columns:
        ratios["BTC/SPY"] = (px["BTC-USD"] / (spy + 1e-12))

    # Z-scores of log-ratios
    z = {}
    for name, s in ratios.items():
        s = np.log(s.replace(0, np.nan)).dropna()
        z[name] = _zscore(s, 252)

    # Align
    Z = pd.concat(z, axis=1).dropna(how="any")

    # Signs: risk-on positive, risk-off negative
    sign = pd.Series(1.0, index=Z.columns)
    for k in Z.columns:
        if k.startswith("Duration") or k.startswith("Gold") or k.startswith("USD"):
            sign[k] = -1.0

    comp = (Z * sign).mean(axis=1)

    last = float(comp.iloc[-1])
    if last >= 0.75:
        reg = "RISK-ON"
    elif last <= -0.75:
        reg = "RISK-OFF"
    else:
        reg = "MIXED"

    fig = plt.figure(figsize=(10,6.6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot((comp).values, label="Composite (z)")
    ax1.axhline(0, linewidth=1)
    ax1.axhline(0.75, linewidth=1)
    ax1.axhline(-0.75, linewidth=1)
    ax1.set_title(f"Risk-On/Risk-Off Composite — {reg} (now {last:+.2f})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    # show top 4 contributors by absolute z today
    last_z = (Z.iloc[-1] * sign).sort_values(key=lambda s: s.abs(), ascending=False)
    top = last_z.head(4).index.tolist()
    for name in top:
        ax2.plot((Z[name]*sign[name]).values, label=name)
    ax2.axhline(0, linewidth=1)
    ax2.set_title("Top components (signed z-scores)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    png = fig_to_png_bytes(fig)

    metrics = {"CompositeZ": round(last, 3), "Regime": reg}
    for name in top:
        metrics[f"{name} z"] = round(float((Z[name]*sign[name]).iloc[-1]), 2)

    bullets = [
        "Composite = mean(signed z-scores) of classic risk-on/off ratios (log-ratio z(252)).",
        "Positive = risk-on tilt; negative = risk-off tilt (heuristic but very desk-like).",
    ]

    return CardResult(
        key="macro.risk_on_off_composite",
        title="Macro: Risk-On / Risk-Off Composite",
        summary="Cross-asset risk appetite dashboard from classic ratios.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="risk_on_off_composite.png", payload=png)]
    )

# -------- 2) VIX term structure proxy --------
@register_card("risk.vix_term_proxy", "Risk: VIX Term Structure Proxy (VIX vs VIX3M)", "risk", min_tier="black", cost=7, heavy=False, slots=("S09",))
def vix_term(ctx: CardContext) -> CardResult:
    # Use yfinance directly for ^VIX / ^VXV (3M proxy) because some OHLCV routers sanitize tickers.
    try:
        import yfinance as yf
    except Exception as e:
        return CardResult(
            key="risk.vix_term_proxy",
            title="Risk: VIX Term Structure Proxy",
            summary="yfinance required for VIX index series.",
            warnings=[repr(e)]
        )

    def fetch(sym: str) -> pd.Series | None:
        try:
            h = yf.download(sym, start=str(ctx.start), progress=False, auto_adjust=False)
            if h is None or h.empty:
                return None
            c = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
            c = c.astype(float).dropna()
            return c.iloc[-1600:]
        except Exception:
            return None

    vix = fetch("^VIX")
    vxv = fetch("^VXV")  # 3-month VIX proxy
    if vix is None or vxv is None or len(vix) < 200 or len(vxv) < 200:
        return CardResult(
            key="risk.vix_term_proxy",
            title="Risk: VIX Term Structure Proxy (VIX vs VIX3M)",
            summary="Could not fetch ^VIX and ^VXV reliably.",
            warnings=["Try later; Yahoo index feeds can be flaky."]
        )

    # Align
    px = pd.concat([vix.rename("VIX"), vxv.rename("VXV")], axis=1).dropna(how="any")
    ratio = px["VIX"] / (px["VXV"] + 1e-12)
    rz = _zscore(ratio, 252)

    last_ratio = float(ratio.iloc[-1])
    last_z = float(rz.dropna().iloc[-1]) if rz.dropna().shape[0] else float("nan")

    # Term structure regime (very common desk read)
    if last_ratio > 1.0:
        ts = "Backwardation (stress)"
    else:
        ts = "Contango (calm)"

    fig = plt.figure(figsize=(10,6.2))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)

    ax1.plot(px["VIX"].values, label="VIX")
    ax1.plot(px["VXV"].values, label="VIX3M (VXV)")
    ax1.set_title("VIX vs VIX3M (proxy)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(ratio.values, label="VIX/VXV")
    ax2.axhline(1.0, linewidth=1)
    ax2.set_title(f"Term structure: {ts} — VIX/VXV now {last_ratio:.3f} (z {last_z:+.2f})")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")

    png = fig_to_png_bytes(fig)

    metrics = {
        "VIX": round(float(px["VIX"].iloc[-1]), 2),
        "VXV": round(float(px["VXV"].iloc[-1]), 2),
        "VIX/VXV": round(last_ratio, 3),
        "Z252": round(last_z, 2) if np.isfinite(last_z) else None,
        "Regime": ts,
    }
    metrics = {k:v for k,v in metrics.items() if v is not None}

    bullets = [
        "VIX/VXV > 1 often aligns with stress/backwardation; < 1 with calm/contango.",
        "This is a proxy for term structure, not a full VIX futures curve.",
    ]

    return CardResult(
        key="risk.vix_term_proxy",
        title="Risk: VIX Term Structure Proxy (VIX vs VIX3M)",
        summary="Desk-style volatility term structure read using VIX and VIX3M proxy.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="vix_term_proxy.png", payload=png)]
    )

# -------- 3) 52-week highs/lows breadth --------
@register_card("breadth.new_highs_lows_52w", "Breadth: 52-Week New Highs/Lows (S&P cap)", "breadth", min_tier="black", cost=10, heavy=True, slots=("S04",))
def highs_lows_52w(ctx: CardContext) -> CardResult:
    spdf = ctx.sp500.df()
    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 350)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    highs=[]
    lows=[]
    near_high=[]
    near_low=[]
    fails=0
    ok=0

    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1
            continue
        c = df["Close"].astype(float).dropna()
        if len(c) < 260:
            fails += 1
            continue
        c = c.iloc[-520:]
        last = float(c.iloc[-1])
        hi = float(c.rolling(252).max().iloc[-1])
        lo = float(c.rolling(252).min().iloc[-1])

        if not (np.isfinite(last) and np.isfinite(hi) and np.isfinite(lo)):
            fails += 1
            continue

        ok += 1
        dist_hi = (last - hi) / (hi + 1e-12)
        dist_lo = (last - lo) / (lo + 1e-12)

        # exact new highs/lows
        if dist_hi >= -1e-6:
            highs.append((tk, dist_hi))
        if dist_lo <= 1e-6:
            lows.append((tk, dist_lo))

        # “near” buckets
        if dist_hi >= -0.02:
            near_high.append((tk, dist_hi))
        if dist_lo <= 0.02:
            near_low.append((tk, dist_lo))

    n_high = len(highs)
    n_low = len(lows)
    n_nh = len(near_high)
    n_nl = len(near_low)

    fig = plt.figure(figsize=(10,4.8))
    ax = fig.add_subplot(1,1,1)
    cats = ["NewHighs", "NearHigh(<=2%)", "NewLows", "NearLow(<=2%)", "Net(H-L)"]
    vals = [n_high, n_nh, n_low, n_nl, n_high - n_low]
    ax.bar(cats, vals)
    ax.set_title(f"52-Week High/Low Breadth (cap={len(uni)}, ok={ok}, fails={fails})")
    ax.grid(True, alpha=0.25, axis="y")
    png_bar = fig_to_png_bytes(fig)

    # Tables (top by proximity)
    highs_sorted = sorted(highs, key=lambda x: x[1], reverse=True)[:12]
    lows_sorted = sorted(lows, key=lambda x: x[1])[:12]

    def tbl(rows, title, kind):
        if not rows:
            df = pd.DataFrame([["(none)",""]], columns=["Symbol", kind])
        else:
            df = pd.DataFrame([[s, f"{d*100:+.2f}%"] for s,d in rows], columns=["Symbol", kind])
        return _table_png(title, df)

    png_hi = tbl(highs_sorted, "New 52-Week Highs (top)", "DistHi")
    png_lo = tbl(lows_sorted, "New 52-Week Lows (top)", "DistLo")

    metrics = {
        "NewHighs": n_high,
        "NewLows": n_low,
        "NearHigh<=2%": n_nh,
        "NearLow<=2%": n_nl,
        "Net(H-L)": n_high - n_low,
        "Ok": ok,
        "Fails": fails,
    }

    bullets = [
        "New highs/lows computed on last close vs rolling 252D high/low.",
        "Near buckets show participation close to extremes (useful for ‘melt-up’ or ‘capitulation’ reads).",
    ]

    return CardResult(
        key="breadth.new_highs_lows_52w",
        title="Breadth: 52-Week New Highs/Lows (S&P cap)",
        summary="New highs/lows + near-extremes participation.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[
            Artifact(kind="image/png", name="breadth_52w_bar.png", payload=png_bar),
            Artifact(kind="image/png", name="breadth_52w_highs.png", payload=png_hi),
            Artifact(kind="image/png", name="breadth_52w_lows.png", payload=png_lo),
        ]
    )
