from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _table_png(title: str, df: pd.DataFrame, scale_y: float = 1.45) -> bytes:
    fig = plt.figure(figsize=(10, 0.55 + 0.33 * max(10, len(df))))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, scale_y)
    return fig_to_png_bytes(fig)

def _ann_vol_hv20(close: pd.Series) -> float:
    r = close.pct_change().dropna()
    if len(r) < 25:
        return float("nan")
    return float(r.rolling(20).std(ddof=1).iloc[-1] * math.sqrt(252))

def _spot_from_market(ctx: CardContext, t: str) -> float:
    try:
        df = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-5:]
        return float(df["Close"].astype(float).iloc[-1])
    except Exception:
        return float("nan")

def _yfticker(t: str):
    import yfinance as yf
    return yf.Ticker(t)

def _nearest_expiry(tk) -> str | None:
    try:
        opts = list(tk.options or [])
        return opts[0] if opts else None
    except Exception:
        return None

def _dte(exp: str, as_of) -> int | None:
    try:
        d = pd.to_datetime(exp).date()
        return int((d - as_of).days)
    except Exception:
        return None

@register_card("options.iv_snapshot_nearest", "Options IV Snapshot (nearest expiry)", "options", min_tier="black", cost=10, heavy=True, slots=("S11","S09"))
def iv_snapshot_nearest(ctx: CardContext) -> CardResult:
    try:
        import yfinance as yf  # noqa: F401
    except Exception as e:
        return CardResult(
            key="options.iv_snapshot_nearest",
            title="Options IV Snapshot (nearest expiry)",
            summary="yfinance required for options chains.",
            warnings=[repr(e)]
        )

    # keep small + useful
    tks = (ctx.tickers + ["SPY","QQQ","IWM","AAPL","MSFT","NVDA"])[:10]
    tks = list(dict.fromkeys([t.replace(".","-").lstrip("$").upper() for t in tks]))

    rows=[]
    fails=0
    for t in tks:
        try:
            tk = _yfticker(t)
            exp = _nearest_expiry(tk)
            spot = None

            # spot: try fast_info/info then fallback to market close
            try:
                fi = getattr(tk, "fast_info", None)
                if fi and fi.get("last_price"):
                    spot = float(fi.get("last_price"))
            except Exception:
                spot = None
            if spot is None:
                try:
                    info = tk.info or {}
                    if info.get("regularMarketPrice"):
                        spot = float(info["regularMarketPrice"])
                except Exception:
                    spot = None
            if spot is None:
                spot = _spot_from_market(ctx, t)

            hv20 = float("nan")
            try:
                dfp = ctx.market.get_ohlcv(t, start=ctx.start, end=ctx.end, interval="1d").iloc[-260:]
                hv20 = _ann_vol_hv20(dfp["Close"].astype(float))
            except Exception:
                pass

            if not exp:
                rows.append((t, "—", "", f"{spot:.2f}" if np.isfinite(spot) else "", "", f"{hv20*100:.1f}%" if np.isfinite(hv20) else "", "no chain (HV fallback)"))
                continue

            dte = _dte(exp, ctx.as_of)
            oc = tk.option_chain(exp)
            calls = oc.calls.copy() if oc and hasattr(oc, "calls") else pd.DataFrame()
            puts  = oc.puts.copy()  if oc and hasattr(oc, "puts")  else pd.DataFrame()

            if calls.empty or puts.empty or not np.isfinite(spot):
                rows.append((t, exp, dte if dte is not None else "", f"{spot:.2f}" if np.isfinite(spot) else "", "", f"{hv20*100:.1f}%" if np.isfinite(hv20) else "", "partial chain (HV fallback)"))
                continue

            calls["d"] = (calls["strike"].astype(float) - spot).abs()
            puts["d"]  = (puts["strike"].astype(float) - spot).abs()

            c_atm = calls.sort_values("d").head(1)
            p_atm = puts.sort_values("d").head(1)

            civ = float(c_atm["impliedVolatility"].iloc[0]) if "impliedVolatility" in c_atm.columns and pd.notna(c_atm["impliedVolatility"].iloc[0]) else float("nan")
            piv = float(p_atm["impliedVolatility"].iloc[0]) if "impliedVolatility" in p_atm.columns and pd.notna(p_atm["impliedVolatility"].iloc[0]) else float("nan")

            if np.isfinite(civ) and np.isfinite(piv):
                atm_iv = 0.5*(civ+piv)
                src = "chain ATM"
                iv_str = f"{atm_iv*100:.1f}%"
            elif np.isfinite(civ) or np.isfinite(piv):
                atm_iv = civ if np.isfinite(civ) else piv
                src = "chain 1-side"
                iv_str = f"{atm_iv*100:.1f}%"
            else:
                src = "HV fallback"
                iv_str = ""

            rows.append((t, exp, dte if dte is not None else "", f"{spot:.2f}", iv_str, f"{hv20*100:.1f}%" if np.isfinite(hv20) else "", src))
        except Exception:
            fails += 1
            continue

    if not rows:
        return CardResult(
            key="options.iv_snapshot_nearest",
            title="Options IV Snapshot (nearest expiry)",
            summary="No rows produced (options data failures).",
            warnings=[f"fails={fails}"]
        )

    df = pd.DataFrame(rows, columns=["Symbol","Expiry","DTE","Spot","ATM IV","HV20","Source"])
    png = _table_png("Options IV Snapshot (nearest expiry)", df)

    warnings=[]
    if fails:
        warnings.append(f"Failures: {fails} (yfinance chain/info calls can be flaky).")

    return CardResult(
        key="options.iv_snapshot_nearest",
        title="Options IV Snapshot (nearest expiry)",
        summary="ATM IV from nearest expiry chain (with HV fallback).",
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="iv_snapshot.png", payload=png)]
    )

@register_card("options.pin_levels_oi", "Options OI Pin Levels (nearest expiry)", "options", min_tier="black", cost=10, heavy=True, slots=("S11","S09"))
def pin_levels_oi(ctx: CardContext) -> CardResult:
    try:
        import yfinance as yf  # noqa: F401
    except Exception as e:
        return CardResult(
            key="options.pin_levels_oi",
            title="Options OI Pin Levels",
            summary="yfinance required for options chains.",
            warnings=[repr(e)]
        )

    t = (ctx.tickers[0] if ctx.tickers else "SPY").replace(".","-").lstrip("$").upper()
    tk = _yfticker(t)
    exp = _nearest_expiry(tk)
    if not exp:
        return CardResult(
            key="options.pin_levels_oi",
            title=f"{t}: Options OI Pin Levels",
            summary="No options expirations available via yfinance.",
            warnings=[f"symbol={t}"]
        )

    # spot
    spot = None
    try:
        fi = getattr(tk, "fast_info", None)
        if fi and fi.get("last_price"):
            spot = float(fi.get("last_price"))
    except Exception:
        spot = None
    if spot is None:
        spot = _spot_from_market(ctx, t)

    oc = tk.option_chain(exp)
    calls = oc.calls.copy()
    puts  = oc.puts.copy()
    if calls.empty or puts.empty:
        return CardResult(
            key="options.pin_levels_oi",
            title=f"{t}: Options OI Pin Levels",
            summary="Options chain returned empty calls/puts.",
            warnings=[f"exp={exp}"]
        )

    # Range around spot (±10%) to keep it readable
    if np.isfinite(spot):
        lo = 0.90 * spot
        hi = 1.10 * spot
        calls = calls[(calls["strike"]>=lo) & (calls["strike"]<=hi)].copy()
        puts  = puts[(puts["strike"]>=lo) & (puts["strike"]<=hi)].copy()

    # Fill OI
    calls["openInterest"] = pd.to_numeric(calls.get("openInterest", 0), errors="coerce").fillna(0.0)
    puts["openInterest"]  = pd.to_numeric(puts.get("openInterest", 0), errors="coerce").fillna(0.0)

    # Max walls
    cmax = calls.sort_values("openInterest", ascending=False).head(1)
    pmax = puts.sort_values("openInterest", ascending=False).head(1)

    c_strike = float(cmax["strike"].iloc[0]) if not cmax.empty else float("nan")
    p_strike = float(pmax["strike"].iloc[0]) if not pmax.empty else float("nan")
    c_oi = float(cmax["openInterest"].iloc[0]) if not cmax.empty else 0.0
    p_oi = float(pmax["openInterest"].iloc[0]) if not pmax.empty else 0.0

    # Plot OI bars
    fig = plt.figure(figsize=(10,5.2))
    ax = fig.add_subplot(1,1,1)
    ax.bar(calls["strike"].astype(float).values, calls["openInterest"].values, label="Call OI", alpha=0.6)
    ax.bar(puts["strike"].astype(float).values, puts["openInterest"].values, label="Put OI", alpha=0.6)
    if np.isfinite(spot):
        ax.axvline(spot, linewidth=1, label="Spot")
    if np.isfinite(c_strike):
        ax.axvline(c_strike, linewidth=1, label=f"Max Call OI @ {c_strike:g}")
    if np.isfinite(p_strike):
        ax.axvline(p_strike, linewidth=1, label=f"Max Put OI @ {p_strike:g}")
    ax.set_title(f"{t} OI by strike (nearest expiry {exp})")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    png = fig_to_png_bytes(fig)

    metrics = {
        "Expiry": exp,
        "DTE": _dte(exp, ctx.as_of) if _dte(exp, ctx.as_of) is not None else "",
        "Spot": round(float(spot), 2) if np.isfinite(spot) else "",
        "MaxCallOI strike": round(c_strike, 2) if np.isfinite(c_strike) else "",
        "MaxPutOI strike": round(p_strike, 2) if np.isfinite(p_strike) else "",
        "MaxCallOI": int(c_oi),
        "MaxPutOI": int(p_oi),
    }
    bullets = [
        "Heuristic 'pin/wall' levels from max open interest near spot.",
        "Use for context only; OI is not positioning certainty.",
    ]

    return CardResult(
        key="options.pin_levels_oi",
        title=f"{t}: Options OI Pin Levels",
        summary="Max call/put OI strikes near spot (nearest expiry).",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="oi_pin_levels.png", payload=png)]
    )

@register_card("options.skew_proxy_smile", "IV Smile + Skew Proxy (nearest expiry)", "options", min_tier="black", cost=10, heavy=True, slots=("S11","S09"))
def skew_proxy_smile(ctx: CardContext) -> CardResult:
    try:
        import yfinance as yf  # noqa: F401
    except Exception as e:
        return CardResult(
            key="options.skew_proxy_smile",
            title="IV Smile + Skew Proxy",
            summary="yfinance required for options chains.",
            warnings=[repr(e)]
        )

    t = (ctx.tickers[0] if ctx.tickers else "SPY").replace(".","-").lstrip("$").upper()
    tk = _yfticker(t)
    exp = _nearest_expiry(tk)
    if not exp:
        return CardResult(
            key="options.skew_proxy_smile",
            title=f"{t}: IV Smile + Skew Proxy",
            summary="No options expirations available via yfinance.",
        )

    # spot
    spot = None
    try:
        fi = getattr(tk, "fast_info", None)
        if fi and fi.get("last_price"):
            spot = float(fi.get("last_price"))
    except Exception:
        spot = None
    if spot is None:
        spot = _spot_from_market(ctx, t)

    oc = tk.option_chain(exp)
    calls = oc.calls.copy()
    puts  = oc.puts.copy()
    if calls.empty or puts.empty or not np.isfinite(spot):
        return CardResult(
            key="options.skew_proxy_smile",
            title=f"{t}: IV Smile + Skew Proxy",
            summary="Missing chain or spot price.",
            warnings=[f"exp={exp}"]
        )

    # Keep strikes around spot ±15%
    lo = 0.85 * spot
    hi = 1.15 * spot
    calls = calls[(calls["strike"]>=lo) & (calls["strike"]<=hi)].copy()
    puts  = puts[(puts["strike"]>=lo) & (puts["strike"]<=hi)].copy()

    for df in (calls, puts):
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", np.nan), errors="coerce")

    calls = calls.dropna(subset=["strike","impliedVolatility"])
    puts  = puts.dropna(subset=["strike","impliedVolatility"])

    if calls.empty or puts.empty:
        return CardResult(
            key="options.skew_proxy_smile",
            title=f"{t}: IV Smile + Skew Proxy",
            summary="No IV points available in the filtered strike window.",
        )

    # Skew proxy: put IV near 95% strike minus call IV near 105% strike
    put_target = 0.95 * spot
    call_target = 1.05 * spot
    p = puts.assign(d=(puts["strike"]-put_target).abs()).sort_values("d").head(1)
    c = calls.assign(d=(calls["strike"]-call_target).abs()).sort_values("d").head(1)

    p_iv = float(p["impliedVolatility"].iloc[0]) if not p.empty else float("nan")
    c_iv = float(c["impliedVolatility"].iloc[0]) if not c.empty else float("nan")
    skew = (p_iv - c_iv) if np.isfinite(p_iv) and np.isfinite(c_iv) else float("nan")

    # Plot smile
    fig = plt.figure(figsize=(10,5.2))
    ax = fig.add_subplot(1,1,1)
    ax.plot(calls["strike"].values, calls["impliedVolatility"].values*100, label="Calls IV%")
    ax.plot(puts["strike"].values, puts["impliedVolatility"].values*100, label="Puts IV%")
    ax.axvline(spot, linewidth=1, label="Spot")
    ax.set_title(f"{t} IV Smile (nearest expiry {exp})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    png = fig_to_png_bytes(fig)

    bullets = [
        "Skew proxy = IV(put ~95% strike) − IV(call ~105% strike).",
        "Positive skew often indicates higher demand for downside protection.",
    ]

    metrics = {
        "Expiry": exp,
        "DTE": _dte(exp, ctx.as_of) if _dte(exp, ctx.as_of) is not None else "",
        "Spot": round(float(spot), 2),
        "Put95 IV%": round(p_iv*100, 2) if np.isfinite(p_iv) else "",
        "Call105 IV%": round(c_iv*100, 2) if np.isfinite(c_iv) else "",
        "Skew%": round(skew*100, 2) if np.isfinite(skew) else "",
    }

    return CardResult(
        key="options.skew_proxy_smile",
        title=f"{t}: IV Smile + Skew Proxy",
        summary="Nearest-expiry smile + simple skew proxy.",
        metrics=metrics,
        bullets=bullets,
        artifacts=[Artifact(kind="image/png", name="iv_smile.png", payload=png)]
    )