from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.registry import register_card
from ..core.models import CardContext, CardResult, Artifact
from ..core.utils import fig_to_png_bytes

def _ann_vol(ret: pd.Series, win: int = 20) -> float:
    r = ret.dropna()
    if len(r) < win + 5:
        return float("nan")
    return float(r.rolling(win).std(ddof=1).iloc[-1] * math.sqrt(252))

@register_card("sector.heatmap_1d", "Sector Performance Heatmap (1D)", "sector", min_tier="pro", cost=9, heavy=True, slots=("S05",))
def sector_heatmap(ctx: CardContext) -> CardResult:
    # Use S&P CSV for sector mapping
    spdf = ctx.sp500.df()
    if "Sector" not in spdf.columns:
        return CardResult(
            key="sector.heatmap_1d",
            title="Sector Performance Heatmap (1D)",
            summary="S&P constituents CSV missing 'Sector' column."
        )

    uni = spdf["Symbol"].tolist()[: min(ctx.max_universe, 160)]
    data = ctx.market.get_ohlcv_many(uni, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    fails=0
    for raw in uni:
        tk = raw.replace(".", "-")
        df = data.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            fails += 1
            continue
        c = df["Close"].astype(float).dropna()
        if len(c) < 2:
            fails += 1
            continue
        r1 = float(c.pct_change().iloc[-1])
        sector = str(spdf.loc[spdf["Symbol"]==raw, "Sector"].iloc[0])
        rows.append((sector, tk, r1))

    if len(rows) < 20:
        return CardResult(
            key="sector.heatmap_1d",
            title="Sector Performance Heatmap (1D)",
            summary="Not enough symbols with data to compute heatmap.",
            warnings=[f"ok={len(rows)} fails={fails}"]
        )

    dfv = pd.DataFrame(rows, columns=["Sector","Symbol","R1"])
    # sector mean return
    sec = dfv.groupby("Sector")["R1"].mean().sort_values(ascending=False)
    order = sec.index.tolist()

    # build heat matrix: each sector column, fill with top movers by abs return
    cols = min(11, len(order))
    order = order[:cols]
    mat_cols = []
    labels_cols = []
    max_rows = 10

    for s in order:
        sub = dfv[dfv["Sector"]==s].copy()
        sub["abs"] = sub["R1"].abs()
        sub = sub.sort_values("abs", ascending=False).head(max_rows)
        vec = sub["R1"].values
        lab = [f"{sym}\n{r*100:+.2f}%" for sym,r in zip(sub["Symbol"].values, sub["R1"].values)]
        # pad
        if len(vec) < max_rows:
            vec = np.pad(vec, (0, max_rows-len(vec)), constant_values=np.nan)
            lab += [""]*(max_rows-len(lab))
        mat_cols.append(vec)
        labels_cols.append(lab)

    mat = np.column_stack(mat_cols)  # shape (rows, sectors)

    fig = plt.figure(figsize=(10,6.2))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(mat, aspect="auto", origin="upper")
    ax.set_title("Sector Heatmap (1D): top movers by |return| within sector")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([])

    for c in range(mat.shape[1]):
        for r in range(mat.shape[0]):
            txt = labels_cols[c][r]
            if txt:
                ax.text(c, r, txt, ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = fig_to_png_bytes(fig)

    metrics = {f"{s} mean%": round(float(sec.loc[s]*100), 2) for s in order if s in sec.index}
    warnings=[]
    if fails: warnings.append(f"Skipped {fails} symbols due to missing data.")

    return CardResult(
        key="sector.heatmap_1d",
        title="Sector Performance Heatmap (1D)",
        summary="Sector means + top movers per sector (capped universe, batched fetch).",
        metrics=metrics,
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="sector_heatmap_1d.png", payload=png)]
    )

@register_card("income.dividends_snapshot", "Dividends Snapshot (top names)", "income", min_tier="black", cost=9, heavy=True, slots=("S12",))
def dividends_snapshot(ctx: CardContext) -> CardResult:
    # Use a short list of mega-cap-ish tickers (or user tickers)
    base = (ctx.tickers + ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","JPM","XOM","JNJ","PG","KO","PEP"])[:20]
    # Normalize
    tks = list(dict.fromkeys([t.replace(".", "-") for t in base]))

    # yfinance info calls are slower; keep count small and tolerate failures
    try:
        import yfinance as yf
    except Exception as e:
        return CardResult(
            key="income.dividends_snapshot",
            title="Dividends Snapshot",
            summary="yfinance required for dividend metadata.",
            warnings=[repr(e)]
        )

    rows=[]
    fails=0
    for t in tks:
        try:
            info = yf.Ticker(t).info or {}
            yld = info.get("dividendYield", None)  # fraction
            rate = info.get("dividendRate", None)
            price = info.get("regularMarketPrice", None)
            name = info.get("shortName", "") or info.get("longName","")
            if yld is None and rate is not None and price:
                yld = float(rate)/float(price)
            if yld is None:
                continue
            rows.append((t, name[:28], float(yld)*100.0, float(rate) if rate is not None else None))
        except Exception:
            fails += 1

    if not rows:
        return CardResult(
            key="income.dividends_snapshot",
            title="Dividends Snapshot",
            summary="No dividend data available for the selected tickers.",
            warnings=[f"fails={fails}"]
        )

    df = pd.DataFrame(rows, columns=["Symbol","Name","Yield%","Rate"])
    df = df.sort_values("Yield%", ascending=False).head(15)

    fig = plt.figure(figsize=(10,5.0))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title("Dividend Yield Snapshot (top 15)")
    show = df.copy()
    show["Yield%"] = show["Yield%"].map(lambda x: f"{x:.2f}%")
    show["Rate"] = show["Rate"].map(lambda x: "" if x is None else f"{x:.2f}")
    table = ax.table(cellText=show[["Symbol","Name","Yield%","Rate"]].values,
                     colLabels=["Symbol","Name","Yield%","Rate"], loc="center")
    table.scale(1, 1.5)
    png = fig_to_png_bytes(fig)

    metrics = {"Rows": int(len(df)), "Failures": int(fails)}
    return CardResult(
        key="income.dividends_snapshot",
        title="Dividends Snapshot",
        summary="Dividend yields from yfinance metadata (best-effort).",
        metrics=metrics,
        artifacts=[Artifact(kind="image/png", name="dividends.png", payload=png)]
    )

@register_card("multi.asset_summary", "Multi-Asset Summary (returns/vol/corr)", "multi", min_tier="pro", cost=8, heavy=False, slots=("S03","S11"))
def multi_asset(ctx: CardContext) -> CardResult:
    basket = ["SPY","QQQ","IWM","TLT","GLD","BTC-USD"]
    data = ctx.market.get_ohlcv_many(basket, start=ctx.start, end=ctx.end, interval="1d")

    rows=[]
    missing=[]
    for raw in basket:
        t = raw.replace(".","-")
        df = data.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            missing.append(raw); continue
        c = df["Close"].astype(float).dropna()
        if len(c) < 25:
            missing.append(raw); continue
        r1 = float(c.pct_change().iloc[-1])
        r5 = float(c.pct_change(5).iloc[-1]) if len(c) >= 7 else float("nan")
        r20 = float(c.pct_change(20).iloc[-1]) if len(c) >= 25 else float("nan")
        v20 = _ann_vol(c.pct_change(), 20)
        rows.append((t, r1, r5, r20, v20))

    if len(rows) < 2:
        return CardResult(
            key="multi.asset_summary",
            title="Multi-Asset Summary",
            summary="Not enough series to build summary table.",
            warnings=[f"missing: {', '.join(missing)}"]
        )

    tab = pd.DataFrame(rows, columns=["Symbol","R1","R5","R20","Vol20"])
    # corr vs SPY
    spy = data.get("SPY")
    if spy is not None and not spy.empty:
        spy_ret = spy["Close"].pct_change().dropna()
        corrs=[]
        for sym in tab["Symbol"].tolist():
            df = data.get(sym)
            if df is None or df.empty: 
                corrs.append(np.nan); continue
            r = df["Close"].pct_change().dropna()
            m = pd.concat([spy_ret, r], axis=1).dropna()
            if len(m) < 60:
                corrs.append(np.nan)
            else:
                corrs.append(float(m.iloc[-60:,0].corr(m.iloc[-60:,1])))
        tab["Corr60vsSPY"] = corrs

    fig = plt.figure(figsize=(10,5.1))
    ax = fig.add_subplot(1,1,1)
    ax.axis("off")
    ax.set_title("Multi-Asset Summary")
    show = tab.copy()
    for c in ["R1","R5","R20"]:
        show[c] = show[c].map(lambda x: "" if not np.isfinite(x) else f"{x*100:+.2f}%")
    show["Vol20"] = show["Vol20"].map(lambda x: "" if not np.isfinite(x) else f"{x*100:.1f}%")
    if "Corr60vsSPY" in show.columns:
        show["Corr60vsSPY"] = show["Corr60vsSPY"].map(lambda x: "" if not np.isfinite(x) else f"{x:.2f}")
        cols = ["Symbol","R1","R5","R20","Vol20","Corr60vsSPY"]
    else:
        cols = ["Symbol","R1","R5","R20","Vol20"]
    table = ax.table(cellText=show[cols].values, colLabels=cols, loc="center")
    table.scale(1, 1.5)
    png = fig_to_png_bytes(fig)

    warnings=[]
    if missing: warnings.append(f"Missing/short series skipped: {', '.join(missing)}")

    return CardResult(
        key="multi.asset_summary",
        title="Multi-Asset Summary",
        summary="Returns + vol + corr vs SPY (where possible).",
        warnings=warnings,
        artifacts=[Artifact(kind="image/png", name="multi_asset.png", payload=png)]
    )