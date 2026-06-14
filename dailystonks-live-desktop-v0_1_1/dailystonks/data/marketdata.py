from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd
import numpy as np
import time

def yahoo_symbol(sym: str) -> str:
    s = str(sym).upper().strip()
    s = s.lstrip("$")
    s = s.replace(".", "-")  # BRK.B -> BRK-B
    return s

def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

@dataclass
class MarketData:
    cache_dir: str
    offline_synth: bool = False
    cache_ttl_seconds: Optional[float] = None

    def _cache_is_fresh(self, path: Path) -> bool:
        """Return True when a cache file may be reused.

        The historical mailout/report path keeps the original behaviour by
        leaving cache_ttl_seconds as None, meaning cached data is reused until
        manually cleared. Live terminal views pass a finite TTL so repeated
        refreshes do not get trapped behind stale TODAY cache files.
        """
        if self.cache_ttl_seconds is None:
            return True
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        return age <= max(0.0, float(self.cache_ttl_seconds))

    def _cache_base(self, ticker: str, interval: str, start: str, end: Optional[str]) -> Path:
        safe_end = end or "TODAY"
        p = Path(self.cache_dir) / "ohlcv" / interval
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{ticker}_{start}_{safe_end}"

    def _read_cache(self, base: Path) -> Optional[pd.DataFrame]:
        # Prefer parquet if available, otherwise pickle
        parquet = base.with_suffix(".parquet")
        pkl = base.with_suffix(".pkl")

        if parquet.exists() and _parquet_engine_available() and self._cache_is_fresh(parquet):
            try:
                return pd.read_parquet(parquet)
            except Exception:
                pass

        if pkl.exists() and self._cache_is_fresh(pkl):
            try:
                return pd.read_pickle(pkl)
            except Exception:
                pass

        # if parquet exists but engine missing, ignore + fall back to fresh download
        return None

    def _write_cache(self, base: Path, df: pd.DataFrame) -> None:
        # Write parquet if possible, else pickle
        if _parquet_engine_available():
            try:
                df.to_parquet(base.with_suffix(".parquet"))
                return
            except Exception:
                pass
        df.to_pickle(base.with_suffix(".pkl"))

    def _synth_ohlcv(self, ticker: str, *, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
        if interval != "1d":
            interval = "1d"
        idx = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="B")
        n = len(idx)
        if n < 60:
            idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=240, freq="B")
            n = len(idx)
        seed = abs(hash(ticker)) % (2**32)
        rng = np.random.default_rng(seed)
        rets = rng.normal(loc=0.0004, scale=0.012, size=n)
        price = 100.0 * np.cumprod(1.0 + rets)
        close = pd.Series(price, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + rng.normal(0, 0.002, size=n))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=n))
        low  = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=n))
        vol  = rng.integers(1_000_000, 20_000_000, size=n)
        return pd.DataFrame({"Open": open_.values, "High": high.values, "Low": low.values, "Close": close.values, "Volume": vol}, index=idx)

    def get_ohlcv(self, ticker: str, *, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
        t = yahoo_symbol(ticker)

        if self.offline_synth:
            return self._synth_ohlcv(t, start=start, end=end, interval=interval)

        base = self._cache_base(t, interval, start, end)
        cached = self._read_cache(base)
        if cached is not None and not cached.empty:
            return cached

        # Use batch path for single ticker (still benefits from retry logic)
        out = self.get_ohlcv_many([t], start=start, end=end, interval=interval)
        if t not in out or out[t].empty:
            raise RuntimeError(f"No data returned for {ticker} (mapped {t}).")
        return out[t]

    def get_ohlcv_many(self, tickers: Iterable[str], *, start: str, end: Optional[str], interval: str) -> Dict[str, pd.DataFrame]:
        # Returns dict of mapped ticker -> OHLCV df (may omit failures)
        mapped = [yahoo_symbol(x) for x in tickers]
        mapped = list(dict.fromkeys(mapped))  # unique preserve order
        out: Dict[str, pd.DataFrame] = {}

        if self.offline_synth:
            for t in mapped:
                out[t] = self._synth_ohlcv(t, start=start, end=end, interval=interval)
            return out

        # Serve from cache if present
        need = []
        for t in mapped:
            base = self._cache_base(t, interval, start, end)
            cached = self._read_cache(base)
            if cached is not None and not cached.empty:
                out[t] = cached
            else:
                need.append(t)

        if not need:
            return out

        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError("yfinance required for live downloads.") from e

        last_err = None
        for attempt in range(1, 4):
            try:
                df = yf.download(
                    tickers=need,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
                if df is None or df.empty:
                    raise RuntimeError("Empty batch download result.")

                # Normalize extraction regardless of yfinance column order
                for t in need:
                    sub = None
                    if isinstance(df.columns, pd.MultiIndex):
                        # group_by='ticker' => (ticker, field)
                        if t in df.columns.get_level_values(0):
                            sub = df[t]
                        # else maybe (field, ticker)
                        elif t in df.columns.get_level_values(1):
                            sub = df.xs(t, axis=1, level=1)
                    else:
                        # Single ticker fallback
                        sub = df

                    if sub is None or sub.empty:
                        continue

                    sub = sub.rename(columns={c: c.title() for c in sub.columns})
                    # Some series may omit Volume (crypto sometimes); handle gracefully
                    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
                    sub = sub[cols].copy()
                    sub.index = pd.to_datetime(sub.index)

                    out[t] = sub

                    base = self._cache_base(t, interval, start, end)
                    self._write_cache(base, sub)

                return out
            except Exception as e:
                last_err = e
                time.sleep(0.6 * attempt)

        # Return whatever we got from cache; caller will degrade gracefully
        return out