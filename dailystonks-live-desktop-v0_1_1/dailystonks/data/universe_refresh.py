from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "sp500_constituents.csv"

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCHANGE_NAMES = {
    "Q": "NASDAQ Global Select Market",
    "G": "NASDAQ Global Market",
    "S": "NASDAQ Capital Market",
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

CANONICAL_COLUMNS = [
    "Symbol",
    "Name",
    "Sector",
    "Industry",
    "Exchange",
    "ETF",
    "Source",
    "DateAdded",
    "CIK",
]


@dataclass(frozen=True)
class RefreshResult:
    out_path: Path
    row_count: int
    source: str
    backup_path: Optional[Path] = None


def project_root() -> Path:
    return PROJECT_ROOT


def _request_text(url: str, *, timeout: int = 30) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "DailyStonksLiveDesktop/0.1 (+https://local.invalid; personal-market-dashboard)",
            "Accept": "text/html,text/plain,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")
    except URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def _normalize_symbol(value: object) -> str:
    s = str(value or "").strip().upper()
    s = s.replace("\u00a0", "")
    s = s.replace(".", "-")  # Yahoo Finance convention: BRK.B -> BRK-B
    s = re.sub(r"\s+", "", s)
    return s


def _clean_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    s = str(value).strip()
    if s.lower() in {"nan", "none", "nat"}:
        return default
    return s


def _canonicalize(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    df = df.copy()
    if "Symbol" not in df.columns:
        raise ValueError("Universe data must contain a Symbol column after normalization.")

    df["Symbol"] = df["Symbol"].map(_normalize_symbol)
    df = df[df["Symbol"].astype(bool)].copy()
    df = df[~df["Symbol"].str.contains("FILE CREATION TIME", case=False, na=False)].copy()

    if "Name" not in df.columns:
        if "Security" in df.columns:
            df["Name"] = df["Security"]
        elif "Security Name" in df.columns:
            df["Name"] = df["Security Name"]
        else:
            df["Name"] = df["Symbol"]

    if "Sector" not in df.columns:
        if "GICS Sector" in df.columns:
            df["Sector"] = df["GICS Sector"]
        else:
            df["Sector"] = "Unknown"

    if "Industry" not in df.columns:
        if "GICS Sub-Industry" in df.columns:
            df["Industry"] = df["GICS Sub-Industry"]
        elif "Sub-Industry" in df.columns:
            df["Industry"] = df["Sub-Industry"]
        else:
            df["Industry"] = "Unknown"

    if "Exchange" not in df.columns:
        if "Market Category" in df.columns:
            df["Exchange"] = df["Market Category"].map(lambda x: EXCHANGE_NAMES.get(_clean_str(x), _clean_str(x, "NASDAQ")))
        elif "Listing Exchange" in df.columns:
            df["Exchange"] = df["Listing Exchange"].map(lambda x: EXCHANGE_NAMES.get(_clean_str(x), _clean_str(x, "Unknown")))
        elif "Exchange Code" in df.columns:
            df["Exchange"] = df["Exchange Code"].map(lambda x: EXCHANGE_NAMES.get(_clean_str(x), _clean_str(x, "Unknown")))
        else:
            df["Exchange"] = "Unknown"

    if "ETF" not in df.columns:
        df["ETF"] = "N"

    if "Source" not in df.columns:
        df["Source"] = source
    else:
        df["Source"] = df["Source"].fillna(source).replace("", source)

    if "DateAdded" not in df.columns:
        if "Date added" in df.columns:
            df["DateAdded"] = df["Date added"]
        else:
            df["DateAdded"] = ""

    if "CIK" not in df.columns:
        if "CIK" in df.columns:
            pass
        else:
            df["CIK"] = ""

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    out = df[CANONICAL_COLUMNS].copy()
    for col in CANONICAL_COLUMNS:
        out[col] = out[col].map(lambda x: _clean_str(x))

    # Keep broad Yahoo-compatible ordinary symbols by default. This intentionally
    # allows class shares normalized with a hyphen, e.g. BRK-B.
    out = out[out["Symbol"].str.match(r"^[A-Z0-9][A-Z0-9-]{0,14}$", na=False)].copy()
    out = out.drop_duplicates(subset=["Symbol"], keep="first")
    out = out.sort_values("Symbol", kind="stable").reset_index(drop=True)
    return out


def load_sp500_from_wikipedia() -> pd.DataFrame:
    try:
        tables = pd.read_html(WIKIPEDIA_SP500_URL)
    except ImportError as exc:
        raise RuntimeError(
            "pandas.read_html needs an HTML parser. Run: python -m pip install lxml"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read S&P 500 table from Wikipedia: {exc}") from exc

    candidates = []
    for table in tables:
        cols = {str(c).strip() for c in table.columns}
        if {"Symbol", "Security", "GICS Sector"}.issubset(cols):
            candidates.append(table)
    if not candidates:
        raise RuntimeError("Could not find the expected S&P 500 constituents table.")

    df = candidates[0].rename(columns={"Security": "Name", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry", "Date added": "DateAdded"})
    df["Exchange"] = "S&P 500"
    df["ETF"] = "N"
    df["Source"] = "wikipedia_sp500"
    return _canonicalize(df, source="wikipedia_sp500")


def _read_nasdaq_pipe_url(url: str) -> pd.DataFrame:
    text = _request_text(url)
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("File Creation Time:")]
    if not lines:
        raise RuntimeError(f"No rows returned from {url}")
    return pd.read_csv(StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def load_nasdaq_listed(*, include_etfs: bool = False) -> pd.DataFrame:
    df = _read_nasdaq_pipe_url(NASDAQ_LISTED_URL)
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].str.upper().eq("N")].copy()
    if "Financial Status" in df.columns:
        # N means normal. Empty can appear on some rows; keep normal rows only when present.
        df = df[df["Financial Status"].str.upper().isin(["N", ""])].copy()
    if not include_etfs and "ETF" in df.columns:
        df = df[~df["ETF"].str.upper().eq("Y")].copy()
    df = df.rename(columns={"Security Name": "Name"})
    df["Source"] = "nasdaqtrader_nasdaqlisted"
    return _canonicalize(df, source="nasdaqtrader_nasdaqlisted")


def load_other_listed(*, include_etfs: bool = False) -> pd.DataFrame:
    df = _read_nasdaq_pipe_url(OTHER_LISTED_URL)
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].str.upper().eq("N")].copy()
    if not include_etfs and "ETF" in df.columns:
        df = df[~df["ETF"].str.upper().eq("Y")].copy()
    df = df.rename(columns={"ACT Symbol": "Symbol", "Security Name": "Name", "Exchange": "Exchange Code"})
    df["Source"] = "nasdaqtrader_otherlisted"
    return _canonicalize(df, source="nasdaqtrader_otherlisted")


def load_us_listed(*, include_etfs: bool = False) -> pd.DataFrame:
    parts = [load_nasdaq_listed(include_etfs=include_etfs), load_other_listed(include_etfs=include_etfs)]
    df = pd.concat(parts, ignore_index=True)
    return _canonicalize(df, source="nasdaqtrader_us_listed")


def import_custom_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    df = pd.read_csv(input_path, dtype=str).fillna("")

    # Accept common alternatives without making users edit their old file first.
    aliases = {
        "Ticker": "Symbol",
        "ticker": "Symbol",
        "symbol": "Symbol",
        "Company": "Name",
        "company": "Name",
        "Security": "Name",
        "security": "Name",
        "Name": "Name",
        "GICS Sector": "Sector",
        "sector": "Sector",
        "GICS Sub-Industry": "Industry",
        "industry": "Industry",
        "Exchange Code": "Exchange Code",
        "exchange": "Exchange",
    }
    rename = {col: aliases[col] for col in df.columns if col in aliases}
    df = df.rename(columns=rename)
    df["Source"] = f"custom_import:{input_path.name}"
    return _canonicalize(df, source=f"custom_import:{input_path.name}")


def write_universe(df: pd.DataFrame, out_path: Path, *, backup: bool = True) -> RefreshResult:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Optional[Path] = None
    if backup and out_path.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = out_path.with_name(f"{out_path.stem}.backup-{stamp}{out_path.suffix}")
        shutil.copy2(out_path, backup_path)
    df.to_csv(out_path, index=False)
    source = str(df["Source"].iloc[0]) if "Source" in df.columns and len(df) else "unknown"
    return RefreshResult(out_path=out_path, row_count=len(df), source=source, backup_path=backup_path)


def build_universe(source: str, *, input_path: Optional[Path] = None, include_etfs: bool = False) -> pd.DataFrame:
    source = source.strip().lower().replace("_", "-")
    if source in {"sp500", "s&p500", "s-and-p-500", "sandp500"}:
        return load_sp500_from_wikipedia()
    if source in {"nasdaq", "nasdaq-listed", "nasdaqlisted"}:
        return load_nasdaq_listed(include_etfs=include_etfs)
    if source in {"us", "us-listed", "all-us", "all-us-listed"}:
        return load_us_listed(include_etfs=include_etfs)
    if source in {"custom", "import", "csv"}:
        if input_path is None:
            raise ValueError("--input is required when --source custom/import/csv")
        return import_custom_csv(input_path)
    raise ValueError(f"Unknown universe source: {source}")


def _print_preview(df: pd.DataFrame, limit: int = 12) -> None:
    print(f"Rows: {len(df)}")
    if len(df) == 0:
        return
    cols = [c for c in ["Symbol", "Name", "Sector", "Exchange", "ETF", "Source"] if c in df.columns]
    print(df[cols].head(limit).to_string(index=False))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dailystonks-refresh-universe",
        description="Refresh or import the local DailyStonks Live Desktop universe CSV.",
    )
    ap.add_argument(
        "--source",
        default="sp500",
        choices=["sp500", "nasdaq", "us-listed", "custom"],
        help="Universe source to build. Default: sp500.",
    )
    ap.add_argument("--input", type=Path, help="CSV to import when --source custom.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output CSV. Default: {DEFAULT_OUT}")
    ap.add_argument("--include-etfs", action="store_true", help="Keep ETF rows for NasdaqTrader sources.")
    ap.add_argument("--no-backup", action="store_true", help="Overwrite output without making a timestamped backup first.")
    ap.add_argument("--dry-run", action="store_true", help="Download/import and preview rows without writing the output CSV.")
    ap.add_argument("--preview", type=int, default=12, help="Number of preview rows to print.")
    args = ap.parse_args(argv)

    try:
        df = build_universe(args.source, input_path=args.input, include_etfs=args.include_etfs)
        _print_preview(df, limit=max(0, args.preview))
        if args.dry_run:
            print("Dry run only; no file written.")
            return 0
        result = write_universe(df, args.out, backup=not args.no_backup)
        print(f"Wrote: {result.out_path}")
        print(f"Source: {result.source}")
        print(f"Rows: {result.row_count}")
        if result.backup_path:
            print(f"Backup: {result.backup_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
