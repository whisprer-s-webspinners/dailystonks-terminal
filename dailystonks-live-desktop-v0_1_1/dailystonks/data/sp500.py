from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class SP500Universe:
    csv_path: str

    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "Symbol" not in df.columns:
            raise ValueError(f"S&P universe CSV must contain a Symbol column: {self.csv_path}")

        # Normalize columns from datasets/s-and-p-500-companies / Wikipedia-like exports.
        # The portable bundle can operate with a tiny Symbol-only fallback CSV; sector
        # cards will degrade to Unknown instead of crashing.
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        df = df[df["Symbol"].astype(bool)].copy()

        if "Name" not in df.columns:
            if "Security" in df.columns:
                df["Name"] = df["Security"].astype(str)
            else:
                df["Name"] = df["Symbol"]

        if "Sector" not in df.columns:
            if "GICS Sector" in df.columns:
                df["Sector"] = df["GICS Sector"].astype(str)
            else:
                df["Sector"] = "Unknown"

        return df

    def tickers(self, *, max_n: Optional[int] = None) -> List[str]:
        syms = self.df()["Symbol"].tolist()
        if max_n is not None:
            syms = syms[:max_n]
        return syms

    def by_sector(self, *, max_n: Optional[int] = None) -> pd.DataFrame:
        df = self.df()
        if max_n is not None:
            df = df.head(max_n)
        return df.groupby("Sector")["Symbol"].count().sort_values(ascending=False).to_frame("count")
