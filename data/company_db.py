"""
NiveshAI — NIFTY 500 Company Database
Search, filter, and look up Indian companies from the NIFTY 500 universe.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import re

_CSV_PATH = Path(__file__).parent / "nifty500.csv"

# Cached in-memory after first load
_df: Optional[pd.DataFrame] = None


# ── Load ───────────────────────────────────────────────────────────────────────

def load_nifty500(force_reload: bool = False) -> pd.DataFrame:
    """Load (and cache) the NIFTY 500 company DataFrame."""
    global _df
    if _df is not None and not force_reload:
        return _df

    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY 500 CSV not found at {_CSV_PATH}. "
            "Run: python scripts/build_nifty500_db.py"
        )

    df = pd.read_csv(_CSV_PATH)

    # Normalise column names (strip whitespace, title-case)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist with fallback
    for col in ["Symbol", "Company Name", "Sector", "Industry", "Cap Category"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["Symbol"]       = df["Symbol"].str.strip().str.upper()
    df["Company Name"] = df["Company Name"].str.strip()
    df["Sector"]       = df["Sector"].str.strip()
    df["Industry"]     = df["Industry"].str.strip()
    df["Cap Category"] = df["Cap Category"].str.strip()

    _df = df
    return _df


# ── Lookups ────────────────────────────────────────────────────────────────────

def get_company(symbol: str) -> Optional[dict]:
    """Return all info for a single stock symbol. Case-insensitive."""
    df = load_nifty500()
    sym = symbol.strip().upper()
    row = df[df["Symbol"] == sym]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_nse_ticker(symbol: str) -> str:
    """Return the yfinance-compatible NSE ticker (e.g. 'RELIANCE' → 'RELIANCE.NS')."""
    return f"{symbol.strip().upper()}.NS"


def search_companies(query: str, limit: int = 10) -> list[dict]:
    """
    Fuzzy search companies by name OR symbol.
    Returns a list of dicts sorted by relevance.
    """
    df = load_nifty500()
    q  = query.strip().upper()

    if not q:
        return df.head(limit).to_dict("records")

    # Exact symbol match first
    exact = df[df["Symbol"] == q]
    if not exact.empty:
        return exact.head(limit).to_dict("records")

    # Partial match on symbol or company name
    mask = (
        df["Symbol"].str.contains(q, na=False, case=False) |
        df["Company Name"].str.contains(query.strip(), na=False, case=False)
    )
    results = df[mask].head(limit)
    return results.to_dict("records")


def get_all_symbols() -> list[str]:
    """Return sorted list of all NIFTY 500 symbols."""
    return sorted(load_nifty500()["Symbol"].tolist())


def get_display_options() -> list[str]:
    """Return list of 'SYMBOL — Company Name' strings for Streamlit selectbox."""
    df = load_nifty500()
    return [f"{row['Symbol']} — {row['Company Name']}" for _, row in df.iterrows()]


def parse_display_option(option: str) -> str:
    """Extract symbol from a display option string like 'RELIANCE — Reliance Industries'."""
    return option.split(" — ")[0].strip()


# ── Filters ────────────────────────────────────────────────────────────────────

def get_by_sector(sector: str) -> pd.DataFrame:
    """Return all companies in a given sector."""
    df = load_nifty500()
    return df[df["Sector"].str.contains(sector, case=False, na=False)]


def get_by_cap(cap: str) -> pd.DataFrame:
    """
    Filter by market cap category.
    cap: 'Large', 'Mid', or 'Small'
    """
    df = load_nifty500()
    return df[df["Cap Category"].str.lower() == cap.lower()]


def get_sectors() -> list[str]:
    """Return sorted unique sector names."""
    return sorted(load_nifty500()["Sector"].dropna().unique().tolist())


def get_industries() -> list[str]:
    """Return sorted unique industry names."""
    return sorted(load_nifty500()["Industry"].dropna().unique().tolist())


def get_sector_summary() -> pd.DataFrame:
    """Return count of companies per sector (for heatmap/charts)."""
    df = load_nifty500()
    return (
        df.groupby("Sector")
        .agg(count=("Symbol", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
    )


# ── Convenience: top N per sector ─────────────────────────────────────────────

def get_top_symbols(n: int = 20) -> list[str]:
    """
    Return the top N symbols by index order (proxy for market cap rank).
    The CSV is ordered largest-first within each cap tier.
    """
    df = load_nifty500()
    return df["Symbol"].head(n).tolist()
