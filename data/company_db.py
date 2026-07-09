"""
NiveshAI — NIFTY 500 Company Database
Query interface for the pre-built NIFTY 500 company database.
"""
# TODO: Implement in Phase 2

import pandas as pd
from pathlib import Path


NIFTY500_PATH = Path(__file__).parent / "nifty500.csv"


def load_nifty500() -> pd.DataFrame:
    """Load the NIFTY 500 company database."""
    if NIFTY500_PATH.exists():
        return pd.read_csv(NIFTY500_PATH)
    return pd.DataFrame()


def search_companies(query: str, limit: int = 10) -> list:
    """Search companies by name or symbol."""
    raise NotImplementedError("Will be implemented in Phase 2")


def get_companies_by_sector(sector: str) -> list:
    """Filter companies by sector."""
    raise NotImplementedError("Will be implemented in Phase 2")


def get_companies_by_cap(cap_category: str) -> list:
    """Filter companies by market cap category (Large/Mid/Small)."""
    raise NotImplementedError("Will be implemented in Phase 2")
