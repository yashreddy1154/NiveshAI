"""
NiveshAI — Build NIFTY 500 Database
Downloads the full NIFTY 500 constituents list from NSE and saves as CSV.

Usage:
    python scripts/build_nifty500_db.py
"""

import pandas as pd
import requests
import io
import time


def fetch_nifty500_from_nse():
    """Try to fetch NIFTY 500 list from NSE India / niftyindices.com."""
    urls = [
        "https://www.niftyindices.com/IndexConstituents/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for url in urls:
        try:
            print(f"Trying: {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"  ✅ Got {len(df)} stocks")
            return df
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            time.sleep(2)

    return None


def enrich_with_cap_category(df):
    """Add a Cap Category column based on known NIFTY indices."""
    # This is a simplified heuristic — first 100 are large, next 150 mid, rest small
    df = df.copy()
    cap_cats = []
    for i in range(len(df)):
        if i < 100:
            cap_cats.append("Large")
        elif i < 250:
            cap_cats.append("Mid")
        else:
            cap_cats.append("Small")
    df["Cap Category"] = cap_cats
    return df


def main():
    print("🏗️ Building NIFTY 500 Database...")
    print("=" * 50)

    df = fetch_nifty500_from_nse()

    if df is not None:
        # Standardize column names
        col_mapping = {
            "Company Name": "Company Name",
            "Industry": "Industry",
            "Symbol": "Symbol",
            "Series": "Series",
            "ISIN Code": "ISIN",
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

        # Add Sector (approximate from Industry)
        if "Sector" not in df.columns:
            df["Sector"] = df.get("Industry", "Unknown")

        # Add Cap Category
        df = enrich_with_cap_category(df)

        # Ensure required columns
        required = ["Symbol", "Company Name", "Sector", "Industry", "Series", "ISIN", "Cap Category"]
        for col in required:
            if col not in df.columns:
                df[col] = "Unknown"

        df = df[required]

        # Save
        output_path = "data/nifty500.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} stocks to {output_path}")
        print(f"\nSample:")
        print(df.head(10).to_string(index=False))
    else:
        print("\n⚠️ Could not fetch from NSE. Using existing data/nifty500.csv")
        print("You can manually download from: https://www.nseindia.com/market-data/live-equity-market")


if __name__ == "__main__":
    main()
