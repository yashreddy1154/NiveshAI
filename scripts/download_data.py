"""
NiveshAI — Download Stock Data
Pre-downloads historical stock data for NIFTY stocks and caches locally.

Usage:
    python scripts/download_data.py                    # Download NIFTY 50
    python scripts/download_data.py --all              # Download all NIFTY 500
    python scripts/download_data.py --symbols TCS INFY # Download specific stocks
"""

import argparse
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "SBIN",
    "ITC", "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "HCLTECH", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "ONGC", "NTPC", "POWERGRID", "AXISBANK",
    "WIPRO", "ADANIENT", "TATASTEEL", "ASIANPAINT", "BAJAJFINSV", "COALINDIA",
    "NESTLEIND", "TITAN", "ULTRACEMCO", "TECHM", "JSWSTEEL", "INDUSINDBK",
    "HINDALCO", "DRREDDY", "CIPLA", "BPCL", "EICHERMOT", "GRASIM",
    "APOLLOHOSP", "DIVISLAB", "TATACONSUM", "ADANIPORTS", "HEROMOTOCO",
    "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "BRITANNIA", "VEDL", "HAL", "BEL",
]


def download_stock_data(symbols, period="2y"):
    """Download and cache stock data for given symbols."""
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return

    os.makedirs("data/cache", exist_ok=True)
    success = 0
    failed = []

    for i, symbol in enumerate(symbols):
        ticker = f"{symbol}.NS"
        print(f"  [{i+1}/{len(symbols)}] Downloading {ticker}...", end=" ")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if len(hist) > 0:
                hist.to_csv(f"data/cache/{symbol}_history.csv")
                print(f"✅ {len(hist)} rows")
                success += 1
            else:
                print("⚠️ No data returned")
                failed.append(symbol)
        except Exception as e:
            print(f"❌ {e}")
            failed.append(symbol)

    print(f"\n{'='*50}")
    print(f"✅ Downloaded: {success}/{len(symbols)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(description="Download stock data for NiveshAI")
    parser.add_argument("--all", action="store_true", help="Download all NIFTY 500 stocks")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    args = parser.parse_args()

    print("📦 NiveshAI — Stock Data Downloader")
    print("=" * 50)

    if args.symbols:
        symbols = args.symbols
        print(f"Downloading {len(symbols)} specified stocks...")
    elif args.all:
        csv_path = "data/nifty500.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            symbols = df["Symbol"].tolist()
            print(f"Downloading all {len(symbols)} NIFTY 500 stocks...")
        else:
            print(f"⚠️ {csv_path} not found. Falling back to NIFTY 50.")
            symbols = NIFTY_50
    else:
        symbols = NIFTY_50
        print(f"Downloading NIFTY 50 stocks ({len(symbols)} stocks)...")

    print(f"Period: {args.period}\n")
    download_stock_data(symbols, period=args.period)


if __name__ == "__main__":
    main()
