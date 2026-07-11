"""
NiveshAI — Build NIFTY 500 Company Database
This script builds the nifty500.csv database file located in the data/ folder.
It uses a robust list of 115 predefined Indian stocks with hardcoded sectors, industries,
and cap categories as fallbacks, and attempts to fetch live names, sectors, and market caps
from yfinance to dynamically categorize them.
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Hardcoded companies database fallback (115 stocks)
PREDEFINED_STOCKS = [
    # Symbol, Company Name, Sector, Industry, Cap Category
    ("RELIANCE", "Reliance Industries Ltd", "Energy", "Oil & Gas", "Large"),
    ("TCS", "Tata Consultancy Services Ltd", "Information Technology", "IT Services", "Large"),
    ("INFY", "Infosys Ltd", "Information Technology", "IT Services", "Large"),
    ("HDFCBANK", "HDFC Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("ICICIBANK", "ICICI Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("BHARTIARTL", "Bharti Airtel Ltd", "Telecommunication", "Telecom Services", "Large"),
    ("SBIN", "State Bank of India", "Financial Services", "Public Sector Bank", "Large"),
    ("ITC", "ITC Ltd", "Consumer Goods", "FMCG", "Large"),
    ("KOTAKBANK", "Kotak Mahindra Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("LT", "Larsen & Toubro Ltd", "Construction", "Engineering & Construction", "Large"),
    ("AXISBANK", "Axis Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("WIPRO", "Wipro Ltd", "Information Technology", "IT Services", "Large"),
    ("BAJFINANCE", "Bajaj Finance Ltd", "Financial Services", "Non Banking Financial Company (NBFC)", "Large"),
    ("HCLTECH", "HCL Technologies Ltd", "Information Technology", "IT Services", "Large"),
    ("MARUTI", "Maruti Suzuki India Ltd", "Automobile", "Passenger Cars & Utility Vehicles", "Large"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("TATAMOTORS", "Tata Motors Ltd", "Automobile", "Passenger Cars & Utility Vehicles", "Large"),
    ("ONGC", "Oil & Natural Gas Corporation Ltd", "Energy", "Oil & Gas", "Large"),
    ("NTPC", "NTPC Ltd", "Utilities", "Power Generation", "Large"),
    ("POWERGRID", "Power Grid Corporation of India Ltd", "Utilities", "Power Transmission", "Large"),
    ("ADANIENT", "Adani Enterprises Ltd", "Metals & Mining", "Trading & Distributors", "Large"),
    ("ADANIPORTS", "Adani Ports and Special Economic Zone Ltd", "Services", "Port & Port Services", "Large"),
    ("COALINDIA", "Coal India Ltd", "Energy", "Coal", "Large"),
    ("ULTRACEMCO", "UltraTech Cement Ltd", "Construction Materials", "Cement & Cement Products", "Large"),
    ("NESTLEIND", "Nestle India Ltd", "Consumer Goods", "Food Products", "Large"),
    ("TITAN", "Titan Company Ltd", "Consumer Goods", "Consumer Durables", "Large"),
    ("ASIANPAINT", "Asian Paints Ltd", "Consumer Goods", "Paints", "Large"),
    ("BAJAJFINSV", "Bajaj Finserv Ltd", "Financial Services", "Insurance", "Large"),
    ("DIVISLAB", "Divi's Laboratories Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("DRREDDY", "Dr. Reddy's Laboratories Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("EICHERMOT", "Eicher Motors Ltd", "Automobile", "Two & Three Wheelers", "Large"),
    ("GRASIM", "Grasim Industries Ltd", "Construction Materials", "Diversified Chemicals", "Large"),
    ("HEROMOTOCO", "Hero MotoCorp Ltd", "Automobile", "Two & Three Wheelers", "Large"),
    ("HINDALCO", "Hindalco Industries Ltd", "Metals & Mining", "Aluminium", "Large"),
    ("HINDUNILVR", "Hindustan Unilever Ltd", "Consumer Goods", "FMCG", "Large"),
    ("INDUSINDBK", "IndusInd Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("JSWSTEEL", "JSW Steel Ltd", "Metals & Mining", "Iron & Steel", "Large"),
    ("M&M", "Mahindra & Mahindra Ltd", "Automobile", "Passenger Cars & Utility Vehicles", "Large"),
    ("SBILIFE", "SBI Life Insurance Company Ltd", "Financial Services", "Life Insurance", "Large"),
    ("TATASTEEL", "Tata Steel Ltd", "Metals & Mining", "Iron & Steel", "Large"),
    ("TECHM", "Tech Mahindra Ltd", "Information Technology", "IT Services", "Large"),
    ("VEDL", "Vedanta Ltd", "Metals & Mining", "Diversified Metals & Mining", "Large"),
    ("CIPLA", "Cipla Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("BPCL", "Bharat Petroleum Corporation Ltd", "Energy", "Oil & Gas", "Large"),
    ("HDFCLIFE", "HDFC Life Insurance Company Ltd", "Financial Services", "Life Insurance", "Large"),
    ("UPL", "UPL Ltd", "Chemicals", "Agrochemicals", "Large"),
    ("SHREECEM", "Shree Cement Ltd", "Construction Materials", "Cement & Cement Products", "Large"),
    ("TATACONSUM", "Tata Consumer Products Ltd", "Consumer Goods", "Food Products", "Large"),
    ("BRITANNIA", "Britannia Industries Ltd", "Consumer Goods", "Food Products", "Large"),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise Ltd", "Healthcare", "Healthcare Services", "Large"),
    ("BAJAJ-AUTO", "Bajaj Auto Ltd", "Automobile", "Two & Three Wheelers", "Large"),
    ("PIDILITIND", "Pidilite Industries Ltd", "Chemicals", "Adhesives & Sealants", "Large"),
    ("HAVELLS", "Havells India Ltd", "Consumer Goods", "Consumer Durables", "Large"),
    ("MUTHOOTFIN", "Muthoot Finance Ltd", "Financial Services", "Non Banking Financial Company (NBFC)", "Large"),
    ("BANDHANBNK", "Bandhan Bank Ltd", "Financial Services", "Private Sector Bank", "Large"),
    ("BIOCON", "Biocon Ltd", "Healthcare", "Biotechnology", "Mid"),
    ("CHOLAFIN", "Cholamandalam Investment and Finance Company Ltd", "Financial Services", "NBFC", "Large"),
    ("COFORGE", "Coforge Ltd", "Information Technology", "IT Services", "Mid"),
    ("CUB", "City Union Bank Ltd", "Financial Services", "Private Sector Bank", "Mid"),
    ("DALBHARAT", "Dalmia Bharat Ltd", "Construction Materials", "Cement", "Mid"),
    ("ESCORTS", "Escorts Kubota Ltd", "Automobile", "Tractors & Agricultural Machinery", "Mid"),
    ("FEDERALBNK", "Federal Bank Ltd", "Financial Services", "Private Sector Bank", "Mid"),
    ("GLENMARK", "Glenmark Pharmaceuticals Ltd", "Healthcare", "Pharmaceuticals", "Mid"),
    ("GODREJCP", "Godrej Consumer Products Ltd", "Consumer Goods", "FMCG", "Large"),
    ("GODREJPROP", "Godrej Properties Ltd", "Real Estate", "Residential & Commercial", "Mid"),
    ("GRANULES", "Granules India Ltd", "Healthcare", "Pharmaceuticals", "Small"),
    ("HAL", "Hindustan Aeronautics Ltd", "Capital Goods", "Aerospace & Defense", "Large"),
    ("IDFCFIRSTB", "IDFC First Bank Ltd", "Financial Services", "Private Sector Bank", "Mid"),
    ("INDUSTOWER", "Indus Towers Ltd", "Telecommunication", "Telecom Infrastructure", "Large"),
    ("IRCTC", "Indian Railway Catering & Tourism Corp Ltd", "Services", "Tourism & Catering Services", "Large"),
    ("JINDALSTEL", "Jindal Steel & Power Ltd", "Metals & Mining", "Iron & Steel", "Large"),
    ("JUBLFOOD", "Jubilant FoodWorks Ltd", "Consumer Goods", "Restaurants", "Mid"),
    ("LUPIN", "Lupin Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("MCDOWELL-N", "United Spirits Ltd", "Consumer Goods", "Beverages", "Large"),
    ("MFSL", "Max Financial Services Ltd", "Financial Services", "Life Insurance", "Mid"),
    ("MGL", "Mahanagar Gas Ltd", "Utilities", "Gas Distribution", "Mid"),
    ("MOTHERSON", "Samvardhana Motherson International Ltd", "Automobile", "Auto Components & Equipments", "Large"),
    ("MPHASIS", "Mphasis Ltd", "Information Technology", "IT Services", "Mid"),
    ("MRF", "MRF Ltd", "Automobile", "Tyres & Tubes", "Large"),
    ("NATIONALUM", "National Aluminium Company Ltd", "Metals & Mining", "Aluminium", "Mid"),
    ("NAVINFLUOR", "Navin Fluorine International Ltd", "Chemicals", "Speciality Chemicals", "Mid"),
    ("NAUKRI", "Info Edge (India) Ltd", "Information Technology", "Internet Software & Services", "Large"),
    ("OBEROIRLTY", "Oberoi Realty Ltd", "Real Estate", "Residential & Commercial", "Mid"),
    ("OFSS", "Oracle Financial Services Software Ltd", "Information Technology", "IT Services", "Large"),
    ("PAGEIND", "Page Industries Ltd", "Consumer Goods", "Apparel & Accessories", "Large"),
    ("PEL", "Piramal Enterprises Ltd", "Financial Services", "NBFC", "Mid"),
    ("PERSISTENT", "Persistent Systems Ltd", "Information Technology", "IT Services", "Large"),
    ("PETRONET", "Petronet LNG Ltd", "Energy", "Gas Distribution", "Large"),
    ("PFC", "Power Finance Corporation Ltd", "Financial Services", "NBFC", "Large"),
    ("POLYCAB", "Polycab India Ltd", "Capital Goods", "Cables & Wires", "Large"),
    ("PVRINOX", "PVR INOX Ltd", "Services", "Media & Entertainment", "Mid"),
    ("RBLBANK", "RBL Bank Ltd", "Financial Services", "Private Sector Bank", "Mid"),
    ("RECLTD", "REC Ltd", "Financial Services", "NBFC", "Large"),
    ("SAIL", "Steel Authority of India Ltd", "Metals & Mining", "Iron & Steel", "Large"),
    ("SBICARD", "SBI Cards and Payment Services Ltd", "Financial Services", "Credit Cards", "Large"),
    ("SIEMENS", "Siemens Ltd", "Capital Goods", "Electrical Equipment", "Large"),
    ("SRF", "SRF Ltd", "Chemicals", "Diversified Chemicals", "Large"),
    ("SUNTVNETWORK", "Sun TV Network Ltd", "Services", "Media & Entertainment", "Mid"),
    ("SUPREMEIND", "Supreme Industries Ltd", "Consumer Goods", "Plastic Products", "Large"),
    ("TATACOMM", "Tata Communications Ltd", "Telecommunication", "Telecom Services", "Large"),
    ("TATAMTRDVR", "Tata Motors Ltd DVR", "Automobile", "Passenger Cars & Utility Vehicles", "Mid"),
    ("TORNTPHARM", "Torrent Pharmaceuticals Ltd", "Healthcare", "Pharmaceuticals", "Large"),
    ("TRENT", "Trent Ltd", "Consumer Goods", "Retailing", "Large"),
    ("TVSMOTOR", "TVS Motor Company Ltd", "Automobile", "Two & Three Wheelers", "Large"),
    ("VBL", "Varun Beverages Ltd", "Consumer Goods", "Beverages", "Large"),
    ("VOLTAS", "Voltas Ltd", "Consumer Goods", "Consumer Durables", "Large"),
    ("WHIRLPOOL", "Whirlpool of India Ltd", "Consumer Goods", "Consumer Durables", "Mid"),
    ("ZOMATO", "Zomato Ltd", "Information Technology", "Internet Software & Services", "Large"),
    ("PAYTM", "One 97 Communications Ltd (Paytm)", "Financial Services", "Fintech", "Mid"),
    ("NYKAA", "FSN E-Commerce Ventures Ltd (Nykaa)", "Consumer Goods", "Retailing", "Mid"),
    ("DELHIVERY", "Delhivery Ltd", "Services", "Logistics", "Mid"),
    ("POLICYBZR", "PB Fintech Ltd (PolicyBazaar)", "Financial Services", "Fintech", "Large")
]

def fetch_live_company_details(symbol: str) -> dict:
    """Fetch live details for a symbol from yfinance."""
    import yfinance as yf
    ticker = f"{symbol}.NS"
    try:
        t = yf.Ticker(ticker)
        # We use fast_info or info (info can be slow/unstable so we try to catch errors)
        info = t.info
        if not info or len(info) < 2:
            return {}
        
        market_cap = info.get("marketCap", 0)
        
        # Categorise by Market Cap
        # marketCap is in INR. yfinance normally returns it in INR.
        # Large Cap: > 20,000 Crores (200,000,000,000 INR)
        # Mid Cap: 5,000 to 20,000 Crores (50,000,000,000 to 200,000,000,000 INR)
        # Small Cap: < 5,000 Crores
        if market_cap:
            if market_cap >= 200_000_000_000:
                cap_cat = "Large"
            elif market_cap >= 50_000_000_000:
                cap_cat = "Mid"
            else:
                cap_cat = "Small"
        else:
            cap_cat = None

        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "cap_cat": cap_cat
        }
    except Exception:
        return {}

def main():
    print("============================================================")
    print("           NiveshAI — NIFTY 500 Database Builder            ")
    print("============================================================")

    # Ensure data/ directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "nifty500.csv"

    print(f"Target path: {csv_path}")
    print(f"Preparing database with {len(PREDEFINED_STOCKS)} predefined stocks.")
    print("Attempting to enrich/validate records using live yfinance data...")
    print("(Press Ctrl+C to stop yfinance enrichment and save current cache/predefined list instantly)\n")

    final_records = []
    try:
        for symbol, fallback_name, fallback_sector, fallback_industry, fallback_cap in tqdm(PREDEFINED_STOCKS, desc="Processing stocks", unit="stock"):
            # Try to fetch live details
            live_details = fetch_live_company_details(symbol)
            
            # Combine live data with fallbacks
            name = live_details.get("name") or fallback_name
            sector = live_details.get("sector") or fallback_sector
            industry = live_details.get("industry") or fallback_industry
            cap_cat = live_details.get("cap_cat") or fallback_cap
            
            # Clean names and strings
            name = str(name).strip()
            sector = str(sector).strip()
            industry = str(industry).strip()
            cap_cat = str(cap_cat).strip()

            final_records.append({
                "Symbol": symbol,
                "Company Name": name,
                "Sector": sector,
                "Industry": industry,
                "Cap Category": cap_cat
            })
            
            # Sleep slightly to avoid spamming the Yahoo Finance API
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Process interrupted by user. Saving current data gathered so far...")
        # Fill in any missing ones with standard fallback values
        gathered_symbols = {r["Symbol"] for r in final_records}
        for symbol, fallback_name, fallback_sector, fallback_industry, fallback_cap in PREDEFINED_STOCKS:
            if symbol not in gathered_symbols:
                final_records.append({
                    "Symbol": symbol,
                    "Company Name": fallback_name,
                    "Sector": fallback_sector,
                    "Industry": fallback_industry,
                    "Cap Category": fallback_cap
                })

    # Save to CSV
    df = pd.DataFrame(final_records)
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Database successfully saved to: {csv_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total Companies   : {len(df)}")
    
    cap_counts = df["Cap Category"].value_counts()
    print("Cap Categories    :")
    for cap, count in cap_counts.items():
        print(f"  - {cap}: {count}")

    sector_counts = df["Sector"].value_counts()
    print("Top Sectors       :")
    for sector, count in list(sector_counts.items())[:5]:
        print(f"  - {sector}: {count}")

if __name__ == "__main__":
    main()
