"""
NiveshAI — Stock Data Pipeline
Real-time and historical stock data from NSE via yfinance.
Includes SQLite caching to avoid redundant downloads.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# Internal imports
from data.cache import get_cache
from data.company_db import get_company, get_nse_ticker


# ── Constants ──────────────────────────────────────────────────────────────────

# Cache TTLs
TTL_PRICE_HOURS       = 0.25   # 15 minutes for live prices
TTL_HISTORY_HOURS     = 6      # 6 hours for historical OHLCV
TTL_FUNDAMENTALS_HOURS = 24    # 24 hours for fundamentals (changes slowly)
TTL_FINANCIALS_HOURS  = 48     # 48 hours for income/balance sheet

# Technical indicator periods
SMA_SHORT  = 20
SMA_LONG   = 50
SMA_TREND  = 200
RSI_PERIOD = 14


# ── Helpers ────────────────────────────────────────────────────────────────────

def _import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance not installed. Run: pip install yfinance"
        )


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to an OHLCV DataFrame."""
    if df.empty or len(df) < 2:
        return df

    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Moving Averages
    for p in [SMA_SHORT, SMA_LONG, SMA_TREND]:
        df[f"SMA_{p}"] = close.rolling(p).mean()

    # EMA 12 and 26 (used in MACD)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day, ±2σ)
    sma20 = close.rolling(SMA_SHORT).mean()
    std20 = close.rolling(SMA_SHORT).std()
    df["BB_Upper"]  = sma20 + 2 * std20
    df["BB_Lower"]  = sma20 - 2 * std20
    df["BB_Middle"] = sma20
    df["BB_PctB"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # Daily returns
    df["Daily_Return"] = close.pct_change() * 100

    # Volume MA
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    return df


def _generate_signal(df: pd.DataFrame) -> dict:
    """
    Generate a simple buy/sell/hold signal from the latest indicator values.
    Returns {'signal': 'BUY'|'SELL'|'HOLD', 'strength': float 0-1, 'reasons': [...]}
    """
    if df.empty:
        return {"signal": "HOLD", "strength": 0.5, "reasons": ["Insufficient data"]}

    latest = df.iloc[-1]
    reasons = []
    bull_count = 0
    bear_count = 0

    # RSI signal
    rsi = latest.get("RSI", 50)
    if rsi < 30:
        reasons.append(f"RSI {rsi:.0f} — Oversold (bullish)")
        bull_count += 2
    elif rsi > 70:
        reasons.append(f"RSI {rsi:.0f} — Overbought (bearish)")
        bear_count += 2
    else:
        reasons.append(f"RSI {rsi:.0f} — Neutral")

    # MACD signal
    macd_h = latest.get("MACD_Hist", 0)
    if macd_h > 0:
        reasons.append("MACD histogram positive (bullish momentum)")
        bull_count += 1
    elif macd_h < 0:
        reasons.append("MACD histogram negative (bearish momentum)")
        bear_count += 1

    # Price vs SMA signals
    close = latest.get("Close", 0)
    sma20 = latest.get("SMA_20", close)
    sma50 = latest.get("SMA_50", close)
    if close > sma20 > sma50:
        reasons.append("Price > SMA20 > SMA50 (uptrend)")
        bull_count += 1
    elif close < sma20 < sma50:
        reasons.append("Price < SMA20 < SMA50 (downtrend)")
        bear_count += 1

    # Determine signal
    total = bull_count + bear_count
    if total == 0:
        return {"signal": "HOLD", "strength": 0.5, "reasons": reasons}

    bull_pct = bull_count / total
    if bull_pct >= 0.65:
        return {"signal": "BUY",  "strength": bull_pct, "reasons": reasons}
    elif bull_pct <= 0.35:
        return {"signal": "SELL", "strength": 1 - bull_pct, "reasons": reasons}
    else:
        return {"signal": "HOLD", "strength": 0.5, "reasons": reasons}


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    add_indicators: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for an NSE stock.

    Args:
        symbol:  NSE symbol (e.g. 'RELIANCE')
        period:  yfinance period string ('1y', '2y', '6mo', '3mo', etc.)
        interval: bar interval ('1d', '1wk', '1h')
        add_indicators: if True, adds SMA, RSI, MACD, Bollinger Bands

    Returns:
        DataFrame with OHLCV + indicators, or empty DataFrame on failure.
    """
    cache = get_cache()
    cache_key = f"history:{symbol}:{period}:{interval}"

    # Try cache first
    cached = cache.get(cache_key, max_age_hours=TTL_HISTORY_HOURS)
    if cached:
        df = pd.DataFrame(cached)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            return df

    yf = _import_yfinance()
    ticker = get_nse_ticker(symbol)

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if hasattr(raw.columns, "levels"):
            raw.columns = raw.columns.get_level_values(0)

        # Standard column names
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.dropna()

        if add_indicators and len(df) >= 20:
            df = _add_technical_indicators(df)

        # Cache (store as dict with string index)
        df_cache = df.copy()
        df_cache.index = df_cache.index.strftime("%Y-%m-%d")
        cache.set(cache_key, df_cache.to_dict(), ttl_hours=TTL_HISTORY_HOURS)

        return df

    except Exception as e:
        print(f"[stock_data] Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_live_price(symbol: str) -> dict:
    """
    Fetch the current (live/delayed) price and basic stats.

    Returns:
        {
            'symbol': str, 'price': float, 'change': float, 'change_pct': float,
            'open': float, 'high': float, 'low': float, 'volume': int,
            'prev_close': float, 'timestamp': str
        }
    """
    cache = get_cache()
    cache_key = f"live:{symbol}"

    cached = cache.get(cache_key, max_age_hours=TTL_PRICE_HOURS)
    if cached:
        return cached

    yf = _import_yfinance()
    ticker_obj = yf.Ticker(get_nse_ticker(symbol))

    try:
        info  = ticker_obj.fast_info
        hist1d = ticker_obj.history(period="2d", interval="1d")

        if hist1d.empty:
            return _empty_price(symbol)

        last   = hist1d.iloc[-1]
        prev   = hist1d.iloc[-2] if len(hist1d) >= 2 else last

        price      = float(last["Close"])
        prev_close = float(prev["Close"])
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        result = {
            "symbol":     symbol,
            "price":      round(price, 2),
            "change":     round(change, 2),
            "change_pct": round(change_pct, 2),
            "open":       round(float(last.get("Open", price)), 2),
            "high":       round(float(last.get("High", price)), 2),
            "low":        round(float(last.get("Low", price)), 2),
            "volume":     int(last.get("Volume", 0)),
            "prev_close": round(prev_close, 2),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        cache.set(cache_key, result, ttl_hours=TTL_PRICE_HOURS)
        return result

    except Exception as e:
        print(f"[stock_data] Error fetching live price for {symbol}: {e}")
        return _empty_price(symbol)


def fetch_fundamentals(symbol: str) -> dict:
    """
    Fetch fundamental data: P/E, market cap, EPS, dividend yield, etc.

    Returns a dict with all available fundamental metrics.
    """
    cache = get_cache()
    cache_key = f"fundamentals:{symbol}"

    cached = cache.get(cache_key, max_age_hours=TTL_FUNDAMENTALS_HOURS)
    if cached:
        return cached

    yf = _import_yfinance()
    ticker_obj = yf.Ticker(get_nse_ticker(symbol))

    try:
        info = ticker_obj.info
        if not info or "symbol" not in info and "shortName" not in info:
            return _empty_fundamentals(symbol)

        def safe(key, default=None):
            v = info.get(key)
            return v if v not in (None, "N/A", "", float("inf")) else default

        market_cap = safe("marketCap", 0)
        result = {
            "symbol":          symbol,
            "company_name":    safe("longName") or safe("shortName", symbol),
            "sector":          safe("sector", "Unknown"),
            "industry":        safe("industry", "Unknown"),
            "website":         safe("website", ""),
            "description":     safe("longBusinessSummary", "")[:500] if safe("longBusinessSummary") else "",
            # Valuation
            "market_cap":      market_cap,
            "market_cap_cr":   round(market_cap / 1e7, 2) if market_cap else None,  # ₹ Crores
            "pe_ratio":        safe("trailingPE"),
            "forward_pe":      safe("forwardPE"),
            "pb_ratio":        safe("priceToBook"),
            "ps_ratio":        safe("priceToSalesTrailing12Months"),
            "ev_ebitda":       safe("enterpriseToEbitda"),
            # Per-share metrics
            "eps":             safe("trailingEps"),
            "book_value":      safe("bookValue"),
            # Profitability
            "roe":             safe("returnOnEquity"),
            "roa":             safe("returnOnAssets"),
            "profit_margin":   safe("profitMargins"),
            "op_margin":       safe("operatingMargins"),
            # Income
            "revenue":         safe("totalRevenue"),
            "ebitda":          safe("ebitda"),
            "net_income":      safe("netIncomeToCommon"),
            "revenue_growth":  safe("revenueGrowth"),
            "earnings_growth": safe("earningsGrowth"),
            # Cash & Debt
            "total_cash":      safe("totalCash"),
            "total_debt":      safe("totalDebt"),
            "debt_to_equity":  safe("debtToEquity"),
            "current_ratio":   safe("currentRatio"),
            "free_cashflow":   safe("freeCashflow"),
            # Dividends
            "dividend_yield":  safe("dividendYield"),
            "dividend_rate":   safe("dividendRate"),
            "payout_ratio":    safe("payoutRatio"),
            # 52-week range
            "week_52_high":    safe("fiftyTwoWeekHigh"),
            "week_52_low":     safe("fiftyTwoWeekLow"),
            "avg_volume":      safe("averageVolume"),
            "beta":            safe("beta"),
            "fetched_at":      datetime.now().isoformat(),
        }

        cache.set(cache_key, result, ttl_hours=TTL_FUNDAMENTALS_HOURS)
        return result

    except Exception as e:
        print(f"[stock_data] Error fetching fundamentals for {symbol}: {e}")
        return _empty_fundamentals(symbol)


def fetch_financials(symbol: str) -> dict:
    """
    Fetch quarterly income statement and balance sheet data.
    Returns {'income': DataFrame, 'balance': DataFrame, 'cashflow': DataFrame}
    """
    cache = get_cache()
    cache_key = f"financials:{symbol}"

    cached = cache.get(cache_key, max_age_hours=TTL_FINANCIALS_HOURS)
    if cached:
        return {
            k: pd.DataFrame(v) for k, v in cached.items()
        }

    yf = _import_yfinance()
    ticker_obj = yf.Ticker(get_nse_ticker(symbol))

    result = {}
    try:
        for key, attr in [
            ("income",   "quarterly_income_stmt"),
            ("balance",  "quarterly_balance_sheet"),
            ("cashflow", "quarterly_cashflow"),
        ]:
            try:
                df = getattr(ticker_obj, attr)
                if df is not None and not df.empty:
                    result[key] = df.head(4)   # last 4 quarters
                else:
                    result[key] = pd.DataFrame()
            except Exception:
                result[key] = pd.DataFrame()

        # Cache as dicts
        cache.set(
            cache_key,
            {k: v.to_dict() for k, v in result.items()},
            ttl_hours=TTL_FINANCIALS_HOURS,
        )
        return result

    except Exception as e:
        print(f"[stock_data] Error fetching financials for {symbol}: {e}")
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def fetch_full_stock_data(symbol: str, period: str = "1y") -> dict:
    """
    All-in-one fetch: history + indicators + live price + fundamentals.
    Used by the AI Research Agent's get_stock_data tool.
    """
    history      = fetch_history(symbol, period=period)
    live_price   = fetch_live_price(symbol)
    fundamentals = fetch_fundamentals(symbol)
    signal       = _generate_signal(history) if not history.empty else {}
    company_info = get_company(symbol) or {}

    return {
        "symbol":        symbol,
        "nse_ticker":    get_nse_ticker(symbol),
        "company_info":  company_info,
        "live_price":    live_price,
        "fundamentals":  fundamentals,
        "history":       history,
        "signal":        signal,
        "fetched_at":    datetime.now().isoformat(),
    }


def fetch_market_indices() -> dict:
    """
    Fetch major Indian market indices: NIFTY 50, SENSEX, BANK NIFTY.
    Returns dict of {name: {price, change, change_pct}}.
    """
    cache = get_cache()
    cache_key = "indices:india"
    cached = cache.get(cache_key, max_age_hours=TTL_PRICE_HOURS)
    if cached:
        return cached

    yf = _import_yfinance()
    indices = {
        "NIFTY 50":   "^NSEI",
        "SENSEX":     "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "NIFTY IT":   "^CNXIT",
        "NIFTY MID":  "^NSEMDCP50",
    }

    result = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="1d")
            if len(hist) >= 2:
                last = float(hist.iloc[-1]["Close"])
                prev = float(hist.iloc[-2]["Close"])
                chg  = last - prev
                pct  = (chg / prev * 100) if prev else 0
                result[name] = {
                    "price":      round(last, 2),
                    "change":     round(chg, 2),
                    "change_pct": round(pct, 2),
                }
            elif len(hist) == 1:
                result[name] = {
                    "price":      round(float(hist.iloc[-1]["Close"]), 2),
                    "change":     0,
                    "change_pct": 0,
                }
        except Exception:
            result[name] = {"price": 0, "change": 0, "change_pct": 0}

    cache.set(cache_key, result, ttl_hours=TTL_PRICE_HOURS)
    return result


def fetch_multiple_prices(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch current prices for a list of symbols simultaneously (faster via yf.download batch).
    Returns DataFrame with symbol as index.
    """
    cache = get_cache()
    cache_key = f"multi:{':'.join(sorted(symbols))}"
    cached = cache.get(cache_key, max_age_hours=TTL_PRICE_HOURS)
    if cached:
        return pd.DataFrame(cached)

    yf = _import_yfinance()
    tickers = [get_nse_ticker(s) for s in symbols]

    try:
        data = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()

        close = data["Close"] if "Close" in data else data.get("close", pd.DataFrame())
        if isinstance(close, pd.Series):
            close = close.to_frame()

        # Get last 2 rows for change calculation
        result_rows = []
        for ticker, sym in zip(tickers, symbols):
            col = ticker if ticker in close.columns else None
            if col is None:
                continue
            prices = close[col].dropna()
            if len(prices) >= 2:
                last = float(prices.iloc[-1])
                prev = float(prices.iloc[-2])
                chg  = last - prev
                pct  = (chg / prev * 100) if prev else 0
                result_rows.append({
                    "symbol": sym, "price": round(last, 2),
                    "change": round(chg, 2), "change_pct": round(pct, 2),
                })
            elif len(prices) == 1:
                result_rows.append({
                    "symbol": sym, "price": round(float(prices.iloc[-1]), 2),
                    "change": 0, "change_pct": 0,
                })

        df = pd.DataFrame(result_rows).set_index("symbol") if result_rows else pd.DataFrame()
        if not df.empty:
            cache.set(cache_key, df.reset_index().to_dict("records"), ttl_hours=TTL_PRICE_HOURS)
        return df

    except Exception as e:
        print(f"[stock_data] Error in batch price fetch: {e}")
        return pd.DataFrame()


# ── Utility ────────────────────────────────────────────────────────────────────

def format_inr(value: float, crore: bool = False) -> str:
    """Format a number in Indian currency style (Lakh / Crore)."""
    if value is None:
        return "N/A"
    if crore:
        return f"₹{value:,.2f} Cr"
    if abs(value) >= 1e7:   # Crore
        return f"₹{value/1e7:,.2f} Cr"
    if abs(value) >= 1e5:   # Lakh
        return f"₹{value/1e5:,.2f} L"
    return f"₹{value:,.2f}"


def is_market_open() -> bool:
    """Check if NSE market is currently open (9:15–15:30 IST, Mon–Fri)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    market_open  = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ── Fallback empty results ─────────────────────────────────────────────────────

def _empty_price(symbol: str) -> dict:
    return {
        "symbol": symbol, "price": 0, "change": 0, "change_pct": 0,
        "open": 0, "high": 0, "low": 0, "volume": 0,
        "prev_close": 0, "timestamp": "N/A",
    }


def _empty_fundamentals(symbol: str) -> dict:
    return {
        "symbol": symbol, "company_name": symbol,
        "sector": "Unknown", "industry": "Unknown",
        "market_cap": None, "market_cap_cr": None,
        "pe_ratio": None, "forward_pe": None,
        "eps": None, "book_value": None,
        "roe": None, "profit_margin": None,
        "dividend_yield": None,
        "week_52_high": None, "week_52_low": None,
        "beta": None, "description": "",
        "fetched_at": datetime.now().isoformat(),
    }
