"""
NiveshAI — Stock Data Pipeline
Fetches real-time and historical stock data from NSE via yfinance.
"""
# TODO: Implement in Phase 2


def fetch_stock_data(symbol: str, period: str = "2y"):
    """Fetch OHLCV data for an NSE stock."""
    raise NotImplementedError("Will be implemented in Phase 2")


def fetch_company_info(symbol: str) -> dict:
    """Fetch company fundamentals (P/E, Market Cap, etc.)."""
    raise NotImplementedError("Will be implemented in Phase 2")


def fetch_financials(symbol: str) -> dict:
    """Fetch income statement, balance sheet, cash flow."""
    raise NotImplementedError("Will be implemented in Phase 2")
