"""
NiveshAI — Agent Tool Definitions
Defines the tools available to the AI Research Agent.
"""
# TODO: Implement in Phase 2

TOOLS = [
    {
        "name": "get_stock_data",
        "description": "Fetch OHLCV price data and company info for an NSE-listed stock",
        "parameters": {"symbol": "str — NSE stock symbol (e.g., RELIANCE)"},
    },
    {
        "name": "get_news",
        "description": "Fetch recent financial news articles for a company",
        "parameters": {"company_name": "str", "num_articles": "int (default 10)"},
    },
    {
        "name": "analyze_sentiment",
        "description": "Run sentiment analysis on news articles using the trained DistilBERT model",
        "parameters": {"texts": "list of str — news headlines/articles"},
    },
    {
        "name": "predict_price",
        "description": "Predict future stock prices using the trained LSTM model",
        "parameters": {"symbol": "str", "days": "int (default 7)"},
    },
    {
        "name": "get_fundamentals",
        "description": "Get fundamental analysis — P/E, Market Cap, Revenue, EPS, etc.",
        "parameters": {"symbol": "str"},
    },
    {
        "name": "calculate_technicals",
        "description": "Calculate technical indicators — RSI, MACD, SMA, Bollinger Bands",
        "parameters": {"symbol": "str"},
    },
    {
        "name": "compare_stocks",
        "description": "Compare multiple stocks on price performance, fundamentals, and technicals",
        "parameters": {"symbols": "list of str"},
    },
    {
        "name": "optimize_portfolio",
        "description": "Run Modern Portfolio Theory optimization on a set of stocks",
        "parameters": {"symbols": "list of str", "risk_tolerance": "str (conservative/moderate/aggressive)"},
    },
]
