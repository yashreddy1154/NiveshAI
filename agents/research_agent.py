"""
NiveshAI — AI Research Agent
Multi-tool agent that orchestrates data fetching, model inference, and analysis
to generate comprehensive investment research.
"""
# TODO: Implement in Phase 2


class ResearchAgent:
    """AI Research Agent that uses tool-calling to answer investment questions.

    Tools available:
        - get_stock_data: Fetch OHLCV + fundamentals from yfinance
        - get_news: Fetch recent financial news
        - analyze_sentiment: Run DistilBERT sentiment on news
        - predict_price: Run LSTM price prediction
        - get_fundamentals: Get P/E, market cap, revenue
        - calculate_technicals: Compute RSI, MACD, Bollinger
        - compare_stocks: Compare multiple tickers
        - optimize_portfolio: Run portfolio optimization
    """

    def __init__(self, provider=None):
        self.provider = provider
        self.tools = []
        # TODO: Register tools

    def answer(self, question: str) -> dict:
        """Answer a user's investment question using tool-calling.

        Returns: {
            'response': str,        # Final text response
            'tool_calls': list,     # List of tools called
            'charts': list,         # Any Plotly charts to display
            'data_tables': list,    # Any DataFrames to display
        }
        """
        raise NotImplementedError("Will be implemented in Phase 2")
