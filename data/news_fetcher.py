"""
NiveshAI — News Fetcher
Collects financial news from NewsAPI, Google News RSS, and web scraping.
"""
# TODO: Implement in Phase 2


def fetch_news_api(query: str, page_size: int = 10) -> list:
    """Fetch news articles from NewsAPI."""
    raise NotImplementedError("Will be implemented in Phase 2")


def fetch_google_news(query: str, num_results: int = 10) -> list:
    """Fetch news from Google News RSS (unlimited, no API key)."""
    raise NotImplementedError("Will be implemented in Phase 2")


def fetch_all_news(stock_name: str) -> list:
    """Aggregate news from all sources, deduplicate, and return."""
    raise NotImplementedError("Will be implemented in Phase 2")
