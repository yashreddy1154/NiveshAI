"""
NiveshAI — News Fetcher
Collects financial news from multiple sources:
  1. NewsAPI (free tier: 100 req/day)
  2. Google News RSS (unlimited, no key)
  3. Moneycontrol RSS (Indian financial news)
All results are deduplicated, cached, and returned in a unified format.
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import xml.etree.ElementTree as ET

from data.cache import get_cache

# Cache TTL
TTL_NEWS_HOURS = 1.0   # refresh news every hour

# ── Unified Article Format ─────────────────────────────────────────────────────
# Every source returns a list of dicts with these keys:
# {
#   "title": str,
#   "url": str,
#   "source": str,
#   "published_at": str (ISO 8601 or human-readable),
#   "summary": str,       # snippet / description (may be empty)
#   "sentiment": str,     # filled in later by sentiment model (default: "neutral")
#   "relevance": float,   # 0-1, how relevant to query (default: 1.0)
# }


def _make_article(title, url, source, published_at="", summary="") -> dict:
    return {
        "title":        title.strip() if title else "",
        "url":          url.strip() if url else "",
        "source":       source,
        "published_at": published_at,
        "summary":      summary.strip() if summary else "",
        "sentiment":    "neutral",
        "relevance":    1.0,
    }


def _deduplicate(articles: list[dict]) -> list[dict]:
    """Remove articles with duplicate titles (case-insensitive)."""
    seen  = set()
    clean = []
    for a in articles:
        key = re.sub(r"[^a-z0-9]", "", a["title"].lower())[:80]
        if key and key not in seen:
            seen.add(key)
            clean.append(a)
    return clean


def _clean_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


# ── Source 1: NewsAPI ──────────────────────────────────────────────────────────

def fetch_newsapi(query: str, api_key: str, page_size: int = 10) -> list[dict]:
    """
    Fetch articles from NewsAPI.org (free tier: 100 req/day).
    Focuses on Indian financial sources.

    Args:
        query:    Search term (e.g. "Reliance Industries stock")
        api_key:  NewsAPI key from newsapi.org
        page_size: Max articles to return (max 100 on free tier)
    """
    if not api_key:
        return []

    try:
        import requests
    except ImportError:
        return []

    # Indian financial sources available on NewsAPI
    INDIAN_SOURCES = (
        "the-times-of-india,the-hindu,business-standard,moneycontrol,"
        "economic-times,livemint,ndtv,hindustan-times"
    )

    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": min(page_size, 100),
        "apiKey":   api_key,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[news] NewsAPI error {resp.status_code}: {resp.json().get('message', '')}")
            return []

        data = resp.json()
        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "") or ""
            if title == "[Removed]" or not title:
                continue
            articles.append(_make_article(
                title        = title,
                url          = item.get("url", ""),
                source       = item.get("source", {}).get("name", "NewsAPI"),
                published_at = item.get("publishedAt", "")[:10],
                summary      = _clean_html(item.get("description") or item.get("content") or ""),
            ))
        return articles

    except Exception as e:
        print(f"[news] NewsAPI fetch failed: {e}")
        return []


# ── Source 2: Google News RSS ──────────────────────────────────────────────────

def fetch_google_news_rss(query: str, num_results: int = 10) -> list[dict]:
    """
    Fetch articles from Google News RSS (free, no API key needed).
    Uses the GNews RSS endpoint which doesn't require authentication.
    """
    try:
        import requests
        url = (
            f"https://news.google.com/rss/search"
            f"?q={requests.utils.quote(query + ' India stock NSE')}"
            f"&hl=en-IN&gl=IN&ceid=IN:en"
        )
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NiveshAI/1.0)"
        })
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        articles = []
        for item in channel.findall("item")[:num_results]:
            title   = item.findtext("title", "")
            link    = item.findtext("link", "")
            pub     = item.findtext("pubDate", "")
            desc    = _clean_html(item.findtext("description", ""))
            source_el = item.find("{https://news.google.com/rss}source")
            source  = source_el.text if source_el is not None else "Google News"

            # Convert pub date to simple format
            try:
                pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
                pub_str = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                pub_str = pub[:10] if len(pub) >= 10 else pub

            if title:
                articles.append(_make_article(title, link, source, pub_str, desc))

        return articles

    except Exception as e:
        print(f"[news] Google News RSS fetch failed: {e}")
        return []


# ── Source 3: Moneycontrol RSS ─────────────────────────────────────────────────

def fetch_moneycontrol_rss(category: str = "markets") -> list[dict]:
    """
    Fetch from Moneycontrol's RSS feed — a top Indian financial news source.
    category: 'markets', 'economy', 'stocks', 'ipo'
    """
    RSS_FEEDS = {
        "markets":  "https://www.moneycontrol.com/rss/marketreports.xml",
        "economy":  "https://www.moneycontrol.com/rss/economy.xml",
        "stocks":   "https://www.moneycontrol.com/rss/latestnews.xml",
        "ipo":      "https://www.moneycontrol.com/rss/ipo.xml",
    }
    url = RSS_FEEDS.get(category, RSS_FEEDS["stocks"])

    try:
        import requests
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NiveshAI/1.0)"
        })
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        articles = []
        for item in channel.findall("item")[:15]:
            title = item.findtext("title", "")
            link  = item.findtext("link", "")
            pub   = item.findtext("pubDate", "")
            desc  = _clean_html(item.findtext("description", ""))
            try:
                pub_dt  = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
                pub_str = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                pub_str = pub[:10] if len(pub) >= 10 else pub

            if title:
                articles.append(_make_article(title, link, "Moneycontrol", pub_str, desc))
        return articles

    except Exception as e:
        print(f"[news] Moneycontrol RSS failed: {e}")
        return []


# ── Source 4: Economic Times RSS ──────────────────────────────────────────────

def fetch_economic_times_rss() -> list[dict]:
    """Fetch from Economic Times markets RSS."""
    try:
        import requests
        url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        resp = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NiveshAI/1.0)"
        })
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        articles = []
        for item in channel.findall("item")[:10]:
            title = item.findtext("title", "")
            link  = item.findtext("link", "")
            pub   = item.findtext("pubDate", "")
            desc  = _clean_html(item.findtext("description", ""))
            try:
                pub_dt  = datetime.strptime(pub.strip(), "%a, %d %b %Y %H:%M:%S %z")
                pub_str = pub_dt.strftime("%Y-%m-%d")
            except Exception:
                pub_str = ""

            if title:
                articles.append(_make_article(title, link, "Economic Times", pub_str, desc))
        return articles

    except Exception as e:
        print(f"[news] Economic Times RSS failed: {e}")
        return []


# ── Relevance Scoring ─────────────────────────────────────────────────────────

def _score_relevance(articles: list[dict], company_name: str, symbol: str) -> list[dict]:
    """Score each article's relevance to the target stock."""
    keywords = set(
        [w.lower() for w in re.split(r"\W+", company_name) if len(w) > 2]
        + [symbol.lower()]
    )

    for a in articles:
        text = (a["title"] + " " + a["summary"]).lower()
        hits = sum(1 for kw in keywords if kw in text)
        a["relevance"] = min(1.0, hits / max(1, len(keywords)))

    return sorted(articles, key=lambda x: x["relevance"], reverse=True)


# ── Main Public API ────────────────────────────────────────────────────────────

def fetch_news(
    symbol: str,
    company_name: Optional[str] = None,
    newsapi_key: Optional[str] = None,
    max_articles: int = 15,
) -> list[dict]:
    """
    Fetch and aggregate news for a stock from all available sources.
    Results are cached for TTL_NEWS_HOURS.

    Args:
        symbol:       NSE symbol (e.g. 'RELIANCE')
        company_name: Full company name for better search (optional)
        newsapi_key:  NewsAPI key (optional — falls back to RSS if not provided)
        max_articles: Max total articles to return

    Returns:
        List of article dicts, sorted by relevance, deduplicated.
    """
    cache = get_cache()
    cache_key = f"news:{symbol}:{max_articles}"

    cached = cache.get(cache_key, max_age_hours=TTL_NEWS_HOURS)
    if cached:
        return cached

    # Resolve company name from DB if not provided
    if not company_name:
        try:
            from data.company_db import get_company
            info = get_company(symbol)
            company_name = info.get("Company Name", symbol) if info else symbol
        except Exception:
            company_name = symbol

    # Build search query
    query_short = symbol           # e.g. "RELIANCE"
    query_full  = company_name     # e.g. "Reliance Industries"

    all_articles = []

    # 1. NewsAPI (if key provided)
    if newsapi_key:
        all_articles += fetch_newsapi(
            f"{query_full} stock India",
            api_key=newsapi_key,
            page_size=10,
        )

    # 2. Google News RSS (always try)
    all_articles += fetch_google_news_rss(query_full, num_results=10)
    if len(all_articles) < 5:
        # Also try with shorter query
        all_articles += fetch_google_news_rss(query_short + " NSE", num_results=8)

    # 3. Moneycontrol RSS (general market news, always)
    all_articles += fetch_moneycontrol_rss("markets")

    # 4. Economic Times
    all_articles += fetch_economic_times_rss()

    # Deduplicate
    all_articles = _deduplicate(all_articles)

    # Score relevance and sort
    all_articles = _score_relevance(all_articles, company_name, symbol)

    # Take top N
    final = all_articles[:max_articles]

    # Cache result
    cache.set(cache_key, final, ttl_hours=TTL_NEWS_HOURS)
    return final


def fetch_market_news(max_articles: int = 20) -> list[dict]:
    """
    Fetch general Indian market news (not stock-specific).
    Used on the Home page for market overview.
    """
    cache = get_cache()
    cache_key = f"news:market_general:{max_articles}"
    cached = cache.get(cache_key, max_age_hours=TTL_NEWS_HOURS)
    if cached:
        return cached

    all_articles = []
    all_articles += fetch_moneycontrol_rss("markets")
    all_articles += fetch_moneycontrol_rss("economy")
    all_articles += fetch_economic_times_rss()
    all_articles += fetch_google_news_rss("Indian stock market NSE NIFTY today", 10)

    final = _deduplicate(all_articles)[:max_articles]
    cache.set(cache_key, final, ttl_hours=TTL_NEWS_HOURS)
    return final


def get_news_headlines(articles: list[dict]) -> list[str]:
    """Extract just the headline strings from a list of articles."""
    return [a["title"] for a in articles if a.get("title")]


def format_news_for_llm(articles: list[dict], max_items: int = 8) -> str:
    """
    Format news articles as a numbered list for LLM context injection.
    Keeps it concise to save tokens.
    """
    lines = []
    for i, a in enumerate(articles[:max_items], 1):
        date = a.get("published_at", "")
        src  = a.get("source", "")
        sent = a.get("sentiment", "neutral")
        lines.append(
            f"{i}. [{src} | {date} | {sent}] {a['title']}"
        )
        if a.get("summary"):
            lines.append(f"   → {a['summary'][:120]}")
    return "\n".join(lines)
