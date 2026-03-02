"""
Service 1: Financial News via Alpha Vantage NEWS_SENTIMENT API

Retrieves current financial news articles with sentiment scores for specific
stock tickers or broad financial topics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import requests
from langchain.tools import tool
from dotenv import load_dotenv

from assignment_chat.utils.logger import get_logger

_logs = get_logger(__name__)

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")
load_dotenv(_root / ".secrets")

# Alpha Vantage NEWS_SENTIMENT endpoint
_AV_URL = "https://www.alphavantage.co/query"

# Predefined topics accepted by Alpha Vantage
_VALID_TOPICS = {
    "blockchain", "earnings", "ipo", "mergers_and_acquisitions",
    "financial_markets", "economy_fiscal", "economy_monetary",
    "economy_macro", "energy_transportation", "finance",
    "life_sciences", "manufacturing", "real_estate",
    "retail_wholesale", "technology"
}

_SENTIMENT_LABELS = {
    "Bearish": "Bearish (negative)",
    "Somewhat-Bearish": "Somewhat Bearish",
    "Neutral": "Neutral",
    "Somewhat-Bullish": "Somewhat Bullish",
    "Bullish": "Bullish (positive)",
}


def _format_article(article: dict) -> str:
    """Formats a single news article into a readable string."""
    title = article.get("title", "No title")
    source = article.get("source", "Unknown source")
    published = article.get("time_published", "")[:8]  # YYYYMMDD
    if len(published) == 8:
        published = f"{published[:4]}-{published[4:6]}-{published[6:]}"
    summary = article.get("summary", "No summary available.")
    sentiment_label = article.get("overall_sentiment_label", "Neutral")
    sentiment_score = article.get("overall_sentiment_score", 0.0)
    sentiment_display = _SENTIMENT_LABELS.get(sentiment_label, sentiment_label)

    return (
        f"**{title}**\n"
        f"Source: {source} | Published: {published} | "
        f"Sentiment: {sentiment_display} (score: {sentiment_score:.3f})\n"
        f"{summary}\n"
    )


@tool
def get_financial_news(ticker: str = "", topics: str = "", limit: int = 5) -> str:
    """
    Retrieves recent financial news articles and market sentiment data.

    Use this tool when the user asks about:
    - Recent news about a specific stock or company (use ticker parameter)
    - Market news about a sector or theme (use topics parameter)
    - Current market sentiment and what is driving price movements

    Args:
        ticker: Stock ticker symbol(s) for company-specific news.
                Examples: "AAPL", "MSFT", "AAPL,MSFT,GOOGL"
                Leave empty if searching by topic instead.
        topics: Financial topic(s) for broader market news.
                Valid values: financial_markets, economy_macro, economy_fiscal,
                economy_monetary, earnings, ipo, mergers_and_acquisitions,
                technology, energy_transportation, finance, real_estate,
                retail_wholesale, manufacturing, life_sciences, blockchain.
                Leave empty if searching by ticker instead.
        limit: Number of news articles to retrieve (default: 5, max: 50).

    Returns:
        Formatted summary of recent financial news articles with sentiment data.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: Alpha Vantage API key not configured."

    params = {
        "function": "NEWS_SENTIMENT",
        "limit": min(int(limit), 50),
        "sort": "LATEST",
        "apikey": api_key,
    }

    if ticker:
        params["tickers"] = ticker.upper().strip()
        search_desc = f"ticker(s): {ticker.upper()}"
    elif topics:
        # Validate and clean topics
        requested = [t.strip().lower() for t in topics.split(",")]
        valid = [t for t in requested if t in _VALID_TOPICS]
        if not valid:
            valid = ["financial_markets"]
        params["topics"] = ",".join(valid)
        search_desc = f"topic(s): {', '.join(valid)}"
    else:
        params["topics"] = "financial_markets"
        search_desc = "general financial markets"

    _logs.debug(f"Fetching Alpha Vantage news for {search_desc} (limit={limit})")

    try:
        response = requests.get(_AV_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        _logs.error(f"Alpha Vantage API request failed: {e}")
        return f"Unable to retrieve news at this time. Error: {str(e)}"

    # Check for API error messages
    if "Note" in data:
        _logs.warning(f"Alpha Vantage rate limit note: {data['Note']}")
        return ("Alpha Vantage API rate limit reached. The free tier allows 25 requests/day "
                "and 5 requests/minute. Please try again shortly.")

    if "Information" in data:
        return f"Alpha Vantage API message: {data['Information']}"

    feed = data.get("feed", [])
    if not feed:
        return f"No recent news found for {search_desc}."

    items_returned = int(data.get("items", len(feed)))
    _logs.info(f"Retrieved {items_returned} news articles for {search_desc}")

    # Format articles
    formatted = [f"## Recent Financial News: {search_desc.title()}\n"]
    for i, article in enumerate(feed[:limit], 1):
        formatted.append(f"### Article {i}\n{_format_article(article)}")

    # Overall sentiment summary if multiple articles
    if len(feed) > 1:
        scores = [a.get("overall_sentiment_score", 0.0) for a in feed[:limit]]
        avg_sentiment = sum(scores) / len(scores)
        if avg_sentiment > 0.15:
            overall = "net Bullish"
        elif avg_sentiment < -0.15:
            overall = "net Bearish"
        else:
            overall = "broadly Neutral"
        formatted.append(
            f"\n**Overall Sentiment**: {overall} "
            f"(average score: {avg_sentiment:.3f} across {len(scores)} articles)"
        )

    return "\n".join(formatted)
