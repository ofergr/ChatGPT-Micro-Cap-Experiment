"""
Fundamentals and news-sentiment enrichment for the trading prompt.

Pulls per-ticker fundamentals (P/E, ROE, revenue growth, analyst target, ...)
and recent news headlines from Alpha Vantage's OVERVIEW and NEWS_SENTIMENT
endpoints. Sentiment labels come from Alpha Vantage's own server-side NLP --
no LLM/AI API call is made here, only the ALPHA_VANTAGE_API_KEY already used
for market data.

Standalone usage (for the manual copy-paste-into-ChatGPT workflow):
    python market_enrichment.py AAPL MSFT NVDA
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.alphaintelligence import AlphaIntelligence

load_dotenv()

# Alpha Vantage's free tier caps requests at 5/minute; we make 2 calls per
# ticker (fundamentals + news), so pacing calls >=12s apart stays under that.
_DEFAULT_CALL_DELAY = 12.5


def _is_rate_limited(data: object) -> bool:
    return isinstance(data, dict) and (
        "Note" in data or "Information" in data or "Error Message" in data
    )


def get_fundamentals_summary(ticker: str, api_key: str) -> Optional[str]:
    """Fetch key fundamentals for `ticker`. Returns None if unavailable."""
    try:
        fd = FundamentalData(key=api_key, output_format="json")
        data, _ = fd.get_company_overview(symbol=ticker)
    except Exception:
        return None

    if _is_rate_limited(data) or not data or not data.get("Symbol"):
        return None

    def fmt(key: str) -> str:
        val = data.get(key)
        return "n/a" if val in (None, "", "None") else str(val)

    return (
        f"{ticker}: P/E {fmt('PERatio')}, PEG {fmt('PEGRatio')}, "
        f"ROE {fmt('ReturnOnEquityTTM')}, Rev Growth YoY {fmt('QuarterlyRevenueGrowthYOY')}, "
        f"Profit Margin {fmt('ProfitMargin')}, Analyst Target ${fmt('AnalystTargetPrice')}, "
        f"52wk Range ${fmt('52WeekLow')}-${fmt('52WeekHigh')}"
    )


def get_news_sentiment_summary(ticker: str, api_key: str, limit: int = 5) -> Optional[str]:
    """Fetch recent headlines + Alpha Vantage's own sentiment scoring for `ticker`.

    Alpha Vantage's `tickers`/`limit` query params aren't reliably honored on
    the free tier -- it can return a generic ~50-article feed regardless. We
    filter to articles that actually tag `ticker` in `ticker_sentiment` and
    sort by that ticker's relevance_score, so at least the most-relevant
    available articles surface first.
    """
    try:
        ai = AlphaIntelligence(key=api_key, output_format="json")
        articles, _ = ai.get_news_sentiment(tickers=ticker, limit=limit)
    except Exception:
        return None

    if _is_rate_limited(articles):
        return None

    if hasattr(articles, "to_dict"):  # the client sometimes returns a DataFrame despite output_format="json"
        articles = articles.to_dict("records")
    if not articles:
        return None

    scored: list[tuple[float, dict, dict]] = []
    for article in articles:
        ticker_sentiment = next(
            (t for t in article.get("ticker_sentiment", []) if t.get("ticker") == ticker),
            None,
        )
        if not ticker_sentiment:
            continue
        try:
            relevance = float(ticker_sentiment.get("relevance_score", 0))
        except (TypeError, ValueError):
            relevance = 0.0
        scored.append((relevance, article, ticker_sentiment))
    scored.sort(key=lambda item: item[0], reverse=True)

    lines = []
    for _, article, ticker_sentiment in scored[:limit]:
        title = str(article.get("title", "")).strip()
        if not title:
            continue
        label = ticker_sentiment.get("ticker_sentiment_label", "n/a")
        source = article.get("source", "")
        lines.append(f"  - [{label}] {title} ({source})")

    if not lines:
        return None
    return f"{ticker} recent news sentiment:\n" + "\n".join(lines)


def build_market_enrichment(
    tickers: list[str],
    api_key: Optional[str],
    delay: float = _DEFAULT_CALL_DELAY,
) -> str:
    """
    Build a text block of fundamentals + news sentiment for each ticker.
    Returns "" if no API key is configured, or if every call failed/was
    rate-limited -- callers should treat that as "skip enrichment", not
    an error.
    """
    if not api_key:
        return ""

    sections = []
    first_call = True
    for i, ticker in enumerate(tickers, start=1):
        ticker_sections = []
        for fetch in (get_fundamentals_summary, get_news_sentiment_summary):
            if not first_call:
                time.sleep(delay)
            first_call = False
            result = fetch(ticker, api_key)
            if result:
                ticker_sections.append(result)
        status = "retrieved" if ticker_sections else "no data available"
        print(f"  [{i}/{len(tickers)}] {ticker}: {status}")
        sections.extend(ticker_sections)

    if not sections:
        return ""
    return "[ Market Data (Alpha Vantage) ]\n" + "\n".join(sections)


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        print("Usage: python market_enrichment.py TICKER [TICKER ...]")
        sys.exit(1)

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("Error: ALPHA_VANTAGE_API_KEY not set (see .env.example)")
        sys.exit(1)

    block = build_market_enrichment(tickers, api_key)
    print(block if block else "No market data available (rate-limited or no data for these tickers).")


if __name__ == "__main__":
    main()
