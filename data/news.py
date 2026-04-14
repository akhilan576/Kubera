"""
Fetches recent news headlines for symbols via the Alpaca News REST API.
Uses direct HTTP calls instead of the SDK (SDK's NewsRequest is broken for multi-symbol).
"""

from __future__ import annotations
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


@dataclass
class NewsItem:
    symbol: str
    headline: str
    summary: str
    published_at: datetime

    @property
    def text(self) -> str:
        """Combined text for sentiment scoring."""
        return f"{self.headline}. {self.summary}".strip()


class NewsFetcher:
    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }

    def get_recent_news(
        self,
        symbols: list[str],
        hours_back: int = 24,
        max_per_symbol: int = 10,
    ) -> dict[str, list[NewsItem]]:
        """
        Fetch recent news for a list of symbols in one batch call.
        Returns {symbol: [NewsItem, ...]} sorted newest-first.
        """
        end   = datetime.utcnow()
        start = end - timedelta(hours=hours_back)

        result: dict[str, list[NewsItem]] = {s: [] for s in symbols}

        try:
            resp = requests.get(
                NEWS_URL,
                headers=self.headers,
                params={
                    "symbols": ",".join(symbols),
                    "start":   start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end":     end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit":   min(max_per_symbol * len(symbols), 50),
                    "sort":    "desc",
                    "include_content": "false",
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("news", [])

            for article in articles:
                for sym in (article.get("symbols") or []):
                    if sym in result:
                        result[sym].append(NewsItem(
                            symbol=sym,
                            headline=article.get("headline", ""),
                            summary=article.get("summary", ""),
                            published_at=datetime.fromisoformat(
                                article["created_at"].replace("Z", "+00:00")
                            ),
                        ))

            total = sum(len(v) for v in result.values())
            logger.debug(f"News batch: {len(articles)} articles → {total} mapped")

        except Exception as e:
            logger.warning(f"News fetch failed — {e}")

        return result
