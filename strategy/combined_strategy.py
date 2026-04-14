"""
Combined strategy: XGBoost price prediction + FinBERT sentiment filter.

Signal logic:
    BUY  → XGB says P(up) >= buy_threshold  AND  sentiment >= sentiment_floor
    SELL → XGB says P(up) <= sell_threshold  OR   sentiment <= sentiment_ceiling (bearish news)
    HOLD → anything else

Sentiment scores are in [-1.0, +1.0]:
    sentiment_floor   = minimum sentiment required to allow a BUY  (default: -0.1)
    sentiment_ceiling = maximum sentiment allowed to suppress SELL  (default: -0.3)
"""

from __future__ import annotations
import pandas as pd
from strategy.base import BaseStrategy, TradeSignal, Signal
from strategy.xgb_strategy import XGBStrategy
from data.news import NewsFetcher
from models.sentiment import SentimentAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)


class CombinedStrategy(BaseStrategy):
    """
    Layers FinBERT news sentiment on top of XGBoost price signals.

    A BUY is only placed when:
      - XGBoost predicts the price will rise (P(up) >= buy_threshold)
      - Recent news sentiment is not significantly negative (>= sentiment_floor)

    A SELL is triggered when:
      - XGBoost predicts the price will fall  OR
      - News sentiment is strongly negative (<= sentiment_ceiling)
    """

    def __init__(
        self,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        sentiment_floor: float = -0.10,
        sentiment_ceiling: float = -0.30,
        auc_threshold: float = 0.50,
        sentiment_buy_floor: float = 0.20,   # min sentiment for a sentiment-only BUY
        news_hours_back: int = 24,
        max_news_per_symbol: int = 10,
    ):
        super().__init__("XGBoost + FinBERT Combined")
        self.sentiment_floor = sentiment_floor
        self.sentiment_ceiling = sentiment_ceiling
        self.sentiment_buy_floor = sentiment_buy_floor
        self.news_hours_back = news_hours_back
        self.max_news_per_symbol = max_news_per_symbol

        self._xgb = XGBStrategy(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            auc_threshold=auc_threshold,
        )
        self._news_fetcher = NewsFetcher()
        self._sentiment = SentimentAnalyzer()

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        symbols = list(bars.keys())

        # --- Step 1: XGBoost price signals ---
        xgb_signals = {s.symbol: s for s in self._xgb.generate_signals(bars)}

        # --- Step 2: Fetch news and score sentiment ---
        sentiment_scores = self._get_sentiment_scores(symbols)

        # --- Step 3: Combine ---
        final_signals: list[TradeSignal] = []

        for symbol in symbols:
            xgb_signal = xgb_signals.get(symbol)
            sentiment = sentiment_scores.get(symbol, 0.0)

            logger.info(f"{symbol}: XGB={xgb_signal.signal.value if xgb_signal else 'NONE'} | "
                        f"sentiment={sentiment:+.3f}")

            if xgb_signal and xgb_signal.signal == Signal.BUY:
                if sentiment >= self.sentiment_floor:
                    # Boost confidence when sentiment agrees
                    sentiment_boost = max(0.0, sentiment) * 0.2
                    confidence = min(1.0, xgb_signal.confidence + sentiment_boost)
                    final_signals.append(TradeSignal(
                        symbol=symbol,
                        signal=Signal.BUY,
                        confidence=round(confidence, 3),
                        reason=f"{xgb_signal.reason} | sentiment={sentiment:+.3f}",
                    ))
                    logger.info(f"BUY confirmed: {symbol} | confidence={confidence:.3f}")
                else:
                    logger.info(
                        f"BUY blocked by sentiment: {symbol} | "
                        f"sentiment={sentiment:+.3f} < floor={self.sentiment_floor}"
                    )

            elif xgb_signal and xgb_signal.signal == Signal.SELL:
                final_signals.append(xgb_signal)

            elif sentiment >= self.sentiment_buy_floor:
                # XGB gated out but sentiment is clearly positive → BUY
                final_signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    confidence=round(sentiment, 3),
                    reason=f"Sentiment-only BUY (XGB gated) sentiment={sentiment:+.3f}",
                ))
                logger.info(f"BUY (sentiment-only): {symbol} | sentiment={sentiment:+.3f}")

            elif sentiment <= self.sentiment_ceiling:
                # Strongly negative news — SELL regardless of XGB
                final_signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    confidence=round(abs(sentiment), 3),
                    reason=f"Sentiment-only SELL sentiment={sentiment:+.3f}",
                ))
                logger.info(f"SELL (sentiment-only): {symbol} | sentiment={sentiment:+.3f}")

        return final_signals

    def _get_sentiment_scores(self, symbols: list[str]) -> dict[str, float]:
        """Fetch news and return {symbol: sentiment_score}."""
        scores = {}
        try:
            news_map = self._news_fetcher.get_recent_news(
                symbols=symbols,
                hours_back=self.news_hours_back,
                max_per_symbol=self.max_news_per_symbol,
            )
            for symbol, articles in news_map.items():
                if articles:
                    texts = [a.text for a in articles]
                    scores[symbol] = self._sentiment.score_symbol(texts)
                    logger.debug(f"{symbol}: {len(texts)} articles → sentiment={scores[symbol]:+.3f}")
                else:
                    scores[symbol] = 0.0
                    logger.debug(f"{symbol}: no recent news, sentiment=0.0")
        except Exception as e:
            logger.error(f"Sentiment scoring failed: {e}")
        return scores
