from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, BAR_TIMEFRAME, LOOKBACK_BARS
from utils.logger import get_logger

logger = get_logger(__name__)

_TIMEFRAME_MAP = {
    "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
    "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
}


class DataFetcher:
    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.timeframe = _TIMEFRAME_MAP.get(BAR_TIMEFRAME, TimeFrame(1, TimeFrameUnit.Day))

    def get_bars(self, symbols: list[str], lookback: int = LOOKBACK_BARS) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for a list of symbols.
        Returns a dict of {symbol: DataFrame} with columns:
            open, high, low, close, volume
        """
        end = datetime.utcnow()
        # Calculate calendar days needed based on timeframe
        # (1.4x multiplier accounts for weekends/holidays)
        bars_per_day = {
            "1Min": 390, "5Min": 78, "15Min": 26, "1Hour": 7, "1Day": 1
        }.get(BAR_TIMEFRAME, 1)
        calendar_days = max(5, int((lookback / bars_per_day) * 7 / 5 * 1.4))
        start = end - timedelta(days=calendar_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,   # free-tier compatible (SIP requires paid subscription)
        )

        try:
            bars = self.client.get_stock_bars(request)
            result = {}
            for symbol in symbols:
                try:
                    raw = bars[symbol]
                    if not raw:
                        logger.warning(f"{symbol}: no bars returned, skipping")
                        continue
                    df = pd.DataFrame([{
                        "open":   b.open,
                        "high":   b.high,
                        "low":    b.low,
                        "close":  b.close,
                        "volume": b.volume,
                    } for b in raw], index=pd.to_datetime([b.timestamp for b in raw]))
                    df.index = df.index.tz_localize(None)
                    df.sort_index(inplace=True)
                    result[symbol] = df.tail(lookback)
                except (KeyError, Exception) as e:
                    logger.warning(f"{symbol}: skipped — {e}")
                    continue
            return result
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbols}: {e}")
            return {}

    def get_latest_price(self, symbol: str) -> float | None:
        """Return the most recent close price for a symbol."""
        bars = self.get_bars([symbol], lookback=1)
        if symbol in bars and not bars[symbol].empty:
            return float(bars[symbol]["close"].iloc[-1])
        return None
