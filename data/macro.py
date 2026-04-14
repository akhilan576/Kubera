"""
Macro data fetcher — VIX, TNX (10Y yield), DXY (dollar index).

These are fetched as regular bar symbols via Alpaca:
  VIX  → VIXY  (VIX ETF proxy — actual VIX index not available via Alpaca)
  TNX  → TLT   (20Y Treasury ETF — inverse proxy for yields)
  DXY  → UUP   (Dollar ETF proxy)

Alternatively yfinance is used as fallback for direct macro indices.
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

# Alpaca-tradeable proxies for macro indices
_MACRO_PROXIES = {
    "vix": "VIXY",   # ProShares VIX Short-Term Futures ETF
    "tnx": "TLT",    # iShares 20+ Year Treasury Bond ETF
    "dxy": "UUP",    # Invesco DB US Dollar Index Bullish Fund
}


class MacroFetcher:
    """
    Fetches macro indicator data and returns a unified DataFrame
    with columns: vix, tnx, dxy — indexed by datetime.
    """

    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    def get_macro(self, lookback: int = 300) -> pd.DataFrame:
        """
        Fetch macro proxy bars and return a combined DataFrame.
        Columns: vix, tnx, dxy (close prices of respective ETFs).
        """
        end   = datetime.utcnow()
        start = end - timedelta(days=lookback * 2)

        symbols = list(_MACRO_PROXIES.values())

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            limit=lookback,
        )

        result = pd.DataFrame()
        try:
            bars = self.client.get_stock_bars(request)
            frames = {}
            for key, sym in _MACRO_PROXIES.items():
                try:
                    df  = bars[sym].df
                    col = df["close"].copy()
                    col.index = pd.to_datetime(col.index)
                    # Remove timezone info for consistent joining
                    col.index = col.index.tz_localize(None) if col.index.tz is None else col.index.tz_convert(None)
                    frames[key] = col
                except KeyError:
                    logger.warning(f"Macro proxy {sym} ({key}) not available")

            if frames:
                result = pd.DataFrame(frames)
                result.sort_index(inplace=True)
                result.ffill(inplace=True)
                logger.debug(f"Macro data: {len(result)} rows, cols={list(result.columns)}")

        except Exception as e:
            logger.warning(f"Macro fetch failed: {e} — macro features will be skipped")

        return result
