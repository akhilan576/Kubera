"""
Yahoo Finance fetcher for NSE stocks.
Drop-in replacement for AngelFetcher — same get_bars() interface.
Uses .NS suffix for NSE symbols (e.g. RELIANCE → RELIANCE.NS).
15-minute delayed data, free, no API key required.
"""
from __future__ import annotations
import yfinance as yf
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

_INTERVAL_MAP = {
    "1Min":  "1m",
    "5Min":  "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "1Day":  "1d",
}

# yfinance period strings that give enough bars per interval
_PERIOD_MAP = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "1h":  "730d",
    "1d":  "5y",
}


class YFinanceFetcher:
    """Fetch NSE OHLCV data via Yahoo Finance. No auth required."""

    def __init__(self, interval: str = "15Min"):
        self.interval = _INTERVAL_MAP.get(interval, "15m")
        self.period   = _PERIOD_MAP.get(self.interval, "60d")
        logger.info(f"YFinanceFetcher ready | interval={self.interval} | period={self.period}")

    def get_bars(self, symbols: list[str], lookback: int = 1000) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for NSE symbols.
        Returns {symbol: DataFrame} with columns: open, high, low, close, volume
        Same interface as AngelFetcher and DataFetcher.
        """
        result = {}
        for sym in symbols:
            sym = sym.upper().strip()
            # Index symbols (^NSEI, ^NSEBANK) already have full Yahoo tickers — don't add .NS
            ticker = sym if sym.startswith("^") else f"{sym}.NS"
            try:
                raw = yf.download(
                    ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False,
                    auto_adjust=True,
                )
                if raw.empty:
                    logger.warning(f"YFinanceFetcher: no data for {ticker}")
                    continue

                df = pd.DataFrame({
                    "open":   raw["Open"].values.flatten(),
                    "high":   raw["High"].values.flatten(),
                    "low":    raw["Low"].values.flatten(),
                    "close":  raw["Close"].values.flatten(),
                    "volume": raw["Volume"].values.flatten(),
                })
                df = df.dropna().tail(lookback).reset_index(drop=True)

                if df.empty:
                    continue

                result[sym] = df
                logger.info(f"YFinanceFetcher: {sym} → {len(df)} bars")
            except Exception as e:
                logger.error(f"YFinanceFetcher: failed for {sym}: {e}")
        return result
