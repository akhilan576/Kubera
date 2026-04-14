"""
Crypto data fetcher using Binance public API for bar data.
Execution still goes through Alpaca — this is data-only.
Crypto trades 24/7 — no market hours restriction.

Supported symbols (Alpaca format): BTC/USD, ETH/USD, SOL/USD, etc.
"""
from __future__ import annotations

import requests
import pandas as pd
from datetime import datetime, timedelta
from config.settings import BAR_TIMEFRAME, LOOKBACK_BARS
from utils.logger import get_logger

logger = get_logger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

_TIMEFRAME_MAP = {
    "1Min":  "1m",
    "5Min":  "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "1Day":  "1d",
}

# Canonical Alpaca crypto symbols
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]


def normalise(symbol: str) -> str:
    """BTC/USD → BTCUSD for use as dict keys."""
    return symbol.replace("/", "")


def alpaca_symbol(symbol: str) -> str:
    """BTCUSD → BTC/USD for Alpaca trading calls."""
    if "/" in symbol:
        return symbol
    return symbol[:-3] + "/" + symbol[-3:]


def binance_symbol(symbol: str) -> str:
    """BTC/USD or BTCUSD → BTCUSDT for Binance API."""
    base = symbol.replace("/", "").replace("USD", "")
    return f"{base}USDT"


class CryptoFetcher:
    def __init__(self, lookback: int = LOOKBACK_BARS):
        self.interval = _TIMEFRAME_MAP.get(BAR_TIMEFRAME, "15m")
        self._default_lookback = lookback

    def get_bars(self, symbols: list[str],
                 lookback: int | None = None) -> dict[str, pd.DataFrame]:
        """
        Fetch historical crypto bars from Binance.
        Accepts both 'BTC/USD' and 'BTCUSD' formats.
        Returns dict keyed by normalised symbol (e.g. 'BTCUSD').
        """
        lookback = lookback if lookback is not None else self._default_lookback
        result = {}

        for symbol in symbols:
            key = normalise(symbol)
            bsym = binance_symbol(symbol)
            try:
                resp = requests.get(
                    BINANCE_KLINES_URL,
                    params={
                        "symbol":   bsym,
                        "interval": self.interval,
                        "limit":    min(lookback, 1000),
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                raw = resp.json()

                if not raw:
                    logger.warning(f"{symbol}: no data from Binance")
                    continue

                df = pd.DataFrame([{
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]),
                } for k in raw], index=pd.to_datetime([k[0] for k in raw], unit="ms"))

                df.sort_index(inplace=True)
                result[key] = df
                logger.debug(f"{symbol}: {len(df)} bars from Binance")

            except Exception as e:
                logger.error(f"{symbol}: Binance fetch failed — {e}")

        return result

    def get_latest_price(self, symbol: str) -> float | None:
        bars = self.get_bars([symbol], lookback=1)
        key = normalise(symbol)
        if key in bars and not bars[key].empty:
            return float(bars[key]["close"].iloc[-1])
        return None
