"""
Angel One SmartAPI fetcher for NSE stocks.
Drop-in replacement for YFinanceFetcher — same get_bars() interface.
Provides real-time data (vs 15-min delayed from yfinance).

Usage:
    fetcher = AngelFetcher()
    bars = fetcher.get_bars(["RELIANCE", "SBIN"], lookback=200)
"""
from __future__ import annotations

import time
import pyotp
import pandas as pd
from datetime import datetime, timedelta
import pytz

from SmartApi import SmartConnect
from config.india_settings import (
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
    INDIA_BAR_TIMEFRAME,
)
from utils.logger import get_logger

logger = get_logger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Angel One interval strings
_INTERVAL_MAP = {
    "1Min":  "ONE_MINUTE",
    "3Min":  "THREE_MINUTE",
    "5Min":  "FIVE_MINUTE",
    "10Min": "TEN_MINUTE",
    "15Min": "FIFTEEN_MINUTE",
    "30Min": "THIRTY_MINUTE",
    "1Hour": "ONE_HOUR",
    "1Day":  "ONE_DAY",
}

# How many days of history to fetch per interval (Angel One limits)
_LOOKBACK_DAYS = {
    "ONE_MINUTE":     30,
    "THREE_MINUTE":   60,
    "FIVE_MINUTE":    100,
    "TEN_MINUTE":     100,
    "FIFTEEN_MINUTE": 200,
    "THIRTY_MINUTE":  200,
    "ONE_HOUR":       400,
    "ONE_DAY":        2000,
}

# Cache for symbol tokens so we don't search on every tick
_TOKEN_CACHE: dict[str, str] = {}


class AngelFetcher:
    """Fetch NSE OHLCV data via Angel One SmartAPI. Requires active account."""

    def __init__(self, interval: str = "15Min"):
        self.interval = _INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE")
        self.lookback_days = _LOOKBACK_DAYS.get(self.interval, 200)
        self._api = self._login()

    def _login(self) -> SmartConnect:
        """Authenticate with Angel One and return a live SmartConnect session."""
        api = SmartConnect(api_key=ANGEL_API_KEY)
        totp_code = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        session = api.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp_code)
        if not session.get("status"):
            raise RuntimeError(f"Angel One login failed: {session.get('message', 'unknown error')}")
        logger.info(f"AngelFetcher: logged in as {ANGEL_CLIENT_ID}")
        return api

    def _get_token(self, symbol: str) -> str | None:
        """Look up the Angel One symbol token for an NSE symbol."""
        if symbol in _TOKEN_CACHE:
            return _TOKEN_CACHE[symbol]
        try:
            result = self._api.searchScrip("NSE", symbol)
            if result.get("status") and result.get("data"):
                # Pick the EQ (equity) scrip, not futures/options
                for item in result["data"]:
                    if item.get("tradingsymbol") == symbol or item.get("tradingsymbol") == f"{symbol}-EQ":
                        token = item["symboltoken"]
                        _TOKEN_CACHE[symbol] = token
                        logger.debug(f"AngelFetcher: {symbol} → token {token}")
                        return token
                # Fallback: take first result
                token = result["data"][0]["symboltoken"]
                _TOKEN_CACHE[symbol] = token
                logger.debug(f"AngelFetcher: {symbol} → token {token} (fallback)")
                return token
        except Exception as e:
            logger.error(f"AngelFetcher: token lookup failed for {symbol}: {e}")
        return None

    def get_bars(self, symbols: list[str], lookback: int = 1000) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for NSE symbols.
        Returns {symbol: DataFrame} with columns: open, high, low, close, volume
        Same interface as YFinanceFetcher.
        """
        now_ist  = datetime.now(IST)
        from_dt  = now_ist - timedelta(days=self.lookback_days)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str   = now_ist.strftime("%Y-%m-%d %H:%M")

        result = {}
        for sym in symbols:
            sym = sym.upper().strip()

            # NIFTY index — use yfinance fallback (SmartAPI uses different token for indices)
            if sym.startswith("^"):
                result.update(self._fetch_index_fallback(sym, lookback))
                continue

            token = self._get_token(sym)
            if token is None:
                logger.warning(f"AngelFetcher: no token found for {sym}, skipping")
                continue

            params = {
                "exchange":    "NSE",
                "symboltoken": token,
                "interval":    self.interval,
                "fromdate":    from_str,
                "todate":      to_str,
            }
            try:
                resp = self._api.getCandleData(params)
                if not resp.get("status") or not resp.get("data"):
                    logger.warning(f"AngelFetcher: no data for {sym}: {resp.get('message', '')}")
                    continue

                raw = resp["data"]  # list of [timestamp, open, high, low, close, volume]
                df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
                df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
                df = df.dropna().tail(lookback).reset_index(drop=True)

                if df.empty:
                    continue

                result[sym] = df
                logger.info(f"AngelFetcher: {sym} → {len(df)} bars")
                time.sleep(0.1)  # gentle rate limiting

            except Exception as e:
                logger.error(f"AngelFetcher: failed for {sym}: {e}")

        return result

    def get_day_gainers(self, symbols: list[str], top_n: int = 5) -> None:
        """
        Fetch today's top gainers and losers from the watchlist and log them.
        Uses daily bars (prev close vs today's close).
        """
        bars = self.get_bars(symbols, lookback=2)
        moves = []
        for sym, df in bars.items():
            if len(df) >= 2:
                prev  = float(df["close"].iloc[-2])
                today = float(df["close"].iloc[-1])
                pct   = (today - prev) / prev * 100
                moves.append((sym, round(pct, 2), round(today, 2)))

        if not moves:
            return

        moves.sort(key=lambda x: x[1], reverse=True)

        logger.info("── Today's Top Gainers (watchlist) ──")
        for sym, pct, price in moves[:top_n]:
            logger.info(f"  {sym:15} {pct:+.2f}%  ₹{price:,.2f}")

        logger.info("── Today's Top Losers (watchlist) ──")
        for sym, pct, price in moves[-top_n:][::-1]:
            logger.info(f"  {sym:15} {pct:+.2f}%  ₹{price:,.2f}")

    def _fetch_index_fallback(self, symbol: str, lookback: int) -> dict[str, pd.DataFrame]:
        """Fall back to yfinance for index symbols like ^NSEI."""
        try:
            from data.yfinance_fetcher import YFinanceFetcher
            yf = YFinanceFetcher(interval="15Min")
            return yf.get_bars([symbol], lookback=lookback)
        except Exception as e:
            logger.error(f"AngelFetcher: index fallback failed for {symbol}: {e}")
            return {}
