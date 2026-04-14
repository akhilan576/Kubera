"""
Crypto broker — handles Alpaca crypto order placement.
Crypto supports fractional quantities, no PDT rules, trades 24/7.
"""
from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from data.crypto_fetcher import alpaca_symbol, normalise
from utils.logger import get_logger

logger = get_logger(__name__)

_IS_PAPER = "paper" in ALPACA_BASE_URL


class CryptoBroker:
    """
    Handles crypto order execution via Alpaca.
    Uses notional (USD) sizing instead of share qty — crypto is fractional.
    """

    def __init__(self):
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=_IS_PAPER,
        )
        mode = "PAPER" if _IS_PAPER else "LIVE"
        logger.info(f"CryptoBroker initialised in {mode} mode")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_portfolio_value(self) -> float:
        return float(self.client.get_account().portfolio_value)

    def get_cash(self) -> float:
        return float(self.client.get_account().cash)

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> dict:
        """Return {normalised_symbol: position} for all open crypto positions."""
        all_positions = self.client.get_all_positions()
        return {
            normalise(p.symbol): p
            for p in all_positions
            if p.asset_class == AssetClass.CRYPTO
        }

    def has_position(self, symbol: str) -> bool:
        return normalise(symbol) in self.get_positions()

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy_notional(self, symbol: str, usd_amount: float):
        """
        Buy `usd_amount` worth of crypto (fractional).
        E.g. buy_notional('BTC/USD', 1000) buys $1000 of BTC.
        """
        if usd_amount < 1:
            logger.warning(f"Skipping crypto BUY for {symbol}: amount=${usd_amount:.2f} too small")
            return None

        sym = alpaca_symbol(symbol)
        request = MarketOrderRequest(
            symbol=sym,
            notional=round(usd_amount, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC,   # crypto uses IOC/GTC, not DAY
        )
        try:
            order = self.client.submit_order(request)
            logger.info(f"CRYPTO BUY: {sym} ${usd_amount:,.2f} | order_id={order.id}")
            return order
        except Exception as e:
            logger.error(f"CRYPTO BUY failed for {sym}: {e}")
            return None

    def sell_position(self, symbol: str):
        """Close entire crypto position for symbol."""
        sym = normalise(symbol)  # close_position needs ETHUSD not ETH/USD
        try:
            order = self.client.close_position(sym)
            logger.info(f"CRYPTO SELL: {sym} | order_id={order.id}")
            return order
        except Exception as e:
            logger.error(f"CRYPTO SELL failed for {sym}: {e}")
            return None

    def close_all(self):
        """Emergency flatten all crypto positions."""
        try:
            positions = self.get_positions()
            for sym in positions:
                self.sell_position(sym)
            logger.warning("All crypto positions closed")
        except Exception as e:
            logger.error(f"Failed to close all crypto positions: {e}")
