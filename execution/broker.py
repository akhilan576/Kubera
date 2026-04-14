from __future__ import annotations
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TrailingStopOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

# Paper trading if base URL contains "paper"
_IS_PAPER = "paper" in ALPACA_BASE_URL


class Broker:
    """
    Thin wrapper around the Alpaca TradingClient.
    Handles order placement, cancellation, and account queries.
    """

    def __init__(self):
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=_IS_PAPER,
        )
        mode = "PAPER" if _IS_PAPER else "LIVE"
        logger.info(f"Broker initialised in {mode} mode")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self):
        """Return Alpaca account object."""
        return self.client.get_account()

    def get_portfolio_value(self) -> float:
        account = self.get_account()
        return float(account.portfolio_value)

    def get_cash(self) -> float:
        account = self.get_account()
        return float(account.cash)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> dict:
        """Return {symbol: position_object} for all open positions."""
        positions = self.client.get_all_positions()
        return {p.symbol: p for p in positions}

    def get_position(self, symbol: str):
        """Return position for a single symbol, or None if not held."""
        try:
            return self.client.get_open_position(symbol)
        except Exception:
            return None

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def market_buy(self, symbol: str, qty: int, stop_loss: float | None = None,
                   take_profit: float | None = None):
        """
        Submit a market buy order, optionally with bracket legs.
        """
        if qty <= 0:
            logger.warning(f"Skipping buy for {symbol}: qty={qty}")
            return None

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            stop_loss=StopLossRequest(stop_price=stop_loss) if stop_loss else None,
            take_profit=TakeProfitRequest(limit_price=take_profit) if take_profit else None,
        )

        try:
            order = self.client.submit_order(request)
            logger.info(f"BUY submitted: {symbol} x{qty} | order_id={order.id}")
            return order
        except Exception as e:
            logger.error(f"BUY failed for {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, qty: int | None = None):
        """
        Submit a market sell order.
        If qty is None, close the entire position.
        """
        if qty is not None and qty <= 0:
            logger.warning(f"Skipping sell for {symbol}: qty={qty}")
            return None

        try:
            if qty is None:
                # Close full position
                order = self.client.close_position(symbol)
                logger.info(f"CLOSE POSITION submitted: {symbol} | order_id={order.id}")
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.client.submit_order(request)
                logger.info(f"SELL submitted: {symbol} x{qty} | order_id={order.id}")
            return order
        except Exception as e:
            logger.error(f"SELL failed for {symbol}: {e}")
            return None

    def trailing_stop_sell(self, symbol: str, qty: int, trail_percent: float):
        """
        Submit a trailing stop sell order.
        trail_percent: e.g. 1.5 means trail by 1.5% below the high-water mark.
        Alpaca manages the trailing automatically once submitted.
        """
        if qty <= 0:
            logger.warning(f"Skipping trailing stop for {symbol}: qty={qty}")
            return None
        try:
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=round(trail_percent, 2),
            )
            order = self.client.submit_order(request)
            logger.info(f"TRAILING STOP submitted: {symbol} x{qty} "
                        f"trail={trail_percent:.1f}% | order_id={order.id}")
            return order
        except Exception as e:
            logger.error(f"TRAILING STOP failed for {symbol}: {e}")
            return None

    def cancel_symbol_orders(self, symbol: str, wait_secs: float = 3.0):
        """Cancel all open orders for a symbol and wait until they are gone."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            open_orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
            cancelled = [o for o in open_orders if o.symbol == symbol]
            for order in cancelled:
                self.client.cancel_order_by_id(order.id)
                logger.info(f"Cancelled open order {order.id} for {symbol}")
            if not cancelled:
                return
            # Wait for Alpaca to process the cancellations before caller submits a sell
            deadline = time.monotonic() + wait_secs
            while time.monotonic() < deadline:
                time.sleep(0.5)
                remaining = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
                if not any(o.symbol == symbol for o in remaining):
                    return
            logger.warning(f"Orders for {symbol} may not be fully cancelled after {wait_secs}s")
        except Exception as e:
            logger.warning(f"Could not cancel orders for {symbol}: {e}")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            self.client.cancel_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    def close_all_positions(self):
        """Emergency: flatten everything."""
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.warning("All positions closed (emergency flatten)")
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")

    # ------------------------------------------------------------------
    # Order status
    # ------------------------------------------------------------------

    def get_open_orders(self) -> list:
        from alpaca.trading.requests import GetOrdersRequest
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return self.client.get_orders(filter=request)
