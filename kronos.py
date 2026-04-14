"""
Kronos — Standalone Crypto Bot
Trades BTC/ETH/SOL/DOGE/AVAX via Alpaca 24/7, independent of the stock bot.
Run with: python kronos.py
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from config.settings import (
    CRYPTO_SYMBOLS, MAX_CRYPTO_POSITION_USD, MAX_CRYPTO_POSITIONS, ALPACA_BASE_URL,
)
from data.crypto_fetcher import CryptoFetcher
from execution.crypto_broker import CryptoBroker
from strategy.xgb_strategy import XGBStrategy
from strategy.base import Signal
from alerts.telegram import Alerter
from db.logger import init_db, log_signal, log_order
from utils.logger import get_logger
from utils.lock import acquire as lock_acquire, release as lock_release

logger = get_logger(__name__)

CRYPTO_INTERVAL_SECONDS = 300   # poll every 5 minutes to avoid rate limits
_IS_PAPER = "paper" in ALPACA_BASE_URL


def run_crypto(fetcher: CryptoFetcher, strategy: XGBStrategy,
               broker: CryptoBroker, alerter: Alerter) -> None:
    logger.info("── Crypto tick ──")

    bars = fetcher.get_bars(CRYPTO_SYMBOLS)
    if not bars:
        logger.warning("No crypto bar data, skipping tick")
        return

    signals   = strategy.generate_signals(bars)
    positions = broker.get_positions()
    open_count = len(positions)
    cash       = broker.get_cash()

    logger.info(f"[CRYPTO] Cash=${cash:,.2f} | Positions={open_count} | Signals={len(signals)}")

    for signal in signals:
        symbol     = signal.symbol
        price_data = bars.get(symbol)
        if price_data is None or price_data.empty:
            continue
        price = float(price_data["close"].iloc[-1])

        log_signal(f"CRYPTO:{symbol}", signal.signal.value, signal.confidence, signal.reason)

        if signal.signal == Signal.BUY:
            if symbol in positions:
                continue
            if open_count >= MAX_CRYPTO_POSITIONS:
                logger.warning(f"[CRYPTO] Max positions reached, skipping {symbol}")
                continue
            usd_amount = min(
                MAX_CRYPTO_POSITION_USD * signal.confidence,
                cash * 0.30,
            )
            if usd_amount >= 1:
                order    = broker.buy_notional(symbol, usd_amount)
                order_id = str(order.id) if order else None
                log_order(f"CRYPTO:{symbol}", "buy", 0, order_id, "submitted", price)
                alerter.trade_opened(f"CRYPTO:{symbol}", "BUY", 0, price, signal.confidence)
                open_count += 1

        elif signal.signal == Signal.SELL:
            if symbol not in positions:
                continue
            pos      = positions[symbol]
            entry    = float(pos.avg_entry_price)
            qty      = float(pos.qty)
            pnl      = (price - entry) * qty
            order    = broker.sell_position(symbol)
            order_id = str(order.id) if order else None
            log_order(f"CRYPTO:{symbol}", "sell", 0, order_id, "submitted", price)
            alerter.trade_closed(f"CRYPTO:{symbol}", "SELL", 0, entry, price, pnl)
            open_count -= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both", action="store_true", help="Allow running alongside Zeno")
    args = parser.parse_args()
    lock_acquire("Kronos", force=args.both)
    logger.info("=== Kronos Crypto Bot starting ===")

    init_db()
    alerter = Alerter()

    fetcher  = CryptoFetcher(lookback=200)
    broker   = CryptoBroker()
    strategy = XGBStrategy(buy_threshold=0.55, sell_threshold=0.45)

    mode = "PAPER" if _IS_PAPER else "LIVE"
    logger.info(
        f"Cash: ${broker.get_cash():,.2f} | "
        f"Portfolio: ${broker.get_portfolio_value():,.2f} | Mode: {mode} | "
        f"Interval: {CRYPTO_INTERVAL_SECONDS}s"
    )
    alerter.bot_started(mode=f"CRYPTO-{mode}")

    try:
        while True:
            try:
                run_crypto(fetcher, strategy, broker, alerter)
            except Exception as e:
                logger.error(f"[CRYPTO] Error: {e}", exc_info=True)
                alerter.error("crypto loop", str(e))
            time.sleep(CRYPTO_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Crypto bot shutting down...")
        alerter.bot_stopped()
        lock_release()


if __name__ == "__main__":
    main()
