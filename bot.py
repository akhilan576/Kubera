"""
Project Zeno — Unified AI Trading Bot
Trades US stocks (Alpaca) + crypto (Alpaca) in a single loop.
Run with: python bot.py
"""
from __future__ import annotations

import argparse
import time
import threading
from datetime import datetime
import pytz

from config.settings import (
    SYMBOLS, CRYPTO_SYMBOLS, CRYPTO_ENABLED,
    MAX_CRYPTO_POSITION_USD, MAX_CRYPTO_POSITIONS,
    RUN_INTERVAL_SECONDS, MARKET_OPEN_TIME, MARKET_CLOSE_TIME, EOD_LIQUIDATE_TIME,
    OPEN_BURST_MINUTES, OPEN_BURST_MAX,
)
from data.fetcher import DataFetcher
from data.crypto_fetcher import CryptoFetcher
from data.screener import DailyScreener
from data.macro import MacroFetcher
from strategy.combined_strategy import CombinedStrategy
from strategy.xgb_strategy import XGBStrategy
from execution.broker import Broker
from execution.crypto_broker import CryptoBroker
from risk.manager import RiskManager
from strategy.base import Signal
from alerts.telegram import Alerter
from db.logger import init_db, log_signal, log_order, log_equity
from utils.logger import get_logger
from utils.lock import acquire as lock_acquire, release as lock_release

logger = get_logger(__name__)

ET = pytz.timezone("America/New_York")

# ── Trailing stop percent (ATR-based, passed to Alpaca) ───────────────────────
TRAILING_STOP_PCT = 1.5   # trail by 1.5% below high-water mark


def is_market_open() -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    open_h,  open_m  = map(int, MARKET_OPEN_TIME.split(":"))
    close_h, close_m = map(int, MARKET_CLOSE_TIME.split(":"))
    open_time  = now_et.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
    close_time = now_et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return open_time <= now_et <= close_time


def is_eod_liquidate_time() -> bool:
    """True when it's time to close all intraday positions before market close."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    liq_h, liq_m = map(int, EOD_LIQUIDATE_TIME.split(":"))
    close_h, close_m = map(int, MARKET_CLOSE_TIME.split(":"))
    liq_time   = now_et.replace(hour=liq_h,   minute=liq_m,   second=0, microsecond=0)
    close_time = now_et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return liq_time <= now_et <= close_time


def liquidate_all_positions(broker: Broker, alerter: Alerter):
    """Close all open stock positions — called at EOD for intraday mode."""
    positions = broker.get_positions()
    if not positions:
        return
    logger.info(f"[EOD] Liquidating {len(positions)} open position(s) before market close")
    for symbol, pos in positions.items():
        try:
            entry = float(pos.avg_entry_price)
            qty   = int(float(pos.qty))
            price = float(pos.current_price)
            pnl   = (price - entry) * qty
            broker.cancel_symbol_orders(symbol)
            order = broker.market_sell(symbol)
            order_id = str(order.id) if order else None
            log_order(symbol, "sell", qty, order_id, "eod-liquidation", price)
            alerter.trade_closed(symbol, "SELL", qty, entry, price, pnl)
            logger.info(f"[EOD] Closed {symbol} | qty={qty} | P&L=${pnl:+.2f}")
        except Exception as e:
            logger.error(f"[EOD] Failed to close {symbol}: {e}")


# ── Stock trading loop ────────────────────────────────────────────────────────

def _load_win_rates(risk: RiskManager):
    """Pull historical win rates from DB fills and update risk manager."""
    try:
        from db.logger import get_recent_trades
        fills = get_recent_trades(limit=500)
        from collections import defaultdict
        wins   = defaultdict(int)
        totals = defaultdict(int)
        # Pair up buys and sells per symbol to count wins
        buys: dict[str, float] = {}
        for f in reversed(fills):
            sym = f["symbol"]
            if f["side"] == "buy":
                buys[sym] = f["fill_price"]
            elif f["side"] == "sell" and sym in buys:
                totals[sym] += 1
                if f["fill_price"] > buys.pop(sym):
                    wins[sym] += 1
        for sym, total in totals.items():
            if total >= 5:  # need at least 5 trades for a meaningful rate
                risk.update_win_rate(sym, wins[sym] / total)
    except Exception as e:
        logger.debug(f"Win rate load skipped: {e}")


def run_stocks(fetcher: DataFetcher, strategy, broker: Broker,
               risk: RiskManager, alerter: Alerter,
               active_symbols: list[str] | None = None,
               macro_fetcher: MacroFetcher | None = None,
               burst_opens_today: list | None = None):
    """One tick of the stock trading loop."""
    logger.info("── Stock tick ──")

    symbols       = active_symbols or SYMBOLS
    fetch_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    bars          = fetcher.get_bars(fetch_symbols)

    # Fetch macro data and inject into strategy for feature enrichment
    macro_df = macro_fetcher.get_macro() if macro_fetcher else None
    if macro_df is not None and not macro_df.empty:
        strategy.macro_df = macro_df
    if not bars:
        logger.warning("No stock bar data, skipping tick")
        return

    # Update market regime from SPY
    spy_df = bars.get("SPY")
    if spy_df is not None:
        risk.update_market_regime(spy_df)

    signals         = strategy.generate_signals(bars)
    portfolio_value = broker.get_portfolio_value()
    cash            = broker.get_cash()
    positions       = broker.get_positions()
    open_count      = len(positions)

    logger.info(f"[STOCKS] Portfolio=${portfolio_value:,.2f} | "
                f"Cash=${cash:,.2f} | Positions={open_count} | "
                f"Signals={len(signals)} | Bear={risk.is_bear_market}")

    log_equity(portfolio_value, cash, open_count)

    # Update trailing stop high-water marks for all open positions
    for sym, pos in positions.items():
        current_price = float(pos.current_price)
        risk.update_trailing_stop(sym, current_price)

        # Check if trailing stop triggered — exit if so
        entry     = float(pos.avg_entry_price)
        price_df  = bars.get(sym)
        if price_df is not None and risk.should_trail_exit(sym, current_price, entry, price_df):
            qty      = int(float(pos.qty))
            pnl      = (current_price - entry) * qty
            broker.cancel_symbol_orders(sym)
            order    = broker.market_sell(sym)
            order_id = str(order.id) if order else None
            log_order(sym, "sell", qty, order_id, "trailing-stop", current_price)
            alerter.trade_closed(sym, "SELL", qty, entry, current_price, pnl)
            risk.clear_trailing(sym)
            open_count -= 1

    for signal in signals:
        symbol     = signal.symbol
        price_data = bars.get(symbol)
        if price_data is None or price_data.empty:
            continue
        price = float(price_data["close"].iloc[-1])

        log_signal(symbol, signal.signal.value, signal.confidence, signal.reason)

        if signal.signal == Signal.BUY:
            if positions.get(symbol):
                continue
            if not risk.can_open_position(open_count):
                continue

            # Burst cap — limit new entries in the first N minutes after open
            if burst_opens_today is not None:
                now_et = datetime.now(ET)
                open_h, open_m = map(int, MARKET_OPEN_TIME.split(":"))
                burst_end = now_et.replace(hour=open_h, minute=open_m + OPEN_BURST_MINUTES, second=0, microsecond=0)
                if now_et <= burst_end and len(burst_opens_today) >= OPEN_BURST_MAX:
                    logger.info(f"[BURST] Skipping {symbol} — burst cap ({OPEN_BURST_MAX}) reached for first {OPEN_BURST_MINUTES}m")
                    continue

            # ATR-based dynamic stops
            stop_price = risk.atr_stop_loss(price_data, price)
            tp_price   = risk.atr_take_profit(price_data, price)

            # Kelly Criterion sizing
            qty = risk.kelly_position_size(
                portfolio_value=portfolio_value,
                price=price,
                confidence=signal.confidence,
                symbol=symbol,
            )
            if qty > 0:
                order    = broker.market_buy(symbol, qty, stop_loss=stop_price, take_profit=tp_price)
                order_id = str(order.id) if order else None
                log_order(symbol, "buy", qty, order_id, "submitted", price)
                alerter.trade_opened(symbol, "BUY", qty, price, signal.confidence)
                # Wait for fill then use actual filled qty for trailing stop
                time.sleep(2)
                filled_qty = qty
                if order_id:
                    try:
                        import uuid
                        filled = broker.client.get_order_by_id(uuid.UUID(order_id))
                        filled_qty = int(float(filled.filled_qty or qty))
                    except Exception:
                        pass
                if filled_qty <= 0:
                    filled_qty = qty
                broker.trailing_stop_sell(symbol, filled_qty, TRAILING_STOP_PCT)
                risk.update_trailing_stop(symbol, price)
                if burst_opens_today is not None:
                    burst_opens_today.append(symbol)
                open_count += 1

        elif signal.signal == Signal.SELL:
            if not positions.get(symbol):
                continue
            pos      = positions[symbol]
            entry    = float(pos.avg_entry_price)
            qty      = int(float(pos.qty))
            pnl      = (price - entry) * qty
            broker.cancel_symbol_orders(symbol)
            order    = broker.market_sell(symbol)
            order_id = str(order.id) if order else None
            log_order(symbol, "sell", qty, order_id, "submitted", price)
            alerter.trade_closed(symbol, "SELL", qty, entry, price, pnl)
            risk.clear_trailing(symbol)
            open_count -= 1


# ── Crypto trading loop ───────────────────────────────────────────────────────

def run_crypto(crypto_fetcher: CryptoFetcher, crypto_strategy,
               crypto_broker: CryptoBroker, alerter: Alerter):
    """One tick of the crypto trading loop — runs 24/7."""
    logger.info("── Crypto tick ──")

    # Fetch bars (use normalised keys: BTCUSD, ETHUSD etc.)
    bars = crypto_fetcher.get_bars(CRYPTO_SYMBOLS)
    if not bars:
        logger.warning("No crypto bar data, skipping tick")
        return

    # Rename keys to match strategy expectations
    signals   = crypto_strategy.generate_signals(bars)
    positions = crypto_broker.get_positions()
    open_count = len(positions)
    cash       = crypto_broker.get_cash()

    logger.info(f"[CRYPTO] Cash=${cash:,.2f} | Positions={open_count} | Signals={len(signals)}")

    for signal in signals:
        symbol = signal.symbol
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
                cash * 0.30,   # never spend more than 30% of cash on one crypto trade
            )
            if usd_amount >= 1:
                order    = crypto_broker.buy_notional(symbol, usd_amount)
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
            order    = crypto_broker.sell_position(symbol)
            order_id = str(order.id) if order else None
            log_order(f"CRYPTO:{symbol}", "sell", 0, order_id, "submitted", price)
            alerter.trade_closed(f"CRYPTO:{symbol}", "SELL", 0, entry, price, pnl)
            open_count -= 1


# ── Crypto background thread ──────────────────────────────────────────────────

def crypto_loop(crypto_fetcher: CryptoFetcher, crypto_strategy,
                crypto_broker: CryptoBroker, alerter: Alerter,
                stop_event: threading.Event):
    """Runs crypto in a background thread — independent of market hours."""
    logger.info("[CRYPTO] Background thread started — trading 24/7")
    while not stop_event.is_set():
        try:
            run_crypto(crypto_fetcher, crypto_strategy, crypto_broker, alerter)
        except Exception as e:
            logger.error(f"[CRYPTO] Error: {e}", exc_info=True)
            alerter.error("crypto loop", str(e))
        stop_event.wait(timeout=RUN_INTERVAL_SECONDS)
    logger.info("[CRYPTO] Background thread stopped")


# ── Pre-market regime check ──────────────────────────────────────────────────

def premarket_regime_check(fetcher: DataFetcher, risk: RiskManager) -> None:
    """
    Before market open, read SPY to prime the bear/bull regime flag.
    Prevents the bot from blindly opening positions on a down-trend day.
    """
    try:
        bars = fetcher.get_bars(["SPY"])
        spy_df = bars.get("SPY")
        if spy_df is not None and not spy_df.empty:
            risk.update_market_regime(spy_df)
            logger.info(f"[PRE-MARKET] Regime check — Bear={risk.is_bear_market} "
                        f"(SPY last close=${spy_df['close'].iloc[-1]:.2f})")
        else:
            logger.warning("[PRE-MARKET] SPY data unavailable — regime unknown")
    except Exception as e:
        logger.warning(f"[PRE-MARKET] Regime check failed: {e}")


# ── Overnight recovery ───────────────────────────────────────────────────────

def recover_overnight_positions(broker: Broker, alerter: Alerter) -> None:
    """
    At startup, close any positions that were opened on a previous trading day.
    Handles the case where the bot was down at EOD and missed liquidation.
    """
    import sqlite3
    positions = broker.get_positions()
    if not positions:
        return

    today = datetime.now(ET).date()

    try:
        conn = sqlite3.connect("db/zeno.db")
        conn.row_factory = sqlite3.Row
        stale = []
        for symbol in positions:
            row = conn.execute(
                "SELECT ts FROM orders WHERE symbol=? AND side='buy' ORDER BY ts DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row:
                buy_date = datetime.fromisoformat(row["ts"]).date()
                if buy_date < today:
                    stale.append(symbol)
        conn.close()
    except Exception as e:
        logger.warning(f"[RECOVERY] DB check failed: {e} — skipping overnight recovery")
        return

    if not stale:
        return

    logger.info(f"[RECOVERY] Found {len(stale)} overnight position(s) from a previous day: {stale}")
    for symbol in stale:
        try:
            pos      = positions[symbol]
            entry    = float(pos.avg_entry_price)
            qty      = int(float(pos.qty))
            price    = float(pos.current_price)
            pnl      = (price - entry) * qty
            broker.cancel_symbol_orders(symbol)
            order    = broker.market_sell(symbol)
            order_id = str(order.id) if order else None
            log_order(symbol, "sell", qty, order_id, "overnight-recovery", price)
            alerter.trade_closed(symbol, "SELL", qty, entry, price, pnl)
            logger.info(f"[RECOVERY] Closed {symbol} | qty={qty} | P&L=${pnl:+.2f}")
        except Exception as e:
            logger.error(f"[RECOVERY] Failed to close {symbol}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both", action="store_true", help="Allow running alongside Kronos")
    args = parser.parse_args()
    lock_acquire("Zeno", force=args.both)
    logger.info("=== Project Zeno Unified Bot starting ===")

    init_db()
    alerter = Alerter()

    # ── Stock components ──
    fetcher       = DataFetcher()
    broker        = Broker()
    risk          = RiskManager()
    macro_fetcher = MacroFetcher()
    screener      = DailyScreener()
    _load_win_rates(risk)

    # Run pre-market screener to get today's dynamic symbol list
    active_symbols  = screener.run()
    last_screen_day = datetime.now(ET).date()
    logger.info(f"[SCREENER] Today's universe: {active_symbols}")
    strategy = CombinedStrategy(
        buy_threshold=0.50,       # lower = more BUY signals intraday
        sell_threshold=0.48,      # higher = more SELL signals intraday
        sentiment_floor=-0.20,
        sentiment_ceiling=-0.40,
        auc_threshold=0.48,       # lower gate = more symbols trade via XGBoost
        sentiment_buy_floor=0.10, # lower = more sentiment-driven entries
    )

    # ── Crypto components ──
    crypto_fetcher  = CryptoFetcher()
    crypto_broker   = CryptoBroker()
    # XGBoost-only for crypto (no news API for crypto symbols)
    crypto_strategy = XGBStrategy(buy_threshold=0.55, sell_threshold=0.45)

    account = broker.get_account()
    mode    = "PAPER" if _IS_PAPER else "LIVE"
    logger.info(
        f"Account: ${float(account.portfolio_value):,.2f} | "
        f"Cash: ${float(account.cash):,.2f} | "
        f"Stocks: {len(SYMBOLS)} | Crypto: {len(CRYPTO_SYMBOLS)} | Mode: {mode}"
    )
    alerter.bot_started(mode=mode)

    # ── Overnight recovery — close stale positions from previous day ──
    recover_overnight_positions(broker, alerter)

    # ── Start crypto background thread ──
    stop_event = threading.Event()
    if CRYPTO_ENABLED:
        crypto_thread = threading.Thread(
            target=crypto_loop,
            args=(crypto_fetcher, crypto_strategy, crypto_broker, alerter, stop_event),
            daemon=True,
            name="crypto-loop",
        )
        crypto_thread.start()
        logger.info("[CRYPTO] Thread started — BTC/ETH/SOL/DOGE/AVAX running 24/7")
    else:
        logger.info("[CRYPTO] Disabled via CRYPTO_ENABLED=false")

    # ── Main stock loop ──
    burst_opens_today: list[str] = []   # tracks entries during open burst window
    try:
        while True:
            try:
                today = datetime.now(ET).date()

                # Reset burst tracker and re-run screener on new day
                if today != last_screen_day:
                    active_symbols    = screener.run()
                    last_screen_day   = today
                    burst_opens_today = []
                    logger.info(f"[SCREENER] New day — universe updated: {active_symbols}")

                if is_eod_liquidate_time():
                    liquidate_all_positions(broker, alerter)
                elif is_market_open():
                    run_stocks(fetcher, strategy, broker, risk, alerter,
                               active_symbols=active_symbols,
                               macro_fetcher=macro_fetcher,
                               burst_opens_today=burst_opens_today)
                else:
                    now_et = datetime.now(ET)
                    # Run pre-market regime check in the 30 mins before open
                    open_h, open_m = map(int, MARKET_OPEN_TIME.split(":"))
                    premarket_start = now_et.replace(hour=open_h, minute=open_m, second=0, microsecond=0).replace(minute=open_m - 30 if open_m >= 30 else 0)
                    market_open_dt  = now_et.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
                    if premarket_start <= now_et < market_open_dt:
                        premarket_regime_check(fetcher, risk)
                    logger.info(f"[STOCKS] Market closed ({now_et.strftime('%Y-%m-%d %H:%M:%S ET')}), sleeping...")

            except Exception as e:
                logger.error(f"[STOCKS] Error: {e}", exc_info=True)
                alerter.error("stock loop", str(e))

            time.sleep(RUN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_event.set()
        alerter.bot_stopped()
        lock_release()


# detect paper mode for logging
from config.settings import ALPACA_BASE_URL
_IS_PAPER = "paper" in ALPACA_BASE_URL

if __name__ == "__main__":
    main()
