"""
Kubera — India Paper Trading Bot
Paper trades NSE stocks using Angel One SmartAPI (real-time data) + virtual SQLite broker.
Runs 9:30 AM – 3:30 PM IST (skips volatile 9:15–9:30 opening auction).

v3 improvements:
- XGBoost confidence threshold at 63%
- Midday lull filter: no new buys 12:30–14:00 IST (low-liquidity chop)
- Daily loss cap: halt new buys if portfolio drops >3% from day open
- ATR-based trailing stops: smarter than fixed % (1.5x ATR normal, 2x ATR cheap)
"""
from __future__ import annotations
import time
import numpy as np
from datetime import datetime, date
import pytz

from config.india_settings import (
    INDIA_SYMBOLS, INDIA_ENABLED,
    INDIA_MARKET_OPEN, INDIA_MARKET_CLOSE, INDIA_EOD_LIQUIDATE,
    INDIA_MAX_POSITION_PCT, INDIA_MAX_OPEN_POSITIONS,
    INDIA_STARTING_CASH, INDIA_RUN_INTERVAL_SECS, INDIA_BAR_TIMEFRAME,
    INDIA_LOOKBACK_BARS,
)
from data.angel_fetcher import AngelFetcher
from execution.india_paper_broker import IndiaPaperBroker
from strategy.xgb_strategy import XGBStrategy, SPY_KEY
from strategy.base import Signal
from utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

NIFTY_SYMBOL   = "^NSEI"      # Yahoo Finance ticker for NIFTY 50
CHEAP_THRESHOLD = 100.0        # stocks below this get wider ATR multiplier

# ── NSE holiday calendar 2026 ─────────────────────────────────────────────────
NSE_HOLIDAYS_2026 = {
    "2026-01-15",  # Municipal Corporation Election – Maharashtra
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali – Balipratipada
    "2026-11-24",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2026-12-25",  # Christmas
}

# ── XGBoost thresholds ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65    # high-conviction longs only
SELL_THRESHOLD       = 0.35    # high-conviction shorts only

# ── Experimental mode flags (set to False to revert to conservative defaults) ─
ENABLE_MIDDAY_TRADING  = True   # if False: no new buys 12:30–14:00
ENABLE_BEAR_TRADING    = True   # if False: no new buys when NIFTY bearish
ENABLE_SHORT_SELLING   = True   # if False: no short positions
NIFTY_REGIME_FILTER    = True   # if True: only long in bull, only short in bear (regime alignment)
SKIP_OPEN_BAR          = True   # if True: skip new entries at 9:30 (noisy opening bar)

# ── Midday lull filter ────────────────────────────────────────────────────────
MIDDAY_START = "12:30"
MIDDAY_END   = "14:00"

# ── Daily loss cap ────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT = 3.0       # halt new buys if portfolio drops >3% intraday

# ── ATR-based trailing stops & profit targets ─────────────────────────────────
ATR_PERIOD              = 14
ATR_MULTIPLIER_CHEAP    = 2.0   # wider stop for volatile cheap stocks
ATR_MULTIPLIER_NORMAL   = 1.5   # tighter stop for higher-priced stocks
PROFIT_TARGET_ATR_MULT  = 3.0   # take profit at 3×ATR above entry (2:1 R:R)
VOLUME_LOOKBACK         = 20

# ── Penny-stock filter ────────────────────────────────────────────────────────
MIN_STOCK_PRICE = 150.0         # skip stocks below this price


def ist_now() -> datetime:
    return datetime.now(IST)


def market_is_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    if now.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2026:
        logger.info(f"[INDIA] NSE holiday today — market closed")
        return False
    t = now.strftime("%H:%M")
    return INDIA_MARKET_OPEN <= t <= INDIA_MARKET_CLOSE


def eod_liquidate_time() -> bool:
    return ist_now().strftime("%H:%M") >= INDIA_EOD_LIQUIDATE


def is_midday_lull() -> bool:
    """True during 12:30–14:00 IST — low-liquidity chop, avoid new entries."""
    t = ist_now().strftime("%H:%M")
    lull = MIDDAY_START <= t < MIDDAY_END
    if lull:
        logger.debug("[INDIA] Midday lull — skipping new buys")
    return lull


def is_open_bar() -> bool:
    """True at 9:30 — opening bar is noisy (auction residue), skip new entries."""
    return ist_now().strftime("%H:%M") == "09:30"


def is_nifty_bearish(nifty_df) -> bool:
    """True if NIFTY is in a short-term downtrend (close < 20-bar SMA)."""
    if nifty_df is None or len(nifty_df) < 20:
        return False
    sma20 = nifty_df["close"].tail(20).mean()
    last  = float(nifty_df["close"].iloc[-1])
    bear  = last < sma20
    if bear:
        logger.info(f"[INDIA] NIFTY bearish: {last:.1f} < SMA20 {sma20:.1f} — blocking new buys")
    return bear


def has_enough_volume(df, lookback: int = VOLUME_LOOKBACK) -> bool:
    """True if latest bar volume >= 50% of recent average."""
    if len(df) < lookback:
        return True
    avg_vol = df["volume"].tail(lookback).mean()
    cur_vol = float(df["volume"].iloc[-1])
    return cur_vol >= avg_vol * 0.5


def compute_atr(df, period: int = ATR_PERIOD) -> float:
    """
    Compute Average True Range over the last `period` bars.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Returns 0.0 if not enough bars.
    """
    if len(df) < period + 1:
        return 0.0
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        )
    )
    return float(np.mean(tr[-period:]))


def atr_stop(price: float, df, entry: bool = True) -> float:
    """
    Compute ATR-based stop price below `price`.
    Uses wider multiplier for cheap stocks.
    Falls back to fixed % if ATR can't be computed.
    """
    atr = compute_atr(df)
    if atr <= 0:
        # Fallback: 2.5% for cheap, 1.5% for normal
        pct = 2.5 if price < CHEAP_THRESHOLD else 1.5
        return price * (1 - pct / 100)
    multiplier = ATR_MULTIPLIER_CHEAP if price < CHEAP_THRESHOLD else ATR_MULTIPLIER_NORMAL
    return price - multiplier * atr


def run():
    if not INDIA_ENABLED:
        logger.info("India bot disabled (INDIA_ENABLED=false in .env)")
        return

    logger.info("── Kubera v3 starting ──")
    fetcher  = AngelFetcher(interval=INDIA_BAR_TIMEFRAME)
    broker   = IndiaPaperBroker(starting_cash=INDIA_STARTING_CASH)
    strategy = XGBStrategy(
        buy_threshold=CONFIDENCE_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
    )

    # Trailing stops and profit targets tracked in memory
    trailing_stops: dict[str, float] = {}
    short_stops:    dict[str, float] = {}   # short stop prices (above entry)
    profit_targets: dict[str, float] = {}   # long profit target prices (above entry)
    short_targets:  dict[str, float] = {}   # short profit target prices (below entry)

    # Daily loss cap state
    day_open_value: float | None = None
    last_trade_date: date | None = None

    while True:
        try:
            if not market_is_open():
                logger.info(f"[INDIA] Market closed ({ist_now().strftime('%H:%M:%S')} IST), sleeping...")
                time.sleep(INDIA_RUN_INTERVAL_SECS)
                continue

            logger.info("── India tick ──")

            # Fetch NIFTY + all symbols
            all_symbols = INDIA_SYMBOLS + [NIFTY_SYMBOL]
            bars        = fetcher.get_bars(all_symbols, lookback=INDIA_LOOKBACK_BARS)

            # Extract NIFTY and rename so XGBStrategy finds it as "SPY"
            nifty_df = bars.pop(NIFTY_SYMBOL, None)
            if nifty_df is not None:
                bars[SPY_KEY] = nifty_df

            positions      = broker.get_positions()
            current_prices = {sym: float(df["close"].iloc[-1])
                              for sym, df in bars.items()
                              if sym != SPY_KEY and not df.empty}

            # ── Daily loss cap: snapshot portfolio at day open ────────────────
            today = ist_now().date()
            if last_trade_date != today:
                last_trade_date = today
                day_open_value  = broker.get_portfolio_value(current_prices)
                logger.info(f"[INDIA] Day open portfolio: ₹{day_open_value:,.2f}")
                fetcher.get_day_gainers(INDIA_SYMBOLS)

            portfolio_value = broker.get_portfolio_value(current_prices)
            daily_loss_pct  = ((day_open_value - portfolio_value) / day_open_value * 100
                               if day_open_value else 0.0)
            daily_loss_halt = daily_loss_pct >= MAX_DAILY_LOSS_PCT
            if daily_loss_halt:
                logger.warning(
                    f"[INDIA] Daily loss cap hit: -{daily_loss_pct:.2f}% "
                    f"(₹{day_open_value - portfolio_value:,.0f} loss) — halting new buys"
                )

            # ── EOD liquidation ───────────────────────────────────────────────
            if eod_liquidate_time() and (positions or broker.get_short_positions()):
                logger.info("[INDIA EOD] Liquidating all positions")
                broker.close_all_positions(current_prices)
                trailing_stops.clear()
                short_stops.clear()
                profit_targets.clear()
                short_targets.clear()
                time.sleep(INDIA_RUN_INTERVAL_SECS)
                continue

            # ── Market regime ─────────────────────────────────────────────────
            bear_market = is_nifty_bearish(nifty_df)
            midday_lull = is_midday_lull()

            # ── Trailing stop & profit target management (longs) ─────────────
            for symbol, pos in list(positions.items()):
                price = current_prices.get(symbol)
                if price is None:
                    continue

                # Profit target — take profit before checking stop
                target = profit_targets.get(symbol)
                if target and price >= target:
                    logger.info(f"[INDIA TARGET] {symbol} hit profit target @ ₹{price:,.2f} (target=₹{target:,.2f})")
                    try:
                        broker.sell(symbol, int(pos["qty"]), price)
                        trailing_stops.pop(symbol, None)
                        profit_targets.pop(symbol, None)
                        broker.clear_stop(symbol)
                    except ValueError as e:
                        logger.warning(f"[INDIA] Target sell failed for {symbol}: {e}")
                    continue

                stop = trailing_stops.get(symbol)
                if stop and price <= stop:
                    logger.info(f"[INDIA STOP] {symbol} hit ATR stop @ ₹{price:,.2f} (stop=₹{stop:,.2f})")
                    try:
                        broker.sell(symbol, int(pos["qty"]), price)
                        trailing_stops.pop(symbol, None)
                        profit_targets.pop(symbol, None)
                        broker.clear_stop(symbol)
                    except ValueError as e:
                        logger.warning(f"[INDIA] Stop-loss sell failed for {symbol}: {e}")
                else:
                    # Trail the stop upward using ATR
                    df_sym = bars.get(symbol)
                    if df_sym is not None:
                        new_stop = atr_stop(price, df_sym)
                        if stop is None or new_stop > stop:
                            trailing_stops[symbol] = new_stop

            # ── Short stop & profit target management ─────────────────────────
            if ENABLE_SHORT_SELLING:
                for symbol, pos in list(broker.get_short_positions().items()):
                    price = current_prices.get(symbol)
                    if price is None:
                        continue

                    # Profit target for short — take profit before stop
                    s_target = short_targets.get(symbol)
                    if s_target and price <= s_target:
                        logger.info(f"[INDIA SHORT TARGET] {symbol} hit profit target @ ₹{price:,.2f} (target=₹{s_target:,.2f})")
                        try:
                            broker.cover(symbol, int(abs(pos["qty"])), price)
                            short_stops.pop(symbol, None)
                            short_targets.pop(symbol, None)
                            broker.clear_stop(symbol)
                        except ValueError as e:
                            logger.warning(f"[INDIA] Cover target failed for {symbol}: {e}")
                        continue

                    stop = short_stops.get(symbol)
                    if stop and price >= stop:
                        logger.info(f"[INDIA SHORT STOP] {symbol} hit stop @ ₹{price:,.2f} (stop=₹{stop:,.2f})")
                        try:
                            broker.cover(symbol, int(abs(pos["qty"])), price)
                            short_stops.pop(symbol, None)
                            short_targets.pop(symbol, None)
                            broker.clear_stop(symbol)
                        except ValueError as e:
                            logger.warning(f"[INDIA] Cover failed for {symbol}: {e}")

            open_count = len(broker.get_positions()) + len(broker.get_short_positions())
            signals    = strategy.generate_signals(bars)

            for sig in signals:
                symbol = sig.symbol
                if symbol not in current_prices:
                    continue
                price = current_prices[symbol]

                if sig.signal == Signal.BUY:
                    # Cover short first if one exists
                    if ENABLE_SHORT_SELLING and broker.has_short(symbol):
                        pos = broker.get_short_positions()[symbol]
                        try:
                            broker.cover(symbol, int(abs(pos["qty"])), price)
                            short_stops.pop(symbol, None)
                            short_targets.pop(symbol, None)
                        except ValueError as e:
                            logger.warning(f"[INDIA] Cover skipped for {symbol}: {e}")
                        continue

                    # Gate: price / regime / open bar / midday / daily loss / capacity / volume
                    if price < MIN_STOCK_PRICE:
                        logger.debug(f"[INDIA] {symbol} @ ₹{price:,.2f} below min price — skipping")
                        continue
                    if SKIP_OPEN_BAR and is_open_bar():
                        logger.debug(f"[INDIA] BUY blocked (opening bar noise): {symbol}")
                        continue
                    if NIFTY_REGIME_FILTER and bear_market:
                        logger.info(f"[INDIA] BUY blocked (NIFTY bearish — regime filter): {symbol}")
                        continue
                    if not ENABLE_BEAR_TRADING and bear_market:
                        logger.info(f"[INDIA] BUY blocked (bear market): {symbol}")
                        continue
                    if not ENABLE_MIDDAY_TRADING and midday_lull:
                        logger.info(f"[INDIA] BUY blocked (midday lull): {symbol}")
                        continue
                    if daily_loss_halt:
                        logger.info(f"[INDIA] BUY blocked (daily loss cap): {symbol}")
                        continue
                    if open_count >= INDIA_MAX_OPEN_POSITIONS or broker.has_position(symbol):
                        continue
                    if not has_enough_volume(bars[symbol]):
                        logger.warning(f"[INDIA] {symbol} skipped — low volume")
                        continue
                    qty = int((portfolio_value * INDIA_MAX_POSITION_PCT) / price)
                    if qty < 1:
                        logger.warning(f"[INDIA] {symbol} @ ₹{price:,.2f} too expensive, skipping")
                        continue
                    try:
                        broker.buy(symbol, qty, price)
                        stop = atr_stop(price, bars[symbol])
                        atr_val = compute_atr(bars[symbol])
                        target = price + PROFIT_TARGET_ATR_MULT * atr_val if atr_val > 0 else price * 1.03
                        trailing_stops[symbol] = stop
                        profit_targets[symbol]  = target
                        broker.set_stop(symbol, stop, target)
                        logger.info(
                            f"[INDIA BUY] {symbol} x{qty} @ ₹{price:,.2f} | "
                            f"stop=₹{stop:,.2f} target=₹{target:,.2f} "
                            f"(conf={sig.confidence:.2f}, ATR={atr_val:.2f})"
                        )
                        open_count += 1
                    except ValueError as e:
                        logger.warning(f"[INDIA] Buy skipped for {symbol}: {e}")

                elif sig.signal == Signal.SELL:
                    # Sell long if we have one
                    if broker.has_position(symbol):
                        pos = broker.get_positions()[symbol]
                        try:
                            broker.sell(symbol, int(pos["qty"]), price)
                            trailing_stops.pop(symbol, None)
                            profit_targets.pop(symbol, None)
                            broker.clear_stop(symbol)
                        except ValueError as e:
                            logger.warning(f"[INDIA] Sell skipped for {symbol}: {e}")
                        continue

                    # Otherwise open a short
                    if not ENABLE_SHORT_SELLING:
                        continue
                    if broker.has_short(symbol):
                        continue
                    if price < MIN_STOCK_PRICE:
                        logger.debug(f"[INDIA] {symbol} @ ₹{price:,.2f} below min price — skipping short")
                        continue
                    if SKIP_OPEN_BAR and is_open_bar():
                        logger.debug(f"[INDIA] SHORT blocked (opening bar noise): {symbol}")
                        continue
                    if NIFTY_REGIME_FILTER and not bear_market:
                        logger.info(f"[INDIA] SHORT blocked (NIFTY bullish — regime filter): {symbol}")
                        continue
                    if daily_loss_halt:
                        continue
                    if not has_enough_volume(bars[symbol]):
                        continue
                    qty = int((portfolio_value * INDIA_MAX_POSITION_PCT) / price)
                    if qty < 1:
                        continue
                    try:
                        broker.short(symbol, qty, price)
                        atr_val = compute_atr(bars[symbol])
                        short_stops[symbol]  = price * 1.02  # 2% stop above entry
                        s_target = price - PROFIT_TARGET_ATR_MULT * atr_val if atr_val > 0 else price * 0.97
                        short_targets[symbol] = s_target
                        broker.set_stop(symbol, short_stops[symbol], s_target)
                        logger.info(
                            f"[INDIA SHORT] {symbol} x{qty} @ ₹{price:,.2f} | "
                            f"stop=₹{short_stops[symbol]:,.2f} target=₹{s_target:,.2f} "
                            f"(conf={sig.confidence:.2f}, ATR={atr_val:.2f})"
                        )
                    except ValueError as e:
                        logger.warning(f"[INDIA] Short skipped for {symbol}: {e}")

            portfolio_value = broker.get_portfolio_value(current_prices)
            broker.log_portfolio_value(portfolio_value)
            broker.log_prices(current_prices)
            logger.info(
                f"[INDIA] Portfolio=₹{portfolio_value:,.2f} | "
                f"Cash=₹{broker.get_cash():,.2f} | "
                f"Positions={len(broker.get_positions())} | "
                f"Signals={len(signals)} | "
                f"Bear={bear_market} | Midday={midday_lull} | "
                f"DailyLoss={daily_loss_pct:.1f}%"
            )

        except Exception as e:
            logger.error(f"[INDIA] Error: {e}", exc_info=True)

        time.sleep(INDIA_RUN_INTERVAL_SECS)


if __name__ == "__main__":
    run()
