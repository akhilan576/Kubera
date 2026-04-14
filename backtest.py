"""
Kubera Backtest — replay a past NSE trading session using yfinance historical data.

Usage:
    python backtest.py                      # replay most recent trading day
    python backtest.py --date 2026-04-11    # replay a specific date
    python backtest.py --date 2026-04-11 --verbose   # show every signal
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import yfinance as yf

from config.india_settings import (
    INDIA_SYMBOLS, INDIA_STARTING_CASH,
    INDIA_MAX_POSITION_PCT, INDIA_MAX_OPEN_POSITIONS,
)
from kubera import (
    NSE_HOLIDAYS_2026,
    CONFIDENCE_THRESHOLD, SELL_THRESHOLD,
    ENABLE_SHORT_SELLING, ENABLE_BEAR_TRADING, ENABLE_MIDDAY_TRADING,
    NIFTY_REGIME_FILTER, SKIP_OPEN_BAR,
    CHEAP_THRESHOLD, MIDDAY_START, MIDDAY_END,
    MIN_STOCK_PRICE, PROFIT_TARGET_ATR_MULT,
    atr_stop, compute_atr, has_enough_volume, is_nifty_bearish,
)
from strategy.xgb_strategy import XGBStrategy, SPY_KEY
from strategy.base import Signal

import logging
logging.disable(logging.CRITICAL)

IST    = pytz.timezone("Asia/Kolkata")
WARMUP = 30   # days of history to fetch before the target date (for indicator warmup)


# ── Date helpers ───────────────────────────────────────────────────────────────

def last_trading_day() -> date:
    """Return the most recent trading weekday that isn't a holiday."""
    d = date.today() - timedelta(days=1)
    for _ in range(10):
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS_2026:
            return d
        d -= timedelta(days=1)
    return d


# ── Data fetch ─────────────────────────────────────────────────────────────────

def fetch_bars(symbols: list[str], target: date) -> dict[str, pd.DataFrame]:
    """
    Fetch 15-min OHLCV bars for all symbols from (target - WARMUP days) → target.
    NSE stocks get .NS suffix; index symbols (^) pass through as-is.
    Returns {symbol: DataFrame} with columns open/high/low/close/volume,
    indexed by UTC timestamps.
    """
    start = (target - timedelta(days=WARMUP)).strftime("%Y-%m-%d")
    end   = (target + timedelta(days=1)).strftime("%Y-%m-%d")

    result: dict[str, pd.DataFrame] = {}
    print(f"Fetching {len(symbols)} symbols ({start} → {end}) …", flush=True)

    for sym in symbols:
        ticker = sym if sym.startswith("^") else f"{sym}.NS"
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="15m", progress=False, auto_adjust=True)
            if df.empty:
                continue
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            result[sym] = df
        except Exception as exc:
            print(f"  Warning: {sym} failed — {exc}")

    print(f"Got data for {len(result)} symbols.\n")
    return result


# ── In-memory portfolio ────────────────────────────────────────────────────────

class SimBroker:
    """Lightweight in-memory paper broker — mirrors IndiaPaperBroker logic."""

    def __init__(self, starting_cash: float):
        self.cash      = starting_cash
        self.positions: dict[str, dict] = {}   # {sym: {qty, avg_entry}}  qty>0 long, qty<0 short
        self.trades:    list[dict]       = []

    def portfolio_value(self, prices: dict[str, float]) -> float:
        equity = sum(pos["qty"] * prices.get(s, pos["avg_entry"])
                     for s, pos in self.positions.items())
        return self.cash + equity

    def buy(self, sym, qty, price, time="", note=""):
        cost = qty * price
        if cost > self.cash:
            return False
        pos = self.positions.get(sym)
        if pos:
            new_qty = pos["qty"] + qty
            pos["avg_entry"] = (pos["qty"] * pos["avg_entry"] + qty * price) / new_qty
            pos["qty"] = new_qty
        else:
            self.positions[sym] = {"qty": qty, "avg_entry": price}
        self.cash -= cost
        self.trades.append({"time": time, "side": "BUY", "sym": sym, "qty": qty, "price": price, "pnl": 0, "note": note})
        return True

    def sell(self, sym, qty, price, time="", note=""):
        pos = self.positions.get(sym)
        if not pos or pos["qty"] < qty:
            return False
        pnl = (price - pos["avg_entry"]) * qty
        pos["qty"] -= qty
        if pos["qty"] == 0:
            del self.positions[sym]
        self.cash += qty * price
        self.trades.append({"time": time, "side": "SELL", "sym": sym, "qty": qty, "price": price, "pnl": pnl, "note": note})
        return True

    def short(self, sym, qty, price, time="", note=""):
        pos = self.positions.get(sym)
        if pos:
            new_qty = pos["qty"] - qty
            pos["avg_entry"] = (abs(pos["qty"]) * pos["avg_entry"] + qty * price) / (abs(pos["qty"]) + qty)
            pos["qty"] = new_qty
        else:
            self.positions[sym] = {"qty": -qty, "avg_entry": price}
        self.cash += qty * price
        self.trades.append({"time": time, "side": "SHORT", "sym": sym, "qty": qty, "price": price, "pnl": 0, "note": note})
        return True

    def cover(self, sym, qty, price, time="", note=""):
        pos = self.positions.get(sym)
        if not pos or pos["qty"] >= 0:
            return False
        pnl = (pos["avg_entry"] - price) * qty
        pos["qty"] += qty
        if pos["qty"] == 0:
            del self.positions[sym]
        self.cash -= qty * price
        self.trades.append({"time": time, "side": "COVER", "sym": sym, "qty": qty, "price": price, "pnl": pnl, "note": note})
        return True

    def longs(self):
        return {s: p for s, p in self.positions.items() if p["qty"] > 0}

    def shorts(self):
        return {s: p for s, p in self.positions.items() if p["qty"] < 0}


# ── Backtest engine ────────────────────────────────────────────────────────────

def run(target: date, verbose: bool = False):
    print("=" * 64)
    print(f"  KUBERA BACKTEST  —  {target.strftime('%A, %d %b %Y')}")
    print(f"  Confidence ≥{CONFIDENCE_THRESHOLD:.0%} buy  |  ≤{SELL_THRESHOLD:.0%} sell")
    print("=" * 64 + "\n")

    # ── Fetch data ─────────────────────────────────────────────────────────────
    all_symbols = INDIA_SYMBOLS + ["^NSEI"]
    all_bars    = fetch_bars(all_symbols, target)

    if not all_bars:
        print("No data returned. Was the market open that day?")
        return

    # Find the bar timestamps for the target date in IST
    sample_df   = next(iter(all_bars.values()))
    sample_ist  = sample_df.index.tz_convert(IST)
    target_mask = sample_ist.date == target
    day_times   = sample_df.index[target_mask]

    if len(day_times) == 0:
        print(f"No bars found for {target}. Market may have been closed.")
        return

    day_ist = day_times.tz_convert(IST)
    print(f"  {len(day_times)} bars  ({day_ist[0].strftime('%H:%M')} → {day_ist[-1].strftime('%H:%M')} IST)\n")

    # ── Simulation state ───────────────────────────────────────────────────────
    broker          = SimBroker(INDIA_STARTING_CASH)
    strategy        = XGBStrategy(buy_threshold=CONFIDENCE_THRESHOLD,
                                  sell_threshold=SELL_THRESHOLD)
    trailing_stops: dict[str, float] = {}
    short_stops:    dict[str, float] = {}
    profit_targets: dict[str, float] = {}
    short_targets:  dict[str, float] = {}
    pnl_history:    list[tuple]      = []   # (time_str, portfolio_value)

    # ── Bar-by-bar replay ──────────────────────────────────────────────────────
    for bar_utc in day_times:
        bar_ist  = bar_utc.tz_convert(IST)
        t_str    = bar_ist.strftime("%H:%M")

        if t_str < "09:30":
            continue

        # Slice all data up to and including this bar
        cur_bars: dict[str, pd.DataFrame] = {}
        for sym, df in all_bars.items():
            sliced = df[df.index <= bar_utc].reset_index(drop=True)
            if not sliced.empty:
                cur_bars[sym] = sliced

        nifty_df = cur_bars.pop("^NSEI", None)
        if nifty_df is not None:
            cur_bars[SPY_KEY] = nifty_df

        prices = {s: float(df["close"].iloc[-1])
                  for s, df in cur_bars.items() if s != SPY_KEY}

        pval = broker.portfolio_value(prices)
        pnl_history.append((t_str, pval))

        # ── EOD liquidation ────────────────────────────────────────────────────
        if t_str >= "15:15":
            for sym, pos in list(broker.longs().items()):
                broker.sell(sym, int(pos["qty"]), prices.get(sym, pos["avg_entry"]), time=t_str, note="EOD")
            for sym, pos in list(broker.shorts().items()):
                broker.cover(sym, int(abs(pos["qty"])), prices.get(sym, pos["avg_entry"]), time=t_str, note="EOD")
            last_prices = prices
            break

        # ── Trailing stop & profit targets — longs ────────────────────────────
        for sym, pos in list(broker.longs().items()):
            price = prices.get(sym)
            if price is None:
                continue

            # Profit target first
            target = profit_targets.get(sym)
            if target and price >= target:
                broker.sell(sym, int(pos["qty"]), price, time=t_str, note="TP")
                trailing_stops.pop(sym, None)
                profit_targets.pop(sym, None)
                if verbose:
                    pnl = (price - pos["avg_entry"]) * pos["qty"]
                    print(f"  {t_str}  TARGET     {sym:12}  @ ₹{price:,.2f}  P&L ₹{pnl:+,.2f}")
                continue

            stop = trailing_stops.get(sym)
            if stop and price <= stop:
                broker.sell(sym, int(pos["qty"]), price, time=t_str, note="SL")
                trailing_stops.pop(sym, None)
                profit_targets.pop(sym, None)
                if verbose:
                    print(f"  {t_str}  STOP-SELL  {sym:12}  @ ₹{price:,.2f}")
            elif sym in cur_bars:
                new_stop = atr_stop(price, cur_bars[sym])
                if stop is None or new_stop > stop:
                    trailing_stops[sym] = new_stop

        # ── Trailing stop & profit targets — shorts ───────────────────────────
        if ENABLE_SHORT_SELLING:
            for sym, pos in list(broker.shorts().items()):
                price = prices.get(sym)
                if price is None:
                    continue

                # Profit target for short
                s_target = short_targets.get(sym)
                if s_target and price <= s_target:
                    broker.cover(sym, int(abs(pos["qty"])), price, time=t_str, note="TP")
                    short_stops.pop(sym, None)
                    short_targets.pop(sym, None)
                    if verbose:
                        pnl = (pos["avg_entry"] - price) * abs(pos["qty"])
                        print(f"  {t_str}  SHORT-TP   {sym:12}  @ ₹{price:,.2f}  P&L ₹{pnl:+,.2f}")
                    continue

                stop = short_stops.get(sym)
                if stop and price >= stop:
                    broker.cover(sym, int(abs(pos["qty"])), price, time=t_str, note="SL")
                    short_stops.pop(sym, None)
                    short_targets.pop(sym, None)
                    if verbose:
                        print(f"  {t_str}  STOP-COVER {sym:12}  @ ₹{price:,.2f}")

        # ── Market regime checks ───────────────────────────────────────────────
        bear_market = is_nifty_bearish(nifty_df)
        midday_lull = MIDDAY_START <= t_str < MIDDAY_END
        open_count  = len(broker.positions)

        # ── Strategy signals ───────────────────────────────────────────────────
        signals = strategy.generate_signals(cur_bars)

        for sig in signals:
            sym = sig.symbol
            if sym not in prices:
                continue
            price = prices[sym]

            if sig.signal == Signal.BUY:
                # Cover short first
                if ENABLE_SHORT_SELLING and sym in broker.shorts():
                    pos = broker.shorts()[sym]
                    broker.cover(sym, int(abs(pos["qty"])), price, time=t_str)
                    short_stops.pop(sym, None)
                    short_targets.pop(sym, None)
                    if verbose:
                        print(f"  {t_str}  COVER      {sym:12}  @ ₹{price:,.2f}  (flip)")
                    continue

                if price < MIN_STOCK_PRICE:
                    continue
                if SKIP_OPEN_BAR and t_str == "09:30":
                    continue
                if NIFTY_REGIME_FILTER and bear_market:
                    continue
                if not ENABLE_BEAR_TRADING and bear_market:
                    continue
                if not ENABLE_MIDDAY_TRADING and midday_lull:
                    continue
                if open_count >= INDIA_MAX_OPEN_POSITIONS or sym in broker.longs():
                    continue
                if sym not in cur_bars or not has_enough_volume(cur_bars[sym]):
                    continue
                qty = int((pval * INDIA_MAX_POSITION_PCT) / price)
                if qty < 1 or qty * price > broker.cash:
                    continue
                broker.buy(sym, qty, price, time=t_str)
                stop = atr_stop(price, cur_bars[sym])
                atr_val = compute_atr(cur_bars[sym])
                target = price + PROFIT_TARGET_ATR_MULT * atr_val if atr_val > 0 else price * 1.03
                trailing_stops[sym] = stop
                profit_targets[sym] = target
                open_count += 1
                if verbose:
                    print(f"  {t_str}  BUY        {sym:12}  x{qty}  @ ₹{price:,.2f}  conf={sig.confidence:.2f}  stop=₹{stop:,.2f}  target=₹{target:,.2f}")

            elif sig.signal == Signal.SELL:
                if sym in broker.longs():
                    pos = broker.longs()[sym]
                    broker.sell(sym, int(pos["qty"]), price, time=t_str)
                    trailing_stops.pop(sym, None)
                    profit_targets.pop(sym, None)
                    if verbose:
                        pnl = (price - pos["avg_entry"]) * pos["qty"]
                        print(f"  {t_str}  SELL       {sym:12}  x{int(pos['qty'])}  @ ₹{price:,.2f}  P&L ₹{pnl:+,.2f}")
                    continue

                if not ENABLE_SHORT_SELLING or sym in broker.shorts():
                    continue
                if price < MIN_STOCK_PRICE:
                    continue
                if SKIP_OPEN_BAR and t_str == "09:30":
                    continue
                if NIFTY_REGIME_FILTER and not bear_market:
                    continue
                if open_count >= INDIA_MAX_OPEN_POSITIONS:
                    continue
                if sym not in cur_bars or not has_enough_volume(cur_bars[sym]):
                    continue
                qty = int((pval * INDIA_MAX_POSITION_PCT) / price)
                if qty < 1:
                    continue
                broker.short(sym, qty, price, time=t_str)
                atr_val = compute_atr(cur_bars[sym])
                short_stops[sym]  = price * 1.02
                s_target = price - PROFIT_TARGET_ATR_MULT * atr_val if atr_val > 0 else price * 0.97
                short_targets[sym] = s_target
                open_count += 1
                if verbose:
                    print(f"  {t_str}  SHORT      {sym:12}  x{qty}  @ ₹{price:,.2f}  conf={sig.confidence:.2f}  stop=₹{short_stops[sym]:,.2f}  target=₹{s_target:,.2f}")

    # ── Results ────────────────────────────────────────────────────────────────
    # Close any remaining positions at last known prices (safety net)
    last_p = locals().get("last_prices", {})
    for sym, pos in list(broker.longs().items()):
        broker.sell(sym, int(pos["qty"]), last_p.get(sym, pos["avg_entry"]), time="15:15", note="EOD")
    for sym, pos in list(broker.shorts().items()):
        broker.cover(sym, int(abs(pos["qty"])), last_p.get(sym, pos["avg_entry"]), time="15:15", note="EOD")

    final_pval  = broker.cash   # all positions closed, cash = final portfolio
    net_pnl     = final_pval - INDIA_STARTING_CASH
    net_pct     = net_pnl / INDIA_STARTING_CASH * 100

    all_trades  = broker.trades
    closing     = [t for t in all_trades if t["side"] in ("SELL", "COVER")]
    wins        = [t for t in closing if t["pnl"] > 0]
    losses      = [t for t in closing if t["pnl"] < 0]
    win_rate    = len(wins) / len(closing) * 100 if closing else 0

    # Max drawdown
    peak, max_dd = INDIA_STARTING_CASH, 0.0
    for _, v in pnl_history:
        if v > peak:
            peak = v
        if peak - v > max_dd:
            max_dd = peak - v
    max_dd_pct = max_dd / INDIA_STARTING_CASH * 100

    print("\n" + "─" * 64)
    print(f"  {'Net P&L':<20} ₹{net_pnl:+,.2f}  ({net_pct:+.2f}%)")
    print(f"  {'Max drawdown':<20} ₹{max_dd:,.2f}  ({max_dd_pct:.2f}%)")
    print(f"  {'Total trades':<20} {len(all_trades)}")
    print(f"  {'Win rate':<20} {len(wins)}/{len(closing)} ({win_rate:.1f}%)" if closing else f"  {'Closed trades':<20} 0")
    if wins:
        print(f"  {'Avg win':<20} ₹{sum(t['pnl'] for t in wins)/len(wins):+,.2f}")
    if losses:
        print(f"  {'Avg loss':<20} ₹{sum(t['pnl'] for t in losses)/len(losses):+,.2f}")
    print("─" * 64)

    print(f"\n  {'Time':6}  {'Side':6}  {'Symbol':12}  {'Qty':>5}  {'Price':>10}  {'P&L':>10}  Note")
    print(f"  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*4}")
    for t in all_trades:
        is_close = t["side"] in ("SELL", "COVER")
        pnl_s    = f"₹{t['pnl']:+,.2f}" if is_close else "—"
        note_s   = f"[{t['note']}]" if t["note"] else ""
        print(f"  {t.get('time',''):6}  {t['side']:6}  {t['sym']:12}  {t['qty']:>5}  "
              f"₹{t['price']:>9,.2f}  {pnl_s:>10}  {note_s}")

    print("\n" + "=" * 64 + "\n")

    # Attach time to trades for display
    return net_pnl, net_pct


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a past NSE session with Kubera")
    parser.add_argument("--date",    default=None,  help="Date to replay YYYY-MM-DD (default: last trading day)")
    parser.add_argument("--verbose", action="store_true", help="Print every signal")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else last_trading_day()
    run(target, verbose=args.verbose)
