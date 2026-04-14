"""
CLI performance dashboard for Project Zeno.

Usage:
    python -m dashboard.cli              # live portfolio view
    python -m dashboard.cli --trades     # recent trade history
    python -m dashboard.cli --equity     # equity curve (ASCII)
    python -m dashboard.cli --backtest AAPL MSFT   # run + print backtest
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.logger import get_recent_trades, get_equity_history, init_db
from utils.logger import get_logger

logger = get_logger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _colour(value: float, fmt: str = ".2f") -> str:
    s = f"{value:+{fmt}}"
    return f"{GREEN}{s}{RESET}" if value >= 0 else f"{RED}{s}{RESET}"


# ── Portfolio view ────────────────────────────────────────────────────────────

def show_portfolio():
    from execution.broker import Broker
    from execution.crypto_broker import CryptoBroker
    from config.settings import CRYPTO_ENABLED

    broker = Broker()
    account   = broker.get_account()
    positions = broker.get_positions()

    pv   = float(account.portfolio_value)
    cash = float(account.cash)
    invested = pv - cash

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}  Project Zeno — Unified Portfolio{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")
    print(f"  Portfolio value   : {BOLD}${pv:,.2f}{RESET}")
    print(f"  Cash              : ${cash:,.2f}")
    print(f"  Invested (stocks) : ${invested:,.2f}")
    print(f"  Stock positions   : {len(positions)}")
    print(f"{CYAN}{'─'*60}{RESET}")

    if positions:
        print(f"\n  {BOLD}── Stocks ──{RESET}")
        print(f"  {'Symbol':<10} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
        print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")
        for sym, pos in positions.items():
            qty            = int(float(pos.qty))
            cost_basis     = float(pos.avg_entry_price)
            current        = float(pos.current_price)
            unrealised     = float(pos.unrealized_pl)
            unrealised_pct = float(pos.unrealized_plpc) * 100
            print(
                f"  {sym:<10} {qty:>6} ${cost_basis:>9,.2f} ${current:>9,.2f} "
                f"{_colour(unrealised, ',.2f'):>20} {_colour(unrealised_pct, '.2f'):>16}%"
            )
    else:
        print(f"\n  {YELLOW}No stock positions.{RESET}")

    # ── Crypto positions ──
    if CRYPTO_ENABLED:
        try:
            crypto_broker     = CryptoBroker()
            crypto_positions  = crypto_broker.get_positions()
            print(f"\n  {BOLD}── Crypto (24/7) ──{RESET}")
            if crypto_positions:
                print(f"  {'Symbol':<10} {'Qty':>12} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
                print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")
                for sym, pos in crypto_positions.items():
                    qty            = float(pos.qty)
                    cost_basis     = float(pos.avg_entry_price)
                    current        = float(pos.current_price)
                    unrealised     = float(pos.unrealized_pl)
                    unrealised_pct = float(pos.unrealized_plpc) * 100
                    print(
                        f"  {sym:<10} {qty:>12.6f} ${cost_basis:>9,.2f} ${current:>9,.2f} "
                        f"{_colour(unrealised, ',.2f'):>20} {_colour(unrealised_pct, '.2f'):>16}%"
                    )
            else:
                print(f"  {YELLOW}No crypto positions.{RESET}")
        except Exception as e:
            print(f"  {RED}Crypto positions unavailable: {e}{RESET}")

    print()


# ── Recent trades ─────────────────────────────────────────────────────────────

def show_trades(limit: int = 20):
    init_db()
    trades = get_recent_trades(limit)

    print(f"\n{BOLD}{CYAN}{'─'*65}{RESET}")
    print(f"{BOLD}  Recent Fills (last {limit}){RESET}")
    print(f"{CYAN}{'─'*65}{RESET}")

    if not trades:
        print(f"  {YELLOW}No fills recorded yet.{RESET}\n")
        return

    print(f"  {'Time':<20} {'Symbol':<8} {'Side':<5} {'Qty':>5} {'Price':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*5} {'─'*5} {'─'*10}")
    for t in trades:
        side_col = f"{GREEN}BUY{RESET}" if t["side"] == "buy" else f"{RED}SELL{RESET}"
        print(f"  {t['ts'][:19]:<20} {t['symbol']:<8} {side_col:<14} {t['qty']:>5} ${t['fill_price']:>9,.2f}")
    print()


# ── Equity curve (ASCII sparkline) ───────────────────────────────────────────

def show_equity():
    init_db()
    history = get_equity_history()

    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}  Equity Curve{RESET}")
    print(f"{CYAN}{'─'*55}{RESET}")

    if len(history) < 2:
        print(f"  {YELLOW}Not enough equity snapshots yet.{RESET}\n")
        return

    values = [h["portfolio_value"] for h in history]
    dates  = [h["ts"][:10] for h in history]
    start  = values[0]
    end    = values[-1]
    total_return = (end - start) / start * 100

    min_v, max_v = min(values), max(values)
    height = 8
    width  = min(len(values), 60)
    step   = max(1, len(values) // width)
    sampled = values[::step][:width]

    # Normalise to height
    def _scale(v):
        if max_v == min_v:
            return height // 2
        return int((v - min_v) / (max_v - min_v) * (height - 1))

    bars = [_scale(v) for v in sampled]
    chart = []
    for row in range(height - 1, -1, -1):
        line = ""
        for b in bars:
            line += "█" if b >= row else " "
        chart.append(line)

    print(f"  ${max_v:,.0f} ┐")
    for line in chart:
        print(f"         │{line}")
    print(f"  ${min_v:,.0f} └{'─' * len(sampled)}")
    print(f"\n  Start  : ${start:,.2f}  →  End: ${end:,.2f}")
    print(f"  Return : {_colour(total_return, '.2f')}%")
    print(f"  Period : {dates[0]}  →  {dates[-1]}")
    print()


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(symbols: list[str], strategy: str, lookback: int):
    from data.fetcher import DataFetcher
    from backtest.engine import Backtester

    fetcher    = DataFetcher()
    backtester = Backtester(strategy_name=strategy)

    for sym in symbols:
        print(f"\n  Fetching {lookback} bars for {sym}...")
        bars = fetcher.get_bars([sym], lookback=lookback)
        if sym not in bars:
            print(f"  {RED}No data for {sym}{RESET}")
            continue
        result = backtester.run(sym, bars[sym])
        print(result.summary())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Zeno dashboard")
    parser.add_argument("--trades",   action="store_true", help="Show recent fills")
    parser.add_argument("--equity",   action="store_true", help="Show equity curve")
    parser.add_argument("--backtest", nargs="+", metavar="SYMBOL", help="Run backtest")
    parser.add_argument("--strategy", choices=["sma", "xgb"], default="sma")
    parser.add_argument("--lookback", type=int, default=500)
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest, args.strategy, args.lookback)
    elif args.trades:
        show_trades()
    elif args.equity:
        show_equity()
    else:
        show_portfolio()
        show_trades(10)
        show_equity()
