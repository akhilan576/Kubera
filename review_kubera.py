"""
Kubera Paper Trading Review
Run: python review_kubera.py
"""
from execution.india_paper_broker import IndiaPaperBroker
from data.yfinance_fetcher import YFinanceFetcher
from config.india_settings import INDIA_SYMBOLS, INDIA_STARTING_CASH
import pandas as pd

broker = IndiaPaperBroker()

# ── Trade Log ─────────────────────────────────────────────────────────────────
trades = broker.get_trade_log()
if not trades:
    print("No trades yet.")
else:
    df = pd.DataFrame(trades)
    df["ts"] = pd.to_datetime(df["ts"])
    df["pnl"] = df["pnl"].astype(float)

    print("\n══════════════════ KUBERA TRADE LOG ══════════════════")
    print(df[["ts", "symbol", "side", "qty", "price", "pnl"]].to_string(index=False))

    sells = df[df["side"] == "sell"]
    total_pnl    = sells["pnl"].sum()
    win_trades   = sells[sells["pnl"] > 0]
    loss_trades  = sells[sells["pnl"] < 0]
    win_rate     = len(win_trades) / len(sells) * 100 if len(sells) else 0
    avg_win      = win_trades["pnl"].mean() if len(win_trades) else 0
    avg_loss     = loss_trades["pnl"].mean() if len(loss_trades) else 0

    print("\n══════════════════ SUMMARY ═══════════════════════════")
    print(f"  Total trades  : {len(df)} ({len(df[df['side']=='buy'])} buys, {len(sells)} sells)")
    print(f"  Total P&L     : ₹{total_pnl:+,.2f}")
    print(f"  Win rate      : {win_rate:.1f}%  ({len(win_trades)}W / {len(loss_trades)}L)")
    print(f"  Avg win       : ₹{avg_win:+,.2f}")
    print(f"  Avg loss      : ₹{avg_loss:+,.2f}")

# ── Current Positions ─────────────────────────────────────────────────────────
positions = broker.get_positions()
print("\n══════════════════ OPEN POSITIONS ════════════════════")
if not positions:
    print("  No open positions.")
else:
    fetcher = YFinanceFetcher(interval="15Min")
    syms = list(positions.keys())
    bars = fetcher.get_bars(syms, lookback=2)
    current_prices = {s: float(bars[s]["close"].iloc[-1]) for s in syms if s in bars}

    for sym, pos in positions.items():
        price     = current_prices.get(sym, pos["avg_entry"])
        unrealised = (price - pos["avg_entry"]) * pos["qty"]
        print(f"  {sym:15s} qty={int(pos['qty']):4d}  entry=₹{pos['avg_entry']:,.2f}  "
              f"now=₹{price:,.2f}  unrealised=₹{unrealised:+,.2f}")

# ── Portfolio ─────────────────────────────────────────────────────────────────
current_prices_all = {**{s: p["avg_entry"] for s, p in positions.items()}, **current_prices} if positions else {}
portfolio_value = broker.get_portfolio_value(current_prices_all)
returns_pct     = (portfolio_value - INDIA_STARTING_CASH) / INDIA_STARTING_CASH * 100

print("\n══════════════════ PORTFOLIO ══════════════════════════")
print(f"  Starting cash  : ₹{INDIA_STARTING_CASH:,.2f}")
print(f"  Current value  : ₹{portfolio_value:,.2f}")
print(f"  Cash available : ₹{broker.get_cash():,.2f}")
print(f"  Total return   : {returns_pct:+.2f}%")
print("═══════════════════════════════════════════════════════\n")
