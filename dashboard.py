"""
Kubera Dashboard
Run once: python dashboard.py
Live mode: python dashboard.py --live   (refreshes every 60s during market hours)
"""
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
import pytz

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import box

import logging
logging.disable(logging.CRITICAL)

from execution.india_paper_broker import IndiaPaperBroker
from data.yfinance_fetcher import YFinanceFetcher
from config.india_settings import INDIA_STARTING_CASH

console = Console()
IST = pytz.timezone("Asia/Kolkata")
BAR_WIDTH = 18


def colour_pnl(val: float, fmt: str = "₹{:+,.2f}") -> Text:
    return Text(fmt.format(val), style="bold green" if val >= 0 else "bold red")


def colour_pct(val: float) -> Text:
    return Text(f"{val:+.2f}%", style="bold green" if val >= 0 else "bold red")


def fetch_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    fetcher = YFinanceFetcher(interval="15Min")
    bars = fetcher.get_bars(symbols, lookback=2)
    return {s: float(bars[s]["close"].iloc[-1]) for s in symbols if s in bars}


def exposure_bar(value: float, max_value: float, width: int = BAR_WIDTH) -> Text:
    if max_value <= 0:
        filled = 0
    else:
        filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    return Text(bar, style="gold1")


def render(broker: IndiaPaperBroker):
    os.system("clear")
    now = datetime.now(IST).strftime("%A, %d %b %Y  %H:%M:%S IST")

    # ── Fetch data ────────────────────────────────────────────────────────────
    positions       = broker.get_positions()
    short_positions = broker.get_short_positions()
    all_syms        = list(set(list(positions.keys()) + list(short_positions.keys())))
    prices          = fetch_prices(all_syms)

    # ── Metrics ───────────────────────────────────────────────────────────────
    cash          = broker.get_cash()
    long_invested = sum(p["qty"] * p["avg_entry"] for p in positions.values())
    long_equity   = sum(p["qty"] * prices.get(s, p["avg_entry"]) for s, p in positions.items())
    short_unreal  = sum((p["avg_entry"] - prices.get(s, p["avg_entry"])) * abs(p["qty"])
                        for s, p in short_positions.items())
    portfolio_val  = broker.get_portfolio_value(prices)
    unrealised_pnl = (long_equity - long_invested) + short_unreal
    total_return   = portfolio_val - INDIA_STARTING_CASH
    ret_pct        = total_return / INDIA_STARTING_CASH * 100

    trades       = broker.get_trade_log()
    closed       = [t for t in trades if t["side"] in ("sell", "cover")]
    realised_pnl = sum(t["pnl"] for t in closed)
    wins         = [t for t in closed if t["pnl"] > 0]
    losses       = [t for t in closed if t["pnl"] < 0]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0.0
    net_pnl      = realised_pnl + unrealised_pnl

    # ── Header ────────────────────────────────────────────────────────────────
    header_tbl = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
    header_tbl.add_column("", style="bold gold1",  ratio=1)
    header_tbl.add_column("", style="dim white",   ratio=3, justify="center")
    header_tbl.add_column("", style="dim",         ratio=2, justify="right")
    header_tbl.add_row("KUBERA", "India Paper Trading Dashboard", now)
    console.print(Panel(header_tbl, box=box.HEAVY, style="gold1", padding=(0, 1)))

    # ── Metric cards ──────────────────────────────────────────────────────────
    def card(title: str, value: str, style: str = "bold white") -> Panel:
        return Panel(
            Align(Text.from_markup(f"[{style}]{value}[/]"), align="left"),
            title=f"[dim]{title}[/dim]",
            box=box.SIMPLE, padding=(0, 1)
        )

    ret_style = "bold green" if total_return >= 0 else "bold red"
    unr_style = "bold green" if unrealised_pnl >= 0 else "bold red"
    rea_style = "bold green" if realised_pnl >= 0 else "bold red"

    cards = [
        card("Portfolio Value",  f"₹{portfolio_val:,.2f}",                         "bold white"),
        card("Cash Available",   f"₹{cash:,.2f}",                                  "white"),
        card("Invested",         f"₹{long_invested:,.2f}",                         "yellow"),
        card("Unrealised P&L",   f"₹{unrealised_pnl:+,.2f}",                      unr_style),
        card("Realised P&L",     f"₹{realised_pnl:+,.2f}",                        rea_style),
        card("Total Return",     f"₹{total_return:+,.2f}\n[dim]{ret_pct:+.2f}%[/dim]", ret_style),
    ]
    console.print(Columns(cards, equal=True, expand=True))

    # ── Build position rows ───────────────────────────────────────────────────
    pos_rows = []
    for sym, pos in positions.items():
        qty     = int(pos["qty"])
        entry   = pos["avg_entry"]
        cur     = prices.get(sym, entry)
        unreal  = (cur - entry) * qty
        ret_p   = (cur - entry) / entry * 100
        exposure = qty * cur
        pos_rows.append((sym, "LONG", qty, entry, cur, unreal, ret_p, exposure))

    for sym, pos in short_positions.items():
        qty     = int(abs(pos["qty"]))
        entry   = pos["avg_entry"]
        cur     = prices.get(sym, entry)
        unreal  = (entry - cur) * qty
        ret_p   = (entry - cur) / entry * 100
        exposure = qty * entry
        pos_rows.append((sym, "SHORT", qty, entry, cur, unreal, ret_p, exposure))

    total_positions = len(pos_rows)
    max_exposure    = max((r[7] for r in pos_rows), default=1)

    # ── Open Positions table ──────────────────────────────────────────────────
    pos_tbl = Table(
        box=box.SIMPLE_HEAVY, show_edge=False,
        header_style="bold gold1", expand=True,
        title=f"[bold white]Open Positions[/bold white]  [dim]{total_positions} positions[/dim]",
        title_justify="left",
    )
    pos_tbl.add_column("Symbol",     style="bold white", no_wrap=True)
    pos_tbl.add_column("Side",       justify="center",   no_wrap=True)
    pos_tbl.add_column("Qty",        justify="right")
    pos_tbl.add_column("Avg Entry",  justify="right")
    pos_tbl.add_column("Current",    justify="right")
    pos_tbl.add_column("Unreal P&L", justify="right")
    pos_tbl.add_column("Return",     justify="right")

    if not pos_rows:
        pos_tbl.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")
    else:
        for sym, side, qty, entry, cur, unreal, ret_p, _ in pos_rows:
            side_text = Text("LONG", style="bold green") if side == "LONG" else Text("SHORT", style="bold red")
            pos_tbl.add_row(
                sym, side_text, str(qty),
                f"₹{entry:,.2f}", f"₹{cur:,.2f}",
                colour_pnl(unreal), colour_pct(ret_p),
            )

    # ── Trade Summary panel ───────────────────────────────────────────────────
    sum_tbl = Table(box=None, show_header=False, show_edge=False, padding=(0, 1), expand=True)
    sum_tbl.add_column("", style="dim",        ratio=1)
    sum_tbl.add_column("", style="bold white", ratio=1, justify="right")
    sum_tbl.add_column("", style="dim",        ratio=1)
    sum_tbl.add_column("", style="bold white", ratio=1, justify="right")

    n_buys   = len([t for t in trades if t["side"] == "buy"])
    n_sells  = len([t for t in trades if t["side"] == "sell"])
    n_shorts = len([t for t in trades if t["side"] == "short"])
    n_covers = len([t for t in trades if t["side"] == "cover"])
    avg_win  = f"₹{sum(t['pnl'] for t in wins)/len(wins):+,.2f}"  if wins   else "—"
    avg_loss = f"₹{sum(t['pnl'] for t in losses)/len(losses):+,.2f}" if losses else "—"

    sum_tbl.add_row("Total trades", str(len(trades)), "Buys",     str(n_buys))
    sum_tbl.add_row("Sells",        str(n_sells),     "Shorts",   str(n_shorts))
    sum_tbl.add_row("Covers",       str(n_covers),    "Win rate", f"{win_rate:.1f}%")
    sum_tbl.add_row("Avg win",      avg_win,           "Avg loss", avg_loss)

    # Profit/loss bar
    total_profit = sum(t["pnl"] for t in wins)
    total_loss   = abs(sum(t["pnl"] for t in losses))
    combined     = total_profit + total_loss
    profit_fill  = int((total_profit / combined) * BAR_WIDTH) if combined > 0 else 0
    loss_fill    = BAR_WIDTH - profit_fill
    pnl_bar      = Text("█" * profit_fill, style="green") + Text("█" * loss_fill, style="red")

    sum_tbl.add_row("", Text(""), "", Text(""))
    sum_tbl.add_row("profit", pnl_bar, "", Text(""))
    sum_tbl.add_row("losing", Text(""), "", Text(""))
    sum_tbl.add_row("net", colour_pnl(net_pnl), "", Text(""))

    # ── Exposure panel ────────────────────────────────────────────────────────
    exp_tbl = Table(box=None, show_header=False, show_edge=False, padding=(0, 1), expand=True)
    exp_tbl.add_column("", style="bold white", no_wrap=True, min_width=10)
    exp_tbl.add_column("", ratio=1)

    for sym, _, _, _, _, _, _, exposure in sorted(pos_rows, key=lambda x: x[7], reverse=True):
        exp_tbl.add_row(sym, exposure_bar(exposure, max_exposure))

    # Right panel: Trade Summary + Exposure
    right_panel = Panel(
        Columns([
            Panel(sum_tbl, title="[bold white]Trade Summary[/bold white]",
                  box=box.SIMPLE, padding=(0, 1)),
        ], expand=True),
        box=box.SIMPLE, padding=(0, 0)
    )

    right_content = Table(box=None, show_header=False, show_edge=False, expand=True)
    right_content.add_column("", ratio=1)
    right_content.add_row(Panel(sum_tbl, title="[bold white]Trade Summary[/bold white]",
                                box=box.SIMPLE, padding=(0,1)))
    right_content.add_row(Panel(exp_tbl, title="[bold white]Exposure[/bold white]",
                                box=box.SIMPLE, padding=(0,1)))

    # ── Two-column layout ─────────────────────────────────────────────────────
    main_tbl = Table(box=None, show_header=False, show_edge=False, expand=True, padding=0)
    main_tbl.add_column("", ratio=3)
    main_tbl.add_column("", ratio=2)
    main_tbl.add_row(
        Panel(pos_tbl,       box=box.SIMPLE, padding=(0, 1)),
        Panel(right_content, box=box.SIMPLE, padding=(0, 1)),
    )
    console.print(main_tbl)

    # ── Recent Trades ─────────────────────────────────────────────────────────
    if trades:
        recent = trades[-10:][::-1]
        rtbl = Table(
            box=box.SIMPLE_HEAVY, show_edge=False, expand=True,
            header_style="bold gold1",
            title="[bold white]Recent Trades[/bold white]  [dim]last 10[/dim]",
            title_justify="left",
        )
        rtbl.add_column("Time",   no_wrap=True)
        rtbl.add_column("Symbol", style="bold white")
        rtbl.add_column("Side",   justify="center")
        rtbl.add_column("Qty",    justify="right")
        rtbl.add_column("Price",  justify="right")
        rtbl.add_column("P&L",    justify="right")

        side_styles = {
            "buy":   Text("BUY",   style="bold green"),
            "sell":  Text("SELL",  style="bold yellow"),
            "short": Text("SHORT", style="bold red"),
            "cover": Text("COVER", style="bold cyan"),
        }
        for t in recent:
            ts   = datetime.fromisoformat(t["ts"]).strftime("%d %b %H:%M")
            side = side_styles.get(t["side"], Text(t["side"].upper()))
            pnl  = colour_pnl(t["pnl"]) if t["side"] in ("sell", "cover") else Text("—", style="dim")
            rtbl.add_row(ts, t["symbol"], side, str(int(t["qty"])), f"₹{t['price']:,.2f}", pnl)

        console.print(Panel(rtbl, box=box.SIMPLE, padding=(0, 1)))

    console.print(f"[dim]  Starting capital: ₹{INDIA_STARTING_CASH:,.2f}[/dim]\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    args = parser.parse_args()

    broker = IndiaPaperBroker()

    if args.live:
        try:
            while True:
                render(broker)
                time.sleep(60)
        except KeyboardInterrupt:
            console.print("\n[dim]Exited.[/dim]")
    else:
        render(broker)


if __name__ == "__main__":
    main()
