"""
Kubera TUI — Interactive paper-trading terminal
────────────────────────────────────────────────────────────────────────────
Run:  python tui.py

Commands (type at the bottom prompt, press Enter):
  buy <sym> <qty>          Enter long at market price
  short <sym> <qty>        Enter short at market price
  sell <sym> <qty>         Exit long
  cover <sym> <qty>        Cover short
  x <sym>                  Close entire position (long or short)
  :export                  Dump trades → ~/kubera/sessions/YYYY-MM-DD.csv
  :note <text>             Add journal entry (stored in DB)
  :tag <sym> <text>        Tag a position in the journal

Keyboard shortcuts (when command prompt is not focused):
  s    Cycle sort: symbol → pnl → qty → age
  ?    Show keybindings in the order log
  q    Quit
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytz
from rich import box as rbox
from rich.align import Align
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Input, Label, RichLog, Static
from textual import events

from execution.india_paper_broker import IndiaPaperBroker
from data.yfinance_fetcher import YFinanceFetcher
from config.india_settings import INDIA_STARTING_CASH

import logging
logging.disable(logging.CRITICAL)

IST        = pytz.timezone("Asia/Kolkata")
REFRESH_S  = 30
SORT_KEYS  = ["symbol", "pnl", "qty", "age"]
BLOCKS     = "▁▂▃▄▅▆▇█"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fetch_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    fetcher = YFinanceFetcher(interval="15Min")
    bars = fetcher.get_bars(symbols, lookback=2)
    return {s: float(bars[s]["close"].iloc[-1]) for s in symbols if s in bars}


def _age(ts_iso: str) -> str:
    try:
        dt    = datetime.fromisoformat(ts_iso)
        secs  = int((datetime.now() - dt).total_seconds())
        h, r  = divmod(secs, 3600)
        return f"{h:02d}:{r // 60:02d}"
    except Exception:
        return "--:--"


def _sparkline(prices: list[float], width: int = 8) -> str:
    if len(prices) < 2:
        return "─" * width
    mn, mx = min(prices), max(prices)
    if mx == mn:
        return "─" * width
    return "".join(BLOCKS[min(7, int((p - mn) / (mx - mn) * 8))] for p in prices[-width:])


def _equity_chart(history: list[float], width: int = 34, height: int = 4) -> list[str]:
    if len(history) < 2:
        return ["  (no history yet)"] + [""] * (height - 1)
    mn, mx = min(history), max(history)
    rng = mx - mn or 1.0
    recent = history[-width:]
    rows = []
    for row_idx in range(height):
        chars = []
        for i, val in enumerate(recent):
            cur_y  = height - 1 - int(((val - mn) / rng) * (height - 1))
            prev_y = cur_y
            if i > 0:
                prev_v = recent[i - 1]
                prev_y = height - 1 - int(((prev_v - mn) / rng) * (height - 1))
            if row_idx == cur_y:
                chars.append("╱" if prev_y > cur_y else ("╲" if prev_y < cur_y else "─"))
            else:
                chars.append(" ")
        rows.append("".join(chars))
    return rows


# ─── App ──────────────────────────────────────────────────────────────────────

class KuberaTUI(App):

    CSS = """
    Screen {
        background: #0a0a0a;
        layout: vertical;
    }

    #header {
        height: 1;
        background: #150800;
        color: gold;
        padding: 0 1;
        content-align: left middle;
    }

    #metrics {
        height: 6;
        padding: 0 0;
        border-bottom: solid #2a2a2a;
        background: #0d0d0d;
    }

    #main {
        height: 1fr;
        layout: horizontal;
    }

    #left-panel {
        width: 3fr;
        height: 1fr;
    }

    #right-panel {
        width: 2fr;
        height: 1fr;
        layout: vertical;
        border-left: solid #2a2a2a;
    }

    #analytics {
        height: auto;
        max-height: 16;
        padding: 0 1;
        border-bottom: solid #2a2a2a;
    }

    #order-log {
        height: 1fr;
        padding: 0 1;
    }

    #cmd-bar {
        height: 3;
        layout: horizontal;
        background: #111111;
        align: left middle;
        padding: 0 1;
        border-top: solid #2a2a2a;
    }

    #cmd-prompt {
        width: auto;
        padding: 0 1 0 0;
        color: gold;
    }

    #cmd-input {
        width: 1fr;
        background: transparent;
        border: none;
        padding: 0;
        color: white;
    }

    Input:focus {
        border: none;
    }

    #status {
        height: 1;
        background: #0a0a0a;
        color: #444444;
        padding: 0 1;
        content-align: left middle;
        border-top: solid #1a1a1a;
    }
    """

    sort_col: reactive[int] = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self.broker   = IndiaPaperBroker()
        self._prices: dict[str, float] = {}

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield Static("", id="metrics")
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield DataTable(id="positions", zebra_stripes=True, cursor_type="row")
            with Vertical(id="right-panel"):
                yield Static("", id="analytics")
                yield RichLog(id="order-log", markup=True, highlight=False, max_lines=500)
        with Horizontal(id="cmd-bar"):
            yield Label("[bold gold1]>[/bold gold1]", id="cmd-prompt")
            yield Input(
                id="cmd-input",
                placeholder="buy NTPC 10  |  short SBIN 5  |  x HDFCBANK  |  :export  |  :note ...",
            )
        yield Static("", id="status")

    def on_mount(self) -> None:
        self._setup_columns()
        self._reload_log()
        self.set_interval(REFRESH_S, self._refresh)
        self.set_interval(1.0, self._tick)
        self._refresh()
        self.query_one("#cmd-input", Input).focus()

    # ── Column setup ──────────────────────────────────────────────────────────

    def _setup_columns(self) -> None:
        t = self.query_one("#positions", DataTable)
        t.add_column("Symbol",  key="symbol",  width=12)
        t.add_column("Side",    key="side",    width=6)
        t.add_column("Qty",     key="qty",     width=6)
        t.add_column("Entry",   key="entry",   width=10)
        t.add_column("Current", key="current", width=10)
        t.add_column("Age",     key="age",     width=6)
        t.add_column("PnL",     key="pnl",     width=12)
        t.add_column("Ret%",    key="ret",     width=8)
        t.add_column("Risk%",   key="risk",    width=6)
        t.add_column("SL",      key="sl",      width=10)
        t.add_column("▾Spark",  key="spark",   width=9)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        longs  = self.broker.get_positions()
        shorts = self.broker.get_short_positions()
        syms   = list(set(list(longs) + list(shorts)))
        if syms:
            self._prices = _fetch_prices(syms)
        pval = self.broker.get_portfolio_value(self._prices)
        self.broker.log_portfolio_value(pval)
        if self._prices:
            self.broker.log_prices(self._prices)
        self._update_metrics(pval)
        self._update_table(longs, shorts, pval)
        self._update_analytics(pval)
        self._update_status(pval)

    def _tick(self) -> None:
        now  = datetime.now(IST).strftime("%A, %d %b %Y  %H:%M:%S IST")
        skey = SORT_KEYS[self.sort_col % len(SORT_KEYS)]
        self.query_one("#header", Static).update(
            f"[bold gold1]KUBERA[/bold gold1]  "
            f"[dim]Interactive Trading Terminal[/dim]  "
            f"[dim]│[/dim]  [dim]{now}[/dim]  "
            f"[dim]│  sort: {skey}[/dim]"
        )

    # ── Metric cards ──────────────────────────────────────────────────────────

    def _update_metrics(self, pval: float) -> None:
        cash   = self.broker.get_cash()
        longs  = self.broker.get_positions()
        shorts = self.broker.get_short_positions()
        n_pos  = len(longs) + len(shorts)

        long_invested  = sum(p["qty"] * p["avg_entry"] for p in longs.values())
        short_exposure = sum(abs(p["qty"]) * p["avg_entry"] for p in shorts.values())
        total_invested = long_invested + short_exposure

        unrealised = (
            sum((self._prices.get(s, p["avg_entry"]) - p["avg_entry"]) * p["qty"]
                for s, p in longs.items())
            + sum((p["avg_entry"] - self._prices.get(s, p["avg_entry"])) * abs(p["qty"])
                  for s, p in shorts.items())
        )
        trades   = self.broker.get_trade_log()
        closed   = [t for t in trades if t["side"] in ("sell", "cover")]
        realised = sum(t["pnl"] for t in closed)
        total_ret = pval - INDIA_STARTING_CASH
        ret_pct   = total_ret / INDIA_STARTING_CASH * 100
        unr_pct   = unrealised / INDIA_STARTING_CASH * 100

        def card(title: str, val: str, sub: str, val_style: str = "bold white") -> Panel:
            body = Text.from_markup(f"[{val_style}]{val}[/]\n[dim]{sub}[/dim]")
            return Panel(
                Align(body, align="left"),
                title=f"[dim]{title}[/dim]",
                box=rbox.SIMPLE, padding=(0, 1),
            )

        ret_s = "bold green" if total_ret >= 0 else "bold red"
        unr_s = "bold green" if unrealised  >= 0 else "bold red"
        rea_s = "bold green" if realised    >= 0 else "bold red"

        cards = [
            card("PORTFOLIO",      f"₹{pval:,.2f}",          f"starting ₹{INDIA_STARTING_CASH:,.0f}", ret_s),
            card("CASH",           f"₹{cash:,.2f}",           "available",                              "bold white"),
            card("INVESTED",       f"₹{total_invested:,.2f}", f"{n_pos} positions",                     "yellow"),
            card("UNREALISED P&L", f"₹{unrealised:+,.2f}",   f"{unr_pct:+.2f}% return",               unr_s),
            card("REALISED P&L",   f"₹{realised:+,.2f}",     f"{len(closed)} closed",                  rea_s),
            card("TOTAL P&L",      f"₹{total_ret:+,.2f}",    f"{ret_pct:+.2f}% session",              ret_s),
        ]
        self.query_one("#metrics", Static).update(
            Columns(cards, equal=True, expand=True)
        )

    # ── Positions table ───────────────────────────────────────────────────────

    def _update_table(self, longs: dict, shorts: dict, pval: float) -> None:
        t = self.query_one("#positions", DataTable)
        t.clear()

        # Entry timestamp per symbol (first buy/short trade)
        first_ts: dict[str, str] = {}
        for tr in self.broker.get_trade_log():
            if tr["side"] in ("buy", "short") and tr["symbol"] not in first_ts:
                first_ts[tr["symbol"]] = tr["ts"]

        stops = self.broker.get_stops()
        rows: list[dict] = []

        for sym, pos in longs.items():
            qty   = int(pos["qty"])
            entry = pos["avg_entry"]
            cur   = self._prices.get(sym, entry)
            pnl   = (cur - entry) * qty
            ret_p = (cur - entry) / entry * 100
            risk  = qty * cur / pval * 100 if pval > 0 else 0
            rows.append(dict(symbol=sym, side="LONG",  qty=qty, entry=entry,
                             cur=cur, age=_age(first_ts.get(sym, "")),
                             pnl=pnl, ret_p=ret_p, risk=risk,
                             sl=stops.get(sym, {}).get("stop"),
                             ph=self.broker.get_price_history(sym, n=8)))

        for sym, pos in shorts.items():
            qty   = int(abs(pos["qty"]))
            entry = pos["avg_entry"]
            cur   = self._prices.get(sym, entry)
            pnl   = (entry - cur) * qty
            ret_p = (entry - cur) / entry * 100
            risk  = qty * entry / pval * 100 if pval > 0 else 0
            rows.append(dict(symbol=sym, side="SHORT", qty=qty, entry=entry,
                             cur=cur, age=_age(first_ts.get(sym, "")),
                             pnl=pnl, ret_p=ret_p, risk=risk,
                             sl=stops.get(sym, {}).get("stop"),
                             ph=self.broker.get_price_history(sym, n=8)))

        # Sort
        sk_map = {"symbol": "symbol", "pnl": "pnl", "qty": "qty", "age": "age"}
        sk = sk_map[SORT_KEYS[self.sort_col % len(SORT_KEYS)]]
        rows.sort(key=lambda r: r[sk])

        if not rows:
            t.add_row("[dim]No open positions[/dim]", *[""] * 10)
            return

        for r in rows:
            side_t = Text("LONG",  style="bold green") if r["side"] == "LONG" \
                else Text("SHORT", style="bold red")
            pnl_t  = Text(f"₹{r['pnl']:+,.2f}",    style="bold green" if r["pnl"]  >= 0 else "bold red")
            ret_t  = Text(f"{r['ret_p']:+.2f}%",     style="bold green" if r["ret_p"] >= 0 else "bold red")
            sl_str = f"₹{r['sl']:,.0f}" if r["sl"] else "—"
            t.add_row(
                r["symbol"], side_t, str(r["qty"]),
                f"₹{r['entry']:,.2f}", f"₹{r['cur']:,.2f}",
                r["age"], pnl_t, ret_t,
                f"{r['risk']:.1f}%", sl_str, _sparkline(r["ph"]),
            )

    # ── Analytics panel ───────────────────────────────────────────────────────

    def _update_analytics(self, pval: float) -> None:
        hist_rows = self.broker.get_portfolio_history(n=40)
        hist      = [r["value"] for r in hist_rows]
        trades    = self.broker.get_trade_log()
        closed    = [t for t in trades if t["side"] in ("sell", "cover")]
        wins      = [t for t in closed if t["pnl"] > 0]
        losses    = [t for t in closed if t["pnl"] < 0]
        win_rate  = len(wins) / len(closed) * 100 if closed else 0.0

        # Max drawdown
        peak, max_dd, dd_at = INDIA_STARTING_CASH, 0.0, ""
        for row in hist_rows:
            v = row["value"]
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
                dd_at  = row["ts"][11:16]
        dd_pct = max_dd / peak * 100 if peak > 0 else 0

        total_ret = pval - INDIA_STARTING_CASH
        ret_pct   = total_ret / INDIA_STARTING_CASH * 100
        rc        = "green" if total_ret >= 0 else "red"

        chart = _equity_chart(hist, width=34, height=4)
        chart_str = "\n".join(f"  [dim]{ln}[/dim]" for ln in chart)

        avg_w = f"₹{sum(t['pnl'] for t in wins)/len(wins):+,.2f}"    if wins   else "—"
        avg_l = f"₹{sum(t['pnl'] for t in losses)/len(losses):+,.2f}" if losses else "—"

        dd_line = f"  [dim]Max DD:[/dim] [red]₹{max_dd:,.2f} ({dd_pct:.2f}%)[/red]"
        if dd_at:
            dd_line += f"  [dim]{dd_at}[/dim]"
        body = (
            f"[bold white]Equity Curve[/bold white]\n"
            f"{chart_str}\n"
            f"{dd_line}\n"
            f"  [dim]Sharpe: —   Sortino: —[/dim]\n"
            f"\n[bold white]Trade Summary[/bold white]\n"
            f"  [dim]Total[/dim] {len(trades)}  "
            f"[dim]Wins[/dim] [green]{len(wins)}[/green]  "
            f"[dim]Losses[/dim] [red]{len(losses)}[/red]  "
            f"[dim]Rate[/dim] {win_rate:.1f}%\n"
            f"  [dim]Avg win[/dim] {avg_w}  [dim]Avg loss[/dim] {avg_l}\n"
            f"  [dim]Net[/dim] [{rc}]₹{total_ret:+,.2f} ({ret_pct:+.2f}%)[/{rc}]"
        )
        self.query_one("#analytics", Static).update(body)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self, pval: float) -> None:
        ret_pct  = (pval - INDIA_STARTING_CASH) / INDIA_STARTING_CASH * 100
        dd_col   = "red" if ret_pct < -2 else ("yellow" if ret_pct < 0 else "green")
        risk_str = "[red]RISK: HALT[/red]" if ret_pct < -3 else "[green]RISK: OK[/green]"

        # Build contextual quick-command hints from open positions
        shorts = list(self.broker.get_short_positions().keys())
        longs  = list(self.broker.get_positions().keys())
        hints  = []
        if shorts:
            hints.append(f"x {shorts[0]}")
        if longs:
            hints.append(f"sell {longs[0]}")
        hints += [":export", ":note ..."]
        hint_str = "  [dim]|[/dim]  ".join(f"[dim]{h}[/dim]" for h in hints)

        self.query_one("#status", Static).update(
            f" {risk_str}  [dim]│[/dim]"
            f"  DD [{dd_col}]{ret_pct:+.2f}%[/{dd_col}]  [dim]│[/dim]"
            f"  {hint_str}  [dim]│[/dim]"
            f"  [dim][s] sort  [?] help  [q] quit[/dim]"
        )

    # ── Order log ─────────────────────────────────────────────────────────────

    SIDE_FMT = {
        "buy":   "[bold green]BUY  [/bold green]",
        "sell":  "[bold yellow]SELL [/bold yellow]",
        "short": "[bold red]SHORT[/bold red]",
        "cover": "[bold cyan]COVER[/bold cyan]",
    }
    ACT_FMT = {
        "BOUGHT":  "[bold green]BOUGHT [/bold green]",
        "SOLD":    "[bold yellow]SOLD   [/bold yellow]",
        "SHORTED": "[bold red]SHORTED[/bold red]",
        "COVERED": "[bold cyan]COVERED[/bold cyan]",
    }

    def _reload_log(self) -> None:
        log = self.query_one("#order-log", RichLog)
        log.clear()
        log.write("[bold gold1]── Order Log ─────────────────────────────────────────────────[/bold gold1]")
        for t in self.broker.get_trade_log()[-50:]:
            ts    = datetime.fromisoformat(t["ts"]).strftime("%H:%M")
            side  = self.SIDE_FMT.get(t["side"], t["side"].upper())
            pnl_s = ""
            if t["side"] in ("sell", "cover"):
                c     = "green" if t["pnl"] >= 0 else "red"
                pnl_s = f"  [{c}]P&L ₹{t['pnl']:+,.2f}[/{c}]"
            log.write(
                f"[dim]{ts}[/dim]  {side}  [bold white]{t['symbol']:<10}[/bold white]"
                f" x{int(t['qty'])}  [dim]@[/dim] ₹{t['price']:,.2f}{pnl_s}"
            )

    def _log_action(self, action: str, sym: str, qty: int, price: float, pnl: float = 0) -> None:
        log   = self.query_one("#order-log", RichLog)
        ts    = datetime.now(IST).strftime("%H:%M")
        act   = self.ACT_FMT.get(action, f"[white]{action}[/white]")
        pnl_s = ""
        if action in ("SOLD", "COVERED"):
            c     = "green" if pnl >= 0 else "red"
            pnl_s = f"  [{c}]P&L ₹{pnl:+,.2f}[/{c}]"
        log.write(
            f"[dim]{ts}[/dim]  {act}  [bold white]{sym:<10}[/bold white]"
            f" x{qty}  [dim]@[/dim] ₹{price:,.2f}{pnl_s}"
        )

    # ── Command parsing ───────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        self.query_one("#cmd-input", Input).value = ""
        if raw:
            self._run(raw)

    def _run(self, raw: str) -> None:  # noqa: C901 (complexity OK for a command parser)
        log   = self.query_one("#order-log", RichLog)
        parts = raw.strip().split()
        verb  = parts[0].lower() if parts else ""

        longs  = self.broker.get_positions()
        shorts = self.broker.get_short_positions()

        def live(sym: str) -> float | None:
            if sym in self._prices:
                return self._prices[sym]
            p = _fetch_prices([sym]).get(sym)
            if p:
                self._prices[sym] = p
            return p

        try:
            # ── x <sym> ── close entire position ─────────────────────────────
            if verb == "x" and len(parts) >= 2:
                sym = parts[1].upper()
                p   = live(sym)
                if p is None:
                    log.write(f"[red]No price for {sym}[/red]")
                    return
                if sym in longs:
                    qty = int(longs[sym]["qty"])
                    pnl = (p - longs[sym]["avg_entry"]) * qty
                    self.broker.sell(sym, qty, p)
                    self.broker.clear_stop(sym)
                    self._log_action("SOLD", sym, qty, p, pnl)
                elif sym in shorts:
                    qty = int(abs(shorts[sym]["qty"]))
                    pnl = (shorts[sym]["avg_entry"] - p) * qty
                    self.broker.cover(sym, qty, p)
                    self.broker.clear_stop(sym)
                    self._log_action("COVERED", sym, qty, p, pnl)
                else:
                    log.write(f"[dim]No position in {sym}[/dim]")
                    return
                self._refresh()
                return

            # ── buy/short/sell/cover <sym> <qty> [@ <price>] ─────────────────
            if verb in ("buy", "short", "sell", "cover") and len(parts) >= 3:
                sym = parts[1].upper()
                qty = int(parts[2])
                if len(parts) >= 5 and parts[3] == "@":
                    p = float(parts[4].replace("₹", "").replace(",", ""))
                else:
                    p = live(sym)
                if p is None:
                    log.write(f"[red]No price for {sym}[/red]")
                    return

                if verb == "buy":
                    self.broker.buy(sym, qty, p)
                    self._log_action("BOUGHT", sym, qty, p)
                elif verb == "short":
                    self.broker.short(sym, qty, p)
                    stop = p * 1.02
                    self.broker.set_stop(sym, stop)
                    self._log_action("SHORTED", sym, qty, p)
                elif verb == "sell":
                    pos = self.broker.get_positions().get(sym, {})
                    pnl = (p - pos.get("avg_entry", p)) * qty
                    self.broker.sell(sym, qty, p)
                    self.broker.clear_stop(sym)
                    self._log_action("SOLD", sym, qty, p, pnl)
                elif verb == "cover":
                    pos = self.broker.get_short_positions().get(sym, {})
                    pnl = (pos.get("avg_entry", p) - p) * qty
                    self.broker.cover(sym, qty, p)
                    self.broker.clear_stop(sym)
                    self._log_action("COVERED", sym, qty, p, pnl)
                self._refresh()
                return

            # ── :export ───────────────────────────────────────────────────────
            if verb in (":export", "export"):
                path = self._export()
                log.write(f"[bold green]:export[/bold green]  → [dim]{path}[/dim]")
                return

            # ── :note <text> ──────────────────────────────────────────────────
            if verb == ":note" or raw.startswith(":note "):
                note = raw[len(":note"):].strip()
                self.broker.add_journal(note)
                log.write(f"[bold white]:note[/bold white]  [dim]{note}[/dim]")
                return

            # ── :tag <sym> <text> ─────────────────────────────────────────────
            if verb == ":tag" and len(parts) >= 3:
                sym  = parts[1].upper()
                text = " ".join(parts[2:])
                self.broker.add_journal(f"[{sym}] {text}")
                log.write(f"[bold white]:tag[/bold white]  {sym}  [dim]{text}[/dim]")
                return

            log.write(
                f"[red]Unknown:[/red] {raw}\n"
                "  [dim]buy/short/sell/cover <sym> <qty> [@price]  |  "
                "x <sym>  |  :export  |  :note <text>  |  :tag <sym> <text>[/dim]"
            )

        except ValueError as exc:
            log.write(f"[bold red]Error:[/bold red] {exc}")
        except Exception as exc:
            log.write(f"[bold red]Error:[/bold red] {exc}")

    # ── CSV export ────────────────────────────────────────────────────────────

    def _export(self) -> str:
        out = Path.home() / "kubera" / "sessions"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts", "symbol", "side", "qty", "price", "pnl"])
            w.writeheader()
            w.writerows(self.broker.get_trade_log())
        return str(path)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def on_key(self, event: events.Key) -> None:
        if self.query_one("#cmd-input", Input).has_focus:
            return
        if event.key == "s":
            self.sort_col = (self.sort_col + 1) % len(SORT_KEYS)
            pval = self.broker.get_portfolio_value(self._prices)
            self._update_table(self.broker.get_positions(), self.broker.get_short_positions(), pval)
            event.prevent_default()
        elif event.key == "question_mark":
            self._help()
            event.prevent_default()
        elif event.key == "q":
            self.exit()

    def _help(self) -> None:
        log = self.query_one("#order-log", RichLog)
        log.write(
            "\n[bold gold1]── Keybindings ──────────────────────────────────────────────[/bold gold1]\n"
            "  [bold white]s[/bold white]    cycle sort: symbol → pnl → qty → age\n"
            "  [bold white]?[/bold white]    show this help\n"
            "  [bold white]q[/bold white]    quit\n"
            "\n[bold gold1]── Commands ─────────────────────────────────────────────────[/bold gold1]\n"
            "  [bold white]buy <sym> <qty>        [/bold white] long at market\n"
            "  [bold white]short <sym> <qty>      [/bold white] short at market\n"
            "  [bold white]sell <sym> <qty>       [/bold white] exit long\n"
            "  [bold white]cover <sym> <qty>      [/bold white] cover short\n"
            "  [bold white]x <sym>                [/bold white] close entire position\n"
            "  [bold white]:export                [/bold white] CSV → ~/kubera/sessions/\n"
            "  [bold white]:note <text>           [/bold white] add journal entry\n"
            "  [bold white]:tag <sym> <text>      [/bold white] tag a position\n"
            "[bold gold1]─────────────────────────────────────────────────────────────[/bold gold1]\n"
        )


def main() -> None:
    KuberaTUI().run()


if __name__ == "__main__":
    main()
