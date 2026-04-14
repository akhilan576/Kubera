"""
Virtual paper broker for Indian market simulation.
Tracks positions and cash in SQLite. No real orders placed.
Simulates fills at the price provided by the caller (market price).
"""
from __future__ import annotations
import sqlite3
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB = "db/india_paper.db"


class IndiaPaperBroker:
    """
    Simulates a brokerage for NSE paper trading.
    All state persists in SQLite so it survives bot restarts.
    """

    def __init__(self, starting_cash: float = 50_000.0, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db(starting_cash)
        logger.info(f"IndiaPaperBroker ready | cash=₹{self.get_cash():,.2f} | db={db_path}")

    def _init_db(self, starting_cash: float):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS account (
                key TEXT PRIMARY KEY, value TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol    TEXT PRIMARY KEY,
                qty       REAL,
                avg_entry REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT,
                symbol TEXT,
                side   TEXT,
                qty    REAL,
                price  REAL,
                pnl    REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stops (
                symbol       TEXT PRIMARY KEY,
                stop_price   REAL,
                target_price REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                ts    TEXT,
                value REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT,
                symbol TEXT,
                price  REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts   TEXT,
                note TEXT
            )
        """)
        cur.execute("INSERT OR IGNORE INTO account VALUES ('cash', ?)", (str(starting_cash),))
        self._conn.commit()

    # ── Account ───────────────────────────────────────────────────────────────

    def get_cash(self) -> float:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM account WHERE key='cash'")
        return float(cur.fetchone()[0])

    def _set_cash(self, amount: float):
        self._conn.execute("UPDATE account SET value=? WHERE key='cash'", (str(amount),))
        self._conn.commit()

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        cur = self._conn.cursor()
        cur.execute("SELECT symbol, qty, avg_entry FROM positions")
        all_positions = cur.fetchall()
        # For longs:  qty * current_price (positive contribution)
        # For shorts: qty * current_price (negative qty = negative contribution = liability offset)
        equity = sum(qty * current_prices.get(sym, avg_entry)
                     for sym, qty, avg_entry in all_positions)
        return self.get_cash() + equity

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, dict]:
        """Returns long positions only (qty > 0)."""
        cur = self._conn.cursor()
        cur.execute("SELECT symbol, qty, avg_entry FROM positions WHERE qty > 0")
        return {row[0]: {"qty": row[1], "avg_entry": row[2]} for row in cur.fetchall()}

    def get_short_positions(self) -> dict[str, dict]:
        """Returns short positions only (qty < 0)."""
        cur = self._conn.cursor()
        cur.execute("SELECT symbol, qty, avg_entry FROM positions WHERE qty < 0")
        return {row[0]: {"qty": row[1], "avg_entry": row[2]} for row in cur.fetchall()}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.get_positions()

    def has_short(self, symbol: str) -> bool:
        return symbol in self.get_short_positions()

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy(self, symbol: str, qty: int, price: float):
        cost = qty * price
        cash = self.get_cash()
        if cost > cash:
            raise ValueError(f"Insufficient cash: need ₹{cost:,.2f}, have ₹{cash:,.2f}")

        cur = self._conn.cursor()
        cur.execute("SELECT qty, avg_entry FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if row:
            old_qty, old_avg = row
            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * price) / new_qty
            cur.execute("UPDATE positions SET qty=?, avg_entry=? WHERE symbol=?",
                        (new_qty, new_avg, symbol))
        else:
            cur.execute("INSERT INTO positions VALUES (?,?,?)", (symbol, qty, price))

        self._set_cash(cash - cost)
        cur.execute("INSERT INTO trades (ts,symbol,side,qty,price,pnl) VALUES (?,?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, "buy", qty, price, 0.0))
        self._conn.commit()
        logger.info(f"PAPER BUY:  {symbol} x{qty} @ ₹{price:,.2f} | cost=₹{cost:,.2f}")

    def sell(self, symbol: str, qty: int, price: float):
        cur = self._conn.cursor()
        cur.execute("SELECT qty, avg_entry FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if not row or row[0] < qty:
            held = row[0] if row else 0
            raise ValueError(f"Insufficient qty for {symbol}: need {qty}, have {held}")

        old_qty, avg_entry = row
        pnl     = (price - avg_entry) * qty
        new_qty = old_qty - qty

        if new_qty == 0:
            cur.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        else:
            cur.execute("UPDATE positions SET qty=? WHERE symbol=?", (new_qty, symbol))

        self._set_cash(self.get_cash() + qty * price)
        cur.execute("INSERT INTO trades (ts,symbol,side,qty,price,pnl) VALUES (?,?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, "sell", qty, price, pnl))
        self._conn.commit()
        logger.info(f"PAPER SELL: {symbol} x{qty} @ ₹{price:,.2f} | P&L=₹{pnl:+,.2f}")

    def short(self, symbol: str, qty: int, price: float):
        """Open a short position — receive cash, record negative qty."""
        proceeds = qty * price
        cur = self._conn.cursor()
        cur.execute("SELECT qty, avg_entry FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if row:
            old_qty, old_avg = row
            new_qty = old_qty - qty  # more negative
            new_avg = (abs(old_qty) * old_avg + qty * price) / (abs(old_qty) + qty)
            cur.execute("UPDATE positions SET qty=?, avg_entry=? WHERE symbol=?",
                        (new_qty, new_avg, symbol))
        else:
            cur.execute("INSERT INTO positions VALUES (?,?,?)", (symbol, -qty, price))
        self._set_cash(self.get_cash() + proceeds)
        cur.execute("INSERT INTO trades (ts,symbol,side,qty,price,pnl) VALUES (?,?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, "short", qty, price, 0.0))
        self._conn.commit()
        logger.info(f"PAPER SHORT: {symbol} x{qty} @ ₹{price:,.2f} | proceeds=₹{proceeds:,.2f}")

    def cover(self, symbol: str, qty: int, price: float):
        """Close a short position — pay cash, realise P&L."""
        cur = self._conn.cursor()
        cur.execute("SELECT qty, avg_entry FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if not row or row[0] >= 0:
            raise ValueError(f"No short position for {symbol}")
        old_qty, avg_entry = row  # old_qty is negative
        pnl     = (avg_entry - price) * qty   # profit when price fell
        new_qty = old_qty + qty               # less negative
        if new_qty == 0:
            cur.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        else:
            cur.execute("UPDATE positions SET qty=? WHERE symbol=?", (new_qty, symbol))
        self._set_cash(self.get_cash() - qty * price)
        cur.execute("INSERT INTO trades (ts,symbol,side,qty,price,pnl) VALUES (?,?,?,?,?,?)",
                    (datetime.now().isoformat(), symbol, "cover", qty, price, pnl))
        self._conn.commit()
        logger.info(f"PAPER COVER: {symbol} x{qty} @ ₹{price:,.2f} | P&L=₹{pnl:+,.2f}")

    def close_all_positions(self, current_prices: dict[str, float]):
        for symbol, pos in list(self.get_positions().items()):
            price = current_prices.get(symbol, pos["avg_entry"])
            self.sell(symbol, int(pos["qty"]), price)
        for symbol, pos in list(self.get_short_positions().items()):
            price = current_prices.get(symbol, pos["avg_entry"])
            self.cover(symbol, int(abs(pos["qty"])), price)

    def get_trade_log(self) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT ts,symbol,side,qty,price,pnl FROM trades ORDER BY id")
        cols = ["ts", "symbol", "side", "qty", "price", "pnl"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Stops ──────────────────────────────────────────────────────────────────

    def set_stop(self, symbol: str, stop: float, target: float = None):
        self._conn.execute(
            "INSERT OR REPLACE INTO stops (symbol, stop_price, target_price) VALUES (?,?,?)",
            (symbol, stop, target),
        )
        self._conn.commit()

    def clear_stop(self, symbol: str):
        self._conn.execute("DELETE FROM stops WHERE symbol=?", (symbol,))
        self._conn.commit()

    def get_stops(self) -> dict[str, dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT symbol, stop_price, target_price FROM stops")
        return {r[0]: {"stop": r[1], "target": r[2]} for r in cur.fetchall()}

    # ── Portfolio history ──────────────────────────────────────────────────────

    def log_portfolio_value(self, value: float):
        self._conn.execute(
            "INSERT INTO portfolio_history (ts, value) VALUES (?,?)",
            (datetime.now().isoformat(), value),
        )
        self._conn.commit()

    def get_portfolio_history(self, n: int = 100) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, value FROM portfolio_history ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = list(reversed(cur.fetchall()))
        return [{"ts": r[0], "value": r[1]} for r in rows]

    # ── Price history ──────────────────────────────────────────────────────────

    def log_prices(self, prices: dict[str, float]):
        ts = datetime.now().isoformat()
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT INTO price_history (ts, symbol, price) VALUES (?,?,?)",
            [(ts, sym, price) for sym, price in prices.items()],
        )
        self._conn.commit()

    def get_price_history(self, symbol: str, n: int = 20) -> list[float]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT price FROM price_history WHERE symbol=? ORDER BY id DESC LIMIT ?",
            (symbol, n),
        )
        return list(reversed([r[0] for r in cur.fetchall()]))

    # ── Journal ────────────────────────────────────────────────────────────────

    def add_journal(self, note: str):
        self._conn.execute(
            "INSERT INTO journal (ts, note) VALUES (?,?)",
            (datetime.now().isoformat(), note),
        )
        self._conn.commit()

    def get_journal(self) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT ts, note FROM journal ORDER BY id")
        return [{"ts": r[0], "note": r[1]} for r in cur.fetchall()]
