# India Paper Trading (Angel One SmartAPI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Indian market (NSE) paper trading to Zeno using Angel One SmartAPI for live data and a local SQLite-backed virtual broker for simulated order execution — no real money, no paid subscription.

**Architecture:** A new `zeno_ind.py` runs alongside the existing bot, trading NIFTY 50 stocks on a 15-min loop during Indian market hours (9:15 AM–3:30 PM IST). `data/angel_fetcher.py` pulls OHLCV from Angel One SmartAPI. `execution/india_paper_broker.py` simulates fills locally in SQLite (no real orders placed). The existing XGBoost strategy, risk manager, and trainer are reused unchanged.

**Tech Stack:** `smartapi-python`, `pyotp` (TOTP auth), `sqlite3`, `pandas`, `pytz`, existing Zeno strategy/risk stack.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `data/angel_fetcher.py` | **Create** | Angel One SmartAPI auth + OHLCV historical/live data |
| `execution/india_paper_broker.py` | **Create** | Virtual paper broker — simulates fills in SQLite |
| `config/india_settings.py` | **Create** | Indian market config (symbols, hours, risk params) |
| `zeno_ind.py` | **Create** | Indian market main loop (mirrors bot.py structure) |
| `.env` | **Modify** | Add Angel One credentials + INDIA_ENABLED flag |
| `config/settings.py` | **Modify** | Import and re-export India settings |
| `db/india_paper.db` | **Auto-created** | SQLite paper trading state |
| `tests/test_angel_fetcher.py` | **Create** | Unit tests for data fetcher |
| `tests/test_india_paper_broker.py` | **Create** | Unit tests for virtual broker |

---

## Prerequisites (manual, before coding)

- [ ] Create a free Angel One demat account at angelone.in
- [ ] Enable SmartAPI: My Profile → API → Generate API Key (free)
- [ ] Enable TOTP: My Profile → Security → Enable TOTP (use Google Authenticator)
- [ ] Note down: `API_KEY`, `CLIENT_ID` (your login ID), `PASSWORD`, `TOTP_SECRET`
- [ ] Install dependencies:
```bash
cd ~/Documents/Project\ Zeno
.venv/bin/pip install smartapi-python pyotp pytz
```

---

## Task 1: Angel One data fetcher

**Files:**
- Create: `data/angel_fetcher.py`
- Create: `tests/test_angel_fetcher.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_angel_fetcher.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

def test_normalise_symbol():
    from data.angel_fetcher import normalise_symbol
    assert normalise_symbol("RELIANCE") == "RELIANCE"
    assert normalise_symbol("reliance") == "RELIANCE"

def test_get_bars_returns_dataframe():
    from data.angel_fetcher import AngelFetcher
    mock_resp = {
        "status": True,
        "data": [
            ["2026-04-10T09:15:00+05:30", 2800.0, 2820.0, 2795.0, 2815.0, 100000],
            ["2026-04-10T09:30:00+05:30", 2815.0, 2830.0, 2810.0, 2825.0, 90000],
        ]
    }
    with patch("data.angel_fetcher.SmartConnect") as MockSC:
        instance = MockSC.return_value
        instance.getCandleData.return_value = mock_resp
        instance.generateSession.return_value = {"status": True, "data": {"jwtToken": "tok", "refreshToken": "ref"}}
        fetcher = AngelFetcher.__new__(AngelFetcher)
        fetcher.smart = instance
        fetcher._token_map = {"RELIANCE": "2885"}
        result = fetcher._parse_candles(mock_resp["data"])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == 2

def test_get_bars_unknown_symbol_skipped():
    from data.angel_fetcher import AngelFetcher
    with patch("data.angel_fetcher.SmartConnect") as MockSC:
        instance = MockSC.return_value
        instance.getCandleData.return_value = {"status": False, "message": "Invalid token"}
        fetcher = AngelFetcher.__new__(AngelFetcher)
        fetcher.smart = instance
        fetcher._token_map = {}
        result = fetcher.get_bars(["UNKNOWN"], lookback=100)
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Documents/Project\ Zeno
.venv/bin/pytest tests/test_angel_fetcher.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create the fetcher**

```python
# data/angel_fetcher.py
"""
Angel One SmartAPI data fetcher for NSE stocks.
Provides same interface as DataFetcher (get_bars returns {symbol: DataFrame}).
"""
from __future__ import annotations
import time
import pyotp
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from config.india_settings import (
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
    INDIA_BAR_TIMEFRAME, INDIA_LOOKBACK_BARS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Angel One instrument token map for NIFTY 50 stocks (NSE)
# Full list: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
_INSTRUMENT_TOKENS: dict[str, str] = {
    # ~₹250–500
    "ONGC":       "2475",   # ~₹250
    "NTPC":       "11630",  # ~₹350
    "COALINDIA":  "20374",  # ~₹400
    "ITC":        "1660",   # ~₹450
    "WIPRO":      "3787",   # ~₹500
    # ~₹500–1000
    "HINDALCO":   "1363",   # ~₹650
    "TATAMOTORS": "3456",   # ~₹700
    "SBIN":       "3045",   # ~₹800
    # ~₹1000–2000
    "AXISBANK":   "5900",   # ~₹1,100
    "ICICIBANK":  "4963",   # ~₹1,200
    "RELIANCE":   "2885",   # ~₹1,300
    "INFY":       "1594",   # ~₹1,500
    "BHARTIARTL": "10604",  # ~₹1,500
    "HDFCBANK":   "1333",   # ~₹1,700
    "SUNPHARMA":  "3351",   # ~₹1,800
    "NIFTY_50":   "99926000",  # index — market regime context (like SPY)
}

_TIMEFRAME_MAP = {
    "1Min":  "ONE_MINUTE",
    "5Min":  "FIVE_MINUTE",
    "15Min": "FIFTEEN_MINUTE",
    "1Hour": "ONE_HOUR",
    "1Day":  "ONE_DAY",
}


def normalise_symbol(symbol: str) -> str:
    return symbol.upper().strip()


class AngelFetcher:
    """Fetch OHLCV bars from Angel One SmartAPI."""

    def __init__(self):
        self.smart = SmartConnect(api_key=ANGEL_API_KEY)
        self._authenticate()
        self._token_map = _INSTRUMENT_TOKENS

    def _authenticate(self):
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        resp = self.smart.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if not resp.get("status"):
            raise RuntimeError(f"Angel One auth failed: {resp.get('message')}")
        logger.info("Angel One SmartAPI authenticated")

    def _parse_candles(self, data: list) -> pd.DataFrame:
        """Parse Angel One candle list → DataFrame with open/high/low/close/volume."""
        rows = []
        for candle in data:
            ts, o, h, l, c, v = candle
            rows.append({"open": float(o), "high": float(h), "low": float(l),
                         "close": float(c), "volume": float(v)})
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
        return df

    def get_bars(self, symbols: list[str], lookback: int = INDIA_LOOKBACK_BARS) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for given symbols.
        Returns {symbol: DataFrame} — same interface as DataFetcher.
        """
        tf = _TIMEFRAME_MAP.get(INDIA_BAR_TIMEFRAME, "FIFTEEN_MINUTE")
        end_dt   = datetime.now()
        bars_per_day = {"1Min": 375, "5Min": 75, "15Min": 25, "1Hour": 7, "1Day": 1}.get(INDIA_BAR_TIMEFRAME, 25)
        cal_days = max(5, int((lookback / bars_per_day) * 7 / 5 * 1.5))
        start_dt = end_dt - timedelta(days=cal_days)

        from_date = start_dt.strftime("%Y-%m-%d %H:%M")
        to_date   = end_dt.strftime("%Y-%m-%d %H:%M")

        result = {}
        for sym in symbols:
            sym = normalise_symbol(sym)
            token = self._token_map.get(sym)
            if not token:
                logger.warning(f"AngelFetcher: no token for {sym}, skipping")
                continue
            try:
                resp = self.smart.getCandleData({
                    "exchange":    "NSE",
                    "symboltoken": token,
                    "interval":    tf,
                    "fromdate":    from_date,
                    "todate":      to_date,
                })
                if not resp.get("status") or not resp.get("data"):
                    logger.warning(f"AngelFetcher: no data for {sym}: {resp.get('message')}")
                    continue
                df = self._parse_candles(resp["data"])
                if df.empty:
                    continue
                result[sym] = df
                time.sleep(0.2)  # respect rate limits
            except Exception as e:
                logger.error(f"AngelFetcher: failed for {sym}: {e}")
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_angel_fetcher.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/Project\ Zeno
git add data/angel_fetcher.py tests/test_angel_fetcher.py
git commit -m "feat: add Angel One SmartAPI data fetcher for NSE stocks"
```

---

## Task 2: India config

**Files:**
- Create: `config/india_settings.py`
- Modify: `.env`
- Modify: `config/settings.py`

- [ ] **Step 1: Add credentials to .env**

Add these lines to `/Users/akhilan_m/Documents/Project Zeno/.env`:
```
ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id_here
ANGEL_PASSWORD=your_password_here
ANGEL_TOTP_SECRET=your_totp_secret_here
INDIA_SYMBOLS=ONGC,NTPC,COALINDIA,ITC,WIPRO,HINDALCO,TATAMOTORS,SBIN,AXISBANK,ICICIBANK,RELIANCE,INFY,BHARTIARTL,HDFCBANK,SUNPHARMA
INDIA_ENABLED=false
INDIA_PAPER_STARTING_CASH=50000
```

- [ ] **Step 2: Create india_settings.py**

```python
# config/india_settings.py
import os
from dotenv import load_dotenv

load_dotenv()

ANGEL_API_KEY      = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID    = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD     = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET  = os.getenv("ANGEL_TOTP_SECRET", "")

INDIA_SYMBOLS      = os.getenv("INDIA_SYMBOLS", "ONGC,NTPC,COALINDIA,ITC,WIPRO,HINDALCO,TATAMOTORS,SBIN,AXISBANK,ICICIBANK,RELIANCE,INFY,BHARTIARTL,HDFCBANK,SUNPHARMA").split(",")
INDIA_ENABLED      = os.getenv("INDIA_ENABLED", "false").lower() == "true"

# Market hours (IST, 24h)
INDIA_MARKET_OPEN  = "09:15"
INDIA_MARKET_CLOSE = "15:25"   # stop new trades 5 min before close
INDIA_EOD_LIQUIDATE = "15:20"  # flatten all positions

# Bar settings
INDIA_BAR_TIMEFRAME = "15Min"
INDIA_LOOKBACK_BARS = 1000

# Risk
INDIA_MAX_POSITION_PCT   = 0.10   # max 10% of paper portfolio per position
INDIA_MAX_OPEN_POSITIONS = 10
INDIA_STARTING_CASH      = float(os.getenv("INDIA_PAPER_STARTING_CASH", "50000"))  # ₹50k default
INDIA_TRAILING_STOP_PCT  = 1.5
INDIA_RUN_INTERVAL_SECS  = 60
```

- [ ] **Step 3: Re-export from settings.py**

Add to the bottom of `config/settings.py`:
```python
# --- India ---
from config.india_settings import (  # noqa: F401
    INDIA_ENABLED, INDIA_SYMBOLS, INDIA_MARKET_OPEN,
    INDIA_MARKET_CLOSE, INDIA_EOD_LIQUIDATE,
)
```

- [ ] **Step 4: Commit**

```bash
git add config/india_settings.py config/settings.py .env
git commit -m "feat: add India market config and Angel One credentials placeholder"
```

---

## Task 3: Virtual paper broker

**Files:**
- Create: `execution/india_paper_broker.py`
- Create: `tests/test_india_paper_broker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_india_paper_broker.py
import pytest
import os
import tempfile
from execution.india_paper_broker import IndiaPaperBroker

@pytest.fixture
def broker(tmp_path):
    db = str(tmp_path / "test_paper.db")
    return IndiaPaperBroker(starting_cash=100000.0, db_path=db)

def test_initial_cash(broker):
    assert broker.get_cash() == 100000.0

def test_buy_reduces_cash(broker):
    broker.buy("RELIANCE", qty=10, price=2800.0)
    assert broker.get_cash() == pytest.approx(100000.0 - 10 * 2800.0)

def test_buy_creates_position(broker):
    broker.buy("TCS", qty=5, price=3500.0)
    positions = broker.get_positions()
    assert "TCS" in positions
    assert positions["TCS"]["qty"] == 5

def test_sell_removes_position(broker):
    broker.buy("INFY", qty=10, price=1500.0)
    broker.sell("INFY", qty=10, price=1600.0)
    positions = broker.get_positions()
    assert "INFY" not in positions

def test_sell_increases_cash(broker):
    broker.buy("SBIN", qty=20, price=700.0)
    cash_after_buy = broker.get_cash()
    broker.sell("SBIN", qty=20, price=750.0)
    assert broker.get_cash() == pytest.approx(cash_after_buy + 20 * 750.0)

def test_portfolio_value(broker):
    broker.buy("RELIANCE", qty=10, price=2800.0)
    prices = {"RELIANCE": 2900.0}
    val = broker.get_portfolio_value(prices)
    expected = (100000.0 - 10 * 2800.0) + 10 * 2900.0
    assert val == pytest.approx(expected)

def test_cannot_sell_more_than_held(broker):
    broker.buy("TCS", qty=5, price=3500.0)
    with pytest.raises(ValueError, match="Insufficient qty"):
        broker.sell("TCS", qty=10, price=3600.0)

def test_cannot_buy_without_cash(broker):
    with pytest.raises(ValueError, match="Insufficient cash"):
        broker.buy("RELIANCE", qty=1000, price=2800.0)

def test_has_position(broker):
    assert not broker.has_position("WIPRO")
    broker.buy("WIPRO", qty=5, price=500.0)
    assert broker.has_position("WIPRO")

def test_trade_log_persists(broker):
    broker.buy("MARUTI", qty=2, price=10000.0)
    broker.sell("MARUTI", qty=2, price=10500.0)
    log = broker.get_trade_log()
    assert len(log) == 2
    assert log[0]["side"] == "buy"
    assert log[1]["side"] == "sell"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_india_paper_broker.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the virtual broker**

```python
# execution/india_paper_broker.py
"""
Virtual paper broker for Indian market simulation.
Tracks positions and cash in SQLite. No real orders placed.
Simulates fills at the price provided by the caller (market price).
"""
from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB = "db/india_paper.db"


class IndiaPaperBroker:
    """
    Simulates a brokerage for NSE paper trading.
    All state persists in SQLite so it survives bot restarts.
    """

    def __init__(self, starting_cash: float = 1_000_000.0, db_path: str = _DEFAULT_DB):
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
                symbol TEXT PRIMARY KEY,
                qty    REAL,
                avg_entry REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT,
                symbol    TEXT,
                side      TEXT,
                qty       REAL,
                price     REAL,
                pnl       REAL
            )
        """)
        # Only set starting cash if account is brand new
        cur.execute("INSERT OR IGNORE INTO account VALUES ('cash', ?)", (str(starting_cash),))
        self._conn.commit()

    # ── Account ──────────────────────────────────────────────────────────────

    def get_cash(self) -> float:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM account WHERE key='cash'")
        return float(cur.fetchone()[0])

    def _set_cash(self, amount: float):
        self._conn.execute("UPDATE account SET value=? WHERE key='cash'", (str(amount),))
        self._conn.commit()

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Total value = cash + sum(qty * current_price) for all positions."""
        positions = self.get_positions()
        equity = sum(pos["qty"] * current_prices.get(sym, pos["avg_entry"])
                     for sym, pos in positions.items())
        return self.get_cash() + equity

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, dict]:
        """Returns {symbol: {qty, avg_entry}}"""
        cur = self._conn.cursor()
        cur.execute("SELECT symbol, qty, avg_entry FROM positions WHERE qty > 0")
        return {row[0]: {"qty": row[1], "avg_entry": row[2]} for row in cur.fetchall()}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.get_positions()

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy(self, symbol: str, qty: int, price: float):
        """Simulate a market buy at `price`."""
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
        logger.info(f"PAPER BUY: {symbol} x{qty} @ ₹{price:,.2f} | cost=₹{cost:,.2f}")

    def sell(self, symbol: str, qty: int, price: float):
        """Simulate a market sell at `price`."""
        cur = self._conn.cursor()
        cur.execute("SELECT qty, avg_entry FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if not row or row[0] < qty:
            held = row[0] if row else 0
            raise ValueError(f"Insufficient qty for {symbol}: need {qty}, have {held}")

        old_qty, avg_entry = row
        pnl = (price - avg_entry) * qty
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

    def close_all_positions(self, current_prices: dict[str, float]):
        """EOD: flatten everything at current prices."""
        for symbol, pos in self.get_positions().items():
            price = current_prices.get(symbol, pos["avg_entry"])
            self.sell(symbol, int(pos["qty"]), price)

    def get_trade_log(self) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT ts,symbol,side,qty,price,pnl FROM trades ORDER BY id")
        cols = ["ts", "symbol", "side", "qty", "price", "pnl"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_india_paper_broker.py -v
```
Expected: 10 PASSED

- [ ] **Step 5: Commit**

```bash
git add execution/india_paper_broker.py tests/test_india_paper_broker.py
git commit -m "feat: virtual paper broker for Indian market simulation (SQLite-backed)"
```

---

## Task 4: Train models for NSE stocks

**Files:**
- No new files — reuse `models/trainer.py` with `DataFetcher` swap

- [ ] **Step 1: Run training for all India symbols**

```bash
cd ~/Documents/Project\ Zeno
.venv/bin/python - <<'EOF'
# Quick check — verify AngelFetcher returns data before committing to a full train
from data.angel_fetcher import AngelFetcher
f = AngelFetcher()
bars = f.get_bars(["RELIANCE", "TCS"], lookback=100)
for sym, df in bars.items():
    print(f"{sym}: {len(df)} bars, columns={list(df.columns)}")
EOF
```
Expected: 2 symbols with ~100 rows each

- [ ] **Step 2: Train all 15 symbols**

```bash
cd ~/Documents/Project\ Zeno
nohup .venv/bin/python - >> logs/india_training.log 2>&1 << 'EOF'
import sys
sys.path.insert(0, ".")
from data.angel_fetcher import AngelFetcher
from models.trainer import train_symbol

symbols = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "BAJFINANCE", "MARUTI",
    "TITAN", "SUNPHARMA", "TATAMOTORS", "LT", "BHARTIARTL"
]
fetcher = AngelFetcher()
for sym in symbols:
    try:
        train_symbol(sym, fetcher, lookback=1000, n_trials=200)
    except Exception as e:
        print(f"FAILED {sym}: {e}")
EOF
echo "Training started in background"
```

- [ ] **Step 3: Monitor training progress**

```bash
tail -f ~/Documents/Project\ Zeno/logs/india_training.log
```
Expected: each symbol logs `Model bundle saved → models/saved/<SYMBOL>.pkl`

- [ ] **Step 4: Verify models saved**

```bash
ls ~/Documents/Project\ Zeno/models/saved/ | grep -E "RELIANCE|TCS|INFY|HDFCBANK|SBIN"
```
Expected: 5+ `.pkl` files listed

- [ ] **Step 5: Commit**

```bash
git add logs/india_training.log
git commit -m "feat: train XGBoost models for NSE NIFTY 50 stocks"
```

---

## Task 5: Indian market bot loop

**Files:**
- Create: `zeno_ind.py`

- [ ] **Step 1: Write bot_india.py**

```python
# zeno_ind.py
"""
Project Zeno — Indian Market Bot
Trades NSE stocks using Angel One SmartAPI data + virtual paper broker.
Runs 9:15 AM – 3:30 PM IST. Fully independent of the US bot.
"""
from __future__ import annotations
import time
from datetime import datetime
import pytz

from config.india_settings import (
    INDIA_SYMBOLS, INDIA_ENABLED,
    INDIA_MARKET_OPEN, INDIA_MARKET_CLOSE, INDIA_EOD_LIQUIDATE,
    INDIA_MAX_POSITION_PCT, INDIA_MAX_OPEN_POSITIONS,
    INDIA_STARTING_CASH, INDIA_TRAILING_STOP_PCT,
    INDIA_RUN_INTERVAL_SECS, INDIA_LOOKBACK_BARS,
)
from data.angel_fetcher import AngelFetcher
from execution.india_paper_broker import IndiaPaperBroker
from strategy.xgb_strategy import XGBStrategy
from strategy.combined_strategy import CombinedStrategy, Signal
from risk.manager import RiskManager
from utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")


def ist_now() -> datetime:
    return datetime.now(IST)


def market_is_open() -> bool:
    now  = ist_now()
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t    = now.strftime("%H:%M")
    return INDIA_MARKET_OPEN <= t <= INDIA_MARKET_CLOSE


def eod_liquidate_time() -> bool:
    t = ist_now().strftime("%H:%M")
    return t >= INDIA_EOD_LIQUIDATE


def run():
    if not INDIA_ENABLED:
        logger.info("India bot disabled (INDIA_ENABLED=false). Set to true in .env to start.")
        return

    logger.info("── India Bot starting ──")
    fetcher  = AngelFetcher()
    broker   = IndiaPaperBroker(starting_cash=INDIA_STARTING_CASH)
    strategy = XGBStrategy()
    risk     = RiskManager()

    while True:
        try:
            if not market_is_open():
                now = ist_now().strftime("%H:%M:%S")
                logger.info(f"[INDIA] Market closed ({now} IST), sleeping...")
                time.sleep(INDIA_RUN_INTERVAL_SECS)
                continue

            logger.info("── India tick ──")
            bars      = fetcher.get_bars(INDIA_SYMBOLS, lookback=INDIA_LOOKBACK_BARS)
            positions = broker.get_positions()

            # Current prices for portfolio valuation and EOD liquidation
            current_prices = {sym: float(df["close"].iloc[-1])
                              for sym, df in bars.items() if not df.empty}

            # EOD liquidation
            if eod_liquidate_time() and positions:
                logger.info("[INDIA EOD] Liquidating all positions")
                broker.close_all_positions(current_prices)
                time.sleep(INDIA_RUN_INTERVAL_SECS)
                continue

            portfolio_value = broker.get_portfolio_value(current_prices)
            open_count      = len(positions)

            # Generate signals
            signals = strategy.generate_signals(bars)

            for signal in signals:
                symbol = signal.symbol
                if symbol not in current_prices:
                    continue
                price = current_prices[symbol]

                if signal.signal == Signal.BUY:
                    if open_count >= INDIA_MAX_OPEN_POSITIONS:
                        continue
                    if broker.has_position(symbol):
                        continue
                    max_spend = portfolio_value * INDIA_MAX_POSITION_PCT
                    qty = int(max_spend / price)
                    if qty < 1:
                        continue
                    try:
                        broker.buy(symbol, qty, price)
                        open_count += 1
                    except ValueError as e:
                        logger.warning(f"[INDIA] Buy skipped for {symbol}: {e}")

                elif signal.signal == Signal.SELL:
                    if not broker.has_position(symbol):
                        continue
                    pos = positions[symbol]
                    try:
                        broker.sell(symbol, int(pos["qty"]), price)
                    except ValueError as e:
                        logger.warning(f"[INDIA] Sell skipped for {symbol}: {e}")

            portfolio_value = broker.get_portfolio_value(current_prices)
            logger.info(f"[INDIA] Portfolio=₹{portfolio_value:,.2f} | "
                        f"Cash=₹{broker.get_cash():,.2f} | "
                        f"Positions={len(broker.get_positions())} | "
                        f"Signals={len(signals)}")

        except Exception as e:
            logger.error(f"[INDIA] Error: {e}", exc_info=True)

        time.sleep(INDIA_RUN_INTERVAL_SECS)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Enable India bot and do a dry run**

In `.env`, set:
```
INDIA_ENABLED=true
ANGEL_API_KEY=<your_key>
ANGEL_CLIENT_ID=<your_id>
ANGEL_PASSWORD=<your_password>
ANGEL_TOTP_SECRET=<your_totp_secret>
```

Run during market hours (9:15 AM – 3:30 PM IST):
```bash
cd ~/Documents/Project\ Zeno
.venv/bin/python zeno_ind.py 2>&1 | tee logs/india_bot.log
```
Expected: `[INDIA] Portfolio=₹50,000.00 | Cash=₹50,000.00 | Positions=0`

- [ ] **Step 3: Commit**

```bash
git add bot_india.py
git commit -m "feat: zeno_ind — Indian market paper trading bot loop (IST hours, virtual broker)"
```

---

## Task 6: Wire up to launchd (auto-start)

**Files:**
- Create: `~/Library/LaunchAgents/com.projectzeno.india.plist`

- [ ] **Step 1: Create the plist**

```bash
cat > ~/Library/LaunchAgents/com.projectzeno.india.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.projectzeno.india</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/akhilan_m/Documents/Project Zeno/.venv/bin/python</string>
        <string>zeno_ind.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/akhilan_m/Documents/Project Zeno</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/akhilan_m/Documents/Project Zeno/logs/india_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/akhilan_m/Documents/Project Zeno/logs/india_launchd.log</string>
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
EOF
```

- [ ] **Step 2: Remove quarantine and load**

```bash
xattr -rc ~/Library/LaunchAgents/com.projectzeno.india.plist
launchctl load ~/Library/LaunchAgents/com.projectzeno.india.plist
launchctl list | grep india
```
Expected: `com.projectzeno.india` listed with PID (not `-`)

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: launchd agent for India bot auto-start"
```

---

## Upgrade Path (when ready for live trading)

When paper trading results look good, switch to a real broker by:

1. **Zerodha Kite** (recommended): replace `IndiaPaperBroker` with a `KiteBroker` wrapper around `kiteconnect`. The `buy(symbol, qty, price)` interface stays identical — only the internals change.
2. Update `.env` with Kite credentials
3. Set `INDIA_PAPER=false` (add this flag to `india_settings.py`)
4. The rest of the bot is unchanged.
