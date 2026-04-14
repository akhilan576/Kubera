"""
Pre-flight health check — run before starting the bot.

Verifies:
  - .env credentials are set
  - Alpaca connection works (paper)
  - At least one XGBoost model exists
  - Database is writable
  - Telegram (optional) is reachable

Usage:
    python scripts/healthcheck.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m  ✓\033[0m"
FAIL = "\033[91m  ✗\033[0m"
WARN = "\033[93m  !\033[0m"


def check(label: str, fn):
    try:
        result = fn()
        status = WARN if result is None else PASS
        note   = "" if result is None else f" — {result}"
        print(f"{status} {label}{note}")
        return True
    except Exception as e:
        print(f"{FAIL} {label} — {e}")
        return False


def main():
    print("\n\033[1mProject Zeno — Health Check\033[0m")
    print("─" * 45)
    failures = 0

    # 1. Env vars
    def _env():
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
        assert ALPACA_API_KEY,    "ALPACA_API_KEY is not set in .env"
        assert ALPACA_SECRET_KEY, "ALPACA_SECRET_KEY is not set in .env"
        return "keys loaded"

    if not check("Environment variables", _env):
        failures += 1

    # 2. Alpaca connection
    def _alpaca():
        from execution.broker import Broker
        broker  = Broker()
        account = broker.get_account()
        pv = float(account.portfolio_value)
        return f"portfolio=${pv:,.2f}, status={account.status}"

    if not check("Alpaca API connection", _alpaca):
        failures += 1

    # 3. Market data
    def _data():
        from data.fetcher import DataFetcher
        from config.settings import SYMBOLS
        fetcher = DataFetcher()
        bars    = fetcher.get_bars([SYMBOLS[0]], lookback=5)
        assert bars and SYMBOLS[0] in bars, "No bar data returned"
        df = bars[SYMBOLS[0]]
        return f"{SYMBOLS[0]}: {len(df)} bars, last close=${df['close'].iloc[-1]:.2f}"

    if not check("Market data feed", _data):
        failures += 1

    # 4. XGBoost models
    def _models():
        from config.settings import SYMBOLS
        from models.trainer import load_model
        found = [s for s in SYMBOLS if load_model(s) is not None]
        if not found:
            raise FileNotFoundError(
                "No trained models found — run: python -m models.trainer --symbols " +
                " ".join(SYMBOLS)
            )
        return f"found for: {', '.join(found)}"

    if not check("XGBoost models", _models):
        failures += 1

    # 5. Database
    def _db():
        from db.logger import init_db, log_equity
        init_db()
        log_equity(0.0, 0.0, 0)
        return "db/zeno.db writable"

    if not check("SQLite database", _db):
        failures += 1

    # 6. Telegram (optional)
    def _telegram():
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return None   # optional — show as warning, not failure
        from alerts.telegram import Alerter
        Alerter().send("✅ Project Zeno health check passed")
        return "test message sent"

    check("Telegram alerts (optional)", _telegram)

    # Summary
    print("─" * 45)
    if failures == 0:
        print("\033[92m\033[1m  All checks passed — bot is ready to run.\033[0m")
        print("  Start with: python bot.py\n")
    else:
        print(f"\033[91m\033[1m  {failures} check(s) failed — fix above before running.\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
