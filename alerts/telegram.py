"""
Telegram alert system for Project Zeno.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your bot once, then get your chat_id:
     curl https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Add to .env:
     TELEGRAM_BOT_TOKEN=your_token
     TELEGRAM_CHAT_ID=your_chat_id

Usage (async-friendly, fire-and-forget):
    from alerts.telegram import Alerter
    alerter = Alerter()
    alerter.send("Hello from Zeno!")
    alerter.trade_opened("AAPL", "BUY", 10, 182.50, confidence=0.72)
"""
from __future__ import annotations

import os
import asyncio
import threading
from datetime import datetime
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


class Alerter:
    """
    Sends Telegram messages synchronously from a background thread.
    Gracefully disabled if credentials are not set.
    """

    def __init__(self):
        self._enabled = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
        if not self._enabled:
            logger.warning(
                "Telegram alerts disabled — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def send(self, message: str):
        """Send a plain text message."""
        if not self._enabled:
            return
        self._dispatch(message)

    def trade_opened(self, symbol: str, side: str, qty: int,
                     price: float, confidence: float = 1.0):
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        msg = (
            f"{emoji} *Trade Opened*\n"
            f"Symbol    : `{symbol}`\n"
            f"Side      : `{side.upper()}`\n"
            f"Qty       : `{qty}`\n"
            f"Price     : `${price:,.2f}`\n"
            f"Value     : `${qty * price:,.2f}`\n"
            f"Confidence: `{confidence:.0%}`\n"
            f"Time      : `{_now()}`"
        )
        self.send(msg)

    def trade_closed(self, symbol: str, side: str, qty: int,
                     entry: float, exit_price: float, pnl: float):
        emoji = "✅" if pnl >= 0 else "❌"
        pct = (exit_price - entry) / entry * 100
        msg = (
            f"{emoji} *Trade Closed*\n"
            f"Symbol    : `{symbol}`\n"
            f"Side      : `{side.upper()}`\n"
            f"Qty       : `{qty}`\n"
            f"Entry     : `${entry:,.2f}`\n"
            f"Exit      : `${exit_price:,.2f}`\n"
            f"P&L       : `${pnl:+,.2f}` (`{pct:+.2f}%`)\n"
            f"Time      : `{_now()}`"
        )
        self.send(msg)

    def portfolio_snapshot(self, portfolio_value: float, cash: float,
                           open_positions: int, daily_pnl: float):
        emoji = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{emoji} *Portfolio Snapshot*\n"
            f"Value     : `${portfolio_value:,.2f}`\n"
            f"Cash      : `${cash:,.2f}`\n"
            f"Positions : `{open_positions}`\n"
            f"Day P&L   : `${daily_pnl:+,.2f}`\n"
            f"Time      : `{_now()}`"
        )
        self.send(msg)

    def error(self, context: str, error: str):
        msg = (
            f"⚠️ *Bot Error*\n"
            f"Context: `{context}`\n"
            f"Error  : `{error}`\n"
            f"Time   : `{_now()}`"
        )
        self.send(msg)

    def bot_started(self, mode: str = "paper"):
        self.send(f"🤖 *Project Zeno started* — mode: `{mode.upper()}` — `{_now()}`")

    def bot_stopped(self):
        self.send(f"🛑 *Project Zeno stopped* — `{_now()}`")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(self, text: str):
        """Fire-and-forget in a background thread."""
        t = threading.Thread(target=self._send_sync, args=(text,), daemon=True)
        t.start()

    def _send_sync(self, text: str):
        try:
            import telegram
            async def _send():
                bot = telegram.Bot(token=TELEGRAM_TOKEN)
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=text,
                    parse_mode="Markdown",
                )
            asyncio.run(_send())
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
