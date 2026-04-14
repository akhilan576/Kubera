"""
Project Zeno — FastAPI backend.

Endpoints:
  GET  /api/portfolio       — live portfolio value, cash, positions
  GET  /api/signals         — recent signals from DB
  GET  /api/trades          — recent fills from DB
  GET  /api/equity          — equity curve data
  GET  /api/status          — bot health + market status
  WS   /ws/live             — WebSocket: pushes updates every 5s

Run standalone:
  uvicorn api.main:app --reload --port 8000

Or via docker-compose (auto-started).
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import pytz

from db.logger import init_db, get_recent_trades, get_equity_history
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Project Zeno API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ET = pytz.timezone("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_broker():
    from execution.broker import Broker
    return Broker()

def _is_market_open() -> bool:
    from config.settings import MARKET_OPEN_TIME, MARKET_CLOSE_TIME
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    oh, om = map(int, MARKET_OPEN_TIME.split(":"))
    ch, cm = map(int, MARKET_CLOSE_TIME.split(":"))
    open_t  = now_et.replace(hour=oh, minute=om, second=0, microsecond=0)
    close_t = now_et.replace(hour=ch, minute=cm, second=0, microsecond=0)
    return open_t <= now_et <= close_t


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Zeno API started")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def get_portfolio():
    try:
        broker    = _get_broker()
        account   = broker.get_account()
        positions = broker.get_positions()

        pos_list = []
        for sym, pos in positions.items():
            pos_list.append({
                "symbol":        sym,
                "qty":           float(pos.qty),
                "entry_price":   float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                "market_value":  float(pos.market_value),
            })

        return {
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "invested":        float(account.portfolio_value) - float(account.cash),
            "positions":       pos_list,
            "open_count":      len(pos_list),
            "market_open":     _is_market_open(),
            "timestamp":       datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    try:
        from db.logger import _conn
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return {"signals": [dict(r) for r in rows]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    try:
        trades = get_recent_trades(limit)
        return {"trades": trades}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/equity")
async def get_equity():
    try:
        history = get_equity_history()
        return {
            "equity": history,
            "start_value": history[0]["portfolio_value"] if history else 100_000,
            "current_value": history[-1]["portfolio_value"] if history else 100_000,
            "total_return_pct": (
                (history[-1]["portfolio_value"] / history[0]["portfolio_value"] - 1) * 100
                if len(history) >= 2 else 0.0
            ),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/status")
async def get_status():
    try:
        from config.settings import SYMBOLS, CRYPTO_ENABLED, CRYPTO_SYMBOLS
        from models.trainer import load_model
        models = {s: load_model(s) is not None for s in SYMBOLS}
        return {
            "market_open":   _is_market_open(),
            "crypto_enabled": CRYPTO_ENABLED,
            "symbols":       SYMBOLS,
            "crypto_symbols": CRYPTO_SYMBOLS if CRYPTO_ENABLED else [],
            "models_trained": models,
            "timestamp":     datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── WebSocket — live updates ──────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                self.active.remove(ws)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    logger.info(f"WebSocket client connected ({len(manager.active)} total)")
    try:
        while True:
            # Push portfolio update every 5 seconds
            try:
                broker    = _get_broker()
                account   = broker.get_account()
                positions = broker.get_positions()
                trades    = get_recent_trades(5)

                await ws.send_json({
                    "type":            "portfolio",
                    "portfolio_value": float(account.portfolio_value),
                    "cash":            float(account.cash),
                    "open_positions":  len(positions),
                    "market_open":     _is_market_open(),
                    "recent_trades":   trades,
                    "timestamp":       datetime.utcnow().isoformat(),
                })
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info(f"WebSocket client disconnected ({len(manager.active)} remaining)")
