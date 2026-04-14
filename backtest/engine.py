"""
Walk-forward backtester for Project Zeno strategies.

Usage:
    python -m backtest.engine --symbols AAPL MSFT --lookback 500
    python -m backtest.engine --symbols AAPL --strategy sma
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from data.fetcher import DataFetcher
from data.features import build_features
from strategy.base import Signal
from strategy.sma_crossover import SMACrossoverStrategy
from strategy.xgb_strategy import XGBStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

COMMISSION_PCT = 0.001   # 0.1% per trade (Alpaca is 0% but model slippage)
INITIAL_CASH   = 100_000.0
STOP_LOSS_PCT  = 0.02
TAKE_PROFIT_PCT = 0.04


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    qty: int
    side: str = "long"
    exit_reason: str = ""

    @property
    def pnl(self) -> float:
        gross = (self.exit_price - self.entry_price) * self.qty
        commission = (self.entry_price + self.exit_price) * self.qty * COMMISSION_PCT
        return gross - commission

    @property
    def pnl_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_returns = self.equity_curve.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - roll_max) / roll_max
        return float(drawdown.min())

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  Backtest: {self.symbol}\n"
            f"{'='*55}\n"
            f"  Trades      : {self.total_trades}\n"
            f"  Win rate    : {self.win_rate:.1%}\n"
            f"  Total P&L   : ${self.total_pnl:,.2f}\n"
            f"  Avg P&L     : ${self.avg_pnl:,.2f}\n"
            f"  Sharpe ratio: {self.sharpe_ratio:.2f}\n"
            f"  Max drawdown: {self.max_drawdown:.1%}\n"
            f"{'='*55}"
        )


class Backtester:
    def __init__(self, strategy_name: str = "sma"):
        self.strategy_name = strategy_name

    def run(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """
        Simulate trading on historical OHLCV data bar-by-bar.
        Uses a simple long-only model with stop-loss and take-profit.
        """
        result = BacktestResult(symbol=symbol)
        cash = INITIAL_CASH
        position_qty = 0
        entry_price = 0.0
        entry_date = None
        stop_price = 0.0
        tp_price = 0.0
        equity_series = {}

        strategy = self._build_strategy(symbol, df)

        # Walk forward: start after enough warm-up bars
        warm_up = 60
        for i in range(warm_up, len(df)):
            window = df.iloc[: i + 1]
            bar = df.iloc[i]
            date = df.index[i]
            price = float(bar["close"])

            # Track equity
            position_value = position_qty * price
            equity_series[date] = cash + position_value

            # --- Exit checks (stop-loss / take-profit) ---
            if position_qty > 0:
                if price <= stop_price:
                    pnl_trade = self._close(
                        result, symbol, entry_date, date,
                        entry_price, stop_price, position_qty, "stop-loss"
                    )
                    cash += stop_price * position_qty * (1 - COMMISSION_PCT)
                    position_qty = 0
                    continue
                if price >= tp_price:
                    self._close(
                        result, symbol, entry_date, date,
                        entry_price, tp_price, position_qty, "take-profit"
                    )
                    cash += tp_price * position_qty * (1 - COMMISSION_PCT)
                    position_qty = 0
                    continue

            # --- Strategy signals ---
            signals = strategy.generate_signals({symbol: window})
            signal = next((s for s in signals if s.symbol == symbol), None)

            if signal is None:
                continue

            if signal.signal == Signal.BUY and position_qty == 0:
                max_spend = cash * 0.10 * signal.confidence
                qty = int(max_spend / price)
                if qty > 0:
                    cost = price * qty * (1 + COMMISSION_PCT)
                    if cost <= cash:
                        cash -= cost
                        position_qty = qty
                        entry_price = price
                        entry_date = date
                        stop_price = price * (1 - STOP_LOSS_PCT)
                        tp_price   = price * (1 + TAKE_PROFIT_PCT)

            elif signal.signal == Signal.SELL and position_qty > 0:
                self._close(
                    result, symbol, entry_date, date,
                    entry_price, price, position_qty, "signal"
                )
                cash += price * position_qty * (1 - COMMISSION_PCT)
                position_qty = 0

        # Force-close any open position at end
        if position_qty > 0:
            last_price = float(df["close"].iloc[-1])
            last_date  = df.index[-1]
            self._close(
                result, symbol, entry_date, last_date,
                entry_price, last_price, position_qty, "end-of-data"
            )
            cash += last_price * position_qty * (1 - COMMISSION_PCT)

        result.equity_curve = pd.Series(equity_series)
        return result

    def _close(self, result, symbol, entry_date, exit_date,
               entry_price, exit_price, qty, reason) -> float:
        trade = Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            exit_reason=reason,
        )
        result.trades.append(trade)
        return trade.pnl

    def _build_strategy(self, symbol: str, df: pd.DataFrame):
        if self.strategy_name == "xgb":
            from models.trainer import load_model
            model = load_model(symbol)
            if model is None:
                logger.warning(f"No XGB model for {symbol}, falling back to SMA")
                return SMACrossoverStrategy()
            return XGBStrategy(buy_threshold=0.60, sell_threshold=0.40)
        return SMACrossoverStrategy(fast=20, slow=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbols",  nargs="+", required=True)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--strategy", choices=["sma", "xgb"], default="sma")
    args = parser.parse_args()

    fetcher = Backtester.__new__(Backtester)  # avoid import loop
    fetcher = DataFetcher()
    backtester = Backtester(strategy_name=args.strategy)

    for sym in args.symbols:
        logger.info(f"Fetching data for {sym}...")
        bars = fetcher.get_bars([sym], lookback=args.lookback)
        if sym not in bars:
            logger.error(f"No data for {sym}")
            continue
        result = backtester.run(sym, bars[sym])
        print(result.summary())
