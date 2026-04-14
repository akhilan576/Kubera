from __future__ import annotations
import pandas as pd
from strategy.base import BaseStrategy, TradeSignal, Signal
from data.indicators import sma, rsi
from utils.logger import get_logger

logger = get_logger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """
    Classic dual SMA crossover + RSI filter.

    BUY  when fast SMA crosses above slow SMA and RSI < overbought threshold.
    SELL when fast SMA crosses below slow SMA or RSI > overbought threshold.
    """

    def __init__(self, fast: int = 20, slow: int = 50, rsi_period: int = 14,
                 rsi_overbought: float = 70.0, rsi_oversold: float = 30.0):
        super().__init__("SMA Crossover + RSI")
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        signals = []

        for symbol, df in bars.items():
            if len(df) < self.slow + 2:
                logger.debug(f"{symbol}: not enough bars ({len(df)}) for strategy")
                continue

            close = df["close"]
            fast_sma = sma(close, self.fast)
            slow_sma = sma(close, self.slow)
            rsi_vals = rsi(close, self.rsi_period)

            # Current and previous bar values
            fast_now, fast_prev = fast_sma.iloc[-1], fast_sma.iloc[-2]
            slow_now, slow_prev = slow_sma.iloc[-1], slow_sma.iloc[-2]
            rsi_now = rsi_vals.iloc[-1]

            crossed_above = fast_prev < slow_prev and fast_now > slow_now
            crossed_below = fast_prev > slow_prev and fast_now < slow_now

            if crossed_above and rsi_now < self.rsi_overbought:
                confidence = 1.0 - (rsi_now / 100)  # higher confidence when RSI is lower
                signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    confidence=round(confidence, 2),
                    reason=f"SMA{self.fast} crossed above SMA{self.slow}, RSI={rsi_now:.1f}",
                ))
                logger.info(f"BUY signal: {symbol} | {signals[-1].reason}")

            elif crossed_below or rsi_now > self.rsi_overbought:
                reason = (
                    f"SMA{self.fast} crossed below SMA{self.slow}"
                    if crossed_below
                    else f"RSI overbought ({rsi_now:.1f})"
                )
                signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    confidence=1.0,
                    reason=reason,
                ))
                logger.info(f"SELL signal: {symbol} | {reason}")

        return signals
