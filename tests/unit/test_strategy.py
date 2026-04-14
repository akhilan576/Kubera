"""
Unit tests for strategy/sma_crossover.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from strategy.sma_crossover import SMACrossoverStrategy
from strategy.base import Signal
from tests.conftest import make_ohlcv


@pytest.fixture
def strategy():
    return SMACrossoverStrategy(fast=20, slow=50)


@pytest.mark.unit
class TestSMACrossoverInit:
    def test_default_params(self):
        s = SMACrossoverStrategy()
        assert s.fast == 20
        assert s.slow == 50
        assert s.rsi_period == 14
        assert s.rsi_overbought == 70.0

    def test_custom_params(self):
        s = SMACrossoverStrategy(fast=10, slow=30, rsi_overbought=65.0)
        assert s.fast == 10
        assert s.slow == 30
        assert s.rsi_overbought == 65.0

    def test_repr(self):
        s = SMACrossoverStrategy()
        assert "SMA Crossover" in repr(s)


@pytest.mark.unit
class TestGenerateSignalsNoSignal:
    def test_not_enough_bars_returns_empty(self, strategy):
        df = make_ohlcv(30)   # less than slow (50) + 2
        signals = strategy.generate_signals({"AAPL": df})
        assert signals == []

    def test_empty_dict_returns_empty(self, strategy):
        assert strategy.generate_signals({}) == []

    def test_flat_price_no_crossover_no_signal(self, strategy):
        """Flat prices: SMAs are equal, no crossover occurs."""
        n = 60
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.full(n, 500_000.0),
        }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))
        signals = strategy.generate_signals({"AAPL": df})
        assert signals == []


def _make_sma_mock(df, fast_val_prev, fast_val_now, slow_val_prev, slow_val_now, fast, slow):
    """
    Return a side-effect function for `sma` that injects controlled crossover
    values at the last two bars while leaving the rest of the series natural.
    """
    def side_effect(series, period):
        result = series.rolling(window=period).mean().copy()
        if period == fast:
            result.iloc[-2] = fast_val_prev
            result.iloc[-1] = fast_val_now
        elif period == slow:
            result.iloc[-2] = slow_val_prev
            result.iloc[-1] = slow_val_now
        return result
    return side_effect


@pytest.mark.unit
class TestGenerateSignalsBuy:
    def test_buy_signal_on_upward_crossover(self, strategy):
        """Fast SMA crosses above slow SMA with RSI below overbought → BUY."""
        df = make_ohlcv(60)

        # Inject a controlled crossover: fast goes from 95→105, slow stays at 100
        sma_mock = _make_sma_mock(df, 95.0, 105.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 50.0), index=df.index)  # RSI=50, below 70

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        assert len(buy_signals) == 1
        assert buy_signals[0].symbol == "AAPL"

    def test_buy_signal_confidence_between_0_and_1(self, strategy):
        df = make_ohlcv(60)
        sma_mock = _make_sma_mock(df, 95.0, 105.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 50.0), index=df.index)

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        for s in signals:
            if s.signal == Signal.BUY:
                assert 0.0 <= s.confidence <= 1.0

    def test_buy_signal_has_reason(self, strategy):
        df = make_ohlcv(60)
        sma_mock = _make_sma_mock(df, 95.0, 105.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 50.0), index=df.index)

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        for s in signals:
            if s.signal == Signal.BUY:
                assert len(s.reason) > 0

    def test_no_buy_when_rsi_overbought(self, strategy):
        """Crossover fires but RSI is above overbought threshold → no BUY."""
        df = make_ohlcv(60)
        sma_mock = _make_sma_mock(df, 95.0, 105.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 80.0), index=df.index)  # RSI=80, overbought

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        assert len(buy_signals) == 0


@pytest.mark.unit
class TestGenerateSignalsSell:
    def test_sell_signal_on_downward_crossover(self, strategy):
        """Fast SMA crosses below slow SMA → SELL."""
        df = make_ohlcv(60)
        # Inject crossover: fast goes from 105→95, slow stays at 100
        sma_mock = _make_sma_mock(df, 105.0, 95.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 50.0), index=df.index)

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        assert len(sell_signals) == 1
        assert sell_signals[0].symbol == "AAPL"

    def test_sell_signal_when_rsi_overbought(self, strategy):
        """RSI above overbought threshold → SELL even without crossover."""
        df = make_ohlcv(60)
        # No crossover: fast stays below slow at both bars
        sma_mock = _make_sma_mock(df, 95.0, 95.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 80.0), index=df.index)  # RSI=80 > 70

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        assert len(sell_signals) == 1

    def test_sell_signal_has_reason(self, strategy):
        df = make_ohlcv(60)
        sma_mock = _make_sma_mock(df, 105.0, 95.0, 100.0, 100.0, fast=20, slow=50)
        rsi_result = pd.Series(np.full(len(df), 50.0), index=df.index)

        with patch("strategy.sma_crossover.sma", side_effect=sma_mock), \
             patch("strategy.sma_crossover.rsi", return_value=rsi_result):
            signals = strategy.generate_signals({"AAPL": df})

        for s in signals:
            if s.signal == Signal.SELL:
                assert len(s.reason) > 0


@pytest.mark.unit
class TestGenerateSignalsMultipleSymbols:
    def test_processes_all_symbols(self, strategy):
        bars = {
            "AAPL": make_ohlcv(100, seed=1),
            "MSFT": make_ohlcv(100, seed=2),
            "TSLA": make_ohlcv(100, seed=3),
        }
        signals = strategy.generate_signals(bars)
        symbols_with_signals = {s.symbol for s in signals}
        # All returned signals must be for known symbols
        assert symbols_with_signals.issubset({"AAPL", "MSFT", "TSLA"})

    def test_skips_symbol_with_missing_data(self, strategy):
        bars = {
            "AAPL": make_ohlcv(100),
            "BAD":  pd.DataFrame(),   # empty
        }
        # Should not raise; empty df is skipped
        signals = strategy.generate_signals(bars)
        assert all(s.symbol != "BAD" for s in signals)
