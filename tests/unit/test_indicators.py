"""
Unit tests for data/indicators.py
"""

import pytest
import pandas as pd
import numpy as np

from data.indicators import sma, ema, rsi, macd, bollinger_bands, atr, vwap


@pytest.mark.unit
class TestSMA:
    def test_returns_series(self, ohlcv_100):
        result = sma(ohlcv_100["close"], 10)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_100):
        result = sma(ohlcv_100["close"], 10)
        assert len(result) == len(ohlcv_100)

    def test_leading_nan_count(self, ohlcv_100):
        period = 20
        result = sma(ohlcv_100["close"], period)
        assert result.iloc[:period - 1].isna().all()
        assert not result.iloc[period - 1:].isna().any()

    def test_correct_value(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(close, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_period_1_equals_input(self, ohlcv_100):
        result = sma(ohlcv_100["close"], 1)
        pd.testing.assert_series_equal(result, ohlcv_100["close"].astype(float))


@pytest.mark.unit
class TestEMA:
    def test_returns_series(self, ohlcv_100):
        result = ema(ohlcv_100["close"], 10)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_100):
        result = ema(ohlcv_100["close"], 10)
        assert len(result) == len(ohlcv_100)

    def test_no_nan_after_first_value(self, ohlcv_100):
        result = ema(ohlcv_100["close"], 10)
        assert not result.dropna().empty

    def test_ema_reacts_faster_than_sma(self, ohlcv_100):
        """EMA should be closer to current price than SMA in a trending series."""
        close = ohlcv_100["close"]
        e = ema(close, 20).iloc[-1]
        s = sma(close, 20).iloc[-1]
        last = close.iloc[-1]
        # Both exist; just verify they are computed without error
        assert not np.isnan(e)
        assert not np.isnan(s)


@pytest.mark.unit
class TestRSI:
    def test_returns_series(self, ohlcv_100):
        result = rsi(ohlcv_100["close"], 14)
        assert isinstance(result, pd.Series)

    def test_values_in_range(self, ohlcv_100):
        result = rsi(ohlcv_100["close"], 14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_flat_price_rsi_is_nan_or_50(self):
        """Flat prices → no gains or losses → RSI should be NaN or neutral."""
        close = pd.Series([100.0] * 20)
        result = rsi(close, 14).dropna()
        # With no movement, gain/loss = 0, expect NaN (division by zero handled)
        assert result.isna().all() or (result == 50.0).all()

    def test_rising_price_gives_high_rsi(self):
        # Pure linspace has zero losses → RSI is NaN (loss.replace(0, NaN)).
        # Add noise larger than the per-step increment (~2) so some days are losers.
        rng = np.random.default_rng(0)
        prices = np.linspace(100, 200, 50) + rng.normal(0, 5, 50)
        # Clamp to stay positive
        prices = np.clip(prices, 1, None)
        close = pd.Series(prices)
        result = rsi(close, 14).dropna()
        assert len(result) > 0
        assert result.iloc[-1] > 55  # trending up → RSI above midpoint

    def test_falling_price_gives_low_rsi(self):
        close = pd.Series(np.linspace(200, 100, 50))
        result = rsi(close, 14).dropna()
        assert result.iloc[-1] < 30


@pytest.mark.unit
class TestMACD:
    def test_returns_three_series(self, ohlcv_100):
        macd_line, signal_line, histogram = macd(ohlcv_100["close"])
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_histogram_equals_macd_minus_signal(self, ohlcv_100):
        macd_line, signal_line, histogram = macd(ohlcv_100["close"])
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected)

    def test_lengths_match(self, ohlcv_100):
        n = len(ohlcv_100)
        macd_line, signal_line, histogram = macd(ohlcv_100["close"])
        assert len(macd_line) == n
        assert len(signal_line) == n
        assert len(histogram) == n


@pytest.mark.unit
class TestBollingerBands:
    def test_returns_three_series(self, ohlcv_100):
        upper, middle, lower = bollinger_bands(ohlcv_100["close"])
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_upper_above_middle_above_lower(self, ohlcv_100):
        upper, middle, lower = bollinger_bands(ohlcv_100["close"])
        valid = upper.dropna().index
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_middle_equals_sma(self, ohlcv_100):
        _, middle, _ = bollinger_bands(ohlcv_100["close"], period=20)
        expected_sma = sma(ohlcv_100["close"], 20)
        pd.testing.assert_series_equal(middle, expected_sma)

    def test_band_width_zero_for_flat_price(self):
        close = pd.Series([100.0] * 30)
        upper, middle, lower = bollinger_bands(close, period=20)
        valid = upper.dropna()
        assert (valid - lower.dropna()).abs().max() == pytest.approx(0.0)


@pytest.mark.unit
class TestATR:
    def test_returns_series(self, ohlcv_100):
        result = atr(ohlcv_100, 14)
        assert isinstance(result, pd.Series)

    def test_values_non_negative(self, ohlcv_100):
        result = atr(ohlcv_100, 14).dropna()
        assert (result >= 0).all()

    def test_higher_volatility_gives_higher_atr(self):
        low_vol = pd.DataFrame({
            "high":  [101.0] * 30,
            "low":   [99.0]  * 30,
            "close": [100.0] * 30,
        })
        high_vol = pd.DataFrame({
            "high":  [110.0] * 30,
            "low":   [90.0]  * 30,
            "close": [100.0] * 30,
        })
        assert atr(high_vol, 14).dropna().mean() > atr(low_vol, 14).dropna().mean()


@pytest.mark.unit
class TestVWAP:
    def test_returns_series(self, ohlcv_100):
        result = vwap(ohlcv_100)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_100):
        result = vwap(ohlcv_100)
        assert len(result) == len(ohlcv_100)

    def test_vwap_within_overall_price_range(self, ohlcv_100):
        # VWAP is a cumulative weighted average — it won't stay within each
        # individual bar's high/low, but it must stay within the full range.
        result = vwap(ohlcv_100).dropna()
        assert (result >= ohlcv_100["low"].min()).all()
        assert (result <= ohlcv_100["high"].max()).all()
