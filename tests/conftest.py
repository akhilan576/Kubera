"""
Shared fixtures for Project Zeno test suite.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# -----------------------------------------------------------------------
# OHLCV data helpers
# -----------------------------------------------------------------------

def make_ohlcv(n: int = 100, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with a realistic price series.
    Useful for deterministic strategy and indicator tests.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)

    index = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_crossover_up(fast: int = 20, slow: int = 50, extra: int = 5) -> pd.DataFrame:
    """
    Build an OHLCV DataFrame whose last bar triggers a fast-above-slow crossover.
    Price rises sharply at the end so fast SMA crosses above slow SMA.
    """
    n = slow + extra + 2
    prices = np.full(n, 100.0)
    # Lift the last (extra+2) bars so fast SMA > slow SMA at the final bar
    prices[-(extra + 2):] = 115.0
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.full(n, 500_000.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))
    return df


def make_crossover_down(fast: int = 20, slow: int = 50, extra: int = 5) -> pd.DataFrame:
    """
    Build an OHLCV DataFrame whose last bar triggers a fast-below-slow crossover.
    Price drops sharply at the end so fast SMA crosses below slow SMA.
    """
    n = slow + extra + 2
    prices = np.full(n, 100.0)
    prices[-(extra + 2):] = 85.0
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.full(n, 500_000.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))
    return df


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def ohlcv_100():
    """100-bar synthetic OHLCV DataFrame."""
    return make_ohlcv(100)


@pytest.fixture
def ohlcv_200():
    """200-bar synthetic OHLCV DataFrame."""
    return make_ohlcv(200)


@pytest.fixture
def bars_dict(ohlcv_100):
    """Dict of {symbol: DataFrame} as produced by DataFetcher."""
    return {
        "AAPL": ohlcv_100.copy(),
        "MSFT": ohlcv_100.copy(),
    }


@pytest.fixture
def mock_trading_client():
    """Mock for alpaca.trading.client.TradingClient."""
    with patch("execution.broker.TradingClient") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        # Default account values
        account = MagicMock()
        account.portfolio_value = "10000.00"
        account.cash = "5000.00"
        account.status = "ACTIVE"
        instance.get_account.return_value = account

        # Default: no open positions
        instance.get_all_positions.return_value = []

        yield instance


@pytest.fixture
def mock_data_client():
    """Mock for alpaca.data.historical.StockHistoricalDataClient."""
    with patch("data.fetcher.StockHistoricalDataClient") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance
