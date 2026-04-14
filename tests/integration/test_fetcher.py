"""
Integration tests for data/fetcher.py — Alpaca client is mocked.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tests.conftest import make_ohlcv


def _make_alpaca_bars_response(symbols: list[str], n: int = 60) -> MagicMock:
    """Build a mock bars response that mimics alpaca-py's return value."""
    response = MagicMock()

    def getitem(symbol):
        df = make_ohlcv(n)
        mock_sym = MagicMock()
        mock_sym.df = df
        return mock_sym

    response.__getitem__ = MagicMock(side_effect=getitem)
    return response


@pytest.mark.integration
class TestDataFetcherGetBars:
    def test_returns_dict(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert isinstance(result, dict)

    def test_keys_are_symbols(self, mock_data_client):
        symbols = ["AAPL", "MSFT"]
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(symbols)
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(symbols)
        assert set(result.keys()) == set(symbols)

    def test_values_are_dataframes(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert isinstance(result["AAPL"], pd.DataFrame)

    def test_dataframe_has_required_columns(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert set(result["AAPL"].columns) == {"open", "high", "low", "close", "volume"}

    def test_returns_empty_dict_on_exception(self, mock_data_client):
        mock_data_client.get_stock_bars.side_effect = Exception("Network error")
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert result == {}

    def test_respects_lookback_limit(self, mock_data_client):
        lookback = 10
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"], n=100)
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"], lookback=lookback)
        assert len(result["AAPL"]) <= lookback

    def test_index_is_datetime(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert pd.api.types.is_datetime64_any_dtype(result["AAPL"].index)

    def test_data_is_sorted_ascending(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        result = fetcher.get_bars(["AAPL"])
        assert result["AAPL"].index.is_monotonic_increasing


@pytest.mark.integration
class TestDataFetcherGetLatestPrice:
    def test_returns_float(self, mock_data_client):
        mock_data_client.get_stock_bars.return_value = _make_alpaca_bars_response(["AAPL"])
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        price = fetcher.get_latest_price("AAPL")
        assert isinstance(price, float)

    def test_returns_none_on_exception(self, mock_data_client):
        mock_data_client.get_stock_bars.side_effect = Exception("error")
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        price = fetcher.get_latest_price("AAPL")
        assert price is None
