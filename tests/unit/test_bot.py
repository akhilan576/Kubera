"""
Unit tests for bot.py — is_market_open() logic.
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import pytz

ET = pytz.timezone("America/New_York")


def make_et(year, month, day, hour, minute, weekday_override=None):
    """Create a timezone-aware ET datetime."""
    dt = ET.localize(datetime(year, month, day, hour, minute, 0))
    return dt


@pytest.mark.unit
class TestIsMarketOpen:
    """Tests for bot.is_market_open()"""

    def _patch_now(self, dt):
        return patch("bot.datetime")

    def test_open_during_trading_hours(self):
        from bot import is_market_open
        # Tuesday 10:00 ET — market is open
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 2, 10, 0))
            assert is_market_open() is True

    def test_closed_before_open(self):
        from bot import is_market_open
        # Tuesday 8:00 ET — before 9:30
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 2, 8, 0))
            assert is_market_open() is False

    def test_closed_after_close(self):
        from bot import is_market_open
        # Tuesday 16:30 ET — after 15:50
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 2, 16, 30))
            assert is_market_open() is False

    def test_closed_on_saturday(self):
        from bot import is_market_open
        # Saturday 11:00 ET
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 6, 11, 0))
            assert is_market_open() is False

    def test_closed_on_sunday(self):
        from bot import is_market_open
        # Sunday 11:00 ET
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 7, 11, 0))
            assert is_market_open() is False

    def test_open_at_exactly_market_open(self):
        from bot import is_market_open
        # Monday 9:30 ET exactly
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 8, 9, 30))
            assert is_market_open() is True

    def test_open_at_market_close_boundary(self):
        from bot import is_market_open
        # Monday 15:50 ET exactly — bot stops 10 min before exchange close
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 8, 15, 50))
            assert is_market_open() is True

    def test_closed_one_minute_after_cutoff(self):
        from bot import is_market_open
        # Monday 15:51 ET — past the 15:50 cutoff
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = ET.localize(datetime(2024, 1, 8, 15, 51))
            assert is_market_open() is False
