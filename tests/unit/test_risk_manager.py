"""
Unit tests for risk/manager.py
"""

import pytest
from risk.manager import RiskManager


@pytest.mark.unit
class TestCanOpenPosition:
    @pytest.fixture
    def risk(self):
        return RiskManager()

    def test_allows_when_below_max(self, risk):
        assert risk.can_open_position(0) is True
        assert risk.can_open_position(4) is True

    def test_blocks_at_max(self, risk):
        assert risk.can_open_position(5) is False

    def test_blocks_above_max(self, risk):
        assert risk.can_open_position(10) is False


@pytest.mark.unit
class TestPositionSize:
    @pytest.fixture
    def risk(self):
        return RiskManager()

    def test_risk_based_sizing(self, risk):
        # portfolio=$10k, price=$100, stop=2%, confidence=1.0
        # dollar_risk = 10000 * 0.02 * 1.0 = 200
        # risk_per_share = 100 * 0.02 = 2
        # shares = 200 / 2 = 100, capped at 10% = 10
        qty = risk.position_size(10_000, 100.0, confidence=1.0, stop_loss_pct=0.02)
        assert qty == 10   # capped by MAX_POSITION_PCT (10% of $10k = $1000 / $100 = 10)

    def test_pct_based_sizing(self, risk):
        # portfolio=$10k, price=$100, no stop_loss_pct
        # max_dollars = 10000 * 0.10 * 1.0 = 1000 → 10 shares
        qty = risk.position_size(10_000, 100.0, confidence=1.0)
        assert qty == 10

    def test_confidence_scales_size(self, risk):
        qty_full = risk.position_size(10_000, 100.0, confidence=1.0)
        qty_half = risk.position_size(10_000, 100.0, confidence=0.5)
        assert qty_half <= qty_full

    def test_returns_zero_for_zero_portfolio(self, risk):
        assert risk.position_size(0, 100.0) == 0

    def test_returns_zero_for_zero_price(self, risk):
        assert risk.position_size(10_000, 0.0) == 0

    def test_returns_zero_when_price_too_high(self, risk):
        # price > 10% of portfolio → less than 1 share
        qty = risk.position_size(100, 10_000.0)
        assert qty == 0

    def test_returns_integer(self, risk):
        qty = risk.position_size(10_000, 100.0)
        assert isinstance(qty, int)


@pytest.mark.unit
class TestStopLossPrice:
    @pytest.fixture
    def risk(self):
        return RiskManager()

    def test_default_2pct_stop(self, risk):
        assert risk.stop_loss_price(100.0) == pytest.approx(98.0)

    def test_custom_stop_pct(self, risk):
        assert risk.stop_loss_price(200.0, 0.05) == pytest.approx(190.0)

    def test_stop_below_entry(self, risk):
        entry = 150.0
        stop = risk.stop_loss_price(entry, 0.03)
        assert stop < entry


@pytest.mark.unit
class TestTakeProfitPrice:
    @pytest.fixture
    def risk(self):
        return RiskManager()

    def test_default_4pct_tp(self, risk):
        assert risk.take_profit_price(100.0) == pytest.approx(104.0)

    def test_custom_tp_pct(self, risk):
        assert risk.take_profit_price(200.0, 0.10) == pytest.approx(220.0)

    def test_tp_above_entry(self, risk):
        entry = 150.0
        tp = risk.take_profit_price(entry, 0.05)
        assert tp > entry

    def test_reward_risk_ratio(self, risk):
        """Default take-profit should be 2x the stop-loss distance (2:1 R/R)."""
        entry = 100.0
        stop_dist = entry - risk.stop_loss_price(entry, 0.02)
        tp_dist = risk.take_profit_price(entry, 0.04) - entry
        assert tp_dist == pytest.approx(stop_dist * 2, rel=0.01)
