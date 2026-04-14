"""
Integration tests for execution/broker.py — all Alpaca calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch, call


@pytest.mark.integration
class TestBrokerInit:
    def test_initialises_in_paper_mode(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        # paper=True because ALPACA_BASE_URL contains "paper"
        assert broker is not None

    def test_paper_flag_from_url(self):
        with patch("execution.broker.TradingClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("execution.broker.ALPACA_BASE_URL", "https://paper-api.alpaca.markets"):
                from execution.broker import Broker
                broker = Broker()
                _, kwargs = mock_cls.call_args
                assert kwargs.get("paper") is True


@pytest.mark.integration
class TestBrokerAccount:
    def test_get_portfolio_value(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        value = broker.get_portfolio_value()
        assert value == pytest.approx(10_000.0)

    def test_get_cash(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        cash = broker.get_cash()
        assert cash == pytest.approx(5_000.0)

    def test_get_account_called_once_per_method(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        broker.get_portfolio_value()
        assert mock_trading_client.get_account.call_count == 1


@pytest.mark.integration
class TestBrokerPositions:
    def test_get_positions_returns_dict(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        positions = broker.get_positions()
        assert isinstance(positions, dict)

    def test_get_positions_keyed_by_symbol(self, mock_trading_client):
        pos = MagicMock()
        pos.symbol = "AAPL"
        mock_trading_client.get_all_positions.return_value = [pos]

        from execution.broker import Broker
        broker = Broker()
        positions = broker.get_positions()
        assert "AAPL" in positions

    def test_has_position_true(self, mock_trading_client):
        mock_trading_client.get_open_position.return_value = MagicMock()
        from execution.broker import Broker
        broker = Broker()
        assert broker.has_position("AAPL") is True

    def test_has_position_false_when_none(self, mock_trading_client):
        mock_trading_client.get_open_position.side_effect = Exception("not found")
        from execution.broker import Broker
        broker = Broker()
        assert broker.has_position("AAPL") is False


@pytest.mark.integration
class TestBrokerOrders:
    def test_market_buy_submits_order(self, mock_trading_client):
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_trading_client.submit_order.return_value = mock_order

        from execution.broker import Broker
        broker = Broker()
        order = broker.market_buy("AAPL", qty=10)

        mock_trading_client.submit_order.assert_called_once()
        assert order.id == "order-123"

    def test_market_buy_skips_zero_qty(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        result = broker.market_buy("AAPL", qty=0)
        mock_trading_client.submit_order.assert_not_called()
        assert result is None

    def test_market_buy_returns_none_on_exception(self, mock_trading_client):
        mock_trading_client.submit_order.side_effect = Exception("API error")
        from execution.broker import Broker
        broker = Broker()
        result = broker.market_buy("AAPL", qty=5)
        assert result is None

    def test_market_sell_closes_position(self, mock_trading_client):
        mock_order = MagicMock()
        mock_order.id = "close-456"
        mock_trading_client.close_position.return_value = mock_order

        from execution.broker import Broker
        broker = Broker()
        order = broker.market_sell("AAPL")

        mock_trading_client.close_position.assert_called_once_with("AAPL")
        assert order.id == "close-456"

    def test_market_sell_specific_qty(self, mock_trading_client):
        mock_order = MagicMock()
        mock_trading_client.submit_order.return_value = mock_order

        from execution.broker import Broker
        broker = Broker()
        broker.market_sell("AAPL", qty=5)

        mock_trading_client.submit_order.assert_called_once()

    def test_market_sell_skips_zero_qty(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        result = broker.market_sell("AAPL", qty=0)
        mock_trading_client.submit_order.assert_not_called()
        assert result is None

    def test_cancel_all_orders(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        broker.cancel_all_orders()
        mock_trading_client.cancel_orders.assert_called_once()

    def test_close_all_positions(self, mock_trading_client):
        from execution.broker import Broker
        broker = Broker()
        broker.close_all_positions()
        mock_trading_client.close_all_positions.assert_called_once_with(cancel_orders=True)


@pytest.mark.integration
class TestBrokerErrorHandling:
    def test_market_sell_returns_none_on_exception(self, mock_trading_client):
        mock_trading_client.close_position.side_effect = Exception("API error")
        from execution.broker import Broker
        broker = Broker()
        result = broker.market_sell("AAPL")
        assert result is None

    def test_cancel_all_orders_silent_on_exception(self, mock_trading_client):
        mock_trading_client.cancel_orders.side_effect = Exception("API error")
        from execution.broker import Broker
        broker = Broker()
        # Should not raise
        broker.cancel_all_orders()
