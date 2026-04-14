"""
Risk Manager — Phase 1 upgrade.

Improvements:
  - ATR-based dynamic stop-loss and take-profit (replaces fixed %)
  - Trailing stop tracking (per-symbol high-water mark)
  - Kelly Criterion position sizing (uses XGBoost confidence + historical win rate)
  - Market breadth filter (SPY 200DMA gate)
"""
from __future__ import annotations

import pandas as pd
from config.settings import (
    MAX_POSITION_PCT,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_OPEN_POSITIONS,
)
from data.indicators import atr, sma
from utils.logger import get_logger

logger = get_logger(__name__)

# ATR multipliers
ATR_STOP_MULTIPLIER   = 2.0   # stop-loss = entry - 2 × ATR
ATR_TP_MULTIPLIER     = 4.0   # take-profit = entry + 4 × ATR (2:1 reward/risk)
ATR_PERIOD            = 14

# Trailing stop: activate once position gains this % then trail by ATR
TRAILING_ACTIVATE_PCT = 0.03  # activate trailing stop at +3% profit
TRAILING_ATR_MULT     = 1.5   # trail by 1.5 × ATR

# Kelly Criterion
KELLY_FRACTION        = 0.25  # use quarter-Kelly for safety (full Kelly is too aggressive)
MIN_WIN_RATE          = 0.45  # minimum historical win rate to trade
MAX_KELLY_PCT         = 0.30  # cap Kelly at 30% of portfolio regardless

# Market breadth
SPY_SMA_PERIOD        = 200   # SPY 200-day MA filter
BEAR_MARKET_SIZE_MULT = 0.50  # cut position size by 50% in bear market


class RiskManager:
    """
    Dynamic risk manager with ATR stops, trailing stops, Kelly sizing,
    and market breadth filter.
    """

    def __init__(self):
        self.max_position_pct   = MAX_POSITION_PCT
        self.max_risk_pct       = MAX_PORTFOLIO_RISK_PCT
        self.max_open_positions = MAX_OPEN_POSITIONS

        # Trailing stop state: {symbol: highest_price_seen}
        self._trailing_highs: dict[str, float] = {}

        # Historical win rate per symbol (updated from DB fills)
        self._win_rates: dict[str, float] = {}

        # Market regime (updated each tick)
        self._bear_market = False

    # ── Position gate ─────────────────────────────────────────────────────────

    def can_open_position(self, current_open_positions: int) -> bool:
        if current_open_positions >= self.max_open_positions:
            logger.warning(
                f"Max open positions reached "
                f"({current_open_positions}/{self.max_open_positions})"
            )
            return False
        return True

    # ── Market breadth filter ─────────────────────────────────────────────────

    def update_market_regime(self, spy_df: pd.DataFrame):
        """
        Call once per tick with SPY OHLCV bars.
        Sets bear market flag if SPY is below its 200-day SMA.
        """
        if spy_df is None or len(spy_df) < SPY_SMA_PERIOD:
            return
        spy_sma200 = sma(spy_df["close"], SPY_SMA_PERIOD).iloc[-1]
        spy_price  = float(spy_df["close"].iloc[-1])
        was_bear   = self._bear_market
        self._bear_market = spy_price < spy_sma200
        if self._bear_market != was_bear:
            regime = "BEAR" if self._bear_market else "BULL"
            logger.warning(f"Market regime change → {regime} "
                           f"(SPY={spy_price:.2f} vs SMA200={spy_sma200:.2f})")

    @property
    def is_bear_market(self) -> bool:
        return self._bear_market

    # ── ATR-based stops ───────────────────────────────────────────────────────

    def atr_stop_loss(self, df: pd.DataFrame, entry_price: float,
                      multiplier: float = ATR_STOP_MULTIPLIER) -> float:
        """
        Dynamic stop-loss = entry - (ATR × multiplier).
        Widens in volatile markets, tightens in quiet markets.
        """
        atr_val = self._get_atr(df)
        stop    = entry_price - (atr_val * multiplier)
        logger.debug(f"ATR stop: entry={entry_price:.2f} ATR={atr_val:.2f} "
                     f"stop={stop:.2f} ({((entry_price-stop)/entry_price):.1%})")
        return round(max(stop, entry_price * 0.80), 4)  # floor at -20%

    def atr_take_profit(self, df: pd.DataFrame, entry_price: float,
                        multiplier: float = ATR_TP_MULTIPLIER) -> float:
        """
        Dynamic take-profit = entry + (ATR × multiplier).
        """
        atr_val = self._get_atr(df)
        tp      = entry_price + (atr_val * multiplier)
        logger.debug(f"ATR TP: entry={entry_price:.2f} ATR={atr_val:.2f} "
                     f"tp={tp:.2f} ({((tp-entry_price)/entry_price):.1%})")
        return round(tp, 4)

    def _get_atr(self, df: pd.DataFrame) -> float:
        """Return latest ATR value from OHLCV DataFrame."""
        try:
            atr_series = atr(df, ATR_PERIOD)
            val = float(atr_series.dropna().iloc[-1])
            return val if val > 0 else df["close"].iloc[-1] * 0.02
        except Exception:
            return float(df["close"].iloc[-1]) * 0.02  # fallback: 2% of price

    # ── Trailing stop ─────────────────────────────────────────────────────────

    def update_trailing_stop(self, symbol: str, current_price: float):
        """
        Call every tick for open positions.
        Updates the high-water mark for trailing stop calculation.
        """
        prev_high = self._trailing_highs.get(symbol, current_price)
        self._trailing_highs[symbol] = max(prev_high, current_price)

    def trailing_stop_price(self, symbol: str, entry_price: float,
                            df: pd.DataFrame) -> float | None:
        """
        Returns trailing stop price once position is profitable enough.
        Returns None if trailing stop not yet activated.
        """
        high_water = self._trailing_highs.get(symbol, entry_price)
        gain_pct   = (high_water - entry_price) / entry_price

        if gain_pct < TRAILING_ACTIVATE_PCT:
            return None  # not yet activated

        atr_val      = self._get_atr(df)
        trail_stop   = high_water - (atr_val * TRAILING_ATR_MULT)
        logger.debug(f"{symbol}: trailing stop activated | "
                     f"high={high_water:.2f} trail={trail_stop:.2f} gain={gain_pct:.1%}")
        return round(trail_stop, 4)

    def should_trail_exit(self, symbol: str, current_price: float,
                          entry_price: float, df: pd.DataFrame) -> bool:
        """Returns True if current price has dropped below the trailing stop."""
        stop = self.trailing_stop_price(symbol, entry_price, df)
        if stop is None:
            return False
        triggered = current_price <= stop
        if triggered:
            logger.info(f"{symbol}: trailing stop triggered at {current_price:.2f} "
                        f"(stop={stop:.2f})")
        return triggered

    def clear_trailing(self, symbol: str):
        """Call when a position is closed."""
        self._trailing_highs.pop(symbol, None)

    # ── Kelly Criterion sizing ────────────────────────────────────────────────

    def kelly_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        symbol: str | None = None,
    ) -> int:
        """
        Kelly Criterion position sizing.

        Kelly % = W - (1-W)/R
          W = win rate (from historical fills or confidence proxy)
          R = reward/risk ratio (ATR_TP_MULTIPLIER / ATR_STOP_MULTIPLIER)

        Uses quarter-Kelly (KELLY_FRACTION) for safety.
        Halved further in bear markets.
        """
        if portfolio_value <= 0 or price <= 0:
            return 0

        # Win rate: use historical if available, else use confidence as proxy
        win_rate = self._win_rates.get(symbol or "", None)
        if win_rate is None:
            # Proxy: XGBoost confidence maps to win rate estimate
            win_rate = 0.45 + (confidence - 0.5) * 0.5
            win_rate = max(0.35, min(0.70, win_rate))

        if win_rate < MIN_WIN_RATE:
            logger.debug(f"{symbol}: win_rate={win_rate:.2%} below minimum, skipping")
            return 0

        reward_risk = ATR_TP_MULTIPLIER / ATR_STOP_MULTIPLIER  # default 2.0
        loss_rate   = 1.0 - win_rate

        # Full Kelly formula
        full_kelly = win_rate - (loss_rate / reward_risk)
        full_kelly = max(0.0, full_kelly)

        # Apply fraction + confidence scaling
        kelly_pct = full_kelly * KELLY_FRACTION * confidence

        # Bear market: halve sizing
        if self._bear_market:
            kelly_pct *= BEAR_MARKET_SIZE_MULT
            logger.debug(f"{symbol}: bear market — Kelly halved to {kelly_pct:.2%}")

        # Cap at MAX_KELLY_PCT and MAX_POSITION_PCT
        kelly_pct = min(kelly_pct, MAX_KELLY_PCT, self.max_position_pct)

        dollar_amount = portfolio_value * kelly_pct
        shares        = int(dollar_amount / price)

        if shares < 1:
            logger.debug(f"{symbol}: Kelly size < 1 share "
                         f"(kelly={kelly_pct:.2%}, price={price:.2f})")
            return 0

        logger.info(f"{symbol}: Kelly sizing → {shares} shares @ ${price:.2f} "
                    f"| kelly={kelly_pct:.2%} win_rate={win_rate:.2%} "
                    f"bear={self._bear_market}")
        return shares

    def update_win_rate(self, symbol: str, win_rate: float):
        """Update historical win rate from DB fills."""
        self._win_rates[symbol] = round(win_rate, 4)
        logger.debug(f"{symbol}: win rate updated → {win_rate:.2%}")

    # ── Legacy helpers (kept for compatibility) ───────────────────────────────

    def position_size(self, portfolio_value: float, price: float,
                      confidence: float = 1.0,
                      stop_loss_pct: float | None = None) -> int:
        """Fallback fixed-fractional sizing. Prefer kelly_position_size."""
        if portfolio_value <= 0 or price <= 0:
            return 0
        if stop_loss_pct and stop_loss_pct > 0:
            dollar_risk    = portfolio_value * self.max_risk_pct * confidence
            risk_per_share = price * stop_loss_pct
            shares         = int(dollar_risk / risk_per_share)
        else:
            shares = int(portfolio_value * self.max_position_pct * confidence / price)
        max_shares = int((portfolio_value * self.max_position_pct) / price)
        shares     = min(shares, max_shares)
        return max(0, shares)

    def stop_loss_price(self, entry_price: float, stop_loss_pct: float = 0.05) -> float:
        return round(entry_price * (1 - stop_loss_pct), 4)

    def take_profit_price(self, entry_price: float, take_profit_pct: float = 0.15) -> float:
        return round(entry_price * (1 + take_profit_pct), 4)
