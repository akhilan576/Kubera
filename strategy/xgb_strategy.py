"""
XGBoost-based trading strategy.

Requires models to be pre-trained:
    python -m models.trainer --symbols AAPL MSFT TSLA NVDA

Then use in bot.py:
    from strategy.xgb_strategy import XGBStrategy
    strategy = XGBStrategy(buy_threshold=0.60, sell_threshold=0.40)
"""

from __future__ import annotations
import pandas as pd
from strategy.base import BaseStrategy, TradeSignal, Signal
from data.features import build_features
from models.trainer import load_model
from utils.logger import get_logger

logger = get_logger(__name__)

# SPY is used as market-regime context — must be present in bars dict
SPY_KEY = "SPY"


class XGBStrategy(BaseStrategy):
    """
    Uses a per-symbol XGBoost classifier to predict the probability that
    the next bar closes higher than the current bar.

    BUY  when predicted up-probability >= buy_threshold
    SELL when predicted up-probability <= sell_threshold (or no model loaded)

    Expects bars dict to include "SPY" for market-regime features.
    SPY is used as context only — no signal is generated for it.
    """

    def __init__(self, buy_threshold: float = 0.60, sell_threshold: float = 0.40,
                 auc_threshold: float = 0.50):
        super().__init__("XGBoost Classifier")
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold
        self.auc_threshold  = auc_threshold
        self._bundles: dict = {}
        self.macro_df       = None   # set externally before generate_signals if available

    def _get_bundle(self, symbol: str) -> dict | None:
        """Lazy-load model bundle for symbol, cache it."""
        if symbol not in self._bundles:
            self._bundles[symbol] = load_model(symbol)  # None if not found
        return self._bundles[symbol]

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        signals = []
        spy_df = bars.get(SPY_KEY)  # may be None — features.py handles gracefully

        for symbol, df in bars.items():
            if symbol == SPY_KEY:
                continue  # context only, never trade SPY here

            bundle = self._get_bundle(symbol)
            if bundle is None:
                logger.warning(f"{symbol}: no trained model found, skipping")
                continue

            holdout_auc = bundle.get("holdout_auc", 1.0)
            if holdout_auc < self.auc_threshold:
                logger.info(f"{symbol}: XGB gated out (holdout_auc={holdout_auc:.3f} < {self.auc_threshold}) — deferring to sentiment")
                continue

            model         = bundle["model"]
            scaler        = bundle["scaler"]
            keep_features = bundle["features"]

            # Build full feature set (with SPY + macro context if available)
            features = build_features(df, spy_df=spy_df, macro_df=self.macro_df)
            if features.empty:
                logger.debug(f"{symbol}: not enough data to build features")
                continue

            # Keep only the features selected during training
            missing = [f for f in keep_features if f not in features.columns]
            if missing:
                logger.warning(f"{symbol}: {len(missing)} trained features missing in live data, skipping")
                continue

            latest = features[keep_features].iloc[[-1]]

            try:
                latest_scaled = scaler.transform(latest)
                prob_up = float(model.predict_proba(latest_scaled)[0, 1])
            except Exception as e:
                logger.error(f"{symbol}: prediction failed — {e}")
                continue

            logger.debug(f"{symbol}: P(up)={prob_up:.3f} | "
                         f"cv_auc={bundle.get('cv_auc', 0):.3f} | "
                         f"holdout_auc={bundle.get('holdout_auc', 0):.3f}")

            if prob_up >= self.buy_threshold:
                signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    confidence=round(prob_up, 3),
                    reason=f"XGB P(up)={prob_up:.3f} >= {self.buy_threshold}",
                ))
                logger.info(f"BUY signal: {symbol} | P(up)={prob_up:.3f}")

            elif prob_up <= self.sell_threshold:
                signals.append(TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    confidence=round(1 - prob_up, 3),
                    reason=f"XGB P(up)={prob_up:.3f} <= {self.sell_threshold}",
                ))
                logger.info(f"SELL signal: {symbol} | P(up)={prob_up:.3f}")

        return signals
