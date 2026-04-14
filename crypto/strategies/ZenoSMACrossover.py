"""
ZenoSMACrossover — freqtrade strategy for Project Zeno crypto bot.
Mirrors the equities SMA crossover + RSI logic used in strategy/sma_crossover.py.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta


class ZenoSMACrossover(IStrategy):
    """
    Dual SMA crossover with RSI filter — crypto equivalent of Zeno's equities strategy.

    BUY  when fast SMA crosses above slow SMA and RSI < overbought threshold.
    SELL when fast SMA crosses below slow SMA or RSI > overbought threshold.
    """

    INTERFACE_VERSION = 3

    # Minimal ROI: exit at 4% profit immediately, scale down over time
    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "60": 0.01,
    }

    stoploss = -0.02          # 2% stop-loss — matches equities bot
    trailing_stop = False
    timeframe = "1h"

    # Hyperopt-tunable parameters
    fast_period = IntParameter(10, 30, default=20, space="buy")
    slow_period = IntParameter(30, 100, default=50, space="buy")
    rsi_overbought = DecimalParameter(65.0, 80.0, default=70.0, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMAs
        for period in range(10, 101, 5):
            dataframe[f"sma_{period}"] = ta.SMA(dataframe, timeperiod=period)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = f"sma_{self.fast_period.value}"
        slow = f"sma_{self.slow_period.value}"

        dataframe.loc[
            (
                (dataframe[fast] > dataframe[slow]) &                          # fast above slow
                (dataframe[fast].shift(1) <= dataframe[slow].shift(1)) &       # crossed above
                (dataframe["rsi"] < self.rsi_overbought.value) &               # RSI filter
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = f"sma_{self.fast_period.value}"
        slow = f"sma_{self.slow_period.value}"

        dataframe.loc[
            (
                (dataframe[fast] < dataframe[slow]) &                          # fast below slow
                (dataframe[fast].shift(1) >= dataframe[slow].shift(1))         # crossed below
            ) |
            (dataframe["rsi"] > self.rsi_overbought.value),                    # RSI overbought
            "exit_long"
        ] = 1

        return dataframe
