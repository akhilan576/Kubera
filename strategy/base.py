from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    symbol: str
    signal: Signal
    confidence: float = 1.0    # 0.0 - 1.0, used for position sizing
    reason: str = ""


class BaseStrategy(ABC):
    """
    All strategies must inherit from this class and implement `generate_signals`.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """
        Given a dict of {symbol: OHLCV DataFrame}, return a list of TradeSignals.
        Called once per bot loop iteration.
        """
        ...

    def __repr__(self):
        return f"<Strategy: {self.name}>"
