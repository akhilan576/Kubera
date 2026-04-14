"""
Pre-market daily screener — powered by yfinance (free, no rate limits).

Runs before market open each day and selects the top N symbols with:
  1. Highest Relative Volume (RVOL) — today's vol vs 20-day average
  2. Largest gap-up % from previous close
  3. Minimum liquidity filter (avg dollar volume)

Downloads all tickers in a single batch call — fast and reliable.

Usage (standalone):
    python -m data.screener
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd
from config.settings import SYMBOLS
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Screener config ───────────────────────────────────────────────────────────
TOP_N          = 10
MIN_PRICE      = 5.0
MAX_PRICE      = 5000.0
MIN_AVG_VOLUME = 500_000   # 20-day avg volume filter
MIN_RVOL       = 1.2       # today vol / 20-day avg
MIN_GAP_PCT    = -0.05     # allow slight gap-down (-5%)
LOOKBACK_DAYS  = "30d"

# Universe to scan
SCAN_UNIVERSE = list(dict.fromkeys(SYMBOLS + [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOGL", "AMD",
    "COIN", "MSTR", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "XYZ",
    "SPY",  "QQQ",  "ARKK", "SQQQ", "TQQQ",
    "JPM",  "BAC",  "GS",   "XOM",  "CVX",
    "NFLX", "UBER", "LYFT", "SNAP", "RBLX",
]))


class DailyScreener:
    def _fetch_all(self) -> dict[str, pd.DataFrame]:
        """Download daily bars for all symbols in one batch call."""
        try:
            raw = yf.download(
                " ".join(SCAN_UNIVERSE),
                period=LOOKBACK_DAYS,
                interval="1d",
                progress=False,
                group_by="ticker",
                auto_adjust=True,
            )

            result: dict[str, pd.DataFrame] = {}

            # Single ticker: raw is a plain DataFrame with columns Open/High/Low/Close/Volume
            if isinstance(raw.columns, pd.Index) and "Close" in raw.columns:
                sym = SCAN_UNIVERSE[0] if len(SCAN_UNIVERSE) == 1 else None
                if sym:
                    result[sym] = raw.dropna(how="all")
                return result

            # Multiple tickers: columns are MultiIndex (ticker, field)
            top_level = raw.columns.get_level_values(0).unique()
            for sym in top_level:
                try:
                    df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                    if not df.empty:
                        df.columns = [c.lower() for c in df.columns]
                        result[sym] = df
                except Exception:
                    pass

            logger.info(f"[SCREENER] Downloaded {len(result)}/{len(SCAN_UNIVERSE)} symbols")
            return result

        except Exception as e:
            logger.error(f"[SCREENER] Batch download failed — {e}")
            return {}

    def run(self, top_n: int = TOP_N) -> list[str]:
        """Run the screener. Returns ordered list of top symbols for today."""
        logger.info(f"[SCREENER] Scanning {len(SCAN_UNIVERSE)} symbols via yfinance...")
        try:
            scores = self._score_symbols()
            if not scores:
                logger.warning("[SCREENER] No qualifying symbols — falling back to configured SYMBOLS")
                return SYMBOLS

            ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
            top    = [r["symbol"] for r in ranked[:top_n]]

            logger.info(f"[SCREENER] Top {top_n} symbols for today:")
            for r in ranked[:top_n]:
                logger.info(
                    f"  {r['symbol']:<6} RVOL={r['rvol']:.2f}x "
                    f"gap={r['gap_pct']:+.2f}% "
                    f"price=${r['price']:.2f} "
                    f"score={r['score']:.3f}"
                )
            return top

        except Exception as e:
            logger.error(f"[SCREENER] Failed: {e} — falling back to configured SYMBOLS")
            return SYMBOLS

    def _score_symbols(self) -> list[dict]:
        bars = self._fetch_all()
        results = []

        for symbol in SCAN_UNIVERSE:
            df = bars.get(symbol)
            if df is None or len(df) < 5:
                continue

            price = float(df["close"].iloc[-1])
            if price < MIN_PRICE or price > MAX_PRICE:
                continue

            # 20-day avg volume (exclude today)
            avg_vol = float(df["volume"].iloc[:-1].mean())
            if avg_vol < MIN_AVG_VOLUME:
                continue

            today_vol = float(df["volume"].iloc[-1])
            rvol = today_vol / avg_vol if avg_vol > 0 else 0.0
            if rvol < MIN_RVOL:
                continue

            gap_pct = (
                (float(df["open"].iloc[-1]) - float(df["close"].iloc[-2]))
                / float(df["close"].iloc[-2])
            ) if len(df) >= 2 else 0.0

            if gap_pct < MIN_GAP_PCT:
                continue

            score = (rvol * 0.6) + (abs(gap_pct) * 100 * 0.4)

            results.append({
                "symbol":  symbol,
                "price":   price,
                "rvol":    rvol,
                "gap_pct": gap_pct * 100,
                "score":   score,
            })

        return results

    def get_gap_ups(self, min_gap_pct: float = 2.0) -> list[str]:
        scores = self._score_symbols()
        return [r["symbol"] for r in scores if r["gap_pct"] >= min_gap_pct]

    def get_high_rvol(self, min_rvol: float = 2.0) -> list[str]:
        scores = self._score_symbols()
        return [r["symbol"] for r in scores if r["rvol"] >= min_rvol]


if __name__ == "__main__":
    screener = DailyScreener()
    symbols  = screener.run()
    print(f"\nToday's trading universe: {symbols}")
