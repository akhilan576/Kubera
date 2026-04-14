"""
Feature engineering for the XGBoost model.
Takes a raw OHLCV DataFrame and returns a DataFrame of ML-ready features.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from data.indicators import sma, ema, rsi, macd, bollinger_bands, atr

# Lag periods for return features
LAG_PERIODS = [1, 2, 3, 5, 10]

# Rolling windows for statistical features
STAT_WINDOWS = [5, 10, 20]


def build_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Input:  OHLCV DataFrame (columns: open, high, low, close, volume)
            spy_df: optional SPY OHLCV for market-regime/correlation features
    Output: Feature DataFrame (same index, no NaNs)

    Feature groups:
        Trend       — SMA/EMA ratios, crossover distances
        Momentum    — RSI, MACD, rate-of-change
        Volatility  — ATR, Bollinger Bands, realised vol
        Volume      — volume ratios, OBV slope
        Candle      — body, wick, direction
        Lag returns — past 1/2/3/5/10 bar returns
        Calendar    — day-of-week, month, quarter effects
        Regime      — trend strength (ADX proxy), volatility regime
        Market      — SPY return correlation, beta (if spy_df provided)
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    open_ = df["open"]

    # ── Trend ────────────────────────────────────────────────────────────────
    for p in [10, 20, 50, 100]:
        feat[f"sma_{p}_ratio"] = close / sma(close, p)
    for p in [9, 21]:
        feat[f"ema_{p}_ratio"] = close / ema(close, p)

    feat["sma_10_20_cross"] = sma(close, 10) - sma(close, 20)
    feat["sma_20_50_cross"] = sma(close, 20) - sma(close, 50)
    feat["ema_9_21_cross"]  = ema(close, 9)  - ema(close, 21)

    # Price distance from 52-week high/low
    feat["dist_52w_high"] = close / close.rolling(252).max()
    feat["dist_52w_low"]  = close / close.rolling(252).min()

    # ── Momentum ─────────────────────────────────────────────────────────────
    feat["rsi_14"] = rsi(close, 14) / 100
    feat["rsi_7"]  = rsi(close, 7)  / 100
    feat["rsi_21"] = rsi(close, 21) / 100

    # RSI divergence proxy: RSI slope vs price slope
    feat["rsi_slope"]   = rsi(close, 14).diff(3)
    feat["price_slope"] = close.pct_change(3)

    macd_line, signal_line, histogram = macd(close, fast=12, slow=26, signal=9)
    feat["macd_hist"]      = histogram
    feat["macd_cross"]     = macd_line - signal_line
    feat["macd_line_norm"] = macd_line / close

    for p in [3, 5, 10, 20, 60]:
        feat[f"roc_{p}"] = close.pct_change(p)

    # ── Volatility ───────────────────────────────────────────────────────────
    feat["atr_14_ratio"] = atr(df, 14) / close
    feat["atr_7_ratio"]  = atr(df, 7)  / close

    upper, middle, lower = bollinger_bands(close, 20, 2.0)
    feat["bb_width"]    = (upper - lower) / middle
    feat["bb_position"] = (close - lower) / (upper - lower + 1e-9)
    feat["bb_squeeze"]  = feat["bb_width"] / feat["bb_width"].rolling(50).mean()  # squeeze detector

    # Realised volatility (rolling std of log returns)
    log_ret = np.log(close / close.shift(1))
    for w in STAT_WINDOWS:
        feat[f"realised_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Volatility regime: current vol vs long-term vol
    feat["vol_regime"] = feat["realised_vol_5"] / feat["realised_vol_20"]

    # High-Low range normalised
    feat["hl_range"]     = (high - low) / close
    feat["hl_range_ma"]  = feat["hl_range"].rolling(10).mean()

    # ── Volume ───────────────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        feat[f"vol_ratio_{w}"] = vol / vol.rolling(w).mean()

    # On-Balance Volume slope
    obv = (np.sign(close.diff()) * vol).cumsum()
    feat["obv_slope"] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-9)

    # Volume × price momentum
    feat["vol_price_trend"] = (close.pct_change() * vol).rolling(5).mean()

    # ── Candle structure ─────────────────────────────────────────────────────
    body       = (close - open_).abs() / (close + 1e-9)
    upper_wick = (high  - close.clip(lower=open_)) / (close + 1e-9)
    lower_wick = (close.clip(upper=open_) - low)   / (close + 1e-9)
    feat["candle_body"]       = body
    feat["candle_upper_wick"] = upper_wick
    feat["candle_lower_wick"] = lower_wick
    feat["candle_direction"]  = np.sign(close - open_)

    # Doji / indecision: small body relative to range
    feat["candle_doji"] = body / (feat["hl_range"] + 1e-9)

    # ── Lag returns ──────────────────────────────────────────────────────────
    for lag in LAG_PERIODS:
        feat[f"lag_ret_{lag}"] = close.pct_change(lag).shift(1)  # shift 1 to avoid leakage

    # Rolling mean and std of past returns (distribution features)
    for w in STAT_WINDOWS:
        past_ret = close.pct_change(1).shift(1)
        feat[f"ret_mean_{w}"] = past_ret.rolling(w).mean()
        feat[f"ret_std_{w}"]  = past_ret.rolling(w).std()
        feat[f"ret_skew_{w}"] = past_ret.rolling(w).skew()

    # ── Calendar effects ─────────────────────────────────────────────────────
    if hasattr(df.index, "dayofweek"):
        feat["day_of_week"] = df.index.dayofweek / 4.0        # 0=Mon, 1=Fri normalised
        feat["month"]       = df.index.month / 12.0
        feat["quarter"]     = df.index.quarter / 4.0
        feat["is_month_end"]   = df.index.is_month_end.astype(float)
        feat["is_month_start"] = df.index.is_month_start.astype(float)

    # ── Market regime (ADX proxy) ─────────────────────────────────────────────
    # Directional movement index approximation
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14     = atr(df, 14)
    plus_di   = 100 * plus_dm.rolling(14).mean()  / (atr14 + 1e-9)
    minus_di  = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    feat["adx"]        = dx.rolling(14).mean() / 100   # normalised 0-1
    feat["adx_trend"]  = (feat["adx"] > 0.25).astype(float)  # 1 = trending market

    # ── Market features (SPY correlation / beta) ─────────────────────────────
    if spy_df is not None and not spy_df.empty:
        spy_ret   = spy_df["close"].pct_change()
        stock_ret = close.pct_change()
        # Align indices
        common = spy_ret.index.intersection(stock_ret.index)
        spy_ret   = spy_ret.reindex(common)
        stock_ret = stock_ret.reindex(common)

        feat["spy_ret_1"]   = spy_ret.shift(1)
        feat["spy_corr_20"] = stock_ret.rolling(20).corr(spy_ret)
        cov20  = stock_ret.rolling(20).cov(spy_ret)
        var20  = spy_ret.rolling(20).var()
        feat["beta_20"] = cov20 / (var20 + 1e-9)

    # ── Macro features (VIX, TNX, DXY) ───────────────────────────────────────
    # macro_df columns: vix, tnx (10Y yield), dxy (dollar index)
    if macro_df is not None and not macro_df.empty:
        common = df.index.intersection(macro_df.index)
        macro  = macro_df.reindex(common)

        if "vix" in macro.columns:
            vix = macro["vix"]
            feat["vix"]          = vix.reindex(df.index).ffill()
            feat["vix_norm"]     = feat["vix"] / feat["vix"].rolling(20).mean()
            feat["vix_spike"]    = (feat["vix"] > feat["vix"].rolling(20).mean() * 1.5).astype(float)

        if "tnx" in macro.columns:
            tnx = macro["tnx"].reindex(df.index).ffill()
            feat["tnx"]          = tnx
            feat["tnx_slope"]    = tnx.diff(5)   # rising yields = risk-off

        if "dxy" in macro.columns:
            dxy = macro["dxy"].reindex(df.index).ffill()
            feat["dxy"]          = dxy / dxy.rolling(20).mean()   # normalised
            feat["dxy_slope"]    = dxy.pct_change(5)              # strong dollar = headwind

    feat.dropna(inplace=True)
    return feat


def build_labels(df: pd.DataFrame, forward_bars: int = 1, threshold: float = 0.001) -> pd.Series:
    """
    Binary label: 1 if close rises by >= threshold over next `forward_bars` bars, else 0.
    """
    future_return = df["close"].shift(-forward_bars) / df["close"] - 1
    labels = (future_return >= threshold).astype(int)
    labels.name = "label"
    return labels
