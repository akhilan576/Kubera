"""
Fine-tuned XGBoost trainer for Project Zeno.

Improvements over v1:
  - Optuna hyperparameter search (200 trials)
  - Purged walk-forward cross-validation (no lookahead leakage)
  - Early stopping per fold
  - SHAP-based feature selection (drop noisy features)
  - Class imbalance correction (scale_pos_weight)
  - Multi-horizon ensemble (1-bar, 3-bar, 5-bar labels combined)
  - SPY as market-regime context feature
  - Default lookback increased to 1000 bars

Usage:
    python -m models.trainer --symbols AAPL MSFT TSLA NVDA
    python -m models.trainer --symbols AAPL --trials 300 --lookback 1500
"""
from __future__ import annotations

import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap

from data.fetcher import DataFetcher
from data.features import build_features
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR       = "models/saved"
FORWARD_BARS    = [1, 3, 5]     # multi-horizon ensemble
LABEL_THRESHOLD = 0.001         # >= 0.1% = positive label
N_SPLITS        = 6             # purged CV folds
PURGE_BARS      = 5             # bars to drop between train/val (removes feature overlap)
MIN_TRAIN_BARS  = 120           # minimum bars required to train a fold
SHAP_DROP_PCT   = 0.05          # drop bottom 5% features by mean |SHAP|


# ── Purged walk-forward CV ────────────────────────────────────────────────────

def purged_cv_splits(n: int, n_splits: int, purge: int):
    """
    Yields (train_idx, val_idx) with a purge gap between train and val
    to eliminate feature-overlap leakage in time-series data.
    """
    fold_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        val_start = i * fold_size
        val_end   = val_start + fold_size
        train_end = val_start - purge
        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(val_start, min(val_end, n))
        if len(train_idx) >= MIN_TRAIN_BARS and len(val_idx) > 0:
            yield train_idx, val_idx


# ── Optuna objective ──────────────────────────────────────────────────────────

def _objective(trial, X: pd.DataFrame, y: pd.Series, pos_weight: float) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight":  pos_weight,
        "eval_metric":       "auc",
        "random_state":      42,
        "n_jobs":            -1,
        "early_stopping_rounds": 30,
    }

    auc_scores = []
    scaler = StandardScaler()

    for train_idx, val_idx in purged_cv_splits(len(X), N_SPLITS, PURGE_BARS):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            continue

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = XGBClassifier(**params)
        model.fit(
            X_tr_s, y_tr,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )
        prob = model.predict_proba(X_val_s)[:, 1]
        auc_scores.append(roc_auc_score(y_val, prob))

    return float(np.mean(auc_scores)) if auc_scores else 0.0


# ── SHAP feature selection ────────────────────────────────────────────────────

def _select_features(model: XGBClassifier, X: pd.DataFrame,
                     scaler: StandardScaler, drop_pct: float = SHAP_DROP_PCT) -> list[str]:
    """Return feature names to keep after dropping low-SHAP features."""
    X_s = scaler.transform(X)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s[:500])   # sample for speed
    mean_abs    = np.abs(shap_values).mean(axis=0)
    threshold   = np.percentile(mean_abs, drop_pct * 100)
    keep = [col for col, imp in zip(X.columns, mean_abs) if imp >= threshold]
    dropped = len(X.columns) - len(keep)
    logger.info(f"SHAP feature selection: keeping {len(keep)}/{len(X.columns)} features "
                f"(dropped {dropped} low-importance)")
    return keep


# ── Multi-horizon label builder ───────────────────────────────────────────────

def _build_ensemble_label(df: pd.DataFrame, horizons: list[int],
                           threshold: float) -> pd.Series:
    """
    Vote across multiple forward horizons.
    Label = 1 if majority of horizon labels are 1.
    """
    votes = pd.DataFrame(index=df.index)
    for h in horizons:
        future_ret   = df["close"].shift(-h) / df["close"] - 1
        votes[f"h{h}"] = (future_ret >= threshold).astype(int)
    label = (votes.sum(axis=1) > len(horizons) / 2).astype(int)
    label.name = "label"
    return label


# ── Main training function ────────────────────────────────────────────────────

def train_symbol(symbol: str, fetcher: DataFetcher,
                 lookback: int = 1000, n_trials: int = 200) -> str:
    """
    Full fine-tuned training pipeline for one symbol.
    Returns path to saved model.
    """
    logger.info(f"[{symbol}] ── Starting fine-tuned training (lookback={lookback}) ──")

    # Fetch symbol data
    bars = fetcher.get_bars([symbol], lookback=lookback)
    if symbol not in bars or bars[symbol].empty:
        raise ValueError(f"No data returned for {symbol}")
    df = bars[symbol]

    # Fetch SPY for market-regime features
    spy_bars = fetcher.get_bars(["SPY"], lookback=lookback)
    spy_df   = spy_bars.get("SPY", pd.DataFrame())

    # Build features with SPY context
    logger.info(f"[{symbol}] Building features...")
    features = build_features(df, spy_df=spy_df)

    # Multi-horizon ensemble label
    label = _build_ensemble_label(df, horizons=FORWARD_BARS, threshold=LABEL_THRESHOLD)

    # Align index
    common = features.index.intersection(label.dropna().index)
    X = features.loc[common]
    y = label.loc[common]

    # Remove last N rows (no future label)
    max_horizon = max(FORWARD_BARS)
    X = X.iloc[:-max_horizon]
    y = y.iloc[:-max_horizon]

    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    logger.info(f"[{symbol}] Samples={len(X)} | "
                f"class balance={y.mean():.2%} positive | "
                f"scale_pos_weight={pos_weight:.2f}")

    # ── Optuna hyperparameter search ─────────────────────────────────────────
    logger.info(f"[{symbol}] Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _objective(trial, X, y, pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    best_params["scale_pos_weight"] = pos_weight
    best_params["eval_metric"]      = "auc"
    best_params["random_state"]     = 42
    best_params["n_jobs"]           = -1
    best_params["early_stopping_rounds"] = 30

    logger.info(f"[{symbol}] Best CV AUC: {study.best_value:.4f}")
    logger.info(f"[{symbol}] Best params: {best_params}")

    # ── Final model: fit on full data with best params ────────────────────────
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Use last 20% as eval set for early stopping
    split  = int(len(X) * 0.8)
    model  = XGBClassifier(**best_params)
    model.fit(
        X_s[:split], y.iloc[:split],
        eval_set=[(X_s[split:], y.iloc[split:])],
        verbose=False,
    )

    # ── SHAP feature selection ────────────────────────────────────────────────
    logger.info(f"[{symbol}] Running SHAP feature selection...")
    keep_features = _select_features(model, X, scaler)

    # Retrain on selected features only
    X_sel  = X[keep_features]
    X_sel_s = scaler.fit_transform(X_sel)

    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_sel_s[:split], y.iloc[:split],
        eval_set=[(X_sel_s[split:], y.iloc[split:])],
        verbose=False,
    )

    # Final AUC
    prob     = final_model.predict_proba(X_sel_s[split:])[:, 1]
    final_auc = roc_auc_score(y.iloc[split:], prob)
    logger.info(f"[{symbol}] Final hold-out AUC: {final_auc:.4f}")

    # ── Save model bundle ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model":        final_model,
        "scaler":       scaler,
        "features":     keep_features,
        "best_params":  best_params,
        "cv_auc":       study.best_value,
        "holdout_auc":  final_auc,
        "symbol":       symbol,
        "trained_bars": len(X),
    }
    path = os.path.join(MODEL_DIR, f"{symbol}.pkl")
    joblib.dump(bundle, path)
    logger.info(f"[{symbol}] Model bundle saved → {path}")
    return path


# ── Load helper ───────────────────────────────────────────────────────────────

def load_model(symbol: str) -> dict | None:
    """
    Load saved model bundle for `symbol`.
    Returns dict with keys: model, scaler, features, best_params, cv_auc, holdout_auc
    Returns None if not found.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}.pkl")
    if not os.path.exists(path):
        logger.warning(f"No saved model for {symbol} at {path}")
        return None
    bundle = joblib.load(path)
    # Support old format (plain Pipeline) for backwards compat
    if not isinstance(bundle, dict):
        logger.warning(f"{symbol}: old model format detected, retraining recommended")
        return None
    return bundle


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned XGBoost trainer")
    parser.add_argument("--symbols",  nargs="+", required=True)
    parser.add_argument("--lookback", type=int,  default=1000)
    parser.add_argument("--trials",   type=int,  default=200)
    parser.add_argument("--crypto",   action="store_true", help="Use CryptoFetcher for crypto symbols")
    parser.add_argument("--india",    action="store_true", help="Use YFinanceFetcher for NSE symbols")
    args = parser.parse_args()

    if args.crypto:
        from data.crypto_fetcher import CryptoFetcher, normalise
        fetcher = CryptoFetcher()
        # Normalise symbols: BTC/USD → BTCUSD for file-safe model names
        symbols = [normalise(s) for s in args.symbols]
        # Monkey-patch fetcher to accept normalised symbols
        _orig_get_bars = fetcher.get_bars
        def _get_bars_normalised(syms, lookback=args.lookback):
            from data.crypto_fetcher import alpaca_symbol
            alpaca_syms = [alpaca_symbol(s) for s in syms]
            raw = _orig_get_bars(alpaca_syms, lookback=lookback)
            return {normalise(k): v for k, v in raw.items()}
        fetcher.get_bars = _get_bars_normalised
    elif args.india:
        from data.yfinance_fetcher import YFinanceFetcher
        fetcher = YFinanceFetcher(interval="15Min")
        symbols = args.symbols
    else:
        fetcher = DataFetcher()
        symbols = args.symbols

    for sym in symbols:
        try:
            train_symbol(sym, fetcher, lookback=args.lookback, n_trials=args.trials)
        except Exception as e:
            logger.error(f"Failed to train {sym}: {e}", exc_info=True)
