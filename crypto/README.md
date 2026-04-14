# Project Zeno — Crypto Bot (freqtrade + Bybit)

## Setup

### 1. Install freqtrade
```bash
pip install freqtrade
```

### 2. Add Bybit API keys to config.json
Edit `crypto/config.json` and fill in your keys:
```json
"exchange": {
    "name": "bybit",
    "key": "YOUR_BYBIT_API_KEY",
    "secret": "YOUR_BYBIT_SECRET_KEY"
}
```
For dry-run (paper trading), keys are optional — leave them empty.

### 3. Run in dry-run mode
```bash
freqtrade trade --config crypto/config.json
```

### 4. Run backtesting
```bash
freqtrade download-data --config crypto/config.json --timerange 20240101-
freqtrade backtesting --config crypto/config.json --timerange 20240101-
```

## Files
- `config.json` — freqtrade config (Bybit, dry-run, pairs)
- `strategies/ZenoSMACrossover.py` — mirrors the equities SMA+RSI strategy
- `user_data/` — freqtrade data directory (logs, backtest results)
