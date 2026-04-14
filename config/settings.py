import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca API ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # paper by default

# --- Stock Trading Universe ---
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,TSLA,NVDA,AMZN,META,GOOGL,AMD,MSTR,PLTR,COIN,NFLX,ARM,SMCI,MARA,RIOT,GME,HOOD,UPST,AVGO,SQ,SOFI,CLSK,HUT,DELL,BABA").split(",")

# --- Crypto Trading Universe (Alpaca format: BASE/USD) ---
CRYPTO_SYMBOLS = os.getenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,AVAX/USD").split(",")
CRYPTO_ENABLED = os.getenv("CRYPTO_ENABLED", "false").lower() == "true"

# --- Crypto Risk (notional USD per position) ---
MAX_CRYPTO_POSITION_USD = float(os.getenv("MAX_CRYPTO_POSITION_USD", "2000"))  # $2000 max per crypto
MAX_CRYPTO_POSITIONS    = int(os.getenv("MAX_CRYPTO_POSITIONS", "3"))           # max 3 crypto at once

# --- Timeframe ---
BAR_TIMEFRAME = "15Min"  # 1Min, 5Min, 15Min, 1Hour, 1Day
LOOKBACK_BARS = 2000      # ~77 trading days of 15-min bars (~4 months)

# --- Risk Management ---
MAX_POSITION_PCT = 0.10       # max 10% of portfolio per position
MAX_PORTFOLIO_RISK_PCT = 0.02 # max 2% portfolio loss per trade
MAX_OPEN_POSITIONS = 15       # max simultaneous positions

# --- Scheduling ---
RUN_INTERVAL_SECONDS = 60     # check every minute
MARKET_OPEN_TIME = "09:30"    # ET
MARKET_CLOSE_TIME = "15:55"   # ET
EOD_LIQUIDATE_TIME = "15:45"  # ET — close all positions before market close
OPEN_BURST_MINUTES = 5        # max 2 new positions in the first N minutes after open
OPEN_BURST_MAX = 2            # max new positions during burst window

# --- Polygon (Massive) ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/bot.log"
