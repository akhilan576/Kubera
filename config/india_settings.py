import os
from dotenv import load_dotenv

load_dotenv()

# --- Angel One (used when account is ready) ---
ANGEL_API_KEY     = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID   = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD    = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")

# --- Symbols (quality NSE stocks ≥ ₹150 — penny stocks removed) ---
INDIA_SYMBOLS = os.getenv(
    "INDIA_SYMBOLS",
    "ONGC,NTPC,COALINDIA,ITC,WIPRO,HINDALCO,POWERGRID,SBIN,AXISBANK,ICICIBANK,RELIANCE,INFY,BHARTIARTL,HDFCBANK,SUNPHARMA,CANBK,IRFC,ETERNAL,TATAPOWER,TATASTEEL,JSWSTEEL,BAJFINANCE,MARUTI,LT"
).split(",")

INDIA_ENABLED = os.getenv("INDIA_ENABLED", "false").lower() == "true"

# --- Market hours (IST, 24h) ---
INDIA_MARKET_OPEN    = "09:30"  # skip volatile opening auction (9:15–9:30)
INDIA_MARKET_CLOSE   = "15:25"
INDIA_EOD_LIQUIDATE  = "15:20"

# --- Bar settings ---
INDIA_BAR_TIMEFRAME = "15Min"
INDIA_LOOKBACK_BARS = 1000

# --- Risk ---
INDIA_MAX_POSITION_PCT   = 0.20   # 20% per position — larger bets, fewer trades
INDIA_MAX_OPEN_POSITIONS = 5      # max 5 concurrent positions (was 10)
INDIA_STARTING_CASH      = float(os.getenv("INDIA_PAPER_STARTING_CASH", "50000"))
INDIA_TRAILING_STOP_PCT  = 1.5
INDIA_RUN_INTERVAL_SECS  = 60
