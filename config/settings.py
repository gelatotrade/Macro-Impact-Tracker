"""
Configuration settings for the Macro Event Impact Tracker.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EVENTS_DIR = DATA_DIR / "events"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, EVENTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys (set these as environment variables or replace with your keys)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
TRADING_ECONOMICS_KEY = os.getenv("TRADING_ECONOMICS_KEY", "")

# Default assets for analysis
DEFAULT_ASSETS = {
    "equities": ["SPY", "QQQ", "IWM", "DIA"],
    "fx": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "DX-Y.NYB"],
    "rates": ["^TNX", "^FVX", "TLT", "IEF"],
    "commodities": ["GC=F", "CL=F"],
}

# Analysis timeframes (in minutes)
ANALYSIS_TIMEFRAMES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "2hour": 120,
    "4hour": 240,
}

# Event window settings
EVENT_WINDOW = {
    "pre_event_minutes": 30,
    "post_event_minutes": 120,
}

# Chart settings
CHART_SETTINGS = {
    "template": "plotly_dark",
    "default_height": 500,
    "colors": {
        "positive": "#26a69a",
        "negative": "#ef5350",
        "neutral": "#78909c",
        "event_line": "#ff9800",
    },
}

# Cache settings
CACHE_SETTINGS = {
    "market_data_ttl_minutes": 5,  # How long to cache market data
    "event_data_ttl_hours": 24,    # How long to cache event data
}

# Logging settings
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
