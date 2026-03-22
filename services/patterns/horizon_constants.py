# services/patterns/horizon_constants.py
"""
Centralized horizon constants for the Pattern Library.
Decouples seconds-per-candle (for wall-clock duration) 
from bar counts (for pattern search windows).
"""

# Seconds per candle — used by pattern_state_manager and trade_enhancer
HORIZON_WINDOWS_SECONDS = {
    "intraday":    15 * 60,    # 900 seconds
    "short_term":  86400,      # 1 day
    "long_term":   604800,     # 7 days (weekly)
    "multibagger": 2592000,    # 30 days (monthly approximation)
}

# Bar counts for pattern search windows — used by detectors for lookback
# Format: {"window": search_range, "min_history": safety_guard}
HORIZON_WINDOWS_BARS = {
    "intraday":    {"window": 30,  "min_history": 50},
    "short_term":  {"window": 60,  "min_history": 90},
    "long_term":   {"window": 120, "min_history": 150},
    "multibagger": {"window": 120, "min_history": 150},
}
