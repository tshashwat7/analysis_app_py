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

# Horizon-aware MA config for crossovers (Golden Cross / Death Cross)
# Matches definitions in indicators.py for consistent structural analysis
HORIZON_MA_CONFIG = {
    "intraday":    {"mid_len": 50, "slow_len": 200, "type": "EMA"},
    "short_term":  {"mid_len": 50, "slow_len": 200, "type": "EMA"},
    "long_term":   {"mid_len": 40, "slow_len": 50,  "type": "WMA"},
    "multibagger": {"mid_len": 12, "slow_len": 24,  "type": "SMA"},
}

# Wall-clock expiry days for pattern state cleanup (W46/Phase 4)
# Intraday: 1 day, Short: 7, Long: 30, Multibagger: 90
HORIZON_EXPIRY_DAYS = {
    "intraday":    1,
    "short_term":  7,
    "long_term":   30,
    "multibagger": 90,
}
