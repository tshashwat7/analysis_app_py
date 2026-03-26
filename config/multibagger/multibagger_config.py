# config/multibagger/multibagger_config.py
"""
Multibagger Configuration — Phase 1 Gatekeeper
================================================
Hard rejection filters for the Phase 1 screener.

ARCHITECTURE:
- This file owns gatekeeper rules and output schema ONLY.
- setup_pattern_matrix and strategy_matrix live in MB_MASTER_CONFIG
  so that MBConfigExtractor.extract_matrix_sections() can read them
  from self.master_config instead of a hard file import.
- Phase 2 resolver config lives in multibagger_master_config.py.

TIMEFRAME: Weekly (1wk) — 5x more data points than monthly for
indicator stability. Uses mma(6/12/24) → keys: maFast/maMid/maSlow.
"""

MULTIBAGGER_CONFIG = {

    # =========================================================================
    # PHASE 1: GATEKEEPER (Absolute Rejection Filters)
    # Any None value means the filter is SKIPPED for that metric (lenient
    # fallback for data provider gaps on NSE stocks via yfinance).
    # =========================================================================
    "gatekeeper": {

        "universe": {
            "exclude_sectors":  ["Miscellaneous"],
            "min_listing_days": 365,    # Stock must be listed for at least 1 year
            "min_price":        20.0,   # Filters penny stocks
            "min_market_cap":   500,    # In Crores — ignores micro-cap manipulation
        },

        "fundamentals": {
            # Each entry: {min|max: value}
            # None = skip filter gracefully if data missing (yfinance gap handling)
            "epsGrowth5y":      {"min": 15.0},   # Consistent earnings accelerator
            "roce":             {"min": 15.0},   # Return on capital employed
            "roe":              {"min": 15.0},   # Return on equity
            "deRatio":          {"max": 1.0},    # Low leverage
            "promoterHolding":  {"min": 30.0},   # Management skin in the game
            "piotroskiF":       {"min": 6},      # Financial health baseline
        },

        "technicals": {
            # Applied on weekly timeframe.
            # Uses maFast (MMA6w), maMid (MMA12w), maSlow (MMA24w) from indicators.py
            "trend_alignment":     {"required": True},           # Close > MMA6 > MMA12 > MMA24
            "distance_from_high":  {"max_drawdown_pct": 30.0},  # Within 30% of 52W High
        },
    },

    # =========================================================================
    # OUTPUT SCHEMA — conviction tier thresholds for Phase 2 evaluator output
    # =========================================================================
    "output_schema": {
        "conviction_tier": {
            "HIGH":   {"score_min": 8.5, "confidence_min": 75},
            "MEDIUM": {"score_min": 7.5, "confidence_min": 65},
            "LOW":    {"score_min": 6.5, "confidence_min": 60},
            "WATCH":  {"score_min": 0,   "confidence_min": 0},  # Explicit fallback state
        }
    },
}
