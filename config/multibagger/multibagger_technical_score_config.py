# config/multibagger/multibagger_technical_score_config.py
"""
Multibagger Technical Score Configuration
==========================================
MB-specific metric inclusion, weights, penalties, and bonuses.

These constants are consumed ONLY by mb_compute_technical_score()
in multibagger_evaluator.py. MBQueryOptimizedExtractor.get_technical_score()
routes there instead of the main pipeline's compute_technical_score().

KEY DESIGN DECISIONS:
- Volatility weight = 0.00 — noise for 5-year holds, not signal
- momentum category = relStrengthNifty only — outperformance vs benchmark
  is the only momentum that matters for a multi-year hold thesis
- trendStrength carries 45% of trend weight — MA stacking is everything
- MB_COMPOSITE_SCORING_CONFIG is NOT defined here — trendStrength is a
  pass-through composite already computed in indicators.py _avg_scores()
- METRIC SUFFIX CONVENTION: Metrics appearing in dual categories (e.g. relStrengthNifty)
  use the "_momentum" suffix in MB_TECH_METRIC_WEIGHTS to allow independent weighting.
"""

# ============================================================================
# METRIC INCLUSION
# ============================================================================

MB_TECH_HORIZON_METRIC_INCLUSION = {
    "multibagger": {
        "trend": [
            "trendStrength",           # Composite — MMA stacking + ADX
            "maTrendSignal",           # MMA alignment signal
            "priceVsPrimaryTrendPct",  # Distance from primary trend
            "relStrengthNifty",        # Outperformance vs Nifty
        ],
        "momentum": [
            "relStrengthNifty",        # Only benchmark outperformance matters
        ],
        "volatility": ["volatilityQuality"],
        "volume": [
            "rvol",                    # Accumulation detection only
        ],
        "structure": [
            "position52w",             # 52W high breakout context
        ],
        "exclude": [
            "momentumStrength",
            "rsi", "rsiSlope", "stochK", "macd", "macdCross",
            "vwapBias", "wickRejection", "gapPercent",
            "volSpikeRatio", "volSpikeSignal",
            "ttmSqueeze", "bbWidth", "atrPct",
            "supertrendSignal", "psarTrend", "ichiCloud",
            "adx",  # Not directly scored — used via trendStrength composite
        ],
    }
}

# ============================================================================
# CATEGORY WEIGHTS (must sum to 1.0)
# ============================================================================

MB_HORIZON_TECHNICAL_WEIGHTS = {
    "multibagger": {
        "trend":      0.50,
        "momentum":   0.30,
        "structure":  0.10,
        "volume":     0.05,
        "volatility": 0.05,
    }
}

# ============================================================================
# METRIC WEIGHTS (flat dict — within each category, sums to 1.0 per category)
# ============================================================================

MB_TECH_METRIC_WEIGHTS = {
    "multibagger": {
        # trend (sum = 1.0)
        "trendStrength":          0.45,  # Primary — MA stacking composite
        "maTrendSignal":          0.30,  # MMA alignment direction
        "priceVsPrimaryTrendPct": 0.15,  # Entry timing vs trend
        "relStrengthNifty":       0.10,  # Benchmark outperformance
        # momentum (sum = 1.0)
        # relStrengthNifty appears in both trend and momentum categories.
        # The category loop uses category-specific active_metrics lists,
        # so weighting is applied independently per category. No double-counting
        # in final score because category weights sum to 1.0.
        "relStrengthNifty_momentum": 1.00,  # Aliased key — see mb_compute_technical_score
        # structure (sum = 1.0)
        "position52w": 1.00,
        # volume (sum = 1.0)
        "rvol": 1.00,
        # volatility (sum = 1.0)
        "volatilityQuality": 1.00,
    }
}

# ============================================================================
# PENALTY RULES
# For passthrough metrics (trendStrength), actual = score (0-10).
# For raw metrics (relStrengthNifty), actual = raw value.
# ============================================================================

MB_TECHNICAL_PENALTIES = {
    "multibagger": [
        {
            "metric":    "trendStrength",
            "operator":  "<",
            "threshold": 4.0,
            "penalty":   2.5,
            "is_passthrough": True,  # Use score value, not raw
            "reason":    "Multi-year trend too weak",
        },
        {
            "metric":    "relStrengthNifty",
            "operator":  "<",
            "threshold": 0,
            "is_passthrough": False,  # Use raw value
            "penalty":   1.5,
            "reason":    "Underperforming Nifty",
        },
        {
            "metric":    "position52w",
            "operator":  ">",
            "threshold": 92,
            "is_passthrough": False,
            "penalty":   1.0,
            "reason":    "Overextended — >92% of 52W high",
        },
    ]
}

# ============================================================================
# BONUS RULES (gate-based, uses evaluate_gates)
# ============================================================================

MB_TECH_BONUSES = [
    {
        "gates": {"trendStrength": {"min": 8.0}, "relStrengthNifty": {"min": 15}},
        "bonus":  2.0,
        "reason": "Strong trend + Nifty outperformance",
    },
    {
        "gates": {"rvol": {"min": 1.5}},
        "bonus":  1.2,
        "reason": "Weekly accumulation volume",
    },
    {
        "gates": {"position52w": {"min": 85, "max": 95}},
        "bonus":  1.5,
        "reason": "Near 52W high breakout zone",
    },
    {
        "gates": {"relStrengthNifty": {"min": 20}},
        "bonus":  1.2,
        "reason": "Significant Nifty outperformance",
    },
]

# ============================================================================
# LIQUIDITY PENALTY (weekly minimum — much lower than daily thresholds)
# ============================================================================

MB_LIQUIDITY_PENALTY_RULE = {
    "multibagger": {
        "min_avg_volume":    10000,   # Weekly — far lower than intraday 100k
        "penalty_multiplier": 1.0,
        "reason":            "Small-cap weekly liquidity risk",
    }
}
