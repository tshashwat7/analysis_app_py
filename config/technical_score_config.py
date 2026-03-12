# config/technical_score_config.py (SIMPLIFIED v3.0)
"""
Technical Score Aggregation Config
Version: 3.0 - Score-First Design

PHILOSOPHY:
- Indicators ALREADY have scores (0-10)
- No text→numeric mapping needed
- Direct score extraction
- Config defines WEIGHTS + CATEGORIES only
"""

# ==============================================================================
# METRIC REGISTRY (Global metadata)
# ==============================================================================

from typing import Any, Dict, Tuple
import logging
logger = logging.getLogger(__name__)


METRIC_REGISTRY = {
    # ===========================
    # MOMENTUM METRICS
    # ===========================
    "rsi": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 30,
            "max_val": 70,
            "score_at_min": 10,  # Oversold = bullish
            "score_at_max": 0    # Overbought = bearish
        },
        "description": "RSI linear scoring (30=10, 70=0)"
    },

    "rsislope": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -2.0,
            "max_val": 2.0,
            "score_at_min": 0,   # Falling momentum = bearish
            "score_at_max": 10   # Rising momentum = bullish
        },
        "description": "RSI slope momentum indicator"
    },

    "macdCross": {
        "type": "text",
        "category": "momentum",
        "scoring_type": "mapping",
        "params": {
            "Bullish": 10,
            "Bearish": 0,
            "Neutral": 5
        },
        "description": "MACD crossover signal"
    },

    "stochCross": {
        "type": "text",
        "category": "momentum",
        "scoring_type": "mapping",
        "params": {
            "Bullish": 10,
            "Bearish": 0,
            "Neutral": 5
        },
        "description": "Stochastic crossover"
    },

    "stochK": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 20,
            "max_val": 80,
            "score_at_min": 10,  # Oversold
            "score_at_max": 0    # Overbought
        },
        "description": "Stochastic K% value"
    },

    "macd": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 0, "score": 10},   # Positive MACD
                {"min": -5, "score": 5},   # Slightly negative
                {"default": 0}             # Very negative
            ]
        },
        "description": "MACD value stepped scoring"
    },

    "macdhistogram": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 0, "score": 10},   # Positive histogram
                {"min": -1, "score": 5},   # Slightly negative
                {"default": 0}             # Very negative
            ]
        },
        "description": "MACD histogram"
    },

    # ===========================
    # TREND METRICS
    # ===========================
    "adx": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 25, "score": 10},  # Strong trend
                {"min": 20, "score": 7},   # Developing
                {"min": 15, "score": 5},   # Weak
                {"default": 0}             # No trend
            ]
        },
        "description": "ADX trend strength"
    },

    "maFastSlope": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -10,
            "max_val": 25,
            "score_at_min": 0,   # Downtrend
            "score_at_max": 10   # Strong uptrend
        },
        "description": "Moving average slope"
    },

    "maTrendSignal": {
        "type": "numeric", 
        "category": "trend", 
        "scoring_type": "mapping",
        "params": {
            1: 10,     # Strong Uptrend
            1.0: 10,   # Float match
            0.5: 7,    # Developing Uptrend (Matches your logic fix)
            0: 5,      # Neutral
            0.0: 5,    # Float match
            -0.5: 3,   # Developing Downtrend
            -1: 0,     # Strong Downtrend
            -1.0: 0    # Float match
        },
        "description": "MA alignment signal (Supports Developing Trend 0.5)"
    },

    "supertrendSignal": {
        "type": "text",
        "category": "trend",
        "scoring_type": "mapping",
        "params": {
            "Bullish": 10,
            "Bearish": 0,
            "Neutral": 5
        },
        "description": "Supertrend signal"
    },

    "maCrossSignal": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "mapping",
        "params": {
            "1": 10,     # Bullish cross
            "0": 5,      # Neutral
            "-1": 0      # Bearish cross
        },
        "description": "MA crossover signal"
    },
    "maSlowSlope": {
        "type": "numeric", "category": "trend", "scoring_type": "linear_range",
        "params": {"min_val": -2, "max_val": 10, "score_at_min": 0, "score_at_max": 10},
        "description": "Slow MA velocity (long-term anchor)"
    },
    "ichiCloud": {
        "type": "text", "category": "trend", "scoring_type": "mapping",
        "params": {
            "Strong Bullish": 10, "Mild Bullish": 8, "Neutral Bullish": 6,
            "Neutral": 5, "Neutral Bearish": 4, "Mild Bearish": 3, "Strong Bearish": 0
        }
    },
   "psarTrend": {
        "type": "text", "category": "trend", "scoring_type": "mapping",
        "params": {"Bullish": 10, "Bearish": 0},
        "description": "Parabolic SAR trend direction"
    },
    "niftyTrendScore": {
        "type": "text", "category": "trend", "scoring_type": "mapping",
        "params": {"Strong Uptrend": 10, "Moderate Uptrend": 7, "Uptrend (Weak)": 5, "Downtrend": 0},
        "description": "Benchmark market regime"
    }, 
    # ===========================
    # VOLATILITY METRICS
    # ===========================
    "atrPct": {
        "type": "numeric",
        "category": "volatility",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 0.5,
            "max_val": 5.0,
            "score_at_min": 10,  # Low volatility = stable
            "score_at_max": 0    # High volatility = risky
        },
        "description": "ATR percentage (lower is better)"
    },

    "bbWidth": {
        "type": "numeric",
        "category": "volatility",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 0.5,
            "max_val": 10.0,
            "score_at_min": 10,  # Squeeze = potential breakout
            "score_at_max": 0    # Wide bands = volatile
        },
        "description": "Bollinger Band width"
    },

    "ttmSqueeze": {
        "type": "text",
        "category": "volatility",
        "scoring_type": "mapping",
        "params": {
            "Squeeze On": 10,
            "Squeeze Off": 5,
            "Off": 5,
            "No Squeeze": 0
        },
        "description": "TTM Squeeze indicator"
    },
    "atrSmaRatio": {
        "type": "numeric", "category": "volatility", "scoring_type": "linear_range",
        "params": {"min_val": 0.01, "max_val": 0.05, "score_at_min": 10, "score_at_max": 2}
    },
    "position52w": {
        "type": "numeric",
        "category": "structure",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 70,       # Below 70% of 52W high = Score 0 (Weak/Bottom)
            "max_val": 100,      # At 52W high = Score 10 (Strong/Breakout zone)
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Price location relative to 52-week high (Bullish near high)"
    },
    # ===========================
    # VOLUME METRICS
    # ===========================
    "rvol": {
        "type": "numeric",
        "category": "volume",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 0.5,
            "max_val": 3.0,
            "score_at_min": 0,   # Low volume
            "score_at_max": 10   # High volume
        },
        "description": "Relative volume"
    },

    "volSpikeRatio": {
        "type": "numeric",
        "category": "volume",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 2.0, "score": 10},  # Strong spike
                {"min": 1.5, "score": 7},   # Moderate spike
                {"min": 1.0, "score": 5},   # Normal
                {"default": 0}              # Low
            ]
        },
        "description": "Volume spike ratio"
    },

    "cmfSignal": {
        "type": "numeric",
        "category": "volume",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -0.2,
            "max_val": 0.2,
            "score_at_min": 0,   # Negative money flow
            "score_at_max": 10   # Positive money flow
        },
        "description": "Chaikin Money Flow"
    },

    "obvDiv": {
        "type": "text",
        "category": "volume",
        "scoring_type": "mapping",
        "params": {
            "Confirming": 10,
            "Diverging": 0,
            "Neutral": 5
        },
        "description": "OBV divergence"
    },
    "vpt": {
        "type": "text", "category": "volume", "scoring_type": "mapping",
        "params": {"Accumulation": 10, "Distribution": 0, "Neutral": 5}
    },
    "volSpikeSignal": { # Match the camelCase in indicators.py
        "type": "text", "category": "volume", "scoring_type": "mapping",
        "params": {"Strong Spike": 10, "Moderate Spike": 7, "Normal": 5, "Low": 0}
    },
    # ===========================
    # STRUCTURE METRICS
    # ===========================
    "wickRejection": {
        "type": "numeric",
        "category": "structure",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 0.5,
            "max_val": 3.0,
            "score_at_min": 10,  # Solid close (good)
            "score_at_max": 0    # Severe rejection (bad)
        },
        "description": "Wick to body ratio"
    },

    "vwapBias": {
        "type": "text",
        "category": "structure",
        "scoring_type": "mapping",
        "params": {
            "Bullish": 10,
            "Bearish": 0,
            "Neutral": 5
        },
        "description": "VWAP bias"
    },

    "priceAction": {
        "type": "numeric",
        "category": "structure",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 20,
            "max_val": 80,
            "score_at_min": 0,   # Weak close
            "score_at_max": 10   # Strong close
        },
        "description": "Price action strength"
    },

    "gapPercent": {
        "type": "numeric",
        "category": "structure",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 2.0, "score": 8},   # Strong gap up
                {"min": 0.5, "score": 6},   # Moderate gap
                {"min": -0.5, "score": 5},  # Flat
                {"min": -2.0, "score": 3},  # Moderate gap down
                {"default": 0}              # Strong gap down
            ]
        },
        "description": "Gap percentage"
    },

    # ===========================
    # COMPOSITE METRICS (Pass-through)
    # ===========================
    "trendStrength": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "passthrough",
        "params": {
            "min_val": 0,
            "max_val": 10,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Composite trend strength (pass-through)"
    },

    "momentumStrength": {
        "type": "numeric",
        "category": "momentum",
        "scoring_type": "passthrough",
        "params": {
            "min_val": 0,
            "max_val": 10,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Composite momentum strength (pass-through)"
    },

    "volatilityQuality": {
        "type": "numeric",
        "category": "volatility",
        "scoring_type": "passthrough",
        "params": {
            "min_val": 0,
            "max_val": 10,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Composite volatility quality (pass-through)"
    },

    # ===========================
    # DIRECTIONAL INDICATORS
    # ===========================
    "diPlus": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "dynamic_crossover",
        "crossover_metric": "diMinus",
        "params": {
            "score_if_higher": 7,   # DI+ > DI- = bullish
            "score_if_lower": 3,    # DI+ < DI- = bearish
            "score_if_equal": 5     # Neutral
        },
        "description": "DI+ with crossover logic"
    },

    "diMinus": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "dynamic_crossover",
        "crossover_metric": "diPlus",
        "params": {
            "score_if_higher": 3,   # DI- > DI+ = bearish
            "score_if_lower": 7,    # DI- < DI+ = bullish
            "score_if_equal": 5     # Neutral
        },
        "description": "DI- with crossover logic"
    },

    # ===========================
    # PRICE POSITION METRICS
    # ===========================
    "priceVsPrimaryTrendPct": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -10,
            "max_val": 10,
            "score_at_min": 0,   # Below trend
            "score_at_max": 10   # Above trend
        },
        "description": "Price vs primary MA percentage"
    },

    "relStrengthNifty": {
        "type": "numeric",
        "category": "trend",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -10,
            "max_val": 20,
            "score_at_min": 0,   # Underperforming
            "score_at_max": 10   # Outperforming
        },
        "description": "Relative strength vs NIFTY"
    }
}

# ==============================================================================
# HORIZON METRIC PARTICIPATION (What metrics apply per horizon)
# ==============================================================================

HORIZON_METRIC_INCLUSION = {
    "intraday": {
        "trend": ["maFastSlope", "maTrendSignal", "supertrendSignal"],  # ← No trendStrength composite
        "momentum": ["momentumStrength", "rsi", "rsislope", "macd", "stochK", "stochCross"],  # ← Fast signals
        "volatility": ["volatilityQuality", "atrPct", "bbWidth", "ttmSqueeze"],
        "volume": ["rvol", "volSpikeRatio", "volSpikeSignal", "cmfSignal"],
        "structure": ["vwapBias", "priceAction", "wickRejection", "gapPercent"],
        
        # ✅ EXPLICIT EXCLUSIONS
        "exclude": [
            "relStrengthNifty",  # Macro irrelevant for scalping
            "position52w",       # Long-term context
            "obvDiv",           # Too slow
            "trendStrength"     # Use raw slope instead
        ]
    },
    
    "short_term": {
        "trend": ["trendStrength", "maTrendSignal", "maFastSlope", "supertrendSignal", "priceVsPrimaryTrendPct"],
        "momentum": ["momentumStrength", "rsi", "rsislope", "macd", "macdCross", "stochK", "stochCross"],
        "volatility": ["volatilityQuality", "atrPct", "bbWidth"],
        "volume": ["rvol", "volSpikeRatio", "obvDiv", "cmfSignal"],
        "structure": ["position52w", "priceAction", "wickRejection", "gapPercent"],
        
        "exclude": [
            "vwapBias"  # Intraday-specific
        ]
    },
    
    "long_term": {
        "trend": ["trendStrength", "maTrendSignal", "adx", "priceVsPrimaryTrendPct", "relStrengthNifty"],
        "momentum": ["momentumStrength", "rsi", "macd"],  # ← relStrengthNifty is already in trend
        "volatility": ["volatilityQuality"],  # ← Only composite, not raw metrics
        "volume": ["rvol", "obvDiv"],
        "structure": ["position52w", "priceAction"],
        
        "exclude": [
            "rsislope",      # Daily noise
            "stochK",        # Too fast
            "vwapBias",      # Intraday-only
            "volSpikeRatio", # Short-term signal
            "wickRejection", # Intraday pattern
            "gapPercent"     # Short-term event
        ]
    },
    
    "multibagger": {
        "trend": ["trendStrength", "maTrendSignal", "priceVsPrimaryTrendPct", "relStrengthNifty"],
        "momentum": ["relStrengthNifty"],  # ← ONLY benchmark outperformance matters
        "volatility": [],  # ← Volatility is NOISE for 5-year holds
        "volume": ["rvol"],  # ← Only for accumulation detection
        "structure": ["position52w"],
        
        "exclude": [
            "volatilityQuality",  # Irrelevant for multi-year
            "momentumStrength",   # Short-term oscillations
            "rsi", "rsislope", "stochK", "macd",  # All noise
            "vwapBias", "wickRejection", "gapPercent",  # Intraday noise
            "volSpikeRatio", "volSpikeSignal",  # Short-term
            "ttmSqueeze", "bbWidth", "atrPct"  # Volatility noise
        ]
    }
}

# ==============================================================================
# HORIZON CATEGORY WEIGHTS (How to aggregate categories)
# ==============================================================================

HORIZON_TECHNICAL_WEIGHTS = {
    "intraday": {
        "momentum": 0.35,
        "trend": 0.20,
        "volume": 0.20,
        "structure": 0.15,
        "volatility": 0.10
    },
    "short_term": {
        "momentum": 0.30,
        "trend": 0.30,
        "volume": 0.15,
        "volatility": 0.15,
        "structure": 0.10
    },
    "long_term": {
        "trend": 0.45,
        "momentum": 0.25,
        "volume": 0.15,
        "structure": 0.10,
        "volatility": 0.05  # ← Very low weight
    },
    "multibagger": {
        "trend": 0.50,
        "momentum": 0.30,  # ← Just relStrengthNifty
        "structure": 0.15,
        "volume": 0.05,
        "volatility": 0.00  # ← ZERO weight
    }
}

# ==============================================================================
# METRIC-LEVEL WEIGHTS (Within each category, per horizon)
# ==============================================================================

METRIC_WEIGHTS = {
    # ==========================================================================
    # INTRADAY: Fast signals, price action, momentum
    # ==========================================================================
    "intraday": {
        "trend": {
            "maFastSlope": 0.40,          # Most important - immediate trend direction
            "maTrendSignal": 0.35,      # MA alignment strength
            "supertrendSignal": 0.25,     # Quick trend filter
        },
        "momentum": {
            "momentumStrength": 0.30,     # Composite momentum
            "rsi": 0.20,                  # Overbought/oversold
            "rsislope": 0.20,             # Momentum acceleration
            "macd": 0.10,                 # Trend momentum
            "macdCross": 0.08,            # Entry signal
            "stochK": 0.07,               # Fast oscillator
            "stochCross": 0.05,          # Stochastic signal
        },
        "volatility": {
            "volatilityQuality": 0.50,    # Composite volatility score
            "atrPct": 0.25,               # Movement potential
            "bbWidth": 0.15,              # Squeeze detection
            "ttmSqueeze": 0.10,           # Pre-breakout compression
        },
        "volume": {
            "rvol": 0.35,                 # Relative volume
            "volSpikeRatio": 0.30,        # Volume surge
            "volSpikeSignal": 0.20,     # Volume signal classification
            "cmfSignal": 0.15,            # Money flow
        },
        "structure": {
            "vwapBias": 0.40,             # Critical for intraday - institutional anchor
            "priceAction": 0.25,          # Candle position
            "wickRejection": 0.20,        # Support/resistance test
            "gapPercent": 0.15,           # Gap moves
        }
    },
    
    # ==========================================================================
    # SHORT-TERM: Balance momentum + trend, pattern recognition
    # ==========================================================================
    "short_term": {
        "trend": {
            "trendStrength": 0.30,        # Composite trend score
            "maTrendSignal": 0.25,      # MA alignment
            "maFastSlope": 0.15,          # Trend slope
            "supertrendSignal": 0.15,     # Trend filter
            "priceVsPrimaryTrendPct": 0.10,  # Pullback depth
            "adx": 0.05,                  # Trend strength confirmation
        },
        "momentum": {
            "momentumStrength": 0.35,     # Composite momentum
            "rsi": 0.20,                  # Momentum level
            "rsislope": 0.15,             # Momentum change
            "macd": 0.12,                 # Trend momentum
            "macdCross": 0.10,            # Entry signal
            "stochK": 0.05,               # Oscillator
            "stochCross": 0.03,          # Stochastic signal
        },
        "volatility": {
            "volatilityQuality": 0.50,    # Composite volatility
            "atrPct": 0.25,               # Volatility level
            "bbWidth": 0.15,              # Bollinger squeeze
            "bbpercentb": 0.10,           # BB position
        },
        "volume": {
            "rvol": 0.35,                 # Relative volume
            "volSpikeRatio": 0.25,        # Volume surge
            "obvDiv": 0.20,               # Volume divergence
            "cmfSignal": 0.20,            # Money flow
        },
        "structure": {
            "position52w": 0.35,          # 52-week position (breakout proximity)
            "priceAction": 0.25,          # Candle analysis
            "wickRejection": 0.20,        # Support/resistance
            "gapPercent": 0.20,           # Gap moves
        }
    },
    
    # ==========================================================================
    # LONG-TERM: Trend dominance, relative strength, minimal noise
    # ==========================================================================
    "long_term": {
        "trend": {
            "trendStrength": 0.35,        # Composite trend (highest priority)
            "maTrendSignal": 0.25,      # Long-term MA alignment
            "adx": 0.20,                  # Sustained trend strength
            "priceVsPrimaryTrendPct": 0.12,  # Pullback opportunity
            "relStrengthNifty": 0.08,     # Outperformance vs benchmark
        },
        "momentum": {
            "momentumStrength": 0.40,     # Composite momentum
            "rsi": 0.25,                  # Momentum level
            "macd": 0.20,                 # Trend momentum
            "relStrengthNifty": 0.15,     # Benchmark comparison
        },
        "volatility": {
            "volatilityQuality": 1.00,    # Only composite - raw vol is noise
        },
        "volume": {
            "rvol": 0.60,                 # Relative volume
            "obvDiv": 0.40,               # Volume trend divergence
        },
        "structure": {
            "position52w": 0.60,          # Multi-year breakout level
            "priceAction": 0.40,          # Weekly candle patterns
        }
    },
    
    # ==========================================================================
    # MULTIBAGGER: Pure trend + benchmark outperformance, zero noise
    # ==========================================================================
    "multibagger": {
        "trend": {
            "trendStrength": 0.45,        # Multi-year trend (supreme priority)
            "maTrendSignal": 0.30,      # Long-term MA alignment
            "priceVsPrimaryTrendPct": 0.15,  # Entry timing
            "relStrengthNifty": 0.10,     # Must outperform market
        },
        "momentum": {
            "relStrengthNifty": 1.00,     # ONLY benchmark outperformance matters
        },
        "volatility": {},
        "volume": {
            "rvol": 1.00,                 # Accumulation detection only
        },
        "structure": {
            "position52w": 1.00,          # Multi-year breakout context
        }
    }
}


# ==============================================================================
# METRIC CATEGORIES (Same structure, but simpler usage)
# ==============================================================================

TECHNICAL_METRIC_CATEGORIES = {
    "trend": [
        "trendStrength",              # Composite (0-10)
        "maTrendSignal",            # Numeric (1/-1/0) → has score
        "adx",                        # Numeric (0-100) → has score
        "supertrendSignal",           # Text but has score!
        "priceVsPrimaryTrendPct", # % → has score
        "ichiCloud",                  # Text but has score!
        "psarTrend"                   # Text but has score!
    ],
    
    "momentum": [
        "momentumStrength",      # Composite (0-10)
        "rsi",                   # Has score
        "macd",                  # Has score
        "macdCross",             # Text but has score
        "stochK",                # Has score
        "stochCross",           # Text but has score
        "relStrengthNifty"       # Has score
    ],
    
    "volatility": [
        "volatilityQuality",     # Composite (0-10)
        "atrPct",                # Has score
        "bbWidth",               # Has score
        "bbpercentb"             # Has score
    ],
    
    "volume": [
        "rvol",                  # Has score
        "volSpikeRatio",         # Has score
        "volSpikeSignal",      # Text but has score
        "obvDiv",                # Text but has score
        "cmfSignal",             # Has score (numeric!)
    ],
    
    "structure": [
        "vwapBias",              # Text but has score
        "position52w",           # Has score
        "wickRejection",         # Has score
        "priceAction",           # Has score
        "gapPercent"             # Has score
    ],
    
    "liquidity": [
        "avg_volume_30Days"      # For penalties only
    ]
}

# ==============================================================================
# PENALTIES (Numeric thresholds only - no text checks needed!)
# ==============================================================================

TECHNICAL_PENALTIES = {
    "intraday": [
        {
            "metric": "rvol",
            "operator": "<",
            "threshold": 0.8,
            "penalty": 0.15,
            "reason": "Low intraday activity"
        },
        {
            "metric": "atrPct",
            "operator": "<",
            "threshold": 0.5,
            "penalty": 0.10,
            "reason": "Insufficient volatility"
        },
        {
            "metric": "wickRejection",
            "operator": ">",
            "threshold": 2.5,
            "penalty": 0.20,
            "reason": "Strong wick rejection (institutional sell-off)"
        }

    ],
    
    "short_term": [
        {
            "metric": "trendStrength",
            "operator": "<",
            "threshold": 3.5,
            "penalty": 0.20,
            "reason": "Weak trend for swing trade"
        },
        {
            "metric": "rvol",
            "operator": "<",
            "threshold": 0.7,
            "penalty": 0.10,
            "reason": "Low volume confirmation"
        },
        {
            "metric": "relStrengthNifty",
            "operator": "<",
            "threshold": -5,
            "penalty": 0.15,
            "reason": "Underperforming benchmark"
        },
        {
            "metric": "position52w",
            "operator": "<",
            "threshold": 30,
            "penalty": 0.10,
            "reason": "Far from breakout zone"
        },
        {
            "metric": "wickRejection",
            "operator": ">",
            "threshold": 2.5,
            "penalty": 0.20,
            "reason": "Strong wick rejection (institutional sell-off)"
        },
        {
            "metric": "priceVsPrimaryTrendPct",
            "operator": ">",
            "threshold": 15.0,
            "penalty": 0.15,
            "reason": "Overextended from primary trend"
        }

    ],
    
    "long_term": [
        {
            "metric": "trendStrength",
            "operator": "<",
            "threshold": 4.0,
            "penalty": 0.25,
            "reason": "Insufficient long-term trend"
        },
        {
            "metric": "adx",
            "operator": "<",
            "threshold": 18,
            "penalty": 0.15,
            "reason": "Weak trend confirmation"
        },
        {
            "metric": "relStrengthNifty",
            "operator": "<",
            "threshold": 0,
            "penalty": 0.20,
            "reason": "Not outperforming benchmark"
        },
        {
            "metric": "priceVsPrimaryTrendPct",
            "operator": ">",
            "threshold": 15.0,
            "penalty": 0.15,
            "reason": "Overextended from primary trend"
        },
        {
            "metric": "priceVsPrimaryTrendPct",
            "operator": "<",
            "threshold": -5.0,
            "penalty": 0.30,
            "reason": "Below long-term trend support"
        }
    ],
    
    "multibagger": [
        {
            "metric": "trendStrength",
            "operator": "<",
            "threshold": 5.0,
            "penalty": 0.30,
            "reason": "Multi-year trend too weak"
        },
        {
            "metric": "relStrengthNifty",
            "operator": "<",
            "threshold": 10,
            "penalty": 0.25,
            "reason": "Insufficient outperformance"
        },
        {
            "metric": "position52w",
            "operator": ">",
            "threshold": 90,
            "penalty": 0.15,
            "reason": "Overextended"
        }
    ]
}

# ==============================================================================
# BONUSES (Simple condition evaluator)
# ==============================================================================

TECHNICAL_BONUSES = [
    {
        "gates": {
            "trendStrength": {"min": 8.0},
            "momentumStrength": {"min": 8.0}
        },
        "bonus": 0.20,
        "reason": "Exceptional trend + momentum synergy"
    },
    {
        "gates": {
            "rvol": {"min": 2.5},
            "volSpikeRatio": {"min": 2.0}
        },
        "bonus": 0.15,
        "reason": "Strong volume confirmation"
    },
    {
        "gates": {
            "volatilityQuality": {"min": 7.0}
        },
        "bonus": 0.10,
        "reason": "High-quality volatility"
    },
    {
        "gates": {
            "position52w": {"min": 85, "max": 95}
        },
        "bonus": 0.15,
        "reason": "Near 52W high (breakout zone)"
    },
    {
        "gates": {
            "relStrengthNifty": {"min": 15}
        },
        "bonus": 0.12,
        "reason": "Strong outperformance vs Nifty"
    },
    {
        "gates": {
            "ttmSqueeze": {"equals": "Squeeze Off"},
            "rvol": {"min": 2.0}
        },
        "bonus": 0.15,
        "reason": "Squeeze release with volume expansion"
    },
    {
        "gates": {
            "cmfSignal": {"min": 0.15}
        },
        "bonus": 0.12,
        "reason": "Institutional accumulation confirmed"
    },
    {
        "gates": {
            "position52w": {"min": 90},
            "rvol": {"min": 1.5}
        },
        "bonus": 0.15,
        "reason": "Approaching 52-week breakout with volume"
    }
]

# ==============================================================================
# LIQUIDITY PENALTIES (Separate from category scoring)
# ==============================================================================

LIQUIDITY_PENALTY_RULES = {
    "intraday": {
        "min_avg_volume": 100000,
        "penalty_multiplier": 0.30,
        "reason": "Insufficient intraday liquidity"
    },
    
    "short_term": {
        "min_avg_volume": 50000,
        "penalty_multiplier": 0.20,
        "reason": "Low liquidity for swing trading"
    },
    
    "long_term": {
        "min_avg_volume": 25000,
        "penalty_multiplier": 0.15,
        "reason": "Poor long-term liquidity"
    },
    
    "multibagger": {
        "min_avg_volume": 10000,
        "penalty_multiplier": 0.10,
        "reason": "Small-cap liquidity risk"
    }
}

# ==============================================================================
# COMPOSITE SCORING CONFIGURATIONS
# ==============================================================================

COMPOSITE_SCORING_CONFIG = {
    "intraday": {
        "trendStrength": {
            "metrics": {
                "maFastSlope": {
                    "weight": 0.40,  # Velocity of immediate trend
                    "thresholds": [
                        {"min": 20, "score": 10}, {"min": 5, "score": 7},
                        {"min": 0, "score": 5}, {"default": 2}
                    ]
                },
                "maTrendSignal": {
                    "weight": 0.30,  # Alignment of 20/50/200 MAs
                    "mapping": {1: 10, 1.0: 10, 0.5: 7, 0: 5, 0.0: 5, -0.5: 3, -1: 0, -1.0: 0}
                },
                "diSpread": {        # Trend conviction (ADX components)
                    "weight": 0.20,
                    "thresholds": [
                        {"min": 15, "score": 10}, {"min": 10, "score": 7},
                        {"min": 5, "score": 5}, {"default": 2}
                    ]
                },
                "supertrendSignal": {
                    "weight": 0.10,
                    "mapping": {"Bullish": 10, "Neutral": 5, "Bearish": 0}
                }
            }
        },
        "momentumStrength": {
            "metrics": {
                "rsi": {
                    "weight": 0.25,
                    "thresholds": [
                        {"min": 60, "score": 10}, {"min": 50, "score": 7},
                        {"min": 40, "score": 4}, {"default": 2}
                    ]
                },
                "rsislope": {
                    "weight": 0.25,
                    "thresholds": [{"min": 1.0, "score": 10}, {"min": 0, "score": 5}, {"default": 2}]
                },
                "macdhistogram": {   # CORRECTED: Uses Histogram for lead signal
                    "weight": 0.30,
                    "thresholds": [
                        {"min": 0.5, "score": 10}, {"min": 0, "score": 7},
                        {"min": -0.5, "score": 4}, {"default": 0}
                    ]
                },
                "stochCross": {     # CORRECTED: Uses Signal instead of K value
                    "weight": 0.20,
                    "mapping": {"Bullish": 10, "Neutral": 5, "Bearish": 0}
                }
            }
        },
        "volatilityQuality": {
            "metrics": {
                "atrPct": {
                    "weight": 0.30,
                    "thresholds": [{"max": 1.5, "score": 10}, {"max": 5.0, "score": 5}, {"default": 2}]
                },
                "bbWidth": {
                    "weight": 0.25,
                    "thresholds": [{"max": 2.0, "score": 10}, {"max": 10.0, "score": 4}, {"default": 0}]
                },
                "trueRangeConsistency": { # NEW: Detects erratic price action
                    "weight": 0.20,
                    "thresholds": [{"max": 0.5, "score": 10}, {"max": 1.5, "score": 4}, {"default": 0}]
                },
                "hvTrend": {              # NEW: Rising volatility = Danger
                    "weight": 0.15,
                    "mapping": {"declining": 10, "stable": 7, "rising": 2}
                },
                "atrSmaRatio": {          # Squeeze proximity
                    "weight": 0.10,
                    "thresholds": [{"max": 0.02, "score": 10}, {"max": 0.05, "score": 4}]
                }
            }
        }
    },
    "short_term": {
        "trendStrength": {
            "metrics": {
                "adx": {
                    "weight": 0.35,  # ADX becomes critical for swing conviction
                    "thresholds": [{"min": 25, "score": 10}, {"min": 20, "score": 8}, {"min": 15, "score": 5}, {"default": 2}]
                },
                "maTrendSignal": {
                    "weight": 0.25,
                    "mapping": {1: 10, 1.0: 10, 0.5: 7, 0: 5, 0.0: 5, -0.5: 3, -1: 0, -1.0: 0}
                },
                "diSpread": {        # Distance between buyers and sellers
                    "weight": 0.25,
                    "thresholds": [{"min": 15, "score": 10}, {"min": 10, "score": 7}, {"min": 5, "score": 5}, {"default": 2}]
                },
                "supertrendSignal": {
                    "weight": 0.15,
                    "mapping": {"Bullish": 10, "Neutral": 5, "Bearish": 0}
                }
            }
        },
        "momentumStrength": {
            "metrics": {
                "rsi": {
                    "weight": 0.25,
                    "thresholds": [{"min": 60, "score": 10}, {"min": 50, "score": 7}, {"min": 40, "score": 4}, {"default": 2}]
                },
                "rsislope": {
                    "weight": 0.20,
                    "thresholds": [{"min": 1.0, "score": 10}, {"min": 0, "score": 5}, {"default": 2}]
                },
                "macdhistogram": {
                    "weight": 0.30,
                    "thresholds": [{"min": 0.5, "score": 10}, {"min": 0, "score": 7}, {"min": -0.5, "score": 4}, {"default": 0}]
                },
                "stochCross": {
                    "weight": 0.25,  # Crosses are high-probability entry signals for swings
                    "mapping": {"Bullish": 10, "Neutral": 5, "Bearish": 0}
                }
            }
        },
        "volatilityQuality": {
            "metrics": {
                "atrPct": {
                    "weight": 0.25,
                    "thresholds": [{"max": 2.5, "score": 10}, {"max": 4.5, "score": 6}, {"default": 2}]
                },
                "bbWidth": {
                    "weight": 0.20,
                    "thresholds": [{"max": 3.0, "score": 10}, {"max": 8.0, "score": 5}, {"default": 0}]
                },
                "trueRangeConsistency": {
                    "weight": 0.20,
                    "thresholds": [{"max": 0.6, "score": 10}, {"max": 1.2, "score": 5}, {"default": 0}]
                },
                "hvTrend": {
                    "weight": 0.20,
                    "mapping": {"declining": 10, "stable": 7, "rising": 3}
                },
                "atrSmaRatio": {
                    "weight": 0.15,
                    "thresholds": [{"max": 0.025, "score": 10}, {"max": 0.045, "score": 5}, {"default": 0}]
                }
            }
        }
    },

    "long_term": {
        "trendStrength": {
            "metrics": {
                "adx": {
                    "weight": 0.40,  # Supreme priority for long-term trend sustainability
                    "thresholds": [{"min": 25, "score": 10}, {"min": 18, "score": 6}, {"default": 2}]
                },
                "maTrendSignal": {
                    "weight": 0.30,
                    "mapping": {1: 10, 1.0: 10, 0.5: 5, 0: 0, 0.0: 0, -0.5: 0, -1: 0, -1.0: 0} # Strict on LT
                },
                "diSpread": {
                    "weight": 0.30,
                    "thresholds": [{"min": 20, "score": 10}, {"min": 12, "score": 7}, {"min": 5, "score": 4}, {"default": 0}]
                }
            }
        },
        "momentumStrength": {
            "metrics": {
                "rsi": {
                    "weight": 0.35,
                    "thresholds": [{"min": 60, "score": 10}, {"min": 50, "score": 7}, {"default": 2}]
                },
                "macdhistogram": {
                    "weight": 0.35,
                    "thresholds": [{"min": 0.5, "score": 10}, {"min": 0, "score": 6}, {"default": 0}]
                },
                "relStrengthNifty": { # Added for Benchmark Outperformance
                    "weight": 0.30,
                    "thresholds": [{"min": 10, "score": 10}, {"min": 0, "score": 6}, {"default": 0}]
                }
            }
        },
        "volatilityQuality": {
            "metrics": {
                "atrPct": {
                    "weight": 0.40,
                    "thresholds": [{"max": 5.0, "score": 10}, {"max": 10.0, "score": 5}, {"default": 0}]
                },
                "trueRangeConsistency": {
                    "weight": 0.30,
                    "thresholds": [{"max": 0.8, "score": 10}, {"max": 1.5, "score": 5}, {"default": 0}]
                },
                "hvTrend": {
                    "weight": 0.30,
                    "mapping": {"declining": 10, "stable": 7, "rising": 2}
                }
            }
        }
    },

    "multibagger": {
        "trendStrength": {
            "metrics": {
                "maTrendSignal": {
                    "weight": 0.50, # Must be in a clear structural uptrend
                    "mapping": {1: 10, 1.0: 10, 0.5: 4, 0: 0, 0.0: 0, -0.5: 0, -1: 0, -1.0: 0}
                },
                "maSlowSlope": {  # Velocity of the 12-month MA
                    "weight": 0.30,
                    "thresholds": [{"min": 5, "score": 10}, {"min": 2, "score": 7}, {"default": 0}]
                },
                "diSpread": {
                    "weight": 0.20,
                    "thresholds": [{"min": 25, "score": 10}, {"min": 15, "score": 6}, {"default": 0}]
                }
            }
        },
        "momentumStrength": {
            "metrics": {
                "relStrengthNifty": { # The only momentum metric that matters for multi-year holds
                    "weight": 1.0,
                    "thresholds": [{"min": 20, "score": 10}, {"min": 10, "score": 7}, {"min": 0, "score": 4}, {"default": 0}]
                }
            }
        }
        # Volatility is noise for Multibagger (5+ year holds)
    }
}

# ==============================================================================
# COMPOSITE SCORING HELPERS
# ==============================================================================

def _apply_composite_scoring_rules(value: Any, rules: dict, metric_name: str = None) -> float:
    """
    Apply scoring rules from composite config.
    
    Supports:
    - thresholds: List of {"min": X, "score": Y} or {"max": X, "score": Y}
    - mapping: Dict of {value: score}
    
    Returns:
        Score between 0-10
    """
    if value is None:
        return None  # Changed from 0.0 to None
    
    # Handle text mapping
    if "mapping" in rules:
        str_value = str(value).strip()
        score = rules["mapping"].get(str_value)
        
        if score is None:
            # Try case-insensitive
            for key, val in rules["mapping"].items():
                if str(key).lower() == str_value.lower():
                    return float(val)
            
            logger.debug(f"No mapping for {metric_name}='{str_value}', using neutral")
            return 5.0
        
        return float(score)
    
    # Handle threshold scoring
    if "thresholds" in rules:
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert {metric_name} value '{value}' to float")
            return 5.0
        
        for entry in rules["thresholds"]:
            # Check "min" threshold
            if "min" in entry and val_float >= entry["min"]:
                return float(entry["score"])
            
            # Check "max" threshold
            if "max" in entry and val_float <= entry["max"]:
                return float(entry["score"])
        
        # Default score (stored inside the list of thresholds)
        default_entry = next((e for e in rules["thresholds"] if "default" in e), None)
        default = default_entry["default"] if default_entry else 0
        return float(default)
    
    logger.warning(f"No valid scoring rules for {metric_name}")
    return 5.0


def compute_single_composite(
    composite_name: str,
    indicators: dict,
    horizon: str
) -> dict:
    """
    Calculate a single composite score (e.g., trendStrength).
    
    Args:
        composite_name: "trendStrength", "momentumStrength", or "volatilityQuality"
        indicators: Full indicators dict
        horizon: "intraday", "short_term", "long_term", "multibagger"
    
    Returns:
        {
            "value": 7.85,
            "score": 7.85,
            "desc": "Trend Composite",
            "alias": "Trend Strength",
            "source": "composite",
            "breakdown": {...}
        }
    """
    config = COMPOSITE_SCORING_CONFIG.get(horizon, {}).get(composite_name)
    
    if not config:
        logger.warning(f"No composite config for {composite_name} in {horizon}")
        return {
            "value": 0.0,
            "score": 0.0,
            "desc": f"{composite_name} (no config)",
            "source": "composite"
        }
    
    metrics_config = config.get("metrics", {})
    
    weighted_sum = 0.0
    total_weight = 0.0
    breakdown = {}
    
    for metric_key, metric_config in metrics_config.items():
        weight = metric_config.get("weight", 0.0)
        
        # Get raw value from indicators
        metric_data = indicators.get(metric_key)
        if metric_data is None:
            logger.debug(f"Metric {metric_key} not found in indicators")
            continue
        
        # Extract raw or value
        raw_value = None
        if isinstance(metric_data, dict):
            raw_value = metric_data.get("raw") or metric_data.get("value")
        else:
            raw_value = metric_data
        
        if raw_value is None:
            continue
        
        # Calculate score using composite rules
        score = _apply_composite_scoring_rules(raw_value, metric_config, metric_key)
        
        # Accumulate
        weighted_sum += score * weight
        total_weight += weight
        
        breakdown[metric_key] = {
            "raw_value": raw_value,
            "score": round(score, 2),
            "weight": weight,
            "contribution": round(score * weight, 2)
        }
    
    # Normalize to 0-10 scale
    final_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    final_score = round(final_score, 2)
    
    return {
        "value": final_score,
        "score": final_score,
        "desc": f"{composite_name.replace('Strength', '').replace('Quality', '').title()} Composite",
        "alias": composite_name.replace('Strength', ' Strength').replace('Quality', ' Quality').title(),
        "source": "composite",
        "breakdown": breakdown
    }


def compute_all_composites(indicators: dict, horizon: str) -> dict:
    """
    Calculate all composite scores for a given horizon.
    
    Args:
        indicators: Full indicators dict with all base metrics
        horizon: Trading horizon
    
    Returns:
        {
            "trendStrength": {...},
            "momentumStrength": {...},
            "volatilityQuality": {...}  # May be missing for multibagger
        }
    """
    composites = {}
    
    # Get list of composites for this horizon
    horizon_config = COMPOSITE_SCORING_CONFIG.get(horizon, {})
    
    for composite_name in horizon_config.keys():
        composite_result = compute_single_composite(
            composite_name,
            indicators,
            horizon
        )
        composites[composite_name] = composite_result
    
    return composites


def get_composite_config(horizon: str) -> dict:
    """
    Get composite scoring config for a horizon.
    
    Args:
        horizon: "intraday", "short_term", "long_term", "multibagger"
    
    Returns:
        Composite config dict or empty dict if not found
    """
    return COMPOSITE_SCORING_CONFIG.get(horizon, {})

# ============================================================================
# DYNAMIC SCORING FUNCTIONS
# ============================================================================

def calculate_dynamic_score(metric_name: str, raw_value: Any, indicators: Dict = None, metric_registry = METRIC_REGISTRY) -> float:
    """
    Translates raw indicator value into 0-10 score based on METRIC_REGISTRY.

    Args:
        metric_name: Name of the metric (e.g., 'rsi', 'wickRejection')
        raw_value: Raw value from indicator
        indicators: Full indicators dict (needed for crossover logic)

    Returns:
        Score between 0-10
    """
    cfg = metric_registry.get(metric_name)
    if not cfg:
        logger.debug(f"Metric '{metric_name}' not in registry, using neutral score")
        return None  # Changed from 0.0 to None

    if raw_value is None:
        return None  # Changed from 0.0 to None

        
    scoring_type = cfg.get("scoring_type")
    params = cfg.get("params", {})
    # ================================
    # 0. PASSTHROUGH (Composites already 0-10)
    # ================================
    if scoring_type == "passthrough":
        try:
            score = float(raw_value)
            return round(max(0.0, min(10.0, score)), 2)
        except (ValueError, TypeError):
            logger.warning(f"Passthrough failed for {metric_name}={raw_value}")
            return 0.0
    # ================================
    # 1. LINEAR RANGE (Smooth scoring)
    # ================================
    if scoring_type == "linear_range":
        min_v = params["min_val"]
        max_v = params["max_val"]
        s_min = params["score_at_min"]
        s_max = params["score_at_max"]

        # Calculate position in range (0-1)
        try:
            raw_float = float(raw_value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert {metric_name} value '{raw_value}' to float")
            return 5.0

        pos = (raw_float - min_v) / (max_v - min_v)
        pos = max(0, min(1, pos))  # Clamp to 0-1

        # Interpolate score
        score = s_min + pos * (s_max - s_min)
        return round(score, 2)

    # ================================
    # 2. STEPPED (Threshold-based)
    # ================================
    elif scoring_type == "stepped":
        try:
            raw_float = float(raw_value)
        except (ValueError, TypeError):
            return params.get("default", 5.0)

        for entry in params.get("thresholds", []):
            if "min" in entry and raw_float >= entry["min"]:
                return float(entry["score"])
            # Support "max" (Lower is better, e.g., Valuation ratios)
            if "max" in entry and raw_float <= entry["max"]:
                return float(entry["score"])

        return float(params.get("default", 0))

    # ================================
    # 3. MAPPING (Text values)
    # ================================
    elif scoring_type == "mapping":
        str_value = str(raw_value).strip()
        score = params.get(str_value)

        if score is None:
            # Try case-insensitive match
            # Try float comparison if both can be floats
            try:
                f_val = float(str_value)
                for key, val in params.items():
                    try:
                        if abs(float(key) - f_val) < 1e-6:
                            return float(val)
                    except (ValueError, TypeError):
                        continue
            except (ValueError, TypeError):
                pass

            logger.debug(f"No mapping for {metric_name}='{str_value}', using neutral")
            return 5.0

        return float(score)

    # ================================
    # 4. DYNAMIC CROSSOVER (Special)
    # ================================
    elif scoring_type == "dynamic_crossover":
        if not indicators:
            logger.warning(f"Crossover scoring for {metric_name} requires indicators dict")
            return 5.0

        crossover_metric = cfg.get("crossover_metric")
        if not crossover_metric:
            return 5.0

        # Get both values
        other_val = _extract_raw_value(indicators.get(crossover_metric))

        if other_val is None:
            return float(params.get("score_if_equal", 5))

        try:
            raw_float = float(raw_value)
            other_float = float(other_val)

            if raw_float > other_float:
                return float(params.get("score_if_higher", 7))
            elif raw_float < other_float:
                return float(params.get("score_if_lower", 3))
            else:
                return float(params.get("score_if_equal", 5))
        except (ValueError, TypeError):
            return 5.0

    # ================================
    # FALLBACK
    # ================================
    logger.warning(f"Unknown scoring_type '{scoring_type}' for {metric_name}")
    return 5.0


def extract_metric_score(metric_data: Any, metric_name: str, indicators: Dict = None) -> float:
    """
    Extract score from metric data using dynamic scoring.

    Priority:
    1. Calculate from 'raw' field using METRIC_REGISTRY
    2. Calculate from 'value' field using METRIC_REGISTRY
    3. Fallback to hardcoded 'score' field (backward compatibility)

    Args:
        metric_name: Name of the metric
        metric_data: Metric data (dict or primitive)
        indicators: Full indicators dict (for crossover logic)

    Returns:
        Score between 0-10
    """
    if metric_data is None:
        return None   # Changed from 0.0 to None

    # Handle dict format
    if isinstance(metric_data, dict):
        # Try 'raw' first (most accurate)
        raw = metric_data.get("raw")
        if raw is not None:
            return calculate_dynamic_score(metric_name, raw, indicators)

        # Try 'value' second
        value = metric_data.get("value")
        if value is not None:
            return calculate_dynamic_score(metric_name, value, indicators)

        # Fallback to hardcoded score (backward compatibility during migration)
        score = metric_data.get("score")
        if score is not None:
            logger.debug(f"Using hardcoded score for {metric_name} (migration mode)")
            return float(score)

        logger.warning(f"No raw/value/score found for {metric_name}, using neutral")
        return 5.0

    # Handle primitive value
    return calculate_dynamic_score(metric_name, metric_data, indicators)

# ==============================================================================
# ✅ SIMPLIFIED HELPER FUNCTIONS (No Text Mapping!)
# ==============================================================================

# def extract_metric_score(metric_data: Any, metric_name: str = None) -> float:
#     """
#     Universal score extractor - DIRECTLY reads 'score' field.
    
#     Your indicators already have scores, so this is trivial!
    
#     Priority:
#     1. If "score" field exists → use it ✅
#     2. If "value" field is numeric → use it (fallback)
#     3. Otherwise → return None
#     """
#     if metric_data is None:
#         return None
    
#     # Handle nested dict (your standard format)
#     if isinstance(metric_data, dict):
#         # ✅ PRIMARY PATH: Direct score extraction
#         score = metric_data.get("score")
#         if score is not None:
#             return float(score)
        
#         # Fallback: Try value if numeric
#         value = metric_data.get("value")
#         if isinstance(value, (int, float)):
#             return float(value)
        
#         # Last resort: raw field
#         raw = metric_data.get("raw")
#         if isinstance(raw, (int, float)):
#             return float(raw)
    
#     # Handle raw numeric (shouldn't happen with your structure)
#     elif isinstance(metric_data, (int, float)):
#         return float(metric_data)
    
#     return None


def check_liquidity_penalty(indicators: Dict, horizon: str) -> Tuple[float, str]:
    """
    Check if liquidity penalty applies (SEPARATE from category scoring).
    
    Returns:
        (penalty_amount, reason) or (0.0, None) if no penalty
    """
    rule = LIQUIDITY_PENALTY_RULES.get(horizon)
    if not rule:
        return 0.0, None
    
    avg_vol = indicators.get("avg_volume_30Days", {})
    
    # Extract value (your standard structure)
    if isinstance(avg_vol, dict):
        vol_value = avg_vol.get("value") or avg_vol.get("raw")
    else:
        vol_value = avg_vol
    
    if vol_value is None:
        return 0.0, None
    
    try:
        vol_value = float(vol_value)
    except:
        return 0.0, None
    
    if vol_value < rule["min_avg_volume"]:
        return rule["penalty_multiplier"], rule["reason"]
    
    return 0.0, None

def _extract_raw_value(metric_data: dict) -> Any:
    """Helper to extract the unscaled raw value from the indicator dict."""
    if not metric_data or not isinstance(metric_data, dict):
        return None
    return metric_data.get("raw", metric_data.get("value"))

def get_active_metrics_for_horizon(horizon: str) -> dict:
    """
    Returns:
        {
            "trend": [...],
            "momentum": [...],
            "volatility": [...],
            "volume": [...],
            "structure": [...]
        }
    """
    cfg = HORIZON_METRIC_INCLUSION.get(horizon, {})
    active = {}

    for category, metrics in cfg.items():
        if category == "exclude":
            continue
        active[category] = list(metrics)

    # Enforce exclusions
    excluded = set(cfg.get("exclude", []))
    for category in active:
        active[category] = [
            m for m in active[category] if m not in excluded
        ]

    return active

def compute_category_score(
    indicators: dict,
    horizon: str,
    category: str,
    active_metrics: list
) -> tuple:
    """
    Returns:
        (category_score_0_to_10, breakdown_dict)
    """
    horizon_weights = METRIC_WEIGHTS.get(horizon, {})
    weights = horizon_weights.get(category, {})
    
    total_weight = 0.0
    weighted_score = 0.0
    breakdown = {}

    for metric in active_metrics:
        metric_weight = weights.get(metric)
        if metric_weight is None:
            continue

        metric_data = indicators.get(metric)
        score = extract_metric_score(metric_data, metric, indicators)

        if score is None:
            continue

        score = max(0.0, min(10.0, score))  # defensive clamp

        weighted_score += score * metric_weight
        total_weight += metric_weight

        breakdown[metric] = {
            "score": score,
            "weight": metric_weight,
            "contribution": score * metric_weight
        }

    if total_weight == 0:
        return None, breakdown

    normalized = weighted_score / total_weight
    return round(normalized, 2), breakdown

def apply_technical_penalties(indicators: dict, horizon: str) -> tuple:
    """
    Returns:
        (total_penalty, list_of_reasons)
    """
    penalties = TECHNICAL_PENALTIES.get(horizon, [])
    total = 0.0
    reasons = []

    for rule in penalties:
        metric = rule["metric"]
        metric_data = indicators.get(metric)
        
        is_passthrough = METRIC_REGISTRY.get(metric, {}).get("scoring_type") == "passthrough"
        if is_passthrough:
            actual = extract_metric_score(metric_data, metric, indicators)
        else:
            actual = _extract_raw_value(metric_data)

        if actual is None:
            continue

        threshold = rule["threshold"]
        op = rule["operator"]

        triggered = (
            (op == "<" and actual < threshold) or
            (op == ">" and actual > threshold) or
            (op == "<=" and actual <= threshold) or
            (op == ">=" and actual >= threshold)
        )

        if triggered:
            total += rule["penalty"]
            reasons.append(rule["reason"])

    return round(total, 2), reasons

def apply_technical_bonuses(indicators: dict, horizon: str = "short_term") -> tuple:
    from config.gate_evaluator import evaluate_gates
    total = 0.0
    reasons = []
    excluded = set(HORIZON_METRIC_INCLUSION.get(horizon, {}).get("exclude", [])) 

    for rule in TECHNICAL_BONUSES:
        # skip bonus rules referencing excluded metrics
        rule_metrics = [k for k in rule["gates"].keys() if not k.startswith("_")]
        if any(m in excluded for m in rule_metrics):
            continue
            
        passes, _ = evaluate_gates(rule["gates"], indicators)
        if passes:
            total += rule["bonus"]
            reasons.append(rule["reason"])

    return round(total, 2), reasons
    

def compute_technical_score(indicators: dict, horizon: str) -> dict:
    """
    Final technical score computation.
    Returns a rich, explainable structure.
    """
    active = get_active_metrics_for_horizon(horizon)
    category_weights = HORIZON_TECHNICAL_WEIGHTS.get(horizon, {})

    category_scores = {}
    category_breakdown = {}
    total_score = 0.0

    for category, metrics in active.items():
        if category not in category_weights:
            continue

        cat_score, breakdown = compute_category_score(
            indicators, horizon, category, metrics
        )

        if cat_score is None:
            continue

        weight = category_weights[category]
        total_score += cat_score * weight

        category_scores[category] = {
            "score": cat_score,
            "weight": weight,
            "weighted": round(cat_score * weight, 2)
        }
        category_breakdown[category] = breakdown

    # Penalties
    penalty, penalty_reasons = apply_technical_penalties(indicators, horizon)

    # Bonuses
    bonus, bonus_reasons = apply_technical_bonuses(indicators , horizon)

    # Liquidity penalty
    liq_penalty, liq_reason = check_liquidity_penalty(indicators, horizon)

    final_score_raw = total_score - penalty - liq_penalty + bonus
    final_score = round(max(0.0, min(10.0, final_score_raw)), 2)
    
    if final_score_raw > 10.0:
        logger.debug(f"[Technical Score] Clamped score from {final_score_raw:.2f} down to {final_score}")
   
    out = {
        "score": final_score,
        "horizon": horizon,
        "base_score": round(total_score, 2),
        "category_scores": category_scores,
        "penalties": {
            "total": round(penalty + liq_penalty, 2),
            "technical": round(penalty, 2),
            "liquidity": round(liq_penalty, 2),
            "reasons": (penalty_reasons + ([liq_reason] if liq_reason else []))
        },
        "bonuses": {
            "total": round(bonus, 2),
            "reasons": bonus_reasons
        },
        "breakdown": category_breakdown
    }
    # logger.debug(f"compute technical score result : {out} calculated for indicators dict with values {indicators}")
    return out


# ==============================================================================
# EXPORT
# ==============================================================================

__all__ = [
    "compute_technical_score",
    "compute_all_composites",
    "compute_single_composite",
    "get_composite_config",
]