# config/master_config.py (REFACTORED - CLEANED VERSION)
"""
Master Configuration - Global Section
Smart inheritance: Horizons inherit from global, only override what's different

REFACTORING NOTES:
- Removed ALL technical scoring weights/composites (now in technical_score_config.py)
- Removed ALL fundamental scoring weights (now in fundamental_score_config.py)
- Kept only: gates, execution, risk management, time estimation, strategy preferences
"""

# ✅ P1-1 FIX: Runtime Import Guard
import os
import sys

# Prevent direct import of raw config from anywhere except the ConfigExtractor or bootstrapper.
# This ensures that all components use the validated, unified Extractor API.
_caller_frame = sys._getframe(1)
_caller_name = _caller_frame.f_globals.get('__name__', '')
# Removed 'config.query_optimized_extractor' from allowed callers
_allowed_callers = ['config.config_extractor', 'main', 'tests', 'scripts', 'services', 'verify_invariants', '__main__', 'config', 'tests.robustness', 'importlib', 'baktest']

if not any(caller in _caller_name for caller in _allowed_callers) and 'pytest' not in sys.modules:
    raise ImportError(
        f"ARCHITECTURAL VIOLATION: Direct import of 'master_config' by '{_caller_name}'. "
        "Use 'config.config_extractor.ConfigExtractor' instead."
    )

HYBRID_METRIC_REGISTRY = {
    # Linear Range Metrics (Growth, Trend, Consistency)
    "fundamentalMomentum": {
        "type": "numeric",
        "category": "growth_momentum",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 0.0,
            "max_val": 20.0,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Avg(Growth + EPS/5)"
    },
    "earningsConsistencyIndex": {
        "type": "numeric",
        "category": "quality_integrity",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 5.0,
            "max_val": 30.0,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Avg(ROE + Net Profit Margin)"
    },
    "trendConsistency": {
        "type": "numeric",
        "category": "technical_reliability",
        "scoring_type": "linear_range",
        "params": {
            "min_val": 15.0,
            "max_val": 30.0,
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "ADX based trend stability"
    },

    # Stepped Metrics (Required for specific directional logic like Max thresholds)
    "priceToIntrinsicValue": {
        "type": "numeric",
        "category": "valuation_price",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"max": 0.8, "score": 10},  # Deep Value
                {"max": 1.0, "score": 7},   # Fair Value
                {"max": 1.2, "score": 3}    # Overvalued
            ],
            "default": 0.0
        },
        "description": "Price / IV (Lower is better)"
    },
    "volatilityAdjustedRoe": {
        "type": "numeric",
        "category": "quality_vol",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 10.0, "score": 10},
                {"min": 5.0, "score": 7},
                {"min": 2.0, "score": 3}
            ],
            "default": 0.0
        },
        "description": "ROE / ATR %"
    },
    "fcfYieldVsVolatility": {
        "type": "numeric",
        "category": "cash_vol",
        "scoring_type": "stepped",
        "params": {
            "thresholds": [
                {"min": 8.0, "score": 10},
                {"min": 4.0, "score": 7},
                {"min": 1.5, "score": 3}
            ],
            "default": 1.0
        },
        "description": "FCF Yield / ATR %"
    },
    "priceVsPrimaryTrendPct": {
        "type": "numeric",
        "category": "technical_anchor",
        "scoring_type": "linear_range",
        "params": {
            "min_val": -0.10, # 10% below 200DMA
            "max_val": 0.15,  # 15% above 200DMA
            "score_at_min": 0,
            "score_at_max": 10
        },
        "description": "Distance from 200DMA"
    }
}

# STEP 6 LAYER: Internal Composition (Walls)
# How the 7 metrics weight into the single "Hybrid Pillar Score"
HYBRID_PILLAR_COMPOSITION = {
    "intraday":    {"trendConsistency": 0.40, "priceVsPrimaryTrendPct": 0.20, "fundamentalMomentum": 0.15, "earningsConsistencyIndex": 0.10, "volatilityAdjustedRoe": 0.05, "priceToIntrinsicValue": 0.05, "fcfYieldVsVolatility": 0.05},
    "short_term":  {"trendConsistency": 0.25, "fundamentalMomentum": 0.15, "volatilityAdjustedRoe": 0.15, "priceVsPrimaryTrendPct": 0.15, "earningsConsistencyIndex": 0.10, "priceToIntrinsicValue": 0.10, "fcfYieldVsVolatility": 0.10},
    "long_term":   {"volatilityAdjustedRoe": 0.20, "priceToIntrinsicValue": 0.20, "earningsConsistencyIndex": 0.15, "fcfYieldVsVolatility": 0.15, "fundamentalMomentum": 0.10, "trendConsistency": 0.10, "priceVsPrimaryTrendPct": 0.10},
    "multibagger": {
        "fundamentalMomentum": 0.25,
        "earningsConsistencyIndex": 0.25,
        "priceToIntrinsicValue": 0.20,
        "fcfYieldVsVolatility": 0.15,
        "volatilityAdjustedRoe": 0.05,
        "trendConsistency": 0.05,
        "priceVsPrimaryTrendPct": 0.05,
    }
}

# STEP 6 LAYER: Global Arbitration (Blueprint)
# How much the Hybrid Pillar counts vs Tech and Fund
HORIZON_PILLAR_WEIGHTS = {
    "intraday":    {"tech": 0.70, "fund": 0.00, "hybrid": 0.30},
    "short_term":  {"tech": 0.50, "fund": 0.20, "hybrid": 0.30},
    "long_term":   {"tech": 0.30, "fund": 0.40, "hybrid": 0.30},
    "multibagger": {"tech": 0.10, "fund": 0.60, "hybrid": 0.30}
}

# ============================================================================
# GATE METRIC REGISTRY (Metadata for gate validation)
# ============================================================================


GATE_METRIC_REGISTRY = {
    # ===========================
    # TREND GATES
    # ===========================
    "maTrendSignal": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Moving Average trend alignment signal (1=Strong Up, 0.5=Developing, -1=Down)",
        "context_paths": [("indicators", "maTrendSignal")],
        "optional": True
    },
    "ichimokuSignals": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Ichimoku cloud and TK signal strength",
        "context_paths": [("indicators", "ichimokuSignals")],
        "optional": True
    },
    "prevSupertrend": {
        "type": "text",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Previous Supertrend direction",
        "context_paths": [("indicators", "prevSupertrend")]
    },
    "trendStrength": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Composite trend strength (0-10)",
        "context_paths": [("indicators", "trendStrength")],
        "optional": True
    },
    
    "adx": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "ADX trend strength indicator",
        "context_paths": [("indicators", "adx")]
    },
    "relStrengthNifty": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Relative Strength versus Nifty (Alpha)",
        "context_paths": [("indicators", "relStrengthNifty")],
        "optional": True
    },
    
    # ===========================
    # MOMENTUM GATES
    # ===========================
    "momentumStrength": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "Composite momentum strength (0-10)",
        "context_paths": [("indicators", "momentumStrength")],
        "optional": True
    },
    
    "rsi": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "RSI momentum indicator",
        "context_paths": [("indicators", "rsi")]
    },
    "rsiSlope": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "RSI slope indicator",
        "context_paths": [("indicators", "rsiSlope")]
    },
    
    "macdHistogram": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "MACD histogram value",
        "context_paths": [("indicators", "macdHistogram")]
    },
    
    "prevMacdHistogram": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "Previous MACD histogram value",
        "context_paths": [("indicators", "prevMacdHistogram")]
    },
    
    # ===========================
    # VOLATILITY GATES
    # ===========================
    "volatilityQuality": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "Composite volatility quality (0-10)",
        "context_paths": [("indicators", "volatilityQuality")],
        "optional": True
    },
    
    "atrPct": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "ATR as percentage of price",
        "context_paths": [("indicators", "atrPct")]
    },
    "ttmSqueeze": {
        "type": "boolean",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "TTM Squeeze signal (True if in squeeze)",
        "context_paths": [("indicators", "ttmSqueeze")],
        "optional": True
    },
    
    # # Special case: alias mapping
    # "volatilitybandsatrPct": {
    #     "type": "numeric",
    #     "category": "volatility",
    #     "validation_type": "threshold",
    #     "description": "Volatility band ATR check",
    #     "context_paths": [("indicators", "atrPct")]  # Maps to atrPct
    # },
    
    # ===========================
    # VOLUME GATES
    # ===========================
    "rvol": {
        "type": "numeric",
        "category": "volume",
        "validation_type": "threshold",
        "description": "Relative volume ratio",
        "context_paths": [("indicators", "rvol")]
    },
    
    "volume": {
        "type": "numeric",
        "category": "volume",
        "validation_type": "threshold",
        "description": "Absolute volume",
        "context_paths": [
            ("price_data", "volume"),
            ("indicators", "volume")  # ✅ Fallback path
        ]
    },
    # ===========================
    # STRUCTURE GATES
    # ===========================
    "prevHigh": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Previous candle high, used by momentumFlow invalidation",
        "context_paths": [("indicators", "prevHigh")],
        "optional": True
    },
    
    "prevClose": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Previous candle close",
        "context_paths": [("indicators", "prevClose")],
        "optional": True
    },
    "bbPercentB": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "Bollinger %B value",
        "context_paths": [("indicators", "bbPercentB")],  # ✅ P2-1 FIX: Corrected path
        "optional": True                                    # ✅ P2-1 FIX: Marked as optional
    },
    
    "bbWidth": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Bollinger Band Width",
        "context_paths": [("indicators", "bbWidth")],  # ✅ P2-1 FIX: Corrected path
        "optional": True                                # ✅ P2-1 FIX: Marked as optional
    },
    
    "position52w": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Distance from 52-week high",
        "context_paths": [("indicators", "position52w")],  # ✅ P2-1 FIX: Corrected path
        "optional": True                                    # ✅ P2-1 FIX: Marked as optional
    },
    
    "priceVs52wHighPct": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Current price as a percentage of 52-week high",
        "context_paths": [("fundamentals", "priceVs52wHighPct")]
    },
    "maxPatternAgeCandles": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Maximum allowed age of a pattern in candles",
        "context_paths": [("indicators", "maxPatternAgeCandles")],
        "optional": True
    },
    "maxSetupStalenessCandles": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Maximum allowed staleness of a setup in candles",
        "context_paths": [("indicators", "maxSetupStalenessCandles")],
        "optional": True
    },
    
    "drawdown52wHigh": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Drawdown percentage from 52-week high",
        "context_paths": [("fundamentals", "drawdown52wHigh")]
    },

    "priceVsPrimaryTrendPct": {
        "type": "numeric",
        "category": "trend_anchor",
        "validation_type": "threshold",
        "description": "Price distance from 200-period EMA (% of price)",
        "context_paths": [
            ("scoring", "hybrid", "metrics", "priceVsPrimaryTrendPct", "raw"),
            ("scoring", "hybrid", "metrics", "priceVsPrimaryTrendPct", "value"),
            ("indicators", "priceVsPrimaryTrendPct")  # ✅ Fallback
        ]
    },

    # ===========================
    # FUNDAMENTAL GATES
    # ===========================
    "roe": {
        "type": "numeric",
        "category": "profitability",
        "validation_type": "threshold",
        "description": "Return on Equity",
        "context_paths": [("fundamentals", "roe")]
    },
    "epsGrowth5y": {
        "type": "numeric",
        "category": "growth",
        "validation_type": "threshold",
        "description": "5-year EPS Growth rate",
        "context_paths": [("fundamentals", "epsGrowth5y")],
        "optional": True
    },
    "revenueGrowth5y": {
        "type": "numeric",
        "category": "growth",
        "validation_type": "threshold",
        "description": "5-year Revenue Growth rate",
        "context_paths": [("fundamentals", "revenueGrowth5y")],
        "optional": True
    },
    "quarterlyGrowth": {
        "type": "numeric",
        "category": "growth",
        "validation_type": "threshold",
        "description": "Recent Quarterly Growth rate",
        "context_paths": [("fundamentals", "quarterlyGrowth")],
        "optional": True
    },
    
    "roce": {
        "type": "numeric",
        "category": "profitability",
        "validation_type": "threshold",
        "description": "Return on Capital Employed",
        "context_paths": [("fundamentals", "roce")]
    },
    
    "deRatio": {
        "type": "numeric",
        "category": "financial_health",
        "validation_type": "threshold",
        "description": "Debt-to-Equity ratio",
        "context_paths": [("fundamentals", "deRatio")]
    },
    
    "piotroskiF": {
        "type": "numeric",
        "category": "quality",
        "validation_type": "threshold",
        "optional": True,
        "description": "Piotroski F-Score",
        "context_paths": [("fundamentals", "piotroskiF")]
    },
    
    "peRatio": {
        "type": "numeric",
        "category": "valuation",
        "validation_type": "threshold",
        "description": "Price to Earnings Ratio",
        "context_paths": [("fundamentals", "peRatio"), ("fundamentals", "pe")]
    },
    
    "fcfYield": {
        "type": "numeric",
        "category": "valuation",
        "validation_type": "threshold",
        "description": "Free Cash Flow Yield",
        "context_paths": [("fundamentals", "fcfYield")]
    },
    
    "dividendYield": {
        "type": "numeric",
        "category": "valuation",
        "validation_type": "threshold",
        "description": "Dividend Yield",
        "context_paths": [("fundamentals", "dividendYield")]
    },
    
    # ===========================
    # MARKET GATES
    # ===========================
    "marketCap": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Market capitalization",
        "context_paths": [
            ("fundamentals", "marketCap"),
            ("price_data", "marketCap")  # ✅ Fallback
        ]
    },
    
    "institutionalOwnership": {
        "type": "numeric",
        "category": "ownership",
        "validation_type": "threshold",
        "description": "Institutional ownership %",
        "context_paths": [
            ("fundamentals", "institutionalOwnership")
        ]
    },
    
    # ===========================
    # COMPOSITE GATES
    # ===========================
    "confidence": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Setup confidence score",
        "context_paths": [("confidence", "clamped")]  # Post-confidence calc
    },
    
    "rrRatio": {
        "type": "numeric",
        "category": "risk_reward",
        "validation_type": "threshold",
        "description": "Risk-reward ratio",
        "context_paths": [("risk_candidates", "rrRatio")],
        "optional": True,
        "skip_reason": "deferred_to_stage2"
    },
    
    "technicalScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Aggregated technical score",
        "context_paths": [("scoring", "technical", "score")]
    },
    
    "fundamentalScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Aggregated fundamental score",
        "context_paths": [("scoring", "fundamental", "score")]
    },
    
    "hybridScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Hybrid metrics score",
        "context_paths": [("scoring", "hybrid", "score")]
    },
    
    # ===========================
    # EXECUTION GATES its not metric
    # ===========================
    # "volatilityguards": {
    #     "type": "boolean",
    #     "category": "execution",
    #     "validation_type": "boolean",
    #     "description": "Volatility guards passed",
    #     "context_paths": [("execution_rules", "rules", "volatilityguards", "passed")],
    #     "optional": True  # ✅ Mark as optional
    # },
    
    # ===========================
    # MISSING METRICS (NOT YET IMPLEMENTED)
    # ===========================
    "marketTrendScore": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Overall market trend score",
        "context_paths": [("indicators", "marketTrendScore")],
        "optional": True,  # ✅ Not yet computed
        "fallback": 5.0    # ✅ Default neutral value
    },
    
    "relStrengthNifty": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Relative strength vs benchmark (Nifty)",
        "context_paths": [("indicators", "relStrengthNifty")],
        "optional": True,
        "fallback": 0.0
    },
    
    # Alias for relStrengthNifty used in legacy configs
    "relativeStrength": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Alias for relStrengthNifty",
        "context_paths": [("indicators", "relStrengthNifty")],
        "optional": True,
        "fallback": 0.0
    },
    
    "sectorTrendScore": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Sector trend alignment",
        "context_paths": [("indicators", "sectorTrendScore")],
        "optional": True,
        "fallback": 5.0
    }
}


MASTER_CONFIG = {
    "hybrid_composition": {
        "multibagger": [
            "fundamentalMomentum", "earningsConsistencyIndex", "trendConsistency",
            "priceToIntrinsicValue", "volatilityAdjustedRoe", "fcfYieldVsVolatility",
            "priceVsPrimaryTrendPct"
        ],
    },
    # ============================================================================
    # GLOBAL CONSTANTS (Universal Physics & Logic)
    # ============================================================================
    "global": {   
        "time_estimation": {
            "candles_per_unit": 1,
            "base_friction": 0.8,
            "velocity_factors": {
                "strong_trend": {"min_strength": 7.0, "factor": 1.2},
                "normal_trend": {"min_strength": 5.0, "factor": 1.0},
                "weak_trend": {"max_strength": 5.0, "factor": 0.8}
            },
            "formatting": {
                "intraday_candle_minutes": 15,
                "days_per_week": 5
            }
        },
        
        "entry_gates": {
            "structural": {
                "description": "Validates setup structure before confidence calculation",
                "execution_order": 1,
                "gates": {
                    "adx": {"min": 18},
                    "trendStrength": {"min": 3.0},
                    "volatilityQuality": {"min": 3.0},
                    "rsi": {"min": None, "max": None},
                    "macdHistogram": {"min": None},
                    "bbPercentB": {"min": None, "max": None},
                    "atrPct": {"min": None},
                    "roe": {"min": None},
                    "roce": {"min": None},
                    "deRatio": {"max": None},
                    "piotroskiF": {"min": None},
                    "rvol": {"min": None},
                    "volume": {"min": None},
                    "marketTrendScore": {"min": None},   
                    "relStrengthNifty": {"min": None},    
                    "sectorTrendScore": {"min": None},   
                    "marketCap": {"min": None},              
                    "institutionalOwnership": {"min": None}
                }
            },
            "execution_rules": {
                "description": "Complex validation rules requiring custom logic",
                "execution_order": 1.5,
                "volatility_guards": {
                    "description": "Dynamic volatility quality requirements",
                    "extreme_vol_buffer": 2.0,
                    "min_quality_breakout": 2.0,
                    "min_quality_normal": 4.0,
                    "enabled": True,
                    "logic": "if atr_pct > extreme_threshold: use min_quality_breakout"
                },
                "structure_validation": {
                    "description": "Price structure alignment checks",
                    "breakout_clearance": 0.001,
                    "breakdown_clearance": 0.001,
                    "enabled": True,
                    "logic": "price must clear resistance/support by clearance %"
                },
                "sl_distance_validation": {
                    "description": "Stop loss feasibility constraints",
                    "min_atr_multiplier": 0.5,
                    "max_atr_multiplier": 5.0,
                    "enabled": True,
                    "logic": "0.5*ATR <= |entry-SL| <= 5*ATR"
                },
                "target_proximity_rejection": {
                    "description": "Reject if targets too close to resistance",
                    "min_t1_distance": 0.005,
                    "min_t2_distance": 0.01,
                    "enabled": True,
                    "logic": "t1 must be > resistance * (1 + min_t1_distance)"
                }
            },
            "opportunity": {
                "description": "Validates trade opportunity quality after confidence calculation",
                "execution_order": 2,
                "gates": {
                    "confidence": {"min": 55},
                    "rrRatio": {"min": 1.5},
                    "technicalScore": {"min": None},
                    "fundamentalScore": {"min": None},
                    "hybridScore": {"min": None},
                    "maxPatternAgeCandles": None,
                    "maxSetupStalenessCandles": None
                }
            },
            
            # ❌ REMOVED: setup_gate_specifications
            # This entire section has been MOVED to setup_pattern_matrix.py
            # Each setup now owns its own gate requirements and horizon overrides
            
            # OLD CODE (DELETED):
            # "setup_gate_specifications": {
            #     "MOMENTUM_BREAKOUT": {
            #         "structural_gates": {...},
            #         "opportunity_gates": {...}
            #     },
            #     ...
            # }
        },

        "calculation_engine": {
            "horizon_priority_overrides": {
                "intraday": {
                    "MOMENTUM_BREAKDOWN":    80,
                    "BEAR_TREND_FOLLOWING":  75,
                    "MOMENTUM_BREAKOUT":     90,
                    "VOLATILITY_SQUEEZE": 88,
                    "TREND_PULLBACK": 85,
                    "REVERSAL_MACD_CROSS_UP": 82,
                    
                    # Blocked for Intraday (Low Priority)
                    "VALUE_TURNAROUND": 40,
                    "DEEP_VALUE_PLAY": 30,
                    "QUALITY_ACCUMULATION": 35
                },
                
                "short_term": {
                    # Elite Patterns (95-100)
                    "PATTERN_DARVAS_BREAKOUT": 98,
                    "PATTERN_VCP_BREAKOUT": 97,
                    "PATTERN_CUP_BREAKOUT": 96,
                    "PATTERN_FLAG_BREAKOUT": 95,
                    "PATTERN_STRIKE_REVERSAL": 94,
                    
                    # Generic Signals (80-90)
                    "MOMENTUM_BREAKOUT": 90,
                    "MOMENTUM_BREAKDOWN": 70,
                    "BEAR_TREND_FOLLOWING": 65,
                    "VOLATILITY_SQUEEZE": 88,
                    "TREND_PULLBACK": 85,
                    "REVERSAL_MACD_CROSS_UP": 82,
                    
                    # Value Plays (Lower for Short-Term)
                    "VALUE_TURNAROUND": 70,
                    "DEEP_VALUE_PLAY": 65,
                    "QUALITY_ACCUMULATION": 68
                },
                
                "long_term": {
                    # Elite Patterns (95-100) - Trend/Value Focus
                    "PATTERN_GOLDEN_CROSS": 99,
                    "PATTERN_CUP_BREAKOUT": 96,
                    "PATTERN_VCP_BREAKOUT": 95,
                    
                    # Value Setups (85-92) - PRIORITY for Long-Term
                    "VALUE_TURNAROUND": 92,
                    "DEEP_VALUE_PLAY": 90,
                    "QUALITY_ACCUMULATION": 88,
                    
                    # Trend Following (75-80)
                    "TREND_PULLBACK": 80,
                    "REVERSAL_MACD_CROSS_UP": 75,
                    
                    # Momentum DEPRIORITIZED (55-65)
                    "MOMENTUM_BREAKOUT": 60,
                    "MOMENTUM_BREAKDOWN": 45,
                    "BEAR_TREND_FOLLOWING": 40,
                    "VOLATILITY_SQUEEZE": 58,
                    "PATTERN_FLAG_BREAKOUT": 65
                }
            },
            

            # Spread Adjustment
            "spread_adjustment": {
                "market_cap_brackets": {
                    "large_cap": {"min": 100000, "spread_pct": 0.001},
                    "mid_cap": {"min": 10000, "max": 100000, "spread_pct": 0.002},
                    "small_cap": {"max": 10000, "spread_pct": 0.005}
                }
            },

            
            # Wick Rejection
            "wickRejection": {
                "max_ratio": 2.5,
                "calculation": "abs(high - close) / abs(close - open)"
            },
            

        },
        
        "position_sizing": {
            "base_risk_pct": 0.01,
            "global_setup_multipliers": {
                "DEEP_VALUE_PLAY": 1.5,
                "VALUE_TURNAROUND": 1.4,
                "QUALITY_ACCUMULATION": 1.3,
                "VOLATILITY_SQUEEZE": 1.3,
                "DEEP_PULLBACK": 1.5,
                "TREND_PULLBACK": 1.0,
                "REVERSAL_MACD_CROSS_UP": 1.0,
                "REVERSAL_RSI_SWING_UP": 0.9,
                "REVERSAL_ST_FLIP_UP": 1.1,
                "QUALITY_ACCUMULATION_DOWNTREND": 0.9,
                "PATTERN_DARVAS_BREAKOUT": 1.0,
                "PATTERN_VCP_BREAKOUT": 1.0,
                "PATTERN_CUP_BREAKOUT": 1.0,
                "PATTERN_FLAG_BREAKOUT": 1.0,
                "PATTERN_STRIKE_REVERSAL": 1.0,
                "PATTERN_GOLDEN_CROSS": 1.0,
                "MOMENTUM_BREAKOUT": 0.8,
                "MOMENTUM_BREAKDOWN": 0.7,
                "SELL_AT_RANGE_TOP": 0.7,
                "TAKE_PROFIT_AT_MID": 0.6,
                "BEAR_TREND_FOLLOWING": 0.7,
                "GENERIC": 0.5
            },
            "volatility_adjustments": {
                "high_quality": {"vol_qual_min": 7.0, "multiplier": 1.2},
                "low_quality": {"vol_qual_max": 5.0, "multiplier": 0.9}
            }
        },
        
        "trend_weights": {
            "primary": 0.50,
            "secondary": 0.30,
            "acceleration": 0.20
        },
        
        "boosts": {
            "pattern": {"single": 0.8, "confluence": 1.5},
            "momentum": {"score": 1.2, "trigger_rvol": 2.0},
            "volatility": {
                "squeeze": {"score": 1.0, "volatilityQuality": {"min": 7.0}},
                "expansion": {"score": 0.6, "volatilityQuality": {"min": 4.0}}
            },
            "relative_strength": {
                "outperforming_weak_market": 1.5,
                "outperforming_strong_market": 0.8
            },
            "proximity": {"support": 0.6, "resistance": 0.8},
            "max_cap": 2.5
        },
        
        "moving_averages": {
            "type": "EMA",
            "fast": 20,
            "mid": 50,
            "slow": 200,
            "keys": ["ema20", "ema50", "ema200"],
            "dip_buy_reference": "maMid"
        },
        
        "indicators": {
            "rsi_period": 14,
            "adx_period": 14,
            "atr_period": 14,
            "supertrend": {"period": 10, "multiplier": 3},
            "bollinger": {"period": 20, "std_dev": 2.0},
            "keltner": {"period": 20, "atr_mult": 1.5},
            "pivot_type": "classic",
            "stochastic": {
                "k": 14,
                "d": 3,
                "smooth": 3,
                "overbought": 80,
                "oversold": 20
            }
        },
        
        "momentum_thresholds": {
            "rsiSlope": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
            "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
        },
        
        "trend_thresholds": { #new addition
            "slope": {"strong": 10.0, "moderate": 3.0}  # neutral mid-range fallback
        },
        
        "volatility": {
            "scoring_thresholds": {
                "atrPct": {"excellent": 2.5, "good": 3.0, "fair": 4.5, "poor": 5.5},
                "bbWidth": {"tight": 3.0, "normal": 6.0, "wide": 12.0}
            },
            "extreme_threshold": 7.0,
            "min_quality": 4.0,
            "squeeze_quality_min": 7.0,
            "breakout_min_quality": 2.0
        },
        
        "risk_management": {
            "risk_per_trade": 500,
            "max_position_pct": 0.02,
            "setup_size_multipliers": {"default": 1.0},
            "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
            "rrRatio": {"min": 1.5},
            "horizon_t2_cap": 0.10,
            "rr_regime_adjustments": {
                "strong_trend": {"adx": {"min": 35}, "t1_mult": 2.0, "t2_mult": 4.0},
                "normal_trend": {"adx": {"min": 20}, "t1_mult": 1.5, "t2_mult": 3.0},
                "weak_trend": {"adx": {"max": 20}, "t1_mult": 1.2, "t2_mult": 2.5}
            },
            "rr_gates": {"min_t1": 1.5,"min_t2": 2.0,"min_structural": 2.0,"execution_floor": 1.0},
            
            # ✅ NEW: Execution Policy Constants (Migrated from trade_enhancer.py)
            "volatility_buffer_factors": {
                'low': 0.0, 'normal': 0.25, 'high': 0.5, 'extreme': 1.0
            },
            "min_sl_atr_multiples": {
                'intraday': 2.0, 'short_term': 2.0, 'long_term': 2.5
            },
            "target_adjustment_factors": {
                'low': 0.85, 'normal': 1.0, 'high': 1.15, 'extreme': 1.3
            },
            "min_rr_by_trend": {
                "explosive": {"strength": 8.5, "min_rr": 1.0},
                "strong": {"strength": 6.5, "min_rr": 1.3}, 
                "normal": {"strength": 4.5, "min_rr": 1.5},
                "weak": {"strength": 0.0, "min_rr": 2.0}
            },
            "base_spread_pct": {
                'intraday': 0.0015, 'short_term': 0.001, 'long_term': 0.0008
            },
            "expiry_penalty": -20.0,
            
            # ✅ PHASE 5 FIX: Adaptive Spread Cost Configuration
            "adaptive_spread": {
                "volume_adjustments": {
                    "high": {"threshold": 3.0, "factor": 0.7},
                    "normal": {"threshold": 1.5, "factor": 1.0},
                    "low": {"factor": 1.3}
                }
            }
        },
        
        "execution": {
            "stop_loss_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "max_hold_candles": 20,
            "dip_buy_reference": "maMid",
            "stop_loss": {
                "vol_qual_high_mult": 1.5,
                "vol_qual_normal_mult": 2.0,
                "vol_qual_low_mult": 3.0,
                "min_distance_mult": 0.5
            },
            "structure_validation": {
                "breakout_tolerance": 1.001,
                "breakdown_tolerance": 0.999
            },
            "confidence_adjustments": {
                "warning_penalty": -5,
                "violation_penalty": -15,
                "risk_score_thresholds": {
                    "high": 80,
                    "moderate": 60,
                    "low": 40
                },
                "risk_score_high_penalty": -10,
                "risk_score_moderate_penalty": -5,
                "risk_score_low_bonus": 5
            }
        },
        
        "lookback": {"python_data": 600},
        
        "targets": {
            "resistance_cushion": 0.96,
            "support_cushion": 1.005,
            "min_distance_pct": 0.5,
            "support_buffer": 0.998,
            "cover_cushion": 1.005
        }
    },
    
    # ============================================================================
    # HORIZON CONFIGURATIONS
    # ============================================================================
    "horizons": {
        "intraday": {
            "timeframe": "15m",
            "description": "Quick scalps and day trades",
            
            "volume_analysis": {
                "rvol": {"surge_threshold": 3.0, "drought_threshold": 0.7}
            },
            "time_estimation": {
                "candles_per_unit": 4
            },
            
            "moving_averages": {
                "type": "EMA",
                "fast": 20,
                "mid": 50,
                "slow": 200,
                "keys": ["ema20", "ema50", "ema200"],
                "dip_buy_reference": "maFast"
            },
            
            "indicators": {
                "rsi_period": 9,
                "adx_period": 10,
                "atr_period": 10,
                "supertrend": {"period": 7, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "fibonacci",
                "stochastic": {
                    "k": 8,
                    "d": 3,
                    "smooth": 3,
                    "overbought": 80,
                    "oversold": 20
                }
            },
            
            "momentum_thresholds": {
                "rsiSlope": {
                    "acceleration_floor": 0.10,
                    "deceleration_ceiling": -0.10
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            
            "volatility": {
                "scoring_thresholds": {
                    "atrPct": {
                        "excellent": 1.5,
                        "good": 3.0,
                        "fair": 4.0,
                        "poor": 5.0
                    },
                    "bbWidth": {
                        "tight": 2.0,
                        "normal": 5.0,
                        "wide": 10.0
                    }
                }
            },
            
            "trend_thresholds": {
                "slope": {
                    "strong": 12.0,
                    "moderate": 5.0
                }
            },
            
            "position_sizing": {
                "base_risk_pct": 0.005
            },
            
            "risk_management": {
                "max_position_pct": 0.01,
                "setup_size_multipliers": {
                    "VOLATILITY_SQUEEZE": 1.3,
                    "MOMENTUM_BREAKOUT": 0.8
                },
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.005},
                "rrRatio": {"min": 1.2},
                "horizon_t2_cap": 0.04,
                "rr_gates": { "min_t1": 1.2, "min_t2": 2.2, "min_structural": 1.3, "execution_floor": 1.0 }
            },
            
            "execution": {
                "stop_loss_atr_mult": 1.5,
                "target_atr_mult": 2.5,
                "max_hold_candles": 25,
                "base_hold_days": 1,
                "proximity_rejection": {
                    "resistance_mult": 1.003,
                    "support_mult": 0.997
                },
                "min_profit_pct": 0.3,
                
                # ✅ Moved from strategy matrix (Feedback R2-2)
                "indian_market_gates": {
                    "min_avg_volume": 500000,
                    "max_spread_pct": 0.003,
                    "min_delivery_pct": 40,
                    "avoid_gsm": True,
                    "time_filters": {
                        "avoid_first_15_min": True,
                        "avoid_last_15_min": True,
                        "reduce_size_lunch": 0.5,
                        "optimal_windows": [
                            {"start": "09:45", "end": "11:30", "multiplier": 1.0},
                            {"start": "13:30", "end": "15:00", "multiplier": 1.0}
                        ]
                    },
                    "risk_controls": {
                        "max_position_pct": 0.01,
                        "mandatory_stop_loss": True,
                        "max_trades_per_day": 3
                    }
                }
            },
            
            "lookback": {"python_data": 500},
            
            # ✅ KEEP: Horizon-specific structural gates (universal defaults)
            "entry_gates": {
                "structural": {
                    "adx": {"min": 20},  # ✅ Raised from 18; closes ADX dead zone vs confidence penalty
                    "atrPct": {"min": 0.3},
                    "trendStrength": {"min": 5.0},
                    "volatilityQuality": {"min": 5.0}
                },
                "execution_rules": {
                    "volatility_guards": {
                        "extreme_vol_buffer": 1.5,     # Tighter than global 2.0
                        "min_quality_breakout": 2.5,   # Higher floor for 15m candles
                        "min_quality_normal": 4.0,
                        "enabled": True
                    },
                    "structure_validation": {
                        "breakout_clearance": 0.002,   # 0.2% clearance for intraday noise
                        "enabled": True
                    }
                },
                "opportunity": {
                    "confidence": {"min": 65},
                    "rrRatio": {"min": 1.5},  # ✅ Aligned with rr_gates.min_t1
                    "technicalScore": {"min": None},
                    "fundamentalScore": {"min": None}
                }
                
                # ❌ REMOVED: setup_gate_overrides
                # This section has been MOVED to setup_pattern_matrix.py
                # Each setup now defines its own horizon_overrides
                
            },
            
            # ❌ REMOVED: enhancements section (migrated to confidence_config.py)
            # All confidence enhancements are now in confidence_config.py
            # See confidence_config.CONFIDENCE_CONFIG['horizons']['intraday']['conditional_adjustments']['bonuses']
        },
        
        "short_term": {
            "timeframe": "1d",
            "description": "Swing trading (Days to Weeks)",
            "volume_analysis": {
                "rvol": {"surge_threshold": 2.5, "drought_threshold": 0.7}
            },
            "time_estimation": {
                "candles_per_unit": 1
            },
            "position_sizing": {
                "base_risk_pct": 0.01
            },
            "risk_management": {
                "max_position_pct": 0.02,
                "setup_size_multipliers": {
                    "DEEP_PULLBACK": 1.5,
                    "MOMENTUM_BREAKOUT": 1.0
                },
                "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},  # ✅ Extended for overnight gap risk
                "rrRatio": {"min": 1.4},
                "horizon_t2_cap": 0.10,
                "rr_gates": { "min_t1": 1.5, "min_t2": 2.5, "min_structural": 1.5, "execution_floor": 1.2 }
            },
            "execution": {
                "stop_loss_atr_mult": 2.0,
                "target_atr_mult": 3.0,
                "max_hold_candles": 15,
                "dip_buy_reference": "maMid",
                "base_hold_days": 10,
                "proximity_rejection": {
                    "resistance_mult": 1.005,
                    "support_mult": 0.995
                },
                "min_profit_pct": 0.5,
                
                "indian_market_gates": {
                    "min_avg_volume": 500000,
                    "max_spread_pct": 0.003,
                    "min_delivery_pct": 40,
                    "avoid_gsm": True,
                    "time_filters": {
                        "avoid_first_15_min": True,
                        "avoid_last_15_min": False,
                        "reduce_size_lunch": 1.0,
                        "optimal_windows": []
                    },
                    "risk_controls": {
                        "max_position_pct": 0.02,
                        "mandatory_stop_loss": True,
                        "max_trades_per_day": 5
                    }
                }
            },
            "lookback": {"python_data": 600},
            "entry_gates": {
                "structural": {
                    "adx": {"min": 15},
                    "atrPct": {"min": 0.8},                # ✅ Wider range acceptable
                    "trendStrength": {"min": 4.0},
                    "volatilityQuality": {"min": 4.0}
                },

                "execution_rules": {
                    "volatility_guards": {
                        "extreme_vol_buffer": 2.5,     # Wider than intraday (1.5) to allow daily swings
                        "min_quality_breakout": 3.0,   # Prevents entry on low-quality "wicky" breakouts
                        "min_quality_normal": 4.5,     # Standard baseline for swing trades
                        "enabled": True
                    },
                    "structure_validation": {
                        "breakout_clearance": 0.005,   # 0.5% clearance; avoids "false starts" on 1D charts
                        "enabled": True
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60},
                    "rrRatio": {"min": 1.5},  # ✅ Entrance Gate (aligned with 3.0/2.0 baseline)
                    "fundamentalScore": {"min": 3.0}
                },
            },
                
            "trend_thresholds": {
                "slope": {
                    "strong": 10.0,
                    "moderate": 3.0
                }
            },
            "momentum_thresholds": {
                "rsiSlope": {
                    "acceleration_floor": 0.05,
                    "deceleration_ceiling": -0.05
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
        },
        
        "long_term": {
            "timeframe": "1wk",
            "description": "Trend Following & Investing",
            
            "trend_thresholds": {
                "slope": {
                    "strong": 5.0,
                    "moderate": 2.0
                }
            },
            "volume_analysis": {
                "rvol": {"surge_threshold": 2.0, "drought_threshold": 0.8}
            },
            "time_estimation": {
                "candles_per_unit": 0.2
            },
            "moving_averages": {
                "type": "WMA",
                "fast": 10,
                "mid": 40,
                "slow": 50,
                "keys": ["wma10", "wma40", "wma50"],
                "dip_buy_reference": "maMid"
            },
            
            "momentum_thresholds": {
                "rsiSlope": {
                    "acceleration_floor": 0.03,
                    "deceleration_ceiling": -0.03
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            
            "volatility": {
                "scoring_thresholds": {
                    "atrPct": {
                        "excellent": 5.5,
                        "good": 9.0,
                        "fair": 13.0,
                        "poor": 18.0
                    },
                    "bbWidth": {
                        "tight": 4.0,
                        "normal": 8.0,
                        "wide": 15.0
                    }
                }
            },
            
            "position_sizing": {
                "base_risk_pct": 0.015
            },
            
            "risk_management": {
                "max_position_pct": 0.03,
                "setup_size_multipliers": {"default": 1.0},
                "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
                "rrRatio": {"min": 1.5},
                "horizon_t2_cap": 0.20,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx": {"min": 35}, "t1_mult": 2.5, "t2_mult": 5.0},
                    "normal_trend": {"adx": {"min": 20}, "t1_mult": 2.0, "t2_mult": 4.0},
                    "weak_trend": {"adx": {"max": 20}, "t1_mult": 1.5, "t2_mult": 3.0}
                },
                "rr_gates": { "min_t1": 1.5, "min_t2": 3.5, "min_structural": 1.5, "execution_floor": 1.4 }
            },
            
            "execution": {
                "stop_loss_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "max_hold_candles": 52,
                "dip_buy_reference": "maMid",
                "base_hold_days": 60,
                "proximity_rejection": {
                    "resistance_mult": 1.01,
                    "support_mult": 0.99
                },
                "min_profit_pct": 1.0,
                
                "indian_market_gates": {
                    "min_avg_volume": 500000,
                    "max_spread_pct": 0.005,
                    "min_delivery_pct": 50,
                    "avoid_gsm": True,
                    "time_filters": {
                        "avoid_first_15_min": False,
                        "avoid_last_15_min": False,
                        "reduce_size_lunch": 1.0,
                        "optimal_windows": []
                    },
                    "risk_controls": {
                        "max_position_pct": 0.03,
                        "mandatory_stop_loss": True,
                        "max_trades_per_day": 5
                    }
                }
            },
            
            "lookback": {"python_data": 800},
            
            "entry_gates": {
                "structural": {
                    "adx": {"min": None},
                    "atrPct": {"min": 1.0},
                    "trendStrength": {"min": 4.0},   # ✅ Raised from 3.0; aligns with penalty & confidence threshold
                    "volatilityQuality": {"min": 2.0}  # ✅ Added floor; was None
                },
                "execution_rules": {
                    "volatility_guards": {
                        "extreme_vol_buffer": 4.0,     # Very loose; long-term trends often start with high vol
                        "min_quality_breakout": 4.0,   # Ensures the weekly breakout has some substance
                        "enabled": True
                    },
                    "structure_validation": {
                        "breakout_clearance": 0.01,    # Requires a full 1% move above resistance to confirm
                        "enabled": True
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55},
                    "rrRatio": {"min": 2.0},  # ✅ Entrance Gate (Short Term Swing Target)
                    "fundamentalScore": {"min": 6.0},
                    "technicalScore": {"min": None}
                },
            },
        },
        "multibagger": {
            "timeframe": "1mo",
            "description": "Long-term wealth creation (Months to Years)",
            "volume_analysis": {"rvol": {"surge_threshold": 1.5, "drought_threshold": 0.8}},
            "time_estimation": {"candles_per_unit": 0.05},
            "position_sizing": {"base_risk_pct": 0.02},
            "risk_management": {
                "max_position_pct": 0.05,
                "rrRatio": {"min": 3.0},
                "horizon_t2_cap": 0.30,
                "rr_gates": {"min_t1": 2.0, "min_t2": 5.0, "min_structural": 2.0, "execution_floor": 1.5}
            },
            "execution": {
                "stop_loss_atr_mult": 3.0,
                "target_atr_mult": 10.0,
                "max_hold_candles": 240,
                "base_hold_days": 365,
                "proximity_rejection": {"resistance_mult": 1.02, "support_mult": 0.98},
                "min_profit_pct": 5.0
            },
            "lookback": {"python_data": 1500},
            "entry_gates": {
                "structural": {"adx": {"min": None}, "atrPct": {"min": 2.0}},
                "opportunity": {"confidence": {"min": 50}, "rrRatio": {"min": 3.0}}
            }
        },
    }
}

# ============================================================================
# SUMMARY OF CHANGES:
# ============================================================================
# ❌ REMOVED FROM MASTER_CONFIG:
#    - global.entry_gates.setup_gate_specifications (entire section)
#    - horizons.*.entry_gates.setup_gate_overrides (from all horizons)
#
# ✅ KEPT IN MASTER_CONFIG:
#    - global.entry_gates.structural (universal baseline)
#    - global.entry_gates.execution_rules (complex validation logic)
#    - global.entry_gates.opportunity (post-confidence gates)
#    - horizons.*.entry_gates (horizon-specific defaults)
#
# 📦 MOVED TO setup_pattern_matrix.py:
#    - Setup-specific gate requirements
#    - Horizon-specific setup overrides
# ============================================================================



