# config/master_config.py (REFACTORED - CLEANED VERSION)
"""
Master Configuration - Global Section
Smart inheritance: Horizons inherit from global, only override what's different

REFACTORING NOTES:
- Removed ALL technical scoring weights/composites (now in technical_score_config.py)
- Removed ALL fundamental scoring weights (now in fundamental_score_config.py)
- Kept only: gates, execution, risk management, time estimation, strategy preferences
"""

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
    "multibagger": {"fundamentalMomentum": 0.25, "earningsConsistencyIndex": 0.25, "priceToIntrinsicValue": 0.20, "fcfYieldVsVolatility": 0.15, "trendConsistency": 0.10, "priceVsPrimaryTrendPct": 0.05}
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
        "context_paths": [("indicators", "maTrendSignal")] # ✅ ADD THIS
    },
    "trendStrength": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Composite trend strength (0-10)",
        "context_paths": [("indicators", "trendStrength")]  # ✅ ADD THIS
    },
    
    "adx": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "ADX trend strength indicator",
        "context_paths": [("indicators", "adx")]  # ✅ ADD THIS
    },
    
    # ===========================
    # MOMENTUM GATES
    # ===========================
    "momentumStrength": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "Composite momentum strength (0-10)",
        "context_paths": [("indicators", "momentumStrength")]
    },
    
    "rsi": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "RSI momentum indicator",
        "context_paths": [("indicators", "rsi")]
    },
    "rsislope": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "RSI slope indicator",
        "context_paths": [("indicators", "rsislope")]
    },
    
    "macdhistogram": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "MACD histogram value",
        "context_paths": [("indicators", "macdhistogram")]
    },
    
    "prevmacdhistogram": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "Previous MACD histogram value",
        "context_paths": [("indicators", "prevmacdhistogram"), ("indicators", "prev_macdhistogram")]
    },
    
    # ===========================
    # VOLATILITY GATES
    # ===========================
    "volatilityQuality": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "Composite volatility quality (0-10)",
        "context_paths": [("indicators", "volatilityQuality")]
    },
    
    "atrPct": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "ATR as percentage of price",
        "context_paths": [("indicators", "atrPct")]
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
    "bbpercentb": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Bollinger Band %B position",
        "context_paths": [("indicators", "bbpercentb")]
    },
    
    "bbWidth": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Bollinger Band Width",
        "context_paths": [("indicators", "bbWidth")]
    },
    
    "position52w": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Distance from 52-week high",
        "context_paths": [("indicators", "position52w")]
    },
    
    "priceVs52wHighPct": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Current price as a percentage of 52-week high",
        "context_paths": [("fundamentals", "priceVs52wHighPct")]
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
    
    "dividendyield": {
        "type": "numeric",
        "category": "valuation",
        "validation_type": "threshold",
        "description": "Dividend Yield",
        "context_paths": [("fundamentals", "dividendyield"), ("fundamentals", "dividendYield")]
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
            ("fundamentals", "institutionalOwnership"),
            ("price_data", "institutionalownership")  # ✅ Case variation
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
        "context_paths": [("risk_model", "rrRatio")]
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
    
    "relativeStrength": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Relative strength vs benchmark",
        "context_paths": [("indicators", "relativeStrength")],
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
                    "macdhistogram": {"min": None},
                    "bbpercentb": {"min": None, "max": None},
                    "atrPct": {"min": None},
                    "roe": {"min": None},
                    "roce": {"min": None},
                    "deRatio": {"max": None},
                    "piotroskiF": {"min": None},
                    "rvol": {"min": None},
                    "volume": {"min": None},
                    "marketTrendScore": {"min": None},   
                    "relativeStrength": {"min": None},    
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
                    "max_pattern_age_candles": None,
                    "max_setup_staleness_candles": None
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
                    # Elite Patterns (95-100) - Always win
                    "PATTERN_DARVAS_BREAKOUT": 98,
                    "PATTERN_FLAG_BREAKOUT": 97,
                    "PATTERN_STRIKE_REVERSAL": 96,
                    "PATTERN_VCP_BREAKOUT": 95,
                    
                    # Generic Signals (85-90)
                    "MOMENTUM_BREAKOUT": 90,
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
                    "VOLATILITY_SQUEEZE": 58,
                    "PATTERN_FLAG_BREAKOUT": 65
                },
                
                "multibagger": {
                    # Value Setups DOMINATE (95-98)
                    "VALUE_TURNAROUND": 98,
                    "DEEP_VALUE_PLAY": 96,
                    "QUALITY_ACCUMULATION": 95,
                    
                    # Long-term Patterns (90-92)
                    "PATTERN_GOLDEN_CROSS": 92,
                    "PATTERN_CUP_BREAKOUT": 90,
                    
                    # Trend Following (70-75)
                    "TREND_PULLBACK": 72,
                    "REVERSAL_MACD_CROSS_UP": 70,
                    
                    # Momentum BLOCKED (55 or below)
                    "MOMENTUM_BREAKOUT": 55,
                    "VOLATILITY_SQUEEZE": 52,
                    "PATTERN_FLAG_BREAKOUT": 50
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
            
            # Volume Signatures
            "volume_signatures": {
                "surge": {"threshold": 3.0, "confidence_adjustment": 15},
                "drought": {"threshold": 0.7, "confidence_adjustment": -25},
                "climax": {"threshold": 2.0, "rsi_condition_min": 70, "confidence_adjustment": -15 }
            },
            
            # Wick Rejection
            "wickRejection": {
                "max_ratio": 2.5,
                "calculation": "abs(high - close) / abs(close - open)"
            },
            
            # Divergence Detection
            # "divergence_detection": {
            #     "lookback": 10,
            #     "slope_diff_min": -0.05,
            #     "confidence_penalties": {
            #         "bearish_divergence": 0.70,
            #         "bullish_divergence": 0.70
            #     },
            #     "severity_bands": {
            #         "minor": {
            #             "rsislope": {"min": -0.03},
            #             "confidence_penalty": 0.90,
            #             "allow_entry": True
            #         },
            #         "moderate": {
            #             "rsislope": {"min": -0.08},
            #             "confidence_penalty": 0.70,
            #             "allow_entry": True
            #         },
            #         "severe": {
            #             "rsislope": {"min": -999},
            #             "confidence_penalty": 0.50,
            #             "allow_entry": False
            #         }
            #     }
            # },
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
                "MOMENTUM_BREAKOUT": 0.8,
                "MOMENTUM_BREAKDOWN": 0.7,
                "SELL_AT_RANGE_TOP": 0.7,
                "TAKE_PROFIT_AT_MID": 0.6,
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
            "rsislope": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
            "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
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
            "position_sizing": {
                "min_capital": 25000,     # Optional: Warn if trade is too small?
                "max_capital": 50000      # Hard Cap: Never invest more than this
            },
            "rr_gates": {"min_t1": 1.5,"min_t2": 2.0,"min_structural": 2.0,"execution_floor": 1.0}
        },
        
        "execution": {
            "stop_loss_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "max_hold_candles": 20,
            "dip_buy_reference": "maMid",
            "risk_reward_min": 2.0,
            "stop_loss": {
                "vol_qual_high_mult": 1.5,
                "vol_qual_normal_mult": 2.0,
                "vol_qual_low_mult": 3.0,
                "min_distance_mult": 0.5
            },
            "structure_validation": {
                "breakout_tolerance": 1.001,
                "breakdown_tolerance": 0.999
            }
        },
        
        "lookback": {"python_data": 600},
        
        # ❌ DELETED: global.scoring — deprecated, never extracted by any
        # section in extract_global_sections(). Thresholds owned by
        # technical_score_config.py.
        
        "strategy_preferences": {
            # This layer controls TRADING preferences per horizon
            # Independent of objective setup classification
            "horizon_strategy_config": {
                "intraday": {
                    "preferred_setups": [
                        "MOMENTUM_BREAKOUT",
                        "VOLATILITY_SQUEEZE", 
                        "TREND_PULLBACK"
                    ],
                    "blocked_setups": [
                        "QUALITY_ACCUMULATION",
                        "DEEP_VALUE_PLAY",
                        "VALUE_TURNAROUND"
                    ],
                    "sizing_multipliers": {
                        "VOLATILITY_SQUEEZE": 1.3,
                        "MOMENTUM_BREAKOUT": 0.8
                    },
                    "fundamentalScore": {"min": None}
                },
                
                "short_term": {
                    "preferred_setups": [
                        "MOMENTUM_BREAKOUT",
                        "TREND_PULLBACK",
                        "VOLATILITY_SQUEEZE",
                        "REVERSAL_MACD_CROSS_UP"
                    ],
                    "blocked_setups": [],
                    "sizing_multipliers": {
                        "DEEP_PULLBACK": 1.5,
                        "TREND_PULLBACK": 1.3,
                        "MOMENTUM_BREAKOUT": 1.0
                    },
                    "fundamentalScore": {"min": 3.0}
                },
                
                "long_term": {
                    "preferred_setups": [
                        "VALUE_TURNAROUND",
                        "DEEP_VALUE_PLAY",
                        "QUALITY_ACCUMULATION",
                        "TREND_PULLBACK"
                    ],
                    "blocked_setups": [
                        "VOLATILITY_SQUEEZE",
                        "MOMENTUM_BREAKDOWN"
                    ],
                    "sizing_multipliers": {
                        "VALUE_TURNAROUND": 1.5,
                        "DEEP_VALUE_PLAY": 1.4,
                        "QUALITY_ACCUMULATION": 1.3,
                        "MOMENTUM_BREAKOUT": 0.8
                    },
                    "fundamentalScore": {"min": 6.0},
                    "filters": {
                        "require_low_debt": True,
                        "roe": {"min": 15.0},
                        "roce": {"min": 15.0},
                        "deRatio": {"max": 0.5}
                    }
                },
                
                "multibagger": {
                    "preferred_setups": [
                        "VALUE_TURNAROUND",
                        "DEEP_VALUE_PLAY",
                        "QUALITY_ACCUMULATION"
                    ],
                    "blocked_setups": [
                        "MOMENTUM_BREAKOUT",
                        "VOLATILITY_SQUEEZE",
                        "REVERSAL_MACD_CROSS_UP"
                    ],
                    "sizing_multipliers": {
                        "VALUE_TURNAROUND": 1.8,
                        "DEEP_VALUE_PLAY": 1.6,
                        "QUALITY_ACCUMULATION": 1.5
                    },
                    "fundamentalScore": {"min": 8.0},
                    "filters": {
                        "require_low_debt": True,
                        "roe": {"min": 20.0},
                        "roce": {"min": 25.0},
                        "piotroskiF": {"min": 7}
                    }
                }
            }
        },
        
        "strategy_priority": {
            "intraday": {
                "blocked_strategies": [
                    "value_investing",
                    "income_investing",
                    "position_trading",
                    "canslim",
                    "quality_growth",
                    "reversal_trading"
                ],
                "priority_multipliers": {
                    "day_trading": 1.3,
                    "momentum": 1.2,
                    "swing_trading": 1.1,
                    "trend_following": 0.8,
                    "minervini_growth": 0.5
                }
            },
            
            "short_term": {
                "blocked_strategies": [],
                "priority_multipliers": {
                    "swing_trading": 1.2,
                    "momentum": 1.15,
                    "quality_growth": 1.1,
                    "trend_following": 1.1,
                    "reversal_trading": 1.1,
                    "minervini_growth": 1.1,
                    "canslim": 1.05,
                    "value_investing": 0.85,
                    "position_trading": 0.8,
                    "income_investing": 0.7
                }
            },
            
            "long_term": {
                "blocked_strategies": ["day_trading"],
                "priority_multipliers": {
                    "quality_growth": 1.35,
                    "value_investing": 1.3,
                    "position_trading": 1.25,
                    "reversal_trading": 1.2,
                    "income_investing": 1.2,
                    "minervini_growth": 1.1,
                    "canslim": 1.05,
                    "trend_following": 1.0,
                    "swing_trading": 0.8,
                    "momentum": 0.7
                }
            },
            
            "multibagger": {
                "blocked_strategies": [
                    "day_trading",
                    "momentum",
                    "swing_trading"
                ],
                "priority_multipliers": {
                    "quality_growth": 1.5,
                    "value_investing": 1.45,
                    "minervini_growth": 1.35,
                    "canslim": 1.25,
                    "position_trading": 1.2,
                    "income_investing": 1.1,
                    "reversal_trading": 1.05,
                    "trend_following": 0.8
                }
            }
        },
        
        "setup_priority_design_philosophy": {
            "long_term": {
                "philosophy": "Value-first, quality-second, momentum-last",
                "reasoning": "Long-term horizon prioritizes fundamental strength and value over technical momentum",
                "conflict_resolution_rules": {
                    "VALUE_TURNAROUND + QUALITY_ACCUMULATION": "VALUE_TURNAROUND wins (turning point is rarer opportunity)",
                    "DEEP_VALUE_PLAY + QUALITY_ACCUMULATION": "DEEP_VALUE wins (cheaper entry with safety margin)",
                    "TREND_PULLBACK + VALUE_TURNAROUND": "VALUE_TURNAROUND wins (fundamental catalyst > technical)"
                },
                "priority_order_rationale": [
                    "90: VALUE_TURNAROUND - Highest conviction: value + improving momentum",
                    "88: DEEP_VALUE_PLAY - Pure value with margin of safety",
                    "85: QUALITY_ACCUMULATION - Quality + patient accumulation",
                    "80: TREND_PULLBACK - Technical entry in established trend",
                    "75: MOMENTUM_BREAKOUT - Pure momentum (lower for long-term)"
                ]
            },
            
            "short_term": {
                "philosophy": "Momentum-first, trend-second, value-tertiary",
                "reasoning": "Short-term trades capitalize on price momentum and technical patterns",
                "conflict_resolution_rules": {
                    "MOMENTUM_BREAKOUT + TREND_PULLBACK": "MOMENTUM_BREAKOUT wins (stronger signal)",
                    "VOLATILITY_SQUEEZE + MOMENTUM_BREAKOUT": "VOLATILITY_SQUEEZE wins (earlier entry)",
                    "VALUE_TURNAROUND + TREND_PULLBACK": "TREND_PULLBACK wins (faster payoff)"
                }
            },
            
            "intraday": {
                "philosophy": "Momentum-only, fundamentals-never",
                "reasoning": "Intraday scalping requires fast price action without fundamental considerations",
                "conflict_resolution_rules": {
                    "Any fundamental setup": "Block or heavily penalize - wrong timeframe"
                }
            },
            
            "multibagger": {
                "philosophy": "Fundamentals-dominant, quality-obsessed, patience-required",
                "reasoning": "Multi-year holds require exceptional business quality and value",
                "conflict_resolution_rules": {
                    "VALUE_TURNAROUND always wins": "Cheapness + improvement = perfect multibagger formula"
                }
            }
        },
    
        "targets": {
            "resistance_cushion": 0.96,
            "support_cushion": 1.005,
            "min_distance_pct": 0.5,
            "support_buffer": 0.998,
            "cover_cushion": 1.005
        },
        "divergence": {
            "rsislope_deceleration_ceiling": -0.08,
            "bearish_penalty": 0.70,
            "bullish_penalty": 0.70,
            "severity_bands": {
                "severe": {
                    "rsislope_threshold": -0.08,
                    "allow_entry": False,
                    "confidence_penalty": 1.0  # Legacy/unused
                },
                "moderate": {
                    "rsislope_threshold": -0.03,
                    "allow_entry": True,         # Does not block Phase 6 execution
                    "confidence_penalty": 0.7    # NOTE: Unused. Score penalties live in confidence_config.py
                },
                "minor": {
                    "rsislope_threshold": 0.0,
                    "allow_entry": True,         # Does not block Phase 6 execution
                    "confidence_penalty": 0.9    # NOTE: Unused. Score penalties live in confidence_config.py
                }
            }
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
                "rsislope": {
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
                    "strong": 15.0,
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
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "rrRatio": {"min": 1.2},
                "horizon_t2_cap": 0.04,
                "rr_gates": { "min_t1": 1.5, "min_t2": 2.2, "min_structural": 2.5, "execution_floor": 1.2 }
            },
            
            "execution": {
                "stop_loss_atr_mult": 1.5,
                "target_atr_mult": 2.5,
                "max_hold_candles": 25,
                "risk_reward_min": 1.5,
                "base_hold_days": 1,
                "proximity_rejection": {
                    "resistance_mult": 1.003,
                    "support_mult": 0.997
                },
                "min_profit_pct": 0.3
            },
            
            "lookback": {"python_data": 500},
            
            # ✅ KEEP: Horizon-specific structural gates (universal defaults)
            "entry_gates": {
                "structural": {
                    "adx": {"min": 18},
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
                    "rrRatio": {"min": 1.2},
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
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "rrRatio": {"min": 1.4},
                "horizon_t2_cap": 0.10,
                "rr_gates": { "min_t1": 1.6, "min_t2": 2.5, "min_structural": 3.0, "execution_floor": 1.4 }
            },
            "execution": {
                "stop_loss_atr_mult": 2.0,
                "target_atr_mult": 3.0,
                "max_hold_candles": 15,
                "dip_buy_reference": "maMid",
                "risk_reward_min": 2.0,
                "base_hold_days": 10,
                "proximity_rejection": {
                    "resistance_mult": 1.005,
                    "support_mult": 0.995
                },
                "min_profit_pct": 0.5
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
                    "rrRatio": {"min": 1.4},
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
                "rsislope": {
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
                "rsislope": {
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
                "rr_gates": { "min_t1": 2.0, "min_t2": 3.0, "min_structural": 4.0, "execution_floor": 1.4 }
            },
            
            "execution": {
                "stop_loss_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "max_hold_candles": 52,
                "dip_buy_reference": "maMid",
                "risk_reward_min": 2.5,
                "base_hold_days": 60,
                "proximity_rejection": {
                    "resistance_mult": 1.01,
                    "support_mult": 0.99
                },
                "min_profit_pct": 1.0
            },
            
            "lookback": {"python_data": 800},
            
            "entry_gates": {
                "structural": {
                    "adx": {"min": None},
                    "atrPct": {"min": 1.0},
                    "trendStrength": {"min": 3.0},
                    "volatilityQuality": {"min": None}
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
                    "rrRatio": {"min": 1.5},
                    "fundamentalScore": {"min": 6.0},
                    "technicalScore": {"min": None}
                },
            },
                
            # ❌ REMOVED: enhancements section (migrated to confidence_config.py)
        },
        
        "multibagger": {
            "timeframe": "1mo",
            "description": "Deep Value & Compounders",
            "trend_thresholds": {
                "slope": {
                    "strong": 30.0,
                    "moderate": 10.0
                }
            },
            "volume_analysis": {
                "rvol": {"surge_threshold": 1.8, "drought_threshold": 0.8}
            },
            "time_estimation": {
                "candles_per_unit": 0.05
            },
            "moving_averages": {
                "type": "MMA",
                "fast": 6,
                "mid": 12,
                "slow": 12,
                "keys": ["mma6", "mma12", "mma12"],
                "dip_buy_reference": "maSlow"
            },
            "indicators": {
                "rsi_period": 14,
                "adx_period": 14,
                "atr_period": 20,
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic",
                "stochastic": {
                    "k": 21,
                    "d": 5,
                    "smooth": 5,
                    "overbought": 85,
                    "oversold": 15
                }
            },
            "momentum_thresholds": {
                "rsislope": {
                    "acceleration_floor": 0.02,
                    "deceleration_ceiling": -0.02
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            "volatility": {
                "scoring_thresholds": {
                    "atrPct": {
                        "excellent": 11.5,
                        "good": 18.0,
                        "fair": 27.0,
                        "poor": 36.0
                    },
                    "bbWidth": {
                        "tight": 6.0,
                        "normal": 12.0,
                        "wide": 20.0
                    }
                }
            },
            "position_sizing": {
                "base_risk_pct": 0.02
            },
            "risk_management": {
                "max_position_pct": 0.05,
                "setup_size_multipliers": {"QUALITY_ACCUMULATION": 1.5},
                "atr_sl_limits": {"max_percent": 0.10, "min_percent": 0.02},
                "rrRatio": {"min": 2.0},
                "horizon_t2_cap": 1.00,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx": {"min": 30}, "t1_mult": 3.0, "t2_mult": 10.0},
                    "normal_trend": {"adx": {"min": 20}, "t1_mult": 2.5, "t2_mult": 8.0},
                    "weak_trend": {"adx": {"max": 20}, "t1_mult": 2.0, "t2_mult": 6.0}
                },
                "rr_gates": { "min_t1": 2.5, "min_t2": 4.0, "min_structural": 5.0, "execution_floor": 1.5 }
            },
            "execution": {
                "stop_loss_atr_mult": 3.0,
                "target_atr_mult": 10.0,
                "max_hold_candles": 60,
                "dip_buy_reference": "maSlow",
                "risk_reward_min": 3.0,
                "base_hold_days": 180,
                "proximity_rejection": {
                    "resistance_mult": 1.02,
                    "support_mult": 0.98
                },
                "min_profit_pct": 2.0
            },
            "lookback": {"python_data": 3000},
            "entry_gates": {
                "structural": {
                    "adx": {"min": None},
                    "atrPct": {"min": 1.0},
                    "trendStrength": {"min": 3.0},
                    "volatilityQuality": {"min": None}
                },
                "execution_rules": {
                    "volatility_guards": {"enabled": False},
                    "sl_distance_validation": {"enabled": False}, 
                    "structure_validation": {"enabled": False} 
                },
                "opportunity": {
                    "confidence": {"min": 60},
                    "rrRatio": {"min": 2.0},
                    "fundamentalScore": {"min": 8.0},
                    "hybridScore": {"min": None}
                },
                
                # ❌ REMOVED: setup_gate_overrides
          # ❌ REMOVED: enhancements section (migrated to confidence_config.py)
        }
    }
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



