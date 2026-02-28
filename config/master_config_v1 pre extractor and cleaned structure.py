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
    "priceVsMaSlowPct": {
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
    "intraday":    {"trendConsistency": 0.40, "priceVsMaSlowPct": 0.20, "fundamentalMomentum": 0.15, "earningsConsistencyIndex": 0.10, "volatilityAdjustedRoe": 0.05, "priceToIntrinsicValue": 0.05, "fcfYieldVsVolatility": 0.05},
    "short_term":  {"trendConsistency": 0.25, "fundamentalMomentum": 0.15, "volatilityAdjustedRoe": 0.15, "priceVsMaSlowPct": 0.15, "earningsConsistencyIndex": 0.10, "priceToIntrinsicValue": 0.10, "fcfYieldVsVolatility": 0.10},
    "long_term":   {"volatilityAdjustedRoe": 0.20, "priceToIntrinsicValue": 0.20, "earningsConsistencyIndex": 0.15, "fcfYieldVsVolatility": 0.15, "fundamentalMomentum": 0.10, "trendConsistency": 0.10, "priceVsMaSlowPct": 0.10},
    "multibagger": {"fundamentalMomentum": 0.20, "earningsConsistencyIndex": 0.20, "volatilityAdjustedRoe": 0.20, "fcfYieldVsVolatility": 0.15, "priceToIntrinsicValue": 0.15, "trendConsistency": 0.05, "priceVsMaSlowPct": 0.05}
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
    "trendStrength": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "Composite trend strength (0-10)"
    },
    
    "adx": {
        "type": "numeric",
        "category": "trend",
        "validation_type": "threshold",
        "description": "ADX trend strength indicator"
    },
    
    # ===========================
    # MOMENTUM GATES
    # ===========================
    "momentumStrength": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "Composite momentum strength (0-10)"
    },
    
    "rsi": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "RSI momentum indicator"
    },
    
    "macdhistogram": {
        "type": "numeric",
        "category": "momentum",
        "validation_type": "threshold",
        "description": "MACD histogram value"
    },
    
    # ===========================
    # VOLATILITY GATES
    # ===========================
    "volatilityQuality": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "Composite volatility quality (0-10)"
    },
    
    "atrPct": {
        "type": "numeric",
        "category": "volatility",
        "validation_type": "threshold",
        "description": "ATR as percentage of price"
    },
    
    # ===========================
    # VOLUME GATES
    # ===========================
    "rvol": {
        "type": "numeric",
        "category": "volume",
        "validation_type": "threshold",
        "description": "Relative volume ratio"
    },
    
    "volume": {
        "type": "numeric",
        "category": "volume",
        "validation_type": "threshold",
        "description": "Absolute volume"
    },
    
    # ===========================
    # STRUCTURE GATES
    # ===========================
    "bbpercentb": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Bollinger Band %B position"
    },
    
    "position52w": {
        "type": "numeric",
        "category": "structure",
        "validation_type": "threshold",
        "description": "Distance from 52-week high"
    },
    
    # ===========================
    # FUNDAMENTAL GATES
    # ===========================
    "roe": {
        "type": "numeric",
        "category": "profitability",
        "validation_type": "threshold",
        "description": "Return on Equity"
    },
    
    "roce": {
        "type": "numeric",
        "category": "profitability",
        "validation_type": "threshold",
        "description": "Return on Capital Employed"
    },
    
    "deRatio": {
        "type": "numeric",
        "category": "financial_health",
        "validation_type": "threshold",
        "description": "Debt-to-Equity ratio"
    },
    
    "piotroskiF": {
        "type": "numeric",
        "category": "quality",
        "validation_type": "threshold",
        "description": "Piotroski F-Score"
    },
    
    # ===========================
    # MARKET GATES
    # ===========================
    "marketCap": {
        "type": "numeric",
        "category": "market",
        "validation_type": "threshold",
        "description": "Market capitalization"
    },
    
    "institutionalOwnership": {
        "type": "numeric",
        "category": "ownership",
        "validation_type": "threshold",
        "description": "Institutional ownership %"
    },
    
    # ===========================
    # COMPOSITE GATES
    # ===========================
    "confidence": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Setup confidence score"
    },
    
    "rrRatio": {
        "type": "numeric",
        "category": "risk_reward",
        "validation_type": "threshold",
        "description": "Risk-reward ratio"
    },
    
    "technicalScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Aggregated technical score"
    },
    
    "fundamentalScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Aggregated fundamental score"
    },
    
    "hybridScore": {
        "type": "numeric",
        "category": "composite",
        "validation_type": "threshold",
        "description": "Hybrid technical + fundamental score"
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
                    "atrPct": {"min": 1.0},
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
                "execution_order": 1.5,  # After structural gates, before confidence
                "volatility_guards": {"description": "Dynamic volatility quality requirements","extreme_vol_buffer": 2.0,"min_quality_breakout": 2.0,"min_quality_normal": 4.0,"enabled": True, "logic": "if atr_pct > extreme_threshold: use min_quality_breakout"},
                "structure_validation": {"description": "Price structure alignment checks","breakout_clearance": 0.001,"breakdown_clearance": 0.001, "enabled": True, "logic": "price must clear resistance/support by clearance %"},
                "sl_distance_validation": {"description": "Stop loss feasibility constraints","min_atr_multiplier": 0.5,"max_atr_multiplier": 5.0, "enabled": True, "logic": "0.5*ATR <= |entry-SL| <= 5*ATR"},
                "target_proximity_rejection": {"description": "Reject if targets too close to resistance","min_t1_distance": 0.005,"min_t2_distance": 0.01, "enabled": True, "logic": "t1 must be > resistance * (1 + min_t1_distance)"}
            },
            "opportunity": {
                "description": "Validates trade opportunity quality after confidence calculation",
                "execution_order": 2,
                
                "gates": {
                    "confidence": {"min": 55},
                    "confidence_requirements": {
                        "breakout_base": 70,
                        "trend_discount": -15,
                        "accumulation_discount": -25
                    },
                    "rrRatio": {"min": 1.5},
                    "technical_score": {"min": None},
                    "fundamental_score": {"min": None},
                    "hybrid_score": {"min": None},
                    "max_pattern_age_candles": None,
                    "max_setup_staleness_candles": None
                }
            },
            "setup_gate_specifications": {
                "MOMENTUM_BREAKOUT": {
                    "nature": "Requires strong established trend with momentum confirmation",
                    "structural_gates": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 3.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 65},
                        "rrRatio": {"min": 1.5}
                    }
                },
                
                "VOLATILITY_SQUEEZE": {
                    "nature": "Pre-breakout compression setup, trend-agnostic",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": 7.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 60},
                        "rrRatio": {"min": 1.8}
                    }
                },
                
                "QUALITY_ACCUMULATION": {
                    "nature": "Consolidation-based accumulation, fundamentally driven",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None},
                        "roe": {"min": 15.0},
                        "roce": {"min": 15.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 45},
                        "fundamental_score": {"min": 6.0}
                    }
                },
                
                "QUALITY_ACCUMULATION_DOWNTREND": {
                    "nature": "Accumulation during downtrend/sideways, quality-first",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "roe": {"min": 20.0},
                        "roce": {"min": 25.0},
                        "deRatio": {"max": 0.5}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 40},
                        "fundamental_score": {"min": 7.0}
                    }
                },
                
                "DEEP_VALUE_PLAY": {
                    "nature": "Deep value based on fundamentals, trend-neutral",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None},
                        "roe": {"min": 12.0},
                        "roce": {"min": 12.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 40},
                        "fundamental_score": {"min": 7.0}
                    }
                },
                
                "VALUE_TURNAROUND": {
                    "nature": "Value stock showing early trend reversal signs",
                    "structural_gates": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 3.0},
                        "roe": {"min": 12.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 50},
                        "fundamental_score": {"min": 6.0}
                    }
                },
                
                "TREND_PULLBACK": {
                    "nature": "Dip-buying in established uptrend",
                    "structural_gates": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 3.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "DEEP_PULLBACK": {
                    "nature": "Deeper retracement in trend, higher risk/reward",
                    "structural_gates": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 2.5}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 50},
                        "rrRatio": {"min": 2.0}
                    }
                },
                
                "REVERSAL_MACD_CROSS_UP": {
                    "nature": "Early reversal signal via MACD momentum shift",
                    "structural_gates": {
                        "adx": {"min": 10},
                        "trendStrength": {"min": None},
                        "macdhistogram": {"min": 0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 50}
                    }
                },
                
                "REVERSAL_RSI_SWING_UP": {
                    "nature": "Oversold bounce reversal",
                    "structural_gates": {
                        "adx": {"min": 12},
                        "trendStrength": {"min": None},
                        "rsi": {"min": 30}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 45}
                    }
                },
                
                "REVERSAL_ST_FLIP_UP": {
                    "nature": "Supertrend reversal confirmation",
                    "structural_gates": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 2.5}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "PATTERN_DARVAS_BREAKOUT": {
                    "nature": "Darvas box breakout pattern",
                    "structural_gates": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 4.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 65}
                    }
                },
                
                "PATTERN_VCP_BREAKOUT": {
                    "nature": "Minervini-style volatility contraction",
                    "structural_gates": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": 6.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 65}
                    }
                },
                
                "PATTERN_CUP_BREAKOUT": {
                    "nature": "Cup and handle breakout",
                    "structural_gates": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 4.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 60}
                    }
                },
                
                "PATTERN_FLAG_BREAKOUT": {
                    "nature": "Flag/pennant continuation pattern",
                    "structural_gates": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.5},
                        "volatilityQuality": {"min": 4.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 65}
                    }
                },
                
                "PATTERN_GOLDEN_CROSS": {
                    "nature": "Long-term MA crossover signal",
                    "structural_gates": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 3.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 60}
                    }
                },
                
                "PATTERN_STRIKE_REVERSAL": {
                    "nature": "Three-line strike reversal pattern",
                    "structural_gates": {
                        "adx": {"min": 12},
                        "trendStrength": {"min": None}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "MOMENTUM_BREAKDOWN": {
                    "nature": "Breakdown from support with strong bearish momentum",
                    "structural_gates": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 65}
                    }
                },
                
                "SELL_AT_RANGE_TOP": {
                    "nature": "Range-bound exit at resistance",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 60}
                    }
                },
                
                "TAKE_PROFIT_AT_MID": {
                    "nature": "Partial profit in range-bound consolidation",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "TREND_FOLLOWING": {
                    "nature": "Classic trend-following entry in established trend",
                    "structural_gates": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "BEAR_TREND_FOLLOWING": {
                    "nature": "Short/sell setup in strong downtrend",
                    "structural_gates": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 55}
                    }
                },
                
                "GENERIC": {
                    "nature": "Fallback for unclassified setups",
                    "structural_gates": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None}
                    },
                    "opportunity_gates": {
                        "confidence": {"min": 45}
                    }
                }
            }
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
            
            # Setup Classification (SINGLE SOURCE - NOT in horizons)
            "setup_classification": {
                "consolidation": {
                    "bb_width_threshold": 0.5,
                    "volume_ratio_max": 0.8
                },
                "MOMENTUM_BREAKOUT": {
                    "bb_percent_b_min": 0.98,
                    "rsi_min": 60,
                    "wick_ratio_max": 2.5,
                    "rvol_min": 1.5
                },
                "MOMENTUM_BREAKDOWN": {
                    "bb_percentb_max": 0.02,
                    "rsi_max": 40,
                    "rvol_min": 1.5
                },
                "TREND_PULLBACK": {
                    "ma_dist_max": 0.05,
                    "rsi_min": 50
                },
                "bear_pullback": {
                    "ma_dist_max": 0.05,
                    "rsi_max": 50
                },
                "trend_following": {
                    "classic": {
                        "rsi_min": 55,
                        "macd_hist_min": 0
                    },
                    "strong_drift": {
                        "trend_strength_min": 7.0
                    }
                },
                "VALUE_TURNAROUND": {
                    "trend_strength_range": [3.0, 5.5],
                    "rsi_min": 45
                },
                "DEEP_VALUE_PLAY": {
                    "pe_ratio_max": 10.0,
                    "fcf_yield_min": 5.0
                },
                "QUALITY_ACCUMULATION": {
                    "consolidation_required": True,
                    "rsi_range": [40, 60]
                },
                "QUALITY_ACCUMULATION_DOWNTREND": {
                    "fundamental_requirements": {
                        "roe_min": 20,
                        "roce_min": 25,
                        "de_ratio_max": 0.5
                    },
                    "bb_percent_b_range": [0.2, 0.5]
                },
                "divergence": {
                    "lookback": 10,
                    "slope_diff_min": -0.05,
                    "confidence_penalties": {
                        "bearish_divergence": 0.70,
                        "bullish_divergence": 0.70
                    }
                },
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
                "climax": {"threshold": 2.0, "rsi_condition_min": 70}
            },
            
            # Wick Rejection
            "wickRejection": {
                "max_ratio": 2.5,
                "calculation": "abs(high - close) / abs(close - open)"
            },
            
            # Divergence Detection
            "divergence_detection": {
                "lookback": 10,
                "slope_diff_min": -0.05,
                "confidence_penalties": {
                    "bearish_divergence": 0.70,
                    "bullish_divergence": 0.70
                },
                "severity_bands": {
                    "minor": {
                        "rsi_slope_min": -0.03,
                        "confidence_penalty": 0.90,
                        "allow_entry": True
                    },
                    "moderate": {
                        "rsi_slope_min": -0.08,
                        "confidence_penalty": 0.70,
                        "allow_entry": True
                    },
                    "severe": {
                        "rsi_slope_min": -999,
                        "confidence_penalty": 0.50,
                        "allow_entry": False
                    }
                }
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
                "squeeze": {"score": 1.0, "min_quality": 7.0},
                "expansion": {"score": 0.6, "min_quality": 4.0}
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
            "dip_buy_reference": "ema50"
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
            "max_position_pct": 0.02,
            "setup_size_multipliers": {"default": 1.0},
            "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
            "min_rr_ratio": 1.5,
            "horizon_t2_cap": 0.10,
            "rr_regime_adjustments": {
                "strong_trend": {"adx_min": 35, "t1_mult": 2.0, "t2_mult": 4.0},
                "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
            }
        },
        
        "execution": {
            "stop_loss_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "max_hold_candles": 20,
            "dip_buy_reference": "ema50",
            "risk_reward_min": 2.0,
            "stop_loss": {
                "vol_qual_high_mult": 1.5,
                "vol_qual_normal_mult": 2.0,
                "vol_qual_low_mult": 3.0,
                "min_distance_mult": 0.5
            },
            "spread_adjustments": {
                "large_cap": {"min_mcap": 100000, "spread": 0.001},
                "mid_cap": {"min_mcap": 10000, "spread": 0.002},
                "small_cap": {"spread": 0.005}
            },
            "structure_validation": {
                "breakout_tolerance": 1.001,
                "breakdown_tolerance": 0.999
            }
        },
        
        "lookback": {"python_data": 600},
        
        # ⚠️ DEPRECATED - Use technical_score_config.py instead
        # Kept for backward compatibility only
        "scoring": {
            "thresholds": {"buy": 6.0, "hold": 5.0, "sell": 4.0}
        },
        
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
                    "min_fundamental_score": None
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
                    "min_fundamental_score": 3.0
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
                    "min_fundamental_score": 6.0,
                    "filters": {
                        "require_low_debt": True,
                        "min_roe": 15.0,
                        "min_roce": 15.0,
                        "max_de_ratio": 0.5 
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
                    "min_fundamental_score": 8.0,
                    "filters": {
                        "require_low_debt": True,
                        "min_roe": 20.0,
                        "min_roce": 25.0,
                        "min_piotroski_f": 7
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
        
        # ============================================================
        # ✅ KEEP: Confidence Floors (entry logic, not scoring)
        # ============================================================
        "confidence": {
            "horizon_discount": 5,
            "floors": {"buy": 55, "wait": 35},
            "volume_penalty": {"rvol_drought_penalty": -15, "ignore_for_squeeze": True},
            "adx_based_floors": {
                "strong": {"adx_min": 30, "floor": 60},
                "moderate": {"adx_min": 25, "floor": 50},
                "weak": {"adx_min": 20, "floor": 40},
                "range_bound": {"adx_max": 20, "floor": 30}
            },
            "setup_type_overrides": {
                "MOMENTUM_BREAKOUT": 5,
                "VOLATILITY_SQUEEZE": 10,
                "QUALITY_ACCUMULATION": 0
            },
            "adx_normalization": {
                "min": 10,
                "max": 40,
                "adjustment_factor": 12
            }
        },
        
        # ============================================================
        # ❌ REMOVE: Composites (moved to technical_score_config.py)
        # ============================================================
        # "composites": {...}  # Now in COMPOSITE_SCORING_CONFIG
        
        # ============================================================
        # ✅ KEEP: Targets (trade execution logic, not scoring)
        # ============================================================
        "targets": {
            "resistance_cushion": 0.96,
            "support_cushion": 1.005,
            "min_distance_pct": 0.5,
            "support_buffer": 0.998,
            "cover_cushion": 1.005
        },
        
        # ============================================================
        # ❌ REMOVE: Enhancements (empty, replaced by technical bonuses)
        # ============================================================
        # "enhancements": {},  # Now in TECHNICAL_BONUSES
        
        # ============================================================
        # ✅ KEEP: Divergence (pattern detection logic, not scoring)
        # ============================================================
        "divergence": {
            "rsi_slope_deceleration_ceiling": -0.08,
            "bearish_penalty": 0.70,
            "bullish_penalty": 0.70,
            "severity_bands": {
                "severe": {
                    "rsi_slope_threshold": -0.08,
                    "allow_entry": False,
                    "confidence_penalty": 1.0
                },
                "moderate": {
                    "rsi_slope_threshold": -0.03,
                    "allow_entry": True,
                    "confidence_penalty": 0.7
                },
                "minor": {
                    "rsi_slope_threshold": 0.0,
                    "allow_entry": True,
                    "confidence_penalty": 0.9
                }
            }
        }
    },
    
    "horizons": {
        "intraday": {
            "timeframe": "15m",
            "description": "Quick scalps and day trades",
            
            # ============================================================
            # STRUCTURAL PARAMETERS (Keep - defines horizon characteristics)
            # ============================================================
            "volume_analysis": {
                "rvol_surge_threshold": 3.0,
                "rvol_drought_threshold": 0.7
            },
            "time_estimation": {
                "candles_per_unit": 4
            },
            
            # ============================================================
            # TECHNICAL SETTINGS (Keep - indicator configuration)
            # ============================================================
            "moving_averages": {
                "type": "EMA",
                "fast": 20,
                "mid": 50,
                "slow": 200,
                "keys": ["ema20", "ema50", "ema200"],
                "dip_buy_reference": "ema20"
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
            
            # ============================================================
            # RISK MANAGEMENT (Keep - position sizing & execution)
            # ============================================================
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
                "min_rr_ratio": 1.2,
                "horizon_t2_cap": 0.04,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
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
            
            # ============================================================
            # GATES (Keep - entry validation)
            # ============================================================
            "entry_gates": {
                "structural": {
                    "adx_min": 18,
                    "trend_strength_min": 5.0,
                    "volatility_quality_min": 5.0,
                    "volatility_bands_atr_pct": {"min": 0.3,"ideal": 3.0,"max": 5.0},
                    "volatility_guards": {"extreme_vol_buffer": 2.0,"min_quality_breakout": 2.5,"min_quality_normal": 4.0}
                },
                "opportunity": {
                    "confidence_min": 65,
                    "min_rr_ratio": 1.2,
                    "technical_score_min": None,
                    "fundamental_score_min": None
                },
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"structural": {"volatility_quality_min": 4.0},"opportunity": {"confidence_min": 55}},
                    "DEEP_VALUE_PLAY": {"structural": {"volatility_quality_min": 2.5},"opportunity": {"confidence_min": 45}},
                    "VOLATILITY_SQUEEZE": {"structural": {"volatility_quality_min": 7.0},"opportunity": {"confidence_min": 60}},
                    "VALUE_TURNAROUND": {"structural": {"volatility_quality_min": 3.0},"opportunity": {"confidence_min": 50}},
                    "GENERIC": {"structural": {"volatility_quality_min": 3.0},"opportunity": {"confidence_min": 45}}
                }
            },
            
            # ============================================================
            # CONFIDENCE (Keep - setup-specific confidence floors)
            # ============================================================
            "confidence": {
                "horizon_discount": 10,
                "floors": {"buy": 55, "wait": 30},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 50,
                    "MOMENTUM_BREAKDOWN": 50,
                    "TREND_PULLBACK": 48,
                    "DEEP_PULLBACK": 45,
                    "QUALITY_ACCUMULATION": 40,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 48,
                    "BEAR_TREND_FOLLOWING": 48,
                    "REVERSAL_MACD_CROSS_UP": 45,
                    "REVERSAL_RSI_SWING_UP": 45,
                    "REVERSAL_ST_FLIP_UP": 50
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,
                    "ignore_for_squeeze": True
                }
            },
            
            # ============================================================
            # ❌ REMOVED: Now in technical_score_config.py
            # ============================================================
            # - scoring.metrics (technical metric weights)
            # - scoring.penalties (technical penalties)
            # - scoring.metric_weights (fundamental weights - not used for intraday)
            # - technical_weight_overrides
            # Setup Confidence - intraday-specific penalties/bonuses
            "setup_confidence": {
                "confidence_clamp": [30, 90],
                "penalties": {
                    "weak_trend": {
                        "condition": "adx < 20",
                        "amount": 10,
                        "reason": "Intraday trend too weak"
                    },
                    "low_liquidity": {
                        "condition": "rvol < 0.8",
                        "amount": 10,
                        "reason": "Thin intraday liquidity"
                    },
                    "wide_spread": {
                        "condition": "spread_pct > 0.005",
                        "amount": 8,
                        "reason": "Spread too wide for intraday"
                    },
                    "choppy_market": {
                        "condition": "trendStrength < 3.0",
                        "amount": 20,
                        "reason": "Choppy intraday - high whipsaw risk"
                    }
                },
                "bonuses": {
                    "momentum_volume_confirmed": {
                        "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol >= 3.0",
                        "amount": 10,
                        "reason": "Breakout with strong volume"
                    },
                    "clean_flag": {
                        "condition": "setup_type == 'PATTERN_FLAG_BREAKOUT' and volatilityQuality >= 6.0",
                        "amount": 8,
                        "reason": "Clean intraday flag breakout"
                    },
                    "explosive_trend": {
                        "condition": "trendStrength >= 8.0",
                        "amount": 20,
                        "reason": "Explosive intraday trend"
                    },
                    "strong_trend": {
                        "condition": "trendStrength >= 5.5",
                        "amount": 15,
                        "reason": "Strong intraday momentum"
                    },
                    "decent_trend": {
                        "condition": "trendStrength >= 4.0",
                        "amount": 8,
                        "reason": "Decent trend support"
                    }
                }
            },
            # Enhancements - intraday-specific
            "enhancements": {
                "volume_surge": {
                    "condition": "rvol >= 2.5",
                    "amount": 12.0,
                    "reason": "Strong intraday volume surge",
                    "max_boost": 15.0
                },
                "squeeze_release": {
                    "condition": "bbWidth < 3.0 and rvol >= 2.0",
                    "amount": 15.0,
                    "reason": "Tight squeeze breaking with volume",
                    "max_boost": 18.0
                },
                "momentum_spike": {
                    "condition": "momentumStrength >= 8.0",
                    "amount": 10.0,
                    "reason": "Explosive momentum for scalp",
                    "max_boost": 12.0
                }
            }
        },
        
        "short_term": {
            "timeframe": "1d",
            "description": "Swing trading (Days to Weeks)",
            "volume_analysis": {
                "rvol_surge_threshold": 2.5,
                "rvol_drought_threshold": 0.7
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
                "min_rr_ratio": 1.4,
                "horizon_t2_cap": 0.10,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 2.0,
                "target_atr_mult": 3.0,
                "max_hold_candles": 15,
                "dip_buy_reference": "ema50",
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
                    "adx_min": 15,
                    "trend_strength_min": 4.0,
                    "volatility_quality_min": 4.0,
                    "volatility_bands_atr_pct": {"min": 0.8,"ideal": 2.5,"max": 12.0},
                    "volatility_guards": {"extreme_vol_buffer": 2.0,"min_quality_breakout": 3.0,"min_quality_normal": 4.0}
                },
                "opportunity": {
                    "confidence_min": 60,
                    "min_rr_ratio": 1.4,
                    "fundamental_score_min": 3.0
                },
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"structural": {"volatility_quality_min": 2.5},"opportunity": {"confidence_min": 45}},
                    "DEEP_VALUE_PLAY": {"structural": {"volatility_quality_min": 2.0},"opportunity": {"confidence_min": 40}},
                    "VOLATILITY_SQUEEZE": {"structural": {"volatility_quality_min": 6.0},"opportunity": {"confidence_min": 55}},
                    "VALUE_TURNAROUND": {"structural": {"volatility_quality_min": None},"opportunity": {"confidence_min": 50}},
                    "GENERIC": {"structural": {"volatility_quality_min": 2.5},"opportunity": {"confidence_min": 40}}
                }
            },
            "trend_thresholds": {
                "slope": {
                    "strong": 10.0,
                    "moderate": 3.0
                }
            },
            "confidence": {
                "horizon_discount": 5,
                "floors": {"buy": 55, "wait": 30},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 55,
                    "MOMENTUM_BREAKDOWN": 55,
                    "TREND_PULLBACK": 53,
                    "DEEP_PULLBACK": 50,
                    "QUALITY_ACCUMULATION": 45,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 50,
                    "BEAR_TREND_FOLLOWING": 50,
                    "REVERSAL_MACD_CROSS_UP": 50,
                    "REVERSAL_RSI_SWING_UP": 50,
                    "REVERSAL_ST_FLIP_UP": 55,
                    "SELL_AT_RANGE_TOP": 60,
                    "TAKE_PROFIT_AT_MID": 55,
                    "PATTERN_STRIKE_REVERSAL": 55
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,
                    "ignore_for_squeeze": True
                }
            },
            "setup_confidence": {
                "confidence_clamp": [35, 95],
                "penalties": {
                    "moderate_divergence": {
                        "condition": "rsislope < -0.03",
                        "amount": 10,
                        "reason": "Moderate bearish RSI divergence"
                    },
                    "low_breakout_volume": {
                        "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol < 1.5",
                        "amount": 8,
                        "reason": "Breakout with insufficient volume"
                    },
                    "sideways_trend": {
                        "condition": "trendStrength < 3.5",
                        "amount": 15,
                        "reason": "Weak trend - sideways risk"
                    },
                    "trend_setup_weak_trend": {
                        "condition": "setup_type in ['TREND_PULLBACK', 'TREND_FOLLOWING'] and trendStrength < 4.0",
                        "amount": 20,
                        "reason": "Trend-following setup without strong trend"
                    }
                },
                "bonuses": {
                    "pattern_confluence": {
                        "condition": "pattern_count >= 2",
                        "amount": 10,
                        "reason": "Multiple bullish patterns aligned"
                    },
                    "exceptional_combo": {
                        "condition": "trendStrength >= 7.0 and momentumStrength >= 7.0",
                        "amount": 25,
                        "reason": "Exceptional trend + momentum synergy"
                    },
                    "very_strong_trend": {
                        "condition": "trendStrength >= 7.5 and momentumStrength < 7.0",
                        "amount": 25,
                        "reason": "Very strong trend (standalone)"
                    },
                    "strong_trend": {
                        "condition": "trendStrength >= 6.0 and trendStrength < 7.5",
                        "amount": 20,
                        "reason": "Strong trend confirmation"
                    },
                    "moderate_trend": {
                        "condition": "trendStrength >= 4.5 and trendStrength < 6.0",
                        "amount": 10,
                        "reason": "Moderate trend support"
                    }
                }
            },
            "enhancements": {
                "pattern_confluence": {
                    "condition": "pattern_count >= 2",
                    "amount": 12.0,
                    "reason": "Multiple patterns confirm swing setup",
                    "max_boost": 15.0
                },
                "trend_momentum_sync": {
                    "condition": "trendStrength >= 7.0 and momentumStrength >= 7.0",
                    "amount": 10.0,
                    "reason": "Strong trend + momentum for swing",
                    "max_boost": 15.0
                },
                "quality_pullback": {
                    "condition": "setup_type == 'TREND_PULLBACK' and volatilityQuality >= 6.0",
                    "amount": 8.0,
                    "reason": "Clean pullback setup",
                    "max_boost": 12.0
                }
            }
            # ============================================================
            # ❌ REMOVED: Now in technical_score_config.py & fundamental_score_config.py
            # ============================================================
            # - scoring.fundamental_weight
            # - scoring.metrics
            # - scoring.penalties
            # - scoring.thresholds
            # - scoring.metric_weights
            # - technical_weight_overrides
        },
        
        "long_term": {
            "timeframe": "1wk",
            "description": "Trend Following & Investing",
            
            # ============================================================
            # STRUCTURAL PARAMETERS
            # ============================================================
            "trend_thresholds": {
                "slope": {
                    "strong": 5.0,
                    "moderate": 2.0
                }
            },
            "volume_analysis": {
                "rvol_surge_threshold": 2.0,
                "rvol_drought_threshold": 0.8
            },
            "time_estimation": {
                "candles_per_unit": 0.2
            },
            
            # ============================================================
            # TECHNICAL SETTINGS
            # ============================================================
            "moving_averages": {
                "type": "WMA",
                "fast": 10,
                "mid": 40,
                "slow": 50,
                "keys": ["wma10", "wma40", "wma50"],
                "dip_buy_reference": "wma40"
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
            
            # ============================================================
            # RISK MANAGEMENT
            # ============================================================
            "position_sizing": {
                "base_risk_pct": 0.015
            },
            
            "risk_management": {
                "max_position_pct": 0.03,
                "setup_size_multipliers": {"default": 1.0},
                "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
                "min_rr_ratio": 1.5,
                "horizon_t2_cap": 0.20,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 35, "t1_mult": 2.5, "t2_mult": 5.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.0, "t2_mult": 4.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.5, "t2_mult": 3.0}
                }
            },
            
            "execution": {
                "stop_loss_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "max_hold_candles": 52,
                "dip_buy_reference": "wma40",
                "risk_reward_min": 2.5,
                "base_hold_days": 60,
                "proximity_rejection": {
                    "resistance_mult": 1.01,
                    "support_mult": 0.99
                },
                "min_profit_pct": 1.0
            },
            
            "lookback": {"python_data": 800},
            
            # ============================================================
            # GATES
            # ============================================================
            "entry_gates": {
                "structural": {
                    "adx_min": None,
                    "trend_strength_min": 3.0,
                    "volatility_quality_min": None,
                    "volatility_bands_atr_pct": {"min": 1.0,"ideal": 5.5,"max": 15.0},
                    "volatility_guards": {"extreme_vol_buffer": 3.0,"min_quality_breakout": 4.0,"min_quality_normal": 5.0}
                },
                "opportunity": {
                    "confidence_min": 55,
                    "min_rr_ratio": 1.5,
                    "fundamental_score_min": 6.0
                },
                
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"structural": {"adx_min": None,"trend_strength_min": None,"volatility_quality_min": None},"opportunity": {"confidence_min": None}},
                    "DEEP_VALUE_PLAY": {"structural": {"volatility_quality_min": None},"opportunity": {"confidence_min": 35}},
                    "VALUE_TURNAROUND": {"structural": {"adx_min": 8,"volatility_quality_min": None},"opportunity": {"confidence_min": 45}},
                    "MOMENTUM_BREAKOUT": {"structural": {"trend_strength_min": 4.5,"volatility_quality_min": None},"opportunity": {"confidence_min": 50}},
                    "GENERIC": {"structural": {"volatility_quality_min": None},"opportunity": {"confidence_min": 35}}
                }
            },
            # ============================================================
            # CONFIDENCE
            # ============================================================
            "confidence": {
                "horizon_discount": 0,
                "floors": {"buy": 60, "wait": 40},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 60,
                    "MOMENTUM_BREAKDOWN": 60,
                    "TREND_PULLBACK": 58,
                    "DEEP_PULLBACK": 55,
                    "QUALITY_ACCUMULATION": 55,
                    "VOLATILITY_SQUEEZE": 55,
                    "TREND_FOLLOWING": 55,
                    "BEAR_TREND_FOLLOWING": 55,
                    "REVERSAL_MACD_CROSS_UP": 55,
                    "REVERSAL_RSI_SWING_UP": 55,
                    "REVERSAL_ST_FLIP_UP": 60
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -10,
                    "ignore_for_squeeze": True
                }
            },
            "setup_confidence": {
                "confidence_clamp": [40, 98],
                "penalties": {
                    "poor_fundamentals": {
                        "condition": "roe < 15 or roce < 15",
                        "amount": 15,
                        "reason": "Weak fundamentals for long-term hold"
                    },
                    "high_debt": {
                        "condition": "deRatio > 1.0",
                        "amount": 12,
                        "reason": "High leverage risk"
                    },
                    "moderate_weak_trend": {
                        "condition": "trendStrength >= 4.5 and trendStrength < 5.5",
                        "amount": 10,
                        "reason": "Moderate trend weakness for long-term"
                    },
                    "very_weak_trend": {
                        "condition": "trendStrength < 4.5",
                        "amount": 20,
                        "reason": "Very weak trend - unstable for long-term hold"
                    }
                },
                "bonuses": {
                    "high_quality_compounder": {
                        "condition": "roe >= 20 and roce >= 25 and epsGrowth5y >= 15",
                        "amount": 15,
                        "reason": "High-quality long-term compounder"
                    },
                    "stable_growth": {
                        "condition": "earningsStability >= 7.0 and revenueGrowth5y >= 10",
                        "amount": 10,
                        "reason": "Stable earnings and revenue growth"
                    },
                    "exceptional_trend": {
                        "condition": "trendStrength >= 8.0",
                        "amount": 25,
                        "reason": "Exceptional sustained trend"
                    },
                    "strong_trend": {
                        "condition": "trendStrength >= 6.5",
                        "amount": 20,
                        "reason": "Strong sustained trend"
                    }
                }
            },
            "enhancements": {
                "quality_fundamentals": {
                    "condition": "roe >= 20 and roce >= 25",
                    "amount": 15.0,
                    "reason": "High-quality company for trend trade",
                    "max_boost": 20.0
                },
                "earnings_acceleration": {
                    "condition": "quarterlyGrowth >= 15 and epsGrowth5y >= 15",
                    "amount": 12.0,
                    "reason": "Consistent earnings growth",
                    "max_boost": 15.0
                },
                "institutional_interest": {
                    "condition": "institutionalOwnership >= 25 and institutionalOwnership <= 75",
                    "amount": 8.0,
                    "reason": "Smart money accumulating",
                    "max_boost": 10.0
                }
            }
            # ============================================================
            # ❌ REMOVED: Now in technical_score_config.py & fundamental_score_config.py
            # ============================================================
            # - scoring (entire section)
            # - technical_weight_overrides
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
                "rvol_surge_threshold": 1.8,
                "rvol_drought_threshold": 0.8
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
                "dip_buy_reference": "mma12"
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
                "min_rr_ratio": 2.0,
                "horizon_t2_cap": 1.00,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 30, "t1_mult": 3.0, "t2_mult": 10.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.5, "t2_mult": 8.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 2.0, "t2_mult": 6.0}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 3.0,
                "target_atr_mult": 10.0,
                "max_hold_candles": 60,
                "dip_buy_reference": "mma12",
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
                    "adx_min": None,
                    "trend_strength_min": 3.0,
                    "volatility_quality_min": None,
                    "volatility_bands_atr_pct": {"min": 1.0,"ideal": 6.0,"max": 20.0},
                    "volatility_guards": {"extreme_vol_buffer": 4.0,"min_quality_breakout": 5.0,"min_quality_normal": 6.0}
                },
                "execution_rules": {
                    "volatility_guards": {
                        "enabled":False
                    }   # ✅ explicitly disabled
                },
                "opportunity": {
                    "confidence_min": 60,
                    "min_rr_ratio": 2.0,
                    "fundamental_score_min": 8.0
                },
                
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"opportunity": {"confidence_min": 60}},
                    "DEEP_VALUE_PLAY": {"opportunity": {"confidence_min": 55}},
                    "VALUE_TURNAROUND": {"opportunity": {"confidence_min": 65}},
                    "GENERIC": {"opportunity": {"confidence_min": 40}}
                }
            },
            "confidence": {
                "horizon_discount": 0,
                "floors": {"buy": 65, "wait": 50},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 65,
                    "MOMENTUM_BREAKDOWN": 65,
                    "TREND_PULLBACK": 60,
                    "DEEP_PULLBACK": 58,
                    "QUALITY_ACCUMULATION": 60,
                    "VOLATILITY_SQUEEZE": 60,
                    "TREND_FOLLOWING": 60,
                    "BEAR_TREND_FOLLOWING": 60,
                    "REVERSAL_MACD_CROSS_UP": 60,
                    "REVERSAL_RSI_SWING_UP": 60,
                    "REVERSAL_ST_FLIP_UP": 65
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -5,
                    "ignore_for_squeeze": True
                }
            },
            "setup_confidence": {
                "confidence_clamp": [45, 99],
                "penalties": {
                    "insufficient_growth": {
                        "condition": "epsGrowth5y < 15 or revenueGrowth5y < 15",
                        "amount": 20,
                        "reason": "Growth too low for multibagger profile"
                    },
                    "weak_quality": {
                        "condition": "roe < 15 or roce < 18",
                        "amount": 30,
                        "reason": "Insufficient quality for multibagger thesis"
                    },
                    "high_leverage": {
                        "condition": "deRatio > 1.2",
                        "amount": 20,
                        "reason": "High debt risk for long-term hold"
                    },
                    "bearish_trend": {
                        "condition": "trendStrength < 3.0 and momentumStrength < 3.0",
                        "amount": 25,
                        "reason": "Bearish trend - wait for reversal"
                    }
                },
                "bonuses": {
                    "exceptional_quality_growth": {
                        "condition": "roe >= 25 and roce >= 30 and epsGrowth5y >= 20 and revenueGrowth5y >= 20",
                        "amount": 20,
                        "reason": "Exceptional quality + growth combo"
                    },
                    "early_stage_leader": {
                        "condition": "relStrengthNifty >= 1.2 and marketCapCagr >= 25",
                        "amount": 15,
                        "reason": "Outperforming market with strong cap growth"
                    },
                    "mega_trend": {
                        "condition": "trendStrength >= 8.5 and roe >= 25",
                        "amount": 30,
                        "reason": "Mega trend with quality fundamentals"
                    },
                    "strong_trend": {
                        "condition": "trendStrength >= 7.0 and trendStrength < 8.5",
                        "amount": 25,
                        "reason": "Strong multi-year trend"
                    },
                    "quality_emerging_trend": {
                        "condition": "trendStrength >= 5.0 and trendStrength < 7.0 and roe >= 25 and epsGrowth5y >= 15",
                        "amount": 20,
                        "reason": "Quality company with emerging trend - ideal multibagger entry"
                    }
                }
            },
            "enhancements": {
                "quality_technical_setup": {
                    "condition": "setup_type == 'QUALITY_ACCUMULATION' and trendStrength >= 4.0",
                    "amount": 20.0,
                    "reason": "Quality stock in early accumulation",
                    "max_boost": 25.0
                },
                "growth_combo": {
                    "condition": "epsGrowth5y >= 25 and marketCapCagr >= 25",
                    "amount": 18.0,
                    "reason": "Consistent compounding + price performance",
                    "max_boost": 20.0
                },
                "moat_indicators": {
                    "condition": "roic >= 20 and roe >= 25 and deRatio < 0.3",
                    "amount": 15.0,
                    "reason": "Economic moat indicators present",
                    "max_boost": 20.0
                },
                "undiscovered_gem": {
                    "condition": "institutionalOwnership < 15 and marketCap < 10000",
                    "amount": 12.0,
                    "reason": "Under-the-radar quality stock",
                    "max_boost": 15.0
                }
            }
            # ============================================================
            # ❌ REMOVED: Now in technical_score_config.py & fundamental_score_config.py
            # ============================================================
            # - technical_weight_overrides
            # - scoring (entire section)
        }
    }
}

